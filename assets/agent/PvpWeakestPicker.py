# -*- coding: utf-8 -*-
"""4v4 圆形角斗场：比较三个对手的队伍数值，挑最弱的那个「开始战斗」。

工作方式
--------

1. OCR 三行「队伍每秒伤害」（或「队伍韧性值」）；
2. 取数值最低的一行；
3. 确认那一行的「开始战斗」按钮还在（黄色像素占比）；
4. 返回该按钮的矩形 —— 节点用不带 `target` 的 `Click`，就会点识别框中心。

坐标基于 1280x720（与 `pvp_4v4.json`、`interface.json` 的
`display_short_side = 720` 一致）。

自定义识别参数（`custom_recognition_param`）
--------------------------------------------
metric
    比较哪个数值。``"dps"``（默认）= 队伍每秒伤害；``"tenacity"`` = 队伍韧性值。
value_rois
    可选，覆盖默认的三行数值格 ROI：``[[x, y, w, h], ...]``。
buttons
    可选，覆盖默认的三个「开始战斗」按钮矩形：``[[x, y, w, h], ...]``。
require_button
    可选，默认 ``True``。为真时只在那一行按钮确实存在（黄色）时才命中；
    若该行按钮没了（打过了 / 次数用完），自动退到次低的一行。

命中时返回的 box = 选中那一行按钮的矩形；未命中返回 ``None``。
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

from maa.agent.agent_server import AgentServer
from maa.custom_recognition import CustomRecognition
from maa.define import OCRResult

# ======================================================================
# 常量
# ======================================================================

#: 借用的 OCR 节点名（定义在 pvp_4v4.json 里，本模块用 pipeline_override 改 ROI）
PROBE_NODE = "对手数值格"

#: 日志落盘（插件会吞掉 agent 的 stdout，只有落盘才查得到）
LOG_FILE = Path(__file__).resolve().parent.parent.parent / "debug" / "pvp_4v4.log"

#: 三行「队伍每秒伤害」数值格 ROI (x, y, w, h)
DPS_ROIS: tuple[tuple[int, int, int, int], ...] = (
    (998, 268, 62, 24),
    (998, 418, 62, 24),
    (998, 568, 62, 24),
)
#: 三行「队伍韧性值」数值格 ROI
TENACITY_ROIS: tuple[tuple[int, int, int, int], ...] = (
    (990, 289, 72, 24),
    (990, 439, 72, 24),
    (990, 589, 72, 24),
)
#: 三行「开始战斗」按钮矩形（点它的中心；与 StarUp/44.png 的匹配位置一致）
BUTTON_RECTS: tuple[tuple[int, int, int, int], ...] = (
    (1095, 236, 125, 60),
    (1095, 386, 125, 60),
    (1095, 536, 125, 60),
)

STAT_ROIS: dict[str, tuple[tuple[int, int, int, int], ...]] = {
    "dps": DPS_ROIS,
    "tenacity": TENACITY_ROIS,
}
STAT_NAMES: dict[str, str] = {
    "dps": "队伍每秒伤害",
    "tenacity": "队伍韧性值",
}

#: 数值形如 ``2,779K`` / ``18,670K``（OCR 偶尔把千分位逗号读成小数点）
_NUMBER_RE = re.compile(r"\d[\d,.]*")
#: 单位后缀 → 倍率
_UNITS: tuple[tuple[str, float], ...] = (
    ("K", 1_000),
    ("M", 1_000_000),
    ("W", 10_000),
    ("万", 10_000),
)
#: 「开始战斗」按钮的黄色占比低于该值 ⇒ 认为按钮不存在
BUTTON_YELLOW_MIN = 0.3


def log(msg: str) -> None:
    """打印 + 落盘。插件会把 agent 的 stdout 收走，不落盘不好排查。

    控制台可能是 GBK，遇到编不出来的字符不能让识别挂掉，退回 ``ascii``。
    """
    line = f"[Pvp4v4] {msg}"
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "backslashreplace").decode("ascii"), flush=True)

    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with LOG_FILE.open("a", encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
    except Exception:  # noqa: BLE001 - 日志失败不能影响识别
        pass


# ======================================================================
# 解析
# ======================================================================

def parse_number(text: str | None) -> float | None:
    """把 ``2,779K`` / ``2.779K`` / ``2779K`` 解析成 ``2779000.0``。"""
    if not text:
        return None

    match = _NUMBER_RE.search(text)
    if match is None:
        return None

    raw = match.group(0).rstrip(",.")
    if not raw:
        return None

    parts = re.split(r"[,.]", raw)
    try:
        if len(parts) == 1:
            value = float(parts[0])
        elif len(parts[-1]) == 3:
            # 2,779 / 2.779 —— 后面正好 3 位，当千分位
            value = float("".join(parts))
        else:
            # 2.5 —— 当小数
            value = float("".join(parts[:-1]) + "." + parts[-1])
    except ValueError:
        return None

    tail = text[match.end():].strip().upper()
    for unit, scale in _UNITS:
        if tail.startswith(unit):
            value *= scale
            break
    return value


def _parse_params(raw_params: Any) -> dict[str, Any]:
    if isinstance(raw_params, dict):
        return raw_params
    if isinstance(raw_params, str):
        try:
            parsed = json.loads(raw_params)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def _as_rects(value: Any) -> tuple[tuple[int, int, int, int], ...] | None:
    """把 ``[[x, y, w, h], ...]`` 规整成 tuple；格式不对返回 None。"""
    if not isinstance(value, (list, tuple)) or not value:
        return None

    rects: list[tuple[int, int, int, int]] = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 4:
            return None
        try:
            rects.append(tuple(int(v) for v in item))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
    return tuple(rects)


# ======================================================================
# 自定义识别
# ======================================================================

@AgentServer.custom_recognition("PvpWeakestPicker")
class PvpWeakestPicker(CustomRecognition):

    def analyze(
        self,
        context,
        argv: CustomRecognition.AnalyzeArg,
    ) -> CustomRecognition.AnalyzeResult | None:
        params = _parse_params(argv.custom_recognition_param)
        metric = str(params.get("metric") or "dps").strip().lower()

        rois = _as_rects(params.get("value_rois")) or STAT_ROIS.get(metric)
        buttons = _as_rects(params.get("buttons")) or BUTTON_RECTS
        if not rois or not buttons:
            return self._miss({"status": "参数无效", "metric": metric})
        if len(rois) != len(buttons):
            return self._miss(
                {
                    "status": "参数数量不匹配",
                    "metric": metric,
                    "value_rois": len(rois),
                    "buttons": len(buttons),
                }
            )

        require_button = params.get("require_button", True)
        image = argv.image
        stat_name = STAT_NAMES.get(metric, metric)

        rows: list[dict[str, Any]] = []
        for index, roi in enumerate(rois):
            text = self._ocr(context, image, roi)
            rows.append(
                {
                    "row": index,
                    "roi": list(roi),
                    "text": text,
                    "value": parse_number(text),
                }
            )

        readable = [row for row in rows if row["value"] is not None]
        if not readable:
            log(f"没读到任何「{stat_name}」数值，放弃本次选择")
            return self._miss(
                {"status": "没有可读数值", "metric": metric, "stat": stat_name, "rows": rows}
            )

        skipped: list[dict[str, Any]] = []
        for row in sorted(readable, key=lambda item: item["value"]):
            index = int(row["row"])
            yellow = self._yellow_fraction(image, buttons[index])
            present = yellow >= BUTTON_YELLOW_MIN

            if require_button and not present:
                # 这一行按钮没了（已经打过 / 次数用完），退到次低的一行
                skipped.append(
                    {"row": index, "value": row["value"], "button_yellow": round(yellow, 3)}
                )
                continue

            detail = {
                "status": "success",
                "metric": metric,
                "stat": stat_name,
                "picked_row": index,
                "picked_value": row["value"],
                "rows": rows,
                "skipped": skipped,
            }
            detail["summary"] = self._format_summary(detail)
            log(detail["summary"])
            return CustomRecognition.AnalyzeResult(
                box=tuple(buttons[index]),
                detail=detail,
            )

        log(f"三行按钮都不在（{skipped}），放弃本次选择")
        return self._miss(
            {
                "status": "没有可点的战斗按钮",
                "metric": metric,
                "stat": stat_name,
                "rows": rows,
                "skipped": skipped,
            }
        )

    # ------------------------------------------------------------------
    # OCR
    # ------------------------------------------------------------------

    def _ocr(self, context, image: np.ndarray, roi: tuple[int, int, int, int]) -> str:
        """借 `PROBE_NODE` 跑一次 OCR，用 override 把 ROI 换成给定的格子。"""
        detail = context.run_recognition(
            PROBE_NODE,
            image,
            pipeline_override={
                PROBE_NODE: {
                    "recognition": {
                        "type": "OCR",
                        "param": {"roi": list(roi), "only_rec": True},
                    }
                }
            },
        )
        if detail is None or not detail.hit:
            return ""

        texts: list[str] = []
        for result in detail.all_results or []:
            if isinstance(result, OCRResult) and result.text:
                texts.append(result.text)
        if not texts:
            best = detail.best_result
            if isinstance(best, OCRResult) and best.text:
                texts.append(best.text)
        return " ".join(texts).strip()

    # ------------------------------------------------------------------
    # 按钮存在性
    # ------------------------------------------------------------------

    def _yellow_fraction(self, image: np.ndarray, rect: tuple[int, int, int, int]) -> float:
        """按钮矩形内「开始战斗」底色（黄）的像素占比；图像是 BGR。"""
        if image is None:
            return 0.0

        x, y, w, h = rect
        if w <= 0 or h <= 0:
            return 0.0

        height, width = image.shape[:2]
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + w, width), min(y + h, height)
        if x1 <= x0 or y1 <= y0:
            return 0.0

        sub = np.asarray(image[y0:y1, x0:x1]).astype(np.int32)
        blue, green, red = sub[:, :, 0], sub[:, :, 1], sub[:, :, 2]
        mask = (blue < 80) & (green > 170) & (green < 240) & (red > 225)
        return float(mask.mean())

    # ------------------------------------------------------------------
    # 输出
    # ------------------------------------------------------------------

    def _miss(self, detail: dict[str, Any]) -> CustomRecognition.AnalyzeResult:
        detail.setdefault("status", "未命中")
        detail["summary"] = self._format_summary(detail)
        log(detail["summary"])
        return CustomRecognition.AnalyzeResult(box=None, detail=detail)

    def _format_summary(self, detail: dict[str, Any]) -> str:
        lines = [
            f"状态: {detail.get('status')}",
            f"比较项: {detail.get('stat')}（{detail.get('metric')}）",
        ]

        if "picked_row" in detail:
            lines.append(
                f"选中: 第 {int(detail['picked_row']) + 1} 个对手"
                f"（数值 {detail.get('picked_value')}）"
            )

        rows = detail.get("rows")
        if isinstance(rows, list):
            lines.append("三行读数:")
            for row in rows:
                if not isinstance(row, dict):
                    continue
                index = row.get("row")
                label = f"  第 {int(index) + 1} 个" if isinstance(index, int) else "  ?"
                lines.append(
                    f"{label}: text={row.get('text')!r} value={row.get('value')}"
                )

        skipped = detail.get("skipped")
        if isinstance(skipped, list) and skipped:
            lines.append(f"跳过（按钮不在）: {skipped}")

        return "\n".join(lines)
