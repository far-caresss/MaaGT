# -*- coding: utf-8 -*-
"""页面判据层（L0）+ 导航层（L2）+ 图内原语（L1）—— 新架构底座。

设计要点：

1. **只做三件事**：探"我在哪一页" → 收弹窗 → 按 `NAV_TABLE` 走一步。
   各任务的业务动作**不在这里**，这里只有"路"和"闸门"。
2. **判据全部来自 pipeline 节点**：本模块只调 `run_recognition`，不自己写 ROI。
   加一个页面 = 在 `page_base.json` 加一个 `Page*` 节点 + 在 `PAGE_NODES` 里登记。
3. **不 import `guild_raid` / `guild_activity`**：它们以后会 import 本模块，反向导入会成环。
   所以日志/截屏/点击这几个小工具在本模块重新实现一份（与既有写法同款）。
4. **认不出 = 不动作**：`probe_page` 返回 `PageUnknown` 时，`NavTo` 只尝试一次
   "回锚点"（复用既有 `Return`），仍然不行就失败退出，绝不盲点。

批次 0 范围：只登记 **已经能复用现有节点** 的判据（见 `assets/resource/pipeline/page_base.json`），
其余 `Page*` 等真机标定 ROI 后再逐个补上。
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

import numpy as np

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction
from maa.define import OCRResult

# ======================================================================
# 常量
# ======================================================================

_ASSETS_DIR = Path(__file__).resolve().parent.parent
_LOG_FILE = _ASSETS_DIR.parent / "debug" / "page_guard.log"

_TAP_WAIT = 0.6
_PAGE_WAIT = 1.5
_MAX_STEPS = 8

PAGE_UNKNOWN = "PageUnknown"
PAGE_MAIN = "PageMain"
PAGE_MENU = "PageMenuOpen"
PAGE_GUIDE = "PageGuide"

#: 判据节点（顺序 = 探测优先级；靠后的兜底页必须排在后面）
#: 批次 0 只登记"复用现有节点"的 4 个，其余等 ROI 标定后追加。
PAGE_NODES: tuple[str, ...] = (
    PAGE_MAIN,
    "PageGuild",
    "PageTask",
    "PageFriend",
)

#: 弹窗判据，自上而下收（列表顺序 = 优先级）
DIALOG_NODES: tuple[str, ...] = ("DlgReward",)

#: 弹窗关闭动作：`("point", (x, y))` 或 `("node", 节点名)`
DIALOG_CLOSERS: dict[str, tuple[str, Any]] = {
    # 「点击空白处关闭弹窗」类奖励弹窗：右上角空白处（与 guild_raid 的 TAP_BLANK_TO_CLOSE 一致）
    "DlgReward": ("point", (1236, 48)),
}

#: 回锚点用的既有节点（`startup.json`），带 `max_hit: 1` → 调用前必须 clear_hit_count
RECOVER_NODE = "Return"

#: 导航表：(当前页, 目标页) -> ((动作节点, 期望到达页或 None), ...)
#: 期望页为 None 表示"这一跳没有判据"，只打日志不校验（过渡期用）。
#: 每条边的最后一跳**必须**有判据，否则 `NavTo` 会拒绝执行整条路线。
NAV_TABLE: dict[tuple[str, str], tuple[tuple[str, str | None], ...]] = {
    (PAGE_MAIN, "PageGuild"): (("公会", "PageGuild"),),
    (PAGE_MAIN, "PageTask"): (("菜单1", PAGE_MENU), ("任务", "PageTask")),
    (PAGE_MAIN, "PageFriend"): (("菜单1", PAGE_MENU), ("好友", "PageFriend")),
}

#: `ReadNum` 的最近一次结果（节点名 -> 数字），供后续动作/日志使用
LAST_NUM: dict[str, int] = {}


# ======================================================================
# 小工具
# ======================================================================

def log(msg: str) -> None:
    """打印并落盘。插件会把 agent 的 stdout 收进自己的终端，不落盘不好排查。"""
    print(f"[PageGuard] {msg}", flush=True)
    try:
        _LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with _LOG_FILE.open("a", encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
    except Exception:  # noqa: BLE001 - 日志失败不能影响主流程
        pass


def _screencap(context: Context) -> np.ndarray | None:
    image = context.tasker.controller.post_screencap().wait().get()
    if image is None:
        return None
    return np.asarray(image)


def _tap(context: Context, x: int, y: int) -> None:
    context.tasker.controller.post_click(int(x), int(y)).wait()
    time.sleep(_TAP_WAIT)


def _box_of(detail: Any) -> tuple[int, int, int, int] | None:
    """把识别框统一成 (x, y, w, h)。maa 侧 box 可能是 list，也可能是对象。"""
    if detail is None or not getattr(detail, "hit", False):
        return None
    best = getattr(detail, "best_result", None)
    if best is None:
        return None
    raw = getattr(best, "box", None)
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)) and len(raw) == 4:
        return tuple(int(v) for v in raw)  # type: ignore[return-value]
    try:
        return (int(raw.x), int(raw.y), int(raw.w), int(raw.h))
    except AttributeError:
        return None


def _hit(context: Context, node: str, image: np.ndarray | None) -> bool:
    """跑一个判据节点；节点不存在 / 异常一律当"没命中"（不抛出）。"""
    if image is None:
        return False
    try:
        detail = context.run_recognition(node, image)
    except Exception as exc:  # noqa: BLE001 - 判据节点缺失不应该炸掉整个任务
        log(f"判据 {node} 执行失败（节点可能还没建）：{exc}")
        return False
    return bool(detail is not None and detail.hit)


def _text(context: Context, node: str, image: np.ndarray | None) -> str:
    """取一个 OCR 节点的全部文字（按位置排序拼接）。"""
    if image is None:
        return ""
    try:
        detail = context.run_recognition(node, image)
    except Exception:  # noqa: BLE001
        return ""
    if detail is None or not detail.hit or detail.best_result is None:
        return ""
    best = detail.best_result
    if isinstance(best, OCRResult):
        return best.text or ""
    return str(getattr(best, "text", "") or "")


def _click_node(context: Context, image: np.ndarray | None, node: str) -> bool:
    """先判定再点击：命中才点识别框中心（绝不盲点）。"""
    if image is None:
        return False
    try:
        detail = context.run_recognition(node, image)
    except Exception as exc:  # noqa: BLE001
        log(f"点击前判定 {node} 失败：{exc}")
        return False
    box = _box_of(detail)
    if box is None:
        log(f"点击前判定 {node} 未命中 → 不点")
        return False
    x, y, w, h = box
    _tap(context, x + w // 2, y + h // 2)
    log(f"已点击 {node} @({x + w // 2},{y + h // 2})")
    return True


# ======================================================================
# 纯函数（可离线单测）
# ======================================================================

def parse_num(text: str, pattern: str = r"(\d+)") -> int | None:
    """按 pattern 从文本里取第一个整数；**不做"取第一个数字串"的模糊兜底**。

    踩坑 §1.9：`2/3` 曾被误读成 `213`，再用 `re.findall(r"\\d+")[0]` 就得到 2 或 213，
    所以这里必须由调用方给出明确 pattern（如 `r"(\\d+)\\s*/\\s*(\\d+)"` 再取第 1 组）。
    """
    m = re.search(pattern, text or "")
    if not m:
        return None
    raw = m.group(1) if m.groups() else m.group(0)
    digits = re.sub(r"[^\d]", "", raw)
    if not digits:
        return None
    return int(digits)


def classify_price(text: str) -> str:
    """把「购买」弹窗右下按钮的文字分类成 `free` / `paid` / `real_money` / `unknown`。

    - `free`       → 按钮写着 `免费`，零消耗，可点
    - `real_money` → 含 `¥`，真实货币，永禁
    - `paid`       → 纯数字（可能带货币符号/千分位），有代价
    - `unknown`    → 认不出，按最保守处理（调用方应当拒绝）
    """
    s = (text or "").strip()
    if not s:
        return "unknown"
    if "¥" in s or "￥" in s:
        return "real_money"
    if "免费" in s or "無料" in s:
        return "free"
    if re.search(r"\d", s):
        return "paid"
    return "unknown"


def route_of(here: str, target: str) -> tuple[tuple[str, str | None], ...] | None:
    """查导航表；认不出当前页时退化为"从锚点出发"（先回主界面）。"""
    route = NAV_TABLE.get((here, target))
    if route:
        return route
    if here == PAGE_UNKNOWN:
        return NAV_TABLE.get((PAGE_MAIN, target))
    return None


def route_is_verifiable(route: tuple[tuple[str, str | None], ...]) -> bool:
    """一条路线必须至少有一跳带判据（通常是最后一跳），否则等于盲点。"""
    return bool(route) and route[-1][1] is not None


# ======================================================================
# 页面探测 / 收弹窗 / 导航
# ======================================================================

def probe_page(context: Context, image: np.ndarray | None) -> str:
    """依次试 `PAGE_NODES`，返回第一个命中的页面 id；都不命中返回 `PageUnknown`。"""
    if image is None:
        log("探页面：截图为空 → PageUnknown")
        return PAGE_UNKNOWN
    for node in PAGE_NODES:
        if _hit(context, node, image):
            return node
    return PAGE_UNKNOWN


def close_dialogs(context: Context, image: np.ndarray | None) -> bool:
    """自顶向下收一次弹窗；收掉了返回 True（调用方应重新截屏再探页面）。"""
    closed = False
    for node in DIALOG_NODES:
        if not _hit(context, node, image):
            continue
        closer = DIALOG_CLOSERS.get(node)
        if closer is None:
            log(f"命中弹窗 {node}，但没有配关闭动作 → 不动它")
            continue
        kind, arg = closer
        if kind == "point":
            x, y = arg
            _tap(context, x, y)
            log(f"收弹窗 {node}：点击空白处 ({x},{y})")
            closed = True
        elif kind == "node":
            if _click_node(context, image, arg):
                log(f"收弹窗 {node}：点击 {arg}")
                closed = True
        if closed:
            time.sleep(_PAGE_WAIT * 0.5)
            break
    return closed


def _recover_to_main(context: Context) -> bool:
    """回锚点：复用既有 `Return`（带 `max_hit:1`，调用前先清命中计数）。"""
    try:
        context.clear_hit_count(RECOVER_NODE)
        detail = context.run_task(RECOVER_NODE)
    except Exception as exc:  # noqa: BLE001
        log(f"回锚点失败（{RECOVER_NODE}）：{exc}")
        return False
    ok = bool(detail and detail.status.succeeded)
    log(f"回锚点 {RECOVER_NODE}：succeeded={ok}")
    return ok


def nav_to(context: Context, target: str, max_steps: int = _MAX_STEPS) -> bool:
    """从任意一页走到 `target`。每步打一行三元日志：页 →(动作)→ 期望页。"""
    if target not in PAGE_NODES:
        log(f"NavTo 拒绝执行：目标页 {target} 还没有判据节点（不能盲走）")
        return False

    for step in range(1, max_steps + 1):
        image = _screencap(context)
        if close_dialogs(context, image):
            continue

        here = probe_page(context, image)
        if here == target:
            log(f"NavTo 已到达 {target}（{step - 1} 步）")
            return True

        if here == PAGE_UNKNOWN:
            log("NavTo：认不出当前页 → 尝试回锚点")
            if not _recover_to_main(context):
                log("NavTo 失败：回锚点也没成功")
                return False
            continue

        route = route_of(here, target)
        if route is None:
            log(f"NavTo 失败：{here} → {target} 没有路线（不猜）")
            return False
        if not route_is_verifiable(route):
            log(f"NavTo 拒绝执行 {here} → {target}：整条路线没有任何判据")
            return False

        action, expect = route[0]
        if not _click_node(context, image, action):
            log(f"NavTo：在 {here} 上找不到 {action} → 放弃")
            return False
        log(f"[nav] {here} →({action})→ {expect or '?'} ({step}/{max_steps})")
        time.sleep(_PAGE_WAIT)

    log(f"NavTo 失败：超过 {max_steps} 步仍未到达 {target}")
    return False


# ======================================================================
# 图内原语（Custom Action）
# ======================================================================

class _Base(CustomAction):
    @staticmethod
    def _param(argv: Any) -> dict[str, Any]:
        raw = getattr(argv, "custom_action_param", None)
        return raw if isinstance(raw, dict) else {}


@AgentServer.custom_action("NavTo")
class NavToAction(_Base):
    """`{"target": "PageGuild", "max_steps": 8}`"""

    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        p = self._param(argv)
        target = str(p.get("target") or "")
        if not target:
            log("NavTo 缺 target 参数")
            return False
        return nav_to(context, target, int(p.get("max_steps") or _MAX_STEPS))


@AgentServer.custom_action("AssertOnPage")
class AssertOnPageAction(_Base):
    """`{"page": "PageGuide"}` —— 不满足就什么都不点，返回失败。"""

    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        p = self._param(argv)
        page = str(p.get("page") or "")
        image = _screencap(context)
        here = probe_page(context, image)
        if here == page:
            log(f"断言通过：当前页 = {here}")
            return True
        log(f"断言失败：期望 {page}，实际 {here} → 不执行任何动作")
        return False


@AgentServer.custom_action("RepeatUntilFound")
class RepeatUntilFoundAction(_Base):
    """`{"inner": "节点名或 tap_blank", "repeat_count": 3, "wait_nodes": ["DlgReward"]}`

    有界重复：每次先看 `wait_nodes` 有没有命中，命中即成功返回。
    """

    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        p = self._param(argv)
        inner = str(p.get("inner") or "")
        count = int(p.get("repeat_count") or 1)
        wait_nodes = [str(n) for n in (p.get("wait_nodes") or [])]
        interval = float(p.get("interval") or _PAGE_WAIT)
        blank = p.get("tap_blank") or (inner == "tap_blank")

        for i in range(1, count + 1):
            image = _screencap(context)
            for node in wait_nodes:
                if _hit(context, node, image):
                    log(f"RepeatUntilFound 第 {i}/{count} 次：命中 {node} → 成功")
                    return True
            if blank:
                _tap(context, 1236, 48)
                log(f"RepeatUntilFound 第 {i}/{count} 次：点空白处")
            elif inner:
                if not _click_node(context, image, inner):
                    log(f"RepeatUntilFound 第 {i}/{count} 次：{inner} 未命中")
            time.sleep(interval)

        image = _screencap(context)
        for node in wait_nodes:
            if _hit(context, node, image):
                log(f"RepeatUntilFound：{count} 次后命中 {node} → 成功")
                return True
        log(f"RepeatUntilFound 失败：{count} 次内未命中 {wait_nodes}")
        return False


@AgentServer.custom_action("ClearHitCount")
class ClearHitCountAction(_Base):
    """`{"nodes": ["Return"]}` —— 把"清命中计数"从 Python 里的隐式一行搬到图上。"""

    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        nodes = [str(n) for n in (self._param(argv).get("nodes") or [])]
        if not nodes:
            log("ClearHitCount 缺 nodes 参数")
            return False
        for node in nodes:
            try:
                context.clear_hit_count(node)
            except Exception as exc:  # noqa: BLE001
                log(f"ClearHitCount {node} 失败：{exc}")
                return False
            log(f"已清命中计数：{node}")
        return True


@AgentServer.custom_action("ReadNum")
class ReadNumAction(_Base):
    """`{"node": "RaidEntries", "pattern": "(\\\\d+)\\\\s*/\\\\s*(\\\\d+)"}`

    读一个 OCR 节点里的数字（按 pattern 严格解析）。读不到 → 返回失败（由 `next`/`on_error` 决定后续）。
    """

    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        p = self._param(argv)
        node = str(p.get("node") or "")
        pattern = str(p.get("pattern") or r"(\d+)")
        if not node:
            log("ReadNum 缺 node 参数")
            return False
        image = _screencap(context)
        text = _text(context, node, image)
        value = parse_num(text, pattern)
        if value is None:
            log(f"ReadNum 失败：{node} 读到 {text!r}，pattern={pattern!r} 解析不出数字")
            return False
        LAST_NUM[node] = value
        log(f"ReadNum {node} = {value}（原文 {text!r}）")
        return True


@AgentServer.custom_action("TapUnlessConsumes")
class TapUnlessConsumesAction(_Base):
    """`{"button": "节点名", "consumes": "gold", "cost": "1,000", "allow_consume": false}`

    消耗闸门：`consumes` 非空且 `allow_consume` 为假 → **拒绝点击**并记日志。
    `consumes` 取 `real_money` 时**永不**放行（只认 PI 开关不够，要改代码）。
    """

    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        p = self._param(argv)
        button = str(p.get("button") or "")
        consumes = p.get("consumes")
        cost = p.get("cost")
        allow = bool(p.get("allow_consume"))
        if not button:
            log("TapUnlessConsumes 缺 button 参数")
            return False

        if consumes == "real_money":
            log(f"拒绝点击 {button}：涉及真实货币（永不放开）")
            return False
        if consumes and not allow:
            log(f"拒绝点击 {button}：这里会消耗 {consumes}（{cost or '代价未知'}），需要开关放开")
            return False

        image = _screencap(context)
        ok = _click_node(context, image, button)
        if ok:
            log(f"已点击 {button}（consumes={consumes or '无'}）")
        return ok


@AgentServer.custom_action("WaitUntil")
class WaitUntilAction(_Base):
    """`{"nodes": ["DlgReward"], "timeout": 180, "interval": 2}`

    纯等待、**不点击**：战斗/加载/结算期间只观察，超时才失败。
    """

    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        p = self._param(argv)
        nodes = [str(n) for n in (p.get("nodes") or [])]
        timeout = float(p.get("timeout") or 120)
        interval = float(p.get("interval") or 2)
        if not nodes:
            log("WaitUntil 缺 nodes 参数")
            return False

        deadline = time.time() + timeout
        while time.time() < deadline:
            image = _screencap(context)
            for node in nodes:
                if _hit(context, node, image):
                    log(f"WaitUntil：命中 {node}")
                    return True
            time.sleep(interval)
        log(f"WaitUntil 超时（{timeout}s）仍未命中 {nodes}")
        return False
