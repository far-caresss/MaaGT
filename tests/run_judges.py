# -*- coding: utf-8 -*-
"""判据回归矩阵（离线、真机帧、PASS/FAIL 退出码）。

**为什么要有这个**：本仓的判据是 `page_base.json` 里一堆 OCR/TemplateMatch 的 ROI，
改一个坐标就可能把别的页判错，而"判错页"在真机上表现为"卡住不动"，很难从日志看出来。
2026-10-31 一天就在真机上发现 6 个"从建库起就没命中过"的判据，全部是单测覆盖不到的类型
（单测只测纯函数）。所以把真机帧固定下来，改判据先跑这里。

用法（在仓库根目录）::

    python tests/run_judges.py              # 跑全部（实测 371 次识别 / 约 7 分钟）
    python tests/run_judges.py --fast       # 只验 expect，不验 reject（实测约 22 秒）
    python tests/run_judges.py --strict-coverage   # CI：待补帧也必须齐全
    python tests/run_judges.py --frame pvp44.png --verbose

规则（`tests/cases.json`）：每帧给 `expect`（必须命中）和 `allow`（允许同时命中），
其余所有 `Page*` / `Dlg*` / `Btn*` / `Val*` 都必须**不命中**。

依赖：`pip install maa-framework Pillow numpy`（与 `assets/agent` 同一套）。

注意事项（都是踩过的）：
1. 用 `CustomController` 喂静态帧，`click` 全部 no-op —— 所以跑 `Btn*` 也**不会真的点**；
2. 每个节点都显式给 `timeout`（默认 20s 空转会把这个套件变成 1 小时），实测 300ms ≈ 0.7s/节点；
3. **不要**用 `Page*` 节点当载体、也不要加 `pipeline_override` 去改被测节点——override 会污染
   同一 context 里其它 `run_recognition`（踩坑 §1.11）；
4. 新增 `Page*`/`Dlg*` 节点**必须**同时在 `cases.json` 里给它一个正样本帧，否则 coverage 会红；
5. 抓不到真机帧的判据登记进 `cases.json:pending_coverage`（附补帧方法）——默认只提示不阻塞，
   加 `--strict-coverage` 才让它红。理由：一条永远红的检查等于没有检查；
   注意"待补帧"只免除**正样本**要求，该节点照常参与每帧的 reject 断言。
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from maa.controller import CustomController
from maa.resource import Resource
from maa.tasker import Tasker
from maa.toolkit import Toolkit

#: 控制台通常还是 GBK：`cases.json` 的 `desc` 里带 `⚠️` 这类字符会直接把整个矩阵跑挂
#: （2026-11-06 实测 `UnicodeEncodeError: 'gbk' codec can't encode character '\u26a0'`，
#: 而且崩在打印 desc 那一行，看起来像"测试框架坏了"）。`backslashreplace` 让编不出的字符
#: 退化成 `\u26a0` 字面量 —— 不至于因为一条说明文字把全量回归废掉。
try:
    sys.stdout.reconfigure(errors="backslashreplace")
except Exception:  # noqa: BLE001 - 老 Python / 重定向到非文本流
    pass

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ASSETS = ROOT / "assets"
CASES = HERE / "cases.json"
GOLDEN = HERE / "golden"

#: 需要"被判定"的节点前缀（按钮也纳入，防止它们乱命中）
JUDGE_PREFIXES = ("Page", "Dlg", "Btn")
#: **读取器**不当判据：`Val*` 是 `only_rec` 的"读数"节点（`x/y`、金币数…），
#: 只要 ROI 里有任何文字就算命中，天然不适合当"该不该命中"的断言。
#: 它们仍然会被 `expect` 检查（用来验证能读到），只是不参与 reject。
READER_PREFIXES = ("Val",)
#: 只有这些前缀要求"至少有一帧正样本"（按钮可能在别的活动里才出现，不强制）
COVERAGE_PREFIXES = ("Page", "Dlg")
#: 不在前缀规范里、但同样要当判据断言的任务内分叉节点
EXTRA_JUDGE_NODES = ("Pvp44ResultLoss", "Pvp44ResultWin")


class FrameController(CustomController):
    """把一张静态帧喂给 MaaFramework；所有输入动作都是 no-op。"""

    def __init__(self, img_bgr: np.ndarray):
        self._img = img_bgr
        super().__init__()

    def connect(self) -> bool:
        return True

    def request_uuid(self) -> str:
        return "judge-matrix"

    def start_app(self, intent: str) -> bool:
        return False

    def stop_app(self, intent: str) -> bool:
        return False

    def screencap(self) -> np.ndarray:
        return self._img

    def click(self, x: int, y: int) -> bool:
        return True

    def swipe(self, x1, y1, x2, y2, duration) -> bool:
        return True

    def touch_down(self, contact, x, y, pressure) -> bool:
        return True

    def touch_move(self, contact, x, y, pressure) -> bool:
        return True

    def touch_up(self, contact) -> bool:
        return True

    def click_key(self, keycode: int) -> bool:
        return True

    def input_text(self, text: str) -> bool:
        return True

    def key_down(self, keycode: int) -> bool:
        return True

    def key_up(self, keycode: int) -> bool:
        return True


def load_all_nodes() -> dict[str, dict]:
    nodes: dict[str, dict] = {}
    for f in glob.glob(str(ASSETS / "resource" / "pipeline" / "*.json")):
        nodes.update(json.loads(Path(f).read_text(encoding="utf-8")))
    return nodes


def judge_nodes(nodes: dict[str, dict]) -> list[str]:
    picked = {k for k in nodes if k.startswith(JUDGE_PREFIXES)}
    picked |= {k for k in EXTRA_JUDGE_NODES if k in nodes}
    return sorted(picked)


def run_node(tasker: Tasker, node: str, timeout_ms: int) -> bool:
    """跑单个节点，返回**识别**是否命中。**不改被测节点的识别参数**（只压 timeout）。

    ⚠️ 判定口径必须是 `recognition.hit`，**不能**用 `job.succeeded`：
    带 `action: Custom` 的节点（`Pvp44ResultLoss/Win` 的 `DumpFrame`/`TapUnlessConsumes`）
    在离线环境下没有 agent 注册，框架报 `Action is null` 并把 job 判失败——
    识别明明命中了却会被记成"应命中但没有"（2026-10-31 实测踩到）。
    本套件只关心判据本身，动作是否可执行不在断言范围内。
    """
    override = {
        "Default": {"timeout": timeout_ms, "rate_limit": 0, "post_delay": 0},
        node: {"timeout": timeout_ms},
    }
    job = tasker.post_task(node, override).wait()
    detail = tasker.get_task_detail(job.job_id)
    if detail is None or not detail.nodes:
        return False
    rd = detail.nodes[0].recognition
    return bool(rd is not None and rd.hit)


class Trace:
    """边跑边写进度文件：终端要等整条命令结束才回显，看不到日志会以为卡住了。"""

    def __init__(self, path: Path | None):
        self.path = path
        self.t0 = time.time()
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")

    def __call__(self, msg: str) -> None:
        line = f"{time.time() - self.t0:7.1f}s  {msg}"
        print(line, flush=True)
        if self.path is not None:
            try:
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
            except Exception:  # noqa: BLE001
                pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=int, default=300, help="单节点识别预算(ms)，默认 300")
    ap.add_argument("--fast", action="store_true", help="只验 expect，跳过 reject")
    ap.add_argument("--frame", help="只跑指定帧（文件名）")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument(
        "--trace",
        default=str(ROOT / "debug" / "judge_matrix.log"),
        help="实时进度文件（默认 debug/judge_matrix.log）",
    )
    ap.add_argument("--no-trace", action="store_true")
    ap.add_argument(
        "--strict-coverage",
        action="store_true",
        help="把 pending_coverage（已知缺真机帧）也算失败（CI 用）",
    )
    args = ap.parse_args()
    trace = Trace(None if args.no_trace else Path(args.trace))

    cases = json.loads(CASES.read_text(encoding="utf-8"))
    frames = cases["frames"]
    if args.frame:
        frames = [f for f in frames if f["file"] == args.frame]
        if not frames:
            print(f"[FAIL] cases.json 里没有帧 {args.frame}")
            return 1

    missing = [f["file"] for f in frames if not (GOLDEN / f["file"]).exists()]
    if missing:
        print(f"[FAIL] 缺金标准帧：{missing}（放 {GOLDEN}）")
        return 1

    Toolkit.init_option(str(ASSETS))
    res = Resource()
    if not res.post_bundle(str(ASSETS / "resource")).wait().succeeded:
        print("[FAIL] resource.post_bundle 失败")
        return 1
    all_nodes = load_all_nodes()
    universe = judge_nodes(all_nodes)

    # ---- coverage：每个 Page*/Dlg* 都得有正样本 ----
    pending = cases.get("pending_coverage") or []
    pending_nodes = {str(it.get("node")) for it in pending if isinstance(it, dict)}
    covered = {n for f in cases["frames"] for n in f["expect"]}
    need = {k for k in universe if k.startswith(COVERAGE_PREFIXES)}
    uncovered = sorted(need - covered - pending_nodes)
    failures: list[str] = []
    if uncovered:
        failures.append(f"coverage: 这些判据没有正样本帧 → {uncovered}")

    # 待补帧：默认只提示（永远红的检查等于没有检查），--strict-coverage 才计入失败。
    # 注意它们仍然在 `universe` 里，每帧的 reject 断言照常覆盖。
    stale = sorted(pending_nodes - need)
    if stale:
        failures.append(f"coverage: pending_coverage 里不是 Page*/Dlg* 判据（登记有误）→ {stale}")
    real_pending = sorted(pending_nodes & (need - covered))
    if real_pending:
        msg = f"coverage: {len(real_pending)} 项判据缺真机正样本帧 → {real_pending}"
        if args.strict_coverage:
            failures.append(msg)
        else:
            trace(f"[待补] {msg}（加 --strict-coverage 可让它变红）")
            for it in pending:
                if isinstance(it, dict) and str(it.get("node")) in real_pending:
                    trace(f"    {it.get('node')}: {it.get('reason', '（未写原因）')}")

    # 先算总步数，好给"第几/共几 + 预计剩余"
    plan: list[tuple[dict, list[str], list[str]]] = []
    for case in frames:
        expect = list(case.get("expect") or [])
        allow = set(case.get("allow") or [])
        reject = [] if args.fast else [n for n in universe if n not in expect and n not in allow]
        plan.append((case, expect, reject))
    total_ops = sum(len(e) + len(r) for _, e, r in plan)

    trace(f"开始：{len(frames)} 帧 / {len(universe)} 判据 / {total_ops} 次识别 / timeout={args.timeout}ms")
    if not args.no_trace:
        trace(f"进度文件：{args.trace}（可以 tail -f 看）")

    done = 0
    t_start = time.time()
    for case, expect, reject in plan:
        path = GOLDEN / case["file"]
        rgb = np.asarray(Image.open(path).convert("RGB"))
        ctrl = FrameController(np.ascontiguousarray(rgb[:, :, ::-1]))  # 框架给的是 BGR
        if not ctrl.post_connection().wait().succeeded:
            trace(f"[FAIL] {case['file']} 控制器连接失败")
            return 1
        tasker = Tasker()
        if not tasker.bind(res, ctrl):
            trace(f"[FAIL] {case['file']} tasker.bind 失败")
            return 1

        trace(f"── {case['file']}  {case.get('desc', '')}（{len(expect)} 正 / {len(reject)} 负）")
        bad: list[str] = []
        for kind, nodes, want_hit in (("期望命中", expect, True), ("不得命中", reject, False)):
            for node in nodes:
                t0 = time.time()
                ok = run_node(tasker, node, args.timeout)
                done += 1
                cost = (time.time() - t0) * 1000
                verdict = "OK" if ok == want_hit else "!!"
                eta = (time.time() - t_start) / done * (total_ops - done)
                line = (f"[{done:>3}/{total_ops}] {verdict} {kind} {node:<20} "
                        f"hit={str(ok):<5} {cost:5.0f}ms  剩余≈{eta:4.0f}s")
                if ok == want_hit:
                    if args.verbose:
                        trace(line)
                else:
                    trace(line)
                    bad.append(f"{case['file']}: {node} {'应命中但没有' if want_hit else '不该命中却命中了'}")
        failures.extend(bad)
        trace(f"   小结 {case['file']}: {'通过' if not bad else f'{len(bad)} 项失败'}")

    cost = time.time() - t_start
    trace(f"跑完：{done} 次识别，用时 {cost:.1f}s")
    if failures:
        trace(f"[FAILED] {len(failures)} 项：")
        for f in failures:
            trace("  - " + f)
        return 1
    trace("[ALL PASS]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
