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

当前已登记 **15 个页面**（`PAGE_NODES`）：
Main / MenuOpen / GuideRift / Guide / Shop / Forge / Pvp44 / Task / Activity / Guild / Friend /
RaidDetail / RaidList / Dig / Arena。

**6 个弹窗**（`DIALOG_NODES`，自上而下收）：GameUpdate / Notice / BattleLog / TipSent / NoticeAck / Reward。
`DlgGameUpdate`（游戏内「下载资源」）排第一位：冷启动和首次进活动页必弹，是模态的。

**18 条导航边**（`NAV_TABLE`）。⚠️ 查表是**单键** `(当前页, 目标页)`，
**不会自动把多条边拼成多跳** —— 需要两跳就必须显式写全（坑见 `踩坑记录.md` §6.17）。

新增页面 = `page_base.json` 加一个 `Page*` 节点（**≥2 个独立信号**）+ 在 `PAGE_NODES` 登记
+ `tests/cases.json` 加正样本帧。判据 ROI 标定后**必须**用 `_tmp/offline_ocr.py roi` 回测，
确认 `all=[...]` 里有完整关键词（只框到半句话一样 `hit=True`，坑见 §6.15）。
"""

from __future__ import annotations

import json
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

_TAP_WAIT = 0.4
#: 页面切换的**上界**，不是固定休眠：点完之后改为轮询"期望页判据"，命中就走（踩坑 §1.8 反对盲等）
_PAGE_WAIT = 3.0
_POLL_INTERVAL = 0.2
#: ⚠️ 别低于 15：`close_dialogs()` 每收一个弹窗就消耗一步，而**开屏广告轮播实测要点 9 次**
#: （2026-11-07 真机 `TapWhilePresent BtnAdClose：点了 9 次`）。原来写 8，连广告都收不完就报
#: 「认不出当前页」——表现为任意任务在弹广告时开局即失败。
_MAX_STEPS = 20

PAGE_UNKNOWN = "PageUnknown"
PAGE_MAIN = "PageMain"
PAGE_MENU = "PageMenuOpen"
PAGE_GUIDE = "PageGuide"
PAGE_GUIDE_RIFT = "PageGuideRift"
PAGE_SHOP = "PageShop"
PAGE_FORGE_EVOLVE = "PageForgeEvolve"
PAGE_FORGE = "PageForge"
PAGE_PVP44 = "PagePvp44"
PAGE_DAILY_TASK = "PageDailyTask"
PAGE_TASK = "PageTask"
PAGE_ACTIVITY = "PageActivity"
PAGE_GUILD = "PageGuild"
PAGE_FRIEND = "PageFriend"
PAGE_RAID_LIST = "PageRaidList"
PAGE_RAID_DETAIL = "PageRaidDetail"
PAGE_DIG = "PageDig"
PAGE_ARENA = "PageArena"

#: 判据节点（顺序 = 探测优先级）。子页必须排在其父页**前面**，
#: 否则 `PageGuide` 会先命中、`PageGuideRift` 永远命中不到。
#: `PageMenuOpen` 必须排在 `PageMain` 前：菜单是覆盖层，打开时底栏仍在，
#: `PageMain` 同时成立（N5）。
PAGE_NODES: tuple[str, ...] = (
    PAGE_MENU,
    PAGE_MAIN,
    PAGE_GUIDE_RIFT,
    PAGE_GUIDE,
    PAGE_SHOP,
    PAGE_FORGE_EVOLVE,
    PAGE_FORGE,
    PAGE_PVP44,
    PAGE_DAILY_TASK,
    PAGE_TASK,
    PAGE_ACTIVITY,
    PAGE_GUILD,
    PAGE_FRIEND,
    PAGE_RAID_DETAIL,
    PAGE_RAID_LIST,
    PAGE_DIG,
    PAGE_ARENA,
)

#: 弹窗判据，自上而下收（列表顺序 = 优先级）。
#: `DlgNotice` 在 `DlgReward` 之前：`提示` 类是**模态确认框**，它开着的时候点空白处没反应。
#: `DlgBattleLog` 必须在 `DlgReward` 之前：44 的「战斗记录」会**自动弹出**且盖住剩余次数，
#: 不收掉的话 `ValPvpTickets` 读不到 `x/y`。
DIALOG_NODES: tuple[str, ...] = ("DlgAdLayer", "DlgAnnouncement", "DlgAutoBattle", "DlgInnPanel", "DlgSweepResult", "DlgGameUpdate", "DlgNotice", "DlgBattleLog", "DlgTipSent", "DlgNoticeAck", "DlgReward")

#: 弹窗关闭动作：`("point", (x, y))` 或 `("node", 节点名)`
DIALOG_CLOSERS: dict[str, tuple[str, Any]] = {
    # 开屏礼包广告轮播：`BtnAdClose` 是 `And{DlgAdLayer}` + target `(1240,41)`，
    # 只点右上 ✕、永不碰黄色价格按钮（哪怕写着 ¥328）。
    "DlgAdLayer": ("node", "BtnAdClose"),
    # 公告浮层：`BtnAnnouncementClose` 是 `And{DlgAnnouncement}` + target `(1228,77)`。
    # 2026-11-14 才登记：它和广告层同属「全屏遮罩」，不登记就会重演当天那三次一串任务全挂。
    # ⚠️ 不要用合并判据 `DlgStartup` 当关闭依据 —— 它 `Or{DlgAdLayer, DlgAnnouncement}`，
    # 两种浮层的 ✕ 位置不同（1240,41 vs 1228,77），会用错坐标。
    "DlgAnnouncement": ("node", "BtnAnnouncementClose"),
    # 「自动战斗设置」：**只点取消**。右边那个绿「扫荡」会按面板数字扣体力（真机帧写 62 (-10)）
    "DlgAutoBattle": ("node", "BtnAutoBattleCancel"),
    # 「扫荡」结算面板：点底部「确认」关掉，不再扣资源
    "DlgSweepResult": ("node", "BtnSweepResultConfirm"),
    # 「25级旅馆」底部抽屉：点上方空白收起（没有 ✕）
    "DlgInnPanel": ("node", "BtnInnPanelClose"),
    # 游戏内「下载资源」：点确认只下载，不花钱不消耗 → 可以自动确认
    "DlgGameUpdate": ("node", "BtnGameUpdateConfirm"),
    # 「提示」类确认弹窗：只点「取消」（安全侧）。`确定退出游戏?` 点确认会把游戏关掉。
    "DlgNotice": ("node", "BtnNoticeCancel"),
    # 好友页「已经向所有好友发送礼物」：**只有确认没有取消**，所以单列一条（判据绑死正文，不会误点别的提示框）
    "DlgTipSent": ("node", "BtnTipSentConfirm"),
    # 任务指南「今日活跃度不足，请先完成日常任务。」：同样只有确认，判据绑死正文，点确认不花钱
    "DlgNoticeAck": ("node", "BtnNoticeAck"),
    # 「战斗记录」：点右上 ✕（判据化的按钮节点，不是盲点坐标）
    "DlgBattleLog": ("node", "BtnBattleLogClose"),
    # 「点击空白处关闭弹窗」类奖励弹窗：右上角空白处（与 guild_raid 的 TAP_BLANK_TO_CLOSE 一致）
    "DlgReward": ("point", (1236, 48)),
}

#: 回锚点用的步骤节点（`page_nav.json`）：左上白色返回箭头，所有子页都有，0 消耗。
#: **不用旧的 `Return`**：它的 `next` 是 6 个 `[JumpBack]` 节点（`取消` / `确认` 是**全屏** OCR），
#: 单独 `run_task("Return")` 实测要等 60s+ 超时链，而且可能在任意位置命中 `取消` 后盲点一次。
RECOVER_NODE = "NavBack"

#: 导航表：(当前页, 目标页) -> ((动作节点, 期望到达页或 None), ...)
#: 期望页为 None 表示"这一跳没有判据"，只打日志不校验（过渡期用）。
#: 每条边的最后一跳**必须**有判据，否则 `NavTo` 会拒绝执行整条路线。
NAV_TABLE: dict[tuple[str, str], tuple[tuple[str, str | None], ...]] = {
    (PAGE_MAIN, PAGE_GUILD): (("公会", PAGE_GUILD),),
    (PAGE_MAIN, PAGE_TASK): (("NavMenu", PAGE_MENU), ("NavMenuTask", PAGE_TASK)),
    (PAGE_MAIN, PAGE_FRIEND): (("NavMenu", PAGE_MENU), ("NavFriend", PAGE_FRIEND)),
    (PAGE_MAIN, PAGE_FORGE): (("NavMenu", PAGE_MENU), ("强化", PAGE_FORGE)),
    (PAGE_MAIN, PAGE_SHOP): (("商店", PAGE_SHOP),),
    (PAGE_MAIN, PAGE_ACTIVITY): (("活动", PAGE_ACTIVITY),),
    (PAGE_MAIN, PAGE_GUIDE): (("游玩", PAGE_GUIDE),),
    (PAGE_MAIN, PAGE_GUIDE_RIFT): (("游玩", PAGE_GUIDE), ("裂痕", PAGE_GUIDE_RIFT)),
    (PAGE_MAIN, PAGE_PVP44): (("游玩", PAGE_GUIDE), ("44", PAGE_PVP44)),
    (PAGE_GUIDE, PAGE_PVP44): (("44", PAGE_PVP44),),
    (PAGE_GUIDE, PAGE_GUIDE_RIFT): (("裂痕", PAGE_GUIDE_RIFT),),
    # 公会讨伐：详情页 → 列表 → 大厅，都是左上 `<` 一步退回。
    # 2026-11-01 真机实测：没有这两条边时走"经 PageMain 中转"的兜底，
    # 会连点两次 NavBack 且各白等 3s（实际没到 PageMain，`_wait_for` 必然超时）；
    # 加上直达边后 12.2s → 约 2s。
    (PAGE_RAID_DETAIL, PAGE_RAID_LIST): ((RECOVER_NODE, PAGE_RAID_LIST),),
    (PAGE_RAID_LIST, PAGE_GUILD): ((RECOVER_NODE, PAGE_GUILD),),
    # `NAV_TABLE` 只按 (当前页, 目标页) 单键查表，**不会自动拼接多条边**，
    # 所以"详情页 → 公会大厅"这种两跳必须显式写全；否则会掉进"经 PageMain 中转"的兜底，
    # 多白等一次 3s 的 `_wait_for(PageMain)`（那个等待在讨伐页里必然超时）。
    (PAGE_RAID_DETAIL, PAGE_GUILD): (
        (RECOVER_NODE, PAGE_RAID_LIST),
        (RECOVER_NODE, PAGE_GUILD),
    ),
    # 已经在菜单面板里（覆盖层）：只需再点一格
    # 菜单是**覆盖层**，`NavBack` 在主页那一带是头像（白色头像会被判据命中）→ 关菜单只能再点一次「菜单」
    (PAGE_MENU, PAGE_MAIN): (("NavMenu", PAGE_MAIN),),
    (PAGE_MENU, PAGE_TASK): (("NavMenuTask", PAGE_TASK),),
    (PAGE_MENU, PAGE_FRIEND): (("NavFriend", PAGE_FRIEND),),
    (PAGE_MENU, PAGE_FORGE): (("强化", PAGE_FORGE),),
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


_DUMP_DIR = _ASSETS_DIR.parent / "debug"


def dump_screen(image: np.ndarray | None, name: str = "probe_live") -> None:
    """把**交给 `run_recognition` 的那张数组**原样落盘（排查"判据结果与离线不一致"用）。"""
    if image is None:
        log("dump_screen：没有截图")
        return
    try:
        from PIL import Image

        _DUMP_DIR.mkdir(parents=True, exist_ok=True)
        out = _DUMP_DIR / f"{name}.png"
        arr = image
        if arr.ndim == 3 and arr.shape[2] == 3:  # 框架给的是 BGR，落盘转回 RGB
            arr = arr[:, :, ::-1]
        Image.fromarray(np.ascontiguousarray(arr)).save(out)
        log(f"dump_screen → {out} shape={image.shape}")
    except Exception as exc:  # noqa: BLE001
        log(f"dump_screen 失败：{exc}")


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


def _node_click_target(context: Context, node: str) -> tuple[int, int, int, int] | None:
    """读节点自己配置的 `action.param.target`（`[x,y,w,h]`）。`And` 这类判据节点没有 `best_result.box`，
    但它可以带固定点击目标（例：`BtnBattleLogClose` 判弹窗、点 ✕ 坐标）。"""
    try:
        data = context.get_node_data(node)
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(data, dict):
        return None
    action = data.get("action") or {}
    target = ((action.get("param") or {}) if isinstance(action, dict) else {}).get("target")
    if isinstance(target, (list, tuple)) and len(target) == 4:
        return tuple(int(v) for v in target)  # type: ignore[return-value]
    return None


def _click_node(context: Context, image: np.ndarray | None, node: str) -> bool:
    """先判定再点击：命中才点（绝不盲点）。

    点击位置**优先用节点自己配的 `action.param.target`**，没配才用识别框中心。
    ⚠️ 2026-11-01 纠正：原来是"框优先"，但 `And` / `Or` 这类**组合节点也会带框**——
    实测 `BtnAdClose`（`And{DlgAdLayer}` + target `(1240,41)`）的框是子判据的
    `(179,601,107,17)`（底部「2周内不再提醒」那行字），于是点击落到了 (232,609) 而不是 ✕；
    `BtnBattleLogClose` 同理点了「战斗记录」标题。凡是显式写了 target 的节点，
    作者想点的就是那个坐标，框中心只应该当兜底。
    """
    if image is None:
        return False
    try:
        detail = context.run_recognition(node, image)
    except Exception as exc:  # noqa: BLE001
        log(f"点击前判定 {node} 失败：{exc}")
        return False
    if detail is None or not getattr(detail, "hit", False):
        log(f"点击前判定 {node} 未命中 → 不点")
        return False
    box = _node_click_target(context, node) or _box_of(detail)
    if box is None:
        log(f"点击前判定 {node} 命中但没有识别框、也没配 target → 不点")
        return False
    x, y, w, h = box
    _tap(context, x + w // 2, y + h // 2)
    log(f"已点击 {node} @({x + w // 2},{y + h // 2})")
    return True


def _wait_for(context: Context, node: str | None, timeout: float = _PAGE_WAIT) -> bool:
    """轮询"期望页判据"直到命中，最多 `timeout` 秒。

    替代原来的 `time.sleep(_PAGE_WAIT)` 盲等：命中就立刻返回（实测一次 screencap + 单条判据
    约 0.3s，页面切得快时能省掉 1s 以上），超时也不报错——留给调用方下一轮 `probe_page` 判定，
    这样"慢一点"和"走错页"仍然由判据说话，而不是由 sleep 时长说话。
    """
    if not node:
        time.sleep(_TAP_WAIT)
        return False
    deadline = time.time() + max(0.0, timeout)
    while time.time() < deadline:
        if _hit(context, node, _screencap(context)):
            log(f"等待 {node}：命中（{timeout:.1f}s 预算内）")
            return True
        time.sleep(_POLL_INTERVAL)
    log(f"等待 {node}：{timeout:.1f}s 内未命中")
    return False


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


#: 多跳拼接的上限。2–3 跳足够覆盖"详情页 → 列表页 → 大厅"这类层级，
#: 再长基本就是没有直达路径、应该补边或走锚点中转。
_MAX_ROUTE_HOPS = 3


def _compose_route(
    here: str, target: str
) -> tuple[tuple[str, str | None], ...] | None:
    """BFS 把已有边拼成多跳路线（最多 `_MAX_ROUTE_HOPS` 跳）。

    ⚠️ 2026-11-06：原来 `NAV_TABLE` 只按 `(当前页, 目标页)` 单键查，
    **相邻两页各写一条是不够的** —— 从 `PageRaidDetail` 到 `PageGuild` 明明有
    `Detail→List`、`List→Guild`，却查不到，只能掉进「经 PageMain 中转」的兜底，
    白等一次 3s 的 `_wait_for(PageMain)`（那个等待在讨伐页里必然超时，踩坑 §6.17）。
    """
    from collections import deque

    queue: deque[tuple[str, tuple[tuple[str, str | None], ...]]] = deque([(here, ())])
    seen = {here}
    while queue:
        node, path = queue.popleft()
        if len(path) >= _MAX_ROUTE_HOPS:
            continue
        for (src, dst), edge in NAV_TABLE.items():
            if src != node or dst in seen:
                continue
            new_path = path + edge
            if dst == target:
                return new_path
            seen.add(dst)
            queue.append((dst, new_path))
    return None


def route_of(here: str, target: str) -> tuple[tuple[str, str | None], ...] | None:
    """查导航表；认不出当前页时退化为"从锚点出发"（先回主界面）。

    直达边优先（显式写的比拼出来的可信）；没有再 BFS 拼接。
    """
    route = NAV_TABLE.get((here, target))
    if route:
        return route
    if here == PAGE_UNKNOWN:
        return NAV_TABLE.get((PAGE_MAIN, target))
    return _compose_route(here, target)


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
            # 不盲等：下一轮循环会重新截屏 + 复探，弹窗真收掉了自然会往下走
            time.sleep(_TAP_WAIT)
            break
    return closed


def _recover_to_main(context: Context, image: np.ndarray | None = None) -> bool:
    """回锚点：点左上返回箭头一步（有判据才点，0 消耗）。

    ⚠️ `NavBack` 判据是**纯白 ColorMatch**，主页左上角正是白色头像 → 在主页上也会命中
    （2026-10-31 `ProbeNodes` 实测：主页 `NavBack hit=True box=(29,21,44,20)`）。
    所以先确认自己不在锚点，否则会把头像点开。
    """
    img = image if image is not None else _screencap(context)
    if probe_page(context, img) == PAGE_MAIN:
        log(f"回锚点：当前已在 {PAGE_MAIN}，不点 {RECOVER_NODE}")
        return True
    if not _click_node(context, img, RECOVER_NODE):
        log(f"回锚点：找不到 {RECOVER_NODE}（当前不在可返回的子页）")
        return False
    _wait_for(context, PAGE_MAIN)
    return True


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
            # 认不出当前页。典型场景：**网页活动页**（全屏 webview，只剩左上 `<`，没有标题/底栏）——
            # 这种页没有可判的稳定元素，而框架没有 `Not` 原语，"有箭头且没有标题"没法表达，
            # 所以不硬造 `PageWebEvent`。但返回箭头本身是**判据化**的（`NavBack`），
            # 因此允许"退一步再看"，这跟盲点不是一回事（守则 §2 反对的是点没判据的东西）。
            log("NavTo：认不出当前页 → 尝试用返回箭头退一步（有判据才点）")
            if not _recover_to_main(context, image):
                log("NavTo：认不出且没有返回箭头 → 不点任何东西（守则 §2）")
                return False
            continue

        route = route_of(here, target)
        if route is None:
            # 架构 §3.5 的 `* → PageMain` 兜底边：任何子页回锚点都走同一步 `NavBack`，
            # 不必给每个子页单独写一条边，也不点任何"看起来像返回"的东西。
            if target == PAGE_MAIN:
                log(f"{here} → {PAGE_MAIN} 没有直达边 → 走回锚点 {RECOVER_NODE}")
                if not _recover_to_main(context, image):
                    log("NavTo 失败：回锚点没成功")
                    return False
                continue
            # 子页 → 子页：**走锚点中转**（枢纽模型）。否则要写 O(n²) 条边，
            # 而且每加一个页面都要回头补一整套。前提是目标页能从锚点直达。
            if here not in (PAGE_MAIN, PAGE_MENU) and route_of(PAGE_MAIN, target) is not None:
                log(f"{here} → {target} 无直达边 → 经 {PAGE_MAIN} 中转")
                if not _recover_to_main(context, image):
                    log("NavTo 失败：回锚点没成功")
                    return False
                continue
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
        _wait_for(context, expect)

    log(f"NavTo 失败：超过 {max_steps} 步仍未到达 {target}")
    return False


# ======================================================================
# 图内原语（Custom Action）
# ======================================================================

class _Base(CustomAction):
    @staticmethod
    def _param(argv: Any) -> dict[str, Any]:
        """`custom_action_param` 在本仓用的框架版本里是 **JSON 字符串**，不是 dict。

        （`maa/custom_action.py` 的 `RunArg` 注明 `custom_action_param: str`；
        批次 0 的单测只覆盖了纯函数，所以这个坑一直到真机才暴露：
        7 个原语全部拿不到参数，`NavTo` 一律报"缺 target"。）
        这里同时兼容 str / dict / 直接传字符串，解析失败只当"没参数"，不抛出。
        """
        raw = argv if isinstance(argv, str) else getattr(argv, "custom_action_param", None)
        if not raw:
            return {}
        if isinstance(raw, dict):
            return raw
        try:
            parsed = json.loads(raw)
        except Exception as exc:  # noqa: BLE001
            log(f"custom_action_param 不是合法 JSON：{exc!r}（原文 {raw!r}）")
            return {}
        return parsed if isinstance(parsed, dict) else {}


@AgentServer.custom_action("ProbeNodes")
class ProbeNodesAction(_Base):
    """诊断用：`{"nodes": ["PageMain", "识别颜色"], "dump": true}`

    用**同一张**截图（也就是 `NavTo` 内部真正喂给 `run_recognition` 的那张数组）逐个判据跑一遍，
    把命中框 / OCR 文字 / 图像形状打进日志，并把这张数组落盘。
    存在的理由：`post_task(节点)` 与 `run_recognition(节点, image)` 两条路径出现过结论不一致，
    没有这个原语就只能靠猜。
    """

    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        p = self._param(argv)
        nodes = [str(n) for n in (p.get("nodes") or [])]
        image = _screencap(context)
        if image is None:
            log("ProbeNodes：截图为空")
            return False
        log(f"ProbeNodes：shape={image.shape} dtype={image.dtype}")
        if p.get("dump"):
            dump_screen(image)
        for node in nodes:
            try:
                detail = context.run_recognition(node, image)
            except Exception as exc:  # noqa: BLE001
                log(f"  {node}: 异常 {exc}")
                continue
            if detail is None:
                log(f"  {node}: None")
                continue
            texts = [str(getattr(r, "text", "") or "") for r in (getattr(detail, "all_results", None) or [])]
            log(f"  {node}: hit={detail.hit} box={_box_of(detail)} texts={[t for t in texts if t][:4]}")
        return True


def diagnose_nav_failure(context: Context, target: str) -> None:
    """`NavTo` 失败时留证：dump 一帧 + 打印**当时命中了哪些判据**。

    为什么要有：2026-11-14 一天里 `DlgAdLayer` / `DlgAutoBattle` / `DlgSweepResult` 三个遮罩
    各让一串任务全挂，日志只有一句「回锚点没成功」，全靠框架的 `debug/on_error/*.png` 才看清
    是哪个弹窗。而 `on_error` 是**任务级**失败才截，`NavTo` 内部失败未必有。
    这里主动截一张，并把 PAGE_NODES / DIALOG_NODES 的命中情况一起打出来 ——
    下次直接看日志就知道卡在什么弹窗，不用再猜。
    """
    image = _screencap(context)
    if image is None:
        log("NavTo 失败诊断：截图为空")
        return
    dump_screen(image, f"nav_fail_{time.strftime('%m%d_%H%M%S')}")
    hits = [node for node in (PAGE_NODES + DIALOG_NODES) if _hit(context, node, image)]
    log(f"NavTo 失败诊断（目标 {target}）：命中判据 = {hits or '（一个都没命中）'}")


@AgentServer.custom_action("NavTo")
class NavToAction(_Base):
    """`{"target": "PageGuild", "max_steps": 8}`"""

    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        p = self._param(argv)
        target = str(p.get("target") or "")
        if not target:
            log("NavTo 缺 target 参数")
            return False
        ok = nav_to(context, target, int(p.get("max_steps") or _MAX_STEPS))
        if not ok:
            diagnose_nav_failure(context, target)
        return ok


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
        raw = p.get("inner")
        # `inner` 可以是单个节点名，也可以是**列表**（2026-11-01 加）：
        # 开屏一轮里可能先后出现「点击开始」「公告 ✕」「广告 ✕」，每轮只点命中率最高的那一个。
        inner_nodes = ([str(n) for n in raw] if isinstance(raw, list)
                       else [str(raw)] if raw else [])
        count = int(p.get("repeat_count") or 1)
        wait_nodes = [str(n) for n in (p.get("wait_nodes") or [])]
        # 重复间隔与"页面切换上界"是两件事，别复用 _PAGE_WAIT（3s 一轮太慢）
        interval = float(p.get("interval") or 1.0)
        blank = p.get("tap_blank") or ("tap_blank" in inner_nodes)

        for i in range(1, count + 1):
            image = _screencap(context)
            for node in wait_nodes:
                if _hit(context, node, image):
                    log(f"RepeatUntilFound 第 {i}/{count} 次：命中 {node} → 成功")
                    return True
            if blank:
                _tap(context, 1236, 48)
                log(f"RepeatUntilFound 第 {i}/{count} 次：点空白处")
            elif inner_nodes:
                clicked = ""
                for node in inner_nodes:
                    if _click_node(context, image, node):
                        clicked = node
                        break
                if clicked:
                    log(f"RepeatUntilFound 第 {i}/{count} 次：点了 {clicked}")
                else:
                    log(f"RepeatUntilFound 第 {i}/{count} 次：{inner_nodes} 都未命中")
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


@AgentServer.custom_action("DumpFrame")
class DumpFrameAction(_Base):
    """`{"tag": "pvp_loss"}` —— **只落盘、不点击、不改变状态**。

    用途：拿不到"正样本"的场景自动留档。典型：44 战败帧（人手打不出假输），
    让链路在判到 `失败` 的瞬间存一张 `debug/pvp_loss_<月日_时分秒>.png`，
    事后直接拿 `_tmp/offline_ocr.py` 把判据钉死。
    """

    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        p = self._param(argv)
        tag = str(p.get("tag") or "frame")
        image = _screencap(context)
        if image is None:
            log(f"DumpFrame({tag})：截图为空")
            return False
        dump_screen(image, f"{tag}_{time.strftime('%m%d_%H%M%S')}")
        return True


@AgentServer.custom_action("TapWhilePresent")
class TapWhilePresentAction(_Base):
    """`{"button": "刷新", "max_times": 3, "interval": 1.0}` —— 只要按钮还在画面上就继续点。

    `button` 传单个节点名；也可以传 `buttons`（列表）：每轮按顺序试，**第一个命中的**才点，
    一轮只点一下。2026-11-14 为"主界面旅店上方那一排奖励图标"加的 —— 那一排有金币/粉心/蓝紫
    等多个模板，串成一条链要点很多节点，用列表一个节点就够。

    用途（44）：开打前先把"已经打过的卡"用 ↻ 换掉。用户 2026-10-31 的规则：
    **点「开始战斗」之前先看这张卡右上角有没有 ↻，有就先点 ↻ 再打**，否则会一直撞同一个对手。
    每次点击都走 `_click_node`（先判定再点，绝不盲点）；按钮不在就正常返回（0 次点击 = 无需处理）。

    **每轮先 `close_dialogs()`**（2026-11-01 加）：模态弹窗压着时点按钮没反应。
    最典型的是每日签到——点一格会弹「领取奖励」，不收掉就没法接着点下一格。
    弹窗被收掉时 `continue`（不计数、重新截图），所以 `max_times` 要按"点击次数 + 收弹窗次数"估。
    """

    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        p = self._param(argv)
        # 两种写法：`button`（单个，向后兼容）或 `buttons`（列表，每轮试第一个命中的）
        raw_list = p.get("buttons")
        if isinstance(raw_list, list) and raw_list:
            buttons = [str(b) for b in raw_list if b]
        else:
            single = str(p.get("button") or "")
            buttons = [single] if single else []
        max_times = int(p.get("max_times") or 3)
        interval = float(p.get("interval") or 1.0)
        if not buttons:
            log("TapWhilePresent 缺 button / buttons 参数")
            return False
        label = "/".join(buttons)
        times = 0
        for _ in range(max_times):
            image = _screencap(context)
            if close_dialogs(context, image):
                # 模态弹窗被收掉 → 重新截图再判，不算一次点击
                continue
            hit = next((b for b in buttons if _hit(context, b, image)), None)
            if hit is None:
                break
            if not _click_node(context, image, hit):
                break
            times += 1
            log(f"TapWhilePresent {hit}：第 {times}/{max_times} 次")
            time.sleep(interval)
        # 收尾：最后一次点击可能又弹了奖励窗
        close_dialogs(context, _screencap(context))
        log(f"TapWhilePresent {label}：点了 {times} 次（上限 {max_times}）")
        return True


@AgentServer.custom_action("SwipeSlider")
class SwipeSliderAction(_Base):
    """`{"begin": [x1,y1], "end": [x2,y2], "duration": 1000}` —— 把数量滑块拖到最右（拉满）。

    2026-11-14 为周常「买英雄成长觉醒石」加：商店的数量选择是一个**长轨道 + 蓝色圆点**，
    左右各一个「-」「+」按钮，但商品上限可能很大（3/3、60/60…），点「+」要按很多次。
    直接按住圆点拖到轨道右端，一次到位。

    真机实测（1280x720，商店→英雄成长→买觉醒石）：圆点 `(432,530)`、轨道右端 `(860,530)`、
    `duration=1000` 能把数量从 1 拉到上限 3（价格 15,000 → 45,000）。

    ⚠️ **只负责拖，不负责按确认**：确认购买是消耗动作，必须由调用方在判据成立后单独点。
    """

    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        p = self._param(argv)
        begin = p.get("begin") or []
        end = p.get("end") or []
        if len(begin) < 2 or len(end) < 2:
            log("SwipeSlider 缺 begin / end 参数")
            return False
        dur = int(p.get("duration") or 1000)
        x1, y1 = int(begin[0]), int(begin[1])
        x2, y2 = int(end[0]), int(end[1])
        log(f"SwipeSlider：({x1},{y1}) -> ({x2},{y2})  {dur}ms")
        try:
            context.tasker.controller.post_swipe(x1, y1, x2, y2, dur).wait()
        except Exception as exc:  # noqa: BLE001
            log(f"SwipeSlider 失败：{exc!r}")
            return False
        time.sleep(0.6)
        return True


@AgentServer.custom_action("BuyShopItem")
class BuyShopItemAction(_Base):
    """`{"target": [x,y], "begin": [x1,y1], "end": [x2,y2], "consumes": "坎特币", "cost": "...", "allow_consume": false}`

    **买商店里的一个商品**，一条龙：点格子 → 等数量弹窗 → 滑块拉满 → 过闸门 → 点价格确认。

    为什么要做成原语：周常要买好几个商品（英雄成长 1 格 + 忒提斯英雄传前 5 格），
    每个商品的差别只有「点哪一格」，其余步骤完全一样。用 pipeline 拼的话每种商品要 4 个节点
    （Pick/WaitDlg/SliderMax/Confirm）→ 6 种商品就是 24 个节点；做成原语只要 6 个节点，
    而且"滑块拉满"的坐标只有一处、改一次全生效。

    ⚠️ **闸门在点确认之前**：`consumes` 非空且 `allow_consume` 为假 → 直接返回失败，
    **数量滑块已经拉满、但一下都没花**。用户界面开关通过 `pipeline_override` 打开 `allow_consume`。
    """

    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        p = self._param(argv)
        target = p.get("target") or []
        begin = p.get("begin") or [432, 530]
        end = p.get("end") or [860, 530]
        dur = int(p.get("duration") or 1000)
        consumes = p.get("consumes")
        cost = p.get("cost")
        allow = bool(p.get("allow_consume"))
        if len(target) < 2:
            log("BuyShopItem 缺 target 参数")
            return False

        # 1) 点商品格（只弹窗，不花钱）
        x, y = int(target[0]), int(target[1])
        log(f"BuyShopItem：点商品格 ({x},{y})")
        context.tasker.controller.post_click(x, y).wait()
        time.sleep(1.2)

        # 2) 等数量弹窗
        if not _wait_for(context, "DlgBuyQuantity", 8.0):
            log("BuyShopItem：数量弹窗没出现 → 停在这里（不滑、不买）")
            return False

        # 3) 滑块拉满
        log(f"BuyShopItem：滑块 ({begin[0]},{begin[1]}) -> ({end[0]},{end[1]})")
        context.tasker.controller.post_swipe(
            int(begin[0]), int(begin[1]), int(end[0]), int(end[1]), dur
        ).wait()
        time.sleep(0.8)

        # 4) 消耗闸门
        if consumes and not allow:
            log(f"BuyShopItem：已拉满数量，但拒绝确认 —— 这单要花 {consumes}（{cost or '代价未知'}），需要界面开关放开")
            # ⚠️ 必须把弹窗关掉再返回：留着它会挡住后面所有步骤
            # （2026-11-14 真机踩到：拒绝后弹窗压着，后续滑左侧栏/识别分类全部失效，流程卡死）
            if _click_node(context, _screencap(context), "BtnBuyCancel"):
                log("BuyShopItem：已点「取消」关掉弹窗，恢复现场")
                # 1.5s 不是随便写的：0.8s 时弹窗还在淡出，紧接着点下一格会被吃掉
                # （2026-11-14 实测：连买 5 格，前 2 格成功、后 3 格「弹窗没出现」）
                time.sleep(1.5)
            else:
                log("BuyShopItem：点不到「取消」，弹窗可能仍压着画面")
            return False

        # 5) 确认购买
        image = _screencap(context)
        if not _click_node(context, image, "BtnBuyConfirm"):
            log("BuyShopItem：点不到价格按钮")
            return False
        log("BuyShopItem：已点确认购买")
        time.sleep(1.5)
        close_dialogs(context, _screencap(context))
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
    """`{"button": "节点名", "consumes": "gold", "cost": "1,000", "allow_consume": false, "optional": false}`

    消耗闸门：`consumes` 非空且 `allow_consume` 为假 → **拒绝点击**并记日志。
    `consumes` 取 `real_money` 时**永不**放行（只认 PI 开关不够，要改代码）。

    `optional: true` → 按钮不在画面上时**不算失败**（只打日志）。用于"补救动作"
    （例：44 战败后换对手的 `刷新`）——没有这个按钮不该让整轮任务失败。
    ⚠️ `optional` 只放宽"按钮不存在"，**不**放宽消耗闸门：闸门拒绝仍然返回失败。
    """

    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        p = self._param(argv)
        button = str(p.get("button") or "")
        consumes = p.get("consumes")
        cost = p.get("cost")
        allow = bool(p.get("allow_consume"))
        optional = bool(p.get("optional"))
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
            return True
        if optional:
            log(f"{button} 不在画面上 → optional，跳过后继续")
            return True
        return False


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
