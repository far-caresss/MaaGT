# -*- coding: utf-8 -*-
"""公会活动统一入口：进公会大厅 → 点最上方活动卡 → 结构判据识别 → 分派给对应流程。

设计要点（2026-10-02 与用户确认，详见 `docs/local/公会协力讨伐-开发进度.md` §14，本地专属不入库）：

1. **单一入口 + 自动分派**：任务列表里只暴露「公会活动」一个任务，
   识别出当前是哪个活动后再分派，用户不用管本期轮到哪个。
2. **结构特征为主，名字只当日志**：公会 3 个活动轮换，当前周期的那个显示在最上方；
   但**活动名每期都变**（讨伐本期叫「死城频率」，下期可能换），
   所以判据一律用各活动页面独有的稳定元素，**绝不写死名字**。
3. **安全优先**：认不出当前是哪个活动时，**不点任何按钮**，直接退出并告警。
   点错卡片可能进到别的活动，接着乱点就可能误触消耗（见文档 §9 纪律）。
4. **从任意页面都能进来（方案 A）**：导航在 agent 侧完成，**复用既有 `Return` 节点**
   （`pipeline/startup.json`，与其它任务的 `next: ["[JumpBack]Return", …]` 同一套用法）：
   关弹窗（取消/确认/点击空白处/人物窗）→ 逐层点「返回」→ 回到主界面（旅店）。
   到主界面后点底栏「公会」进大厅，再读最上方活动卡。
   见 `_reach_activity` 的窗口→动作表。
"""

from __future__ import annotations

import re
import time

import numpy as np

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction
from maa.define import OCRResult

import guild_arena
import guild_dig
import guild_raid
from guild_raid import (
    LOBBY_KEYWORD,
    NODE_ENTRIES,
    NODE_LOBBY_PANEL,
    NODE_GUILD_ENTRY,
    NODE_PLAY_CARD,
    NODE_POPUP_HINT,
    PAGE_WAIT,
    TAP_BLANK_TO_CLOSE,
    TAP_MAIN_GUILD,
    TAP_WAIT,
    _to_box,
    detect_bosses,
    load_glyph_templates,
    log,
)

ACTIVITY_RAID = "raid"
ACTIVITY_ARENA = "arena"
ACTIVITY_DIG = "dig"

ACTIVITY_CN = {
    ACTIVITY_RAID: "公会协力讨伐",
    ACTIVITY_ARENA: "地牢王国竞技场",
    ACTIVITY_DIG: "陨石挖掘计划",
}

#: 判据节点（定义在 pipeline/guild_activity.json）
NODE_ARENA_TABS = "ArenaTabs"
NODE_ARENA_STATUS = "ArenaStatus"
NODE_DIG_PANEL = "DigPanel"

#: 竞技场判据：**任一**强关键词命中即可（实测 OCR 会把「首领信息」读成「领信息」，
#: 要求两个词都命中太脆，会让路由白跑一趟导航）
ARENA_STRONG = ("兑换所", "今日竞技场")
#: 「首领信息」的拆词兜底（整词被 OCR 切坏时仍能认）
ARENA_PAIR = ("首领", "信息")
#: 挖掘判据关键词：右侧面板的「挖掘进度 / 十字镐」
DIG_KEYWORDS = ("挖掘进度", "十字镐")

#: 既有导航节点（`pipeline/startup.json`）——**复用，不自造**
NODE_RETURN_TASK = "Return"      # 关弹窗 + 逐层返回；末尾「停止」= 主界面判定
NODE_MAIN_COLOR = "识别颜色"      # 主界面（旅店）背景色
NODE_MAIN_MENU = "识别菜单"       # 主界面左下「菜单」

#: 导航步数上限：每步都打日志，超了就安全退出（不盲点、不死循环）
MAX_NAV_STEPS = 8

#: `剩余挑战次数 x/3` —— 讨伐判据之一
_BATTLES_RE = re.compile(r"(\d+)\s*/\s*(\d+)")
#: 活动卡/标题文字必须含中文（名字每期变，不写死，但别把 HUD 数字当卡片）
_CJK_RE = re.compile(r"[\u4e00-\u9fff]{2,}")


# ======================================================================
# 活动判据（探针）：返回活动标识；返回 None = 认不出
# ======================================================================

def _probe_raid(context: Context, image: np.ndarray, glyphs: dict) -> str | None:
    """讨伐判据：读到 `剩余挑战次数 x/3` **且** 至少认出一个 BOSS 属性字形。

    两个信号都要：单看 `x/y` 形式的数字太弱（别的活动页也可能有计数），
    而 4 个 BOSS 属性图标是讨伐列表页独有的。
    """
    detail = context.run_recognition(NODE_ENTRIES, image)
    text = ""
    if detail is not None and detail.hit and detail.best_result is not None:
        best = detail.best_result
        if isinstance(best, OCRResult):
            text = best.text
    if not _BATTLES_RE.search(text or ""):
        log(f"讨伐判据：没读到 `x/y` 形式的剩余次数（读到 {text!r}）")
        return None

    bosses = [b for b in detect_bosses(image, glyphs) if b.element]
    if not bosses:
        log("讨伐判据：次数形态像讨伐，但 0 个 BOSS 属性字形识别成功 → 判为无法确认")
        return None
    log(f"讨伐判据：剩余次数={text!r}，认出 {len(bosses)} 个 BOSS 属性")
    return ACTIVITY_RAID


def _ocr_all_text(context: Context, node: str, image: np.ndarray) -> str:
    """把一个 OCR 节点 ROI 内的所有文字拼起来（判据只看关键词，不看位置）。"""
    detail = context.run_recognition(node, image)
    if detail is None or not detail.hit:
        return ""
    parts = []
    for result in detail.all_results:
        if isinstance(result, OCRResult) and result.text:
            parts.append(result.text)
    return " ".join(parts)


def _probe_arena(context: Context, image: np.ndarray, glyphs: dict) -> str | None:
    """竞技场判据：任一强关键词命中即可（多信号容错）。

    两个 ROI 都是实测的（2026-10-02，游戏停在该页）：
    - `ArenaTabs [960,655,290,40]`：兑换所 (981,667,51,19) / 排名 (1088,669) / 首领信息 (1173,668)
    - `ArenaStatus [935,198,160,34]`：「今日竞技场状况」(950,207,112,19)

    ⚠️ 实测 OCR 会把「首领信息」读成「领信息」，所以**不要求整词**：
    任一强词命中就算（这两个词都是竞技场独有），另有拆词兜底。
    """
    tabs = _ocr_all_text(context, NODE_ARENA_TABS, image)
    status = _ocr_all_text(context, NODE_ARENA_STATUS, image)
    text = f"{tabs} {status}"

    hits = [kw for kw in ARENA_STRONG if kw in text]
    if not hits and all(kw in tabs for kw in ARENA_PAIR):
        hits = ["首领+信息(拆词)"]
    if not hits:
        log(f"竞技场判据：没读到关键词（页签={tabs!r} 状态={status!r}）")
        return None
    log(f"竞技场判据：命中 {hits}")
    return ACTIVITY_ARENA


def _probe_dig(context: Context, image: np.ndarray, glyphs: dict) -> str | None:
    """挖掘判据：右侧面板读到 `挖掘进度` 或 `十字镐`。

    ⚠️ ROI `DigPanel [930,240,350,320]` 目前是**按截图估的宽松区域**（还没实机量），
    但判据只要求"命中关键词"、不需要精确点击，所以宽松是安全的；
    等游戏停在挖掘页时用 `_tmp/measure_activity.py` 量一次再把 ROI 收紧。
    """
    text = _ocr_all_text(context, NODE_DIG_PANEL, image)
    hits = [kw for kw in DIG_KEYWORDS if kw in text]
    if not hits:
        log(f"挖掘判据：面板区没读到关键词（文字={text!r}）")
        return None
    log(f"挖掘判据：命中关键词 {hits}")
    return ACTIVITY_DIG


#: 探针顺序 = 判据优先级（互斥判据，顺序无所谓，但保持稳定便于复现）
PROBES = (_probe_raid, _probe_arena, _probe_dig)


# ======================================================================
# 入口
# ======================================================================

@AgentServer.custom_action("GuildActivityRouter")
class GuildActivityRouter(CustomAction):
    """「公会活动」任务的入口：识别当前活动 → 分派。"""

    _glyphs: dict = {}

    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        param = getattr(argv, "custom_action_param", "") or ""
        try:
            ok = self._run(context, param)
        except Exception as exc:  # noqa: BLE001 - 顶层兜底，避免中断框架
            log(f"异常终止：{exc!r}")
            import traceback

            log(traceback.format_exc())
            return CustomAction.RunResult(success=False)
        return CustomAction.RunResult(success=ok)

    # ------------------------------------------------------------------
    def _run(self, context: Context, param: str) -> bool:
        glyphs = load_glyph_templates()
        if not glyphs:
            log("字形模板缺失，终止")
            return False
        self._glyphs = glyphs

        activity = self._reach_activity(context, glyphs)
        if activity is None:
            log("⚠️ 无法确认当前是哪个公会活动 —— 已停止操作，安全退出（见 §14 方案 A）")
            return False

        log(f"当前活动 = {ACTIVITY_CN.get(activity, activity)}")
        return self._dispatch(context, activity, param)

    def _identify(self, context: Context, glyphs: dict) -> str | None:
        """拿当前屏幕跑一遍所有探针。"""
        image = self._screencap(context)
        if image is None:
            log("截图失败")
            return None
        for probe in PROBES:
            activity = probe(context, image, glyphs)
            if activity:
                log(f"识别命中：{ACTIVITY_CN.get(activity, activity)}（{probe.__name__}）")
                return activity
        return None

    def _reach_activity(self, context: Context, glyphs: dict) -> str | None:
        """从**任意页面**识别出当前活动；必要时收拾页面回到主界面再走。

        每步先认窗口、再决定动作（方案 A）：

        | 当前窗口 | 动作 |
        |---|---|
        | 某个活动页 | 直接用（`_identify`） |
        | 公会大厅 | 读最上方活动卡 → 点卡 → 再识别 |
        | 主界面（旅店） | 点底栏「公会」 |
        | 其它（弹窗 / 子页） | 跑既有 `Return` 节点：关弹窗 + 逐层返回 |

        ⚠️ 认不出窗口时**不盲点固定坐标**，只跑 `Return`；步数用尽就安全退出。
        """
        for step in range(1, MAX_NAV_STEPS + 1):
            activity = self._identify(context, glyphs)
            if activity:
                return activity

            image = self._screencap(context)
            if image is None:
                log("截图失败，终止导航")
                return None

            if self._in_guild_lobby(context, image):
                log(f"[{step}/{MAX_NAV_STEPS}] 当前窗口 = 公会大厅")
                activity = self._open_top_card(context, glyphs)
                if activity:
                    return activity
                log(f"[{step}/{MAX_NAV_STEPS}] 大厅里点开最上方活动卡后仍认不出，稍后重试")
                time.sleep(PAGE_WAIT)
                continue

            if self._in_main_screen(context, image):
                log(f"[{step}/{MAX_NAV_STEPS}] 当前窗口 = 主界面（旅店），点底栏「公会」")
                self._tap_guild_entry(context)
                time.sleep(PAGE_WAIT)
                self._close_popup_if_present(context)
                continue

            log(f"[{step}/{MAX_NAV_STEPS}] 当前窗口 = 认不出（弹窗 / 子页），用 Return 收拾页面")
            self._run_return(context)
            time.sleep(PAGE_WAIT)

        log(f"⚠️ 导航 {MAX_NAV_STEPS} 步仍未抵达公会活动，安全退出")
        return None

    def _open_top_card(self, context: Context, glyphs: dict) -> str | None:
        """在大厅里读最上方活动卡 → 点它 → 识别活动。

        轮换活动里当前周期那一张显示在最上方，所以"点第一张卡"是稳的先验；
        卡上的名字只读出来打日志（每期都变，不能当判据）。
        """
        self._close_popup_if_present(context)
        card = self._activity_card_box(context, self._screencap(context))
        if card is None:
            log("公会大厅「最上方活动卡」未识别")
            return None

        box, text = card
        log(f"最上方活动卡文字 = {text!r}（仅作日志：活动名每期都变）")
        x, y, w, h = box
        self._tap(context, x + w // 2, y + h // 2)
        time.sleep(PAGE_WAIT * 1.5)
        self._close_popup_if_present(context)
        return self._identify(context, glyphs)

    def _run_return(self, context: Context) -> bool:
        """跑既有 `Return` 节点：关弹窗（取消/确认/点击空白处/人物窗）+ 逐层返回。

        它在 `startup.json` 里写了 `max_hit: 1`（同一次任务里只跑一遍），
        所以循环调用前必须先清命中计数，否则第二次会被框架直接跳过。
        """
        context.clear_hit_count(NODE_RETURN_TASK)
        detail = context.run_task(NODE_RETURN_TASK)
        ok = bool(detail and detail.status.succeeded)
        log(f"Return 完成：succeeded={ok}")
        return ok

    def _in_main_screen(self, context: Context, image: np.ndarray | None) -> bool:
        """主界面（旅店）判据：复用既有节点 `识别颜色` + `识别菜单`（= `停止` 节点同款）。"""
        if image is None:
            return False
        color = context.run_recognition(NODE_MAIN_COLOR, image)
        menu = context.run_recognition(NODE_MAIN_MENU, image)
        return bool(color and color.hit and menu and menu.hit)

    def _dispatch(self, context: Context, activity: str, param: str) -> bool:
        """把控制权交给对应流程。"""
        if activity == ACTIVITY_RAID:
            # 显式传 param，不依赖框架的 pipeline_override 继承（见 guild_raid.run_raid）；
            # nav 让讨伐流程丢页时把控制权交回本类，走同一套"带判据、不盲点"的导航
            return guild_raid.run_raid(context, param, nav=self._nav_to_raid)
        if activity == ACTIVITY_ARENA:
            return guild_arena.run(context)
        if activity == ACTIVITY_DIG:
            return guild_dig.run(context)
        log(f"未知活动标识 {activity!r}，不执行任何操作")
        return False

    def _nav_to_raid(self, context: Context) -> bool:
        """给讨伐流程用的"重新进入活动"回调（它丢页时会调这里）。

        复用本类的 `_reach_activity`：回主界面 / 进大厅 / 点最上方卡，全程带判据、不盲点。
        """
        log("讨伐流程请求重新进入活动（走统一导航）")
        return self._reach_activity(context, self._glyphs) == ACTIVITY_RAID

    # ------------------------------------------------------------------
    # 基础能力（与 guild_raid 同款的小工具；不跨类调私有方法，避免耦合）
    # ------------------------------------------------------------------
    def _screencap(self, context: Context) -> np.ndarray | None:
        image = context.tasker.controller.post_screencap().wait().get()
        if image is None:
            return None
        return np.asarray(image)

    def _tap(self, context: Context, x: int, y: int) -> None:
        context.tasker.controller.post_click(int(x), int(y)).wait()
        time.sleep(TAP_WAIT)

    def _ocr_results(
        self, context: Context, node: str, image: np.ndarray
    ) -> list[tuple[str, tuple[int, int, int, int]]]:
        detail = context.run_recognition(node, image)
        if detail is None or not detail.hit:
            return []
        out: list[tuple[str, tuple[int, int, int, int]]] = []
        for result in detail.all_results:
            if not isinstance(result, OCRResult):
                continue
            box = _to_box(result.box)
            if box is not None:
                out.append((result.text, box))
        return out

    def _in_guild_lobby(self, context: Context, image: np.ndarray | None) -> bool:
        """是否在公会大厅：右侧面板标题「公会玩法」可读（固定 UI 文案）。"""
        if image is None:
            return False
        return LOBBY_KEYWORD in _ocr_all_text(context, NODE_LOBBY_PANEL, image)

    def _activity_card_box(
        self, context: Context, image: np.ndarray | None
    ) -> tuple[tuple[int, int, int, int], str] | None:
        """公会大厅「最上方活动卡」的框 + 卡上文字（含中文）。

        先确认在公会大厅：活动卡那一带的文字在别的活动页也会出现
        （实测竞技场页同位置是「未找到记录」），光靠 ROI 收紧治不了。
        """
        if image is None or not self._in_guild_lobby(context, image):
            return None
        for text, box in self._ocr_results(context, NODE_PLAY_CARD, image):
            if _CJK_RE.search(text or ""):
                return box, text
        return None

    def _tap_guild_entry(self, context: Context) -> bool:
        """点底栏「公会」；OCR 不到就退回固定坐标。"""
        image = self._screencap(context)
        if image is not None:
            detail = context.run_recognition(NODE_GUILD_ENTRY, image)
            if detail is not None and detail.hit:
                box = _to_box(detail.box)
                if box is not None:
                    x, y, w, h = box
                    log(f"{NODE_GUILD_ENTRY} -> ({x + w // 2},{y + h // 2})")
                    self._tap(context, x + w // 2, y + h // 2)
                    return True
        log(f"{NODE_GUILD_ENTRY} 未识别，退回固定坐标 {TAP_MAIN_GUILD}")
        self._tap(context, *TAP_MAIN_GUILD)
        return False

    def _close_popup_if_present(self, context: Context) -> bool:
        """入口/结算可能弹领奖页：点空白处关闭。"""
        image = self._screencap(context)
        if image is None:
            return False
        detail = context.run_recognition(NODE_POPUP_HINT, image)
        if detail is None or not detail.hit:
            return False
        log("检测到弹窗，点击空白处关闭")
        self._tap(context, *TAP_BLANK_TO_CLOSE)
        time.sleep(PAGE_WAIT)
        return True
