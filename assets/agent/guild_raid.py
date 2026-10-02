# -*- coding: utf-8 -*-
"""公会协力讨伐：识别首领属性 → 按配置队伍分配 → 逐个作战。

设计文档：docs/zh_cn/公会协力讨伐-设计方案.md

流程（已实测，坐标基于 1280x720）：
    首领列表页 --图标框中心+(120,54)--> 详情页
    详情页 --≡(770,647)--> 「选择预设」弹窗 --点名字框中心--> 详情页
    详情页 --演习(905,647)/入场(1115,647)--> 战斗 --自动战斗--> 结果 --退出--> 首领列表页

v1 使用「演习」（不消耗次数）；验证通过后把 PRACTICE_MODE 改为 False 走「入场」。
"""

from __future__ import annotations

import itertools
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction
from maa.define import OCRResult

# ======================================================================
# 常量
# ======================================================================

#: 属性相克环，箭头 = 前者克后者
COUNTER: dict[str, str] = {
    "fire": "earth",
    "earth": "water",
    "water": "fire",
    "light": "dark",
    "dark": "virtual",
    "virtual": "light",
}
ELEMENTS: tuple[str, ...] = ("fire", "water", "earth", "light", "dark", "virtual")
ELEMENT_CN: dict[str, str] = {
    "fire": "火", "water": "水", "earth": "土",
    "light": "光", "dark": "暗", "virtual": "虚",
}

RELATION_COUNTER = "counter"      # 队伍克制首领
RELATION_NEUTRAL = "neutral"      # 互不克制
RELATION_COUNTERED = "countered"  # 队伍被首领克制

#: v1 = True 走「演习」（不消耗次数）；验证通过后改 False 走「入场」
PRACTICE_MODE = True

# ---------------- 坐标 ----------------
TAP_MAIN_GUILD = (238, 675)        # 兜底坐标（优先用 OCR 定位）
TAP_GUILD_PLAY_CARD = (1100, 185)  # 兜底坐标（优先用 OCR 定位）

#: 首领列表页 4 个槽位的属性图标 ROI (x, y, w, h)
BOSS_ICON_ROIS: tuple[tuple[int, int, int, int], ...] = (
    (34, 376, 52, 52),
    (460, 229, 52, 52),
    (913, 256, 52, 52),
    (374, 526, 52, 52),
)
#: 图标外接框中心 → 首领详情页点击点的偏移
ICON_TO_ITEM_OFFSET = (120, 54)

TAP_DETAIL_PRESET = (770, 647)     # ≡
TAP_DETAIL_PRACTICE = (905, 647)   # 演习
TAP_DETAIL_ENTER = (1115, 647)     # 入场
TAP_PRESET_CLOSE = (985, 64)       # 「选择预设」弹窗关闭 X
TAP_BLANK_TO_CLOSE = (1236, 48)    # 「点击空白处关闭弹窗」的点击位置

#: 弹窗「预设设置」按钮：用它判断弹窗是否已经打开
TAP_PRESET_SETTINGS = (640, 632)

PRESET_SCROLL_FROM = (640, 540)
PRESET_SCROLL_TO = (640, 240)
PRESET_SCROLL_MAX = 8

#: 「选择预设」弹窗内行的点击 x。实测点名字文字无效，必须点行中间。
PRESET_ROW_TAP_X = 640

# ---------------- 字形匹配 ----------------
GLYPH_WHITE_THRESHOLD = 205
GLYPH_SIZE = 24
GLYPH_MIN_IOU = 0.55

# ---------------- 路径 ----------------
_ASSETS_DIR = Path(__file__).resolve().parent.parent            # .../assets
TEMPLATE_DIR = _ASSETS_DIR / "resource" / "image" / "Guild" / "Element"
LOG_FILE = _ASSETS_DIR.parent / "debug" / "guild_raid.log"
CONFIG_CANDIDATES: tuple[Path, ...] = (
    _ASSETS_DIR / "config" / "teams.json",
    Path.cwd() / "config" / "teams.json",
    Path.cwd() / "assets" / "config" / "teams.json",
)

# ---------------- 节奏（秒） ----------------
TAP_WAIT = 0.6
PAGE_WAIT = 1.5
BATTLE_START_TIMEOUT = 40
BATTLE_END_TIMEOUT = 120

#: 识别节点名（定义在 pipeline/guild_raid.json）
NODE_ENTRIES = "RaidEntries"
NODE_PRESET_NAMES = "RaidPresetNames"
NODE_AUTO_BATTLE = "RaidAutoBattle"
NODE_RESULT_EXIT = "RaidResultExit"
NODE_POPUP_HINT = "RaidPopupHint"
NODE_ENTER_BUTTON = "RaidEnterButton"
NODE_GUILD_ENTRY = "RaidGuildEntry"
NODE_PLAY_CARD = "RaidPlayCard"

#: 去掉 JSONC 的尾逗号
_TRAILING_COMMA = re.compile(r",(\s*[}\]])")


def log(msg: str) -> None:
    """打印并落盘。插件会把 agent 的 stdout 收进自己的终端，不落盘不好排查。"""
    print(f"[GuildRaid] {msg}", flush=True)
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with LOG_FILE.open("a", encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
    except Exception:  # noqa: BLE001 - 日志失败不能影响主流程
        pass


def _to_box(raw: object) -> tuple[int, int, int, int] | None:
    """把识别框统一成 (x, y, w, h)。

    注意：maa 反序列化时 box 是 **list**（`[x, y, w, h]`）而不是 `Rect` 对象，
    两种形式都要兼容。
    """
    if raw is None:
        return None
    if hasattr(raw, "x"):
        return (int(raw.x), int(raw.y), int(raw.w), int(raw.h))
    try:
        x, y, w, h = raw  # type: ignore[misc]
        return (int(x), int(y), int(w), int(h))
    except Exception:  # noqa: BLE001
        return None


# ======================================================================
# 数据结构
# ======================================================================

@dataclass(frozen=True)
class Team:
    """配置里的一支队伍。"""

    name: str
    element: str


@dataclass(frozen=True)
class Boss:
    """首领列表页上的一个槽位。"""

    slot: int
    element: str | None      # None = 识别失败


@dataclass(frozen=True)
class Assignment:
    """一对象（首领槽位, 队伍）。"""

    boss_slot: int
    team: Team
    relation: str


# ======================================================================
# 配置与模板
# ======================================================================

def _strip_jsonc(text: str) -> str:
    """去掉 // 与 /* */ 注释，以及尾逗号，使其成为标准 JSON。"""
    out: list[str] = []
    i, n = 0, len(text)
    in_str = False
    while i < n:
        ch = text[i]
        if in_str:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 2
                continue
            if ch == '"':
                in_str = False
            i += 1
            continue
        if ch == '"':
            in_str = True
            out.append(ch)
            i += 1
            continue
        if ch == "/" and i + 1 < n and text[i + 1] == "/":
            while i < n and text[i] not in "\r\n":
                i += 1
            continue
        if ch == "/" and i + 1 < n and text[i + 1] == "*":
            i += 2
            while i + 1 < n and not (text[i] == "*" and text[i + 1] == "/"):
                i += 1
            i += 2
            continue
        out.append(ch)
        i += 1
    return _TRAILING_COMMA.sub(r"\1", "".join(out))


def load_teams() -> list[Team]:
    """读取 teams.json，返回配置顺序（= 平局优先级）的队伍列表。"""
    for path in CONFIG_CANDIDATES:
        if not path.exists():
            continue
        data = json.loads(_strip_jsonc(path.read_text(encoding="utf-8")))
        teams: list[Team] = []
        for item in data.get("teams", []):
            name = str(item.get("name", "")).strip()
            element = str(item.get("element", "")).strip().lower()
            if not name:
                log(f"配置跳过：缺少 name -> {item}")
                continue
            if element not in ELEMENTS:
                log(f"配置跳过：属性非法 {element!r} -> {name}")
                continue
            teams.append(Team(name=name, element=element))
        log(f"已加载队伍({path.name})：" + "、".join(f"{t.name}/{t.element}" for t in teams))
        return teams

    log("未找到队伍配置，候选路径：" + " | ".join(str(p) for p in CONFIG_CANDIDATES))
    return []


def load_glyph_templates() -> dict[str, np.ndarray]:
    """加载 6 张属性字形模板（24x24 二值）。"""
    templates: dict[str, np.ndarray] = {}
    if not TEMPLATE_DIR.exists():
        log(f"模板目录不存在：{TEMPLATE_DIR}")
        return templates
    for name in ELEMENTS:
        path = TEMPLATE_DIR / f"{name}.png"
        if not path.exists():
            continue
        gray = np.asarray(Image.open(path).convert("L"))
        templates[name] = gray > 127
    log("已加载字形模板：" + "、".join(sorted(templates)))
    return templates


# ======================================================================
# 纯函数：属性识别 / 分配
# ======================================================================

def _glyph_mask(image: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray | None:
    """取 ROI 内的白色字形，归一化到 GLYPH_SIZE。"""
    x, y, w, h = roi
    sub = image[y:y + h, x:x + w]
    if sub.size == 0:
        return None
    mask = (
        (sub[:, :, 0] > GLYPH_WHITE_THRESHOLD)
        & (sub[:, :, 1] > GLYPH_WHITE_THRESHOLD)
        & (sub[:, :, 2] > GLYPH_WHITE_THRESHOLD)
    )
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    tight = (mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1] * 255).astype(np.uint8)
    resized = Image.fromarray(tight).resize((GLYPH_SIZE, GLYPH_SIZE), Image.BILINEAR)
    return np.asarray(resized) > 127


def classify_glyph(
    mask: np.ndarray | None, templates: dict[str, np.ndarray]
) -> tuple[str | None, float]:
    """返回 (属性, IoU)；低于阈值返回 (None, 最高分)。"""
    if mask is None or not templates:
        return None, 0.0
    best_name, best_iou = None, 0.0
    for name, tpl in templates.items():
        inter = np.logical_and(mask, tpl).sum()
        union = np.logical_or(mask, tpl).sum()
        iou = float(inter / union) if union else 0.0
        if iou > best_iou:
            best_name, best_iou = name, iou
    if best_iou < GLYPH_MIN_IOU:
        return None, best_iou
    return best_name, best_iou


def detect_bosses(image: np.ndarray, templates: dict[str, np.ndarray]) -> list[Boss]:
    """识别 4 个首领槽位的属性。"""
    bosses: list[Boss] = []
    for slot, roi in enumerate(BOSS_ICON_ROIS):
        element, iou = classify_glyph(_glyph_mask(image, roi), templates)
        bosses.append(Boss(slot=slot, element=element))
        log(f"槽位{slot} 属性={ELEMENT_CN.get(element or '', '未知')} IoU={iou:.3f}")
    return bosses


def relation(team_element: str, boss_element: str) -> str:
    """队伍属性 vs 首领属性 的克制关系。"""
    if COUNTER.get(team_element) == boss_element:
        return RELATION_COUNTER
    if COUNTER.get(boss_element) == team_element:
        return RELATION_COUNTERED
    return RELATION_NEUTRAL


def _rank_better(
    a: tuple[int, int, tuple[int, ...]], b: tuple[int, int, tuple[int, ...]]
) -> bool:
    """a 是否优于 b：先比克制数、再比中性数，最后优先配置靠前的队伍。"""
    if a[0] != b[0]:
        return a[0] > b[0]
    if a[1] != b[1]:
        return a[1] > b[1]
    return a[2] < b[2]


def choose_plan(
    bosses: Sequence[Boss],
    teams: Sequence[Team],
    battles: int,
    banned_teams: Iterable[str] = (),
) -> list[Assignment]:
    """选 battles 对 (首领, 队伍)，首领与队伍均不重复。

    目标（字典序）：
      1. 最大化「克制」对数
      2. 最大化「中性」对数（等价于最小化「被克」对数）
      3. 平局：优先使用配置里靠前的队伍
    """
    banned = set(banned_teams)
    usable_bosses = [b for b in bosses if b.element]
    usable_teams = [(i, t) for i, t in enumerate(teams) if t.name not in banned]

    k = min(battles, len(usable_bosses), len(usable_teams))
    if k <= 0:
        log("可用首领或队伍不足，无法分配")
        return []

    best_rank: tuple[int, int, tuple[int, ...]] | None = None
    best_plan: list[Assignment] = []

    for team_combo in itertools.combinations(usable_teams, k):
        for boss_combo in itertools.permutations(usable_bosses, k):
            assigns: list[Assignment] = []
            counters = neutrals = 0
            for (_, team), boss in zip(team_combo, boss_combo):
                rel = relation(team.element, boss.element or "")
                if rel == RELATION_COUNTER:
                    counters += 1
                elif rel == RELATION_NEUTRAL:
                    neutrals += 1
                assigns.append(Assignment(boss.slot, team, rel))
            rank = (counters, neutrals, tuple(sorted(idx for idx, _ in team_combo)))
            if best_rank is None or _rank_better(rank, best_rank):
                best_rank, best_plan = rank, assigns

    log(f"分配方案：克制 {best_rank[0]} 对 / 中性 {best_rank[1]} 对 / 共 {len(best_plan)} 场")
    for a in best_plan:
        boss_el = next((b.element for b in bosses if b.slot == a.boss_slot), None)
        log(
            f"  → 槽位{a.boss_slot}({ELEMENT_CN.get(boss_el or '', '?')})"
            f" ← {a.team.name}({ELEMENT_CN.get(a.team.element, '?')}) [{a.relation}]"
        )
    return best_plan


# ======================================================================
# 编排动作
# ======================================================================

@AgentServer.custom_action("GuildRaidOrchestrator")
class GuildRaidOrchestrator(CustomAction):
    """公会协力讨伐总编排。"""

    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        try:
            self._run(context)
        except Exception as exc:  # noqa: BLE001 - 顶层兜底，避免中断框架
            log(f"异常终止：{exc!r}")
            import traceback

            log(traceback.format_exc())
            return CustomAction.RunResult(success=False)
        return CustomAction.RunResult(success=True)

    # ------------------------------------------------------------------
    def _run(self, context: Context) -> None:
        teams = load_teams()
        templates = load_glyph_templates()
        if not teams or not templates:
            log("配置或模板缺失，终止")
            return

        if not self._goto_boss_list(context):
            log("未能抵达首领列表页，终止")
            return

        image = self._screencap(context)
        if image is None:
            log("截图失败，终止")
            return

        bosses = detect_bosses(image, templates)
        if not any(b.element for b in bosses):
            log("首领属性全部识别失败，终止")
            return

        battles = self._read_battles(context, image)
        log(f"剩余次数={battles}")
        if battles <= 0:
            log("没有剩余次数，结束")
            return

        banned: set[str] = set()
        for assignment in choose_plan(bosses, teams, battles, banned_teams=banned):
            if not self._fight(context, assignment):
                banned.add(assignment.team.name)
                log(f"本场失败，跳过队伍 {assignment.team.name}")

            if not self._goto_boss_list(context):
                log("无法回到首领列表页，终止")
                return

        log("协力讨伐结束")

    # ------------------------------------------------------------------
    # 导航
    # ------------------------------------------------------------------
    def _goto_boss_list(self, context: Context) -> bool:
        """确保当前在首领列表页。

        退出战斗后页面切换有动画/奖励弹窗，必须轮询等待，不能只扫一次。
        """
        for _ in range(3):
            if self._wait_boss_list(context, 6.0):
                return True

            # 可能停在上一次失败留下的「选择预设」弹窗上：先关掉
            self._tap(context, *TAP_PRESET_CLOSE)
            time.sleep(TAP_WAIT)
            self._close_popup_if_present(context)

            image = self._screencap(context)
            if image is None:
                return False

            # 已在公会大厅（玩法卡可见）→ 直接点卡片；否则先回村庄点底栏「公会」
            if self._ocr_hit(context, NODE_PLAY_CARD, image) is None:
                self._tap_node_or_point(context, NODE_GUILD_ENTRY, TAP_MAIN_GUILD)
                time.sleep(PAGE_WAIT)
            self._tap_node_or_point(context, NODE_PLAY_CARD, TAP_GUILD_PLAY_CARD)
            time.sleep(PAGE_WAIT * 1.5)
            self._close_popup_if_present(context)

        return self._wait_boss_list(context, 4.0)

    def _wait_boss_list(self, context: Context, timeout: float) -> bool:
        """轮询等待「剩余挑战次数 x/3」出现。"""
        deadline = time.time() + timeout
        while True:
            image = self._screencap(context)
            if image is not None and self._is_boss_list(context, image):
                return True
            if time.time() >= deadline:
                return False
            time.sleep(1.0)

    def _tap_node_or_point(
        self, context: Context, node: str, fallback: tuple[int, int]
    ) -> bool:
        """优先按 OCR 结果点击，识别不到时退回固定坐标。"""
        image = self._screencap(context)
        if image is not None:
            box = self._ocr_hit(context, node, image)
            if box is not None:
                x, y, w, h = box
                log(f"{node} -> ({x + w // 2},{y + h // 2})")
                self._tap(context, x + w // 2, y + h // 2)
                return True
        log(f"{node} 未识别，退回固定坐标 {fallback}")
        self._tap(context, *fallback)
        return False

    def _is_boss_list(self, context: Context, image: np.ndarray) -> bool:
        """首领列表页判据：能读到 `剩余挑战次数 x/3`。"""
        return self._read_battles(context, image) >= 0

    def _close_popup_if_present(self, context: Context) -> bool:
        """若出现「点击空白处关闭弹窗」提示，则点空白处关闭。"""
        image = self._screencap(context)
        if image is None:
            return False
        if self._ocr_hit(context, NODE_POPUP_HINT, image) is None:
            return False
        log("检测到弹窗，点击空白处关闭")
        self._tap(context, *TAP_BLANK_TO_CLOSE)
        time.sleep(PAGE_WAIT)
        return True

    # ------------------------------------------------------------------
    # 单场作战
    # ------------------------------------------------------------------
    def _fight(self, context: Context, assignment: Assignment) -> bool:
        boss, team = assignment.boss_slot, assignment.team
        log(f"开战：槽位{boss} ← {team.name}")

        # 1) 进详情页：图标框中心 + 偏移
        roi = BOSS_ICON_ROIS[boss]
        cx = roi[0] + roi[2] // 2 + ICON_TO_ITEM_OFFSET[0]
        cy = roi[1] + roi[3] // 2 + ICON_TO_ITEM_OFFSET[1]
        self._tap(context, cx, cy)
        time.sleep(PAGE_WAIT)

        # 2) 打开「选择预设」并选中目标队伍
        if not self._pick_preset(context, team.name):
            log(f"未能选中预设 {team.name}")
            self._tap(context, *TAP_PRESET_CLOSE)
            return False

        # 3) 入场（v1 走演习）
        if PRACTICE_MODE:
            self._tap(context, *TAP_DETAIL_PRACTICE)
        else:
            if not self._enter_enabled(context):
                log("「入场」不可点（队伍英雄今日已用），跳过")
                return False
            self._tap(context, *TAP_DETAIL_ENTER)
        time.sleep(PAGE_WAIT)

        # 4) 开自动战斗
        if not self._enable_auto_battle(context):
            log("未找到「自动战斗」，可能没进战斗")

        # 5) 等结算 → 退出
        return self._wait_and_exit(context)

    def _pick_preset(self, context: Context, team_name: str) -> bool:
        """点 ≡ 打开弹窗，OCR 找到目标预设行并点击（必要时上滑）。"""
        self._tap(context, *TAP_DETAIL_PRESET)
        time.sleep(PAGE_WAIT)

        for attempt in range(PRESET_SCROLL_MAX + 1):
            image = self._screencap(context)
            if image is None:
                return False
            box = self._find_preset_box(context, image, team_name)
            if box is not None:
                row_y = box[1] + box[3] // 2
                log(f"命中预设 {team_name} @ ({PRESET_ROW_TAP_X},{row_y})")
                self._tap(context, PRESET_ROW_TAP_X, row_y)
                time.sleep(PAGE_WAIT)
                if self._preset_popup_gone(context, team_name):
                    return True
                log(f"预设 {team_name} 未生效（弹窗仍在），重试一次")
                self._tap(context, PRESET_ROW_TAP_X, row_y)
                time.sleep(PAGE_WAIT)
                if self._preset_popup_gone(context, team_name):
                    return True
                log(f"预设 {team_name} 选择失败")
                self._tap(context, *TAP_PRESET_CLOSE)
                time.sleep(TAP_WAIT)
                return False
            if attempt < PRESET_SCROLL_MAX:
                log(f"未命中 {team_name}，上滑（第 {attempt + 1} 次）")
                context.tasker.controller.post_swipe(
                    *PRESET_SCROLL_FROM, *PRESET_SCROLL_TO, 400
                ).wait()
                time.sleep(TAP_WAIT)
        return False

    def _find_preset_box(
        self, context: Context, image: np.ndarray, team_name: str
    ) -> tuple[int, int, int, int] | None:
        results = self._ocr_results(context, NODE_PRESET_NAMES, image)
        target = _normalize_name(team_name)
        for text, box in results:
            if _normalize_name(text) == target:
                return box
        for text, box in results:      # 退化：包含匹配
            if target and target in _normalize_name(text):
                return box
        return None

    def _preset_popup_gone(self, context: Context, team_name: str) -> bool:
        """弹窗已关闭（ROI 内再也读不到该预设名）视为选中成功。"""
        image = self._screencap(context)
        if image is None:
            return False
        return self._find_preset_box(context, image, team_name) is None

    def _enable_auto_battle(self, context: Context) -> bool:
        """轮询「自动战斗」，出现即点（只点一次，避免切换回手动）。"""
        deadline = time.time() + BATTLE_START_TIMEOUT
        while time.time() < deadline:
            image = self._screencap(context)
            if image is None:
                return False
            box = self._ocr_hit(context, NODE_AUTO_BATTLE, image)
            if box is not None:
                x, y, w, h = box
                log("点击「自动战斗」")
                self._tap(context, x + w // 2, y + h // 2)
                return True
            time.sleep(1.5)
        return False

    def _wait_and_exit(self, context: Context) -> bool:
        """等战斗结束（出现「退出」），点退出回到首领列表。"""
        deadline = time.time() + BATTLE_END_TIMEOUT
        while time.time() < deadline:
            image = self._screencap(context)
            if image is None:
                time.sleep(2)
                continue
            box = self._ocr_hit(context, NODE_RESULT_EXIT, image)
            if box is not None:
                x, y, w, h = box
                log("战斗结束，点击「退出」")
                self._tap(context, x + w // 2, y + h // 2)
                time.sleep(PAGE_WAIT * 1.5)
                return True
            time.sleep(2)
        log("等待结算超时")
        return False

    def _enter_enabled(self, context: Context) -> bool:
        """「入场」按钮是否可点。"""
        image = self._screencap(context)
        if image is None:
            return False
        return self._ocr_hit(context, NODE_ENTER_BUTTON, image) is not None

    # ------------------------------------------------------------------
    # 基础能力
    # ------------------------------------------------------------------
    def _screencap(self, context: Context) -> np.ndarray | None:
        image = context.tasker.controller.post_screencap().wait().get()
        if image is None:
            return None
        return np.asarray(image)

    def _tap(self, context: Context, x: int, y: int) -> None:
        context.tasker.controller.post_click(int(x), int(y)).wait()
        time.sleep(TAP_WAIT)

    def _read_battles(self, context: Context, image: np.ndarray | None) -> int:
        """读 `剩余挑战次数 x/3`；读不到返回 -1。"""
        if image is None:
            return -1
        text = self._ocr_text(context, NODE_ENTRIES, image)
        for token in re.findall(r"\d+", text or ""):
            return int(token)
        return -1

    # ---- OCR 封装 ----
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

    def _ocr_text(self, context: Context, node: str, image: np.ndarray) -> str:
        detail = context.run_recognition(node, image)
        if detail is None or not detail.hit or detail.best_result is None:
            return ""
        best = detail.best_result
        return best.text if isinstance(best, OCRResult) else ""

    def _ocr_hit(
        self, context: Context, node: str, image: np.ndarray
    ) -> tuple[int, int, int, int] | None:
        detail = context.run_recognition(node, image)
        if detail is None or not detail.hit:
            return None
        return _to_box(detail.box)


def _normalize_name(text: str) -> str:
    """去掉空白，便于预设名比对。"""
    return "".join(ch for ch in (text or "") if not ch.isspace())
