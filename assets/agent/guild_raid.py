# -*- coding: utf-8 -*-
"""公会协力讨伐：识别首领属性 → 按配置队伍分配 → 逐个作战。

流程（坐标基于 1280x720）：
    首领列表页 --图标框中心+(120,54)--> 详情页
    详情页 --≡(770,647)--> 「选择预设」弹窗 --点名字框中心--> 详情页
    详情页 --演习(905,647)/入场(1115,647)--> 战斗 --自动战斗--> 结果 --退出--> 首领列表页

入场会消耗挑战次数、演习不会，由 PRACTICE_MODE 切换（改回 True 即退回演习）。
"""

from __future__ import annotations

import itertools
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

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

#: 属性中文别名（界面配置容错）
_ELEMENT_ALIASES: dict[str, str] = {cn: en for en, cn in ELEMENT_CN.items()}

RELATION_COUNTER = "counter"      # 队伍克制首领
RELATION_NEUTRAL = "neutral"      # 互不克制
RELATION_COUNTERED = "countered"  # 队伍被首领克制

#: False = 走「入场」（消耗挑战次数）；True = 走「演习」
PRACTICE_MODE = False

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

#: 「选择预设」弹窗内行的点击 x。点名字文字无效，必须点行中间。
PRESET_ROW_TAP_X = 640

# ---------------- 详情页「打不了」判据 ----------------
#: 英雄栏出现这个词 = 该预设里有英雄今日已参与
HERO_UNUSABLE_KEYWORD = "无法使用"
#: 「注意」拦截弹窗正文特征词（子串匹配，容忍 OCR 抖动）
NOTICE_KEYWORDS: tuple[str, ...] = ("无法再次参与", "当日已参与")
#: 详情页左上角 ← 返回，复用 startup.json 的 `返回`（白色 ColorMatch，箭头
#: bbox (32,17)-(58,53)，落在它的 ROI [20,7,53,58] 内）。
#: ⚠️ 曾用「入场」按钮颜色判可用性：可用态均值 149、禁用态 124.7，差值太小
#: 会把亮的按钮误判成灰 → **已废弃**，只看英雄栏的「无法使用」。
NODE_DETAIL_BACK = "返回"

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
#: 战斗中轮询「退出」的间隔（一场战斗 60~90s，判太密没意义还费截图）
BATTLE_POLL_INTERVAL = 5

#: `_wait_battle_entry` 的三种返回
BATTLE_READY = "ready"      # 已点到「自动战斗」
BATTLE_NOTICE = "notice"    # 弹了「注意」（英雄当日已参与），已点「确认」关掉
BATTLE_TIMEOUT = "timeout"  # 一直没等到

#: 识别节点名（定义在 pipeline/*.json）
NODE_ENTRIES = "RaidEntries"
NODE_PRESET_NAMES = "RaidPresetNames"
NODE_AUTO_BATTLE = "RaidAutoBattle"
NODE_RESULT_EXIT = "RaidResultExit"
NODE_POPUP_HINT = "RaidPopupHint"     # 等价副本，guild_activity.py 还在用
#: 既有节点（startup.json）「点击空白处关闭弹窗」：OCR 判据 + 点 (1236,48)。
#: ⚠️ **只借它的"识别"**（`run_recognition`，一次就返回）；
#: **绝不要 `run_task` 它** —— 节点没命中时框架会一直重判到超时，
#: 弹窗不在时会一直卡在判「点击空白处关闭弹窗」这句文案上。
NODE_CLOSE_POPUP = "点击空白处关闭弹窗"
#: 「注意」拦截弹窗（如"当日已参与公会协力讨伐的英雄，无法再次参与。"）
NODE_NOTICE_DIALOG = "RaidNoticeDialog"
#: 英雄栏（详情页底部 4 个槽位）。不带 expected，Python 侧找「无法使用」。
NODE_HERO_SLOTS = "RaidHeroSlots"
NODE_ENTER_BUTTON = "RaidEnterButton"
#: 首领详情页底部「演习」按钮（用来判断"确实在详情页"，进而避免盲点 ≡）
NODE_PRACTICE_BUTTON = "RaidPracticeButton"
NODE_GUILD_ENTRY = "RaidGuildEntry"
#: 公会大厅「最上方活动卡」。**不带 expected**：活动名每期都变（本期讨伐叫「死城频率」），
#: 静态匹配名字下期必失效；改为 Python 侧要求命中中文（见 `_activity_card_box`）。
NODE_PLAY_CARD = "GuildActivityCard"
#: 公会大厅右侧面板标题。用来判断"是否在公会大厅"——因为活动卡那一带的文字
#: 在别的活动页也会出现（例如竞技场页同位置有「未找到记录」），光靠 ROI 治不了。
NODE_LOBBY_PANEL = "GuildLobbyPanel"
#: 大厅面板标题的固定文案（UI 文案，不随活动名变）
LOBBY_KEYWORD = "公会玩法"

#: 去掉 JSONC 的尾逗号
_TRAILING_COMMA = re.compile(r",(\s*[}\]])")

#: 活动卡/标题必须是中文（活动名每期都变，不写死但也不能把 HUD 数字当卡片）
_CJK_RE = re.compile(r"[\u4e00-\u9fff]{2,}")


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


def _load_argv_param(argv: object) -> object:
    """安全解析 custom_action_param（它是个 JSON 字符串）。

    兼容两种入参：`CustomAction.RunArg`（pipeline 入口）或已经是字符串
    （`GuildActivityRouter` 直接透传参数时）。
    """
    raw = argv if isinstance(argv, str) else getattr(argv, "custom_action_param", None)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception as exc:  # noqa: BLE001 - 参数异常时退回 teams.json
        log(f"custom_action_param 不是合法 JSON：{exc!r}")
        return None


def _resolve_element(token: str) -> str | None:
    """把用户写的属性归一成内部名；认不出来返回 None。"""
    key = token.strip().lower()
    key = _ELEMENT_ALIASES.get(key, key)
    return key if key in ELEMENTS else None


def parse_teams_param(entries: Sequence[object]) -> list[Team]:
    """解析界面传入的队伍配置，每项 `预设名:属性`。

    容错：全角冒号、前后空白、属性大小写与中文名、`属性:预设名` 顺序颠倒、
    同一预设名重复（保留靠前的一条）。每项跳过都打日志说明原因。
    """
    teams: list[Team] = []
    seen: set[str] = set()
    skipped = 0
    for idx, item in enumerate(entries, 1):
        line = str(item or "").strip()
        if not line:
            continue
        parts = re.split(r"[:：]", line, maxsplit=1)
        if len(parts) != 2:
            log(f"界面配置第{idx}项跳过：缺少 `:` 分隔符（应为 `预设名:属性`）-> {line!r}")
            skipped += 1
            continue
        left, right = parts[0].strip(), parts[1].strip()
        if not left or not right:
            log(f"界面配置第{idx}项跳过：冒号两侧都不能为空 -> {line!r}")
            skipped += 1
            continue
        element, name = _resolve_element(right), left
        if element is None:
            swapped = _resolve_element(left)
            if swapped is None:
                log(
                    f"界面配置第{idx}项跳过：属性 {right!r} 非法"
                    f"（可用 {'/'.join(ELEMENTS)} 或 {'/'.join(ELEMENT_CN.values())}）"
                    f"-> {line!r}"
                )
                skipped += 1
                continue
            element, name = swapped, right
            log(f"界面配置第{idx}项：按 `属性:预设名` 顺序解析 -> {name}/{element}")
        key = _normalize_name(name)
        if key in seen:
            log(f"界面配置第{idx}项跳过：预设名重复 {name!r}（保留靠前的那条）")
            skipped += 1
            continue
        seen.add(key)
        teams.append(Team(name=name, element=element))
    if skipped:
        log(f"界面配置解析结果：有效 {len(teams)} 队、跳过 {skipped} 项")
    return teams


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
    """公会协力讨伐总编排。

    `_nav`：可选的"重新进入活动"回调，由 `guild_activity.GuildActivityRouter` 注入。
    丢页时交给它（它自带判据：Return / 主界面 / 大厅 / 活动卡）。
    没有它时（开发调试脚本直跑 `GuildRaid`）只认「确实在大厅」才点卡，**绝不盲点**。
    """

    _nav: Callable[[Context], bool] | None = None

    def run(
        self, context: Context, argv: CustomAction.RunArg
    ) -> CustomAction.RunResult:
        param = getattr(argv, "custom_action_param", "") or ""
        return CustomAction.RunResult(success=run_raid(context, param))

    # ------------------------------------------------------------------
    def _run(
        self,
        context: Context,
        param_json: str,
        nav: Callable[[Context], bool] | None = None,
    ) -> None:
        self._nav = nav
        teams = self._load_teams(param_json)
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
        if battles <= 0:
            log("没有剩余次数，结束")
            return

        # 每场重新规划（不是"进循环前一次性算完"）：
        # 一次运行里「队伍最多上一场、BOSS 最多打一次」，所以每轮都从
        # 「剩余次数 + 没打过的 BOSS + 没试过的队伍」重算 → 某队打不了时后面的队伍顶上。
        attempts: set[str] = set()
        done_bosses: set[int] = set()

        while True:
            fresh = self._screencap(context)
            now = self._read_battles(context, fresh)
            if now >= 0:
                battles = now
            log(f"剩余次数={battles}")
            if battles <= 0:
                log("没有剩余次数，结束")
                break

            plan = choose_plan(
                [b for b in bosses if b.slot not in done_bosses],
                teams,
                battles,
                banned_teams=attempts,
            )
            if not plan:
                log("没有还能打的 (BOSS, 队伍) 组合，结束")
                break

            assignment = plan[0]
            if self._fight(context, assignment):
                done_bosses.add(assignment.boss_slot)
                battles -= 1
            else:
                log(f"本场失败，跳过队伍 {assignment.team.name}")
            attempts.add(assignment.team.name)

            # ⚠️ 这里**不主动**跑「点击空白处关闭弹窗」：该节点的判据只是一句
            # OCR 文案，万一误判就会平白点一次右上角 (1236,48) —— 不划算。
            # 奖励窗真挡住列表页时，`_goto_boss_list()` 内部本来就会兜底关它
            # （先 `_wait_boss_list` 读不到「剩余挑战次数」= 确实有东西挡着，才动手）。
            if not self._goto_boss_list(context):
                log("无法回到首领列表页，终止")
                return

        log("协力讨伐结束")

    # ------------------------------------------------------------------
    # 配置来源：界面（custom_action_param.teams）优先，全空退回 teams.json
    # ------------------------------------------------------------------
    def _load_teams(self, param_json: str) -> list[Team]:
        raw = _load_argv_param(param_json)
        entries = raw.get("teams") if isinstance(raw, dict) else None
        if isinstance(entries, (list, tuple)) and any(
            str(x or "").strip() for x in entries
        ):
            teams = parse_teams_param(entries)
            if teams:
                log(
                    "已加载队伍(界面配置)："
                    + "、".join(f"{t.name}/{t.element}" for t in teams)
                )
                return teams
            log("界面配置有内容但全部无效，退回 teams.json")
        else:
            log("界面队伍配置为空，退回 teams.json（开发期兜底）")
        return load_teams()

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

            # 只关「确实开着」的弹窗（先判后点，不做盲点）
            self._close_preset_popup_if_present(context)
            self._close_popup_if_present(context)

            if self._nav is not None:
                # 丢页了：交给统一导航（回主界面 / 进大厅 / 点卡，全程带判据）
                log("未能直接回到首领列表页，交给统一导航重新进入活动")
                return bool(self._nav(context))

            # 没有导航回调（直接跑 GuildRaid 的开发脚本）：只认「确实在大厅」才点卡
            card = self._activity_card_box(context, self._screencap(context))
            if card is None:
                log("不在公会大厅、也没有导航回调 —— 不盲点，放弃")
                return False
            log("在公会大厅，点最上方活动卡重新进入")
            self._tap_activity_card(context, card)
            time.sleep(PAGE_WAIT * 1.5)
            self._close_popup_if_present(context)

        return self._wait_boss_list(context, 4.0)

    def _in_guild_lobby(self, context: Context, image: np.ndarray | None) -> bool:
        """是否在公会大厅：右侧面板标题「公会玩法」是否可读。

        这是**固定 UI 文案**，不随活动名/期数变，所以可以放心匹配。
        没有这道门的话，别的活动页右侧面板同一水平位置的文字
        （竞技场页同位置是「未找到记录」）会被误当成活动卡。
        """
        if image is None:
            return False
        texts = " ".join(text for text, _ in self._ocr_results(context, NODE_LOBBY_PANEL, image))
        return LOBBY_KEYWORD in texts

    def _activity_card_box(
        self, context: Context, image: np.ndarray | None
    ) -> tuple[tuple[int, int, int, int], str] | None:
        """公会大厅「最上方活动卡」的框 + 卡上文字（含中文）。

        - 先确认在公会大厅（见 `_in_guild_lobby`）；
        - 活动名每期都变，所以节点故意不带 `expected`，只要求命中中文。
        """
        if image is None or not self._in_guild_lobby(context, image):
            return None
        for text, box in self._ocr_results(context, NODE_PLAY_CARD, image):
            if _CJK_RE.search(text or ""):
                return box, text
        return None

    def _tap_activity_card(
        self, context: Context, card: tuple[tuple[int, int, int, int], str] | None
    ) -> bool:
        """点活动卡：只点识别到的文本框中心；识别不到就**不点**（不盲点固定坐标）。"""
        if card is None:
            log(f"{NODE_PLAY_CARD} 未识别 —— 不点（避免盲点）")
            return False
        box, text = card
        x, y, w, h = box
        log(f"{NODE_PLAY_CARD}「{text}」 -> ({x + w // 2},{y + h // 2})")
        self._tap(context, x + w // 2, y + h // 2)
        return True

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

    def _in_boss_detail(self, context: Context) -> bool:
        """是否在首领详情页：底部按钮区能读到「演习」或「入场」。

        作用：避免"没进详情页却点了 ≡ / 演习"这种盲点。
        回退判据：读不到两个按钮、但**也不在首领列表页**时，
        认为很可能在详情页（OCR 抖动），继续；反之明确拒绝。
        """
        image = self._screencap(context)
        if image is None:
            return False
        for node in (NODE_PRACTICE_BUTTON, NODE_ENTER_BUTTON):
            if self._ocr_hit(context, node, image) is not None:
                return True
        if self._read_battles(context, image) >= 0:
            log("读不到「演习/入场」，且仍能读到剩余次数 —— 判断为没进详情页")
            return False
        log("读不到「演习/入场」，但也不在首领列表页 —— 按详情页继续（可能 OCR 抖动）")
        return True

    def _is_boss_list(self, context: Context, image: np.ndarray) -> bool:
        """首领列表页判据：能读到 `剩余挑战次数 x/3`。"""
        return self._read_battles(context, image) >= 0

    def _close_popup_if_present(self, context: Context) -> bool:
        """若出现「点击空白处关闭弹窗」提示，则点空白处关闭。

        ⚠️ **不用 `context.run_task(...)`**：节点没命中时框架会一直重判到超时
        （弹窗不在时会一直卡在判这句文案）。
        这里只借既有节点的**识别**（`run_recognition`，一次就返回），点击沿用
        `TAP_BLANK_TO_CLOSE`（与该节点自己的 action target 一致）。
        识别不到直接返回 False —— **不盲点**。
        """
        image = self._screencap(context)
        if image is None:
            return False
        if self._ocr_hit(context, NODE_CLOSE_POPUP, image) is None:
            return False
        log("检测到弹窗，点击空白处关闭")
        self._tap(context, *TAP_BLANK_TO_CLOSE)
        time.sleep(PAGE_WAIT)
        return True

    def _notice_dialog_hit(
        self, context: Context, image: np.ndarray | None
    ) -> tuple[int, int, int, int] | None:
        """「注意」拦截弹窗是否开着：正文特征词 **且** 有「确认」按钮。

        节点故意不带 `expected`（硬匹配失败会整页误判），这里按子串匹配；
        两项找不全就返回 None —— 先判后点。
        """
        if image is None:
            return None
        results = self._ocr_results(context, NODE_NOTICE_DIALOG, image)
        joined = " ".join(text for text, _ in results)
        if not any(key in joined for key in NOTICE_KEYWORDS):
            return None
        for text, box in results:
            if "确认" in text:
                return box
        return None

    def _dismiss_notice_if_present(
        self, context: Context, image: np.ndarray | None = None
    ) -> bool:
        """弹窗确实开着才点它的「确认」关掉；返回是否命中。**不盲点**。"""
        if image is None:
            image = self._screencap(context)
        box = self._notice_dialog_hit(context, image)
        if box is None:
            return False
        x, y, w, h = box
        log("检测到「注意」弹窗（英雄当日已参与），点「确认」关闭")
        self._tap(context, x + w // 2, y + h // 2)
        time.sleep(PAGE_WAIT)
        return True

    def _hero_unusable(self, context: Context, image: np.ndarray | None) -> bool:
        """英雄栏是否出现「无法使用」（该预设里有英雄今日已参与）。"""
        if image is None:
            return False
        for text, _ in self._ocr_results(context, NODE_HERO_SLOTS, image):
            if HERO_UNUSABLE_KEYWORD in text:
                return True
        return False

    def _back_to_boss_list(self, context: Context) -> bool:
        """从首领详情页**就地**回到首领列表页（省掉 `Return` + 重进活动那一套）。

        只在「确实在详情页」且读到左上角 ← 时才点（先判后点）；点完轮询
        「剩余挑战次数」，回到列表页返回 True。失败也没关系：调用方随后仍会走
        `_goto_boss_list`（→ 统一导航兜底），只是多花点时间。
        """
        if not self._in_boss_detail(context):
            return False
        image = self._screencap(context)
        box = self._ocr_hit(context, NODE_DETAIL_BACK, image)
        if box is None:
            log("读不到详情页「返回」—— 不点，交给统一导航")
            return False
        x, y, w, h = box
        log(f"点详情页「返回」-> ({x + w // 2},{y + h // 2})")
        self._tap(context, x + w // 2, y + h // 2)
        return self._wait_boss_list(context, 6.0)

    def _close_preset_popup_if_present(self, context: Context) -> bool:
        """「选择预设」弹窗**确实开着**才点它的 X（先判后点，不再无条件盲点）。

        判据：`RaidPresetNames` 的 ROI 在弹窗内，弹窗开着才有预设名可读。
        """
        image = self._screencap(context)
        if image is None:
            return False
        if not self._ocr_results(context, NODE_PRESET_NAMES, image):
            return False
        log("检测到「选择预设」弹窗，点 X 关闭")
        self._tap(context, *TAP_PRESET_CLOSE)
        time.sleep(TAP_WAIT)
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
            self._close_preset_popup_if_present(context)
            self._back_to_boss_list(context)
            return False

        # 2.5) 选完预设先判一次：英雄栏有「无法使用」= 该预设今日已参与 → 跳过
        if self._hero_unusable(context, self._screencap(context)):
            log(f"预设 {team.name} 里有英雄「无法使用」（今日已参与）—— 跳过该队伍")
            self._back_to_boss_list(context)
            return False

        # 3) 开打：入场（PRACTICE_MODE=True 时走演习）
        if PRACTICE_MODE:
            self._tap(context, *TAP_DETAIL_PRACTICE)
        else:
            if not self._enter_enabled(context):
                log("「入场」不可点（队伍英雄今日已用），跳过")
                return False
            self._tap(context, *TAP_DETAIL_ENTER)
        time.sleep(PAGE_WAIT)

        # 4) 开自动战斗；期间若弹「注意」（英雄当日已参与）则本场作废
        outcome = self._wait_battle_entry(context)
        if outcome == BATTLE_NOTICE:
            log("本场打不了：英雄当日已参与（「注意」弹窗已关）—— 跳过该队伍")
            self._back_to_boss_list(context)
            return False
        if outcome != BATTLE_READY:
            log("未找到「自动战斗」，可能没进战斗，仍尝试等结算")

        # 5) 等结算 → 退出
        return self._wait_and_exit(context)

    def _pick_preset(self, context: Context, team_name: str) -> bool:
        """点 ≡ 打开弹窗，OCR 找到目标预设行并点击（必要时上滑）。"""
        if not self._in_boss_detail(context):
            log("不在首领详情页 —— 不点 ≡（避免盲点），放弃本场")
            return False
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
                self._close_preset_popup_if_present(context)
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

    def _wait_battle_entry(self, context: Context) -> str:
        """等「自动战斗」出现并点掉；期间同时拦截「注意」弹窗（先判后点）。

        返回 BATTLE_READY / BATTLE_NOTICE / BATTLE_TIMEOUT。
        """
        deadline = time.time() + BATTLE_START_TIMEOUT
        while time.time() < deadline:
            image = self._screencap(context)
            if image is None:
                return BATTLE_TIMEOUT
            if self._dismiss_notice_if_present(context, image):
                return BATTLE_NOTICE
            box = self._ocr_hit(context, NODE_AUTO_BATTLE, image)
            if box is not None:
                x, y, w, h = box
                log("点击「自动战斗」")
                self._tap(context, x + w // 2, y + h // 2)
                return BATTLE_READY
            time.sleep(1.5)
        return BATTLE_TIMEOUT

    def _wait_and_exit(self, context: Context) -> bool:
        """等战斗结束（出现「退出」），点退出回到首领列表。"""
        deadline = time.time() + BATTLE_END_TIMEOUT
        while time.time() < deadline:
            image = self._screencap(context)
            if image is None:
                time.sleep(BATTLE_POLL_INTERVAL)
                continue
            box = self._ocr_hit(context, NODE_RESULT_EXIT, image)
            if box is not None:
                x, y, w, h = box
                log("战斗结束，点击「退出」")
                self._tap(context, x + w // 2, y + h // 2)
                time.sleep(PAGE_WAIT * 1.5)
                return True
            time.sleep(BATTLE_POLL_INTERVAL)
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


def run_raid(
    context: Context,
    param_json: str = "",
    nav: Callable[[Context], bool] | None = None,
) -> bool:
    """跑一轮协力讨伐，返回是否正常走完（异常返回 False）。

    两个调用方：
    - `GuildRaidOrchestrator`：pipeline 入口 `GuildRaid`（开发调试用，不带 nav）；
    - `guild_activity.GuildActivityRouter`：统一入口「公会活动」识别后分派过来，
      并把 `nav`（重新进入活动）注入进来。

    `param_json` = `custom_action_param`（JSON 字符串，含界面配置的 `teams`）；
    `nav(context) -> bool` = 丢了页面时用的"重新进入活动"回调（返回是否已回到讨伐页）。

    ⚠️ 分派**故意不走** `context.run_task("GuildRaid")`：那会依赖
    「外层任务的 `pipeline_override` 能否被子任务继承」这一框架细节；
    显式传参更稳（发行版里没有 `teams.json`，参数一旦丢了就直接失败）。
    """
    flow = GuildRaidOrchestrator()
    try:
        flow._run(context, param_json, nav)
    except Exception as exc:  # noqa: BLE001 - 顶层兜底，避免中断框架
        log(f"异常终止：{exc!r}")
        import traceback

        log(traceback.format_exc())
        return False
    return True
