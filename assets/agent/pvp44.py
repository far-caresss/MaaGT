# -*- coding: utf-8 -*-
"""44（圆形角斗场）多场循环 —— L3 编排。

**为什么要单独一个循环**：44 每场消耗 1 次挑战次数，次数用完后再点「开始战斗」
可能弹购买（钻石/真实货币）→ 必须"打一场、读一次次数、到 0 立刻停"，
不能靠 `repeat_count` 之类的固定次数硬打。

安全边界（都在本文件里，图上看得到）：
1. **只认次数**：每轮开打前读 `ValPvpTickets` 的 `x/y`；读不到 → 立刻停（不猜）；
2. `0` 次立刻停，不再点任何按钮；
3. `max_rounds` 兜底（默认 6，次数的每日上限就是 5，多留 1 的余量）；
4. 单场失败就停，不在异常状态里继续消耗。

与 pipeline 的分工：
- 单场链路在 `pvp_44_chain.json:Pvp44Run`（判据、等待、点哪里都在图上）；
- 本文件只负责"打几场"和"什么时候必须停"。
"""

from __future__ import annotations

import re
from typing import Any

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction

import page_guard as pg

TICKETS_NODE = "ValPvpTickets"
#: 次数用完时按钮会变成"红票 + (-1)"的灰色态，模板命中即"打完了"（0/5 命中、1/5 与 3/5 不命中，实测于 2026-10-31）
EXHAUSTED_NODE = "44打完"
SINGLE_ROUND_ENTRY = "Pvp44Run"
#: `2/3` 这种；踩坑 §1.9：绝不能退化成"取第一个数字串"
_TICKETS_RE = r"(\d+)\s*/\s*(\d+)"


def read_tickets(context: Context) -> int | None:
    """读剩余挑战次数；读不到返回 None（调用方必须当"失败"处理，不许当成 0 或沿用旧值）。"""
    image = pg._screencap(context)
    text = pg._text(context, TICKETS_NODE, image)
    value = pg.parse_num(text, _TICKETS_RE)
    pg.log(f"剩余挑战次数：{value}（原文 {text!r}）")
    return value


def _round_override(allow: bool) -> dict[str, Any]:
    """把单场链里的**消耗闸门**按 `allow` 设置。

    闸门在 `pvp_44_chain.json:Pvp44Confirm`。这里显式传参，**不依赖** `run_task` 是否继承
    外层 `pipeline_override`（踩坑 §1.11 只说过"会被继承"，没保证反过来也成立）。

    `allow` 来自本节点自己的 `custom_action_param.allow_consume`——PI 开关
    「44消耗挑战次数」覆盖的就是这条路径（pure override 已有字段，不新建节点）。
    """
    return {
        "Pvp44Confirm": {
            "action": {
                "param": {
                    "custom_action_param": {
                        "button": "BtnPvpFormationFight",
                        "consumes": "ticket",
                        "cost": "1 次挑战次数",
                        "allow_consume": bool(allow),
                    }
                }
            }
        }
    }


@AgentServer.custom_action("Pvp44Rounds")
class Pvp44RoundsAction(CustomAction):
    """`{"max_rounds": 6}` —— 把当天剩余次数打完，打完即停。

    `max_rounds` 只是**上界**：真正的停止条件是"读到的剩余次数为 0"。
    """

    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        raw = getattr(argv, "custom_action_param", "")
        param: dict[str, Any] = {}
        if isinstance(raw, dict):
            param = raw
        elif raw:
            try:
                import json

                param = json.loads(raw) if isinstance(raw, str) else {}
            except Exception:  # noqa: BLE001
                pg.log(f"Pvp44Rounds 参数不是合法 JSON：{raw!r}")
                param = {}
        max_rounds = int(param.get("max_rounds") or 6)
        allow = bool(param.get("allow_consume"))
        pg.log(f"44 多场：max_rounds={max_rounds} allow_consume={allow}"
               f"（关掉时闸门会拒点，任务只探路不消耗）")

        if not pg.nav_to(context, pg.PAGE_PVP44):
            pg.log("44 多场：到不了圆型角斗场面板 → 中止（不盲点）")
            return False

        done = 0
        for i in range(1, max_rounds + 1):
            left = read_tickets(context)
            if left is None:
                # 次数读不到时**只认"明确的没次数证据"**：按钮已变成红票灰态（`44打完`）
                image = pg._screencap(context)
                if pg._hit(context, EXHAUSTED_NODE, image):
                    pg.log("44 多场：读不到次数，但命中「44打完」（按钮已是红票灰态）→ 按次数用完结束")
                    break
                pg.log("44 多场：读不到剩余次数 → 中止（不猜，避免超打）")
                return False
            if left <= 0:
                pg.log("44 多场：次数已用完 → 结束")
                break

            pg.log(f"44 多场：第 {i} 轮开始（剩余 {left} 次）")
            detail = context.run_task(SINGLE_ROUND_ENTRY, _round_override(allow))
            ok = bool(detail is not None and detail.status.succeeded)
            if not ok:
                pg.log(f"44 多场：第 {i} 轮失败 → 中止（不在异常状态里继续消耗）")
                return False
            done += 1

        pg.log(f"44 多场：结束，共打 {done} 场")
        return True
