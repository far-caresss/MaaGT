# -*- coding: utf-8 -*-
"""公会「陨石挖掘计划」流程 —— **占位，尚未实现**。

本模块目前不执行任何操作，只告警并返回失败，
避免半成品被误当成可用流程跑起来。
"""

from maa.context import Context

from guild_raid import log

ACTIVITY_NAME = "陨石挖掘计划"


def run(context: Context) -> bool:
    """占位实现：不点任何按钮，只告警。"""
    log(f"{ACTIVITY_NAME}：流程尚未实现（占位模块），不执行任何操作")
    return False
