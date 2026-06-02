"""多轮 Agent / 工具调用任务的奖励函数示例。

被配置引用：reward.fn: "common.rewards.agent_reward:compute_reward"
思路：最终任务成败为主奖励，过程合法性（工具调用格式、轮数惩罚）为辅。
"""
from __future__ import annotations

from typing import Any


def compute_reward(trajectory: dict[str, Any], **kwargs) -> float:
    """trajectory 约定包含: success(bool), turns(int), invalid_tool_calls(int)。

    具体字段以 NeMo-RL 多轮环境返回结构为准，这里给出可调的组合形态。
    """
    success = bool(trajectory.get("success", False))
    turns = int(trajectory.get("turns", 0))
    invalid = int(trajectory.get("invalid_tool_calls", 0))

    reward = 1.0 if success else 0.0
    reward -= 0.02 * turns          # 轻微惩罚过长轨迹
    reward -= 0.1 * invalid         # 惩罚非法工具调用
    return reward
