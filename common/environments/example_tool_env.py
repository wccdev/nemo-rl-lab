"""自定义多轮 / 工具调用 Environment 骨架（NeMo-RL 0.6.0）。

在 NeMo-RL 里，GRPO 的奖励来自 Environment（不是单独的 reward 函数）。多轮 Agent
训练 = GRPO + grpo.max_rollout_turns>1 + 这样的自定义 Environment。

Environment 通常是一个 Ray actor，实现 EnvironmentInterface：
  - step(message_log_batch, metadata) -> EnvironmentReturn
    在每一轮拿到模型输出，执行工具/推进环境，返回 observation、奖励、是否终止等。

下面是结构骨架——**确切的基类路径、方法签名、返回类型以你安装的 0.6.0 源码为准**
（参考 nemo_rl/environments/ 下的内置环境，如 math、sliding_puzzle）。
"""
from __future__ import annotations

from typing import Any

import ray

# 实际导入以源码为准，例如：
# from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn


@ray.remote
class ExampleToolEnv:  # 实现 EnvironmentInterface
    """一个最小的多轮工具调用环境示例。"""

    def __init__(self, cfg: dict[str, Any] | None = None):
        self.cfg = cfg or {}
        self.max_turns = int(self.cfg.get("max_turns", 8))

    def step(self, message_log_batch, metadata):
        """对一个 batch 的对话推进一轮。

        返回需符合 NeMo-RL 的 EnvironmentReturn（observations / metadata /
        next_stop_strings / rewards / terminateds）。这里只给思路：
          1. 从 message_log_batch 解析模型这一轮的输出（可能含工具调用）
          2. 执行工具 / 推进环境，得到 observation
          3. 计算奖励：最终成败为主，过程合法性（工具格式、轮数）为辅
          4. 判定是否终止（任务完成或超过 max_turns）
        """
        raise NotImplementedError("按 0.6.0 的 EnvironmentReturn 结构实现 step()")

    def global_post_process_and_metrics(self, batch):
        """（可选）批级别的后处理与指标统计。"""
        raise NotImplementedError
