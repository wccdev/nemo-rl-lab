# common/environments — 自定义环境（奖励来源）

NeMo-RL 里 GRPO 的奖励由 **Environment** 产生（而非独立 reward 函数）。把跨实验复用的
自定义环境放这里：

- 数学/通用单轮任务：通常用内置环境（配置 `data.default.env_name=math` 等），无需自写。
- 多轮 Agent / 工具调用：实现自定义 Environment（见 `example_tool_env.py`），在配置里
  通过 `data.default.env_name=<你的环境名>` 引用，并设 `grpo.max_rollout_turns>1`。

实现时参考你本地 NeMo-RL 0.6.0 源码 `nemo_rl/environments/` 下的内置环境
（如 `math`、`sliding_puzzle`）作为模板，确保 `step()` 返回结构与版本一致。
