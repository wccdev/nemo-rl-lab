# common/environments — 自定义环境（奖励来源）

NeMo-RL 里 GRPO 的奖励由 **Environment** 产生（而非独立 reward 函数）。把跨实验复用的
自定义环境放这里：

- 数学/通用单轮任务：通常用内置环境（配置 `data.default.env_name=math` 等），无需自写。
- 多轮 Agent / 工具调用：实现自定义 Environment + 自定义 run 脚本。

## `example_tool_env.py` — 可运行的工具调用环境

一个**计算器工具调用**多轮环境，照官方 `nemo_rl/environments/games/sliding_puzzle.py`
结构写成，可直接训练。导出：

- `ToolAgentEnv`：`@ray.remote` 的 `EnvironmentInterface`，实现 `step()` 返回 6 字段
  `EnvironmentReturn(observations, metadata, next_stop_strings, rewards, terminateds, answers)`。
- `TOOLS`：工具注册表（`name -> callable(arg)->str`），加工具就往这里加。
- `safe_eval`：安全算术求值（给计算器工具与答案校验用）。

配套用法见 `experiments/agent-grpo_qwen3.5-9b_calc-tool_v1/`（含自定义 `run.py`：生成
`DatumSpec` 任务 + 构建 `task_to_env` + `grpo_train`）。

> 实现要点：环境是 Ray actor；多轮训练需要**自定义 run 脚本**来喂数据和环境（纯改配置不够）；
> `step()` 的返回结构、`DatumSpec` 字段以你装的 0.6.0 源码为准（参考 `nemo_rl/environments/` 内置环境）。
