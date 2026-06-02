# agent-grpo_qwen3.5-9b_multitool_v1

多轮多工具 Agent GRPO 练习实验。改编自官方 `run_grpo_sliding_puzzle.py`。

## 目标

让 `qwen3.5-9b` 学会**多轮调用多个工具**完成需要外部能力的任务：检索资料（search）、
做算术（calc）、跑代码（python），最后给出正确数值答案。跑通 NeMo-RL 0.6.0 的多轮
Agent / 自定义环境 / 多工具链路。

## 任务类型（run.py 随机生成，均为数值可验证）

| 类型 | 示例问题 | 需要的工具 |
| --- | --- | --- |
| calc | 计算 12 + 7 * 3 | `calc` |
| search_calc | 买 5 个苹果一共多少钱（KB 含单价 + 干扰项） | `search` → `calc` |
| code | 求 1..15 的平方和 | `python` |

## 工具协议

| 动作 | 模型输出 | 环境响应 |
| --- | --- | --- |
| 计算 | `<tool>calc: 2+3*4</tool>` | `[calc] 14` |
| 检索 | `<tool>search: 苹果 单价</tool>` | `[search] 苹果的单价是 8 元` |
| 代码 | `<tool>python: print(sum(i*i for i in range(1,6)))</tool>` | `[python] 55` |
| 答题 | `<answer>14</answer>` | 校验对错，对则奖励 1.0 并结束 |

停止串 `</tool>` / `</answer>` 让模型每轮在动作处停下，交给环境推进。

## 组成

- `config.yaml` — 继承 `grpo_sliding_puzzle.yaml`（多轮基底）+ `qwen3.5-9b`，`_override_` 替换 env。
- `run.py` — 自定义训练脚本：生成三类任务 + KB，实例化 `ToolAgentEnv`，调 `grpo_train`。
- `run.sh` — 检测到本目录有 `run.py`，自动以它为入口。

## SwanLab

- project：`agent-grpo_qwen3.5-9b_multitool_v1`，run：`turns6-g8-kl0.01`，链接：<回填>
- 指标：`tool_agent_success_rate`（答对率）/ reward / 平均轮数

## 运行

```bash
NEMO_RL_DIR=/path/to/NeMo-RL CLUSTER_PROFILE=gb10-spark bash run.sh
```

> ⚠️ `python` 工具会执行模型生成的代码，请只在隔离的 NeMo-RL 训练容器内运行。

## 扩展成你的真实工具

1. 在 `common/environments/example_tool_env.py` 的 `TOOLS` 加工具（如真实检索 API、SQL、HTTP）。
2. 改 `run.py` 的 `_make_task` / `PROMPT_HEADER` 换成你的任务、数据与知识库。
3. 调 `config.yaml` 的 `max_rollout_turns`、`env.tool_agent.cfg`、超参。
