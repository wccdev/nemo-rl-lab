# agent-grpo_qwen3.5-9b_calc-tool_v1

多轮 Agent（工具调用）GRPO 练习实验。改编自官方 `run_grpo_sliding_puzzle.py`。

## 目标

让 `qwen3.5-9b` 学会**多轮调用计算器工具**求解算式：模型先用 `<tool>calc: …</tool>`
调用计算器、拿到结果，再用 `<answer>…</answer>` 给出最终答案；答对得 1.0 奖励。
用来跑通 NeMo-RL 0.6.0 的多轮 Agent / 自定义环境链路。

## 组成

- `config.yaml` — 继承 `configs/base/grpo_sliding_puzzle.yaml`（多轮基底）+ `configs/models/qwen3.5-9b.yaml`，
  只写差异；`env.calc_tool.cfg` 配置工具环境（`_override_` 整段替换基底的 sliding_puzzle env）。
- `run.py` — 自定义训练脚本：随机生成算术题（`DatumSpec`）、实例化
  `common/environments/example_tool_env.py::ToolAgentEnv`、调用 `grpo_train`。
- `run.sh` — 检测到本目录有 `run.py`，自动以它为入口。

## 环境协议

| 动作 | 模型输出 | 环境响应 |
| --- | --- | --- |
| 调用工具 | `<tool>calc: 2+3*4</tool>` | 返回 `calc(2+3*4) = 14` 作为下一轮 observation |
| 给出答案 | `<answer>14</answer>` | 校验对错，对则奖励 1.0 并结束 |

停止串 `</tool>` / `</answer>` 让模型每轮在动作处停下，交给环境推进。

## SwanLab

- project：`agent-grpo_qwen3.5-9b_calc-tool_v1`
- run：`turns6-g8-kl0.01`
- 链接：<首次运行后回填>

## 运行

```bash
# 需先准备 NeMo-RL 0.6.0 源码，并（2 节点 GB10 时）拉起 Ray 集群
NEMO_RL_DIR=/path/to/NeMo-RL CLUSTER_PROFILE=gb10-spark bash run.sh
# 单机 H200：
NEMO_RL_DIR=/path/to/NeMo-RL CLUSTER_PROFILE=h200 bash run.sh
```

产物落到本目录 `outputs/`（已 .gitignore）。`global_post_process_and_metrics` 会上报
`calc_tool_success_rate`（答对率）。

## 结果与结论

- 关键指标：calc_tool_success_rate / reward / 平均轮数
- 结论 / 下一步：

## 改造成你自己的工具任务

1. 在 `common/environments/example_tool_env.py` 的 `TOOLS` 里加工具（`name -> callable`）。
2. 改 `run.py` 的 `_make_problem` / `_build_prompt` 换成你的任务与数据。
3. 调 `config.yaml` 的 `grpo.max_rollout_turns`、`env.calc_tool.cfg`、超参。
