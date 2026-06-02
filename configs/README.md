# configs/ — 按方法的配置速查 & 推荐 override

NeMo-RL 0.6.0 的训练配置文件较长且字段众多，**直接改官方 example 配置容易出错**。本仓库采用更稳的工作流：

> **以官方 v0.6.0 example 配置为 `--config` 基底，用 CLI override 改我们关心的字段。**

实验里不存一份完整 yaml，而是存一份 `overrides.conf`（`key=value` 清单），由 `run.sh` 合成命令。

## 方法 → 入口脚本 → 基础配置

| 方法 | ENTRY（NeMo-RL 入口） | BASE_CONFIG（官方 example） |
| --- | --- | --- |
| SFT | `examples/run_sft.py` | `examples/configs/sft.yaml` |
| GRPO | `examples/run_grpo.py` | `examples/configs/grpo_math_1B.yaml` |
| 多轮 Agent | `examples/run_grpo.py` | 多轮示例如 `examples/configs/grpo_sliding_puzzle.yaml`（或自定义环境配置） |

> 以上路径相对 NeMo-RL 0.6.0 源码目录。更大模型可换 `grpo_math_8B.yaml` 等官方 example。

## 各方法的推荐 override

- `sft/overrides.example.conf`
- `grpo/overrides.example.conf`
- `agent/overrides.example.conf`

新建实验时，把对应文件内容拷进实验目录的 `overrides.conf` 再改。

## 关键字段对照（0.6.0）

| 作用 | override key |
| --- | --- |
| 基础模型 | `policy.model_name` |
| 序列长度 | `policy.max_total_sequence_length` |
| 学习率 | `policy.optimizer.kwargs.lr` |
| KL 惩罚 | `loss_fn.reference_policy_kl_penalty` |
| GRPO 每步 prompt 数 | `grpo.num_prompts_per_step` |
| 每 prompt 采样数 | `grpo.num_generations_per_prompt` |
| 多轮轮数 | `grpo.max_rollout_turns` |
| 节点 / 卡数 | `cluster.num_nodes` / `cluster.gpus_per_node` |
| checkpoint 目录 | `checkpointing.checkpoint_dir` |
| 启用 SwanLab | `logger.swanlab_enabled=true` |
| SwanLab 项目 / run | `logger.swanlab.project` / `logger.swanlab.name` |
