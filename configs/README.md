# configs/ — 配置继承体系

NeMo-RL 0.6.0 **原生支持配置继承**（`nemo_rl/utils/config.py` 的 `load_config`）：配置里写
`defaults: parent.yaml` 即可继承，支持多继承、嵌套继承、`${}` 插值、`_override_` 整段覆盖，
再叠加命令行 Hydra override。官方自己就这么用（`grpo_sliding_puzzle.yaml` 继承 `grpo_math_1B.yaml`）。

所以每个模型 / 每个实验都有自己的配置、不断调参，但**只写差异**，公共部分继承。

## 三层结构

```
configs/
├── base/      祖父：官方 v0.6.0 example 原样副本（version-locked，勿手改）
│   ├── grpo_math_1B.yaml          # GRPO 基底
│   ├── sft.yaml                   # SFT 基底
│   └── grpo_sliding_puzzle.yaml   # 多轮 Agent 基底（已继承 grpo_math_1B）
└── models/    父：各基础模型的公共片段（model_name / tokenizer / 显存策略…）
    ├── qwen3.5-4b.yaml
    └── qwen3.5-9b.yaml
```

实验目录里的 `config.yaml` 是**子**：

```yaml
defaults:
  - ../../configs/base/grpo_math_1B.yaml   # 方法基底
  - ../../configs/models/qwen3.5-9b.yaml   # 模型片段
# 下面只写本实验差异：数据集 / lr / kl / swanlab / 步数 ...
grpo:
  num_generations_per_prompt: 16
loss_fn:
  reference_policy_kl_penalty: 0.01
logger:
  swanlab_enabled: true
  swanlab: { project: "grpo_qwen3.5-9b_gsm8k_v1", name: "lr1e6-g16-kl0.01" }
```

> 合并规则：多继承中**后面覆盖前面**，实验自身再覆盖基底/模型。需要整段替换（不合并）时，
> 在该段加 `_override_: true`。

## 方法 → 入口 → 基底

| 方法 | `ENTRY`（run.sh） | 基底（defaults 第一项） |
| --- | --- | --- |
| SFT | `examples/run_sft.py` | `../../configs/base/sft.yaml` |
| GRPO | `examples/run_grpo.py` | `../../configs/base/grpo_math_1B.yaml` |
| 多轮 Agent | `examples/run_grpo.py` | `../../configs/base/grpo_sliding_puzzle.yaml` |

> 更大模型可加 `grpo_math_8B.yaml` 等官方基底：把文件名加进 `scripts/sync_base_configs.sh` 同步。

## 集群 / 硬件

不进 `config.yaml`，由 `cluster/<profile>/overrides.conf` 在运行时以 CLI override 叠加（见 `cluster/README.md`），
这样切硬件（GB10 ↔ H200）不动实验配置。

## 常用字段（0.6.0）

| 作用 | key |
| --- | --- |
| 基础模型 | `policy.model_name` |
| 序列长度 | `policy.max_total_sequence_length` |
| 学习率 | `policy.optimizer.kwargs.lr` |
| KL 惩罚 | `loss_fn.reference_policy_kl_penalty` |
| 每步 prompt 数 / 每 prompt 采样数 | `grpo.num_prompts_per_step` / `grpo.num_generations_per_prompt` |
| 多轮轮数 | `grpo.max_rollout_turns` |
| 节点 / 卡数 | `cluster.num_nodes` / `cluster.gpus_per_node` |
| 启用 SwanLab | `logger.swanlab_enabled` + `logger.swanlab.{project,name}` |
