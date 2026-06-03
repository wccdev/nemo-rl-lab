# grpo_qwen3.5-4b_gsm8k_v1

单轮 GRPO：`qwen3.5-4b` 在 GSM8K 上做数学推理。来自 2× DGX Spark GB10 实测的
`grpo_math_4B_megatron` 配方（LoRA dim16 / lr 2e-4），并改为**非 colocated 生成**。

## 与 9B 版的区别

| 项 | 4B（本实验） | 9B（`grpo_qwen3.5-9b_gsm8k_v1`） |
| --- | --- | --- |
| LoRA | dim16 / alpha32 / **lr 2e-4** | dim8 / alpha16 / lr 1e-4 |
| batch | num_prompts=4 / gen=4 / global=16 | num_prompts=4 / gen=8 / global=32 |
| 序列 | 1024 | 1250 |
| 生成 | **非 colocated**（1 卡生成 / 1 卡训练） | 非 colocated |

> **非 colocated + 2×GB10 ⇒ PP=1**：2 张卡里 1 张专跑生成、1 张训练，训练侧只剩 1 卡，
> 无法 PP=2（PP=2 需要 2 张训练卡，那要回 colocated）。并行度在 `cluster/gb10-spark/overrides.conf`。

## 数据准备（与 9B 共用）

```bash
lab prepare gsm8k                              # 写到 datasets/gsm8k/{train,val}.jsonl
export GSM8K_DATA_DIR="$(pwd)/datasets/gsm8k"  # 供 config 的 ${oc.env:GSM8K_DATA_DIR} 解析
```

## 组成

- `config.yaml` — 继承 `grpo_math_1B` + `qwen3.5-4b` + `grpo_megatron`(Megatron+GB10) + `grpo_lora`(覆盖成 4B 配方) + `grpo_noncolocated`。
- `run.sh` — 无自定义 `run.py`，自动用官方入口 `examples/run_grpo.py`。

## 运行

```bash
# 提交到集群（推荐）
lab submit grpo_qwen3.5-4b_gsm8k_v1
#   或在集群容器内直接跑：
NEMO_RL_DIR=/opt/NeMo-RL CLUSTER_PROFILE=gb10-spark lab run grpo_qwen3.5-4b_gsm8k_v1
```

## SwanLab

- project：`grpo_qwen3.5-4b_gsm8k_v1`，run：`lora-lr2e4-dim16-noncolo`，链接：<回填>

## 结果与结论

- 关键指标：val:accuracy / reward
- 结论 / 下一步：
