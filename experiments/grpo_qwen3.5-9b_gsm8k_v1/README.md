# grpo_qwen3.5-9b_gsm8k_v1

单轮 GRPO 实验：`qwen3.5-9b` 在 GSM8K 上做数学推理。作为多轮 Agent 实验的**对照**
（同模型、同 GRPO，但单轮、无工具）。

## 目标

用标准 GRPO（`max_rollout_turns=1`）在 GSM8K 上训练，观察答对率随训练提升，与
`agent-grpo_qwen3.5-9b_multitool_v1` 的多轮工具范式对比。

## 数据准备（先做）

GSM8K 的答案是「推理 + #### 数字」，需抽取干净金标准答案：

```bash
# 在仓库根目录
python common/data/prepare_gsm8k.py          # 写到 datasets/gsm8k/{train,val}.jsonl
export GSM8K_DATA_DIR="$(pwd)/datasets/gsm8k"  # 供 config.yaml 的 ${oc.env:GSM8K_DATA_DIR} 解析
```

## 组成

- `config.yaml` — 继承 `configs/base/grpo_math_1B.yaml` + `qwen3.5-9b`，`_override_` 替换 `data`
  为 `ResponseDataset` 指向上面的 jsonl，处理器 `math_hf_data_processor`、环境 `math`。
- `run.sh` — 无自定义 `run.py`，自动用官方入口 `examples/run_grpo.py`。

## SwanLab

- project：`grpo_qwen3.5-9b_gsm8k_v1`，run：`lr1e6-g16-kl0.01`，链接：<回填>

## 运行

```bash
# 确保已 export GSM8K_DATA_DIR
NEMO_RL_DIR=/path/to/NeMo-RL CLUSTER_PROFILE=gb10-spark bash run.sh
```

产物落到本目录 `outputs/`（已 .gitignore）。

## 结果与结论

- 关键指标：val:accuracy / reward
- 结论 / 下一步：
