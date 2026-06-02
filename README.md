# nemo-rl-lab

基于 **NVIDIA NeMo-RL 0.6.0** 的大模型微调实验室。涵盖：

- **SFT**（监督微调）
- **GRPO / 强化学习**（RL）
- **多轮 Agent 训练**（NeMo-RL 较新特性，工具调用 / 多轮对话）

横跨多个基础模型（如 `qwen3.5-4b`、`qwen3.5-9b` 等）× 多个数据集。所有训练日志统一上传到云端 **SwanLab**。

## 硬件

| Profile | 说明 | 配置目录 |
| --- | --- | --- |
| `gb10-spark` | 2× NVIDIA DGX Spark（GB10 Grace-Blackwell），通过 Ray 组成 2 节点集群 | `cluster/gb10-spark/` |
| `h200` | NVIDIA H200（后续使用），SFT / GRPO 均会涉及 | `cluster/h200/` |

训练配置与硬件解耦：NeMo-RL 0.6.0 通过 CLI override 调集群（`cluster.num_nodes` / `cluster.gpus_per_node`）；硬件相关 override 抽到 `cluster/<profile>/overrides.conf`，切换硬件只换 profile。

## 目录结构

```
nemo-rl-lab/
├── README.md                 # 本文件：总览
├── .gitignore
├── docs/                     # 文档
│   ├── naming-convention.md  # 命名规范（务必先读）
│   ├── setup-dgx-spark-gb10.md
│   ├── setup-h200.md
│   └── swanlab.md            # SwanLab 接入说明
├── env/                      # 环境与依赖
│   ├── README.md
│   └── requirements.txt
├── cluster/                  # 硬件 / 分布式 profile（与训练解耦）
│   ├── gb10-spark/           # 2× DGX Spark GB10
│   └── h200/                 # H200
├── configs/                  # 按方法的推荐 override（速查表，非完整 yaml）
│   ├── sft/
│   ├── grpo/
│   └── agent/
├── common/                   # 跨实验复用代码
│   ├── data/                 # 数据处理 / data processor
│   ├── environments/         # 自定义 Environment（GRPO 奖励来源 / 多轮 Agent）
│   └── utils/
├── datasets/                 # 数据集「元数据」（不放大文件，见下方约定）
├── templates/                # 新实验脚手架模板
│   └── experiment-template/
├── experiments/              # 练习 / 探索性实验
└── projects/                 # 正式 / 交付级项目
```

> 工作流：实验不存完整 yaml，而是用官方 v0.6.0 example 配置作 `--config` 基底
> （见 `configs/README.md` 的方法对照表），实验级改动写在各实验的 `overrides.conf`，
> 由 `run.sh` 合成 `uv run python <entry> --config <base> key=value ...` 命令。

## experiments vs projects

- **`experiments/`**：练习、调参、试错、复现。允许快糙猛，但每个目录必须有 `README.md` 记录目标、结论、SwanLab 链接。
- **`projects/`**：正式项目，要求可复现：固定依赖、固定数据版本、完整 eval、产出 checkpoint 导出流程。

两者内部目录布局一致（见 `templates/experiment-template/`），区别只是成熟度要求。

## 命名规范（核心）

每个实验目录统一命名为：

```
<method>_<model>_<dataset>[_<tag>]
```

- `method`：`sft` | `grpo` | `dpo` | `ppo` | `rm`（奖励模型）| `agent-grpo`（多轮 Agent）
- `model`：`qwen3.5-4b` | `qwen3.5-9b` | ...
- `dataset`：`gsm8k` | `alpaca` | `toolbench` | ...
- `tag`：可选，`v1` / `v2` 或日期 `20260602`

示例：

```
sft_qwen3.5-4b_alpaca_v1
grpo_qwen3.5-9b_gsm8k_v2
agent-grpo_qwen3.5-9b_toolbench_v1
```

字段间用 `_` 分隔，字段内（如模型名 `qwen3.5-4b`）用 `-`，避免歧义。完整规则见 [`docs/naming-convention.md`](docs/naming-convention.md)。

## 新建一个实验

```bash
# 从模板生成新实验（练习放 experiments/，正式放 projects/）
bash scripts/new_experiment.sh experiments grpo_qwen3.5-4b_gsm8k_v1
cd experiments/grpo_qwen3.5-4b_gsm8k_v1
# 1. 改 README.md：目标 / 基础模型 / 数据集 / SwanLab 项目名
# 2. 在 run.sh 顶部设 ENTRY 与 BASE_CONFIG（见 configs/README.md）
# 3. 把 configs/<method>/overrides.example.conf 拷进 overrides.conf 并修改
# 4. 选硬件 profile 运行：
NEMO_RL_DIR=/path/to/NeMo-RL CLUSTER_PROFILE=gb10-spark bash run.sh
```

## 快速开始

1. 装 NeMo-RL 0.6.0 与依赖：[`env/README.md`](env/README.md)
2. 配置 SwanLab：[`docs/swanlab.md`](docs/swanlab.md)
3. 集群 / 硬件 profile：[`cluster/README.md`](cluster/README.md)、[`docs/setup-dgx-spark-gb10.md`](docs/setup-dgx-spark-gb10.md)
4. 命名规范：[`docs/naming-convention.md`](docs/naming-convention.md)
