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
├── lab                       # CLI 薄 shim（= uv run lab）
├── nemo_rl_lab/              # 统一 CLI 实现（Typer；cli.py 为入口）
├── pyproject.toml            # uv 项目：依赖 + lab 命令入口（[project.scripts]）
├── uv.lock                   # 锁定版本（uv sync 用）
├── README.md                 # 本文件：总览
├── .gitignore
├── docs/                     # 文档
│   ├── naming-convention.md  # 命名规范（务必先读）
│   ├── remote-submit.md      # 从 Mac 提交训练到 Ray 集群（完整操作指南）
│   ├── setup-dgx-spark-gb10.md
│   ├── setup-h200.md
│   └── swanlab.md            # SwanLab 接入说明
├── env/                      # 环境与依赖说明
│   └── README.md
├── cluster/                  # 硬件 / 分布式 profile（与训练解耦）
│   ├── gb10-spark/           # 2× DGX Spark GB10
│   └── h200/                 # H200
├── configs/                  # 配置继承体系（NeMo-RL 原生 defaults）
│   ├── base/                 # 祖父：官方 v0.6.0 example 原样副本（勿手改）
│   └── models/               # 父：各基础模型公共片段（qwen3.5-4b / 9b ...）
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

> 配置工作流：每个实验有自己的 `config.yaml`，通过 `defaults` **继承基底 + 模型片段，只写差异**
> （NeMo-RL 0.6.0 原生支持，官方亦如此）。`run.sh` 以该 `config.yaml` 为 `--config`，运行时再叠加
> `cluster/<profile>/overrides.conf` 的硬件 override。详见 `configs/README.md`。

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

## 统一 CLI（`lab`）

所有操作都通过 `lab` 入口（[Typer](https://typer.tiangolo.com) 实现，项目正式命令入口）：

```bash
uv run lab ls                                # 列出实验 / 项目
uv run lab new grpo_qwen3.5-4b_gsm8k_v1      # 从模板新建实验
uv run lab prepare gsm8k                     # 预处理数据集（gsm8k / alpaca / qa_rl）
uv run lab submit agent-grpo_qwen3.5-9b_multitool_v1   # 从本机提交作业到 Ray 集群（执行在集群）
uv run lab run grpo_qwen3.5-9b_gsm8k_v1 --nemo-rl /opt/NeMo-RL   # 在集群容器内直接跑
uv run lab ray head                          # 启动 Ray head（在 head 节点容器内）
uv run lab sync-base --nemo-rl /opt/NeMo-RL  # 升级版本时同步官方基底配置
```

三种等价调用方式：

| 方式 | 说明 |
| --- | --- |
| `uv run lab ...` | 推荐；uv 自动同步项目环境再运行，对任何人都生效（无需手动装包） |
| `./lab ...` | 仓库根的薄 shim，内部就是 `uv run lab`，可在任意目录用绝对路径调用 |
| `lab ...` | `uv sync` 后 `.venv/bin/lab` 已生成；激活 venv 即可直接用 |

`uv run lab <子命令> --help` 看每个命令的参数。CLI 只是对 `scripts/` 与各实验脚本的封装，单一事实来源。
实现见 `nemo_rl_lab/cli.py`。

### 终端补全（Tab）

子命令、实验名、数据集、profile 都能补全（bash / zsh / fish / powershell）。安装一次即可：

```bash
uv run lab --install-completion      # 安装到当前 shell，重开终端生效
uv run lab --show-completion         # 只打印脚本，自行决定放哪
```

之后 `lab sub<Tab>` → `submit`，`lab submit <Tab>` 列出实验名，`lab submit --profile <Tab>` 列出 profile。
（补全基于 `lab` 命令名；建议 `uv sync` 后用激活的 venv，或把 `.venv/bin` 加进 PATH。）

## 新建一个实验（细节）

```bash
uv run lab new grpo_qwen3.5-4b_gsm8k_v1   # 或 bash scripts/new_experiment.sh experiments <name>
cd experiments/grpo_qwen3.5-4b_gsm8k_v1
# 1. 改 README.md：目标 / 基础模型 / 数据集 / SwanLab 项目名
# 2. 改 config.yaml：选 defaults（基底 + 模型片段），写本实验差异（lr/kl/数据集/swanlab）
# 3. 若是 SFT/Agent，在 run.sh 顶部把 ENTRY 改成对应入口（见 configs/README.md）
# 4. 提交到集群（推荐）或在集群容器内直接跑：
uv run lab submit grpo_qwen3.5-4b_gsm8k_v1
```

## 示例实验（覆盖三种方法）

| 实验 | 方法 | 说明 |
| --- | --- | --- |
| [`experiments/sft_qwen3.5-4b_alpaca_v1`](experiments/sft_qwen3.5-4b_alpaca_v1) | SFT | Alpaca 指令监督微调（本地 jsonl + ResponseDataset） |
| [`experiments/grpo_qwen3.5-4b_gsm8k_v1`](experiments/grpo_qwen3.5-4b_gsm8k_v1) | GRPO（单轮） | GSM8K 数学推理（4B + LoRA dim16/lr2e-4，非 colocated） |
| [`experiments/grpo_qwen3.5-9b_gsm8k_v1`](experiments/grpo_qwen3.5-9b_gsm8k_v1) | GRPO（单轮） | GSM8K 数学推理，math 环境验证 |
| [`experiments/grpo_qwen3.5-9b_qa-rl_v1`](experiments/grpo_qwen3.5-9b_qa-rl_v1) | GRPO（单轮，自定义判分） | 自有技术培训考题；客观题规则判分 + 简答 LLM 裁判 |
| [`experiments/agent-grpo_qwen3.5-9b_multitool_v1`](experiments/agent-grpo_qwen3.5-9b_multitool_v1) | GRPO（多轮 Agent） | 多工具（检索/计算/代码）调用，自定义环境 |

数据预处理脚本见 `common/data/`（gsm8k / alpaca / qa_rl）。自定义环境见 `common/environments/`，判分逻辑见 `common/rewards/`。

## 训练工作流（Mac → 集群）

**在 Mac 上写代码 + 提交，训练跑在 Spark GB10 容器里**，日常提交不进容器、不需要 GPU、代码随作业自动上传。
底层是 Ray 官方的 Job Submission（`ray job submit` → head dashboard:8265）。

```bash
# A. 一次性：在两台容器里把 Ray 集群组好（之后反复提交都不用再做；也可从 Mac 用 ssh 远程触发）
#    node1 容器: bash cluster/gb10-spark/start_ray_head.sh
#    node2 容器: bash cluster/gb10-spark/start_ray_worker.sh

# B. 一次性：Mac 端装 Ray CLI（uv 管理，版本对齐集群）+ 填提交配置
uv sync --extra submit
cp cluster/submit.env.example cluster/submit.env   # 填 RAY_DASHBOARD_ADDRESS / NEMO_RL_DIR / 密钥

# C. 每次：在 Mac 上提交、看/停作业（lab job 自动读 submit.env 的地址）
uv run lab submit grpo_qwen3.5-4b_gsm8k_v1
uv run lab job list                 # 查看作业
uv run lab job logs <job_id> -f     # 实时日志
uv run lab job stop <job_id>        # 停止作业
```

> 完整步骤、网络/SSH 隧道、上传规则、监控、排错 → **[`docs/remote-submit.md`](docs/remote-submit.md)**。

## 快速开始

1. 装 NeMo-RL 0.6.0 与依赖：[`env/README.md`](env/README.md)
2. 配置 SwanLab：[`docs/swanlab.md`](docs/swanlab.md)
3. 集群 / 硬件 profile：[`cluster/README.md`](cluster/README.md)、[`docs/setup-dgx-spark-gb10.md`](docs/setup-dgx-spark-gb10.md)
4. **从 Mac 提交到集群**：[`docs/remote-submit.md`](docs/remote-submit.md)
5. 命名规范：[`docs/naming-convention.md`](docs/naming-convention.md)
