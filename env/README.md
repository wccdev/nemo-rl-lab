# 环境与依赖

本仓库以 **NVIDIA NeMo-RL 0.6.0** 为主框架。NeMo-RL 用 **`uv`** 管理依赖与运行（`uv run python ...`）。由于硬件横跨两种架构，依赖按架构分别维护。

## 架构差异

| 硬件 | 架构 | 安装方式 |
| --- | --- | --- |
| DGX Spark GB10 | aarch64 + Blackwell | aarch64 容器 / wheel |
| H200 | x86_64 + Hopper | x86_64 容器 / wheel |

强烈建议用 **NeMo-RL 官方容器镜像** 跑训练，避免手装 CUDA / 依赖踩坑。

## 安装 NeMo-RL 0.6.0

```bash
git clone --branch v0.6.0 https://github.com/NVIDIA-NeMo/RL.git NeMo-RL
cd NeMo-RL
# NeMo-RL 用 uv 管理环境；首次运行会自动建虚拟环境
uv sync
# 验证
uv run python examples/run_grpo.py --help
```

> 具体安装命令、容器 tag、各 backend（DTensor / Megatron）的额外依赖以 v0.6.0 官方文档为准。
> 把克隆下来的 NeMo-RL 目录路径设为 `NEMO_RL_DIR`，实验 `run.sh` 会用到。

## 运行实验

```bash
NEMO_RL_DIR=/path/to/NeMo-RL CLUSTER_PROFILE=gb10-spark bash experiments/<exp>/run.sh
```

## 本机开发期依赖（uv 项目）

本仓库根目录用 **`pyproject.toml` + `uv`** 管理开发期依赖（`typer` + `datasets`），并把统一 CLI 注册为命令入口 `lab`（`[project.scripts] lab = nemo_rl_lab.cli:app`）。重型框架（NeMo-RL / vLLM / Ray / CUDA）不在此管理，由 NeMo-RL 的 `uv` 环境或官方容器提供。

```bash
uv run lab ls           # 首次会自动 uv sync（建 .venv、装 typer/datasets、editable 安装 lab）
uv run lab prepare gsm8k
# 等价：./lab ...（薄 shim）；或 uv sync 后激活 venv 直接用 `lab ...`
```

> 三种调用方式见根目录 README 的「统一 CLI」一节。`uv run lab` 对任何 clone 仓库的人都开箱即用，无需手动装包。

## 密钥

`SWANLAB_API_KEY`、HuggingFace token 等放仓库根目录本地 `.env`（已 `.gitignore`），不要入库。
