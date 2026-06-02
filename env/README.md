# 环境与依赖

本仓库以 **NVIDIA NeMo-RL** 为主框架。由于硬件横跨两种架构，依赖按架构分别维护。

## 架构差异

| 硬件 | 架构 | 安装方式 |
| --- | --- | --- |
| DGX Spark GB10 | aarch64 + Blackwell | 用 NVIDIA 提供的 aarch64 容器 / wheel |
| H200 | x86_64 + Hopper | 用 x86_64 容器 / wheel |

强烈建议用 **NeMo / NeMo-RL 官方容器镜像** 跑训练，避免手装 CUDA / 依赖踩坑。

## 安装 NeMo-RL

```bash
# 方式一（推荐）：官方容器
# docker run --gpus all -it --rm nvcr.io/nvidia/nemo:<tag>

# 方式二：源码安装（按官方文档为准）
git clone https://github.com/NVIDIA/NeMo-RL.git
cd NeMo-RL
pip install -e .
```

> NeMo-RL 的具体安装命令、镜像 tag 以官方文档为准；不同版本对多轮 Agent 训练的支持不同，记录在各实验 README 里。

## 通用 Python 依赖

见 `requirements.txt`（轻量工具依赖；重型框架走容器）。

## 密钥

`SWANLAB_API_KEY`、HuggingFace token 等放到仓库根目录的本地 `.env`（已被 `.gitignore` 排除），不要入库。
