# cluster/ — 硬件 / 分布式 profile

把「在什么硬件上、用多少资源、什么并行度」从实验里抽出来，做到训练配置与硬件解耦。

每个 profile 一个子目录：

- `gb10-spark/` — 2× DGX Spark GB10（Ray 2 节点）
- `h200/` — H200（后续使用）

## 用法

实验的 `run.sh` 通过环境变量选择 profile：

```bash
CLUSTER_PROFILE=gb10-spark bash run.sh
# 或
CLUSTER_PROFILE=h200 bash run.sh
```

`run.sh` 会读取 `cluster/$CLUSTER_PROFILE/profile.yaml` 中的资源/并行度，叠加到该实验的训练配置上。

## 每个 profile 包含

- `profile.yaml` — 节点数、每节点 GPU 数、并行度、rollout 资源划分等
- 启动脚本（如 `start_ray_head.sh` / `start_ray_worker.sh`）
- `hosts`（本地维护，含真实 IP，已 .gitignore）
