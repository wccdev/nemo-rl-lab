# cluster/ — 硬件 / 分布式 profile

NeMo-RL 0.6.0 的集群设置（`cluster.num_nodes`、`cluster.gpus_per_node`）和并行度都在训练配置里，通过 **CLI override** 调整。本目录把「不同硬件的 override」抽出来复用，训练配置与硬件解耦。

每个 profile 一个子目录，核心是 `overrides.conf`（每行一个 `key=value`，`#` 为注释）：

- `gb10-spark/` — 2× DGX Spark GB10（Ray 2 节点）
- `h200/` — H200（后续使用）

## 用法

实验的 `run.sh` 通过环境变量选 profile，自动把对应 `overrides.conf` 追加到训练命令：

```bash
CLUSTER_PROFILE=gb10-spark bash run.sh
CLUSTER_PROFILE=h200       bash run.sh
```

等价于：

```bash
uv run python examples/run_grpo.py --config <base.yaml> \
    cluster.num_nodes=2 cluster.gpus_per_node=1 ...   # 来自 profile
```

## 各 profile 包含

- `overrides.conf` — 节点数 / 每节点 GPU / 并行度等 NeMo-RL 覆盖项
- `start_ray_head.sh` / `start_ray_worker.sh` — 启动 Ray 集群（多节点需要）
- `hosts.example` — 复制为 `hosts` 填真实 IP（`hosts` 已 .gitignore）

> 多节点训练前，先在各节点用 `start_ray_*.sh` 拉起 Ray 集群，再跑 `run.sh`。
> NeMo-RL 在 Slurm 上用官方 `ray.sub`；裸机 2 节点用这里的脚本手动起 Ray。
