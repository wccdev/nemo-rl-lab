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

- `overrides.conf` — 节点数 / 每节点 GPU / 并行度等 NeMo-RL 覆盖项（CLI override）
- `env.sh` — 集群 env（NCCL/RoCE 网络 + Ray 内存监控 + PyTorch 显存分配）；GB10 实测配置，被 ray 启动脚本和实验 `run.sh` 统一 source，一处改处处生效
- `start_ray_head.sh` / `start_ray_worker.sh` — 启动 Ray 集群（多节点需要）
- `hosts.example` — 复制为 `hosts` 填真实 IP（`hosts` 已 .gitignore）

> `overrides.conf` 走 CLI override（进 NeMo-RL 配置）；`env.sh` 走进程环境变量（NCCL/Ray/PyTorch 这类不属于训练配置的开关）。两者互补。

## 多节点起 Ray（GB10，2 节点）

> 单机不用手动起 Ray（NeMo-RL 自动拉起本地 Ray）。多机才需要先把集群起好，作业再连上去。

在 **NeMo-RL 容器内**执行（`ray` 由 NeMo-RL 的 uv 环境提供，所以脚本用 `uv run ray` 并 `cd` 到 `NEMO_RL_DIR`）：

```bash
# head 节点容器（默认 HEAD_IP=192.168.1.4）
NEMO_RL_DIR=/opt/nemo-rl bash cluster/gb10-spark/start_ray_head.sh
#   或： uv run lab ray head --nemo-rl /opt/nemo-rl

# worker 节点容器（默认 NODE_IP=192.168.1.5，连 HEAD_IP:6379）
NEMO_RL_DIR=/opt/nemo-rl bash cluster/gb10-spark/start_ray_worker.sh
#   或： uv run lab ray worker --nemo-rl /opt/nemo-rl --head 192.168.1.4:6379

# 确认两个节点都在
uv run lab ray status --nemo-rl /opt/nemo-rl     # 应看到 2 nodes
```

可调环境变量：`HEAD_IP / NODE_IP / RAY_PORT / HEAD_ADDRESS / OBJECT_STORE_MEM / NEMO_RL_DIR`。
网卡名 / HCA / IB 参数在 `env.sh` 里改（当前是两台 Spark 的实测值）。

集群起好后再跑训练（`lab submit` 提交到 dashboard，或在 head 容器 `lab run`）。
NeMo-RL 在 Slurm 上用官方 `ray.sub`；裸机 2 节点用这里的脚本手动起 Ray。

> **从本机（Mac）提交训练到这个集群**的完整操作（含网络/SSH 隧道、上传规则、监控、排错）：
> 见 [`docs/remote-submit.md`](../docs/remote-submit.md)。
