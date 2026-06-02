# 2× DGX Spark (GB10) 集群

硬件：2 台 NVIDIA DGX Spark，每台 GB10 Grace-Blackwell 超级芯片（统一内存，aarch64 架构）。NeMo-RL 用 **Ray** 把两台机器组成一个集群。

> 注意：GB10 为 ARM64 + Blackwell 架构，安装依赖时请使用对应的 aarch64 / CUDA 容器或 wheel。详见 `env/README.md`。

## 1. 组网

两台机器需在同一内网互通，建议固定主机名 / IP：

| 角色 | 主机名（示例） | IP（示例） |
| --- | --- | --- |
| head | `spark-0` | `192.168.1.10` |
| worker | `spark-1` | `192.168.1.11` |

把真实地址填到 `cluster/gb10-spark/hosts`（该文件已被 .gitignore 排除，本地维护）。

## 2. 启动 Ray 集群

head 节点：

```bash
ray start --head --port=6379 --dashboard-host=0.0.0.0
```

worker 节点：

```bash
ray start --address='spark-0:6379'
```

校验：

```bash
ray status   # 应能看到 2 个节点的资源
```

`cluster/gb10-spark/` 下提供了启动脚本与 Ray 集群配置示例。

## 3. 在该 profile 上跑训练

实验的 `run.sh` 通过 `CLUSTER_PROFILE=gb10-spark` 选择本 profile：

```bash
CLUSTER_PROFILE=gb10-spark bash run.sh
```

## 4. 资源与并行度建议

- GB10 为统一内存架构，单卡可用显存较大但带宽 / 算力与数据中心卡不同，**batch size、序列长度需实测**。
- 2 节点优先用数据并行；模型较大时再叠加张量 / 流水并行（在 `cluster/gb10-spark/profile.yaml` 配置）。
- GRPO 的 rollout（生成）开销大，建议把 vLLM 推理与训练的资源划分写进 profile。
