# 2× DGX Spark (GB10) 集群

硬件：2 台 NVIDIA DGX Spark，每台 GB10 Grace-Blackwell 超级芯片（统一内存，aarch64 架构）。NeMo-RL 用 **Ray** 把两台机器组成一个集群。

> 注意：GB10 为 ARM64 + Blackwell 架构，安装依赖时请使用对应的 aarch64 / CUDA 容器或 wheel。详见 `env/README.md`。

## 0. 运行位置（重要）

训练实际计算**必须在 Spark 的 GPU 上、NeMo-RL 容器里**执行。Mac 只是开发机（写代码 / git / SSH / 看 SwanLab），**不能在本机跑训练**——Mac 无 NVIDIA GPU/CUDA，驱动进程 import torch/vllm/nemo_rl 就起不来。

两种「代码」要分清：

| 代码 | 怎么进集群 |
| --- | --- |
| **NeMo-RL 框架**（torch/vllm/nemo_rl + CUDA） | 预先装在容器里（一次），不随作业走 |
| **本仓库**（configs/common/run.py） | 方式 A 在 Spark 上 git pull / 共享挂载；方式 B 由 `ray job submit --working-dir` 自动上传分发到所有节点 |

> 自定义环境（如 `common/environments/` 的多工具 Agent）会作为 Ray actor 跑在 **worker** 上。
> 方式 B 会把 working-dir 同步到所有 worker，actor 才能 `import common.*`；方式 A 则需保证 worker 也能 import 到本仓库（共享路径 / PYTHONPATH）。

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

> Ray head/worker 建议在**容器内**启动（用 `cluster/gb10-spark/start_ray_*.sh`），
> 这样作业才跑在 NeMo-RL 环境里。

## 3. 跑训练（两种方式）

实验的 `run.sh` 通过 `CLUSTER_PROFILE=gb10-spark` 选择本 profile，并自动追加
`cluster/gb10-spark/overrides.conf` 里的覆盖项（`cluster.num_nodes=2` 等）。

**方式 A：SSH 到 head 容器内直接跑**

```bash
NEMO_RL_DIR=/opt/NeMo-RL CLUSTER_PROFILE=gb10-spark bash experiments/<exp>/run.sh
```

**方式 B：从 Mac 一键提交（推荐，执行仍在集群）**

```bash
# 一次性：开发机装 Ray CLI + 配好集群地址
pip install "ray[default]"
cp cluster/submit.env.example cluster/submit.env   # 填 RAY_DASHBOARD_ADDRESS / NEMO_RL_DIR / SWANLAB_API_KEY
# 提交（代码自动上传，无需手动在 Spark git pull）
bash scripts/submit_job.sh experiments/<exp> [gb10-spark]
```

## 4. 资源与并行度建议

- GB10 为统一内存架构，单卡可用显存较大但带宽 / 算力与数据中心卡不同，**batch size、序列长度需实测**。
- 2 节点优先用数据并行（DTensor/FSDP 在 `num_nodes*gpus_per_node` 张卡上分片）；跨节点张量并行（TP）通常不划算，保持 `policy.dtensor_cfg.tensor_parallel_size=1`。
- GRPO 的 rollout（vLLM 生成）开销大：默认 `colocated.enabled=true` 与训练共用 GPU；资源紧张可在 `overrides.conf` 调整。
- 这些都写在 `cluster/gb10-spark/overrides.conf`，按实测调整。
