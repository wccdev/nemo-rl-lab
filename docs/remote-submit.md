# 从本机（Mac）提交训练到 Ray 集群

本仓库的标准工作流：**在 Mac 上写代码、提交作业；训练真正跑在 Spark GB10 集群的容器里。**
日常提交**不需要进容器、不需要 GPU、不需要手动在集群上 `git pull`**——代码随作业自动上传。

底层用的是 Ray 官方的 **Ray Job Submission**（`ray job submit` → head 节点 dashboard:8265），
这是向「已存在的 Ray 集群」远程提交作业的通用做法。

---

## 1. 全景图

```
   你的 Mac（开发机，无需 GPU）                Spark GB10 集群（2 台，每台 1 卡）
 ┌─────────────────────────────┐          ┌──────────────────────────────────────┐
 │ 写 config / 代码            │          │  node1 容器: Ray head + NeMo-RL 环境   │
 │ uv run lab submit <exp>     │  ──网络── │  node2 容器: Ray worker + NeMo-RL 环境 │
 │   = ray job submit          │  RoCE/IB  │  作业在这两个容器里执行（用 GPU）       │
 │   --working-dir .（传代码） │          │                                        │
 └─────────────────────────────┘          └──────────────────────────────────────┘
        ↑ 反复提交，全程不进容器                   ↑ 只在「开集群」时碰一次（见 §3）
```

**职责划分（重要）**

| 事项 | 在哪做 | 频率 |
| --- | --- | --- |
| 起 Ray 集群（head/worker） | 集群容器内（或从 Mac 用 ssh 远程触发，见 §3） | 一次性（每次开集群一次） |
| 提交训练作业 | **Mac**（`lab submit`） | 每个作业 |
| 看日志 / 状态 | **Mac**（`ray job logs` / dashboard / SwanLab） | 随时 |

为什么起 Ray 必须在容器里：Ray 的 worker 进程就是「带 GPU 的那两个容器」本身，且 `ray` 由容器内
NeMo-RL 的 uv 环境提供。集群一旦起好，后面所有提交都在 Mac 上，不用再碰容器。

---

## 2. 前置条件（一次性）

### 2.1 Mac 端

Ray CLI（job submission 客户端，仅提交用、无需 GPU）由 **uv 管理**，作为可选依赖 `submit`，
版本已对齐 NeMo-RL 0.6.0 的 `ray[default]==2.54.0`（避免与集群 Ray 协议不兼容）。**不要用 `pip install`**
（Homebrew Python 会被 PEP 668 拦，且版本不可控）。

```bash
# 安装 Mac 端 Ray CLI（一次）。也可不手动装：lab submit 会用 `uv run --extra submit` 自动按需装。
uv sync --extra submit

# 本仓库的提交配置（含集群地址 / 容器内路径 / 密钥），从模板复制后填写。
cp cluster/submit.env.example cluster/submit.env
```

> `lab submit` 内部用 `uv run --extra submit ray job submit`，所以即使没先 `uv sync --extra submit`，
> 首次提交时 uv 也会把 Ray 装好。手动 `uv sync --extra submit` 只是想提前装好 / 让激活的 venv 里有 `ray` 命令。

`cluster/submit.env`（已 `.gitignore`，不会入库）关键字段：

```bash
RAY_DASHBOARD_ADDRESS=http://192.168.1.4:8265   # head 节点 dashboard 地址
NEMO_RL_DIR=/opt/NeMo-RL                          # 容器内 NeMo-RL 路径（不是 Mac 路径）
DEFAULT_CLUSTER_PROFILE=gb10-spark
SWANLAB_API_KEY=...                               # 训练日志上云
HF_TOKEN=...                                       # 下载 gated 模型/数据
# HF_ENDPOINT=...                                  # 【勿在 submit 里设】仅本机 lab prepare 时用镜像
HF_HOME=/home/aidenlu/nemo-rl-work/hf_cache       # 集群模型缓存（须先 prefetch，见 §4）
HF_HUB_ENABLE_HF_TRANSFER=0
```

### 2.2 网络可达

Mac 必须能访问 head 的 `8265`（dashboard/提交）和 `6379`（Ray GCS）。

- **同一局域网**（Mac 和 Spark 都在 `192.168.1.x`）：直接通，填真实 IP 即可。
- **不在同一网**：用 SSH 隧道把端口转发到本地：
  ```bash
  ssh -N -L 8265:192.168.1.4:8265 -L 6379:192.168.1.4:6379 <跳板机或集群可达主机>
  # 然后 submit.env 里改成：
  #   RAY_DASHBOARD_ADDRESS=http://127.0.0.1:8265
  ```
  `start_ray_head.sh` 已带 `--dashboard-host=0.0.0.0`，dashboard 对外可达。

### 2.3 集群容器

- 容器里已安装 **NeMo-RL 0.6.0**（框架本身不随作业上传，必须预装在镜像/容器里）。
- 容器里有 `uv`（`ray` / 训练都靠 `uv run` 在 NeMo-RL 环境跑）。

---

## 3. 起 Ray 集群（一次性，每次开集群）

> 单机不用手动起（NeMo-RL 会自动拉本地 Ray）。**2 台机器必须先把集群组好**，Mac 提交的作业才有地方跑。

### 方式 A：在容器内执行

```bash
# node1（head）容器内：
bash cluster/gb10-spark/start_ray_head.sh
#   等价：uv run lab ray head --nemo-rl /opt/NeMo-RL

# node2（worker）容器内：
bash cluster/gb10-spark/start_ray_worker.sh
#   等价：uv run lab ray worker --nemo-rl /opt/NeMo-RL --head 192.168.1.4:6379

# 任一容器内确认两个节点都在：
uv run lab ray status --nemo-rl /opt/NeMo-RL   # 应看到 2 nodes
```

这两个脚本会自动 `source cluster/gb10-spark/env.sh`（NCCL/RoCE 网络 + Ray 内存等 GB10 实测 env），
并用 `uv run ray start` 在 NeMo-RL 环境里拉起 Ray。可调变量：`HEAD_IP / NODE_IP / RAY_PORT / OBJECT_STORE_MEM / NEMO_RL_DIR`。

### 方式 B：从 Mac 远程触发（不进容器交互）

如果不想 ssh 进容器,用 `ssh + docker exec` 一行触发（把 `<容器名>` / 路径换成你的）：

```bash
ssh spark-1 'docker exec <容器名> bash -lc "cd /work/llm && bash cluster/gb10-spark/start_ray_head.sh"'
ssh spark-2 'docker exec <容器名> bash -lc "cd /work/llm && bash cluster/gb10-spark/start_ray_worker.sh"'
```

> 集群只要不关，这步做一次即可，之后反复提交作业都不用重做。

---

## 4. 提交作业（每次，在 Mac）

```bash
# 1)（如需）先在本机预处理数据。GRPO 用 ResponseDataset 的实验（gsm8k）需要先生成 jsonl。
#    生成到 datasets/gsm8k/ 后会随作业一起上传，run.sh 自动把 GSM8K_DATA_DIR 指向它，无需手动 export。
uv run lab prepare gsm8k

# 2)（首次）在 Spark 容器内预拉模型到 HF_HOME（submit.env 里已配 HF_HOME / HF_TOKEN）
#    ssh 进容器后：cd <仓库> && bash scripts/prefetch_hf_model.sh Qwen/Qwen3.5-4B

# 3) 提交
uv run lab submit grpo_qwen3.5-4b_gsm8k_v1
#   指定 profile：uv run lab submit <exp> --profile gb10-spark
```

`lab submit` 实际执行 `scripts/submit_job.sh`，等价于：

```bash
ray job submit \
  --address "$RAY_DASHBOARD_ADDRESS" \
  --working-dir . \                       # 上传整个仓库代码到所有节点
  --runtime-env-json '{...}' \            # 排除大文件/密钥，转发 HF/SwanLab 等环境变量
  -- bash experiments/grpo_qwen3.5-4b_gsm8k_v1/run.sh
```

**会上传什么 / 不上传什么**

- ✅ 上传：仓库代码（`common/`、`configs/`、`experiments/` 等）＋ 已准备好的小数据集 jsonl（如 `datasets/gsm8k/`）。自定义环境/奖励靠这个在所有节点被 `import`。
- ❌ 不上传：`datasets/**/raw/`、`datasets/**/data/`（原始/中间缓存）、`**/outputs/**`、`.git/**`、`__pycache__`、`cluster/submit.env`、`cluster/secrets.env`、`*.key`，以及 `.gitignore` 里命中的路径（如内部数据 `datasets/qa_rl/`）。Ray 默认也遵循 `.gitignore`。
- 🔑 环境变量：`NEMO_RL_DIR` / `CLUSTER_PROFILE` 必传；`SWANLAB_API_KEY` / `HF_TOKEN` / `HF_ENDPOINT` / `HF_HUB_ENABLE_HF_TRANSFER` / `HF_HOME`、以及 `GSM8K_DATA_DIR` / `ALPACA_DATA_DIR` / `QA_RL_DATA_DIR` 填了才转发（经 Ray `runtime_env`，不落盘）。

**数据目录怎么定位（重要）**

- **小数据集（随作业上传，推荐）**：本机 `lab prepare gsm8k` 后，`datasets/gsm8k/` 随作业上传；各实验 `run.sh` 会在未显式设置时自动把 `GSM8K_DATA_DIR` 指向上传后的目录。**无需手动 export，也无需在集群预处理**。
- **大数据 / 内部数据（留在集群，不上传）**：被 `.gitignore` 命中的（如 `datasets/qa_rl/`）不会上传。请先在集群上准备好，然后在 `cluster/submit.env` 里把对应的 `QA_RL_DATA_DIR=/容器内/绝对路径` 设好——它会被转发并**覆盖**上面的自动推导。
- **模型权重（大文件，不上传）**：在 `submit.env` 配好 `HF_HOME` + `HF_TOKEN`，**不要**设 `HF_ENDPOINT`（Spark 容器常连不上 `hf-mirror.com`）。首次在容器内 `bash scripts/prefetch_hf_model.sh Qwen/Qwen3.5-4B` 拉到缓存后再 submit；若集群能直连 `huggingface.co` 也可不 prefetch、由训练自动下载。

---

## 5. 监控 / 管理作业（在 Mac）

```bash
ADDR=http://192.168.1.4:8265

ray job list   --address $ADDR              # 所有作业
ray job logs -f <job_id> --address $ADDR    # 实时日志（-f 跟随）
ray job status <job_id>  --address $ADDR
ray job stop   <job_id>  --address $ADDR    # 停止作业
```

- **Ray Dashboard**：浏览器开 `http://192.168.1.4:8265`，看节点/资源/作业/各 actor 日志。
- **SwanLab**：训练曲线（reward / val:accuracy / GPU 利用率）在云端看，链接回填到实验 `README.md`。

---

## 6. 常见问题

| 现象 | 原因 / 排查 |
| --- | --- |
| `ConnectionError` / 连不上 8265 | Mac 到 head 网络不通。检查 IP、防火墙；不同网用 §2.2 的 SSH 隧道。 |
| 作业 `import common...` 失败 | 没走 `--working-dir`（用 `lab submit`，别手敲漏了），或在节点上 `cwd` 不对。 |
| 找不到 NeMo-RL / `run_grpo.py` | 容器里没装 NeMo-RL，或 `NEMO_RL_DIR` 指错（必须是**容器内**路径）。 |
| `OSError: couldn't connect to 'https://hf-mirror.com'` | **不要在 `submit.env` 里设 `HF_ENDPOINT`**（镜像只适合本机 `lab prepare`；Spark 容器常连不上）。在容器内先 `bash scripts/prefetch_hf_model.sh Qwen/Qwen3.5-4B` 把模型拉到 `HF_HOME`，再 submit。gated 模型要 `HF_TOKEN`。 |
| 模型/数据下载慢或失败 | 本机 prepare 可临时 `HF_ENDPOINT=https://hf-mirror.com`；集群 submit 留空 `HF_ENDPOINT`，用 `prefetch_hf_model.sh` 或集群直连 `huggingface.co`。 |
| `ray job submit` 版本协议报错 | Mac 的 `ray` 版本与集群差太多。对齐到相近版本（pyproject 已锁 `ray[default]==2.54.0`）。 |
| `Failed to merge the Job's runtime env ... conflict` | 作业的 `runtime_env` 与 NeMo-RL `init_ray` 里 `ray.init` 的 `runtime_env` 键重叠。各实验 `run.sh` 已 `export RAY_OVERRIDE_JOB_RUNTIME_ENV=1` 让 Ray 合并；若你自定义入口，记得也设这个。 |
| `KeyError ... 'GSM8K_DATA_DIR' not found` | 数据没准备/没上传。先 `lab prepare gsm8k`（生成 `datasets/gsm8k/`，随作业上传，`run.sh` 自动指向）；用集群上已有数据则在 `submit.env` 设 `GSM8K_DATA_DIR=/容器内绝对路径`。 |
| 只有 1 个节点参与 | worker 没起或没连上 head。在容器里 `uv run ray status` 看节点数；检查 `HEAD_IP`/端口。 |
| 跨机 NCCL 卡住/超时 | 网卡名/RoCE 配置不对。核对 `cluster/gb10-spark/env.sh`（`NCCL_SOCKET_IFNAME` / `NCCL_IB_HCA` 等）。 |

---

## 7. 其他方式（对比）

| 方式 | 场景 | 本仓库 |
| --- | --- | --- |
| **Ray Job Submission**（本文） | 向已存在的裸机 Ray 集群远程提交 | ✅ `lab submit`，推荐 |
| Slurm + `ray.sub` | Slurm/HPC 管理的集群 | NeMo-RL 官方 HPC 用法，不适用本环境 |
| SSH 进容器手跑 `run.sh` | 调试单条命令 | `lab run <exp> --nemo-rl ...`（容器内用） |
