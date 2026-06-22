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
- 🔑 环境变量：`NEMO_RL_DIR` / `CLUSTER_PROFILE` 必传；`SWANLAB_API_KEY` / `HF_TOKEN` / `HF_ENDPOINT` / `HF_HUB_ENABLE_HF_TRANSFER` / `HF_HOME` / `OUTPUT_ROOT`、以及 `GSM8K_DATA_DIR` / `ALPACA_DATA_DIR` / `QA_RL_DATA_DIR` 填了才转发（经 Ray `runtime_env`，不落盘）。`OUTPUT_ROOT` 决定 checkpoint / 样本 jsonl 落到哪（见 §5.1）。

**数据目录怎么定位（重要）**

- **小数据集（随作业上传，推荐）**：本机 `lab prepare gsm8k` 后，`datasets/gsm8k/` 随作业上传；各实验 `run.sh` 会在未显式设置时自动把 `GSM8K_DATA_DIR` 指向上传后的目录。**无需手动 export，也无需在集群预处理**。
- **大数据 / 内部数据（留在集群，不上传）**：被 `.gitignore` 命中的（如 `datasets/qa_rl/`）不会上传。请先在集群上准备好，然后在 `cluster/submit.env` 里把对应的 `QA_RL_DATA_DIR=/容器内/绝对路径` 设好——它会被转发并**覆盖**上面的自动推导。
- **模型权重（大文件，不上传）**：在 `submit.env` 配好 `HF_HOME` + `HF_TOKEN`，**不要**设 `HF_ENDPOINT`（Spark 容器常连不上 `hf-mirror.com`）。首次在容器内 `bash scripts/prefetch_hf_model.sh Qwen/Qwen3.5-4B` 拉到缓存后再 submit；若集群能直连 `huggingface.co` 也可不 prefetch、由训练自动下载。

---

## 5. 监控 / 管理作业（在 Mac）

推荐用 `lab job`（自动从 `cluster/submit.env` 读 `RAY_DASHBOARD_ADDRESS`，无需手敲地址）：

```bash
uv run lab job list                 # 所有作业
uv run lab job logs <job_id> -f     # 实时日志（-f 跟随）
uv run lab job status <job_id>
uv run lab job stop <job_id>        # 停止作业
#   临时连别的集群：加 --address http://其它IP:8265
```

等价的原生命令（需自己带地址）：

```bash
ADDR=http://192.168.1.4:8265
ray job list   --address $ADDR
ray job logs -f <job_id> --address $ADDR
ray job status <job_id>  --address $ADDR
ray job stop   <job_id>  --address $ADDR
```

- **Ray Dashboard**：浏览器开 `http://192.168.1.4:8265`，看节点/资源/作业/各 actor 日志。
- **SwanLab**：训练曲线（reward / val:accuracy / 回答长度 / GPU 利用率）在云端看，链接回填到实验 `README.md`。

### 5.1 排查微调是否走偏 / 工具是否调对（看生成的 token）

SwanLab 只有**指标曲线**，不含模型实际输出的 token。

**先看指标（SwanLab）**：奖励是 答对=1.0/答错=0.0，所以"答对率"就是奖励均值——
看 `validation/accuracy`（验证集答对率）和 `train/reward`（训练答对率）。
走偏诊断看 `train/natural_termination_rate`（正常给 `<answer>` 收尾比例）、`train/truncation_rate`（超长截断比例）、
`train/avg_turns_per_sample`（平均轮数）。
> ⚠️ 自定义环境 `global_post_process_and_metrics` 返回的指标（如 `tool_agent_success_rate`）在当前 NeMo-RL 的 GRPO 流程里**不会被记录**，别去 SwanLab 找它，用上面的 `validation/accuracy` 即可。

**再看具体生成内容（含工具调用）**，有四种方式：

0. **本地 Web 面板**（最直观，一条命令）：
   ```bash
   lab web                 # 起本地服务并自动开浏览器（数据取自 dashboard 日志，纯本地只读）
   lab web --port 9000     # 自定端口；--no-open 不自动开浏览器
   ```
   两个视图：
   - **对比总览**（默认）：点选 2~4 个作业，一眼看「基线 → 最终」准确率与进步/退步——含结论 banner（谁最终最高、相差多少 pp）、每实验 KPI 卡（最终准确率大数字 + Δpp 徽章）、多实验**验证准确率轨迹**叠加曲线（核心指标）、训练 Reward 叠加曲线、关键指标明细表（基线/最终/ΔAcc/Δreward）。点任意 KPI 卡或表格行可下钻。
   - **作业详情**：单作业的 reward 曲线 + 验证准确率、每次验证的完整对话（`<think>`/`<tool_call>`/`\boxed{}` 高亮、reward 彩色徽章、含完整 model response），按页「加载更多」。
   默认 15s 自动刷新。对话条数由 `logger.num_val_samples_to_print` 决定（qa 实验已调到 16）。服务用 FastAPI + uvicorn（`uv` 的 `web` extra，`lab web` 自动启用），API 文档在 `/api/docs`。无需数据库、无需在集群起服务。
1. **本地抽取验证轨迹**（命令行，无需登集群）：
   ```bash
   lab job samples <job_id>            # 全部验证的样本面板
   lab job samples <job_id> --last 1   # 只看最近一次验证
   ```
   它走 dashboard HTTP 把作业日志拉到本机，只保留 prompt+生成（多轮 Agent 含每轮 tool call、工具返回、reward）与结果摘要。条数由 config 的 `logger.num_val_samples_to_print`（默认 3）决定，想看更多就调大重提交。
2. **整段作业日志**：`lab job logs <job_id> -f`（实时跟随），样本面板也在其中，但夹杂大量进度条。
3. **每步落盘的 jsonl**（信息最全，但只在集群）：训练每步写 `train_data_step{N}.jsonl`，每次验证写 `val_data_step{N}.jsonl`，到 `OUTPUT_ROOT/<实验名>/logs/`；`content` 是完整生成文本，另含 `rewards` / `advantages` / `token_ids`。该文件**不会上传 SwanLab**，要原始 jsonl 才需进容器看。
   - ⚠️ 必须在 `submit.env` 设 `OUTPUT_ROOT`（持久路径/共享盘），否则产物落在 Ray 临时目录、训练结束被清理。

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
