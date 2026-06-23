# cluster/ — 硬件 / 分布式 profile

NeMo-RL 0.6.0 的集群设置（`cluster.num_nodes`、`cluster.gpus_per_node`）和并行度都在训练配置里，通过 **CLI override** 调整。本目录把「不同硬件的 override」抽出来复用，训练配置与硬件解耦。

每个 profile 一个子目录，核心是 `overrides.conf`（每行一个 `key=value`，`#` 为注释）：

- `h100/` — 单机 1× H100 80GB（单节点单卡，本机直跑）
- `gb10-spark/` — 2× DGX Spark GB10（Ray 2 节点）
- `b300/` — B300（后续使用）

## 用法

每个实验**自带目标集群**（实验目录下一行 `cluster` 文件，记录 profile 名）。`run.sh` 默认读它选 profile，自动把对应 `overrides.conf` 追加到训练命令；`CLUSTER_PROFILE` 环境变量 / `lab submit --profile` 可临时覆盖：

```bash
bash run.sh                              # 用实验自带 cluster（软绑定）
CLUSTER_PROFILE=h100 bash run.sh         # 临时换单机单卡
```

> **profile 优先级**：`--profile`（显式）> 实验自带 `cluster` 文件 > `submit.env` 的 `DEFAULT_CLUSTER_PROFILE` > `gb10-spark` 兜底。
> 新建实验时用 `lab new <名字> --cluster h100` 写好绑定；`lab new <名字> --from <实验>` 会继承来源实验的绑定。

### submit.env 分两层（地址跟集群走，密钥填一次）

`lab submit` 的配置拆成两层，**换集群不必改同一个文件、也不会互相覆盖**：


| 层   | 文件                             | 放什么                                                                  | 入库              |
| --- | ------------------------------ | -------------------------------------------------------------------- | --------------- |
| 通用层 | `cluster/submit.env`           | 密钥、`RUN_USER`、`DEFAULT_CLUSTER_PROFILE`（填一次，换集群不动）                   | 否（`.gitignore`） |
| 集群层 | `cluster/<profile>/submit.env` | `RAY_DASHBOARD_ADDRESS`、`NEMO_RL_DIR`、`OUTPUT_ROOT`、数据/裁判/检索地址（随集群走） | 否（`.gitignore`） |


加载顺序：先通用层、再集群层（集群层覆盖）。各配一份模板：

```bash
cp cluster/submit.env.example          cluster/submit.env          # 通用层：密钥 + RUN_USER
cp cluster/h100/submit.env.example     cluster/h100/submit.env     # 集群层：改 IP / 路径
# 切到另一台集群只需再 cp 它的集群层模板，互不影响：
cp cluster/gb10-spark/submit.env.example cluster/gb10-spark/submit.env
```

> 多人共用同一集群时在通用层设 `RUN_USER`，产物隔离到 `OUTPUT_ROOT/<RUN_USER>/<实验名>`，互不覆盖。

等价于：

```bash
uv run python examples/run_grpo.py --config <base.yaml> \
    cluster.num_nodes=2 cluster.gpus_per_node=1 ...   # 来自 profile
```

## 各 profile 包含

- `overrides.conf` — 节点数 / 每节点 GPU / 并行度等 NeMo-RL 覆盖项（CLI override）
- `env.sh` — 集群 env（NCCL/RoCE 网络 + Ray 内存监控 + PyTorch 显存分配），被 ray 启动脚本和实验 `run.sh` 统一 source，一处改处处生效
- `submit.env.example` — 该集群的提交配置模板（地址 / 容器路径）；复制为 `submit.env`（已 .gitignore）
- `start_ray_head.sh` / `start_ray_worker.sh` — 启动 Ray 集群（多节点需要）
- `hosts.example` — 复制为 `hosts` 填真实 IP（多节点 profile 有；`hosts` 已 .gitignore）

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
> 见 `[docs/remote-submit.md](../docs/remote-submit.md)`。

## 依赖与环境

本仓库以 **NVIDIA NeMo-RL 0.6.0** 为训练框架；NeMo-RL 用 **`uv`** 管理依赖与运行（`uv run python ...`）。依赖按**架构**分别维护，强烈建议用 **NeMo-RL 官方容器镜像**跑训练，避免手装 CUDA / wheel 踩坑。

| 硬件 profile | 架构 | 安装方式 |
| --- | --- | --- |
| `gb10-spark` | aarch64 + Blackwell | aarch64 容器 / wheel |
| `h100` / `b300` | x86_64 + Hopper / Blackwell | x86_64 容器 / wheel |

### 集群容器内：NeMo-RL 0.6.0

训练框架**预装在容器镜像里**，不随 `lab submit` 上传。容器内路径填到该集群的 `cluster/<profile>/submit.env` → `NEMO_RL_DIR`（必须是**容器内**绝对路径）。

若需自行克隆 / 升级 NeMo-RL（容器内执行）：

```bash
git clone --branch v0.6.0 https://github.com/NVIDIA-NeMo/RL.git NeMo-RL
cd NeMo-RL
uv sync
uv run python examples/run_grpo.py --help   # 验证
```

> 具体容器 tag、各 backend（DTensor / Megatron）的额外依赖以 v0.6.0 官方文档为准。
> 源码与容器 fingerprint 不一致时，见 NeMo-RL 文档的 `NRL_FORCE_REBUILD_VENVS` / 重建镜像说明。

### 本机（Mac）：开发期依赖

本仓库根目录 **`pyproject.toml` + `uv`** 只管理**提交客户端**与 lab CLI（`typer`、`datasets`、可选 `ray[default]`），不含 NeMo-RL / vLLM / CUDA：

```bash
uv sync --extra submit    # 装 Ray job submission 客户端 + lab CLI
uv run lab ls
```

三种调用方式见根目录 README「统一 CLI」一节。

### 密钥与路径

`SWANLAB_API_KEY`、`HF_TOKEN`、`RAY_DASHBOARD_ADDRESS`、`NEMO_RL_DIR` 等放 **`cluster/submit.env`**（通用层 + 各 profile 集群层，均已 `.gitignore`），不要入库。详见上文「submit.env 分两层」与 [`docs/remote-submit.md`](../docs/remote-submit.md) §2.1。

#### 密钥转发的两种模式（多人共用集群务必看）

`lab submit` 让远程作业拿到密钥（`SWANLAB_API_KEY` / `HF_TOKEN` / `JUDGE_API_KEY` / `KB_API_KEY`）有两种方式：

| 模式 | 怎么配 | 密钥是否暴露 | 适用 |
| --- | --- | --- | --- |
| **明文转发（默认/兜底）** | 在 `submit.env` 填密钥 | ⚠️ 会写进 Ray runtime_env，出现在 **Dashboard 的 job 详情**，同集群他人可见 | 单人 / 私有集群 |
| **集群侧 secrets 文件（推荐）** | 集群层 `submit.env` 设 `CLUSTER_SECRETS_FILE=<容器内路径>`，把密钥放到该容器内文件（`chmod 600`） | ✅ 只转发**路径**，密钥不进 Dashboard；作业在集群侧 `source` | 多人共用集群 |

> 配了 `CLUSTER_SECRETS_FILE` 后，通用层 `submit.env` 里的 `SWANLAB_API_KEY` / `HF_TOKEN` 可留空。
> `submit` 检测到将明文转发密钥时会打印安全提示。

#### 提交可追溯（git / config 指纹）

`lab submit` 每次会自动：

- 注入 `NRL_GIT_COMMIT` / `NRL_GIT_DIRTY` / `NRL_CONFIG_SHA` / `NRL_RUN_ID` 到作业（落在 `ray job logs` 的 `[run] version: ...` 行）；
- 在本地 `.lab/runs.jsonl`（已 `.gitignore`）追加一行台账：run_id / 时间 / action / 用户 / 实验 / profile / commit / dirty / config 指纹 / 地址；
- 把 `run_id` 写进 Ray 作业 **metadata**（`lab_run_id`），用于把本地台账与集群作业状态对上（不依赖随机 submission_id）；
- 工作目录 **dirty**（有未提交改动）时给出警告——上传的是当前磁盘内容，但 commit 无法完整复现本次运行，建议先提交再 submit。

`lab export` / `lab eval` 也写同一份台账（`action=export|eval`）。随时 `uv run lab runs`（`--all` / `--exp <名>`）查看历史，
事后看到一个 SwanLab / 作业，用日志里的 `git=<commit>` 即可回到对应代码版本。

#### 关联作业状态 / 集群预检

- `lab runs --status`：连 dashboard（`/api/jobs/`，纯 HTTP 无需 ray），按 `run_id` 把台账对上集群作业状态（RUNNING/SUCCEEDED/FAILED…），一屏看「这次提交跑成没」；连不上时自动降级为纯本地台账。
- `lab status`：提交前预检——整集群空闲 GPU（解析 `/api/cluster_status`）+ 自己的活跃作业（RUNNING/PENDING），避免撞满卡。`--all` 看全部状态作业。

