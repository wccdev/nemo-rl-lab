#!/usr/bin/env bash
# 实验启动·通用逻辑（NeMo-RL 0.6.0）——所有实验的 run.sh 都把通用部分收口到这里，
# 单一事实来源：改一次，所有实验生效。各实验 run.sh 只声明自己的差异（主要是 ENTRY），
# 然后 `exec bash scripts/_run_experiment.sh "${EXP_DIR}"`。
#
# 入参：$1 = 实验目录绝对路径（EXP_DIR）。
# 约定的可选环境变量（由各实验 run.sh / lab submit / submit.env 注入）：
#   ENTRY            训练入口（不设则：本目录有 run.py 用之，否则 examples/run_grpo.py）
#   NEMO_RL_DIR      容器/本机 NeMo-RL 0.6.0 源码目录（必填）
#   CLUSTER_PROFILE  硬件 profile（不设则读实验自带 cluster 文件，再兜底 gb10-spark）
#   OUTPUT_ROOT      产物根目录（不设则落到 EXP_DIR/outputs）；RUN_USER 再做多人隔离
set -euo pipefail

EXP_DIR="${1:?用法: _run_experiment.sh <实验目录绝对路径>（由各实验 run.sh 传入）}"
[[ -d "${EXP_DIR}" ]] || { echo "实验目录不存在: ${EXP_DIR}"; exit 1; }
REPO_ROOT="$(cd "${EXP_DIR}/../.." && pwd)"
EXP_NAME="$(basename "${EXP_DIR}")"

# 本地 NeMo-RL 0.6.0 源码目录（必填）
NEMO_RL_DIR="${NEMO_RL_DIR:?请设置 NEMO_RL_DIR 指向 NeMo-RL 0.6.0 源码目录}"

# 硬件 profile：默认读本实验绑定的集群（同目录 cluster 文件，可选 cluster/ 下 h100 | gb10-spark | b300）。
# 本实验超参（batch/seq/LoRA/并行度/显存）都是按该集群的卡调出来的，换卡通常要重调。
# 优先级：环境 CLUSTER_PROFILE（lab submit 注入 / --profile）> 自带 cluster 文件 > gb10-spark 兜底。
if [[ -z "${CLUSTER_PROFILE:-}" && -f "${EXP_DIR}/cluster" ]]; then
  CLUSTER_PROFILE="$(tr -d '[:space:]' < "${EXP_DIR}/cluster")"
fi
CLUSTER_PROFILE="${CLUSTER_PROFILE:-gb10-spark}"
CONFIG="${EXP_DIR}/config.yaml"                        # 继承基底 + 本实验差异
PROFILE_CONF="${REPO_ROOT}/cluster/${CLUSTER_PROFILE}/overrides.conf"
PROFILE_ENV="${REPO_ROOT}/cluster/${CLUSTER_PROFILE}/env.sh"

# 训练入口：实验 run.sh 显式 export ENTRY（SFT / 自定义示例）优先；否则本目录有 run.py 用它，
# 再否则用 GRPO 官方入口。
if [[ -z "${ENTRY:-}" ]]; then
  if [[ -f "${EXP_DIR}/run.py" ]]; then ENTRY="${EXP_DIR}/run.py"; else ENTRY="examples/run_grpo.py"; fi
fi

read_conf() { [[ -f "$1" ]] && grep -vE '^[[:space:]]*(#|$)' "$1" || true; }

# 集群/硬件 override（CLI，运行时按 profile 叠加）+ 产物落盘
OVERRIDES=()
while IFS= read -r l; do [[ -n "$l" ]] && OVERRIDES+=("$l"); done < <(read_conf "${PROFILE_CONF}")
# 产物（checkpoint + 每步样本 jsonl + 日志）落盘位置。
# 远程 lab submit 时 EXP_DIR 在 Ray 上传的临时包目录里（训练结束被清理、不回传 Mac），
# 故设 OUTPUT_ROOT（建议在 submit.env 配成集群持久路径/共享盘）后产物落到 OUTPUT_ROOT[/<用户>]/<实验名>。
# 多人共用平台时设 RUN_USER（如名字/工号），产物隔离到 OUTPUT_ROOT/<用户>/<实验名>，互不覆盖。
if [[ -n "${OUTPUT_ROOT:-}" ]]; then OUT_DIR="${OUTPUT_ROOT%/}${RUN_USER:+/${RUN_USER}}/${EXP_NAME}"; else OUT_DIR="${EXP_DIR}/outputs"; fi
OVERRIDES+=("checkpointing.checkpoint_dir=${OUT_DIR}")
OVERRIDES+=("logger.log_dir=${OUT_DIR}/logs")

echo "[run] exp     : ${EXP_NAME}"
echo "[run] profile : ${CLUSTER_PROFILE}"
echo "[run] entry   : ${ENTRY}"
echo "[run] config  : ${CONFIG}"
echo "[run] cluster/产物 overrides:"; printf '          %s\n' "${OVERRIDES[@]}"

# 本地密钥/HF 配置（容器内直跑用；lab submit 路径由 runtime_env 注入，不依赖此文件）
SECRETS_ENV="${REPO_ROOT}/cluster/secrets.env"
[[ -f "${SECRETS_ENV}" ]] && { set -a; source "${SECRETS_ENV}"; set +a; }

# 硬件/网络 env（NCCL、Ray 内存、PyTorch 分配）；多节点须与 ray start 用同一份
[[ -f "${PROFILE_ENV}" ]] && source "${PROFILE_ENV}"

# 数据目录：未显式设置 *_DATA_DIR 时，默认指向本仓库 datasets/<name>。
# lab submit 时该目录随作业上传（submit_job.sh 仅排除 raw/data 缓存），
# 故 config 里的 ${oc.env:GSM8K_DATA_DIR} 等无需手填即可解析；
# 想用集群上已有的大数据，则在 submit.env / secrets.env 显式设置同名变量覆盖。
for _ds in gsm8k:GSM8K_DATA_DIR alpaca:ALPACA_DATA_DIR qa_rl:QA_RL_DATA_DIR; do
  _name="${_ds%%:*}"; _var="${_ds##*:}"
  if [[ -z "${!_var:-}" && -d "${REPO_ROOT}/datasets/${_name}" ]]; then
    export "${_var}=${REPO_ROOT}/datasets/${_name}"
    echo "[run] ${_var}=${REPO_ROOT}/datasets/${_name} (默认指向仓库内数据)"
  fi
done

# 经 ray job submit 时，作业自带 runtime_env（working_dir + 转发的 env_vars）；NeMo-RL 的
# init_ray 还会再 ray.init(runtime_env=...) 传一份 env_vars，键重叠会被 Ray 判为冲突报错。
# 置 1 让 Ray 合并 Job 与 Driver 的 runtime_env（冲突以 Driver 为准，值相同无副作用；直跑无害）。
export RAY_OVERRIDE_JOB_RUNTIME_ENV=1

cd "${NEMO_RL_DIR}"
exec uv run python "${ENTRY}" --config "${CONFIG}" "${OVERRIDES[@]}"
