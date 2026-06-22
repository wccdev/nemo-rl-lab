#!/usr/bin/env bash
# 从开发机（Mac）把训练作业提交到远程 Ray 集群；实际执行在集群容器内。
# 本仓库代码随 --working-dir 自动上传并分发到所有节点（含 worker，自定义环境靠这个被 import）；
# NeMo-RL 框架须已装在容器里（不随作业上传）。
#
# 准备（两层配置，各填一次）：
#   uv sync --extra submit                                            # 开发机装 Ray CLI（lab submit 也会自动按需装）
#   cp cluster/submit.env.example cluster/submit.env                  # 通用层：密钥 / RUN_USER / 默认 profile（换集群不动）
#   cp cluster/<profile>/submit.env.example cluster/<profile>/submit.env  # 集群专属：地址 / 容器路径（随 profile 走）
# 用法：
#   bash scripts/submit_job.sh experiments/agent-grpo_qwen3.5-9b_multitool_v1 [gb10-spark]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 分层加载：先通用层（密钥/默认 profile），再集群专属层（地址/路径，覆盖通用层）。
# 换集群只需改 profile（参数 / DEFAULT_CLUSTER_PROFILE），不必回头编辑同一个文件、也不会互相覆盖。
SHARED_ENV="${SUBMIT_ENV:-${REPO_ROOT}/cluster/submit.env}"
[[ -f "${SHARED_ENV}" ]] && { set -a; source "${SHARED_ENV}"; set +a; }

EXP_REL="${1:?用法: submit_job.sh <实验相对路径> [profile]，如 experiments/grpo_qwen3.5-9b_gsm8k_v1}"
EXP_REL="${EXP_REL%/}"
# 实验自带的默认集群（软绑定）：experiments/<name>/cluster 一行 profile 名。
# 这些超参都是按该集群的卡调出来的；换卡用 --profile（下方 $2）临时覆盖。
EXP_PROFILE=""
[[ -f "${REPO_ROOT}/${EXP_REL}/cluster" ]] && EXP_PROFILE="$(tr -d '[:space:]' < "${REPO_ROOT}/${EXP_REL}/cluster")"
# 选集群：--profile($2) > 实验绑定 > 环境 CLUSTER_PROFILE > 通用层 DEFAULT_CLUSTER_PROFILE > gb10-spark
export CLUSTER_PROFILE="${2:-${EXP_PROFILE:-${CLUSTER_PROFILE:-${DEFAULT_CLUSTER_PROFILE:-gb10-spark}}}}"
PROFILE_ENV="${REPO_ROOT}/cluster/${CLUSTER_PROFILE}/submit.env"
[[ -f "${PROFILE_ENV}" ]] && { set -a; source "${PROFILE_ENV}"; set +a; }

if [[ ! -f "${SHARED_ENV}" && ! -f "${PROFILE_ENV}" ]]; then
  echo "缺少提交配置：${SHARED_ENV} 与 ${PROFILE_ENV} 都不存在。"
  echo "请先：cp cluster/submit.env.example cluster/submit.env                      # 通用层：密钥/RUN_USER/默认 profile"
  echo "      cp cluster/${CLUSTER_PROFILE}/submit.env.example cluster/${CLUSTER_PROFILE}/submit.env   # 集群专属：地址/路径"
  exit 1
fi

: "${RAY_DASHBOARD_ADDRESS:?请在 cluster/${CLUSTER_PROFILE}/submit.env 设置 RAY_DASHBOARD_ADDRESS（如 http://192.168.1.10:8265）}"
: "${NEMO_RL_DIR:?请在 cluster/${CLUSTER_PROFILE}/submit.env 设置 NEMO_RL_DIR（容器内 NeMo-RL 路径）}"

[[ -d "${REPO_ROOT}/${EXP_REL}" ]] || { echo "找不到实验目录: ${EXP_REL}"; exit 1; }
[[ -f "${REPO_ROOT}/${EXP_REL}/run.sh" ]] || { echo "实验缺少 run.sh: ${EXP_REL}"; exit 1; }

# 组装 runtime_env：排除大文件/密钥，转发必要环境变量给作业进程
RUNTIME_ENV="$(python3 - <<'PY'
import json, os
env_vars = {
    "NEMO_RL_DIR": os.environ["NEMO_RL_DIR"],
    "CLUSTER_PROFILE": os.environ["CLUSTER_PROFILE"],
}
# 可选转发：密钥 + HF 下载配置 + 数据目录覆盖（在 submit.env 里设了才转发）。
# *_DATA_DIR 只在你想用「集群上已有的大数据」时才设（值是容器内路径）；
# 不设时各实验 run.sh 会自动指向随作业上传的 datasets/<name>。
for k in ("SWANLAB_API_KEY", "HF_TOKEN", "HF_ENDPOINT", "HF_HUB_ENABLE_HF_TRANSFER", "HF_HOME",
          "GSM8K_DATA_DIR", "ALPACA_DATA_DIR", "QA_RL_DATA_DIR", "OUTPUT_ROOT",
          # 多人共用平台：产物隔离到 OUTPUT_ROOT/<RUN_USER>/<实验名>（见各实验 run.sh）。
          "RUN_USER",
          # UV_NO_SYNC=1：让集群 run.sh 的 `uv run` 跳过 sync、直接用已装好的 venv，
          # 避开 GitHub 直链依赖(flash-attn)偶发 504 拖垮整个作业（venv 已建好时强烈建议设 1）。
          "UV_NO_SYNC",
          # NeMo-RL 环境一致性开关：容器代码/依赖与源码漂移时用。
          #   NRL_FORCE_REBUILD_VENVS=true 强制 worker 按当前源码重建隔离 venv（修「configure_worker
          #     返回值个数不符」之类的 driver 新/worker 旧不一致；首跑较慢，建好后删掉此项）。
          #   NRL_IGNORE_VERSION_MISMATCH=1 仅压制告警、不修复（不推荐长期开）。
          "NRL_FORCE_REBUILD_VENVS", "NRL_IGNORE_VERSION_MISMATCH",
          # 简答题裁判 LLM（qa-rl / qa-rl-agent）；外部知识库检索（qa-rl-agent）。
          "JUDGE_BASE_URL", "JUDGE_MODEL", "JUDGE_API_KEY", "JUDGE_CONCURRENCY", "JUDGE_TIMEOUT",
          "KB_BASE_URL", "KB_API_KEY", "KB_DATASET_IDS", "KB_TOP_K", "KB_TIMEOUT",
          "KB_SIMILARITY_THRESHOLD", "KB_MAX_CHARS"):
    v = os.environ.get(k)
    if v:
        env_vars[k] = v
print(json.dumps({
    # 上传整个仓库（含已准备好的小 jsonl，自定义环境/数据随作业分发到各节点）；
    # 仅排除：原始/中间缓存、产物、密钥、git/pycache。
    "excludes": [
        "datasets/**/raw/**", "datasets/**/data/**",
        "**/outputs/**", ".git/**", "**/__pycache__/**",
        "cluster/submit.env", "cluster/*/submit.env", "cluster/secrets.env", "**/*.key",
    ],
    "env_vars": env_vars,
}))
PY
)"

echo "[submit] 集群        : ${RAY_DASHBOARD_ADDRESS}"
echo "[submit] 实验        : ${EXP_REL}  (profile=${CLUSTER_PROFILE})"
echo "[submit] NEMO_RL_DIR : ${NEMO_RL_DIR} (容器内)"
if [[ -n "${HF_ENDPOINT:-}" ]]; then
  echo "[submit] 警告: 已设置 HF_ENDPOINT=${HF_ENDPOINT}"
  echo "         集群容器常连不上国内镜像；若训练报 OSError 连不上 mirror，请注释 submit.env 里的 HF_ENDPOINT，"
  echo "         并在容器内先运行: bash scripts/prefetch_hf_model.sh <模型名>"
fi
if [[ -n "${HF_HOME:-}" ]]; then
  echo "[submit] HF_HOME     : ${HF_HOME} (容器内，模型须已缓存或能访问 huggingface.co)"
fi

cd "${REPO_ROOT}"
# 用 uv 管理的 Ray CLI（pyproject 可选依赖 submit；版本对齐集群）。
# uv 会按需把 submit extra 装好，无需手动 pip install ray。
exec uv run --extra submit ray job submit \
  --address "${RAY_DASHBOARD_ADDRESS}" \
  --working-dir . \
  --runtime-env-json "${RUNTIME_ENV}" \
  -- bash "${EXP_REL}/run.sh"
