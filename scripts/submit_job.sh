#!/usr/bin/env bash
# 从开发机（Mac）把训练作业提交到远程 Ray 集群；实际执行在集群容器内。
# 本仓库代码随 --working-dir 自动上传并分发到所有节点（含 worker，自定义环境靠这个被 import）；
# NeMo-RL 框架须已装在容器里（不随作业上传）。
#
# 准备：
#   pip install "ray[default]"                       # 开发机装 Ray CLI（无需 GPU）
#   cp cluster/submit.env.example cluster/submit.env  # 填好地址 / 路径 / 密钥
# 用法：
#   bash scripts/submit_job.sh experiments/agent-grpo_qwen3.5-9b_multitool_v1 [gb10-spark]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ENV_FILE="${SUBMIT_ENV:-${REPO_ROOT}/cluster/submit.env}"
if [[ -f "${ENV_FILE}" ]]; then
  set -a; source "${ENV_FILE}"; set +a
else
  echo "缺少 ${ENV_FILE}"
  echo "请先： cp cluster/submit.env.example cluster/submit.env 并填写地址/路径。"
  exit 1
fi

EXP_REL="${1:?用法: submit_job.sh <实验相对路径> [profile]，如 experiments/grpo_qwen3.5-9b_gsm8k_v1}"
EXP_REL="${EXP_REL%/}"
export CLUSTER_PROFILE="${2:-${DEFAULT_CLUSTER_PROFILE:-gb10-spark}}"

: "${RAY_DASHBOARD_ADDRESS:?请在 cluster/submit.env 设置 RAY_DASHBOARD_ADDRESS（如 http://192.168.1.10:8265）}"
: "${NEMO_RL_DIR:?请在 cluster/submit.env 设置 NEMO_RL_DIR（容器内 NeMo-RL 路径）}"

[[ -d "${REPO_ROOT}/${EXP_REL}" ]] || { echo "找不到实验目录: ${EXP_REL}"; exit 1; }
[[ -f "${REPO_ROOT}/${EXP_REL}/run.sh" ]] || { echo "实验缺少 run.sh: ${EXP_REL}"; exit 1; }

# 组装 runtime_env：排除大文件，传必要环境变量给作业进程
RUNTIME_ENV="$(python3 - <<'PY'
import json, os
env_vars = {
    "NEMO_RL_DIR": os.environ["NEMO_RL_DIR"],
    "CLUSTER_PROFILE": os.environ["CLUSTER_PROFILE"],
}
key = os.environ.get("SWANLAB_API_KEY")
if key:
    env_vars["SWANLAB_API_KEY"] = key
print(json.dumps({
    "excludes": ["datasets/**", "**/outputs/**", ".git/**", "**/__pycache__/**"],
    "env_vars": env_vars,
}))
PY
)"

echo "[submit] 集群        : ${RAY_DASHBOARD_ADDRESS}"
echo "[submit] 实验        : ${EXP_REL}  (profile=${CLUSTER_PROFILE})"
echo "[submit] NEMO_RL_DIR : ${NEMO_RL_DIR} (容器内)"

cd "${REPO_ROOT}"
exec ray job submit \
  --address "${RAY_DASHBOARD_ADDRESS}" \
  --working-dir . \
  --runtime-env-json "${RUNTIME_ENV}" \
  -- bash "${EXP_REL}/run.sh"
