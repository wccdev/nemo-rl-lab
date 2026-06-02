#!/usr/bin/env bash
# 实验启动脚本。用法： CLUSTER_PROFILE=gb10-spark bash run.sh
set -euo pipefail

# 仓库根目录（templates/experiment-template -> 上两级）
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${EXP_DIR}/../.." && pwd)"

CLUSTER_PROFILE="${CLUSTER_PROFILE:-gb10-spark}"
PROFILE_FILE="${REPO_ROOT}/cluster/${CLUSTER_PROFILE}/profile.yaml"
CONFIG_FILE="${EXP_DIR}/configs/train.yaml"

echo "[run] 实验目录   : ${EXP_DIR}"
echo "[run] 硬件 profile: ${CLUSTER_PROFILE} (${PROFILE_FILE})"
echo "[run] 训练配置   : ${CONFIG_FILE}"

# TODO: 在此调用 NeMo-RL 训练入口，把 CONFIG_FILE 与 PROFILE_FILE 合并传入。
# 例如（命令以所用 NeMo-RL 版本为准）：
#   python -m nemo_rl.train --config "${CONFIG_FILE}" --cluster "${PROFILE_FILE}"
echo "[run] 请在 run.sh 中填入 NeMo-RL 训练命令。"
