#!/usr/bin/env bash
# 实验启动脚本（NeMo-RL 0.6.0）。
# 用法： NEMO_RL_DIR=/path/to/NeMo-RL CLUSTER_PROFILE=gb10-spark bash run.sh
set -euo pipefail

# ===================== 按实验修改 =====================
# 训练入口（相对 NeMo-RL 源码目录）：examples/run_grpo.py | examples/run_sft.py
ENTRY="${ENTRY:-examples/run_grpo.py}"
# =====================================================

# 硬件 profile：gb10-spark | h200
CLUSTER_PROFILE="${CLUSTER_PROFILE:-gb10-spark}"
# 本地 NeMo-RL 0.6.0 源码目录（必填）
NEMO_RL_DIR="${NEMO_RL_DIR:?请设置 NEMO_RL_DIR 指向本地 NeMo-RL 0.6.0 源码目录}"

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${EXP_DIR}/../.." && pwd)"
CONFIG="${EXP_DIR}/config.yaml"                        # 继承基底 + 本实验差异
PROFILE_CONF="${REPO_ROOT}/cluster/${CLUSTER_PROFILE}/overrides.conf"

read_conf() { [[ -f "$1" ]] && grep -vE '^[[:space:]]*(#|$)' "$1" || true; }

# 集群/硬件 override（CLI，运行时按 profile 叠加）+ 产物落到实验目录
OVERRIDES=()
while IFS= read -r l; do [[ -n "$l" ]] && OVERRIDES+=("$l"); done < <(read_conf "${PROFILE_CONF}")
OVERRIDES+=("checkpointing.checkpoint_dir=${EXP_DIR}/outputs")
OVERRIDES+=("logger.log_dir=${EXP_DIR}/outputs/logs")

echo "[run] profile : ${CLUSTER_PROFILE}"
echo "[run] entry   : ${ENTRY}"
echo "[run] config  : ${CONFIG}"
echo "[run] cluster/产物 overrides:"; printf '          %s\n' "${OVERRIDES[@]}"

cd "${NEMO_RL_DIR}"
exec uv run python "${ENTRY}" --config "${CONFIG}" "${OVERRIDES[@]}"
