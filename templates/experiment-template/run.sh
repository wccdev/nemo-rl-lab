#!/usr/bin/env bash
# 实验启动脚本（NeMo-RL 0.6.0）。
# 用法： NEMO_RL_DIR=/path/to/NeMo-RL CLUSTER_PROFILE=gb10-spark bash run.sh
set -euo pipefail

# ===================== 按实验修改 =====================
# 训练入口（相对 NeMo-RL 源码目录）：run_sft.py / run_grpo.py
ENTRY="${ENTRY:-examples/run_grpo.py}"
# 基础配置（官方 v0.6.0 example，相对 NeMo-RL 源码目录）
BASE_CONFIG="${BASE_CONFIG:-examples/configs/grpo_math_1B.yaml}"
# =====================================================

# 硬件 profile：gb10-spark | h200
CLUSTER_PROFILE="${CLUSTER_PROFILE:-gb10-spark}"
# 本地 NeMo-RL 0.6.0 源码目录（必填）
NEMO_RL_DIR="${NEMO_RL_DIR:?请设置 NEMO_RL_DIR 指向本地 NeMo-RL 0.6.0 源码目录}"

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${EXP_DIR}/../.." && pwd)"
PROFILE_CONF="${REPO_ROOT}/cluster/${CLUSTER_PROFILE}/overrides.conf"
EXP_CONF="${EXP_DIR}/overrides.conf"

read_conf() { [[ -f "$1" ]] && grep -vE '^[[:space:]]*(#|$)' "$1" || true; }

OVERRIDES=()
# 1) 硬件 profile 覆盖
while IFS= read -r l; do [[ -n "$l" ]] && OVERRIDES+=("$l"); done < <(read_conf "${PROFILE_CONF}")
# 2) 自动把产物落到实验目录（可被下方 overrides.conf 覆盖）
OVERRIDES+=("checkpointing.checkpoint_dir=${EXP_DIR}/outputs")
OVERRIDES+=("logger.log_dir=${EXP_DIR}/outputs/logs")
# 3) 实验自身覆盖（模型/数据/超参/SwanLab）
while IFS= read -r l; do [[ -n "$l" ]] && OVERRIDES+=("$l"); done < <(read_conf "${EXP_CONF}")

echo "[run] profile : ${CLUSTER_PROFILE}"
echo "[run] entry   : ${ENTRY}"
echo "[run] base    : ${BASE_CONFIG}"
echo "[run] overrides:"; printf '          %s\n' "${OVERRIDES[@]}"

cd "${NEMO_RL_DIR}"
exec uv run python "${ENTRY}" --config "${BASE_CONFIG}" "${OVERRIDES[@]}"
