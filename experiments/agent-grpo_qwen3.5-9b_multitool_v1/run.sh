#!/usr/bin/env bash
# 实验启动脚本（NeMo-RL 0.6.0）。
# 用法： NEMO_RL_DIR=/path/to/NeMo-RL CLUSTER_PROFILE=gb10-spark bash run.sh
set -euo pipefail

# 硬件 profile：gb10-spark | h200
CLUSTER_PROFILE="${CLUSTER_PROFILE:-gb10-spark}"
# 本地 NeMo-RL 0.6.0 源码目录（必填）
NEMO_RL_DIR="${NEMO_RL_DIR:?请设置 NEMO_RL_DIR 指向本地 NeMo-RL 0.6.0 源码目录}"

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${EXP_DIR}/../.." && pwd)"
CONFIG="${EXP_DIR}/config.yaml"                        # 继承基底 + 本实验差异
PROFILE_CONF="${REPO_ROOT}/cluster/${CLUSTER_PROFILE}/overrides.conf"

# 训练入口：
#   - 自定义环境（如多轮 Agent）：本目录有 run.py 则自动用它
#   - 否则按方法用官方入口（GRPO 默认 / SFT 改成 examples/run_sft.py）
if [[ -z "${ENTRY:-}" ]]; then
  if [[ -f "${EXP_DIR}/run.py" ]]; then ENTRY="${EXP_DIR}/run.py"; else ENTRY="examples/run_grpo.py"; fi
fi

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
