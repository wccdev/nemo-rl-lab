#!/usr/bin/env bash
# 多轮 Agent GRPO·滑块拼图。入口：NeMo-RL 自带 examples/run_grpo_sliding_puzzle.py（自带 env + 数据，无需 run.py）。
# 用法： NEMO_RL_DIR=/path/to/NeMo-RL CLUSTER_PROFILE=h100 bash run.sh
# 通用逻辑（profile / override / 产物 / 数据 / 密钥）在 scripts/_run_experiment.sh，单一事实来源。
set -euo pipefail
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ENTRY="${ENTRY:-examples/run_grpo_sliding_puzzle.py}"   # 本实验差异：自带多轮拼图示例入口
exec bash "${EXP_DIR}/../../scripts/_run_experiment.sh" "${EXP_DIR}"
