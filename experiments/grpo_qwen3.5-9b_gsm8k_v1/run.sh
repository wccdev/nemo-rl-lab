#!/usr/bin/env bash
# GRPO·GSM8K 数学推理（9B + LoRA）。入口：官方 examples/run_grpo.py（本目录无 run.py，走兜底）。
# 用法： NEMO_RL_DIR=/path/to/NeMo-RL CLUSTER_PROFILE=gb10-spark bash run.sh
# 通用逻辑（profile / override / 产物 / 数据 / 密钥）在 scripts/_run_experiment.sh，单一事实来源。
set -euo pipefail
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${EXP_DIR}/../../scripts/_run_experiment.sh" "${EXP_DIR}"
