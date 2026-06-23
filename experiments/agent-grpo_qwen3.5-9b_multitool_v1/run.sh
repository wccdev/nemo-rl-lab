#!/usr/bin/env bash
# 多轮 Agent GRPO·多工具调用。入口：本目录 run.py（存在则自动优先，含自定义多工具环境）。
# 用法： NEMO_RL_DIR=/path/to/NeMo-RL CLUSTER_PROFILE=gb10-spark bash run.sh
# 通用逻辑（profile / override / 产物 / 数据 / 密钥）在 scripts/_run_experiment.sh，单一事实来源。
set -euo pipefail
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${EXP_DIR}/../../scripts/_run_experiment.sh" "${EXP_DIR}"
