#!/usr/bin/env bash
# SFT·Alpaca 指令监督微调（4B）。入口：官方 examples/run_sft.py（无 run.py，须显式声明）。
# 用法： NEMO_RL_DIR=/path/to/NeMo-RL CLUSTER_PROFILE=gb10-spark bash run.sh
# 通用逻辑（profile / override / 产物 / 数据 / 密钥）在 scripts/_run_experiment.sh，单一事实来源。
set -euo pipefail
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ENTRY="${ENTRY:-examples/run_sft.py}"   # 本实验差异：SFT 入口
exec bash "${EXP_DIR}/../../scripts/_run_experiment.sh" "${EXP_DIR}"
