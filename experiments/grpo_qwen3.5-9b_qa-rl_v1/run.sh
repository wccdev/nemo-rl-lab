#!/usr/bin/env bash
# GRPO·QA 题库（自定义判分）。入口：本目录 run.py（存在则自动优先，含规则判分 + 简答 LLM 裁判）。
# 用法： NEMO_RL_DIR=/path/to/NeMo-RL CLUSTER_PROFILE=gb10-spark bash run.sh
# 通用逻辑（profile / override / 产物 / 数据 / 密钥）在 scripts/_run_experiment.sh，单一事实来源。
set -euo pipefail
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${EXP_DIR}/../../scripts/_run_experiment.sh" "${EXP_DIR}"
