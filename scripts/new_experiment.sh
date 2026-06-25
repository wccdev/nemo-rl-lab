#!/usr/bin/env bash
# 薄封装：逻辑在 nemo_rl_lab/new_experiment.py（跨平台单一事实来源）。
# 用法同旧版： scripts/new_experiment.sh <experiments|projects> <实验名> [来源实验] [集群profile]
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec uv run --project "${REPO_ROOT}" python -m nemo_rl_lab.new_experiment "$@"
