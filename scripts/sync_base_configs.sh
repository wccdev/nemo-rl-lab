#!/usr/bin/env bash
# 薄封装：逻辑在 nemo_rl_lab/sync_base.py（跨平台单一事实来源）。
# 用法： NEMO_RL_DIR=/path/to/NeMo-RL bash scripts/sync_base_configs.sh
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec uv run --project "${REPO_ROOT}" python -m nemo_rl_lab.sync_base "$@"
