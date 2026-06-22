#!/usr/bin/env bash
# 单机 1× H100：可选地手动起一个本地 Ray（带 dashboard）。
#
# 多数情况【不需要】跑这个：直接 `bash experiments/<exp>/run.sh` 时，
# NeMo-RL 会自动拉起本地 Ray。只有当你想用 `lab submit` / `lab web`
# 连本地 dashboard（默认 127.0.0.1:8265）观察作业时，才先跑这个起好集群。
#
# 在 NeMo-RL 容器里跑（ray 由 NeMo-RL 的 uv 环境提供）。
# 可调环境变量：HEAD_IP / RAY_PORT / OBJECT_STORE_MEM / NEMO_RL_DIR
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/env.sh"

HEAD_IP="${HEAD_IP:-127.0.0.1}"                      # 单机绑本地回环即可
RAY_PORT="${RAY_PORT:-6379}"
OBJECT_STORE_MEM="${OBJECT_STORE_MEM:-8589934592}"   # 8GB（H100 机器 host RAM 通常充足）

cd "${NEMO_RL_DIR:-.}"   # 让 `uv run ray` 使用 NeMo-RL 的 uv 环境
uv run ray start --head \
  --node-ip-address="${HEAD_IP}" \
  --port="${RAY_PORT}" \
  --num-gpus=1 \
  --dashboard-host=0.0.0.0 \
  --object-store-memory="${OBJECT_STORE_MEM}"
uv run ray status
