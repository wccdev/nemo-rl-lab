#!/usr/bin/env bash
# 在 head 节点容器内启动 Ray 集群头（GB10 / DGX Spark，2 节点）。
# 在 NeMo-RL 容器里跑（需 uv + NeMo-RL 环境，ray 由其提供）。
# 可调环境变量：HEAD_IP / RAY_PORT / OBJECT_STORE_MEM / NEMO_RL_DIR
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/env.sh"

HEAD_IP="${HEAD_IP:-192.168.1.4}"
RAY_PORT="${RAY_PORT:-6379}"
OBJECT_STORE_MEM="${OBJECT_STORE_MEM:-4294967296}"   # 4GB

cd "${NEMO_RL_DIR:-.}"   # 让 `uv run ray` 使用 NeMo-RL 的 uv 环境
uv run ray start --head \
  --node-ip-address="${HEAD_IP}" \
  --port="${RAY_PORT}" \
  --dashboard-host=0.0.0.0 \
  --object-store-memory="${OBJECT_STORE_MEM}"
uv run ray status
