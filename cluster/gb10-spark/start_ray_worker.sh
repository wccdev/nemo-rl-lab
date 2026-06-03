#!/usr/bin/env bash
# 在 worker 节点容器内加入 Ray 集群（GB10 / DGX Spark）。
# 在 NeMo-RL 容器里跑（需 uv + NeMo-RL 环境，ray 由其提供）。
# 可调环境变量：HEAD_IP / RAY_PORT / HEAD_ADDRESS / NODE_IP / NEMO_RL_DIR
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/env.sh"

HEAD_IP="${HEAD_IP:-192.168.1.4}"
RAY_PORT="${RAY_PORT:-6379}"
HEAD_ADDRESS="${HEAD_ADDRESS:-${HEAD_IP}:${RAY_PORT}}"
NODE_IP="${NODE_IP:-192.168.1.5}"

cd "${NEMO_RL_DIR:-.}"   # 让 `uv run ray` 使用 NeMo-RL 的 uv 环境
uv run ray start --address="${HEAD_ADDRESS}" --node-ip-address="${NODE_IP}"
uv run ray status
