#!/usr/bin/env bash
# 在 worker 节点（spark-1）加入 Ray 集群
set -euo pipefail
HEAD_ADDRESS="${HEAD_ADDRESS:-spark-0:6379}"
ray start --address="${HEAD_ADDRESS}"
