#!/usr/bin/env bash
# 在 head 节点（spark-0）启动 Ray 集群头
set -euo pipefail
ray start --head --port=6379 --dashboard-host=0.0.0.0
ray status
