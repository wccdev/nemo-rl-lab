#!/usr/bin/env bash
# 从本地 NeMo-RL 源码同步官方 example 配置到 configs/base/（升级版本时用）。
# 用法： NEMO_RL_DIR=/path/to/NeMo-RL bash scripts/sync_base_configs.sh
set -euo pipefail

NEMO_RL_DIR="${NEMO_RL_DIR:?请设置 NEMO_RL_DIR 指向本地 NeMo-RL 源码目录}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${NEMO_RL_DIR}/examples/configs"
DST="${REPO_ROOT}/configs/base"

FILES=(grpo_math_1B.yaml sft.yaml grpo_sliding_puzzle.yaml)
# 需要更大模型基底时，把对应文件名加进来，例如 grpo_math_8B.yaml
# 注意：configs/base/grpo_megatron.yaml 是本仓库自定义 overlay（非官方副本），不在此同步、勿加入。

mkdir -p "${DST}"
for f in "${FILES[@]}"; do
  if [[ -f "${SRC}/${f}" ]]; then
    cp "${SRC}/${f}" "${DST}/${f}"
    echo "synced ${f}"
  else
    echo "WARN 未找到 ${SRC}/${f}，跳过"
  fi
done
echo "完成。请 git diff 检查变化后再提交。"
