#!/usr/bin/env bash
# 快速新建实验： scripts/new_experiment.sh <experiments|projects> <实验名>
# 例： scripts/new_experiment.sh experiments grpo_qwen3.5-4b_gsm8k_v1
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KIND="${1:?用法: new_experiment.sh <experiments|projects> <实验名>}"
NAME="${2:?用法: new_experiment.sh <experiments|projects> <实验名>}"

case "${KIND}" in
  experiments|projects) ;;
  *) echo "第一个参数必须是 experiments 或 projects"; exit 1 ;;
esac

DEST="${REPO_ROOT}/${KIND}/${NAME}"
if [[ -e "${DEST}" ]]; then
  echo "已存在: ${DEST}"; exit 1
fi

cp -r "${REPO_ROOT}/templates/experiment-template" "${DEST}"
rm -f "${DEST}/.gitkeep"
echo "已创建实验: ${DEST}"
echo "下一步:"
echo "  1. 编辑 ${DEST}/README.md（目标 / 模型 / 数据 / SwanLab）"
echo "  2. 编辑 ${DEST}/config.yaml（选 defaults 基底+模型，写本实验差异）"
echo "  3. 若是 SFT/Agent，改 ${DEST}/run.sh 的 ENTRY（见 configs/README.md）"
