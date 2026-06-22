#!/usr/bin/env bash
# 新建实验： scripts/new_experiment.sh <experiments|projects> <实验名> [来源实验]
#   无来源：从 templates/experiment-template 起一个空白实验
#   有来源：fork 一个现成实验（copy 目录，并把 config.yaml 的 swanlab project/name 与 README 标题改成新名）
# 例：
#   scripts/new_experiment.sh experiments grpo_qwen3.5-4b_gsm8k_v1
#   scripts/new_experiment.sh experiments grpo_qwen3.5-4b_gsm8k_lr1e4 grpo_qwen3.5-4b_gsm8k_v1
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KIND="${1:?用法: new_experiment.sh <experiments|projects> <实验名> [来源实验]}"
NAME="${2:?用法: new_experiment.sh <experiments|projects> <实验名> [来源实验]}"
SRC="${3:-}"

case "${KIND}" in
  experiments|projects) ;;
  *) echo "第一个参数必须是 experiments 或 projects"; exit 1 ;;
esac

DEST="${REPO_ROOT}/${KIND}/${NAME}"
if [[ -e "${DEST}" ]]; then
  echo "已存在: ${DEST}"; exit 1
fi

if [[ -n "${SRC}" ]]; then
  # —— fork 现成实验 ——
  SRC_DIR=""
  for c in "${SRC}" "experiments/${SRC}" "projects/${SRC}"; do
    if [[ -d "${REPO_ROOT}/${c}" ]]; then SRC_DIR="${REPO_ROOT}/${c}"; break; fi
  done
  [[ -n "${SRC_DIR}" ]] || { echo "找不到来源实验: ${SRC}（试过 ${SRC} / experiments/${SRC} / projects/${SRC}）"; exit 1; }

  cp -r "${SRC_DIR}" "${DEST}"
  rm -rf "${DEST}/outputs"  # 别把来源的训练产物也 fork 过来

  # 仅做行级文本替换（保留 config.yaml 里的注释/调参速查），把 swanlab project/name 与 README 标题改成新名。
  python3 - "${DEST}" "${NAME}" <<'PY'
import re, sys, pathlib
dest, name = pathlib.Path(sys.argv[1]), sys.argv[2]
cfg = dest / "config.yaml"
if cfg.is_file():
    lines = cfg.read_text().splitlines()
    in_sw, sw_indent = False, 0
    for i, ln in enumerate(lines):
        s, indent = ln.strip(), len(ln) - len(ln.lstrip())
        if s == "swanlab:":
            in_sw, sw_indent = True, indent
            continue
        if in_sw:
            if s and indent <= sw_indent:
                in_sw = False
            else:
                m = re.match(r'^(\s*)(project|name):\s*.*$', ln)
                if m:
                    lines[i] = f'{m.group(1)}{m.group(2)}: "{name}"'
    cfg.write_text("\n".join(lines) + "\n")
readme = dest / "README.md"
if readme.is_file():
    rl = readme.read_text().splitlines()
    for i, ln in enumerate(rl):
        if ln.startswith("# "):
            rl[i] = f"# {name}"
            break
    readme.write_text("\n".join(rl) + "\n")
PY

  echo "已 fork 实验: ${DEST}（来源: ${SRC}）"
  echo "  · config.yaml 的 swanlab project/name 与 README 标题已改为: ${NAME}"
  echo "下一步: 改 ${DEST}/config.yaml 顶部【① 调参区】试你的超参，然后 lab submit ${NAME}"
else
  # —— 从空白模板新建 ——
  cp -r "${REPO_ROOT}/templates/experiment-template" "${DEST}"
  rm -f "${DEST}/.gitkeep"
  echo "已创建实验: ${DEST}"
  echo "下一步:"
  echo "  1. 编辑 ${DEST}/README.md（目标 / 模型 / 数据 / SwanLab）"
  echo "  2. 编辑 ${DEST}/config.yaml（选 defaults 基底+模型，写本实验差异）"
  echo "  3. 若是 SFT/Agent，改 ${DEST}/run.sh 的 ENTRY（见 configs/README.md）"
fi
