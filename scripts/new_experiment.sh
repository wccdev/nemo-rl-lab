#!/usr/bin/env bash
# 新建实验： scripts/new_experiment.sh <experiments|projects> <实验名> [来源实验] [集群profile]
#   无来源：从 templates/experiment-template 起一个空白实验
#   有来源：fork 一个现成实验（copy 目录，并把 config.yaml 的 swanlab project/name 与 README 标题改成新名）
#   集群profile：写入实验自带的 cluster 文件（软绑定的默认集群）；fork 时不给则继承来源实验。
# 例：
#   scripts/new_experiment.sh experiments grpo_qwen3.5-4b_gsm8k_v1 "" h100
#   scripts/new_experiment.sh experiments grpo_qwen3.5-4b_gsm8k_lr1e4 grpo_qwen3.5-4b_gsm8k_v1
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KIND="${1:?用法: new_experiment.sh <experiments|projects> <实验名> [来源实验] [集群profile]}"
NAME="${2:?用法: new_experiment.sh <experiments|projects> <实验名> [来源实验] [集群profile]}"
SRC="${3:-}"
CLUSTER="${4:-}"

case "${KIND}" in
  experiments|projects) ;;
  *) echo "第一个参数必须是 experiments 或 projects"; exit 1 ;;
esac

# 给了集群 profile 就校验它存在（cluster/<profile>/overrides.conf）。
if [[ -n "${CLUSTER}" && ! -f "${REPO_ROOT}/cluster/${CLUSTER}/overrides.conf" ]]; then
  echo "未知集群 profile: ${CLUSTER}（cluster/${CLUSTER}/overrides.conf 不存在）"
  echo "可选: $(cd "${REPO_ROOT}/cluster" && for d in */overrides.conf; do printf '%s ' "${d%/overrides.conf}"; done)"
  exit 1
fi

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

  # 集群绑定：给了 --cluster 就覆盖；否则继承来源实验自带的 cluster（cp -r 已带过来）。
  [[ -n "${CLUSTER}" ]] && printf '%s\n' "${CLUSTER}" > "${DEST}/cluster"
  echo "已 fork 实验: ${DEST}（来源: ${SRC}）"
  echo "  · config.yaml 的 swanlab project/name 与 README 标题已改为: ${NAME}"
  echo "  · 目标集群(cluster): $(tr -d '[:space:]' < "${DEST}/cluster" 2>/dev/null || echo 未设置)"
  echo "下一步: 改 ${DEST}/config.yaml 顶部【① 调参区】试你的超参，然后 lab submit ${NAME}"
else
  # —— 从空白模板新建 ——
  cp -r "${REPO_ROOT}/templates/experiment-template" "${DEST}"
  rm -f "${DEST}/.gitkeep"
  # 集群绑定：给了 --cluster 就覆盖模板默认；否则用模板自带的 cluster。
  [[ -n "${CLUSTER}" ]] && printf '%s\n' "${CLUSTER}" > "${DEST}/cluster"

  # 按方法塑形骨架（LAB_METHOD 由 lab new --method 传入，默认 grpo=模板原样）。
  METHOD="${LAB_METHOD:-grpo}"
  case "${METHOD}" in
    grpo) ;;  # 模板默认即 GRPO，无需改
    sft)
      # 切基底 → sft.yaml；删掉 grpo/loss_fn 块、补 sft 块（SFT schema 没有 grpo/loss_fn）。
      python3 - "${DEST}/config.yaml" <<'PY'
import re, sys, pathlib
f = pathlib.Path(sys.argv[1])
t = f.read_text().replace("../../configs/base/grpo_math_1B.yaml", "../../configs/base/sft.yaml")

def drop_block(text, key):
    lines, out, i = text.splitlines(keepends=True), [], 0
    while i < len(lines):
        if re.match(rf'^{key}:\s*$', lines[i]):
            i += 1
            while i < len(lines) and not re.match(r'^[A-Za-z_]+:', lines[i]):
                i += 1
        else:
            out.append(lines[i]); i += 1
    return "".join(out)

t = drop_block(drop_block(t, "grpo"), "loss_fn")
sft_block = (
    "sft:\n"
    "  max_num_epochs: 1\n"
    "  val_period: 50\n"
    "  val_batches: 8\n\n"
    "# 数据集：SFT 读指令数据（见 common/data/README.md 与官方 examples/run_sft.py）\n"
    "# data:\n"
    "#   train:\n"
    "#     data_path: /abs/path/train.jsonl\n\n"
)
t = t.replace("logger:", sft_block + "logger:", 1)
f.write_text(t)
PY
      # 取消模板里 SFT 入口那行的注释（行首是 '# export ENTRY=...'）。
      python3 - "${DEST}/run.sh" <<'PY'
import sys, pathlib, re
f = pathlib.Path(sys.argv[1])
t = re.sub(r'^#\s*export ENTRY="\$\{ENTRY:-examples/run_sft\.py\}"',
           'export ENTRY="${ENTRY:-examples/run_sft.py}"', f.read_text(), flags=re.M)
f.write_text(t)
PY
      ;;
    agent)
      # 切基底 → grpo_sliding_puzzle.yaml（含 env + max_rollout_turns），bump 多轮上限，放入 run.py 骨架。
      python3 - "${DEST}/config.yaml" <<'PY'
import re, sys, pathlib
f = pathlib.Path(sys.argv[1])
t = f.read_text().replace("../../configs/base/grpo_math_1B.yaml", "../../configs/base/grpo_sliding_puzzle.yaml")
t = re.sub(r'^(\s*max_rollout_turns:\s*)1\b.*$',
           r'\g<1>6            # 多轮 Agent：工具调用 + 答题轮数上限', t, flags=re.M)
f.write_text(t)
PY
      cp "${REPO_ROOT}/templates/agent-run.py.tmpl" "${DEST}/run.py"
      ;;
    *)
      echo "未知 --method: ${METHOD}（可选 grpo | sft | agent）"; rm -rf "${DEST}"; exit 1 ;;
  esac

  echo "已创建实验: ${DEST}（method=${METHOD}）"
  echo "  · 目标集群(cluster): $(tr -d '[:space:]' < "${DEST}/cluster" 2>/dev/null || echo 未设置)（按需改：echo h100 > ${DEST}/cluster）"
  echo "下一步:"
  echo "  1. 编辑 ${DEST}/README.md（目标 / 模型 / 数据 / SwanLab）"
  echo "  2. 编辑 ${DEST}/config.yaml（基底已设为 ${METHOD}；写本实验差异）"
  case "${METHOD}" in
    sft)   echo "  3. SFT 入口已设好（run.sh 的 ENTRY=examples/run_sft.py）；填好数据后 lab submit ${NAME}" ;;
    agent) echo "  3. 编辑 ${DEST}/run.py（已放骨架：实现你的环境 + 数据，见文件内 TODO 与 multitool 范例）" ;;
    *)     echo "  3. 自定义多轮环境：写 ${DEST}/run.py（自动选用）。见 configs/README.md" ;;
  esac
fi
