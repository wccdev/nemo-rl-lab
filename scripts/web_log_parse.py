#!/usr/bin/env python
"""Ray 作业日志解析（被 scripts/web_dashboard.py 复用）。

把 NeMo-RL GRPO 作业日志解析成结构化数据：
  - steps：训练每步 {step, total, avg_reward, step_time}
  - validations：每次验证 {step, avg_reward, accuracy, avg_len, dist, samples}
    其中 samples 是打印出来的 N 条对话 {idx, reward, user, assistant, env}
"""
from __future__ import annotations

import re

# 训练步分隔：========= Step 12/300 =========
_STEP_HDR = re.compile(r"=+\s*Step\s+(\d+)\s*/\s*(\d+)\s*=+")
# 训练步指标（带 • 项目符号，区别于验证框里的同名字段）
_STEP_REWARD = re.compile(r"[•·]\s*Avg Reward:\s*([-\d.]+)")
_STEP_TIME = re.compile(r"[•·]\s*Total step time:\s*([\d.]+)\s*s")
# 验证块
_VAL_START = re.compile(r"Starting validation at step\s+(\d+)")
_VAL_ACC = re.compile(r"[•·]\s*Accuracy:\s*([-\d.]+)")
_VAL_LEN = re.compile(r"[•·]\s*Average response length:\s*([\d.]+)")
_VAL_AVG_REWARD = re.compile(r"Avg Reward:\s*([-\d.]+)")
# 奖励分布：Reward -0.5000: ███ (49 samples)
_VAL_DIST = re.compile(r"Reward\s+([-\d.]+):.*?\((\d+)\s+samples?\)")
# 样本面板标题：╭─ 🔥 Sample 1 | Reward: 1.0000 ─╮
_SAMPLE_HDR = re.compile(r"Sample\s+(\d+)\s*\|\s*Reward:\s*([-\d.]+)")

_BORDER_CHARS = "│┃|"
_BOTTOM_CHARS = "╰╯"

# 这些进度/告警噪声行直接丢弃，避免污染解析与体积
_NOISE_SUBSTR = (
    "Processed prompts",
    "Rendering prompts",
    "VllmGenerationWorker",
    "warnings.warn",
    "FutureWarning",
    "This can lead",
    "TypedStorage",
    "torch._dynamo",
    "UserWarning",
)


def _is_noise(line: str) -> bool:
    return any(s in line for s in _NOISE_SUBSTR)


def _debox(line: str):
    """若是 Rich 框内容行（│ ... │），剥掉边框返回内部文本；否则返回 None。"""
    s = line.rstrip("\n")
    stripped = s.strip()
    if not stripped or stripped[0] not in _BORDER_CHARS:
        return None
    inner = stripped.strip(_BORDER_CHARS)
    if inner.startswith(" "):
        inner = inner[1:]
    return inner.rstrip()


def _finalize_sample(sample: dict) -> dict:
    return {
        "idx": sample["idx"],
        "reward": sample["reward"],
        "user": "\n".join(sample["user"]).strip(),
        "assistant": "\n".join(sample["assistant"]).strip(),
        "env": "\n".join(sample["env"]).strip(),
    }


def parse_logs(logs: str) -> dict:
    """把作业日志解析成 {steps, validations}（见模块 docstring）。"""
    steps: list[dict] = []
    validations: list[dict] = []

    cur_step: dict | None = None
    cur_val: dict | None = None
    cur_sample: dict | None = None
    section: str | None = None  # user / assistant / env

    def close_val():
        nonlocal cur_val, cur_sample, section
        if cur_sample is not None:
            cur_val["samples"].append(_finalize_sample(cur_sample))
            cur_sample = None
        if cur_val is not None:
            validations.append(cur_val)
        cur_val = None
        section = None

    for raw in logs.splitlines():
        if _is_noise(raw):
            continue

        m = _STEP_HDR.search(raw)
        if m:
            # 训练步分隔同时意味着上一个验证块结束
            close_val()
            cur_step = {
                "step": int(m.group(1)),
                "total": int(m.group(2)),
                "avg_reward": None,
                "step_time": None,
            }
            steps.append(cur_step)
            continue

        mv = _VAL_START.search(raw)
        if mv:
            close_val()
            cur_val = {
                "step": int(mv.group(1)),
                "avg_reward": None,
                "accuracy": None,
                "avg_len": None,
                "dist": [],
                "samples": [],
            }
            continue

        if cur_val is not None:
            inner = _debox(raw)
            if inner is not None:
                # 框内：样本对话内容
                if inner.startswith("USER:"):
                    section = "user"
                    rest = inner[len("USER:"):].strip()
                    if cur_sample is not None and rest:
                        cur_sample["user"].append(rest)
                    continue
                if inner.startswith("ASSISTANT:"):
                    section = "assistant"
                    rest = inner[len("ASSISTANT:"):].strip()
                    if cur_sample is not None and rest:
                        cur_sample["assistant"].append(rest)
                    continue
                if inner.startswith("ENVIRONMENT:"):
                    section = "env"
                    rest = inner[len("ENVIRONMENT:"):].strip()
                    if cur_sample is not None and rest:
                        cur_sample["env"].append(rest)
                    continue
                # 验证统计框里的平均奖励（取第一次出现，且不在样本面板内）
                ar = _VAL_AVG_REWARD.search(inner)
                if ar and cur_val["avg_reward"] is None and cur_sample is None:
                    cur_val["avg_reward"] = float(ar.group(1))
                dm = _VAL_DIST.search(inner)
                if dm and cur_sample is None:
                    cur_val["dist"].append(
                        {"reward": float(dm.group(1)), "count": int(dm.group(2))}
                    )
                # 普通正文行，归到当前 section
                if cur_sample is not None and section in ("user", "assistant", "env"):
                    cur_sample[section].append(inner)
                continue

            # 非框行：样本标题 / 面板收尾 / 验证摘要
            sh = _SAMPLE_HDR.search(raw)
            if sh:
                if cur_sample is not None:
                    cur_val["samples"].append(_finalize_sample(cur_sample))
                cur_sample = {
                    "idx": int(sh.group(1)),
                    "reward": float(sh.group(2)),
                    "user": [],
                    "assistant": [],
                    "env": [],
                }
                section = None
                continue
            if any(c in raw for c in _BOTTOM_CHARS) and cur_sample is not None:
                cur_val["samples"].append(_finalize_sample(cur_sample))
                cur_sample = None
                section = None
                continue
            acc = _VAL_ACC.search(raw)
            if acc:
                cur_val["accuracy"] = float(acc.group(1))
                continue
            ln = _VAL_LEN.search(raw)
            if ln:
                cur_val["avg_len"] = float(ln.group(1))
                continue
            dm = _VAL_DIST.search(raw)
            if dm and cur_sample is None:
                cur_val["dist"].append(
                    {"reward": float(dm.group(1)), "count": int(dm.group(2))}
                )
                continue
            continue

        # 训练步指标（非验证块）
        if cur_step is not None:
            sr = _STEP_REWARD.search(raw)
            if sr:
                cur_step["avg_reward"] = float(sr.group(1))
                continue
            st = _STEP_TIME.search(raw)
            if st:
                cur_step["step_time"] = float(st.group(1))
                continue

    close_val()

    steps = [s for s in steps if s["avg_reward"] is not None]
    return {"steps": steps, "validations": validations}
