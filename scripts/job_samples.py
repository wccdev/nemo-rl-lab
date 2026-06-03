#!/usr/bin/env python
"""从 Ray 作业日志里抽取「验证样本轨迹」（被 `lab job samples` 调用）。

微调时 `logger.num_val_samples_to_print` 条完整对话（含多轮工具调用）会被打印到 stdout，
进而落到 Ray 作业日志里。本脚本用 JobSubmissionClient.get_job_logs 把日志拉到本机
（走 dashboard HTTP，不用登集群），只保留验证样本面板与结果摘要，方便在 Mac 上排查
「有没有走偏、有没有正确调工具」。

用法：
    python scripts/job_samples.py --address http://192.168.1.4:8265 <job_id> [--last N]
"""
from __future__ import annotations

import argparse

from ray.job_submission import JobSubmissionClient

# 命中这些关键字的行直接保留（验证分隔/结果摘要）
_KEEP_SUBSTR = (
    "Starting validation at step",
    "End of Samples",
    "Validation Results",
    "Accuracy:",
    "Average response length",
    "Samples processed",
)
# Rich 面板的边框/正文字符（每条样本对话都画在面板里）
_BOX_CHARS = set("│╭╮╰╯")


def _keep(line: str) -> bool:
    if any(s in line for s in _KEEP_SUBSTR):
        return True
    return any(ch in line for ch in _BOX_CHARS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("job_id", help="作业 ID（见 lab job list）")
    ap.add_argument("--address", required=True, help="Ray dashboard 地址")
    ap.add_argument(
        "--last", type=int, default=0,
        help="只看最近 N 次验证的样本（默认 0=全部）",
    )
    args = ap.parse_args()

    client = JobSubmissionClient(args.address)
    logs = client.get_job_logs(args.job_id)

    kept = [ln for ln in logs.splitlines() if _keep(ln)]
    if not kept:
        print(
            "未在日志里找到验证样本面板。\n"
            "可能原因：① 还没跑到首次验证（val_at_start 也要等模型加载完）；"
            "② config 里 logger.num_val_samples_to_print=0。\n"
            "把它设为 >0（如 3）重新提交即可。"
        )
        return

    # 按「Starting validation at step」切块，便于 --last 取最近几次
    blocks: list[list[str]] = []
    cur: list[str] = []
    for ln in kept:
        if "Starting validation at step" in ln and cur:
            blocks.append(cur)
            cur = []
        cur.append(ln)
    if cur:
        blocks.append(cur)

    if args.last and args.last > 0:
        blocks = blocks[-args.last :]

    for blk in blocks:
        print("\n".join(blk))
        print()


if __name__ == "__main__":
    main()
