#!/usr/bin/env python
"""精简展示 Ray 集群作业列表（被 `lab job list` 调用）。

直接用 ray 原生 `ray job list` 输出整块 JobDetails 太啰嗦；这里用 JobSubmissionClient
取结构化数据，排成「ID / 状态 / 用时 / 入口」表格。须在装了 Ray 的环境运行
（CLI 用 `uv run --extra submit python` 调起）。

用法：
    python scripts/ray_jobs.py --address http://192.168.1.4:8265 [--all]
"""
from __future__ import annotations

import argparse
import time

from ray.job_submission import JobSubmissionClient

# 状态 -> 终端颜色（ANSI），让运行中/失败一眼可辨
_COLOR = {
    "RUNNING": "\033[36m",    # 青
    "SUCCEEDED": "\033[32m",  # 绿
    "FAILED": "\033[31m",     # 红
    "STOPPED": "\033[33m",    # 黄
    "PENDING": "\033[35m",    # 紫
}
_RESET = "\033[0m"


def _fmt_duration(start_ms: int | None, end_ms: int | None) -> str:
    if not start_ms:
        return "-"
    end = end_ms if end_ms else int(time.time() * 1000)
    secs = max(0, (end - start_ms) // 1000)
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _fmt_start(start_ms: int | None) -> str:
    if not start_ms:
        return "-"
    return time.strftime("%m-%d %H:%M", time.localtime(start_ms / 1000))


_TERMINAL = {"FAILED", "SUCCEEDED", "STOPPED"}


def _clean(client: JobSubmissionClient, assume_yes: bool = False) -> None:
    """删除所有处于终态的作业（RUNNING/PENDING 会跳过；不删磁盘日志）。"""
    jobs = client.list_jobs()

    def _is_terminal(j) -> bool:
        return str(j.status) in _TERMINAL or getattr(j.status, "value", "") in _TERMINAL

    def _is_submission(j) -> bool:
        # 只有 submission 类型可删；driver 类型（head 上直接 python 跑的）删不了，跳过避免报错
        t = getattr(j, "type", None)
        return j.submission_id is not None and str(getattr(t, "value", t)).upper().endswith("SUBMISSION")

    targets = [j for j in jobs if _is_terminal(j) and _is_submission(j)]
    skipped_driver = [j for j in jobs if _is_terminal(j) and not _is_submission(j)]
    if not targets:
        print("没有可清理的已结束提交作业。")
        if skipped_driver:
            print(f"（{len(skipped_driver)} 个 driver 类型作业无法删除，已跳过）")
        return
    keep = len(jobs) - len(targets)
    note = f"，其中 {len(skipped_driver)} 个 driver 作业无法删" if skipped_driver else ""
    print(f"将删除 {len(targets)} 个已结束提交作业（保留 {keep} 个：运行中/等待中{note}）：")
    for j in targets:
        print(f"  - {j.submission_id or j.job_id}  {j.status}")
    if not assume_yes:
        ans = input("确认删除？[y/N] ").strip().lower()
        if ans not in ("y", "yes"):
            print("已取消。")
            return
    ok = 0
    for j in targets:
        jid = j.submission_id or j.job_id
        try:
            client.delete_job(jid)
            ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"  ⚠️ 删除 {jid} 失败: {e}")
    print(f"已删除 {ok}/{len(targets)} 个作业。")


_ACTIVE = {"RUNNING", "PENDING"}


def _cancel_all(client: JobSubmissionClient, assume_yes: bool = False) -> None:
    """停止所有运行中/等待中的作业（区别于 --clean 只删终态记录）。"""

    def _is_active(j) -> bool:
        return str(j.status) in _ACTIVE or getattr(j.status, "value", "") in _ACTIVE

    targets = [j for j in client.list_jobs() if _is_active(j)]
    if not targets:
        print("没有运行中/等待中的作业。")
        return
    print(f"将停止 {len(targets)} 个活跃作业：")
    for j in targets:
        print(f"  - {j.submission_id or j.job_id}  {j.status}")
    if not assume_yes:
        ans = input("确认全部停止？[y/N] ").strip().lower()
        if ans not in ("y", "yes"):
            print("已取消。")
            return
    ok = 0
    for j in targets:
        jid = j.submission_id or j.job_id
        try:
            client.stop_job(jid)
            ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"  ⚠️ 停止 {jid} 失败: {e}")
    print(f"已请求停止 {ok}/{len(targets)} 个作业（停止是异步的，稍后 lab job list 复核状态）。")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--address", required=True, help="Ray dashboard 地址")
    ap.add_argument("--all", action="store_true", help="显示全部（默认只显示最近 15 条）")
    ap.add_argument("--clean", action="store_true", help="删除所有已结束(FAILED/SUCCEEDED/STOPPED)的作业")
    ap.add_argument("--cancel-all", action="store_true", help="停止所有运行中/等待中的作业")
    ap.add_argument("--yes", action="store_true", help="清理/停止时跳过确认")
    args = ap.parse_args()

    client = JobSubmissionClient(args.address)

    if args.clean:
        _clean(client, assume_yes=args.yes)
        return

    if args.cancel_all:
        _cancel_all(client, assume_yes=args.yes)
        return

    jobs = client.list_jobs()
    # 按开始时间倒序（新的在上）
    jobs.sort(key=lambda j: j.start_time or 0, reverse=True)
    if not args.all:
        jobs = jobs[:15]

    if not jobs:
        print("（没有作业）")
        return

    rows = []
    for j in jobs:
        entry = j.entrypoint or ""
        if len(entry) > 46:
            entry = entry[:43] + "..."
        rows.append(
            (
                j.submission_id or (j.job_id or "-"),
                j.status,
                _fmt_start(j.start_time),
                _fmt_duration(j.start_time, j.end_time),
                entry,
            )
        )

    headers = ("JOB ID", "STATUS", "START", "DUR", "ENTRYPOINT")
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    # STATUS 列按无色长度对齐，颜色码不计入宽度
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(line)
    print("  ".join("-" * widths[i] for i in range(len(headers))))
    for r in rows:
        status = r[1]
        color = _COLOR.get(status, "")
        status_cell = f"{color}{status}{_RESET}".ljust(widths[1] + len(color) + len(_RESET)) if color else status.ljust(widths[1])
        cells = [
            r[0].ljust(widths[0]),
            status_cell,
            r[2].ljust(widths[2]),
            r[3].ljust(widths[3]),
            r[4],
        ]
        print("  ".join(cells))

    print(f"\n共 {len(jobs)} 条" + ("" if args.all else "（最近 15 条；--all 看全部）"))
    print("详情/日志： lab job logs <JOB ID> -f")


if __name__ == "__main__":
    main()
