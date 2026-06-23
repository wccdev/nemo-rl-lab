"""本地提交台账读取。"""
from __future__ import annotations

import json
from pathlib import Path


def read_ledger(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def enrich_runs_with_status(entries: list[dict], status_map: dict[str, str]) -> list[dict]:
    out = []
    for e in entries:
        rid = str(e.get("run_id", ""))
        row = dict(e)
        row["job_status"] = status_map.get(rid, "-")
        out.append(row)
    return sorted(out, key=lambda x: str(x.get("time", "")), reverse=True)
