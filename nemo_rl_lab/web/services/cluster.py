"""Ray dashboard HTTP 辅助（GPU 概览）。"""
from __future__ import annotations

import json
import urllib.request
from typing import Optional


def dashboard_get(address: str, path: str, timeout: float = 6.0) -> dict:
    with urllib.request.urlopen(f"{address.rstrip('/')}{path}", timeout=timeout) as r:
        return json.loads(r.read())


def fetch_jobs_http(address: str) -> list[dict]:
    data = dashboard_get(address, "/api/jobs/")
    return data if isinstance(data, list) else list(data.values())


def run_status_map(address: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for j in fetch_jobs_http(address):
        meta = j.get("metadata") or {}
        rid = meta.get("lab_run_id")
        if rid:
            out[str(rid)] = str(j.get("status", "?"))
    return out


def gpu_summary(address: str) -> Optional[dict]:
    data = dashboard_get(address, "/api/cluster_status")
    usage = (
        (((data.get("data") or {}).get("clusterStatus") or {}).get("loadMetricsReport") or {})
        .get("usage")
        or {}
    )
    if not usage:
        return None
    accel = [k.split(":", 1)[1] for k in usage if k.lower().startswith(("acceleratortype:", "accelerator_type:"))]
    gu, gt = usage.get("GPU", [0, 0])
    cu, ct = usage.get("CPU", [0, 0])
    mu, mt = usage.get("memory", [0, 0])
    return {
        "accel": accel,
        "gpu_used": float(gu),
        "gpu_total": float(gt),
        "gpu_free": float(gt) - float(gu),
        "cpu_used": float(cu),
        "cpu_total": float(ct),
        "memory_used_gib": float(mu) / 2**30,
        "memory_total_gib": float(mt) / 2**30,
    }
