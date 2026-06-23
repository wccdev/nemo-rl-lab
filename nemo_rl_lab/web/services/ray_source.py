"""Ray 作业数据层（从 scripts/web_dashboard.py 拆出）。"""
from __future__ import annotations

import json
import re
import sys
import threading
import time
from pathlib import Path

from fastapi import HTTPException
from ray.job_submission import JobSubmissionClient

# scripts/ 下的 web_log_parse
_REPO = Path(__file__).resolve().parents[3]
_SCRIPTS = _REPO / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from web_log_parse import parse_logs  # noqa: E402

from nemo_rl_lab.config_resolve import resolve as resolve_config  # noqa: E402

_TERMINAL = {"FAILED", "SUCCEEDED", "STOPPED"}
_EXP_RE = re.compile(r"experiments/([^/\s]+)")


def _exp_name(entrypoint: str) -> str:
    if not entrypoint:
        return "-"
    m = _EXP_RE.search(entrypoint)
    if m:
        return m.group(1)
    for tok in entrypoint.split():
        if "/run.py" in tok or tok.endswith(".py"):
            parts = tok.strip().split("/")
            return parts[-2] if len(parts) >= 2 else parts[-1]
    return entrypoint[:32]


def _fmt_start(start_ms):
    if not start_ms:
        return "-"
    return time.strftime("%m-%d %H:%M", time.localtime(start_ms / 1000))


def _fmt_dur(start_ms, end_ms):
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


def _summary(steps: list, vmeta: list) -> dict:
    accs = [v for v in vmeta if v.get("accuracy") is not None]
    base_acc = accs[0]["accuracy"] if accs else None
    final_acc = accs[-1]["accuracy"] if accs else None
    delta_acc = (final_acc - base_acc) if (base_acc is not None and final_acc is not None) else None
    rsteps = [s for s in steps if s.get("avg_reward") is not None]
    reward_first = rsteps[0]["avg_reward"] if rsteps else None
    reward_last = rsteps[-1]["avg_reward"] if rsteps else None
    delta_reward = (reward_last - reward_first) if (reward_first is not None and reward_last is not None) else None
    last_step = steps[-1]["step"] if steps else None
    total = steps[-1]["total"] if steps else None
    return {
        "base_acc": base_acc,
        "final_acc": final_acc,
        "delta_acc": delta_acc,
        "reward_first": reward_first,
        "reward_last": reward_last,
        "delta_reward": delta_reward,
        "last_step": last_step,
        "total": total,
        "n_val": len(accs),
    }


_cfg_cache: dict[str, tuple[float, dict | None]] = {}


def resolve_exp_config(repo_root: Path, exp: str) -> dict | None:
    if not exp:
        return None
    path = repo_root / "experiments" / exp / "config.yaml"
    if not path.is_file():
        for kind in ("experiments", "projects"):
            alt = repo_root / kind / exp / "config.yaml"
            if alt.is_file():
                path = alt
                break
        else:
            return None
    mtime = path.stat().st_mtime
    key = str(path)
    hit = _cfg_cache.get(key)
    if hit and hit[0] == mtime:
        return hit[1]
    try:
        resolved = resolve_config(path)
    except Exception:  # noqa: BLE001
        resolved = None
    _cfg_cache[key] = (mtime, resolved)
    return resolved


def _flatten(d: dict, prefix: str = "") -> dict:
    out: dict = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


class RayDataSource:
    def __init__(self, address: str, repo_root: Path, ttl: float = 5.0):
        self.address = address
        self.repo_root = repo_root
        self.ttl = ttl
        self._client = JobSubmissionClient(address)
        self._lock = threading.Lock()
        self._jobs_cache: tuple[float, list] | None = None
        self._log_cache: dict[str, tuple[float, dict]] = {}

    def list_jobs(self) -> list[dict]:
        now = time.time()
        with self._lock:
            if self._jobs_cache and now - self._jobs_cache[0] < self.ttl:
                return self._jobs_cache[1]
        jobs = self._client.list_jobs()
        jobs.sort(key=lambda j: j.start_time or 0, reverse=True)
        rows = []
        for j in jobs:
            jid = j.submission_id or j.job_id
            if not jid:
                continue
            status = str(getattr(j.status, "value", j.status))
            meta = getattr(j, "metadata", None) or {}
            rows.append({
                "id": jid,
                "exp": _exp_name(j.entrypoint or ""),
                "status": status,
                "entrypoint": (j.entrypoint or "")[:120],
                "start": _fmt_start(j.start_time),
                "dur": _fmt_dur(j.start_time, j.end_time),
                "running": status not in _TERMINAL,
                "lab_run_id": meta.get("lab_run_id") if isinstance(meta, dict) else None,
                "start_time": j.start_time,
            })
        with self._lock:
            self._jobs_cache = (now, rows)
        return rows

    def _parsed(self, job_id: str) -> dict:
        now = time.time()
        with self._lock:
            hit = self._log_cache.get(job_id)
            if hit and now - hit[0] < self.ttl:
                return hit[1]
        logs = self._client.get_job_logs(job_id)
        parsed = parse_logs(logs)
        with self._lock:
            self._log_cache[job_id] = (now, parsed)
        return parsed

    def _exp_for(self, job_id: str) -> str:
        for row in self.list_jobs():
            if row["id"] == job_id:
                return row["exp"]
        return job_id[:16]

    def job_overview(self, job_id: str) -> dict:
        parsed = self._parsed(job_id)
        vmeta = [
            {
                "step": v["step"],
                "avg_reward": v["avg_reward"],
                "accuracy": v["accuracy"],
                "avg_len": v["avg_len"],
                "dist": v["dist"],
                "sample_count": len(v["samples"]),
            }
            for v in parsed["validations"]
        ]
        exp = self._exp_for(job_id)
        cfg = resolve_exp_config(self.repo_root, exp)
        model = (cfg.get("policy", {}) or {}).get("model_name") if cfg else None
        return {
            "job_id": job_id,
            "exp": exp,
            "model": model,
            "steps": parsed["steps"],
            "validations": vmeta,
            "summary": _summary(parsed["steps"], vmeta),
        }

    def config_diff(self, ids: list[str]) -> dict:
        cols, flats = [], []
        for jid in ids:
            exp = self._exp_for(jid)
            cfg = resolve_exp_config(self.repo_root, exp)
            flat = _flatten(cfg) if cfg else {}
            cols.append({
                "id": jid,
                "exp": exp,
                "model": flat.get("policy.model_name"),
                "has_config": bool(cfg),
            })
            flats.append(flat)
        keys = sorted(set().union(*[set(f) for f in flats])) if flats else []
        rows = []
        n_both_diff = n_one_sided = 0
        for k in keys:
            vals = [f.get(k, None) if k in f else "__MISSING__" for f in flats]
            missing = any(v == "__MISSING__" for v in vals)
            norm = [json.dumps(v, ensure_ascii=False, sort_keys=True) for v in vals]
            differ = len(set(norm)) > 1
            if missing:
                kind = "one_sided"
                n_one_sided += 1
            elif differ:
                kind = "both_diff"
                n_both_diff += 1
            else:
                kind = "both_same"
            rows.append({"key": k, "values": vals, "diff": differ, "kind": kind})
        return {
            "jobs": cols,
            "rows": rows,
            "n_both_diff": n_both_diff,
            "n_one_sided": n_one_sided,
            "n_total": len(rows),
        }

    def stop_job(self, job_id: str) -> dict:
        try:
            ok = self._client.stop_job(job_id)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"停止失败：{e}") from e
        with self._lock:
            self._jobs_cache = None
        return {"ok": bool(ok), "id": job_id}

    def delete_job(self, job_id: str) -> dict:
        try:
            self._client.delete_job(job_id)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"删除失败：{e}") from e
        with self._lock:
            self._jobs_cache = None
            self._log_cache.pop(job_id, None)
        return {"ok": True, "id": job_id}

    def samples_page(self, job_id: str, vidx: int, offset: int, limit: int) -> dict:
        parsed = self._parsed(job_id)
        vals = parsed["validations"]
        if vidx < 0 or vidx >= len(vals):
            raise HTTPException(status_code=404, detail="validation index out of range")
        samples = vals[vidx]["samples"]
        page = samples[offset: offset + limit]
        return {
            "vidx": vidx,
            "step": vals[vidx]["step"],
            "total": len(samples),
            "offset": offset,
            "limit": limit,
            "samples": page,
        }

    def get_logs(self, job_id: str) -> str:
        return self._client.get_job_logs(job_id)
