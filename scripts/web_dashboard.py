#!/usr/bin/env python
"""本地 Web 面板：查看 Ray 作业的训练 reward 曲线 + 验证对话样本（被 `lab web` 调用）。

数据来源与 `lab job samples` 完全一致——`JobSubmissionClient.get_job_logs`（走 dashboard HTTP），
纯本地运行、零训练侧改动、不依赖任何数据库或集群服务。HTTP 服务用 FastAPI + uvicorn，
前端单页内嵌（无构建步骤）。对话样本走分页接口，支持「加载更多」。

接口：
  GET /                         单页前端
  GET /api/jobs                 作业列表
  GET /api/job?id=              曲线 + 验证元信息（不含对话正文，payload 小）
  GET /api/samples?id=&vidx=&offset=&limit=   某次验证的对话样本（分页）

解析三类信息：
  1. 训练每步 reward 曲线（`Step N/总数` + `• Avg Reward` + `• Total step time`）
  2. 每次验证的准确率 / 平均 reward / 奖励分布（`Starting validation` 块）
  3. 每次验证打印的 N 条完整对话（USER / ASSISTANT / ENVIRONMENT 得分；N=num_val_samples_to_print）

用法：
    python scripts/web_dashboard.py --address http://192.168.1.4:8265 --port 8080 [--open]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
import webbrowser

import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from ray.job_submission import JobSubmissionClient

from web_log_parse import parse_logs

# ----------------------------- 实验配置解析（defaults 继承 + _override_）-----------------------------
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EXP_DIR = os.path.join(_REPO, "experiments")
_cfg_cache: dict[str, tuple[float, dict]] = {}  # config.yaml 路径 -> (mtime, 解析结果)


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base, over):
    """深合并；over 覆盖 base。over 是带 `_override_: true` 的 dict 时整段替换（剥掉该标记）。"""
    if isinstance(over, dict) and over.get("_override_"):
        return {k: v for k, v in over.items() if k != "_override_"}
    if isinstance(base, dict) and isinstance(over, dict):
        out = dict(base)
        for k, v in over.items():
            if k == "_override_":
                continue
            out[k] = _deep_merge(out.get(k), v) if k in out else v
        return out
    return over


def _resolve(path: str) -> dict:
    """加载 yaml 并按 `defaults` 顺序合并（相对本文件路径），最后叠加自身键。"""
    data = _load_yaml(path)
    defaults = data.pop("defaults", []) or []
    merged: dict = {}
    base_dir = os.path.dirname(path)
    for d in defaults:
        if not isinstance(d, str):
            continue
        dp = os.path.normpath(os.path.join(base_dir, d))
        if not dp.endswith((".yaml", ".yml")):
            dp += ".yaml"
        if os.path.isfile(dp):
            merged = _deep_merge(merged, _resolve(dp))
    return _deep_merge(merged, data)


def _flatten(d: dict, prefix: str = "") -> dict:
    out: dict = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def resolve_exp_config(exp: str) -> dict | None:
    """解析某实验的 config.yaml（含 defaults 继承）。无此实验返回 None。带 mtime 缓存。"""
    if not exp:
        return None
    path = os.path.join(_EXP_DIR, exp, "config.yaml")
    if not os.path.isfile(path):
        return None
    mtime = os.path.getmtime(path)
    hit = _cfg_cache.get(path)
    if hit and hit[0] == mtime:
        return hit[1]
    try:
        resolved = _resolve(path)
    except Exception:  # noqa: BLE001
        return None
    _cfg_cache[path] = (mtime, resolved)
    return resolved

# ----------------------------- Ray 取数（带短缓存）-----------------------------
_TERMINAL = {"FAILED", "SUCCEEDED", "STOPPED"}
_EXP_RE = re.compile(r"experiments/([^/\s]+)")


def _exp_name(entrypoint: str) -> str:
    """从 entrypoint 里提取实验名（experiments/<name>/...），取不到就回退到末段路径。"""
    if not entrypoint:
        return "-"
    m = _EXP_RE.search(entrypoint)
    if m:
        return m.group(1)
    # 回退：找形如 .../xxx/run.py 的目录名
    for tok in entrypoint.split():
        if "/run.py" in tok or tok.endswith(".py"):
            parts = tok.strip().split("/")
            return parts[-2] if len(parts) >= 2 else parts[-1]
    return entrypoint[:32]


def _summary(steps: list, vmeta: list) -> dict:
    """从曲线 + 验证元信息计算「一眼看懂」的关键指标：基线/最终准确率与 reward 增量。"""
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


class DataSource:
    """封装 JobSubmissionClient，对作业列表与日志做短 TTL 缓存以免频繁打 dashboard。"""

    def __init__(self, address: str, ttl: float = 5.0):
        self.address = address
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
            rows.append(
                {
                    "id": jid,
                    "exp": _exp_name(j.entrypoint or ""),
                    "status": status,
                    "entrypoint": (j.entrypoint or "")[:70],
                    "start": _fmt_start(j.start_time),
                    "dur": _fmt_dur(j.start_time, j.end_time),
                    "running": status not in _TERMINAL,
                }
            )
        with self._lock:
            self._jobs_cache = (now, rows)
        return rows

    def _parsed(self, job_id: str) -> dict:
        """取并解析作业日志（带 TTL 缓存）。返回 {steps, validations}。"""
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
        """曲线 + 验证元信息 + 关键指标摘要（不含对话正文，便于前端轻量轮询与对比）。"""
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
        cfg = resolve_exp_config(exp)
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
        """解析每个作业对应实验的 config.yaml，逐字段对比，标记差异。"""
        cols, flats = [], []
        for jid in ids:
            exp = self._exp_for(jid)
            cfg = resolve_exp_config(exp)
            flat = _flatten(cfg) if cfg else {}
            cols.append({"id": jid, "exp": exp,
                         "model": flat.get("policy.model_name"),
                         "has_config": bool(cfg)})
            flats.append(flat)
        keys = sorted(set().union(*[set(f) for f in flats])) if flats else []
        rows = []
        n_both_diff = n_one_sided = 0
        for k in keys:
            vals = [f.get(k, None) if k in f else "__MISSING__" for f in flats]
            missing = any(v == "__MISSING__" for v in vals)
            norm = [json.dumps(v, ensure_ascii=False, sort_keys=True) for v in vals]
            differ = len(set(norm)) > 1
            # kind: one_sided=有一方未声明(无此项，落框架默认，是否不同未知)；
            #       both_diff=两边都声明且值不同(确定差异)；both_same=两边都声明且相同
            if missing:
                kind = "one_sided"
                n_one_sided += 1
            elif differ:
                kind = "both_diff"
                n_both_diff += 1
            else:
                kind = "both_same"
            rows.append({"key": k, "values": vals, "diff": differ, "kind": kind})
        return {"jobs": cols, "rows": rows,
                "n_both_diff": n_both_diff, "n_one_sided": n_one_sided, "n_total": len(rows)}

    def stop_job(self, job_id: str) -> dict:
        """停止运行中的作业（SIGTERM），并失效作业列表缓存。"""
        try:
            ok = self._client.stop_job(job_id)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"停止失败：{e}")
        with self._lock:
            self._jobs_cache = None
        return {"ok": bool(ok), "id": job_id}

    def delete_job(self, job_id: str) -> dict:
        """删除已结束（FAILED/SUCCEEDED/STOPPED）的作业；运行中会被 Ray 拒绝。"""
        try:
            self._client.delete_job(job_id)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"删除失败（仅能删已结束作业，运行中请先停止）：{e}")
        with self._lock:
            self._jobs_cache = None
            self._log_cache.pop(job_id, None)
        return {"ok": True, "id": job_id}

    def samples_page(self, job_id: str, vidx: int, offset: int, limit: int) -> dict:
        """某次验证（按验证序号 vidx）的对话样本分页。"""
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


# ----------------------------- 前端（内嵌单页）-----------------------------
INDEX_HTML = r"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>NeMo-RL Lab · 训练面板</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600;700&family=Fira+Sans:wght@300;400;500;600;700&display=swap');
  :root{
    --bg:#0a0c11; --panel:#12151d; --panel2:#181c27; --panel3:#1f2433; --line:#262b3a; --line2:#323a4e;
    --fg:#e8ebf2; --muted:#8a93a6; --muted2:#646d80;
    --accent:#5b9dff; --accent2:#3b82f6; --amber:#f59e0b; --good:#3ecf8e; --bad:#ff6b6b; --warn:#f5b454;
    --think:#8a93a6; --tool:#f59e0b; --tooresp:#3ecf8e; --ans:#5b9dff;
    --sans:"Fira Sans",-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;
    --mono:"Fira Code",ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.55 var(--sans)}
  header{display:flex;align-items:center;gap:14px;padding:11px 18px;border-bottom:1px solid var(--line);background:linear-gradient(180deg,#141824,#10131b);position:sticky;top:0;z-index:50}
  header h1{font-size:15px;margin:0;font-weight:700;letter-spacing:.3px;white-space:nowrap}
  header h1 span{color:var(--muted);font-weight:400}
  .grow{flex:1}
  .tabs{display:flex;gap:4px;background:var(--panel2);border:1px solid var(--line);border-radius:10px;padding:3px}
  .tab{background:transparent;border:none;color:var(--muted);border-radius:7px;padding:6px 14px;font:600 13px var(--sans);cursor:pointer;transition:color .2s,background .2s}
  .tab:hover{color:var(--fg)}
  .tab.active{background:var(--accent2);color:#fff}
  button{background:var(--panel2);color:var(--fg);border:1px solid var(--line);border-radius:8px;padding:7px 10px;font:13px var(--sans);cursor:pointer;transition:border-color .2s,background .2s}
  button:hover{border-color:var(--accent)}
  /* 作业选择器（替代原生 select） */
  .jobpick{position:relative;flex:1;min-width:280px;max-width:640px}
  .jobpick-btn{width:100%;display:flex;align-items:center;gap:10px;padding:9px 12px;text-align:left;background:var(--panel2);border:1px solid var(--line);border-radius:10px;cursor:pointer;transition:border-color .2s,box-shadow .2s}
  .jobpick-btn:hover,.jobpick.open .jobpick-btn{border-color:var(--accent);box-shadow:0 0 0 2px rgba(91,157,255,.12)}
  .jobpick-btn .jpbody{flex:1;min-width:0;display:flex;align-items:center;gap:10px;flex-wrap:wrap}
  .jobpick-btn .jpexp{font-weight:600;font-size:13px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:360px}
  .jobpick-btn .jpmeta{font:11px var(--mono);color:var(--muted2)}
  .jobpick-chev{width:16px;height:16px;flex:none;color:var(--muted);transition:transform .2s}
  .jobpick.open .jobpick-chev{transform:rotate(180deg);color:var(--accent)}
  .jobpick-menu{position:absolute;top:calc(100% + 6px);left:0;right:0;z-index:40;background:var(--panel);border:1px solid var(--line2);border-radius:12px;box-shadow:0 12px 40px rgba(0,0,0,.45);overflow:hidden}
  .jobpick-menu[hidden]{display:none}
  .jobpick-search{width:100%;border:none;border-bottom:1px solid var(--line);background:var(--panel2);color:var(--fg);padding:10px 12px;font:13px var(--sans);outline:none}
  .jobpick-search::placeholder{color:var(--muted2)}
  .jobpick-search:focus{background:var(--panel3)}
  .jobpick-list{max-height:min(360px,50vh);overflow-y:auto;padding:6px}
  .jobpick-item{display:flex;align-items:center;gap:10px;width:100%;padding:9px 10px;border:none;border-radius:8px;background:transparent;color:var(--fg);text-align:left;cursor:pointer;transition:background .15s}
  .jobpick-item:hover{background:var(--panel2)}
  .jobpick-item.on{background:rgba(91,157,255,.12);outline:1px solid rgba(91,157,255,.35)}
  .jobpick-item .jibody{flex:1;min-width:0}
  .jobpick-item .jiexp{font-weight:600;font-size:13px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .jobpick-item .jimeta{font:11px var(--mono);color:var(--muted2);margin-top:3px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .jobpick-empty{padding:20px;text-align:center;color:var(--muted);font-size:13px}
  .stbadge{font:600 10px var(--mono);padding:2px 7px;border-radius:5px;border:1px solid var(--line2);white-space:nowrap;text-transform:uppercase;letter-spacing:.3px}
  .st-run{color:var(--good);border-color:rgba(62,207,142,.35);background:rgba(62,207,142,.1)}
  .st-ok{color:var(--accent);border-color:rgba(91,157,255,.35);background:rgba(91,157,255,.1)}
  .st-stop{color:var(--muted);border-color:var(--line2);background:var(--panel2)}
  .st-fail{color:var(--bad);border-color:rgba(255,107,107,.35);background:rgba(255,107,107,.1)}
  .st-pend{color:var(--warn);border-color:rgba(245,180,84,.35);background:rgba(245,180,84,.1)}
  button.warn{border-color:rgba(245,180,84,.4);color:var(--warn)}
  button.warn:hover{border-color:var(--warn);background:rgba(245,180,84,.1)}
  button.danger{border-color:rgba(255,107,107,.4);color:var(--bad)}
  button.danger:hover{border-color:var(--bad);background:rgba(255,107,107,.1)}
  main{padding:18px;max-width:1240px;margin:0 auto}
  .panel{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:16px;margin-bottom:16px}
  .panel h2{font-size:12px;margin:0 0 12px;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.7px;display:flex;align-items:center;gap:8px}
  .panel h2 .hint{font-size:11px;font-weight:400;text-transform:none;letter-spacing:0;color:var(--muted2)}
  svg{width:100%;display:block}
  .legend{display:flex;flex-wrap:wrap;gap:16px;font-size:12px;color:var(--muted);margin-top:10px}
  .legend i{width:18px;height:3px;display:inline-block;vertical-align:middle;margin-right:5px;border-radius:2px}
  /* 选择 chips */
  .chips{display:flex;flex-wrap:wrap;gap:9px}
  .chip{display:flex;align-items:center;gap:9px;background:var(--panel2);border:1px solid var(--line);border-radius:10px;padding:8px 12px;cursor:pointer;transition:border-color .2s,background .2s;max-width:340px}
  .chip:hover{border-color:var(--line2)}
  .chip.on{border-color:var(--accent);background:rgba(91,157,255,.10)}
  .chip .cdot{width:10px;height:10px;border-radius:3px;flex:none;background:var(--muted2)}
  .chip.on .cdot{box-shadow:0 0 0 2px rgba(255,255,255,.08)}
  .chip .ctxt{min-width:0;flex:1}
  .chip .cn{font-weight:600;font-size:12.5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .chip .cm{font:11px var(--mono);color:var(--muted2);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .chipact{flex:none;font-size:11px;padding:3px 9px;border-radius:6px}
  .chiptoggle{margin-top:10px}
  .chiptoggle button{font-size:12px;padding:5px 12px}
  .chip.hide{display:none}
  .live{width:7px;height:7px;border-radius:50%;background:var(--good);display:inline-block;box-shadow:0 0 6px var(--good)}
  /* 结论 banner */
  .verdict{border:1px solid var(--line);border-left:3px solid var(--accent2);background:linear-gradient(90deg,rgba(59,130,246,.08),transparent 60%);border-radius:12px;padding:14px 16px;margin-bottom:16px}
  .verdict .vhead{font-size:14px;font-weight:600;margin-bottom:10px;display:flex;align-items:center;gap:10px;flex-wrap:wrap}
  .verdict .vlines{display:flex;flex-direction:column;gap:7px}
  .vline{display:flex;align-items:center;gap:10px;font-size:13px;flex-wrap:wrap}
  .vline .vn{font-weight:600;display:flex;align-items:center;gap:7px;min-width:200px}
  .vline .vflow{font:12.5px var(--mono);color:var(--muted)}
  .tag{font-size:11px;font-weight:600;padding:2px 8px;border-radius:6px;border:1px solid var(--line2);color:var(--muted)}
  .tag.best{color:var(--amber);border-color:rgba(245,158,11,.4);background:rgba(245,158,11,.12)}
  /* KPI cards */
  .kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px;margin-bottom:16px}
  .kpi{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:15px 16px;border-top:3px solid var(--accent);cursor:pointer;transition:transform .15s,border-color .2s}
  .kpi:hover{border-color:var(--line2)}
  .kpi .kt{display:flex;align-items:center;gap:8px;font-weight:600;font-size:13px;margin-bottom:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .kpi .kbig{font:700 30px var(--mono);line-height:1;letter-spacing:-.5px}
  .kpi .ksub{display:flex;align-items:baseline;gap:10px;margin-top:8px;color:var(--muted);font-size:12px}
  .kpi .krow{display:flex;justify-content:space-between;font-size:11.5px;color:var(--muted2);margin-top:10px;padding-top:9px;border-top:1px solid var(--line);font-family:var(--mono)}
  /* delta badge */
  .db{font:600 12px var(--mono);padding:2px 7px;border-radius:6px;white-space:nowrap}
  .db.up{color:var(--good);background:rgba(62,207,142,.12);border:1px solid rgba(62,207,142,.3)}
  .db.down{color:var(--bad);background:rgba(255,107,107,.12);border:1px solid rgba(255,107,107,.3)}
  .db.flat{color:var(--muted);background:var(--panel2);border:1px solid var(--line)}
  /* 明细表 */
  .tablewrap{overflow-x:auto}
  table{width:100%;border-collapse:collapse;font-size:13px}
  th,td{text-align:right;padding:9px 12px;border-bottom:1px solid var(--line);white-space:nowrap}
  th{color:var(--muted);font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.5px}
  th:first-child,td:first-child{text-align:left}
  td.mono{font-family:var(--mono)}
  tbody tr{cursor:pointer;transition:background .15s}
  tbody tr:hover{background:var(--panel2)}
  .cdotinline{width:9px;height:9px;border-radius:3px;display:inline-block;margin-right:7px;vertical-align:middle}
  .modeltag{font:11px var(--mono);color:var(--muted);background:var(--panel2);border:1px solid var(--line);border-radius:5px;padding:1px 6px;white-space:nowrap}
  /* 配置差异表 */
  #diffTable th,#diffTable td{text-align:left}
  #diffTable th{vertical-align:bottom}
  td.dk{font:12px var(--mono);color:var(--muted);max-width:300px;overflow:hidden;text-overflow:ellipsis}
  td.dv{font:12px var(--mono);max-width:280px;overflow:hidden;text-overflow:ellipsis}
  td.miss{color:var(--muted2);font-style:italic}
  tr.drow td{background:rgba(245,158,11,.06)}
  tr.drow td.dk{border-left:3px solid var(--amber);color:var(--amber);font-weight:600}
  tr.drow td.dv{color:#ffd591}
  tr.orow td.dk{border-left:3px solid var(--line2);color:var(--muted)}
  .difftools{display:flex;align-items:center;gap:12px}
  /* 详情：stat cards */
  .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:16px}
  .stat{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
  .stat .k{color:var(--muted);font-size:12px;margin-bottom:6px}
  .stat .v{font:700 22px var(--mono)}
  .valbar{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:14px}
  .valbar button{padding:6px 12px;font-size:12px}
  .valbar button.active{border-color:var(--accent);background:rgba(91,157,255,.14);color:var(--accent)}
  .sample{border:1px solid var(--line);border-radius:12px;margin-bottom:14px;overflow:hidden;background:var(--panel2)}
  .sample .sh{display:flex;align-items:center;gap:10px;padding:10px 14px;border-bottom:1px solid var(--line);background:rgba(255,255,255,.02)}
  .badge{font:700 12px var(--mono);padding:3px 9px;border-radius:6px}
  .b-good{background:rgba(62,207,142,.15);color:var(--good);border:1px solid rgba(62,207,142,.3)}
  .b-bad{background:rgba(255,107,107,.15);color:var(--bad);border:1px solid rgba(255,107,107,.3)}
  .b-mid{background:rgba(245,180,84,.15);color:var(--warn);border:1px solid rgba(245,180,84,.3)}
  .seg{padding:12px 14px;border-bottom:1px solid var(--line)}
  .seg:last-child{border-bottom:none}
  .seg .lbl{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px}
  .seg pre{margin:0;white-space:pre-wrap;word-break:break-word;font:12.5px/1.6 var(--mono)}
  .seg.user pre{color:#c3cad8}
  .tok-think{color:var(--think);font-style:italic}
  .tok-tool{color:var(--tool);font-weight:700}
  .tok-resp{color:var(--tooresp);font-weight:700}
  .tok-ans{color:var(--ans);font-weight:700}
  .boxed{background:rgba(91,157,255,.16);color:#aecbff;border-radius:4px;padding:0 3px;font-weight:700}
  .collapsed pre{max-height:84px;overflow:hidden;-webkit-mask-image:linear-gradient(#000 50%,transparent)}
  .toggle{font-size:11px;color:var(--accent);cursor:pointer;margin-top:6px;display:inline-block}
  .empty{color:var(--muted);text-align:center;padding:40px}
  .pill{font-size:11px;color:var(--muted)}
  .updated{font-size:11px;color:var(--muted);font-family:var(--mono)}
  .more{display:flex;justify-content:center;margin:6px 0 2px}
  .more button{padding:8px 18px}
  .loading{opacity:.5;pointer-events:none}
  /* 自定义弹窗（替代原生 confirm/alert） */
  .dlg-overlay{position:fixed;inset:0;z-index:200;display:flex;align-items:center;justify-content:center;padding:20px;background:rgba(0,0,0,.55);backdrop-filter:blur(4px)}
  .dlg-overlay[hidden]{display:none}
  .dlg{width:min(420px,100%);background:var(--panel);border:1px solid var(--line2);border-radius:14px;box-shadow:0 20px 60px rgba(0,0,0,.5);overflow:hidden;animation:dlgIn .18s ease}
  @keyframes dlgIn{from{opacity:0;transform:scale(.96) translateY(8px)}to{opacity:1;transform:none}}
  .dlg-hd{padding:16px 18px 0;font-size:15px;font-weight:700}
  .dlg-bd{padding:12px 18px 18px;color:var(--muted);font-size:13px;line-height:1.6}
  .dlg-bd p{margin:0 0 10px}
  .dlg-meta{background:var(--panel2);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
  .dlg-meta strong{display:block;font-size:13px;color:var(--fg);margin-bottom:6px;word-break:break-word}
  .dlg-meta code{font:12px var(--mono);color:var(--muted2);word-break:break-all}
  .dlg-ft{display:flex;justify-content:flex-end;gap:10px;padding:0 18px 16px}
  .dlg-ft button{min-width:80px;padding:8px 16px}
  .dlg-ft .primary{background:var(--accent2);border-color:var(--accent2);color:#fff}
  .dlg-ft .primary:hover{background:#2563eb;border-color:#2563eb}
  @media (prefers-reduced-motion:reduce){*{transition:none!important}.dlg{animation:none}}
</style>
</head>
<body>
<header>
  <h1>NeMo-RL Lab <span>· 实验对比面板</span></h1>
  <div class="tabs">
    <button class="tab active" data-view="compare">对比总览</button>
    <button class="tab" data-view="detail">作业详情</button>
  </div>
  <span class="grow"></span>
  <button id="refreshBtn" title="立即刷新">↻ 刷新</button>
  <label class="pill"><input type="checkbox" id="autoChk" checked/> 自动刷新</label>
  <span class="updated" id="updated"></span>
</header>
<main>
  <!-- ================= 对比总览 ================= -->
  <section id="view-compare">
    <div class="panel">
      <h2>选择实验对比 <span class="hint">点选 2~4 个作业，自动对齐「基线 → 最终」准确率</span></h2>
      <div class="chips" id="cmpChips"></div>
      <div class="chiptoggle" id="cmpToggle"></div>
    </div>
    <div class="verdict" id="verdict"></div>
    <div class="kpis" id="kpis"></div>
    <div class="panel">
      <h2>验证准确率轨迹 <span class="hint">核心指标 · 线越往右上 = 进步越大</span></h2>
      <svg id="accChart" viewBox="0 0 1000 260" preserveAspectRatio="none"></svg>
      <div class="legend" id="accLegend"></div>
    </div>
    <div class="panel">
      <h2>训练 Reward 曲线对比 <span class="hint">每步 Avg Reward</span></h2>
      <svg id="rewChart" viewBox="0 0 1000 240" preserveAspectRatio="none"></svg>
      <div class="legend" id="rewLegend"></div>
    </div>
    <div class="panel">
      <h2>关键指标明细 <span class="hint">点任意一行下钻到该作业的验证对话</span></h2>
      <div class="tablewrap"><table id="cmpTable"></table></div>
    </div>
    <div class="panel">
      <h2>配置差异 (config.yaml)
        <span class="hint">解析 defaults 继承后逐字段对比 · <b style="color:var(--amber)">琥珀=双边确定不同</b> · <span style="color:var(--muted2)">灰=单边「无此项」（缺的一方用框架默认）</span></span>
        <span class="grow"></span>
        <span class="difftools">
          <span class="pill" id="diffStat"></span>
          <label class="pill"><input type="checkbox" id="diffOnly" checked/> 只看差异</label>
          <label class="pill"><input type="checkbox" id="hideMissing" checked/> 隐藏「无此项」</label>
        </span>
      </h2>
      <div class="tablewrap"><table id="diffTable"></table></div>
    </div>
  </section>

  <!-- ================= 作业详情 ================= -->
  <section id="view-detail" hidden>
    <div class="panel" style="display:flex;align-items:center;gap:12px">
      <span class="pill">作业</span>
      <div class="jobpick" id="jobPick">
        <button type="button" class="jobpick-btn" id="jobPickBtn" aria-haspopup="listbox" aria-expanded="false">
          <span class="jpbody" id="jobPickLabel"><span class="jpexp">选择作业…</span></span>
          <svg class="jobpick-chev" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M6 9l6 6 6-6"/></svg>
        </button>
        <div class="jobpick-menu" id="jobPickMenu" hidden>
          <input type="text" class="jobpick-search" id="jobPickSearch" placeholder="搜索实验名 / job id / 状态…" autocomplete="off"/>
          <div class="jobpick-list" id="jobPickList" role="listbox"></div>
        </div>
      </div>
      <span class="grow"></span>
      <button id="stopBtn" class="warn" hidden>停止</button>
      <button id="delBtn" class="danger" hidden>删除</button>
    </div>
    <div class="cards" id="stats"></div>
    <div class="panel">
      <h2>训练 reward 曲线（每步）+ 验证准确率</h2>
      <svg id="curve" viewBox="0 0 1000 240" preserveAspectRatio="none"></svg>
      <div class="legend">
        <span><i style="background:#5b9dff"></i>训练 Avg Reward</span>
        <span><i style="background:#3ecf8e"></i>验证 Accuracy</span>
      </div>
    </div>
    <div class="panel">
      <h2>验证对话样本 <span class="hint">含完整 model response · 分页加载</span></h2>
      <div class="valbar" id="valbar"></div>
      <div id="samples"></div>
      <div class="more" id="moreWrap"></div>
    </div>
  </section>
</main>
<div class="dlg-overlay" id="dlgOverlay" hidden>
  <div class="dlg" role="dialog" aria-modal="true" aria-labelledby="dlgTitle">
    <div class="dlg-hd" id="dlgTitle"></div>
    <div class="dlg-bd" id="dlgBody"></div>
    <div class="dlg-ft" id="dlgFt"></div>
  </div>
</div>
<script>
const $ = s => document.querySelector(s);
const PAGE = 6;                            // 每页对话条数
const PALETTE = ['#5b9dff','#f59e0b','#3ecf8e','#c084fc','#ff6b6b','#22d3ee'];
let JOBS=[];                               // /api/jobs
let OVS={};                                // 客户端缓存 jobId -> overview
let CMP=[];                                // 对比选中的 jobId（保序）
let VIEW='compare';
let CHIPS_OPEN=false;                      // 实验 chips 是否展开（默认只显示 2 行）
// 详情视图状态
let CUR_JOB=null, CUR_VAL=-1;
let LOADED=[], OFFSET=0, TOTAL=0, LOADING=false;

// --------- 工具 ---------
function rewardClass(r){ if(r>=0.999) return 'b-good'; if(r<0) return 'b-bad'; return 'b-mid'; }
function esc(s){ return (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
function pct(x){ return x==null? '—' : (x*100).toFixed(1)+'%'; }
function pp(d){ return (d>=0?'+':'')+(d*100).toFixed(1)+' pp'; }
function arrow(d){ return d>0.0005?'▲':(d<-0.0005?'▼':'—'); }
function dcls(d){ return d==null?'flat':(d>0.0005?'up':(d<-0.0005?'down':'flat')); }
function accBadge(d){ if(d==null) return '<span class="db flat">—</span>'; return `<span class="db ${dcls(d)}">${arrow(d)} ${pp(d)}</span>`; }
function rewBadge(d){ if(d==null) return '<span class="db flat">—</span>'; return `<span class="db ${dcls(d)}">${arrow(d)} ${(d>=0?'+':'')}${d.toFixed(3)}</span>`; }
function colorOf(id){ const i=CMP.indexOf(id); return PALETTE[(i<0?0:i)%PALETTE.length]; }
function expLabel(id){ const j=JOBS.find(x=>x.id===id); return j? j.exp : id.slice(0,12); }
function highlight(text){
  let h = esc(text);
  h = h.replace(/&lt;\/?think&gt;/g, m=>`<span class="tok-think">${m}</span>`);
  h = h.replace(/&lt;\/?tool_call&gt;/g, m=>`<span class="tok-tool">${m}</span>`);
  h = h.replace(/&lt;\/?(tool_response|search_result)&gt;/g, m=>`<span class="tok-resp">${m}</span>`);
  h = h.replace(/&lt;\/?(answer|search)&gt;/g, m=>`<span class="tok-ans">${m}</span>`);
  h = h.replace(/\\boxed\{[^}]*\}/g, m=>`<span class="boxed">${m}</span>`);
  return h;
}

// --------- 自定义弹窗（替代原生 confirm / alert）---------
let _dlgResolve=null;
function closeDlg(result){
  $('#dlgOverlay').hidden=true;
  document.body.style.overflow='';
  if(_dlgResolve){ const r=_dlgResolve; _dlgResolve=null; r(result); }
}
function showDlg({title, body, buttons}){
  return new Promise(resolve=>{
    _dlgResolve=resolve;
    $('#dlgTitle').textContent=title;
    $('#dlgBody').innerHTML=body;
    $('#dlgFt').innerHTML=buttons.map(b=>`<button type="button" class="${b.cls||''}" data-act="${b.act}">${esc(b.label)}</button>`).join('');
    $('#dlgFt').querySelectorAll('button').forEach(btn=>{
      btn.onclick=()=>closeDlg(btn.dataset.act==='confirm'||btn.dataset.act==='ok');
    });
    $('#dlgOverlay').hidden=false;
    document.body.style.overflow='hidden';
    const primary=$('#dlgFt').querySelector('.primary,.danger,.warn')||$('#dlgFt').querySelector('button');
    if(primary) setTimeout(()=>primary.focus(), 30);
  });
}
function dlgConfirm(title, body, {variant='primary', okLabel='确认'}={}){
  const cls=variant==='danger'?'danger':(variant==='warn'?'warn':'primary');
  return showDlg({title, body, buttons:[{label:'取消',act:'cancel'},{label:okLabel,act:'confirm',cls}]}).then(r=>r===true);
}
function dlgAlert(title, body, {error=false}={}){
  return showDlg({title, body, buttons:[{label:'知道了',act:'ok',cls:error?'danger':''}]});
}

// --------- 数据 ---------
async function loadJobs(){
  const r = await fetch('/api/jobs'); JOBS = await r.json();
  if(!CUR_JOB && JOBS.length){ CUR_JOB = (JOBS.find(j=>j.running)||JOBS[0]).id; }
  if(!CMP.length){
    // 默认：每个实验取最近一条，最多 2 个，便于一眼对比
    const seen=new Set(); for(const j of JOBS){ if(!seen.has(j.exp)){ seen.add(j.exp); CMP.push(j.id); } if(CMP.length>=2) break; }
  }
}
async function ensureOverview(id, force){
  if(!force && OVS[id]) return OVS[id];
  const r = await fetch('/api/job?id='+encodeURIComponent(id));
  OVS[id] = await r.json(); return OVS[id];
}
async function refresh(){
  await loadJobs();
  $('#updated').textContent = '更新于 '+new Date().toLocaleTimeString();
  if(VIEW==='compare'){ await renderCompare(true); }
  else { await loadDetail(true); }
}

// --------- 视图切换 ---------
function switchView(v){
  VIEW=v;
  document.querySelectorAll('.tab').forEach(t=>t.classList.toggle('active', t.dataset.view===v));
  $('#view-compare').hidden = v!=='compare';
  $('#view-detail').hidden = v!=='detail';
  if(v==='compare') renderCompare(true); else loadDetail(true);
}

// ========================= 对比总览 =========================
function renderChips(){
  // 选中的排前面（保证默认两行内一定能看到），其余按时间序
  const ordered = [...JOBS].sort((a,b)=> (CMP.includes(b.id)?1:0)-(CMP.includes(a.id)?1:0));
  $('#cmpChips').innerHTML = ordered.map(j=>{
    const on = CMP.includes(j.id);
    const col = on? colorOf(j.id) : 'var(--muted2)';
    const act = j.running
      ? `<button class="chipact warn" onclick="event.stopPropagation();jobAction('stop','${j.id}')">停止</button>`
      : `<button class="chipact danger" onclick="event.stopPropagation();jobAction('delete','${j.id}')">删除</button>`;
    return `<div class="chip ${on?'on':''}" onclick="toggleCmp('${j.id}')">
      <span class="cdot" style="background:${col}"></span>
      <span class="ctxt"><span class="cn">${j.running?'<span class=live></span> ':''}${esc(j.exp)}</span>
      <span class="cm">${j.id.slice(0,18)} · ${j.status} · ${j.dur}</span></span>${act}</div>`;
  }).join('');
  applyChipRows();
}
// 折叠到 2 行：按实际换行位置（offsetTop）判断行，超出的隐藏
function applyChipRows(){
  const cont=$('#cmpChips'); const toggle=$('#cmpToggle');
  const chips=[...cont.querySelectorAll('.chip')];
  chips.forEach(c=>c.classList.remove('hide'));
  if(CHIPS_OPEN){
    toggle.innerHTML = chips.length>0? `<button onclick="toggleChips()">收起</button>` : '';
    return;
  }
  const tops=[]; chips.forEach(c=>{ if(!tops.includes(c.offsetTop)) tops.push(c.offsetTop); });
  const allow=tops.slice(0,2);
  let hidden=0;
  chips.forEach(c=>{ if(!allow.includes(c.offsetTop)){ c.classList.add('hide'); hidden++; } });
  toggle.innerHTML = hidden>0? `<button onclick="toggleChips()">展开全部（共 ${chips.length} 个，还有 ${hidden} 个）</button>` : '';
}
function toggleChips(){ CHIPS_OPEN=!CHIPS_OPEN; applyChipRows(); }
function toggleCmp(id){
  const i=CMP.indexOf(id);
  if(i>=0) CMP.splice(i,1);
  else { if(CMP.length>=4){ dlgAlert('已达上限','最多同时对比 <b>4</b> 个作业，请先取消一个再选。'); return; } CMP.push(id); }
  renderCompare(false);
}

async function renderCompare(refetch){
  renderChips();
  const sel = CMP.slice();
  if(!sel.length){ $('#verdict').innerHTML=''; $('#kpis').innerHTML='<div class="empty">上面点选至少一个作业开始对比。</div>'; $('#accChart').innerHTML=''; $('#rewChart').innerHTML=''; $('#cmpTable').innerHTML=''; $('#accLegend').innerHTML=''; $('#rewLegend').innerHTML=''; return; }
  await Promise.all(sel.map(id=>ensureOverview(id, refetch)));
  const items = sel.map(id=>({id, exp:expLabel(id), color:colorOf(id), ov:OVS[id], s:(OVS[id]||{}).summary||{}, model:(OVS[id]||{}).model}));
  renderVerdict(items); renderKpis(items); renderAccChart(items); renderRewChart(items); renderCmpTable(items);
  renderDiff(sel);
}
function modelShort(m){ if(!m) return '—'; const p=String(m).split('/'); return p[p.length-1]; }
function statusBadge(st, running){
  const map={RUNNING:'st-run',SUCCEEDED:'st-ok',STOPPED:'st-stop',FAILED:'st-fail',PENDING:'st-pend'};
  const cls=map[st]||'st-stop';
  return `<span class="stbadge ${cls}">${running?'<span class="live"></span> ':''}${esc(st)}</span>`;
}
function jobPickLabel(j){
  if(!j) return '<span class="jpexp">选择作业…</span>';
  return `<span class="jpexp" title="${esc(j.exp)}">${esc(j.exp)}</span>${statusBadge(j.status,j.running)}<span class="jpmeta">${esc(j.id.slice(0,18))} · ${esc(j.dur)}</span>`;
}

function renderVerdict(items){
  const withAcc = items.filter(it=>it.s.final_acc!=null);
  let head = '<span>结论</span>';
  if(withAcc.length){
    const best = withAcc.reduce((a,b)=> b.s.final_acc>a.s.final_acc? b:a);
    head = `<span>最终准确率最高：</span><span class="tag best" style="border-color:${best.color}66;color:${best.color}">${esc(best.exp)} · ${pct(best.s.final_acc)}</span>`;
    if(withAcc.length===2){
      const [a,b]=withAcc; const hi=a.s.final_acc>=b.s.final_acc?a:b, lo=hi===a?b:a;
      const gap=hi.s.final_acc-lo.s.final_acc;
      head += `<span class="vflow">　${esc(hi.exp)} 比 ${esc(lo.exp)} 高 ${(gap*100).toFixed(1)} pp</span>`;
    }
  }
  const lines = items.map(it=>{
    const s=it.s;
    const flow = (s.base_acc!=null&&s.final_acc!=null)? `基线 ${pct(s.base_acc)} → 最终 ${pct(s.final_acc)}` : '（暂无验证准确率）';
    const verdictTag = s.delta_acc==null? '' : (s.delta_acc>0.0005? '<span class="tag" style="color:var(--good);border-color:rgba(62,207,142,.4)">进步</span>' : (s.delta_acc<-0.0005? '<span class="tag" style="color:var(--bad);border-color:rgba(255,107,107,.4)">退步</span>' : '<span class="tag">持平</span>'));
    return `<div class="vline"><span class="vn"><span class="cdotinline" style="background:${it.color}"></span>${esc(it.exp)}</span>
      <span class="vflow">${flow}</span>${accBadge(s.delta_acc)}${verdictTag}</div>`;
  }).join('');
  $('#verdict').innerHTML = `<div class="vhead">${head}</div><div class="vlines">${lines}</div>`;
}

function renderKpis(items){
  $('#kpis').innerHTML = items.map(it=>{
    const s=it.s;
    return `<div class="kpi" style="border-top-color:${it.color}" onclick="drillTo('${it.id}')" title="点击下钻详情">
      <div class="kt"><span class="cdotinline" style="background:${it.color}"></span>${esc(it.exp)}</div>
      <div style="margin:-6px 0 10px"><span class="modeltag" title="${esc(it.model||'')}">${esc(modelShort(it.model))}</span></div>
      <div class="kbig">${pct(s.final_acc)}</div>
      <div class="ksub"><span>基线 ${pct(s.base_acc)}</span>${accBadge(s.delta_acc)}</div>
      <div class="krow"><span>reward ${s.reward_first==null?'—':s.reward_first.toFixed(2)}→${s.reward_last==null?'—':s.reward_last.toFixed(2)}</span><span>step ${s.last_step==null?'—':s.last_step}/${s.total==null?'—':s.total}</span></div>
    </div>`;
  }).join('');
}

// 通用多序列折线图
function multiline(svgId, H, series, ylo, yhi, yfmt, maxX, emptyMsg){
  const svg=$(svgId), W=1000,padL=46,padR=16,padT=14,padB=26;
  if(!series.some(s=>s.pts.length)){ svg.setAttribute('viewBox',`0 0 ${W} ${H}`); svg.innerHTML=`<text x="20" y="34" fill="#8a93a6">${emptyMsg}</text>`; return; }
  svg.setAttribute('viewBox',`0 0 ${W} ${H}`);
  const px=x=> padL+(x/Math.max(maxX,1))*(W-padL-padR);
  const py=v=> padT+(1-(v-ylo)/((yhi-ylo)||1))*(H-padT-padB);
  const grid=[ylo,(ylo+yhi)/2,yhi].map(t=>`<text x="8" y="${(py(t)+4).toFixed(1)}" fill="#646d80" font-size="11" font-family="Fira Code,monospace">${yfmt(t)}</text><line x1="${padL}" y1="${py(t).toFixed(1)}" x2="${W-padR}" y2="${py(t).toFixed(1)}" stroke="#1d2230"/>`).join('');
  const body=series.map(s=>{
    if(!s.pts.length) return '';
    const path=s.pts.map((p,i)=>`${i?'L':'M'}${px(p.x).toFixed(1)},${py(p.y).toFixed(1)}`).join(' ');
    const dots=s.pts.map(p=>`<circle cx="${px(p.x).toFixed(1)}" cy="${py(p.y).toFixed(1)}" r="${s.pts.length<=12?3.5:2}" fill="${s.color}"><title>${esc(s.label)} · step ${p.x}: ${p.t}</title></circle>`).join('');
    return `<path d="${path}" fill="none" stroke="${s.color}" stroke-width="2"/>${dots}`;
  }).join('');
  svg.innerHTML = `${grid}${body}<text x="${padL}" y="${H-7}" fill="#646d80" font-size="11" font-family="Fira Code,monospace">step 0</text><text x="${W-padR-34}" y="${H-7}" fill="#646d80" font-size="11" font-family="Fira Code,monospace">${maxX}</text>`;
}
function legendHtml(items){ return items.map(it=>`<span><i style="background:${it.color}"></i>${esc(it.exp)}</span>`).join(''); }

function renderAccChart(items){
  const maxX=Math.max(1,...items.flatMap(it=>(it.ov.validations||[]).map(v=>v.step)));
  const all=items.flatMap(it=>(it.ov.validations||[]).filter(v=>v.accuracy!=null).map(v=>v.accuracy));
  let lo=Math.min(...all,1), hi=Math.max(...all,0); if(!all.length){lo=0;hi=1;}
  lo=Math.max(0,Math.floor(lo*10)/10-0.05); hi=Math.min(1,Math.ceil(hi*10)/10+0.05); if(hi<=lo)hi=lo+0.1;
  const series=items.map(it=>({label:it.exp,color:it.color,pts:(it.ov.validations||[]).filter(v=>v.accuracy!=null).map(v=>({x:v.step,y:v.accuracy,t:(v.accuracy*100).toFixed(1)+'%'}))}));
  multiline('#accChart',260,series,lo,hi,t=>(t*100).toFixed(0)+'%',maxX,'暂无验证准确率（等首次验证完成）');
  $('#accLegend').innerHTML=legendHtml(items);
}
function renderRewChart(items){
  const maxX=Math.max(1,...items.flatMap(it=>(it.ov.steps||[]).map(s=>s.step)));
  const all=items.flatMap(it=>(it.ov.steps||[]).filter(s=>s.avg_reward!=null).map(s=>s.avg_reward));
  let lo=Math.min(0,...all), hi=Math.max(1,...all); if(!all.length){lo=0;hi=1;}
  const series=items.map(it=>({label:it.exp,color:it.color,pts:(it.ov.steps||[]).filter(s=>s.avg_reward!=null).map(s=>({x:s.step,y:s.avg_reward,t:s.avg_reward.toFixed(3)}))}));
  multiline('#rewChart',240,series,lo,hi,t=>t.toFixed(1),maxX,'暂无训练步数据');
  $('#rewLegend').innerHTML=legendHtml(items);
}

function renderCmpTable(items){
  const head=`<thead><tr><th>实验</th><th>模型</th><th>步数</th><th>基线 Acc</th><th>最终 Acc</th><th>Δ Acc</th><th>起 Reward</th><th>末 Reward</th><th>Δ Reward</th></tr></thead>`;
  const rows=items.map(it=>{const s=it.s; return `<tr onclick="drillTo('${it.id}')">
    <td><span class="cdotinline" style="background:${it.color}"></span>${esc(it.exp)}</td>
    <td><span class="modeltag" title="${esc(it.model||'')}">${esc(modelShort(it.model))}</span></td>
    <td class="mono">${s.last_step==null?'—':s.last_step}/${s.total==null?'—':s.total}</td>
    <td class="mono">${pct(s.base_acc)}</td>
    <td class="mono" style="font-weight:700">${pct(s.final_acc)}</td>
    <td>${accBadge(s.delta_acc)}</td>
    <td class="mono">${s.reward_first==null?'—':s.reward_first.toFixed(3)}</td>
    <td class="mono">${s.reward_last==null?'—':s.reward_last.toFixed(3)}</td>
    <td>${rewBadge(s.delta_reward)}</td></tr>`;}).join('');
  $('#cmpTable').innerHTML=head+`<tbody>${rows}</tbody>`;
}

// ---- 配置差异 ----
let DIFF=null;
async function renderDiff(ids){
  if(!ids.length){ DIFF=null; $('#diffTable').innerHTML=''; $('#diffStat').textContent=''; return; }
  try{
    const r=await fetch('/api/diff?ids='+encodeURIComponent(ids.join(',')));
    DIFF=await r.json();
  }catch(e){ $('#diffTable').innerHTML=`<tbody><tr><td class="miss">配置解析失败：${esc(String(e))}</td></tr></tbody>`; return; }
  drawDiff();
}
function diffCell(v){
  if(v==='__MISSING__') return `<td class="dv miss">无此项</td>`;
  let s = (v===null||v===undefined)? 'null' : (typeof v==='object'? JSON.stringify(v) : String(v));
  return `<td class="dv" title="${esc(s)}">${esc(s)}</td>`;
}
function drawDiff(){
  if(!DIFF){ return; }
  const onlyDiff=$('#diffOnly').checked, hideMiss=$('#hideMissing').checked;
  const cols=DIFF.jobs;
  const noCfg=cols.some(c=>!c.has_config);
  $('#diffStat').innerHTML = `<b style="color:var(--amber)">双边差异 ${DIFF.n_both_diff}</b> · 单边「无此项」 ${DIFF.n_one_sided} · 共 ${DIFF.n_total} 项`;
  const head=`<thead><tr><th>配置字段</th>${cols.map(c=>{
    const col=colorOf(c.id);
    const m=c.model? `<div class="modeltag" title="${esc(c.model)}" style="margin-top:4px">${esc(modelShort(c.model))}</div>`:'';
    return `<th><span class="cdotinline" style="background:${col}"></span>${esc(c.exp)}${c.has_config?'':' <span class="miss">(非实验作业)</span>'}${m}</th>`;
  }).join('')}</tr></thead>`;
  let rows=DIFF.rows;
  if(onlyDiff) rows=rows.filter(r=>r.kind!=='both_same');   // 隐藏两边相同
  if(hideMiss) rows=rows.filter(r=>r.kind!=='one_sided');   // 隐藏单边「无此项」
  if(!rows.length){
    const tip = hideMiss && DIFF.n_one_sided>0
      ? `两边都显式声明且不同的字段为 0；另有 ${DIFF.n_one_sided} 个「单边声明」字段（取消勾选「隐藏无此项」可查看）。`
      : (noCfg?'选中的作业里有非实验作业（无 config.yaml）。':'所选作业配置完全一致，无差异字段。');
    $('#diffTable').innerHTML=head+`<tbody><tr><td class="miss" colspan="${cols.length+1}">${tip}</td></tr></tbody>`;
    return;
  }
  const cls={both_diff:'drow', one_sided:'orow', both_same:''};
  const body=rows.map(r=>`<tr class="${cls[r.kind]||''}"><td class="dk" title="${esc(r.key)}">${esc(r.key)}</td>${r.values.map(diffCell).join('')}</tr>`).join('');
  $('#diffTable').innerHTML=head+`<tbody>${body}</tbody>`;
}

function drillTo(id){ CUR_JOB=id; CUR_VAL=-1; switchView('detail'); syncJobSel(); }

// 停止 / 删除作业（本机对集群操作，同 lab job stop/delete）
async function jobAction(action, id){
  const verb = action==='stop' ? '停止' : '删除';
  const variant = action==='delete' ? 'danger' : 'warn';
  const ok = await dlgConfirm(
    `确认${verb}作业`,
    `<p>确定要${verb}以下作业吗？${action==='delete'?'此操作不可撤销。':''}</p>
     <div class="dlg-meta"><strong>${esc(expLabel(id))}</strong><code>${esc(id)}</code></div>`,
    {variant, okLabel: verb}
  );
  if(!ok) return;
  try{
    const r = await fetch('/api/job/'+action+'?id='+encodeURIComponent(id), {method:'POST'});
    const d = await r.json().catch(()=>({}));
    if(!r.ok){ await dlgAlert(`${verb}失败`, esc(d.detail||('HTTP '+r.status)), {error:true}); return; }
  }catch(e){ await dlgAlert(`${verb}失败`, esc(String(e)), {error:true}); return; }
  if(action==='delete'){ const i=CMP.indexOf(id); if(i>=0) CMP.splice(i,1); delete OVS[id]; if(CUR_JOB===id) CUR_JOB=null; }
  await refresh();
}

// ========================= 作业详情 =========================
let JOB_PICK_OPEN=false;
function toggleJobPick(open){
  const pick=$('#jobPick'), menu=$('#jobPickMenu'), btn=$('#jobPickBtn');
  JOB_PICK_OPEN = open!=null? open : !JOB_PICK_OPEN;
  pick.classList.toggle('open', JOB_PICK_OPEN);
  menu.hidden = !JOB_PICK_OPEN;
  btn.setAttribute('aria-expanded', JOB_PICK_OPEN?'true':'false');
  if(JOB_PICK_OPEN){ $('#jobPickSearch').value=''; renderJobPickList(''); setTimeout(()=>$('#jobPickSearch').focus(), 30); }
}
function renderJobPickList(q){
  const ql=(q||'').trim().toLowerCase();
  const list=$('#jobPickList');
  const filtered=JOBS.filter(j=>{
    if(!ql) return true;
    return j.exp.toLowerCase().includes(ql)||j.id.toLowerCase().includes(ql)||j.status.toLowerCase().includes(ql)||j.entrypoint.toLowerCase().includes(ql);
  });
  if(!filtered.length){ list.innerHTML='<div class="jobpick-empty">无匹配作业</div>'; return; }
  list.innerHTML = filtered.map(j=>`<button type="button" class="jobpick-item ${j.id===CUR_JOB?'on':''}" role="option" aria-selected="${j.id===CUR_JOB?'true':'false'}" onclick="pickJob('${j.id}')">
    ${statusBadge(j.status,j.running)}
    <span class="jibody"><div class="jiexp" title="${esc(j.exp)}">${esc(j.exp)}</div><div class="jimeta">${esc(j.id.slice(0,22))} · ${esc(j.dur)}</div></span>
  </button>`).join('');
}
function pickJob(id){
  CUR_JOB=id; CUR_VAL=-1; toggleJobPick(false); syncJobSel(); loadDetail(true);
}
function syncJobSel(){
  if(CUR_JOB && !JOBS.some(j=>j.id===CUR_JOB)) CUR_JOB=null;
  if(!CUR_JOB && JOBS.length) CUR_JOB=(JOBS.find(j=>j.running)||JOBS[0]).id;
  $('#jobPickLabel').innerHTML = jobPickLabel(JOBS.find(j=>j.id===CUR_JOB));
  renderJobPickList($('#jobPickSearch').value);
  const j=JOBS.find(x=>x.id===CUR_JOB);
  $('#stopBtn').hidden = !(j && j.running);
  $('#delBtn').hidden  = !(j && !j.running);
}
async function loadDetail(refetch){
  if(!JOBS.length) return;
  syncJobSel();
  if(!CUR_JOB) return;
  const ov = await ensureOverview(CUR_JOB, refetch);
  renderStats(ov); renderCurve(ov); 
  const vals=ov.validations||[]; if(CUR_VAL<0||CUR_VAL>=vals.length) CUR_VAL=vals.length-1;
  renderValbar(ov); await loadSamples(true);
}

function renderStats(ov){
  const steps=ov.steps||[], vals=ov.validations||[], s=ov.summary||{};
  const last=steps.length?steps[steps.length-1]:null;
  const cards=[
    ['当前步', last?`${last.step}/${last.total}`:'—'],
    ['最新 Avg Reward', last&&last.avg_reward!=null? last.avg_reward.toFixed(3):'—'],
    ['验证准确率(最新)', pct(s.final_acc)],
    ['基线准确率(step0)', pct(s.base_acc)],
    ['准确率变化', `<span style="font-size:15px">${accBadge(s.delta_acc)}</span>`],
    ['单步耗时', last&&last.step_time? last.step_time.toFixed(0)+'s':'—'],
  ];
  $('#stats').innerHTML=cards.map(c=>`<div class="stat"><div class="k">${c[0]}</div><div class="v">${c[1]}</div></div>`).join('');
}
function renderCurve(ov){
  const steps=(ov.steps||[]).filter(s=>s.avg_reward!=null);
  const vals=(ov.validations||[]).filter(v=>v.accuracy!=null);
  const W=1000,H=240,padL=46,padR=16,padT=14,padB=26;
  if(!steps.length){ $('#curve').innerHTML=`<text x="20" y="34" fill="#8a93a6">暂无训练步数据</text>`; return; }
  const maxX=Math.max(...steps.map(s=>s.step), ...vals.map(v=>v.step),1);
  const rewards=steps.map(s=>s.avg_reward);
  let lo=Math.min(0,...rewards), hi=Math.max(1,...rewards);
  const px=x=>padL+(x/maxX)*(W-padL-padR);
  const py=r=>padT+(1-(r-lo)/((hi-lo)||1))*(H-padT-padB);
  const line=steps.map((s,i)=>`${i?'L':'M'}${px(s.step).toFixed(1)},${py(s.avg_reward).toFixed(1)}`).join(' ');
  const accPts=vals.map(v=>`<circle cx="${px(v.step).toFixed(1)}" cy="${py(v.accuracy).toFixed(1)}" r="4" fill="#3ecf8e"><title>step ${v.step} acc ${(v.accuracy*100).toFixed(1)}%</title></circle>`).join('');
  const yticks=[lo,(lo+hi)/2,hi].map(t=>`<text x="8" y="${(py(t)+4).toFixed(1)}" fill="#646d80" font-size="11" font-family="Fira Code,monospace">${t.toFixed(1)}</text><line x1="${padL}" y1="${py(t).toFixed(1)}" x2="${W-padR}" y2="${py(t).toFixed(1)}" stroke="#1d2230"/>`).join('');
  $('#curve').innerHTML=`${yticks}<path d="${line}" fill="none" stroke="#5b9dff" stroke-width="2"/>${steps.map(s=>`<circle cx="${px(s.step).toFixed(1)}" cy="${py(s.avg_reward).toFixed(1)}" r="2.2" fill="#5b9dff"><title>step ${s.step}: ${s.avg_reward.toFixed(3)}</title></circle>`).join('')}${accPts}<text x="${padL}" y="${H-7}" fill="#646d80" font-size="11" font-family="Fira Code,monospace">step 0</text><text x="${W-padR-34}" y="${H-7}" fill="#646d80" font-size="11" font-family="Fira Code,monospace">${maxX}</text>`;
}
function renderValbar(ov){
  const vals=ov.validations||[];
  if(!vals.length){ $('#valbar').innerHTML=''; return; }
  $('#valbar').innerHTML=vals.map((v,i)=>{
    const acc=v.accuracy!=null?` · ${(v.accuracy*100).toFixed(0)}%`:'';
    return `<button class="${i===CUR_VAL?'active':''}" onclick="selVal(${i})">step ${v.step}${acc} <span class="pill">(${v.sample_count})</span></button>`;
  }).join('');
}
function selVal(i){ CUR_VAL=i; renderValbar(OVS[CUR_JOB]); loadSamples(true); }

async function loadSamples(reset){
  const ov=OVS[CUR_JOB]; const vals=ov? ov.validations||[]:[]; const v=vals[CUR_VAL];
  if(!v){ $('#samples').innerHTML=`<div class="empty">该作业暂无验证对话样本。<br/>验证条数由 config 的 <code>logger.num_val_samples_to_print</code> 决定（已为 qa 实验设为 16）。</div>`; $('#moreWrap').innerHTML=''; return; }
  if(reset){ LOADED=[]; OFFSET=0; TOTAL=v.sample_count; $('#samples').innerHTML=''; }
  if(LOADING) return;
  if(TOTAL>0 && OFFSET>=TOTAL){ renderMore(); return; }
  LOADING=true; renderMore();
  const r=await fetch(`/api/samples?id=${encodeURIComponent(CUR_JOB)}&vidx=${CUR_VAL}&offset=${OFFSET}&limit=${PAGE}`);
  const data=await r.json();
  TOTAL=data.total; OFFSET+=data.samples.length; LOADED=LOADED.concat(data.samples);
  LOADING=false; renderSamples(); renderMore();
}
function renderSamples(){
  if(!LOADED.length){ $('#samples').innerHTML=`<div class="empty">本次验证还在生成中，或未打印对话（num_val_samples_to_print=0）。</div>`; return; }
  $('#samples').innerHTML=LOADED.map((s,si)=>`
    <div class="sample">
      <div class="sh"><span class="badge ${rewardClass(s.reward)}">Reward ${s.reward.toFixed(3)}</span><span class="pill">Sample ${s.idx}</span></div>
      <div class="seg user collapsed" id="u${si}">
        <div class="lbl">用户 / 题目</div><pre>${highlight(s.user)}</pre>
        <span class="toggle" onclick="document.getElementById('u${si}').classList.toggle('collapsed')">展开 / 收起</span>
      </div>
      <div class="seg"><div class="lbl">模型作答（model response）</div><pre>${highlight(s.assistant)}</pre></div>
      <div class="seg"><div class="lbl">环境评分</div><pre>${highlight(s.env)}</pre></div>
    </div>`).join('');
}
function renderMore(){
  const w=$('#moreWrap');
  if(LOADING){ w.innerHTML=`<button class="loading">加载中…</button>`; return; }
  if(TOTAL>0 && OFFSET<TOTAL){ w.innerHTML=`<button onclick="loadSamples(false)">加载更多（${OFFSET}/${TOTAL}）</button>`; }
  else if(TOTAL>0){ w.innerHTML=`<span class="pill">已全部加载 ${TOTAL} 条</span>`; }
  else { w.innerHTML=''; }
}

// --------- 事件 ---------
document.querySelectorAll('.tab').forEach(t=>t.addEventListener('click',()=>switchView(t.dataset.view)));
$('#jobPickBtn').addEventListener('click', e=>{ e.stopPropagation(); toggleJobPick(); });
$('#jobPickSearch').addEventListener('input', e=>renderJobPickList(e.target.value));
$('#jobPickSearch').addEventListener('keydown', e=>e.stopPropagation());
document.addEventListener('click', e=>{ if(JOB_PICK_OPEN && !$('#jobPick').contains(e.target)) toggleJobPick(false); });
document.addEventListener('keydown', e=>{
  if(e.key!=='Escape') return;
  if(!$('#dlgOverlay').hidden){ closeDlg(false); return; }
  if(JOB_PICK_OPEN) toggleJobPick(false);
});
$('#dlgOverlay').addEventListener('click', e=>{ if(e.target===$('#dlgOverlay')) closeDlg(false); });
$('#stopBtn').addEventListener('click', ()=>{ if(CUR_JOB) jobAction('stop', CUR_JOB); });
$('#delBtn').addEventListener('click', ()=>{ if(CUR_JOB) jobAction('delete', CUR_JOB); });
$('#refreshBtn').addEventListener('click', refresh);
let timer=null;
function setAuto(on){ if(timer){clearInterval(timer);timer=null;} if(on){ timer=setInterval(refresh, 15000); } }
$('#autoChk').addEventListener('change', e=>setAuto(e.target.checked));
$('#diffOnly').addEventListener('change', drawDiff);
$('#hideMissing').addEventListener('change', drawDiff);
let rzTimer=null;
window.addEventListener('resize', ()=>{ clearTimeout(rzTimer); rzTimer=setTimeout(()=>{ if(VIEW==='compare') applyChipRows(); }, 150); });

(async()=>{ await loadJobs(); $('#updated').textContent='更新于 '+new Date().toLocaleTimeString(); await renderCompare(true); setAuto(true); })();
</script>
</body>
</html>
"""

# ----------------------------- FastAPI 应用 -----------------------------
app = FastAPI(title="NeMo-RL Lab 训练面板", docs_url="/api/docs", redoc_url=None)
_SRC: DataSource | None = None


def _src() -> DataSource:
    if _SRC is None:  # pragma: no cover - main() 一定会先设置
        raise HTTPException(status_code=503, detail="data source not ready")
    return _SRC


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML


@app.get("/api/jobs")
def api_jobs():
    return _src().list_jobs()


@app.get("/api/job")
def api_job(id: str = Query(...)):
    return _src().job_overview(id)


@app.get("/api/diff")
def api_diff(ids: str = Query(..., description="逗号分隔的作业 id")):
    id_list = [x for x in ids.split(",") if x]
    if not id_list:
        raise HTTPException(status_code=400, detail="ids 为空")
    return _src().config_diff(id_list)


@app.post("/api/job/stop")
def api_stop(id: str = Query(...)):
    return _src().stop_job(id)


@app.post("/api/job/delete")
def api_delete(id: str = Query(...)):
    return _src().delete_job(id)


@app.get("/api/samples")
def api_samples(
    id: str = Query(...),
    vidx: int = Query(...),
    offset: int = Query(0, ge=0),
    limit: int = Query(6, ge=1, le=50),
):
    return _src().samples_page(id, vidx, offset, limit)


def main() -> None:
    ap = argparse.ArgumentParser(description="本地训练面板（reward 曲线 + 验证对话，FastAPI）")
    ap.add_argument("--address", required=True, help="Ray dashboard 地址")
    ap.add_argument("--port", type=int, default=8080, help="本地服务端口（默认 8080）")
    ap.add_argument("--open", action="store_true", help="启动后自动打开浏览器")
    args = ap.parse_args()

    global _SRC
    _SRC = DataSource(args.address)
    url = f"http://127.0.0.1:{args.port}"
    print(f"✓ 训练面板已启动: {url}")
    print(f"  数据来源: {args.address}（Ray dashboard 日志，只读）")
    print(f"  API 文档: {url}/api/docs   |   Ctrl-C 退出。")
    if args.open:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
