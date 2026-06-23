"""lab CLI 辅助函数的单元测试（不触发真实提交 / 网络）。"""
from __future__ import annotations

from pathlib import Path

import pytest
import typer

from nemo_rl_lab import cli


def test_read_env_file(tmp_path: Path):
    f = tmp_path / "submit.env"
    f.write_text(
        "# 注释行\n"
        "\n"
        "RAY_DASHBOARD_ADDRESS=http://1.2.3.4:8265\n"
        "  NEMO_RL_DIR = /opt/nemo-rl  \n"
        "BAD_LINE_NO_EQUALS\n"
    )
    env = cli._read_env_file(f)
    assert env["RAY_DASHBOARD_ADDRESS"] == "http://1.2.3.4:8265"
    assert env["NEMO_RL_DIR"] == "/opt/nemo-rl"  # 两侧空白被裁掉
    assert "BAD_LINE_NO_EQUALS" not in env


def test_read_env_file_missing(tmp_path: Path):
    assert cli._read_env_file(tmp_path / "nope.env") == {}


def test_resolve_exp_known():
    # 仓库内现有实验应可解析为 experiments/<name>
    assert cli._resolve_exp("grpo_qwen3.5-4b_gsm8k_v1") == "experiments/grpo_qwen3.5-4b_gsm8k_v1"


def test_resolve_exp_unknown_raises():
    with pytest.raises(typer.BadParameter):
        cli._resolve_exp("不存在的实验_xyz")


def test_list_exps_nonempty():
    exps = cli._list_exps()
    assert "grpo_qwen3.5-4b_gsm8k_v1" in exps


def test_list_profiles_has_h100():
    profiles = cli._list_profiles()
    assert "h100" in profiles


def test_pinned_ray_version_matches_pyproject():
    v = cli._pinned_ray_version()
    assert v is not None
    assert v.count(".") >= 1  # 形如 2.55.1


def test_validate_exp_clean_on_real_experiment():
    errors, _ = cli._validate_exp("experiments/grpo_qwen3.5-4b_gsm8k_v1")
    assert errors == []


# --------------------------- run 台账 ---------------------------
def test_read_ledger_skips_bad_lines(tmp_path: Path):
    led = tmp_path / "runs.jsonl"
    led.write_text(
        '{"run_id": "a", "exp": "experiments/x", "time": "2026-01-01 00:00:00"}\n'
        "\n"
        "这是坏行 not json\n"
        '{"run_id": "b", "exp": "experiments/y", "time": "2026-01-02 00:00:00"}\n'
    )
    rows = cli._read_ledger(led)
    assert [r["run_id"] for r in rows] == ["a", "b"]


def test_read_ledger_missing_file(tmp_path: Path):
    assert cli._read_ledger(tmp_path / "nope.jsonl") == []


def test_filter_runs_sorts_desc_and_limits():
    entries = [
        {"run_id": "a", "exp": "experiments/x", "time": "2026-01-01 00:00:00"},
        {"run_id": "b", "exp": "experiments/y", "time": "2026-01-03 00:00:00"},
        {"run_id": "c", "exp": "experiments/x", "time": "2026-01-02 00:00:00"},
    ]
    rows = cli._filter_runs(entries, exp=None, limit=2)
    assert [r["run_id"] for r in rows] == ["b", "c"]  # 时间倒序后取前 2


def test_filter_runs_by_exp_basename():
    entries = [
        {"run_id": "a", "exp": "experiments/x", "time": "1"},
        {"run_id": "b", "exp": "projects/y", "time": "2"},
    ]
    # 末段名匹配：传 "x" 或 "experiments/x" 都命中
    assert [r["run_id"] for r in cli._filter_runs(entries, exp="x", limit=None)] == ["a"]
    assert [r["run_id"] for r in cli._filter_runs(entries, exp="experiments/x", limit=None)] == ["a"]


# --------------------------- dashboard 解析（对齐真实 JSON 形状，monkeypatch 掉 HTTP）---------------------------
def test_run_status_map_matches_by_metadata(monkeypatch):
    jobs = [
        {"submission_id": "raysubmit_1", "status": "RUNNING",
         "metadata": {"lab_run_id": "exp-20260101", "lab_exp": "experiments/x"}},
        {"submission_id": "raysubmit_2", "status": "SUCCEEDED", "metadata": {}},  # 非 lab 作业
        {"submission_id": "raysubmit_3", "status": "FAILED",
         "metadata": {"lab_run_id": "export-20260102"}},
    ]
    monkeypatch.setattr(cli, "_dashboard_get", lambda addr, path, **kw: jobs)
    m = cli._run_status_map("http://x:8265")
    assert m == {"exp-20260101": "RUNNING", "export-20260102": "FAILED"}


def test_gpu_summary_parses_usage(monkeypatch):
    # 形状取自真实 /api/cluster_status：data.clusterStatus.loadMetricsReport.usage
    payload = {"data": {"clusterStatus": {"loadMetricsReport": {"usage": {
        "GPU": [2.0, 8.0], "CPU": [10.0, 96.0],
        "memory": [0.0, 1613528158208.0], "acceleratorType:H100": [0.0, 8.0],
    }}}}}
    monkeypatch.setattr(cli, "_dashboard_get", lambda addr, path, **kw: payload)
    g = cli._gpu_summary("http://x:8265")
    assert g is not None
    assert g["usage"]["GPU"] == [2.0, 8.0]
    assert g["accel"] == ["H100"]


def test_gpu_summary_empty_returns_none(monkeypatch):
    monkeypatch.setattr(cli, "_dashboard_get", lambda addr, path, **kw: {"data": {}})
    assert cli._gpu_summary("http://x:8265") is None


def test_latest_job_id_picks_newest(monkeypatch):
    jobs = [
        {"submission_id": "old", "start_time": 100},
        {"submission_id": "new", "start_time": 300},
        {"submission_id": "mid", "start_time": 200},
    ]
    monkeypatch.setattr(cli, "_fetch_jobs", lambda addr: jobs)
    assert cli._latest_job_id("http://x:8265") == "new"


def test_latest_job_id_empty(monkeypatch):
    monkeypatch.setattr(cli, "_fetch_jobs", lambda addr: [])
    assert cli._latest_job_id("http://x:8265") is None


# --------------------------- config diff（_flatten）---------------------------
def test_flatten_nested_and_lists():
    flat = cli._flatten({"a": {"b": 1, "c": [10, 20]}, "d": None})
    assert flat == {"a.b": "1", "a.c[0]": "10", "a.c[1]": "20", "d": "null"}


def test_flatten_diff_keys():
    a = cli._flatten({"x": 1, "only_a": 5, "nested": {"k": "v1"}})
    b = cli._flatten({"x": 2, "only_b": 9, "nested": {"k": "v1"}})
    changed = {k for k in a if k in b and a[k] != b[k]}
    assert changed == {"x"}
    assert (set(a) - set(b)) == {"only_a"}
    assert (set(b) - set(a)) == {"only_b"}


# --------------------------- init（_set_env_line）---------------------------
def test_set_env_line_replaces_existing(tmp_path: Path):
    f = tmp_path / "submit.env"
    f.write_text("# 注释\nFOO=old\nBAR=keep\n")
    cli._set_env_line(f, "FOO", "new")
    env = cli._read_env_file(f)
    assert env["FOO"] == "new" and env["BAR"] == "keep"


def test_set_env_line_uncomments(tmp_path: Path):
    f = tmp_path / "submit.env"
    f.write_text("# RAY_DASHBOARD_ADDRESS=http://x\nOTHER=1\n")
    cli._set_env_line(f, "RAY_DASHBOARD_ADDRESS", "http://1.2.3.4:8265")
    assert cli._read_env_file(f)["RAY_DASHBOARD_ADDRESS"] == "http://1.2.3.4:8265"
    # 不应再有被注释的旧行
    assert "# RAY_DASHBOARD_ADDRESS=" not in f.read_text()


def test_set_env_line_appends_when_missing(tmp_path: Path):
    f = tmp_path / "submit.env"
    f.write_text("A=1\n")
    cli._set_env_line(f, "NEW_KEY", "v")
    assert cli._read_env_file(f)["NEW_KEY"] == "v"


# --------------------------- tunnel（_addr_host）---------------------------
def test_addr_host():
    assert cli._addr_host("http://172.19.12.24:8265") == "172.19.12.24"
    assert cli._addr_host("http://127.0.0.1:8265") == "127.0.0.1"
    assert cli._addr_host(None) is None
