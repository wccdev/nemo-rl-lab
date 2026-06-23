"""训练后闭环测试：cli 的 runtime_env 密钥分流 + 集群侧 post_train.sh 的 step/后端逻辑（干跑）。"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from nemo_rl_lab import cli

REPO_ROOT = Path(__file__).resolve().parent.parent
POST_TRAIN = REPO_ROOT / "scripts" / "post_train.sh"


# --------------------------- runtime_env 密钥分流 ---------------------------
def test_post_runtime_env_forwards_secret_by_default():
    merged = {"OUTPUT_ROOT": "/data/runs", "HF_TOKEN": "hf-secret", "RUN_USER": "alice"}
    re = json.loads(cli._build_post_runtime_env(merged, "/opt/nemo-rl", "h100", "rid", "alice"))
    ev = re["env_vars"]
    assert ev["NEMO_RL_DIR"] == "/opt/nemo-rl"
    assert ev["HF_TOKEN"] == "hf-secret"  # 未配 secrets 文件 → 兜底明文转发
    assert ev["NRL_RUN_ID"] == "rid"
    assert "CLUSTER_SECRETS_FILE" not in ev


def test_post_runtime_env_secrets_file_hides_secret():
    merged = {"HF_TOKEN": "hf-secret", "CLUSTER_SECRETS_FILE": "/data/secrets.env"}
    re = json.loads(cli._build_post_runtime_env(merged, "/opt/nemo-rl", "h100", "rid", "bob"))
    ev = re["env_vars"]
    assert ev["CLUSTER_SECRETS_FILE"] == "/data/secrets.env"
    assert "HF_TOKEN" not in ev  # 配了 secrets 文件 → 不明文转发


def test_post_runtime_env_excludes_outputs_and_secrets():
    re = json.loads(cli._build_post_runtime_env({}, "/opt/x", "h100", "rid", "u"))
    assert "**/outputs/**" in re["excludes"]
    assert "cluster/*/submit.env" in re["excludes"]


# --------------------------- post_train.sh 干跑（step 发现 + 后端检测）---------------------------
def _run_post(tmp_ckpt: Path, *args: str) -> str:
    env = {"LAB_DRY_RUN": "1", "NEMO_RL_DIR": "/opt/nemo-rl", "PATH": "/usr/bin:/bin"}
    proc = subprocess.run(
        ["bash", str(POST_TRAIN), *args, "--ckpt-dir", str(tmp_ckpt)],
        capture_output=True, text=True, env=env,
    )
    return proc.stdout


def _mk_step(root: Path, n: int, megatron: bool):
    step = root / f"step_{n}"
    (step / "policy" / "tokenizer").mkdir(parents=True)
    (step / "config.yaml").write_text("policy: {}\n")
    if megatron:
        (step / "policy" / "weights" / f"iter_{n:07d}").mkdir(parents=True)
    else:
        w = step / "policy" / "weights"
        w.mkdir(parents=True)
        (w / ".metadata").write_text("")


@pytest.mark.skipif(shutil.which("bash") is None, reason="需要 bash")
def test_export_picks_latest_step_and_megatron(tmp_path: Path):
    _mk_step(tmp_path, 5, megatron=True)
    _mk_step(tmp_path, 12, megatron=True)
    out = _run_post(tmp_path, "export", "experiments/foo")
    assert "step=12" in out and "backend=megatron" in out
    assert "convert_megatron_to_hf.py" in out
    assert "iter_0000012" in out


@pytest.mark.skipif(shutil.which("bash") is None, reason="需要 bash")
def test_export_explicit_step(tmp_path: Path):
    _mk_step(tmp_path, 5, megatron=True)
    _mk_step(tmp_path, 12, megatron=True)
    out = _run_post(tmp_path, "export", "experiments/foo", "--step", "5")
    assert "step=5" in out and "iter_0000005" in out


@pytest.mark.skipif(shutil.which("bash") is None, reason="需要 bash")
def test_export_dcp_backend(tmp_path: Path):
    _mk_step(tmp_path, 10, megatron=False)
    out = _run_post(tmp_path, "export", "experiments/foo")
    assert "backend=dcp" in out
    assert "convert_dcp_to_hf.py" in out
    assert "policy/weights --hf-ckpt-path" in out  # dcp 用 weights 目录本身


@pytest.mark.skipif(shutil.which("bash") is None, reason="需要 bash")
def test_eval_without_model_exports_first(tmp_path: Path):
    _mk_step(tmp_path, 8, megatron=False)
    out = _run_post(tmp_path, "eval", "experiments/foo")
    assert "convert_dcp_to_hf.py" in out  # 先导出
    assert "run_eval.py" in out  # 再评测
