"""客户端登录/门控单测（纯客户端，无 server 依赖）。"""
from __future__ import annotations

import base64
import hashlib
import json

import pytest

from nemo_rl_lab import cli_login
from nemo_rl_lab.client_device import collect_cli_device, encode_device_param


def test_pkce_pair_self_consistent():
    """CLI 生成的 verifier/challenge 自洽：challenge == base64url(sha256(verifier))，无填充。"""
    verifier, challenge = cli_login.pkce_pair()
    expect = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
    assert challenge == expect
    assert "=" not in challenge


@pytest.fixture()
def isolated_lab(tmp_path, monkeypatch):
    monkeypatch.setattr(cli_login, "CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.setattr(cli_login, "CRED_PATH", tmp_path / "credentials.json")
    monkeypatch.delenv("LAB_SERVER", raising=False)
    return tmp_path


def test_server_mode_detection(isolated_lab, monkeypatch):
    assert cli_login.current_server() is None
    assert cli_login.is_server_mode() is False
    cli_login._save_server("https://lab.x.com/")
    assert cli_login.current_server() == "https://lab.x.com"  # 去尾斜杠
    monkeypatch.setenv("LAB_SERVER", "https://env.x.com")
    assert cli_login.current_server() == "https://env.x.com"  # 环境优先
    assert cli_login.current_server("https://explicit.com") == "https://explicit.com"


def test_creds_roundtrip(isolated_lab):
    cli_login._save_creds("https://lab.x.com", {"access_token": "t", "expires_at": None})
    assert cli_login._load_creds("https://lab.x.com")["access_token"] == "t"
    cli_login._clear_creds("https://lab.x.com")
    assert cli_login._load_creds("https://lab.x.com") is None


def test_gate_requires_server(isolated_lab):
    # 未接入中心化服务：gate 必须报错引导登录（不再有 direct no-op），不触发任何网络/浏览器
    import typer

    with pytest.raises(typer.BadParameter):
        cli_login.gate("submit")


def test_get_access_token_valid_and_expired(isolated_lab):
    import time

    cli_login._save_creds("https://s", {"access_token": "valid", "expires_at": time.time() + 100})
    assert cli_login.get_access_token("https://s") == "valid"
    cli_login._save_creds("https://s", {"access_token": "old", "expires_at": time.time() - 10,
                                        "refresh_token": None})
    assert cli_login.get_access_token("https://s") is None


# ----------------------------- Phase B：打包 / 可追溯 -----------------------------
def _git_repo(tmp_path):
    import subprocess

    def git(*a):
        subprocess.run(["git", "-C", str(tmp_path), *a], check=True,
                       capture_output=True, text=True)

    git("init", "-q")
    git("config", "user.email", "t@t.com")
    git("config", "user.name", "t")
    return git


def test_pack_working_dir_respects_gitignore(tmp_path):
    git = _git_repo(tmp_path)
    (tmp_path / ".gitignore").write_text("ignored.txt\noutputs/\n")
    (tmp_path / "keep.py").write_text("print(1)\n")
    (tmp_path / "ignored.txt").write_text("secret\n")
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs" / "big.bin").write_text("x" * 100)
    (tmp_path / "untracked_new.txt").write_text("new but not ignored\n")
    git("add", ".gitignore", "keep.py")
    git("commit", "-qm", "init")

    blob = cli_login.pack_working_dir(tmp_path)
    import io
    import tarfile

    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        names = set(tar.getnames())
    assert "keep.py" in names
    assert "untracked_new.txt" in names  # 未提交但未被忽略 → 应上传（贴近 Ray 语义）
    assert "ignored.txt" not in names
    assert "outputs/big.bin" not in names


def test_git_provenance(tmp_path):
    git = _git_repo(tmp_path)
    exp = tmp_path / "experiments" / "demo_v1"
    exp.mkdir(parents=True)
    (exp / "config.yaml").write_text("a: 1\n")
    git("add", ".")
    git("commit", "-qm", "init")

    prov = cli_login.git_provenance(tmp_path, "experiments/demo_v1")
    assert prov["git_commit"] != "unknown"
    assert prov["git_dirty"] is False
    assert len(prov["config_sha"]) == 12

    (exp / "config.yaml").write_text("a: 2\n")  # 改动 → dirty
    assert cli_login.git_provenance(tmp_path, "experiments/demo_v1")["git_dirty"] is True


def test_collect_cli_device():
    info = collect_cli_device()
    assert info["source"] == "lab-cli"
    assert info["hostname"]
    assert info["os"]
    assert info.get("device_id")  # 16 hex chars
    assert len(info["device_id"]) == 16


def test_encode_device_roundtrip():
    raw = encode_device_param(collect_cli_device())
    assert raw
    assert "=" not in raw
    pad = "=" * (-len(raw) % 4)
    data = json.loads(base64.urlsafe_b64decode(raw + pad))
    assert data["source"] == "lab-cli"
