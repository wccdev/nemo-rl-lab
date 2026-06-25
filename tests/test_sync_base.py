"""sync_base 跨平台逻辑单测。"""
from __future__ import annotations

import pytest

from nemo_rl_lab.sync_base import SyncBaseError, sync_base_configs


def test_sync_base_copies_files(tmp_path):
    repo = tmp_path / "repo"
    nemo = tmp_path / "NeMo-RL"
    src = nemo / "examples" / "configs"
    src.mkdir(parents=True)
    (src / "grpo_math_1B.yaml").write_text("grpo: 1\n", encoding="utf-8")
    (src / "sft.yaml").write_text("sft: 1\n", encoding="utf-8")

    sync_base_configs(repo, nemo)
    dst = repo / "configs" / "base"
    assert (dst / "grpo_math_1B.yaml").read_text(encoding="utf-8") == "grpo: 1\n"
    assert (dst / "sft.yaml").is_file()


def test_sync_base_missing_nemo_dir(tmp_path):
    with pytest.raises(SyncBaseError, match="配置目录不存在"):
        sync_base_configs(tmp_path / "repo", tmp_path / "missing")
