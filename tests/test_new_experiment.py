"""new_experiment 跨平台逻辑单测。"""
from __future__ import annotations

import pytest

from nemo_rl_lab.new_experiment import NewExperimentError, create_experiment


def test_create_grpo_from_template(tmp_path):
    repo = tmp_path / "repo"
    template = repo / "templates" / "experiment-template"
    template.mkdir(parents=True)
    (template / "config.yaml").write_text("defaults:\n  - ../../configs/base/grpo_math_1B.yaml\n", encoding="utf-8")
    (template / "run.sh").write_text("# export ENTRY\n", encoding="utf-8")
    (template / "cluster").write_text("h100\n", encoding="utf-8")
    (repo / "cluster" / "h100").mkdir(parents=True)
    (repo / "cluster" / "h100" / "overrides.conf").write_text("# test\n", encoding="utf-8")
    (repo / "experiments").mkdir()

    create_experiment(repo, "experiments", "test_exp_v1", cluster="h100", method="grpo")
    dest = repo / "experiments" / "test_exp_v1"
    assert dest.is_dir()
    assert (dest / "config.yaml").is_file()
    assert (dest / "cluster").read_text(encoding="utf-8").strip() == "h100"


def test_create_rejects_unknown_cluster(tmp_path):
    repo = tmp_path / "repo"
    template = repo / "templates" / "experiment-template"
    template.mkdir(parents=True)
    (template / "config.yaml").write_text("x: 1\n", encoding="utf-8")
    (repo / "experiments").mkdir()

    with pytest.raises(NewExperimentError, match="未知集群 profile"):
        create_experiment(repo, "experiments", "x", cluster="nope", method="grpo")


def test_fork_patches_swanlab_and_readme(tmp_path):
    repo = tmp_path / "repo"
    src = repo / "experiments" / "src_exp"
    src.mkdir(parents=True)
    (src / "config.yaml").write_text(
        "swanlab:\n  project: \"old\"\n  name: \"old\"\nother: 1\n",
        encoding="utf-8",
    )
    (src / "README.md").write_text("# old title\n", encoding="utf-8")
    (src / "cluster").write_text("h100\n", encoding="utf-8")
    (repo / "experiments").mkdir(exist_ok=True)

    create_experiment(repo, "experiments", "new_exp", src="src_exp")
    cfg = (repo / "experiments" / "new_exp" / "config.yaml").read_text(encoding="utf-8")
    assert 'project: "new_exp"' in cfg
    assert 'name: "new_exp"' in cfg
    assert (repo / "experiments" / "new_exp" / "README.md").read_text(encoding="utf-8").startswith("# new_exp")


def test_create_sft_method(tmp_path):
    repo = tmp_path / "repo"
    template = repo / "templates" / "experiment-template"
    template.mkdir(parents=True)
    (template / "config.yaml").write_text(
        "defaults:\n  - ../../configs/base/grpo_math_1B.yaml\n\ngrpo:\n  x: 1\n\nloss_fn:\n  y: 2\n\nlogger:\n  z: 3\n",
        encoding="utf-8",
    )
    (template / "run.sh").write_text(
        '# export ENTRY="${ENTRY:-examples/run_sft.py}"\n',
        encoding="utf-8",
    )
    (repo / "experiments").mkdir()

    create_experiment(repo, "experiments", "sft_test", method="sft")
    cfg = (repo / "experiments" / "sft_test" / "config.yaml").read_text(encoding="utf-8")
    assert "../../configs/base/sft.yaml" in cfg
    assert "grpo:" not in cfg
    assert "sft:" in cfg
    run_sh = (repo / "experiments" / "sft_test" / "run.sh").read_text(encoding="utf-8")
    assert run_sh.startswith('export ENTRY="${ENTRY:-examples/run_sft.py}"')
