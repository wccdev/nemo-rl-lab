"""lab completion 子命令与 ./lab shim 补全脚本。"""
from __future__ import annotations

from pathlib import Path

import pytest
import typer

from nemo_rl_lab import shell_completion


def test_show_zsh_contains_compdef():
    script = shell_completion.show_script(shell_completion.ShellChoice.zsh)
    assert "#compdef lab" in script
    assert "compdef" in script


def test_show_bash_wrapper_uses_uv_run():
    repo = Path("/tmp/nemo-rl-lab-test")
    script = shell_completion.show_script(
        shell_completion.ShellChoice.bash, wrapper=True, repo_root=repo
    )
    assert "uv run --project" in script and " lab" in script
    assert str(repo.resolve()) in script
    assert "complete -o default -F" in script
    assert "./lab" in script


def test_wrapper_requires_bash():
    with pytest.raises(typer.BadParameter):
        shell_completion.show_script(
            shell_completion.ShellChoice.zsh, wrapper=True, repo_root=Path("/x")
        )


def test_install_script_bash(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    path = shell_completion.install_script(shell_completion.ShellChoice.bash)
    assert path.is_file()
    assert "complete" in path.read_text()
    assert (home / ".bashrc").is_file()


def test_install_wrapper_bash(tmp_path, monkeypatch):
    home = tmp_path / "home"
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "lab").write_text("#!/bin/bash\n")
    monkeypatch.setenv("HOME", str(home))
    path = shell_completion.install_script(
        shell_completion.ShellChoice.bash, wrapper=True, repo_root=repo
    )
    assert path.name == "lab-shim.sh"
    text = path.read_text()
    assert f"uv run --project '{repo.resolve()}' lab" in text
    assert "complete -o default -F" in text
