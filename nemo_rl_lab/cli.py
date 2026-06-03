"""nemo-rl-lab 统一 CLI（Typer 实现）。

所有操作都通过 `lab <子命令>` 执行；内部调用 scripts/ 与各实验脚本（单一事实来源）。
本 CLI 是项目的命令入口（pyproject [project.scripts] lab = nemo_rl_lab.cli:app）。

调用方式（任选）：
    uv run lab ls            # 推荐：uv 自动同步项目环境再运行（对任何人都生效）
    ./lab ls                 # 仓库根的薄 shim，等价于上面那条
    lab ls                   # `uv sync` 后 .venv/bin/lab 已存在；激活 venv 即可直接用

常用：
    uv run lab ls                                  列出实验 / 项目
    uv run lab new grpo_qwen3.5-4b_gsm8k_v1        从模板新建实验
    uv run lab prepare gsm8k                       预处理数据集
    uv run lab submit agent-grpo_qwen3.5-9b_multitool_v1   从本机提交作业到 Ray 集群
    uv run lab run grpo_qwen3.5-9b_gsm8k_v1 --nemo-rl /opt/NeMo-RL   在集群容器内直接跑
    uv run lab ray head                            启动 Ray head（在 head 节点容器内）
    uv run lab sync-base --nemo-rl /opt/NeMo-RL    同步官方基底配置

补全（Tab，支持 bash/zsh/fish/powershell）：
    uv run lab --install-completion     # 安装到当前 shell
    uv run lab --show-completion        # 仅打印脚本
"""
from __future__ import annotations

import os
import subprocess
import sys
from enum import Enum
from pathlib import Path
from typing import Optional

import typer

# 包位于 <repo>/nemo_rl_lab/，仓库根是上一级（editable 安装下 __file__ 指向源码）。
ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
DATA_PREP = {
    "gsm8k": ROOT / "common" / "data" / "prepare_gsm8k.py",
    "alpaca": ROOT / "common" / "data" / "prepare_alpaca.py",
    "qa_rl": ROOT / "common" / "data" / "prepare_qa_rl.py",
}

app = typer.Typer(
    add_completion=True,
    no_args_is_help=True,
    rich_markup_mode="rich",
    help="nemo-rl-lab 统一 CLI",
    context_settings={"help_option_names": ["-h", "--help"]},
)


# ----------------------------- 辅助 -----------------------------
def _run(cmd: list[str], env: dict | None = None, cwd: Path | None = None) -> int:
    """打印并执行命令，返回退出码。"""
    typer.echo("› " + " ".join(str(c) for c in cmd))
    full_env = {**os.environ, **(env or {})}
    return subprocess.run(cmd, env=full_env, cwd=str(cwd or ROOT)).returncode


def _resolve_exp(name: str) -> str:
    """把实验名解析为相对仓库根的路径，接受 'experiments/x' / 'projects/x' / 'x'。"""
    cands = [name] if "/" in name else [f"experiments/{name}", f"projects/{name}"]
    for c in cands:
        if (ROOT / c).is_dir():
            return c
    raise typer.BadParameter(f"找不到实验: {name}（已尝试: {', '.join(cands)}）")


def _list_exps() -> list[str]:
    out: list[str] = []
    for kind in ("experiments", "projects"):
        base = ROOT / kind
        if base.is_dir():
            out += [p.name for p in base.iterdir() if p.is_dir()]
    return sorted(set(out))


def _list_profiles() -> list[str]:
    base = ROOT / "cluster"
    if not base.is_dir():
        return []
    return sorted(p.name for p in base.iterdir() if (p / "overrides.conf").is_file())


# ----------------------------- 动态补全回调 -----------------------------
def _complete_exp(incomplete: str) -> list[str]:
    return [e for e in _list_exps() if e.startswith(incomplete)]


def _complete_profile(incomplete: str) -> list[str]:
    return [p for p in _list_profiles() if p.startswith(incomplete)]


def _complete_dataset(incomplete: str) -> list[str]:
    return [d for d in sorted(DATA_PREP) if d.startswith(incomplete)]


# ----------------------------- 选择项 -----------------------------
class Kind(str, Enum):
    experiments = "experiments"
    projects = "projects"


class RayAction(str, Enum):
    head = "head"
    worker = "worker"
    status = "status"


# ----------------------------- 子命令 -----------------------------
@app.command(help="列出实验 / 项目")
def ls() -> None:
    for kind in ("experiments", "projects"):
        base = ROOT / kind
        if not base.is_dir():
            continue
        exps = sorted(p.name for p in base.iterdir() if p.is_dir())
        typer.echo(f"\n[{kind}] ({len(exps)})")
        for e in exps:
            typer.echo(f"  - {e}")


@app.command(help="从模板新建实验")
def new(
    name: str = typer.Argument(..., help="实验名（见 docs/naming-convention.md）"),
    kind: Kind = typer.Option(Kind.experiments, "--kind", help="放到 experiments 还是 projects"),
) -> None:
    raise typer.Exit(_run(["bash", str(SCRIPTS / "new_experiment.sh"), kind.value, name]))


@app.command(
    help="预处理数据集（gsm8k / alpaca / qa_rl）",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def prepare(
    ctx: typer.Context,
    dataset: str = typer.Argument(..., autocompletion=_complete_dataset, help="数据集名"),
) -> None:
    script = DATA_PREP.get(dataset)
    if not script:
        raise typer.BadParameter(f"未知数据集: {dataset}（可选: {', '.join(DATA_PREP)}）")
    # 用当前解释器（项目 uv 环境，含 datasets）跑数据脚本。
    raise typer.Exit(_run([sys.executable, str(script), *ctx.args]))


@app.command(help="从本机提交作业到 Ray 集群（执行在集群）")
def submit(
    exp: str = typer.Argument(..., autocompletion=_complete_exp, help="实验名或路径"),
    profile: Optional[str] = typer.Option(
        None, "--profile", autocompletion=_complete_profile, help="硬件 profile（默认取 submit.env）"
    ),
) -> None:
    cmd = ["bash", str(SCRIPTS / "submit_job.sh"), _resolve_exp(exp)]
    if profile:
        cmd.append(profile)
    raise typer.Exit(_run(cmd))


@app.command(help="直接运行实验 run.sh（在集群容器内用）")
def run(
    exp: str = typer.Argument(..., autocompletion=_complete_exp, help="实验名或路径"),
    profile: Optional[str] = typer.Option(
        None, "--profile", autocompletion=_complete_profile, help="硬件 profile：gb10-spark | h200"
    ),
    nemo_rl: Optional[str] = typer.Option(None, "--nemo-rl", help="NeMo-RL 源码目录"),
) -> None:
    exp_path = _resolve_exp(exp)
    env: dict[str, str] = {}
    if nemo_rl:
        env["NEMO_RL_DIR"] = nemo_rl
    if profile:
        env["CLUSTER_PROFILE"] = profile
    raise typer.Exit(_run(["bash", str(ROOT / exp_path / "run.sh")], env=env))


@app.command(name="sync-base", help="同步官方基底配置到 configs/base/")
def sync_base(
    nemo_rl: Optional[str] = typer.Option(None, "--nemo-rl", help="NeMo-RL 源码目录"),
) -> None:
    env = {"NEMO_RL_DIR": nemo_rl} if nemo_rl else {}
    raise typer.Exit(_run(["bash", str(SCRIPTS / "sync_base_configs.sh")], env=env))


@app.command(help="Ray 集群管理（在对应节点容器内执行；ray 由 NeMo-RL 的 uv 环境提供）")
def ray(
    action: RayAction = typer.Argument(..., help="head | worker | status"),
    profile: str = typer.Option("gb10-spark", "--profile", autocompletion=_complete_profile),
    head: Optional[str] = typer.Option(None, "--head", help="worker 加入的 head 地址，如 192.168.1.4:6379"),
    nemo_rl: Optional[str] = typer.Option(
        None, "--nemo-rl", help="NeMo-RL 源码目录（uv run ray 用其环境）"
    ),
) -> None:
    prof_dir = ROOT / "cluster" / profile
    env: dict[str, str] = {}
    if nemo_rl:
        env["NEMO_RL_DIR"] = nemo_rl
    if action is RayAction.status:
        if nemo_rl:
            raise typer.Exit(_run(["uv", "run", "ray", "status"], cwd=Path(nemo_rl)))
        raise typer.Exit(_run(["ray", "status"]))
    if action is RayAction.head:
        raise typer.Exit(_run(["bash", str(prof_dir / "start_ray_head.sh")], env=env))
    if head:
        env["HEAD_ADDRESS"] = head
    raise typer.Exit(_run(["bash", str(prof_dir / "start_ray_worker.sh")], env=env))


if __name__ == "__main__":
    app()
