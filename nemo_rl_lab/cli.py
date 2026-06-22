"""nemo-rl-lab 统一 CLI（Typer 实现）。

所有操作都通过 `lab <子命令>` 执行；内部调用 scripts/ 与各实验脚本（单一事实来源）。
本 CLI 是项目的命令入口（pyproject [project.scripts] lab = nemo_rl_lab.cli:app）。

调用方式（任选）：
    uv run lab ls            # 推荐：uv 自动同步项目环境再运行（对任何人都生效）
    ./lab ls                 # 仓库根的薄 shim，等价于上面那条
    lab ls                   # `uv sync` 后 .venv/bin/lab 已存在；激活 venv 即可直接用

常用：
    uv run lab ls                                  列出实验 / 项目
    uv run lab new grpo_qwen3.5-4b_gsm8k_v1 --cluster h100   从模板新建实验（绑定目标集群）
    uv run lab new my_run --from grpo_qwen3.5-4b_gsm8k_v1   fork 现成实验来调参（继承其集群）
    uv run lab prepare gsm8k                       预处理数据集
    uv run lab submit agent-grpo_qwen3.5-9b_multitool_v1   从本机提交作业到 Ray 集群
    uv run lab job list                            查看集群上的作业（地址取 cluster/<profile>/submit.env）
    uv run lab job logs <job_id> -f                实时看作业日志
    uv run lab job stop <job_id>                   停止作业
    uv run lab web                                 本地 Web 面板：reward 曲线 + 验证对话
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


def _read_env_file(path: Path) -> dict[str, str]:
    """读取一个 env 文件的键值（忽略注释/空行）。"""
    env: dict[str, str] = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env


def _read_submit_env(profile: Optional[str] = None) -> dict[str, str]:
    """分层读取提交配置：通用层 cluster/submit.env（密钥/默认 profile），
    叠加集群专属层 cluster/<profile>/submit.env（地址/容器路径，覆盖通用层）。
    profile 未显式给出时取 环境 CLUSTER_PROFILE > 通用层 DEFAULT_CLUSTER_PROFILE。"""
    shared = _read_env_file(ROOT / "cluster" / "submit.env")
    prof = profile or os.environ.get("CLUSTER_PROFILE") or shared.get("DEFAULT_CLUSTER_PROFILE")
    merged = dict(shared)
    if prof:
        merged.update(_read_env_file(ROOT / "cluster" / prof / "submit.env"))
    return merged


def _ray_address(explicit: Optional[str], profile: Optional[str] = None) -> str:
    """确定 Ray dashboard 地址：显式 > 环境变量 > 分层 submit.env（集群专属层优先）。"""
    addr = explicit or os.environ.get("RAY_DASHBOARD_ADDRESS") or _read_submit_env(profile).get(
        "RAY_DASHBOARD_ADDRESS"
    )
    if not addr:
        prof = profile or os.environ.get("CLUSTER_PROFILE") or "<profile>"
        raise typer.BadParameter(
            f"未找到 Ray 地址。请在 cluster/{prof}/submit.env 设置 RAY_DASHBOARD_ADDRESS，"
            "或用 --address 指定（也可用 --profile 选集群）。"
        )
    return addr


def _ray_job(args: list[str], address: str) -> int:
    """用 uv 管理的 Ray CLI（submit extra，版本对齐集群）执行 `ray job ...`。"""
    return _run(["uv", "run", "--extra", "submit", "ray", "job", *args, "--address", address])


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
    stop = "stop"


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


@app.command(help="新建实验：默认从空白模板；--from 则 fork 现成实验（自动改 SwanLab/README 名）")
def new(
    name: str = typer.Argument(..., help="新实验名（见 docs/naming-convention.md）"),
    from_exp: Optional[str] = typer.Option(
        None, "--from", autocompletion=_complete_exp,
        help="从此现成实验 fork：copy 目录 + 把 config.yaml 的 swanlab project/name 与 README 标题改成新名",
    ),
    cluster: Optional[str] = typer.Option(
        None, "--cluster", autocompletion=_complete_profile,
        help="本实验默认集群（写入实验自带 cluster 文件；不给则用模板默认 / 继承来源实验）",
    ),
    kind: Kind = typer.Option(Kind.experiments, "--kind", help="放到 experiments 还是 projects"),
) -> None:
    cmd = ["bash", str(SCRIPTS / "new_experiment.sh"), kind.value, name]
    # new_experiment.sh 位置参数： <kind> <name> [来源实验] [集群profile]；
    # 只给 --cluster 时也要占住第 3 位（空串=空白模板），让 profile 落到第 4 位。
    if from_exp or cluster:
        cmd.append(_resolve_exp(from_exp) if from_exp else "")
    if cluster:
        cmd.append(cluster)
    raise typer.Exit(_run(cmd))


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
    action: RayAction = typer.Argument(..., help="head | worker | status | stop"),
    profile: str = typer.Option("gb10-spark", "--profile", autocompletion=_complete_profile),
    head: Optional[str] = typer.Option(None, "--head", help="worker 加入的 head 地址，如 192.168.1.4:6379"),
    force: bool = typer.Option(False, "--force", help="stop 时强杀(SIGKILL)残留进程，更彻底释放显存"),
    nemo_rl: Optional[str] = typer.Option(
        None, "--nemo-rl", help="NeMo-RL 源码目录（uv run ray 用其环境）"
    ),
) -> None:
    prof_dir = ROOT / "cluster" / profile
    env: dict[str, str] = {}
    if nemo_rl:
        env["NEMO_RL_DIR"] = nemo_rl

    def _ray_cli(args: list[str]) -> int:
        # ray 由 NeMo-RL 的 uv 环境提供：给了 --nemo-rl 就 `uv run ray`，否则用 PATH 里的 ray
        if nemo_rl:
            return _run(["uv", "run", "ray", *args], cwd=Path(nemo_rl))
        return _run(["ray", *args])

    if action is RayAction.status:
        raise typer.Exit(_ray_cli(["status"]))
    if action is RayAction.stop:
        # ray stop 会终止本节点所有 Ray 进程(含训练/vLLM worker)，从而释放它们占用的 GPU 显存。
        # 注意：这会停掉本机参与的集群，正在跑的训练也会被杀。每个节点都要各自执行。
        raise typer.Exit(_ray_cli(["stop"] + (["--force"] if force else [])))
    if action is RayAction.head:
        raise typer.Exit(_run(["bash", str(prof_dir / "start_ray_head.sh")], env=env))
    if head:
        env["HEAD_ADDRESS"] = head
    raise typer.Exit(_run(["bash", str(prof_dir / "start_ray_worker.sh")], env=env))


# ----------------------------- Ray 作业管理（本机对集群操作）-----------------------------
job_app = typer.Typer(
    no_args_is_help=True,
    help="Ray 作业管理（在本机对集群操作；地址默认取 cluster/<profile>/submit.env 的 RAY_DASHBOARD_ADDRESS，--profile 切集群）",
    context_settings={"help_option_names": ["-h", "--help"]},
)
app.add_typer(job_app, name="job")

_ADDR_OPT = typer.Option(None, "--address", "-a", help="Ray dashboard 地址（默认取 submit.env）")
_PROF_OPT = typer.Option(
    None, "--profile", autocompletion=_complete_profile,
    help="集群 profile（决定读 cluster/<profile>/submit.env 的地址；默认取 submit.env 的 DEFAULT_CLUSTER_PROFILE）",
)


@app.command(help="启动本地 Web 面板：看训练 reward 曲线 + 验证对话（数据取自 Ray 日志，纯本地只读）")
def web(
    port: int = typer.Option(8080, "--port", "-p", help="本地服务端口"),
    address: Optional[str] = _ADDR_OPT,
    profile: Optional[str] = _PROF_OPT,
    no_open: bool = typer.Option(False, "--no-open", help="不自动打开浏览器"),
) -> None:
    # 取数走 ray JobSubmissionClient（同 lab job samples）→ --extra submit；
    # HTTP 服务用 FastAPI + uvicorn → --extra web。一条命令即起。
    cmd = ["uv", "run", "--extra", "submit", "--extra", "web",
           "python", str(SCRIPTS / "web_dashboard.py"),
           "--address", _ray_address(address, profile), "--port", str(port)]
    if not no_open:
        cmd.append("--open")
    raise typer.Exit(_run(cmd))


@job_app.command("list", help="列出集群作业（精简表格）")
def job_list(
    all_jobs: bool = typer.Option(False, "--all", help="显示全部（默认最近 15 条）"),
    address: Optional[str] = _ADDR_OPT,
    profile: Optional[str] = _PROF_OPT,
) -> None:
    cmd = ["uv", "run", "--extra", "submit", "python", str(SCRIPTS / "ray_jobs.py"),
           "--address", _ray_address(address, profile)]
    if all_jobs:
        cmd.append("--all")
    raise typer.Exit(_run(cmd))


@job_app.command("logs", help="查看作业日志（-f 实时跟随）")
def job_logs(
    job_id: str = typer.Argument(..., help="作业 ID（见 lab job list）"),
    follow: bool = typer.Option(False, "--follow", "-f", help="实时跟随输出"),
    address: Optional[str] = _ADDR_OPT,
    profile: Optional[str] = _PROF_OPT,
) -> None:
    args = ["logs", job_id]
    if follow:
        args.append("--follow")
    raise typer.Exit(_ray_job(args, _ray_address(address, profile)))


@job_app.command("samples", help="本地查看验证样本轨迹（含多轮工具调用，从作业日志抽取）")
def job_samples(
    job_id: str = typer.Argument(..., help="作业 ID（见 lab job list）"),
    last: int = typer.Option(0, "--last", "-n", help="只看最近 N 次验证（默认 0=全部）"),
    address: Optional[str] = _ADDR_OPT,
    profile: Optional[str] = _PROF_OPT,
) -> None:
    cmd = ["uv", "run", "--extra", "submit", "python", str(SCRIPTS / "job_samples.py"),
           job_id, "--address", _ray_address(address, profile)]
    if last:
        cmd += ["--last", str(last)]
    raise typer.Exit(_run(cmd))


@job_app.command("status", help="查看作业状态")
def job_status(
    job_id: str = typer.Argument(..., help="作业 ID"),
    address: Optional[str] = _ADDR_OPT,
    profile: Optional[str] = _PROF_OPT,
) -> None:
    raise typer.Exit(_ray_job(["status", job_id], _ray_address(address, profile)))


@job_app.command("stop", help="停止作业（运行中 → 终止）")
def job_stop(
    job_id: str = typer.Argument(..., help="作业 ID"),
    address: Optional[str] = _ADDR_OPT,
    profile: Optional[str] = _PROF_OPT,
) -> None:
    raise typer.Exit(_ray_job(["stop", job_id], _ray_address(address, profile)))


@job_app.command("delete", help="删除某个已结束的作业记录（运行中需先 stop）")
def job_delete(
    job_id: str = typer.Argument(..., help="作业 ID"),
    address: Optional[str] = _ADDR_OPT,
    profile: Optional[str] = _PROF_OPT,
) -> None:
    raise typer.Exit(_ray_job(["delete", job_id], _ray_address(address, profile)))


@job_app.command("clean", help="批量删除所有已结束(FAILED/SUCCEEDED/STOPPED)的作业记录")
def job_clean(
    yes: bool = typer.Option(False, "--yes", "-y", help="跳过确认"),
    address: Optional[str] = _ADDR_OPT,
    profile: Optional[str] = _PROF_OPT,
) -> None:
    cmd = ["uv", "run", "--extra", "submit", "python", str(SCRIPTS / "ray_jobs.py"),
           "--address", _ray_address(address, profile), "--clean"]
    if yes:
        cmd.append("--yes")
    raise typer.Exit(_run(cmd))


if __name__ == "__main__":
    app()
