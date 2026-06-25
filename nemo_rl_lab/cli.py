"""nemo-rl-lab 统一 CLI（Typer 实现）。

所有操作都通过 `lab <子命令>` 执行；内部调用 nemo_rl_lab/ 与各实验脚本（集群侧 run.sh 仍在 Linux 容器执行）。
本 CLI 是项目的命令入口（pyproject [project.scripts] lab = nemo_rl_lab.cli:app）。

提交一律走【中心化 Lab 服务】：先 `lab login` 接入服务，`lab submit` 把工作目录打包上传，
服务端注入密钥 / 路径 / 数据目录，并在集群上代理执行（Ray 地址、密钥都在服务端，本机不直连 Ray、
不读任何 submit.env）。

    调用方式（任选）：
    uv run lab ls            # 推荐：uv 自动同步项目环境再运行（macOS / Linux / Windows）
    ./lab ls                 # macOS/Linux 薄 shim，等价于上面那条
    lab.cmd ls               # Windows 薄 shim，等价于上面那条
    lab ls                   # uv sync 后 venv 内 lab 已存在；激活 venv 即可直接用

常用：
    uv run lab login --server https://lab.company.com   接入中心化服务（鉴权/配额/监控）
    uv run lab ls                                  列出实验 / 项目
    uv run lab new grpo_qwen3.5-4b_gsm8k_v1 --method grpo --cluster h100   从骨架新建实验（grpo|sft|agent）
    uv run lab new my_run --from grpo_qwen3.5-4b_gsm8k_v1   fork 现成实验来调参（继承其集群）
    uv run lab diff grpo_qwen3.5-4b_gsm8k_v1 grpo_qwen3.5-9b_gsm8k_v1   对比两实验有效 config 差异
    uv run lab prepare gsm8k                       预处理数据集
    uv run lab doctor                              体检中心化服务连通 / 登录态
    uv run lab validate grpo_qwen3.5-4b_gsm8k_v1   提交前静态校验 config（batch 三者相等等）
    uv run lab submit agent-grpo_qwen3.5-9b_multitool_v1   提交作业到集群（经服务端，自动先校验）
    uv run lab status                              我的配额 / 用量 / 活跃作业
    uv run lab logs                                跟随最近一个作业的日志
    uv run lab job ls                              我的作业列表
    uv run lab clean grpo_qwen3.5-9b_gsm8k_v1      清理该实验在集群上的产物目录（让同名实验下次从头训练）
    uv run lab export grpo_qwen3.5-9b_gsm8k_v1     把 checkpoint 转 HF（自适应 dcp/megatron），可 --push-repo 推 Hub
    uv run lab eval grpo_qwen3.5-9b_gsm8k_v1       对 checkpoint 跑独立评测（未给 --model 先自动导出）
    uv run lab runs                                我的提交历史（服务端台账）
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

from nemo_rl_lab import cli_login
from nemo_rl_lab.new_experiment import NewExperimentError, create_experiment
from nemo_rl_lab.sync_base import SyncBaseError, sync_base_configs

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


# 共享的 profile 选项（submit/export/eval 提交时把硬件 profile 转发给服务端，决定集群 overrides）。
_PROF_OPT = typer.Option(
    None, "--profile", autocompletion=_complete_profile,
    help="硬件 profile（决定集群 cluster/<profile>/overrides.conf；提交时转发给服务端，默认用实验自带 cluster 文件）",
)


# ----------------------------- 选择项 -----------------------------
class Kind(str, Enum):
    experiments = "experiments"
    projects = "projects"


class Method(str, Enum):
    """训练方法骨架。agent 本质是 GRPO 的多轮变体（base=grpo_sliding_puzzle + 自定义 run.py 环境）。"""
    grpo = "grpo"
    sft = "sft"
    agent = "agent"


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


@app.command(help="新建实验：默认从空白模板（--method 选骨架）；--from 则 fork 现成实验（自动改 SwanLab/README 名）")
def new(
    name: str = typer.Argument(..., help="新实验名（见 docs/naming-convention.md）"),
    from_exp: Optional[str] = typer.Option(
        None, "--from", autocompletion=_complete_exp,
        help="从此现成实验 fork：copy 目录 + 把 config.yaml 的 swanlab project/name 与 README 标题改成新名",
    ),
    method: Method = typer.Option(
        Method.grpo, "--method", "-m",
        help="空白模板的训练方法骨架：grpo（默认）| sft | agent（=GRPO 多轮，含 env+run.py 骨架）。--from 时忽略",
    ),
    cluster: Optional[str] = typer.Option(
        None, "--cluster", autocompletion=_complete_profile,
        help="本实验默认集群（写入实验自带 cluster 文件；不给则用模板默认 / 继承来源实验）",
    ),
    kind: Kind = typer.Option(Kind.experiments, "--kind", help="放到 experiments 还是 projects"),
) -> None:
    if from_exp and method is not Method.grpo:
        typer.secho("注：--from（fork）会继承来源实验的方法/配置，--method 被忽略。", fg=typer.colors.YELLOW)
    src = ""
    if from_exp:
        src = Path(_resolve_exp(from_exp)).name
    try:
        create_experiment(
            ROOT,
            kind.value,
            name,
            src=src,
            cluster=cluster or "",
            method=method.value,
        )
    except NewExperimentError as e:
        typer.secho(str(e), fg=typer.colors.RED)
        raise typer.Exit(1) from None


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


def _validate_exp(exp_path: str) -> tuple[list[str], list[str]]:
    """解析 + 校验某实验 config，打印问题，返回 (errors, warns)。解析失败按 1 个 error 计。"""
    from nemo_rl_lab.config_resolve import resolve, validate_config

    cfg_file = ROOT / exp_path / "config.yaml"
    if not cfg_file.is_file():
        return [f"实验缺少 config.yaml: {exp_path}"], []
    try:
        cfg = resolve(cfg_file)
    except Exception as e:  # noqa: BLE001
        return [f"解析 config 失败: {e}"], []
    issues = validate_config(cfg, repo_root=ROOT)
    errors = [m for lvl, m in issues if lvl == "error"]
    warns = [m for lvl, m in issues if lvl == "warn"]
    for m in errors:
        typer.secho(f"  ✗ {m}", fg=typer.colors.RED)
    for m in warns:
        typer.secho(f"  ! {m}", fg=typer.colors.YELLOW)
    return errors, warns


@app.command(help="提交作业到集群（经中心化服务执行；提交前自动校验 config）")
def submit(
    exp: str = typer.Argument(..., autocompletion=_complete_exp, help="实验名或路径"),
    profile: Optional[str] = _PROF_OPT,
    no_validate: bool = typer.Option(False, "--no-validate", help="跳过提交前 config 校验"),
) -> None:
    cli_login.gate("submit")
    exp_path = _resolve_exp(exp)
    if not no_validate:
        errors, _ = _validate_exp(exp_path)
        if errors:
            typer.secho(
                f"\n{exp_path}: config 校验未通过（{len(errors)} 个错误）。"
                "修复后再 submit，或 --no-validate 强制提交。",
                fg=typer.colors.RED,
            )
            raise typer.Exit(1)
    # 打包 working-dir → 上传到中心化服务 → 服务端注入密钥/路径后代理提交（密钥/地址不外泄）。
    res = cli_login.submit_via_server(exp_path, profile, ROOT)
    gpus = res.get("requested_gpus")
    typer.secho(
        f"✓ 已提交：job={res.get('job_id')}  run_id={res.get('run_id')}"
        + (f"  GPU={gpus}" if gpus is not None else "")
        + ("  [dry-run]" if res.get("dry_run") else ""),
        fg=typer.colors.GREEN,
    )
    typer.echo(f"  跟随日志：lab logs {res.get('job_id')}")


@app.command(
    help="清理某实验在集群上的产物目录（checkpoint/日志），让同名实验下次从头训练。"
    "经服务端在集群侧删除（fan-out 所有节点，兼容共享盘/本地盘）；本实验有活跃作业时会被拒绝。"
)
def clean(
    exp: str = typer.Argument(..., autocompletion=_complete_exp, help="实验名或路径"),
    yes: bool = typer.Option(False, "-y", "--yes", help="跳过确认"),
) -> None:
    cli_login.gate("clean")
    exp_path = _resolve_exp(exp)
    if not yes:
        typer.confirm(
            f"将删除集群上 {exp_path} 的产物目录（checkpoint/日志），不可恢复。确认清理？",
            abort=True,
        )
    res = cli_login.clean_via_server(exp_path)
    typer.secho(
        f"✓ 已提交清理：job={res.get('job_id')}  目标={res.get('target')}"
        + ("  [dry-run]" if res.get("dry_run") else ""),
        fg=typer.colors.GREEN,
    )
    typer.echo(f"  查看清理结果：lab logs {res.get('job_id')}")


@app.command(help="提交前静态校验实验 config（batch 三者相等等；本地秒级，省得跑到集群才报错）")
def validate(
    exp: str = typer.Argument(..., autocompletion=_complete_exp, help="实验名或路径"),
) -> None:
    exp_path = _resolve_exp(exp)
    errors, warns = _validate_exp(exp_path)
    if errors:
        typer.secho(
            f"\n{exp_path}: {len(errors)} 个错误，{len(warns)} 个告警 —— 修复后再 submit。",
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)
    suffix = f"（{len(warns)} 个告警）" if warns else ""
    typer.secho(f"✓ {exp_path}: 校验通过{suffix}", fg=typer.colors.GREEN)


def _flatten(obj, prefix: str = "") -> dict[str, str]:
    """把嵌套 config 拍平成 点路径 -> 标量字符串，便于逐键对比。"""
    out: dict[str, str] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out.update(_flatten(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = "null" if obj is None else str(obj)
    return out


@app.command(help="对比两实验 config 差异（默认解析后的语义 diff；fork 调参时看改了哪些键）")
def diff(
    exp_a: str = typer.Argument(..., autocompletion=_complete_exp, help="实验 A（基准）"),
    exp_b: str = typer.Argument(..., autocompletion=_complete_exp, help="实验 B（对比）"),
    raw: bool = typer.Option(False, "--raw", help="改为对两个 config.yaml 原文做逐行 diff（含注释）"),
) -> None:
    from nemo_rl_lab.config_resolve import resolve

    pa, pb = _resolve_exp(exp_a), _resolve_exp(exp_b)
    fa, fb = ROOT / pa / "config.yaml", ROOT / pb / "config.yaml"
    for f in (fa, fb):
        if not f.is_file():
            raise typer.BadParameter(f"缺少 config.yaml: {f.relative_to(ROOT)}")

    if raw:
        import difflib

        lines = difflib.unified_diff(
            fa.read_text().splitlines(), fb.read_text().splitlines(),
            fromfile=str(fa.relative_to(ROOT)), tofile=str(fb.relative_to(ROOT)), lineterm="",
        )
        any_line = False
        for ln in lines:
            any_line = True
            if ln.startswith("+") and not ln.startswith("+++"):
                typer.secho(ln, fg=typer.colors.GREEN)
            elif ln.startswith("-") and not ln.startswith("---"):
                typer.secho(ln, fg=typer.colors.RED)
            elif ln.startswith("@@"):
                typer.secho(ln, fg=typer.colors.CYAN)
            else:
                typer.echo(ln)
        if not any_line:
            typer.secho("两个 config.yaml 原文完全一致。", fg=typer.colors.GREEN)
        return

    try:
        ca, cb = _flatten(resolve(fa)), _flatten(resolve(fb))
    except Exception as e:  # noqa: BLE001
        raise typer.BadParameter(f"解析 config 失败：{e}") from None

    keys = sorted(set(ca) | set(cb))
    changed = [(k, ca[k], cb[k]) for k in keys if k in ca and k in cb and ca[k] != cb[k]]
    only_a = [(k, ca[k]) for k in keys if k in ca and k not in cb]
    only_b = [(k, cb[k]) for k in keys if k in cb and k not in ca]

    typer.echo(f"A = {pa}\nB = {pb}\n（对比解析后的有效 config；A→B）")
    if not (changed or only_a or only_b):
        typer.secho("\n两实验解析后的有效配置完全一致。", fg=typer.colors.GREEN)
        return
    if changed:
        typer.secho(f"\n改动（{len(changed)}）：", bold=True)
        for k, va, vb in changed:
            typer.echo(f"  {k}: ")
            typer.secho(f"    - {va}", fg=typer.colors.RED)
            typer.secho(f"    + {vb}", fg=typer.colors.GREEN)
    if only_a:
        typer.secho(f"\n仅 A 有（B 缺失，{len(only_a)}）：", bold=True)
        for k, v in only_a:
            typer.secho(f"  - {k} = {v}", fg=typer.colors.RED)
    if only_b:
        typer.secho(f"\n仅 B 有（A 缺失，{len(only_b)}）：", bold=True)
        for k, v in only_b:
            typer.secho(f"  + {k} = {v}", fg=typer.colors.GREEN)
    typer.echo(f"\n小结：改 {len(changed)}，A 独有 {len(only_a)}，B 独有 {len(only_b)}")


@app.command(help="体检中心化服务：是否已登录 / 服务是否可达 / 当前配额")
def doctor() -> None:
    srv = cli_login.current_server()
    if not srv:
        typer.secho("未接入中心化服务：先 `lab login --server <https://lab.company.com>`。", fg=typer.colors.RED)
        raise typer.Exit(1)
    typer.echo(f"中心化服务  {srv}")
    token = cli_login.get_access_token(srv)
    if not token:
        typer.secho("  ✗ 未登录：先 `lab login`（集群命令会自动跳浏览器认证）", fg=typer.colors.RED)
        raise typer.Exit(1)
    try:
        who = cli_login.usage_via_server()
        q = who.get("quota") or {}
        typer.secho("  ✓ 已登录且服务可达（/api/usage/mine）", fg=typer.colors.GREEN)
        cap = q.get("max_concurrent_gpus")
        typer.secho(
            f"  ✓ 配额：GPU≤{'不限' if cap is None else cap}，作业≤{q.get('max_concurrent_jobs') or '不限'}",
            fg=typer.colors.GREEN,
        )
    except Exception as e:  # noqa: BLE001
        typer.secho(f"  ✗ 服务不可达或登录失效：{type(e).__name__}", fg=typer.colors.RED)
        raise typer.Exit(1) from None
    typer.secho("\n✓ 体检通过，可以 lab submit 了。", fg=typer.colors.GREEN)


# ----------------------------- 训练后闭环（export / eval；提交到集群执行）-----------------------------
def _submit_post(action: str, exp_path: str, profile: Optional[str], flags: list[str], dry_run: bool) -> int:
    """把 export/eval 作业经服务端代理提交到集群（入口 scripts/post_train.sh）。"""
    cli_login.gate(action)  # export / eval：需登录
    if dry_run:
        typer.secho("（dry-run 由服务端 LAB_SUBMIT_DRY_RUN 控制）", fg=typer.colors.YELLOW)
    res = cli_login.submit_post_via_server(action, exp_path, profile, flags, ROOT)
    gpus = res.get("requested_gpus")
    typer.secho(
        f"✓ 已提交 [{action}]：job={res.get('job_id')}  run_id={res.get('run_id')}"
        + (f"  GPU={gpus}" if gpus is not None else "")
        + ("  [dry-run]" if res.get("dry_run") else ""),
        fg=typer.colors.GREEN,
    )
    typer.echo(f"  跟随日志：lab logs {res.get('job_id')}")
    return 0


@app.command(name="export", help="把训练 checkpoint 转成 HF 格式（按后端自适应 dcp/megatron），可选推 HF Hub；执行在集群")
def export_ckpt(
    exp: str = typer.Argument(..., autocompletion=_complete_exp, help="实验名或路径"),
    step: Optional[int] = typer.Option(None, "--step", help="checkpoint 步数（默认最新 step_<N>）"),
    out: Optional[str] = typer.Option(None, "--out", help="HF 输出目录（容器内；默认 <ckpt>/hf_export/step_<N>）"),
    push_repo: Optional[str] = typer.Option(None, "--push-repo", help="转换后上传到 HF Hub repo（user/name，需 HF_TOKEN）"),
    ckpt_dir: Optional[str] = typer.Option(None, "--ckpt-dir", help="覆盖 checkpoint 根目录（容器内绝对路径）"),
    profile: Optional[str] = _PROF_OPT,
    dry_run: bool = typer.Option(False, "--dry-run", help="只打印将提交的命令，不实际提交"),
) -> None:
    flags: list[str] = []
    if step is not None:
        flags += ["--step", str(step)]
    if out:
        flags += ["--out", out]
    if push_repo:
        flags += ["--push-repo", push_repo]
    if ckpt_dir:
        flags += ["--ckpt-dir", ckpt_dir]
    raise typer.Exit(_submit_post("export", _resolve_exp(exp), profile, flags, dry_run))


@app.command(
    name="eval",
    help="对 checkpoint 跑独立评测（run_eval.py，仅吃 HF 格式；未给 --model 时先自动导出）；执行在集群。"
    "额外的 NeMo-RL 覆盖项写在 `--` 之后，如：lab eval <exp> -- generation.temperature=0.6",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def eval_ckpt(
    ctx: typer.Context,
    exp: str = typer.Argument(..., autocompletion=_complete_exp, help="实验名或路径"),
    step: Optional[int] = typer.Option(None, "--step", help="checkpoint 步数（默认最新；给 --model 时忽略）"),
    model: Optional[str] = typer.Option(None, "--model", help="直接评测此 HF 模型路径/Hub id（给了就跳过导出）"),
    eval_config: Optional[str] = typer.Option(None, "--eval-config", help="NeMo-RL 评测配置（默认 examples/configs/evals/eval.yaml）"),
    ckpt_dir: Optional[str] = typer.Option(None, "--ckpt-dir", help="覆盖 checkpoint 根目录（容器内绝对路径）"),
    profile: Optional[str] = _PROF_OPT,
    dry_run: bool = typer.Option(False, "--dry-run", help="只打印将提交的命令，不实际提交"),
) -> None:
    flags: list[str] = []
    if step is not None:
        flags += ["--step", str(step)]
    if model:
        flags += ["--model", model]
    if eval_config:
        flags += ["--eval-config", eval_config]
    if ckpt_dir:
        flags += ["--ckpt-dir", ckpt_dir]
    extra = list(ctx.args)  # `--` 之后透传给 run_eval.py 的覆盖项
    if extra:
        flags += ["--", *extra]
    raise typer.Exit(_submit_post("eval", _resolve_exp(exp), profile, flags, dry_run))


@app.command(name="sync-base", help="同步官方基底配置到 configs/base/")
def sync_base(
    nemo_rl: Optional[str] = typer.Option(None, "--nemo-rl", help="NeMo-RL 源码目录"),
) -> None:
    path = nemo_rl or os.environ.get("NEMO_RL_DIR")
    if not path:
        typer.secho("请设置 NEMO_RL_DIR 或使用 --nemo-rl 指向本地 NeMo-RL 源码目录", fg=typer.colors.RED)
        raise typer.Exit(1)
    try:
        sync_base_configs(ROOT, path)
    except SyncBaseError as e:
        typer.secho(str(e), fg=typer.colors.RED)
        raise typer.Exit(1) from None


# ----------------------------- 提交历史 / 状态 / 日志（经服务端）-----------------------------
@app.command(help="查看我的提交历史（服务端台账：每次 submit/export/eval 的 run_id / 状态 / GPU）")
def runs(
    all_runs: bool = typer.Option(False, "--all", help="显示全部（默认最近 20 条）"),
    exp: Optional[str] = typer.Option(
        None, "--exp", autocompletion=_complete_exp, help="只看某实验（接受全名或末段名）"
    ),
    limit: int = typer.Option(20, "-n", "--limit", help="显示条数（--all 时忽略）"),
) -> None:
    cli_login.gate("status")
    jobs = cli_login.list_my_jobs(limit=200 if all_runs else limit)
    if exp:
        jobs = [j for j in jobs if exp in (j.get("exp") or "")]
    if not jobs:
        typer.echo("（服务端没有你的作业记录；lab submit 后会出现）")
        raise typer.Exit(0)
    typer.echo(f"{'TIME':<20} {'STATUS':<10} {'GPU':>4}  {'EXP':<28} RUN_ID")
    for j in jobs:
        typer.echo(
            f"{str(j.get('submitted_at','-'))[:19]:<20} {str(j.get('status','-')):<10} "
            f"{str(j.get('requested_gpus') or '-'):>4}  {Path(str(j.get('exp','-'))).name:<28} {j.get('lab_run_id','-')}"
        )


@app.command(help="我的配额 / 用量 / 活跃作业（submit 前预检）")
def status() -> None:
    cli_login.gate("status")
    data = cli_login.usage_via_server()
    q, u = data.get("quota") or {}, data.get("usage") or {}
    cap = q.get("max_concurrent_gpus")
    typer.echo("我的用量")
    typer.echo(f"  并发 GPU : {u.get('active_gpus', 0)} / {'不限' if cap is None else cap}")
    typer.echo(f"  并发作业 : {u.get('active_jobs', 0)} / {q.get('max_concurrent_jobs') or '不限'}")
    typer.echo(f"  今日/累计 GPU-hours : {u.get('gpu_hours_today', 0):.1f} / {u.get('gpu_hours_total', 0):.1f}")
    running = u.get("running") or []
    typer.echo("\n我的活跃作业")
    if not running:
        typer.echo("  （无）")
    else:
        for r in running:
            jid = (r.get("ray_submission_id") or r.get("lab_run_id") or "-")[:26]
            typer.echo(f"  {jid:<26} {r.get('status','-'):<10} GPU={r.get('gpus') or '-'}  {r.get('exp','-')}")

    cluster = cli_login.cluster_status_via_server()
    gpu = (cluster or {}).get("gpu") or {}
    if gpu:
        accel = "/".join(gpu.get("accel") or []) or "GPU"
        typer.echo("\n集群 GPU")
        typer.echo(
            f"  {accel} : 空闲 {gpu.get('gpu_free', 0):g} / 共 {gpu.get('gpu_total', 0):g}"
            f"（占用 {gpu.get('gpu_used', 0):g}）"
        )
        typer.echo(f"  活跃作业 : {cluster.get('active_count', 0)}")
    typer.echo("\n详情/日志： lab logs <JOB ID>")


@app.command(help="看作业日志：不给 job_id 默认跟随【最近一个】作业（经服务端转发）")
def logs(
    job_id: Optional[str] = typer.Argument(None, help="作业 ID（见 lab job ls）；省略=最近一个"),
    tail: Optional[int] = typer.Option(
        None, "-n", "--tail", help="只回放最后 N 行历史日志再跟随（省略=从头全量）"
    ),
) -> None:
    cli_login.gate("logs")
    jid = job_id or cli_login.latest_job_via_server()
    if not jid:
        typer.secho("没有可跟随的作业（先 lab submit）。", fg=typer.colors.YELLOW)
        raise typer.Exit(1)
    cli_login.stream_logs_via_server(jid, tail=tail)


# ----------------------------- 作业管理（经服务端）-----------------------------
job_app = typer.Typer(
    no_args_is_help=True,
    help="作业管理（经中心化服务：列出 / 看日志 / 状态 / 停止 / 删除我的作业）",
    context_settings={"help_option_names": ["-h", "--help"]},
)
app.add_typer(job_app, name="job")


def _server_jobs_table(jobs: list[dict]) -> None:
    if not jobs:
        typer.echo("（无作业）")
        return
    typer.echo(f"{'JOB ID':<26} {'状态':<10} {'GPU':>4}  实验")
    for j in jobs:
        jid = (j.get("ray_submission_id") or j.get("lab_run_id") or "-")[:26]
        typer.echo(f"{jid:<26} {str(j.get('status','-')):<10} {str(j.get('requested_gpus') or '-'):>4}  {j.get('exp','-')}")


@job_app.command("ls", help="列出我的集群作业（精简表格）")
def job_ls(
    all_jobs: bool = typer.Option(False, "--all", help="显示全部（默认最近 15 条）"),
) -> None:
    cli_login.gate("job-list")
    _server_jobs_table(cli_login.list_my_jobs(limit=200 if all_jobs else 15))


@job_app.command("logs", help="查看作业日志（实时跟随）")
def job_logs(
    job_id: str = typer.Argument(..., help="作业 ID（见 lab job ls）"),
    tail: Optional[int] = typer.Option(
        None, "-n", "--tail", help="只回放最后 N 行历史日志再跟随（省略=从头全量）"
    ),
) -> None:
    cli_login.gate("job-logs")
    cli_login.stream_logs_via_server(job_id, tail=tail)


@job_app.command("status", help="查看作业状态")
def job_status(
    job_id: str = typer.Argument(..., help="作业 ID"),
) -> None:
    cli_login.gate("job-status")
    match = [j for j in cli_login.list_my_jobs(limit=200)
             if job_id in (j.get("ray_submission_id") or "", j.get("lab_run_id") or "")]
    if not match:
        typer.secho(f"未找到作业 {job_id}", fg=typer.colors.RED)
        raise typer.Exit(1)
    _server_jobs_table(match)


@job_app.command("samples", help="查看某次验证的多轮对话轨迹（默认最近一次验证）")
def job_samples(
    job_id: str = typer.Argument(..., help="作业 ID（见 lab job ls）"),
    vidx: int = typer.Option(-1, "--vidx", help="验证轮次下标（默认 -1=最近一次）"),
    n: int = typer.Option(6, "-n", "--limit", help="显示样本条数"),
) -> None:
    cli_login.gate("job-samples")
    overview = cli_login.job_overview_via_server(job_id)
    vals = overview.get("validations") or []
    if not vals:
        typer.secho("该作业暂无验证样本（可能尚未跑到验证步）。", fg=typer.colors.YELLOW)
        raise typer.Exit(1)
    idx = vidx if vidx >= 0 else len(vals) + vidx
    if idx < 0 or idx >= len(vals):
        typer.secho(f"验证下标越界：vidx={vidx}，共 {len(vals)} 轮。", fg=typer.colors.RED)
        raise typer.Exit(1)
    page = cli_login.samples_via_server(job_id, idx, 0, n)
    samples = page.get("samples") or []
    typer.echo(
        f"验证 step={page.get('step', '?')}（第 {idx + 1}/{len(vals)} 轮）  "
        f"样本 {len(samples)}/{page.get('total', len(samples))}"
    )
    for s in samples:
        typer.echo("")
        typer.secho(f"── Sample {s.get('idx', '?')} | reward={s.get('reward', '?')} ──", fg=typer.colors.CYAN)
        if s.get("user"):
            typer.secho("USER:", fg=typer.colors.GREEN)
            typer.echo(s["user"])
        if s.get("assistant"):
            typer.secho("ASSISTANT:", fg=typer.colors.BLUE)
            typer.echo(s["assistant"])
        if s.get("env"):
            typer.secho("ENVIRONMENT:", fg=typer.colors.MAGENTA)
            typer.echo(s["env"])


@job_app.command("stop", help="停止作业（运行中 → 终止）")
def job_stop(
    job_id: str = typer.Argument(..., help="作业 ID"),
) -> None:
    cli_login.gate("job-stop")
    res = cli_login.job_control_via_server("stop", job_id)
    typer.secho(f"✓ stop: {res}", fg=typer.colors.GREEN)


@job_app.command("delete", help="删除某个已结束的作业记录（运行中需先 stop）")
def job_delete(
    job_id: str = typer.Argument(..., help="作业 ID"),
) -> None:
    cli_login.gate("job-delete")
    res = cli_login.job_control_via_server("delete", job_id)
    typer.secho(f"✓ delete: {res}", fg=typer.colors.GREEN)


@job_app.command("cancel-all", help="停止我所有运行中 / 排队中的作业")
def job_cancel_all(
    yes: bool = typer.Option(False, "-y", "--yes", help="跳过确认"),
) -> None:
    cli_login.gate("job-cancel-all")
    if not yes:
        typer.confirm("将停止你【全部】运行中/排队中的作业，确认？", abort=True)
    res = cli_login.batch_via_server("cancel-all")
    typer.secho(f"✓ 已停止 {res.get('stopped', 0)} 个作业", fg=typer.colors.GREEN)


@job_app.command("clean", help="清理我已结束作业的集群记录（dashboard 清理，保留台账历史）")
def job_clean() -> None:
    cli_login.gate("job-clean")
    res = cli_login.batch_via_server("clean")
    typer.secho(f"✓ 已清理 {res.get('deleted', 0)} 个终态作业记录", fg=typer.colors.GREEN)


# ----------------------------- 中心化 Lab 服务：登录/身份/配额 -----------------------------
app.command(help="登录中心化 Lab 服务（--server 指定地址；未登录的集群命令会自动跳浏览器认证）")(cli_login.login)
app.command(help="登出：吊销 refresh 并清除本地凭据")(cli_login.logout)
app.command(help="显示当前登录身份 / 角色 / 配额")(cli_login.whoami)
app.command(help="查看配额与实时用量")(cli_login.quota)
app.add_typer(cli_login.admin_app, name="admin")


if __name__ == "__main__":
    app()
