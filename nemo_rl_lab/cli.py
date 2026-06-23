"""nemo-rl-lab 统一 CLI（Typer 实现）。

所有操作都通过 `lab <子命令>` 执行；内部调用 scripts/ 与各实验脚本（单一事实来源）。
本 CLI 是项目的命令入口（pyproject [project.scripts] lab = nemo_rl_lab.cli:app）。

调用方式（任选）：
    uv run lab ls            # 推荐：uv 自动同步项目环境再运行（对任何人都生效）
    ./lab ls                 # 仓库根的薄 shim，等价于上面那条
    lab ls                   # `uv sync` 后 .venv/bin/lab 已存在；激活 venv 即可直接用

常用：
    uv run lab init                                交互式引导首次配置两层 submit.env
    uv run lab ls                                  列出实验 / 项目
    uv run lab new grpo_qwen3.5-4b_gsm8k_v1 --method grpo --cluster h100   从骨架新建实验（grpo|sft|agent）
    uv run lab new my_run --from grpo_qwen3.5-4b_gsm8k_v1   fork 现成实验来调参（继承其集群）
    uv run lab diff grpo_qwen3.5-4b_gsm8k_v1 grpo_qwen3.5-9b_gsm8k_v1   对比两实验有效 config 差异
    uv run lab prepare gsm8k                       预处理数据集
    uv run lab doctor                              体检提交环境（配置/连通/Ray 版本对齐）
    uv run lab tunnel                              开 SSH 隧道转发 dashboard/GCS 端口（不同网段时）
    uv run lab cluster up                          远程 ssh+docker exec 起 Ray（head+worker）
    uv run lab validate grpo_qwen3.5-4b_gsm8k_v1   提交前静态校验 config（batch 三者相等等）
    uv run lab submit agent-grpo_qwen3.5-9b_multitool_v1   从本机提交作业到 Ray 集群（自动先校验）
    uv run lab status                              集群一览：空闲 GPU + 活跃作业（submit 前预检）
    uv run lab job list                            查看集群上的作业（地址取 cluster/<profile>/submit.env）
    uv run lab logs                                跟随最近一个作业的日志（= lab job logs 便捷版）
    uv run lab export grpo_qwen3.5-9b_gsm8k_v1     把 checkpoint 转 HF（自适应 dcp/megatron），可 --push-repo 推 Hub
    uv run lab eval grpo_qwen3.5-9b_gsm8k_v1       对 checkpoint 跑独立评测（未给 --model 先自动导出）
    uv run lab runs --status                       本地台账并关联集群作业状态（这次提交跑成没）
    uv run lab job cancel-all                      停止所有运行中/等待中作业（clean 只删终态记录）
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


def _set_env_line(path: Path, key: str, value: str) -> None:
    """在 env 文件里写入 KEY=value：命中已有（含被注释的 `# KEY=`）则原位替换，否则追加。"""
    import re

    lines = path.read_text().splitlines() if path.is_file() else []
    pat = re.compile(rf"^\s*#?\s*{re.escape(key)}=")
    for i, ln in enumerate(lines):
        if pat.match(ln):
            lines[i] = f"{key}={value}"
            break
    else:
        lines.append(f"{key}={value}")
    path.write_text("\n".join(lines) + "\n")


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


# ----------------------------- Ray dashboard HTTP（纯 urllib，无需 ray 依赖）-----------------------------
_STATUS_COLOR = {
    "RUNNING": typer.colors.CYAN,
    "PENDING": typer.colors.MAGENTA,
    "SUCCEEDED": typer.colors.GREEN,
    "FAILED": typer.colors.RED,
    "STOPPED": typer.colors.YELLOW,
}


def _dashboard_get(address: str, path: str, timeout: float = 6.0):
    """GET dashboard 的 JSON 接口（/api/jobs/、/api/cluster_status）。"""
    import json
    import urllib.request

    with urllib.request.urlopen(f"{address.rstrip('/')}{path}", timeout=timeout) as r:
        return json.loads(r.read())


def _fetch_jobs(address: str) -> list[dict]:
    """取集群作业列表（/api/jobs/ 返回 list 或 dict）。"""
    data = _dashboard_get(address, "/api/jobs/")
    return data if isinstance(data, list) else list(data.values())


def _run_status_map(address: str) -> dict[str, str]:
    """lab_run_id（作业 metadata）-> 状态，用于把本地台账对上集群作业。"""
    out: dict[str, str] = {}
    for j in _fetch_jobs(address):
        meta = j.get("metadata") or {}
        rid = meta.get("lab_run_id")
        if rid:
            out[rid] = str(j.get("status", "?"))
    return out


def _gpu_summary(address: str) -> Optional[dict]:
    """从 /api/cluster_status 解析整集群资源使用：GPU/CPU/内存 的 [已用, 总量]。"""
    data = _dashboard_get(address, "/api/cluster_status")
    usage = (
        (((data.get("data") or {}).get("clusterStatus") or {}).get("loadMetricsReport") or {})
        .get("usage")
        or {}
    )
    if not usage:
        return None
    accel = [k.split(":", 1)[1] for k in usage if k.lower().startswith(("acceleratortype:", "accelerator_type:"))]
    return {"usage": usage, "accel": accel}


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


# 共享的地址 / profile 选项单例（job、web、runs、status 等连集群的命令共用）。
_ADDR_OPT = typer.Option(None, "--address", "-a", help="Ray dashboard 地址（默认取 submit.env）")
_PROF_OPT = typer.Option(
    None, "--profile", autocompletion=_complete_profile,
    help="集群 profile（决定读 cluster/<profile>/submit.env 的地址；默认取 submit.env 的 DEFAULT_CLUSTER_PROFILE）",
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
    cmd = ["bash", str(SCRIPTS / "new_experiment.sh"), kind.value, name]
    # new_experiment.sh 位置参数： <kind> <name> [来源实验] [集群profile]；
    # 只给 --cluster 时也要占住第 3 位（空串=空白模板），让 profile 落到第 4 位。
    if from_exp or cluster:
        cmd.append(_resolve_exp(from_exp) if from_exp else "")
    if cluster:
        cmd.append(cluster)
    # method 经环境变量传给脚本（仅空白模板分支用），避免再加一个位置参数。
    raise typer.Exit(_run(cmd, env={"LAB_METHOD": method.value}))


@app.command(help="交互式引导首次配置两层 submit.env（替代手动 cp + 编辑；密钥只写本地、已 .gitignore）")
def init(
    profile: Optional[str] = typer.Option(
        None, "--profile", autocompletion=_complete_profile, help="只初始化某集群层（默认问你选）"
    ),
) -> None:
    import shutil

    cluster_dir = ROOT / "cluster"
    profiles = _list_profiles()

    # ---------- 通用层 cluster/submit.env ----------
    typer.secho("① 通用层 cluster/submit.env（密钥 / 默认 profile，跨集群通用）", bold=True)
    shared = cluster_dir / "submit.env"
    shared_eg = cluster_dir / "submit.env.example"
    do_shared = True
    if shared.is_file():
        do_shared = typer.confirm(f"  {shared.relative_to(ROOT)} 已存在，重新初始化？（会保留你已填的值，仅改下面几项）", default=False)
        if do_shared and not shared_eg.is_file():
            pass  # 直接在现有文件上改
    if do_shared:
        if not shared.is_file():
            if not shared_eg.is_file():
                raise typer.BadParameter(f"缺少模板 {shared_eg.relative_to(ROOT)}")
            shutil.copyfile(shared_eg, shared)
            typer.echo(f"  已从模板创建 {shared.relative_to(ROOT)}")
        cur = _read_env_file(shared)
        default_prof = profile or cur.get("DEFAULT_CLUSTER_PROFILE") or (profiles[0] if profiles else "h100")
        hint = f"（可选: {', '.join(profiles)}）" if profiles else ""
        dprof = typer.prompt(f"  默认集群 profile{hint}", default=default_prof)
        sw = typer.prompt("  SwanLab API Key（留空=不上传云端，用 lab web 本地看）", default=cur.get("SWANLAB_API_KEY", ""))
        hf = typer.prompt("  HuggingFace token（下载 gated 模型/数据需要，可留空）", default=cur.get("HF_TOKEN", ""))
        _set_env_line(shared, "DEFAULT_CLUSTER_PROFILE", dprof)
        _set_env_line(shared, "SWANLAB_API_KEY", sw)
        _set_env_line(shared, "HF_TOKEN", hf)
        typer.secho(f"  ✓ 已写入 {shared.relative_to(ROOT)}", fg=typer.colors.GREEN)
        profile = profile or dprof

    # ---------- 集群层 cluster/<profile>/submit.env ----------
    prof = profile or (_read_env_file(shared).get("DEFAULT_CLUSTER_PROFILE") if shared.is_file() else None)
    if not prof:
        ask = "  要初始化哪个集群层 profile" + (f"（可选: {', '.join(profiles)}）" if profiles else "")
        prof = typer.prompt(ask)
    typer.secho(f"\n② 集群层 cluster/{prof}/submit.env（地址 / 容器内路径，随集群走）", bold=True)
    prof_dir = cluster_dir / prof
    if not (prof_dir / "overrides.conf").is_file():
        typer.secho(
            f"  ! 未知 profile '{prof}'（cluster/{prof}/ 不存在）。先建好该集群目录或换个 profile。"
            "（已完成通用层，集群层跳过）",
            fg=typer.colors.YELLOW,
        )
        raise typer.Exit(0)
    pe = prof_dir / "submit.env"
    pe_eg = prof_dir / "submit.env.example"
    do_prof = True
    if pe.is_file():
        do_prof = typer.confirm(f"  {pe.relative_to(ROOT)} 已存在，重新初始化？", default=False)
    if do_prof:
        if not pe.is_file():
            if not pe_eg.is_file():
                raise typer.BadParameter(f"缺少模板 {pe_eg.relative_to(ROOT)}")
            shutil.copyfile(pe_eg, pe)
            typer.echo(f"  已从模板创建 {pe.relative_to(ROOT)}")
        cur = _read_env_file(pe)
        addr = typer.prompt("  Ray dashboard 地址（head 节点，端口 8265）", default=cur.get("RAY_DASHBOARD_ADDRESS", "http://x.x.x.x:8265"))
        nemo = typer.prompt("  容器内 NeMo-RL 路径（不是本机路径）", default=cur.get("NEMO_RL_DIR", "/opt/nemo-rl"))
        out = typer.prompt("  产物落盘根目录 OUTPUT_ROOT（集群持久路径/共享盘）", default=cur.get("OUTPUT_ROOT", "/data/nemo-rl-runs"))
        _set_env_line(pe, "RAY_DASHBOARD_ADDRESS", addr)
        _set_env_line(pe, "NEMO_RL_DIR", nemo)
        _set_env_line(pe, "OUTPUT_ROOT", out)
        typer.secho(f"  ✓ 已写入 {pe.relative_to(ROOT)}", fg=typer.colors.GREEN)

    typer.secho("\n完成。其余项（裁判 LLM / RAGFlow / CLUSTER_SECRETS_FILE 等）按需手动编辑对应文件。", fg=typer.colors.GREEN)
    typer.echo("下一步： uv run lab doctor   # 体检配置/连通/Ray 版本对齐")


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


@app.command(help="从本机提交作业到 Ray 集群（执行在集群；提交前自动校验 config）")
def submit(
    exp: str = typer.Argument(..., autocompletion=_complete_exp, help="实验名或路径"),
    profile: Optional[str] = typer.Option(
        None, "--profile", autocompletion=_complete_profile, help="硬件 profile（默认取 submit.env）"
    ),
    no_validate: bool = typer.Option(False, "--no-validate", help="跳过提交前 config 校验"),
) -> None:
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
    cmd = ["bash", str(SCRIPTS / "submit_job.sh"), exp_path]
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


def _pinned_ray_version() -> Optional[str]:
    """从 pyproject 的 submit extra 读对齐集群用的 Ray 版本（ray[default]==X.Y.Z）。"""
    import re

    try:
        txt = (ROOT / "pyproject.toml").read_text()
    except OSError:
        return None
    m = re.search(r"ray\[default\]==([0-9][0-9.]*)", txt)
    return m.group(1) if m else None


@app.command(help="体检本机提交环境：配置是否填全 / 能否连上集群 dashboard / Ray 版本是否对齐")
def doctor(
    profile: Optional[str] = typer.Option(
        None, "--profile", autocompletion=_complete_profile,
        help="集群 profile（默认取 submit.env 的 DEFAULT_CLUSTER_PROFILE）",
    ),
) -> None:
    import json
    import urllib.error
    import urllib.request

    failed = 0
    warned = 0

    def line(status: str, msg: str) -> None:
        nonlocal failed, warned
        sym = {"ok": "✓", "warn": "!", "fail": "✗"}[status]
        color = {"ok": typer.colors.GREEN, "warn": typer.colors.YELLOW, "fail": typer.colors.RED}[status]
        if status == "fail":
            failed += 1
        elif status == "warn":
            warned += 1
        typer.secho(f"  {sym} {msg}", fg=color)

    shared_path = ROOT / "cluster" / "submit.env"
    shared = _read_env_file(shared_path)
    typer.echo("提交配置")
    if shared_path.is_file():
        line("ok", f"通用层存在: {shared_path}")
    else:
        line("warn", "通用层 cluster/submit.env 不存在（cp cluster/submit.env.example cluster/submit.env）")

    prof = profile or os.environ.get("CLUSTER_PROFILE") or shared.get("DEFAULT_CLUSTER_PROFILE")
    if not prof:
        line("fail", "无法确定 profile：--profile 指定，或在通用层设 DEFAULT_CLUSTER_PROFILE")
        typer.secho("\n体检中止：先确定集群 profile。", fg=typer.colors.RED)
        raise typer.Exit(1)
    line("ok", f"目标 profile: {prof}")

    prof_path = ROOT / "cluster" / prof / "submit.env"
    if prof_path.is_file():
        line("ok", f"集群层存在: {prof_path}")
    else:
        line("warn", f"集群层 cluster/{prof}/submit.env 不存在（cp cluster/{prof}/submit.env.example 它）")

    merged = _read_submit_env(prof)
    address = merged.get("RAY_DASHBOARD_ADDRESS")
    if address and "x.x.x.x" not in address:
        line("ok", f"RAY_DASHBOARD_ADDRESS={address}")
    else:
        line("fail", f"RAY_DASHBOARD_ADDRESS 未填或仍是占位（当前: {address or '空'}）")
    if merged.get("NEMO_RL_DIR"):
        line("ok", f"NEMO_RL_DIR={merged['NEMO_RL_DIR']}（容器内）")
    else:
        line("fail", "NEMO_RL_DIR 未设置（容器内 NeMo-RL 路径）")

    typer.echo("密钥")
    if merged.get("CLUSTER_SECRETS_FILE"):
        line("ok", f"集群侧 secrets：{merged['CLUSTER_SECRETS_FILE']}（密钥不明文转发）")
    elif merged.get("SWANLAB_API_KEY") or merged.get("HF_TOKEN"):
        line("warn", "检测到密钥将随作业明文转发（Dashboard 可见）。多人共用建议设 CLUSTER_SECRETS_FILE")
    else:
        line("warn", "未配置密钥（SwanLab 不上传云端，可用 lab web 本地看；下载 gated 模型需 HF_TOKEN）")

    typer.echo("集群连通 / 版本")
    pinned = _pinned_ray_version()
    if address and "x.x.x.x" not in address:
        try:
            with urllib.request.urlopen(f"{address.rstrip('/')}/api/version", timeout=5) as r:
                info = json.loads(r.read())
            cluster_ray = info.get("ray_version", "?")
            line("ok", f"dashboard 可达，集群 Ray={cluster_ray}")
            if pinned and cluster_ray != "?" and cluster_ray != pinned:
                line("warn", f"本机锁定 Ray={pinned} 与集群 {cluster_ray} 不一致 → 改 pyproject 的 submit extra 后 uv sync --extra submit")
            elif pinned:
                line("ok", f"本机锁定 Ray={pinned} 与集群一致")
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as e:
            line("fail", f"连不上 dashboard（{type(e).__name__}）：检查 VPN/SSH 隧道与地址")
    else:
        line("warn", "地址未填，跳过连通性检查")

    typer.echo("")
    if failed:
        typer.secho(f"体检：{failed} 项失败，{warned} 项告警 —— 修复失败项后再 submit。", fg=typer.colors.RED)
        raise typer.Exit(1)
    typer.secho(f"体检通过（{warned} 项告警）。可以 lab submit 了。", fg=typer.colors.GREEN)


# ----------------------------- 训练后闭环（export / eval；提交到集群执行）-----------------------------
# 转发给训练后作业的环境变量：非密钥配置 + 下载/推送用的 HF_TOKEN（密钥规则同 submit）。
_POST_CONFIG_KEYS = ("OUTPUT_ROOT", "RUN_USER", "HF_HOME", "HF_ENDPOINT", "HF_HUB_ENABLE_HF_TRANSFER", "UV_NO_SYNC")
_POST_SECRET_KEYS = ("HF_TOKEN",)


def _resolve_profile(profile: Optional[str]) -> Optional[str]:
    """与提交链路一致地确定 profile：显式 > 环境 CLUSTER_PROFILE > 通用层 DEFAULT_CLUSTER_PROFILE。"""
    shared = _read_env_file(ROOT / "cluster" / "submit.env")
    return profile or os.environ.get("CLUSTER_PROFILE") or shared.get("DEFAULT_CLUSTER_PROFILE")


def _git_provenance() -> dict[str, str]:
    """取当前 git commit 短码与 dirty 标记（用于训练后作业的可追溯）。"""
    def _g(args: list[str]) -> str:
        try:
            return subprocess.run(
                ["git", *args], cwd=str(ROOT), capture_output=True, text=True
            ).stdout.strip()
        except Exception:  # noqa: BLE001
            return ""

    return {
        "commit": _g(["rev-parse", "--short", "HEAD"]) or "unknown",
        "dirty": "1" if _g(["status", "--porcelain"]) else "0",
    }


def _build_post_runtime_env(
    merged: dict[str, str], nemo_rl_dir: str, profile: str, run_id: str, user: str
) -> str:
    """为 export/eval 作业组装 runtime_env JSON：注入路径/元数据，密钥按 CLUSTER_SECRETS_FILE 决定是否明文转发。"""
    import json

    prov = _git_provenance()
    env_vars: dict[str, str] = {
        "NEMO_RL_DIR": nemo_rl_dir,
        "CLUSTER_PROFILE": profile,
        "NRL_GIT_COMMIT": prov["commit"],
        "NRL_GIT_DIRTY": prov["dirty"],
        "NRL_RUN_ID": run_id,
        "NRL_SUBMIT_USER": user,
    }

    def val(k: str) -> Optional[str]:
        return merged.get(k) or os.environ.get(k)

    for k in _POST_CONFIG_KEYS:
        v = val(k)
        if v:
            env_vars[k] = v
    secrets_file = (val("CLUSTER_SECRETS_FILE") or "").strip()
    if secrets_file:
        env_vars["CLUSTER_SECRETS_FILE"] = secrets_file  # 集群侧 source，不明文转发密钥
    else:
        for k in _POST_SECRET_KEYS:
            v = val(k)
            if v:
                env_vars[k] = v
    return json.dumps({
        "excludes": [
            "datasets/**/raw/**", "datasets/**/data/**",
            "**/outputs/**", ".git/**", "**/__pycache__/**", ".lab/**",
            "cluster/submit.env", "cluster/*/submit.env", "cluster/secrets.env", "**/*.key",
        ],
        "env_vars": env_vars,
    })


LEDGER_PATH = ROOT / ".lab" / "runs.jsonl"


def _append_ledger(entry: dict) -> None:
    """向本地台账 .lab/runs.jsonl 追加一行（与 submit 同一台账，统一追溯）。"""
    import json
    import time

    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry.setdefault("time", time.strftime("%Y-%m-%d %H:%M:%S"))
    with LEDGER_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _read_ledger(path: Path) -> list[dict]:
    """读本地台账（每行一个 JSON）；坏行跳过，保证看历史不被一条脏数据中断。"""
    import json

    out: list[dict] = []
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                out.append(obj)
        except json.JSONDecodeError:
            continue
    return out


def _filter_runs(entries: list[dict], exp: Optional[str], limit: Optional[int]) -> list[dict]:
    """按实验过滤（接受全名或末段名）并按时间倒序，截取 limit 条（limit=None 不截取）。"""
    rows = entries
    if exp:
        want = Path(exp).name
        rows = [e for e in rows if Path(str(e.get("exp", ""))).name == want]
    rows = sorted(rows, key=lambda e: str(e.get("time", "")), reverse=True)
    if limit is not None:
        rows = rows[:limit]
    return rows


@app.command(help="查看本地提交台账（.lab/runs.jsonl：每次 submit/export/eval 的 commit/config 指纹/run_id）；--status 关联集群作业状态")
def runs(
    all_runs: bool = typer.Option(False, "--all", help="显示全部（默认最近 20 条）"),
    exp: Optional[str] = typer.Option(
        None, "--exp", autocompletion=_complete_exp, help="只看某实验（接受全名或末段名）"
    ),
    limit: int = typer.Option(20, "-n", "--limit", help="显示条数（--all 时忽略）"),
    status: bool = typer.Option(False, "--status", "-s", help="关联集群作业状态（连 dashboard，按 run_id 对上）"),
    address: Optional[str] = _ADDR_OPT,
    profile: Optional[str] = _PROF_OPT,
) -> None:
    entries = _read_ledger(LEDGER_PATH)
    if not entries:
        typer.echo(f"（台账为空：{LEDGER_PATH} 还没有记录；lab submit / export / eval 后会写入）")
        return
    rows = _filter_runs(entries, exp, None if all_runs else limit)
    if not rows:
        typer.echo(f"（没有匹配的记录{f'：exp={exp}' if exp else ''}）")
        return

    # --status：拉集群作业，按 run_id 对上状态；连不上则降级为纯本地。
    status_map: dict[str, str] = {}
    if status:
        try:
            status_map = _run_status_map(_ray_address(address, profile))
        except Exception as e:  # noqa: BLE001
            typer.secho(f"（连集群取状态失败，降级为本地台账：{type(e).__name__}）", fg=typer.colors.YELLOW)

    show_status = status
    headers = ["TIME", "ACTION"] + (["STATUS"] if show_status else []) + ["EXP", "PROFILE", "COMMIT", "USER", "RUN_ID"]

    def cells(e: dict) -> list[str]:
        commit = str(e.get("git_commit", "-"))
        if e.get("git_dirty"):
            commit += "*"
        rid = str(e.get("run_id", "-"))
        base = [str(e.get("time", "-")), str(e.get("action", "submit"))]
        st = [status_map.get(rid, "-")] if show_status else []
        return base + st + [
            Path(str(e.get("exp", "-"))).name,
            str(e.get("profile", "-")),
            commit,
            str(e.get("user", "-")),
            rid,
        ]

    table = [cells(e) for e in rows]
    widths = [max(len(headers[i]), *(len(r[i]) for r in table)) for i in range(len(headers))]
    st_col = headers.index("STATUS") if show_status else -1

    def render(r: list[str]) -> str:
        out = []
        for i, c in enumerate(r):
            padded = c.ljust(widths[i])
            if i == st_col and (color := _STATUS_COLOR.get(c)):
                padded = typer.style(padded, fg=color)
            out.append(padded)
        return "  ".join(out)

    typer.echo("  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    typer.echo("  ".join("-" * widths[i] for i in range(len(headers))))
    for r in table:
        typer.echo(render(r))
    note = "（* = 提交时 dirty）"
    if show_status:
        note += "；STATUS=- 表示集群上已无该作业记录（已清理/未提交成功）"
    typer.echo(f"\n共 {len(rows)} 条" + ("" if all_runs else f"（最近 {limit}；--all 看全部）") + note)


@app.command(help="集群一览：空闲 GPU + 我的活跃作业（RUNNING/PENDING），submit 前预检")
def status(
    address: Optional[str] = _ADDR_OPT,
    profile: Optional[str] = _PROF_OPT,
    all_jobs: bool = typer.Option(False, "--all", help="作业列出全部状态（默认只看 RUNNING/PENDING）"),
) -> None:
    addr = _ray_address(address, profile)

    # 1) 集群 GPU / 资源
    typer.echo(f"集群资源  {addr}")
    try:
        gpu = _gpu_summary(addr)
    except Exception as e:  # noqa: BLE001
        gpu = None
        typer.secho(f"  ! 取集群资源失败（{type(e).__name__}）", fg=typer.colors.YELLOW)
    if gpu:
        u = gpu["usage"]

        def pair(key: str) -> tuple[float, float]:
            v = u.get(key) or [0, 0]
            return float(v[0]), float(v[1])

        gu, gt = pair("GPU")
        cu, ct = pair("CPU")
        mu, mt = pair("memory")
        free = gt - gu
        accel = ("/".join(gpu["accel"])) if gpu["accel"] else "GPU"
        gpu_line = f"  {accel}: {gu:.0f}/{gt:.0f} 占用，空闲 {free:.0f}"
        typer.secho(gpu_line, fg=(typer.colors.GREEN if free > 0 else typer.colors.RED))
        typer.echo(f"  CPU: {cu:.0f}/{ct:.0f}   内存: {mu / 2**30:.0f}/{mt / 2**30:.0f} GiB")

    # 2) 作业
    typer.echo("\n作业")
    try:
        jobs = _fetch_jobs(addr)
    except Exception as e:  # noqa: BLE001
        typer.secho(f"  ! 取作业列表失败（{type(e).__name__}）", fg=typer.colors.YELLOW)
        raise typer.Exit(1) from None

    active = {"RUNNING", "PENDING"}
    if not all_jobs:
        jobs = [j for j in jobs if str(j.get("status")) in active]
    jobs.sort(key=lambda j: j.get("start_time") or 0, reverse=True)
    if not jobs:
        typer.echo("  （无活跃作业）" if not all_jobs else "  （没有作业）")
        return

    import time as _time

    def dur(j: dict) -> str:
        s = j.get("start_time")
        if not s:
            return "-"
        e = j.get("end_time") or int(_time.time() * 1000)
        secs = max(0, (e - s) // 1000)
        h, rem = divmod(secs, 3600)
        m, sec = divmod(rem, 60)
        return f"{h}h{m:02d}m" if h else (f"{m}m{sec:02d}s" if m else f"{sec}s")

    rows = []
    for j in jobs:
        meta = j.get("metadata") or {}
        exp = meta.get("lab_exp") or "-"
        rows.append((
            str(j.get("submission_id") or j.get("job_id") or "-"),
            str(j.get("status") or "-"),
            dur(j),
            Path(str(exp)).name if exp != "-" else "-",
        ))
    headers = ("JOB ID", "STATUS", "DUR", "EXP")
    widths = [max(len(headers[i]), *(len(r[i]) for r in rows)) for i in range(len(headers))]
    typer.echo("  " + "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    typer.echo("  " + "  ".join("-" * widths[i] for i in range(len(headers))))
    for r in rows:
        st = typer.style(r[1].ljust(widths[1]), fg=color) if (color := _STATUS_COLOR.get(r[1])) else r[1].ljust(widths[1])
        typer.echo("  " + "  ".join([r[0].ljust(widths[0]), st, r[2].ljust(widths[2]), r[3]]))
    typer.echo("\n详情/日志： lab job logs <JOB ID> -f")


def _latest_job_id(address: str) -> Optional[str]:
    """集群上 start_time 最新的作业 ID（用于 lab logs 无参跟随）。"""
    jobs = _fetch_jobs(address)
    if not jobs:
        return None
    jobs.sort(key=lambda j: j.get("start_time") or 0, reverse=True)
    top = jobs[0]
    return str(top.get("submission_id") or top.get("job_id") or "")


@app.command(help="看作业日志：不给 job_id 默认跟随【最近一个】作业（= lab job logs 的便捷版）")
def logs(
    job_id: Optional[str] = typer.Argument(None, help="作业 ID（见 lab job list）；省略=最近一个"),
    follow: bool = typer.Option(True, "--follow/--no-follow", "-f/-F", help="实时跟随（无参时默认开）"),
    address: Optional[str] = _ADDR_OPT,
    profile: Optional[str] = _PROF_OPT,
) -> None:
    addr = _ray_address(address, profile)
    jid = job_id
    if not jid:
        try:
            jid = _latest_job_id(addr)
        except Exception as e:  # noqa: BLE001
            raise typer.BadParameter(f"取最近作业失败（{type(e).__name__}）；可显式给 job_id。") from None
        if not jid:
            typer.echo("（集群上没有作业）")
            raise typer.Exit(0)
        typer.secho(f"最近作业： {jid}", fg=typer.colors.CYAN)
    args = ["logs", jid] + (["--follow"] if follow else [])
    raise typer.Exit(_ray_job(args, addr))


def _submit_post(action: str, exp_path: str, profile: Optional[str], flags: list[str], dry_run: bool) -> int:
    """把 export/eval 作业提交到集群（ray job submit，入口 scripts/post_train.sh）。"""
    import time

    prof = _resolve_profile(profile)
    if not prof:
        raise typer.BadParameter("无法确定集群 profile：用 --profile 指定，或在 cluster/submit.env 设 DEFAULT_CLUSTER_PROFILE。")
    merged = _read_submit_env(prof)
    nemo_rl_dir = merged.get("NEMO_RL_DIR")
    if not nemo_rl_dir:
        raise typer.BadParameter(f"cluster/{prof}/submit.env 未设 NEMO_RL_DIR（容器内 NeMo-RL 路径）。")
    address = _ray_address(None, prof)
    run_id = f"{action}-{Path(exp_path).name}-{time.strftime('%Y%m%d-%H%M%S')}"
    user = merged.get("RUN_USER") or os.environ.get("RUN_USER") or os.environ.get("USER") or "unknown"
    runtime_env = _build_post_runtime_env(merged, nemo_rl_dir, prof, run_id, user)

    import json

    meta_json = json.dumps({"lab_run_id": run_id, "lab_exp": exp_path, "lab_action": action})
    cmd = [
        "uv", "run", "--extra", "submit", "ray", "job", "submit",
        "--address", address, "--working-dir", ".", "--runtime-env-json", runtime_env,
        "--metadata-json", meta_json,
        "--", "bash", "scripts/post_train.sh", action, exp_path, *flags,
    ]
    typer.echo(f"[{action}] 集群     : {address} (profile={prof})")
    typer.echo(f"[{action}] 实验     : {exp_path}  run_id={run_id}")
    if dry_run:
        typer.echo("› " + " ".join(str(c) for c in cmd))
        typer.secho("（--dry-run：未实际提交）", fg=typer.colors.YELLOW)
        return 0
    _append_ledger({
        "run_id": run_id, "action": action, "user": user, "exp": exp_path,
        "profile": prof, "git_commit": _git_provenance()["commit"], "address": address,
    })
    return _run(cmd)


@app.command(name="export", help="把训练 checkpoint 转成 HF 格式（按后端自适应 dcp/megatron），可选推 HF Hub；执行在集群")
def export_ckpt(
    exp: str = typer.Argument(..., autocompletion=_complete_exp, help="实验名或路径"),
    step: Optional[int] = typer.Option(None, "--step", help="checkpoint 步数（默认最新 step_<N>）"),
    out: Optional[str] = typer.Option(None, "--out", help="HF 输出目录（容器内；默认 <ckpt>/hf_export/step_<N>）"),
    push_repo: Optional[str] = typer.Option(None, "--push-repo", help="转换后上传到 HF Hub repo（user/name，需 HF_TOKEN）"),
    ckpt_dir: Optional[str] = typer.Option(None, "--ckpt-dir", help="覆盖 checkpoint 根目录（容器内绝对路径）"),
    profile: Optional[str] = typer.Option(
        None, "--profile", autocompletion=_complete_profile, help="集群 profile（默认取 submit.env）"
    ),
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
    profile: Optional[str] = typer.Option(
        None, "--profile", autocompletion=_complete_profile, help="集群 profile（默认取 submit.env）"
    ),
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


@job_app.command("cancel-all", help="停止所有运行中/等待中的作业（区别于 clean：clean 只删终态记录、不停运行）")
def job_cancel_all(
    yes: bool = typer.Option(False, "--yes", "-y", help="跳过确认"),
    address: Optional[str] = _ADDR_OPT,
    profile: Optional[str] = _PROF_OPT,
) -> None:
    cmd = ["uv", "run", "--extra", "submit", "python", str(SCRIPTS / "ray_jobs.py"),
           "--address", _ray_address(address, profile), "--cancel-all"]
    if yes:
        cmd.append("--yes")
    raise typer.Exit(_run(cmd))


def _addr_host(addr: Optional[str]) -> Optional[str]:
    """从 http://host:port 取出 host。"""
    if not addr:
        return None
    from urllib.parse import urlparse

    return urlparse(addr).hostname


@app.command(help="开 SSH 隧道：把集群 head 的 8265(dashboard)/6379(GCS) 转发到本机 127.0.0.1（不同网段时用）")
def tunnel(
    host: Optional[str] = typer.Option(None, "--host", help="SSH 目标（跳板机/集群可达主机）；默认取 submit.env 的 SSH_HOST"),
    head_ip: Optional[str] = typer.Option(None, "--head-ip", help="集群 head 节点 IP；默认 RAY_HEAD_IP 或从 RAY_DASHBOARD_ADDRESS 解析"),
    ports: str = typer.Option("8265,6379", "--ports", help="要转发的端口，逗号分隔"),
    profile: Optional[str] = _PROF_OPT,
) -> None:
    env = _read_submit_env(profile)
    ssh_host = host or env.get("SSH_HOST")
    if not ssh_host:
        raise typer.BadParameter("缺少 SSH 目标：用 --host 指定，或在 cluster/<profile>/submit.env 设 SSH_HOST。")
    hip = head_ip or env.get("RAY_HEAD_IP") or _addr_host(env.get("RAY_DASHBOARD_ADDRESS"))
    if not hip or hip in ("127.0.0.1", "localhost"):
        raise typer.BadParameter(
            "无法确定 head IP：用 --head-ip 指定，或设 RAY_HEAD_IP（RAY_DASHBOARD_ADDRESS 指向 127.0.0.1 时无法反推）。"
        )
    cmd = ["ssh", "-N"]
    for p in [x.strip() for x in ports.split(",") if x.strip()]:
        cmd += ["-L", f"{p}:{hip}:{p}"]
    cmd.append(ssh_host)
    typer.secho(f"转发 {ports} ← {hip}（经 {ssh_host}）。保持本窗口开启；另开终端把 RAY_DASHBOARD_ADDRESS 指到 http://127.0.0.1:8265。", fg=typer.colors.CYAN)
    typer.secho("Ctrl-C 关闭隧道。", fg=typer.colors.YELLOW)
    raise typer.Exit(_run(cmd))


# ----------------------------- 远程编排集群（本机 ssh+docker exec 起/停 Ray）-----------------------------
cluster_app = typer.Typer(
    no_args_is_help=True,
    help="从本机远程编排集群：ssh + docker exec 起/停 Ray（区别于在容器内用的 lab ray）。"
    "依赖 cluster/<profile>/submit.env 的 CLUSTER_SSH_HEAD / CLUSTER_SSH_WORKERS / CLUSTER_CONTAINER / CLUSTER_REPO_DIR",
    context_settings={"help_option_names": ["-h", "--help"]},
)
app.add_typer(cluster_app, name="cluster")


def _docker_exec_cmd(ssh_host: str, container: str, repo: str, inner: str, env_prefix: str = "") -> list[str]:
    """ssh <host> 'docker exec <container> bash -lc "cd <repo> && <env>inner"'。"""
    remote = f'docker exec {container} bash -lc "cd {repo} && {env_prefix}{inner}"'
    return ["ssh", ssh_host, remote]


def _cluster_cfg(profile: Optional[str], container: Optional[str], repo: Optional[str]) -> tuple[dict, str, str, str]:
    env = _read_submit_env(profile)
    prof = profile or os.environ.get("CLUSTER_PROFILE") or env.get("DEFAULT_CLUSTER_PROFILE")
    if not prof:
        raise typer.BadParameter("无法确定 profile：用 --profile 指定，或设 DEFAULT_CLUSTER_PROFILE。")
    cont = container or env.get("CLUSTER_CONTAINER")
    rp = repo or env.get("CLUSTER_REPO_DIR")
    missing = [n for n, v in (("CLUSTER_CONTAINER", cont), ("CLUSTER_REPO_DIR", rp)) if not v]
    if missing:
        raise typer.BadParameter(
            f"缺少 {', '.join(missing)}：在 cluster/{prof}/submit.env 配置（容器名 / 容器内仓库路径），或用 --container/--repo 指定。"
        )
    return env, prof, cont, rp


@cluster_app.command("up", help="远程起 Ray：head（CLUSTER_SSH_HEAD）+ 各 worker（CLUSTER_SSH_WORKERS，逗号分隔）")
def cluster_up(
    profile: Optional[str] = _PROF_OPT,
    container: Optional[str] = typer.Option(None, "--container", help="容器名（默认 CLUSTER_CONTAINER）"),
    repo: Optional[str] = typer.Option(None, "--repo", help="容器内仓库路径（默认 CLUSTER_REPO_DIR）"),
    dry_run: bool = typer.Option(False, "--dry-run", help="只打印将执行的远程命令，不真正执行"),
) -> None:
    env, prof, cont, rp = _cluster_cfg(profile, container, repo)
    head_ssh = env.get("CLUSTER_SSH_HEAD")
    if not head_ssh:
        raise typer.BadParameter(f"缺少 CLUSTER_SSH_HEAD：在 cluster/{prof}/submit.env 设 head 节点的 ssh 目标。")
    head_ip = env.get("RAY_HEAD_IP") or _addr_host(env.get("RAY_DASHBOARD_ADDRESS"))

    rc = 0
    head_cmd = _docker_exec_cmd(
        head_ssh, cont, rp, f"bash cluster/{prof}/start_ray_head.sh",
        env_prefix=(f"HEAD_IP={head_ip} " if head_ip else ""),
    )
    typer.secho(f"[head] {head_ssh}", bold=True)
    rc |= _run(head_cmd) if not dry_run else (typer.echo("› " + " ".join(head_cmd)) or 0)

    workers = [w.strip() for w in (env.get("CLUSTER_SSH_WORKERS") or "").split(",") if w.strip()]
    if not workers:
        typer.secho("（未设 CLUSTER_SSH_WORKERS：单机/或 worker 手动起）", fg=typer.colors.YELLOW)
    for w in workers:
        wprefix = f"HEAD_ADDRESS={head_ip}:6379 " if head_ip else ""
        wcmd = _docker_exec_cmd(w, cont, rp, f"bash cluster/{prof}/start_ray_worker.sh", env_prefix=wprefix)
        typer.secho(f"[worker] {w}", bold=True)
        rc |= _run(wcmd) if not dry_run else (typer.echo("› " + " ".join(wcmd)) or 0)
    if not dry_run:
        typer.echo("\n确认节点数： uv run lab status   （或在容器内 lab ray status）")
    raise typer.Exit(rc)


@cluster_app.command("down", help="远程停 Ray：在 head + 各 worker 容器内 ray stop（释放 GPU）")
def cluster_down(
    profile: Optional[str] = _PROF_OPT,
    container: Optional[str] = typer.Option(None, "--container", help="容器名（默认 CLUSTER_CONTAINER）"),
    repo: Optional[str] = typer.Option(None, "--repo", help="容器内仓库路径（默认 CLUSTER_REPO_DIR）"),
    dry_run: bool = typer.Option(False, "--dry-run", help="只打印将执行的远程命令，不真正执行"),
) -> None:
    env, prof, cont, rp = _cluster_cfg(profile, container, repo)
    hosts = [h for h in [env.get("CLUSTER_SSH_HEAD")] if h]
    hosts += [w.strip() for w in (env.get("CLUSTER_SSH_WORKERS") or "").split(",") if w.strip()]
    if not hosts:
        raise typer.BadParameter(f"缺少 CLUSTER_SSH_HEAD：在 cluster/{prof}/submit.env 配置。")
    rc = 0
    for h in hosts:
        cmd = _docker_exec_cmd(h, cont, rp, "uv run ray stop")
        typer.secho(f"[stop] {h}", bold=True)
        rc |= _run(cmd) if not dry_run else (typer.echo("› " + " ".join(cmd)) or 0)
    raise typer.Exit(rc)


if __name__ == "__main__":
    app()
