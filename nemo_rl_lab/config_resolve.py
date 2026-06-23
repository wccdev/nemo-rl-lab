"""实验 config.yaml 解析（defaults 继承 + `_override_`）与提交前静态校验。

与 NeMo-RL 0.6.0 的配置机制对齐：每个实验 config 通过 `defaults` 继承基底 + 模型片段，
只写差异；`_override_: true` 的 dict 整段替换。`lab validate` 与单元测试都用本模块，
在【本地秒级】抓出「跑到集群才报错」的低级配置错误（如 batch 三者不相等）。

注意：这里只做**轻量静态校验**，不依赖 NeMo-RL / OmegaConf；带插值（`${...}`）的字段
按字符串保留、不参与数值校验（取不到具体值就跳过，避免误报）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: Any, over: Any) -> Any:
    """深合并；over 覆盖 base。over 带 `_override_: true` 时整段替换（剥掉标记）。"""
    if isinstance(over, dict) and over.get("_override_"):
        return {k: v for k, v in over.items() if k != "_override_"}
    if isinstance(base, dict) and isinstance(over, dict):
        out = dict(base)
        for k, v in over.items():
            if k == "_override_":
                continue
            out[k] = deep_merge(out.get(k), v) if k in out else v
        return out
    return over


def resolve(path: str | Path) -> dict:
    """加载 yaml 并按 `defaults` 顺序合并（相对本文件路径），最后叠加自身键。"""
    path = Path(path)
    data = load_yaml(path)
    defaults = data.pop("defaults", []) or []
    merged: dict = {}
    for d in defaults:
        if not isinstance(d, str):
            continue
        dp = (path.parent / d).resolve()
        if dp.suffix not in (".yaml", ".yml"):
            dp = dp.with_suffix(".yaml")
        if dp.is_file():
            merged = deep_merge(merged, resolve(dp))
    return deep_merge(merged, data)


def _as_int(v: Any) -> int | None:
    """只接受可确定为整数的字面值；插值字符串 `${...}` 等返回 None（跳过校验）。"""
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        s = v.strip()
        if s.isdigit():
            return int(s)
    return None


# *_DATA_DIR 环境变量名 → 仓库内默认数据目录（与 scripts/_run_experiment.sh 的自动推导一致）
_DATA_DIR_MAP = {
    "GSM8K_DATA_DIR": "datasets/gsm8k",
    "ALPACA_DATA_DIR": "datasets/alpaca",
    "QA_RL_DATA_DIR": "datasets/qa_rl",
}


def _resolve_data_path(raw: str, repo_root: Path) -> Path | None:
    """把 config 里的 data_path 尽力解析为本地路径用于存在性检查。

    支持 `${oc.env:VAR}/x.jsonl` 与 `${VAR}/x.jsonl`：若 VAR 是已知 *_DATA_DIR，
    用仓库内默认目录替换。无法确定时返回 None（跳过检查）。
    """
    if not isinstance(raw, str) or not raw:
        return None
    s = raw
    for var, default in _DATA_DIR_MAP.items():
        for pat in (f"${{oc.env:{var}}}", f"${{{var}}}"):
            if s.startswith(pat):
                return repo_root / default / s[len(pat):].lstrip("/")
    if "${" in s:  # 还有其它插值，无法确定
        return None
    p = Path(s)
    return p if p.is_absolute() else repo_root / s


def validate_config(cfg: dict, repo_root: Path | None = None) -> list[tuple[str, str]]:
    """对合并后的 config 做静态校验，返回 [(level, message)]，level ∈ {error, warn}。

    规则（取不到具体数值的项自动跳过，不误报）：
      - GRPO：train_global_batch_size == num_prompts_per_step × num_generations_per_prompt
      - val_batch_size <= max_val_samples
      - max_num_steps / val_period > 0
      - policy.max_total_sequence_length > 0
      - 数据文件（能解析为本地路径时）存在性 → warn
    """
    issues: list[tuple[str, str]] = []
    grpo = cfg.get("grpo") or {}
    policy = cfg.get("policy") or {}

    # 1) GRPO batch 三者相等（最常踩的坑）
    if grpo:
        npp = _as_int(grpo.get("num_prompts_per_step"))
        ngp = _as_int(grpo.get("num_generations_per_prompt"))
        gbs = _as_int(policy.get("train_global_batch_size"))
        if npp is not None and ngp is not None and gbs is not None:
            if gbs != npp * ngp:
                issues.append((
                    "error",
                    f"train_global_batch_size({gbs}) ≠ num_prompts_per_step({npp}) × "
                    f"num_generations_per_prompt({ngp})={npp * ngp}。三者必须相等。",
                ))
        # 2) 验证 batch 不超过验证样本数
        vbs = _as_int(grpo.get("val_batch_size"))
        mvs = _as_int(grpo.get("max_val_samples"))
        if vbs is not None and mvs is not None and vbs > mvs:
            issues.append((
                "error",
                f"val_batch_size({vbs}) > max_val_samples({mvs})。val_batch_size 应 ≤ max_val_samples。",
            ))
        # 3) 步数 / 验证周期为正
        for key in ("max_num_steps", "val_period"):
            iv = _as_int(grpo.get(key))
            if iv is not None and iv <= 0:
                issues.append(("error", f"grpo.{key} 必须 > 0（当前 {iv}）。"))

    # 4) 上下文长度为正
    mtsl = _as_int(policy.get("max_total_sequence_length"))
    if mtsl is not None and mtsl <= 0:
        issues.append(("error", f"policy.max_total_sequence_length 必须 > 0（当前 {mtsl}）。"))

    # 5) 数据文件存在性（best-effort，缺失只 warn——可能在集群上）
    if repo_root is not None:
        data = cfg.get("data") or {}
        for split in ("train", "validation"):
            sec = data.get(split)
            if isinstance(sec, dict):
                dp = sec.get("data_path")
                local = _resolve_data_path(dp, repo_root) if dp else None
                if local is not None and not local.exists():
                    issues.append((
                        "warn",
                        f"data.{split}.data_path 本地不存在: {local}"
                        f"（如在集群上已有可忽略；否则先 `lab prepare <数据集>`）。",
                    ))
    return issues
