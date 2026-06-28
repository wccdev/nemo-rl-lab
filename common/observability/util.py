"""轻量工具（避免依赖 NeMo-RL 内部 helper）。"""
from __future__ import annotations

from typing import Any, Mapping


def flatten_dict(d: Mapping[str, Any], sep: str = ".") -> dict[str, Any]:
    out: dict[str, Any] = {}

    def _walk(obj: Mapping[str, Any], prefix: str = "") -> None:
        for key, value in obj.items():
            nk = f"{prefix}{sep}{key}" if prefix else key
            if isinstance(value, dict):
                _walk(value, nk)
            else:
                out[nk] = value

    _walk(d)
    return out


def scalarize_metric(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            if value.size == 1:
                return float(value.reshape(-1)[0])
            if np.issubdtype(value.dtype, np.number):
                return float(np.mean(value))
    except Exception:
        pass
    try:
        import torch

        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return float(value.detach().cpu().item())
            if value.is_floating_point():
                return float(value.detach().float().mean().cpu().item())
    except Exception:
        pass
    if isinstance(value, list):
        nums = [x for x in value if isinstance(x, (int, float))]
        if nums:
            return float(sum(nums) / len(nums))
    return None
