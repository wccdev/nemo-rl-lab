"""验证上下文：从日志行识别「Starting validation at step N」。"""
from __future__ import annotations

import re

_VAL_START = re.compile(r"Starting validation at step\s+(\d+)", re.I)

_active_step: int | None = None


def feed_log_text(text: str) -> None:
    global _active_step
    for line in text.splitlines():
        m = _VAL_START.search(line)
        if m:
            _active_step = int(m.group(1))


def active_validation_step() -> int | None:
    return _active_step


def clear_validation_step() -> None:
    global _active_step
    _active_step = None
