"""SwanLab 日志辅助。

优先用 NeMo-RL 配置里原生的 swanlab logger。若版本不支持，可用此模块手动初始化，
统一把仓库的命名规范（project=实验名、experiment=超参组合）落到 SwanLab 上。
"""
from __future__ import annotations

import os
from typing import Any


def init_swanlab(project: str, experiment_name: str, config: dict[str, Any] | None = None,
                 tags: list[str] | None = None):
    """初始化 SwanLab run。需先 `swanlab login` 或设置 SWANLAB_API_KEY。"""
    import swanlab

    return swanlab.init(
        project=project,
        experiment_name=experiment_name,
        config=config or {},
        tags=tags or [],
        mode="cloud" if os.getenv("SWANLAB_API_KEY") else "local",
    )


def log_metrics(run, metrics: dict[str, float], step: int | None = None) -> None:
    run.log(metrics, step=step)
