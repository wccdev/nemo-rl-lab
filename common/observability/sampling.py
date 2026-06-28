"""硬件监控采样策略（对齐 SwanLab System Monitor 文档）。"""

from __future__ import annotations

# SwanLab: 0~10 点 10s，10~50 点 30s，50+ 点 60s
SWANLAB_TIER1 = 10
SWANLAB_TIER2 = 50
SWANLAB_INTERVAL_SHORT = 10.0
SWANLAB_INTERVAL_MID = 30.0
SWANLAB_INTERVAL_LONG = 60.0
SWANLAB_MIN_INTERVAL = 5.0


def swanlab_monitor_interval(
    samples_collected: int,
    *,
    base_interval: float = SWANLAB_INTERVAL_SHORT,
    dynamic: bool = True,
) -> float:
    """根据已采集轮次返回下一次 sleep 间隔（秒）。"""
    base = max(SWANLAB_MIN_INTERVAL, float(base_interval))
    if not dynamic:
        return base
    n = max(0, int(samples_collected))
    if n < SWANLAB_TIER1:
        return base
    if n < SWANLAB_TIER2:
        return max(base, SWANLAB_INTERVAL_MID)
    return max(base, SWANLAB_INTERVAL_LONG)
