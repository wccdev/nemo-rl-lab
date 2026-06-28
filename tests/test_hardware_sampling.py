"""SwanLab 对齐的动态硬件采样间隔。"""
from common.observability.sampling import swanlab_monitor_interval


def test_swanlab_monitor_interval_tiers():
    assert swanlab_monitor_interval(0) == 10
    assert swanlab_monitor_interval(9) == 10
    assert swanlab_monitor_interval(10) == 30
    assert swanlab_monitor_interval(49) == 30
    assert swanlab_monitor_interval(50) == 60
    assert swanlab_monitor_interval(200) == 60


def test_swanlab_monitor_interval_respects_base():
    assert swanlab_monitor_interval(0, base_interval=15) == 15
    assert swanlab_monitor_interval(10, base_interval=15) == 30


def test_swanlab_monitor_interval_static():
    assert swanlab_monitor_interval(999, dynamic=False) == 10
    assert swanlab_monitor_interval(999, base_interval=20, dynamic=False) == 20
