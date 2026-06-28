"""本地硬件探测（pynvml + psutil，指标 key 对齐 SwanLab）。"""
from __future__ import annotations

import os
import socket
from typing import Any


def collect_local_hw() -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    try:
        import psutil

        metrics["cpu.pct"] = float(psutil.cpu_percent(interval=None))
        metrics["cpu.thds"] = float(psutil.Process().num_threads())
        vm = psutil.virtual_memory()
        proc = psutil.Process()
        metrics["mem.pct"] = float(vm.percent)
        metrics["mem.proc"] = float(proc.memory_info().rss) / (1024**2)
        metrics["mem.proc.pct"] = float(proc.memory_percent())
        metrics["mem.proc.avail"] = float(vm.available) / (1024**2)
    except Exception:
        pass

    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            n = pynvml.nvmlDeviceGetCount()
            visible_ids = (
                [int(x) for x in visible.split(",") if x.strip()]
                if visible
                else list(range(n))
            )
            for idx, _logical in enumerate(visible_ids):
                try:
                    physical = _physical_device_id(idx, visible_ids)
                    handle = pynvml.nvmlDeviceGetHandleByIndex(physical)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    metrics[f"gpu.{idx}.pct"] = float(util.gpu)
                    metrics[f"gpu.{idx}.mem.pct"] = float(100.0 * mem.used / mem.total)
                    metrics[f"gpu.{idx}.mem.value"] = float(mem.used >> 20)
                    try:
                        metrics[f"gpu.{idx}.temp"] = float(
                            pynvml.nvmlDeviceGetTemperature(
                                handle, pynvml.NVML_TEMPERATURE_GPU
                            )
                        )
                    except Exception:
                        pass
                    try:
                        metrics[f"gpu.{idx}.power"] = float(
                            pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                        )
                    except Exception:
                        pass
                    try:
                        metrics[f"gpu.{idx}.mem.time"] = float(util.memory)
                    except Exception:
                        pass
                except Exception:
                    continue
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
    except Exception:
        pass

    return metrics


def collect_hw_snapshot() -> dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "metrics": collect_local_hw(),
    }


def _physical_device_id(logical_idx: int, visible_ids: list[int]) -> int:
    if logical_idx < len(visible_ids):
        return visible_ids[logical_idx]
    return logical_idx
