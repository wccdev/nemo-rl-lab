"""仿 SwanLab Monitor：经 Ray 在各节点 pynvml/psutil 直采。

ray 仅在集群容器内可用，本地开发环境通常未安装；故全程懒加载 ray。
集群里 ray 必然可用：init_ray() 在 NeMo-RL 构造 Logger 之前调用，监控正常采集。
仅当 ray 模块不可导入（非集群环境，本就不跑训练）才放弃；ray 暂未 init 则在循环里等待。
"""
from __future__ import annotations

import threading
import time
from datetime import datetime, timezone

from common.observability.hw_probe import collect_hw_snapshot
from common.observability.util import scalarize_metric


class HardwareMonitor:
    def __init__(
        self,
        ingest,
        *,
        collection_interval: float = 10.0,
    ):
        self.ingest = ingest
        self.collection_interval = collection_interval
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        # ray 仅集群容器内存在；本地开发/单测环境无 ray，此时才真正放弃硬件监控
        # （那些环境本就不跑训练）。集群里 ray 一定可导入，监控线程必然启动。
        try:
            import ray  # noqa: F401
        except ImportError:
            print("NeMoLab hardware monitor skipped: ray not available (non-cluster env)")
            return
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="NeMoLab·Monitor"
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=self.collection_interval * 2)

    def _loop(self) -> None:
        import ray

        while self._running:
            try:
                # 不依赖 Logger 与 init_ray() 的构造先后：ray 未 ready 时本轮等待，下轮重试，
                # 一旦集群 ray 起来就开始采集，避免“构造时序变化导致永久不采”。
                if not ray.is_initialized():
                    time.sleep(self.collection_interval)
                    continue
                points = self._collect_cluster_hw()
                if points:
                    self.ingest.enqueue_hardware(points)
            except Exception as e:
                print(f"NeMoLab hardware monitor error: {e}")
            time.sleep(self.collection_interval)

    def _collect_cluster_hw(self) -> list[dict]:
        import ray
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        remote_collect = ray.remote(num_cpus=0)(collect_hw_snapshot)
        ts = datetime.now(timezone.utc).isoformat()
        points: list[dict] = []
        nodes = [n for n in ray.nodes() if n.get("Alive")]
        futures = []
        node_ids = []
        for node in nodes:
            node_id = node.get("NodeID")
            if not node_id:
                continue
            futures.append(
                remote_collect.options(
                    scheduling_strategy=NodeAffinitySchedulingStrategy(
                        node_id=node_id, soft=False
                    )
                ).remote()
            )
            node_ids.append(node_id)
        if not futures:
            return points
        snapshots = ray.get(futures)
        for node_id, snap in zip(node_ids, snapshots, strict=False):
            worker_id = snap.get("hostname") or node_id
            for key, value in (snap.get("metrics") or {}).items():
                gpu_idx = None
                if key.startswith("gpu."):
                    parts = key.split(".")
                    if len(parts) > 1 and parts[1].isdigit():
                        gpu_idx = int(parts[1])
                scalar = scalarize_metric(value)
                if scalar is None:
                    continue
                points.append(
                    {
                        "key": key,
                        "value": scalar,
                        "node_id": node_id,
                        "worker_id": worker_id,
                        "gpu_idx": gpu_idx,
                        "ts": ts,
                    }
                )
        return points
