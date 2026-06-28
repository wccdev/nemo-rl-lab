"""发现当前 Ray 作业占用的节点（driver + 本 job 的 alive actors）。"""
from __future__ import annotations

import os
from typing import Callable


def runtime_ray_job_id() -> str | None:
    try:
        import ray

        jid = ray.get_runtime_context().get_job_id()
        if jid:
            return str(jid)
    except Exception:
        pass
    for env in ("RAY_JOB_ID", "JOB_ID"):
        val = os.environ.get(env)
        if val:
            return val
    return None


def current_ray_node_id() -> str | None:
    try:
        import ray

        if not ray.is_initialized():
            return None
        return str(ray.get_runtime_context().get_node_id())
    except Exception:
        return None


def discover_job_node_ids(
    *,
    list_actors: Callable | None = None,
    job_id: str | None = None,
) -> set[str]:
    """返回本作业相关的 Ray node_id 集合（含 driver 所在节点）。"""
    nodes: set[str] = set()
    cur = current_ray_node_id()
    if cur:
        nodes.add(cur)

    jid = job_id or runtime_ray_job_id()
    if not jid:
        return nodes

    if list_actors is None:
        try:
            from ray.util.state import list_actors as _list_actors

            list_actors = _list_actors
        except Exception:
            return nodes

    try:
        actors = list_actors(
            filters=[("job_id", "=", jid)],
            limit=500,
            detail=True,
            timeout=5,
        )
    except Exception:
        return nodes

    for actor in actors or []:
        state = getattr(actor, "state", None) or ""
        if str(state).upper() in ("DEAD", "RESTARTING"):
            continue
        node_id = getattr(actor, "node_id", None)
        if node_id:
            nodes.add(str(node_id))
    return nodes
