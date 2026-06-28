"""job_nodes：发现本 Ray 作业占用的节点。"""
from common.observability.job_nodes import discover_job_node_ids, runtime_ray_job_id


class _Actor:
    def __init__(self, node_id: str, state: str = "ALIVE"):
        self.node_id = node_id
        self.state = state


def test_discover_job_node_ids_from_actors():
    def _list(**kwargs):
        assert kwargs["filters"] == [("job_id", "=", "job-abc")]
        return [_Actor("node-a"), _Actor("node-b"), _Actor("node-a")]

    nodes = discover_job_node_ids(
        list_actors=_list,
        job_id="job-abc",
    )
    assert nodes == {"node-a", "node-b"}


def test_discover_job_node_ids_skips_dead():
    def _list(**kwargs):
        return [_Actor("node-a", "ALIVE"), _Actor("node-b", "DEAD")]

    nodes = discover_job_node_ids(list_actors=_list, job_id="j1")
    assert nodes == {"node-a"}


def test_runtime_ray_job_id_env_fallback(monkeypatch):
    monkeypatch.setenv("RAY_JOB_ID", "env-job")
    assert runtime_ray_job_id() == "env-job"
