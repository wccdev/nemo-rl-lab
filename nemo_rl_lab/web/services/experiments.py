"""实验目录扫描。"""
from __future__ import annotations

from pathlib import Path


def list_experiments(repo_root: Path) -> list[dict]:
    out: list[dict] = []
    for kind in ("experiments", "projects"):
        base = repo_root / kind
        if not base.is_dir():
            continue
        for p in sorted(base.iterdir()):
            if not p.is_dir() or p.name.startswith("."):
                continue
            cluster_file = p / "cluster"
            profile = cluster_file.read_text().strip() if cluster_file.is_file() else None
            out.append({
                "name": p.name,
                "kind": kind,
                "path": f"{kind}/{p.name}",
                "profile": profile,
                "has_config": (p / "config.yaml").is_file(),
            })
    return out
