"""Web 服务配置。"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class WebSettings:
    repo_root: Path
    ray_address: str
    host: str = "127.0.0.1"
    port: int = 8080
    no_auth: bool = False
    jwt_secret: str = "change-me-in-production"
    jwt_hours: int = 72
    db_path: Path = field(default_factory=lambda: Path(".lab/web.db"))
    static_dir: Path | None = None  # web/dist
    cache_ttl: float = 5.0

    @property
    def ledger_path(self) -> Path:
        return self.repo_root / ".lab" / "runs.jsonl"
