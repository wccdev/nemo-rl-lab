"""lab web 启动前自动构建前端（web/dist）。"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

_WEB_TRACK = frozenset({"index.html", "vite.config.ts", "package.json", "pnpm-lock.yaml"})


def _dist_stale(web_dir: Path) -> bool:
    dist_index = web_dir / "dist" / "index.html"
    if not dist_index.is_file():
        return True
    dist_mtime = dist_index.stat().st_mtime
    for rel in _WEB_TRACK:
        p = web_dir / rel
        if p.is_file() and p.stat().st_mtime > dist_mtime:
            return True
    src = web_dir / "src"
    if src.is_dir():
        for p in src.rglob("*"):
            if p.is_file() and p.stat().st_mtime > dist_mtime:
                return True
    return False


def _needs_install(web_dir: Path) -> bool:
    nm = web_dir / "node_modules"
    if not nm.is_dir():
        return True
    lock = web_dir / "pnpm-lock.yaml"
    return lock.is_file() and lock.stat().st_mtime > nm.stat().st_mtime


def _run(cmd: list[str], cwd: Path) -> None:
    print("› " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def ensure_frontend_built(repo_root: Path, *, skip: bool = False) -> Path:
    """确保 web/dist 存在且不过期；返回 dist 目录。"""
    web_dir = repo_root / "web"
    dist = web_dir / "dist"
    if skip:
        return dist
    if not (web_dir / "package.json").is_file():
        raise RuntimeError(f"找不到前端工程：{web_dir / 'package.json'}")

    if not shutil.which("node"):
        raise RuntimeError("未找到 Node.js。请安装 Node 18+（https://nodejs.org）后再运行 lab web。")
    pnpm = shutil.which("pnpm")
    if not pnpm:
        raise RuntimeError("未找到 pnpm。安装：npm install -g pnpm 或 corepack enable")

    if _needs_install(web_dir):
        _run([pnpm, "install", "--frozen-lockfile"], web_dir)
    if _dist_stale(web_dir):
        _run([pnpm, "build"], web_dir)

    if not (dist / "index.html").is_file():
        raise RuntimeError("前端构建失败：web/dist/index.html 不存在")
    return dist
