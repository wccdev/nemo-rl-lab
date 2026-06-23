#!/usr/bin/env python
"""NeMo-RL Lab Console 服务入口（被 `lab web` 调用）。"""
from __future__ import annotations

import argparse
import os
import secrets
import threading
import webbrowser
from pathlib import Path

import uvicorn

from nemo_rl_lab.web.app import create_app
from nemo_rl_lab.web.config import WebSettings


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description="NeMo-RL Lab Console")
    ap.add_argument("--address", required=True, help="Ray dashboard 地址")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--no-auth", action="store_true", help="本机模式：跳过登录")
    ap.add_argument("--serve", action="store_true", help="绑定 0.0.0.0（团队部署）")
    ap.add_argument("--open", action="store_true", help="启动后打开浏览器")
    args = ap.parse_args()

    static = repo / "web" / "dist"
    jwt_secret = os.environ.get("LAB_WEB_JWT_SECRET") or secrets.token_hex(32)
    settings = WebSettings(
        repo_root=repo,
        ray_address=args.address,
        host="0.0.0.0" if args.serve else args.host,
        port=args.port,
        no_auth=args.no_auth,
        jwt_secret=jwt_secret,
        static_dir=static if static.is_dir() else None,
    )
    app = create_app(settings)
    url = f"http://{'127.0.0.1' if settings.host == '0.0.0.0' else settings.host}:{settings.port}"
    print(f"✓ NeMo-RL Lab Console: {url}")
    print(f"  Ray: {args.address} | auth={'off (--no-auth)' if args.no_auth else 'on'}")
    if not static.is_dir():
        print("  ⚠ 无前端静态资源（lab web 请加 --no-build 跳过构建时会出现）")
    if args.open:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="warning")


if __name__ == "__main__":
    main()
