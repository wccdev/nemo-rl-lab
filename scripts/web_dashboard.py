#!/usr/bin/env python
"""[已废弃] 旧版内嵌 HTML 面板。请改用 NeMo-RL Lab Console：

    uv run lab web              # 新控制台（需 pnpm -C web build 或 dev 联调）
    python -m nemo_rl_lab.web.server --address ... --no-auth

本脚本保留为兼容 shim，转发到新服务。
"""
from __future__ import annotations

import sys


def main() -> None:
    print("web_dashboard.py 已废弃，请使用: uv run lab web", file=sys.stderr)
    from nemo_rl_lab.web.server import main as run

    run()


if __name__ == "__main__":
    main()
