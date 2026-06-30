"""训练入口包装器：先给 NeMo-RL Logger 挂上 NeMoLabLogger 后端，再运行原始入口。

由 scripts/_run_experiment.sh 调用：
    uv run python scripts/nemolab_boot.py <ENTRY> [args...]
等价于 `python <ENTRY> [args...]`，唯一区别是运行前 apply_patch()。
无 NEMOLAB_TOKEN（本地直跑）时 patch 为 no-op，行为与直接 `python <ENTRY>` 完全一致。
"""
from __future__ import annotations

import os
import runpy
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python scripts/nemolab_boot.py <entry.py> [args...]", file=sys.stderr)
        return 2

    try:
        from common.observability.session import start_observability

        start_observability()
        from common.observability.patch import apply_patch

        apply_patch()
    except Exception as e:  # 采集是旁路，任何异常都不应影响训练
        print(f"[nemolab] patch skipped: {e}")

    entry = sys.argv[1]
    sys.argv = [entry, *sys.argv[2:]]
    try:
        runpy.run_path(entry, run_name="__main__")
    except SystemExit:
        raise
    except Exception:
        import traceback

        traceback.print_exc()
        return 1
    finally:
        try:
            from common.observability.session import stop_observability

            stop_observability()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
