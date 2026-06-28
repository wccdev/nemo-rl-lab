"""Monkey-patch nemo_rl.utils.logger：NeMoLabLogger + 验证样本结构化上报。"""
from __future__ import annotations

import os

_PATCHED = False


def apply_patch() -> None:
    global _PATCHED
    if _PATCHED:
        return
    if not os.environ.get("NEMOLAB_TOKEN"):
        return
    try:
        import nemo_rl.utils.logger as logger_mod
    except ImportError:
        print("NeMoLab patch skipped: nemo_rl not importable")
        return

    from common.observability.logger import NeMoLabLogger
    from common.observability.session import get_ingest
    from common.observability.validation_ctx import active_validation_step, clear_validation_step
    from common.observability.validation_extract import extract_message_log_samples

    _orig_init = logger_mod.Logger.__init__
    _orig_del = getattr(logger_mod.Logger, "__del__", None)
    _orig_print_samples = logger_mod.print_message_log_samples

    def _patched_init(self, cfg):
        _orig_init(self, cfg)
        nemolab_log_dir = os.path.join(self.base_log_dir, "nemolab")
        os.makedirs(nemolab_log_dir, exist_ok=True)
        try:
            self.nemolab_logger = NeMoLabLogger({}, log_dir=nemolab_log_dir)
            self.loggers.append(self.nemolab_logger)
        except Exception as e:
            print(f"NeMoLab logger init failed (training continues): {e}")
            self.nemolab_logger = None

    def _patched_del(self):
        nl = getattr(self, "nemolab_logger", None)
        if nl is not None:
            nl.finish()
        if _orig_del is not None:
            _orig_del(self)

    def _patched_print_message_log_samples(
        message_logs, rewards, num_samples=5, step=0
    ):
        _orig_print_samples(message_logs, rewards, num_samples=num_samples, step=step)
        ingest = get_ingest()
        if ingest is None:
            return
        val_step = active_validation_step()
        if val_step is None or val_step != step:
            return
        try:
            samples, dist, avg_reward = extract_message_log_samples(
                message_logs, rewards, num_samples=num_samples
            )
            if not samples:
                return
            ingest.enqueue_validation(
                {
                    "run_id": ingest.run_id,
                    "step": val_step,
                    "avg_reward": avg_reward,
                    "dist": dist,
                    "samples": samples,
                }
            )
        except Exception as e:
            print(f"NeMoLab validation upload failed (training continues): {e}")
        finally:
            clear_validation_step()

    logger_mod.Logger.__init__ = _patched_init
    logger_mod.Logger.__del__ = _patched_del
    logger_mod.print_message_log_samples = _patched_print_message_log_samples
    _PATCHED = True
    print("NeMoLab logger patch applied")
