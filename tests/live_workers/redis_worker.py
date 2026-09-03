"""Re-export worker_main for tests.live_workers.redis_worker compatibility."""

from __future__ import annotations

from semanticmemo.testing.live_worker import worker_main

__all__ = ["worker_main"]


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
