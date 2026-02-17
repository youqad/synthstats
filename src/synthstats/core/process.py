"""Process group cleanup for subprocess isolation."""

from __future__ import annotations

import logging
import os
import signal
import time
from typing import Any

logger = logging.getLogger(__name__)


def _group_exited(group_id: int) -> bool:
    try:
        os.killpg(group_id, 0)
        return False
    except (ProcessLookupError, OSError):
        return True


def _wait_group_exit(group_id: int, timeout_s: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if _group_exited(group_id):
            return True
        time.sleep(0.05)
    return _group_exited(group_id)


def cleanup_process_group(proc: Any) -> None:
    """Best-effort SIGTERM -> SIGKILL on the process group, then reap."""
    pgid: int | None = None
    if hasattr(os, "killpg") and hasattr(os, "getpgid"):
        try:
            pgid = os.getpgid(proc.pid)
        except (ProcessLookupError, OSError) as exc:
            logger.debug("Could not resolve process group for pid %s: %s", proc.pid, exc)

    if pgid is not None:
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(pgid, sig)
            except (ProcessLookupError, OSError):
                break
            if _wait_group_exit(pgid):
                break

    try:
        proc.kill()
    except Exception as exc:
        logger.debug("Parent process kill failed during cleanup: %s", exc)
    try:
        proc.wait(timeout=2.0)
    except Exception as exc:
        logger.debug("Parent process wait failed during cleanup: %s", exc)
