import os
import signal
import subprocess

import pytest


def test_execute_program_for_elpd_timeout_kills_process_group(monkeypatch):
    """Timeout cleanup should not NameError and should attempt group kill."""
    if not hasattr(os, "killpg") or not hasattr(os, "getpgid"):
        pytest.skip("os.killpg/getpgid not available on this platform")

    from synthstats.envs.boxing_env import BoxingEnv

    killpg_calls: list[tuple[int, int]] = []

    def fake_getpgid(pid: int) -> int:  # noqa: ARG001
        return 999

    def fake_killpg(pgid: int, sig: int) -> None:
        killpg_calls.append((pgid, sig))

    monkeypatch.setattr(os, "getpgid", fake_getpgid)
    monkeypatch.setattr(os, "killpg", fake_killpg)

    class FakeProc:
        pid = 123

        def __init__(self) -> None:
            self._wait_calls = 0

        def communicate(self, input=None, timeout=None):  # noqa: ANN001
            raise subprocess.TimeoutExpired(cmd="python", timeout=timeout)

        def wait(self, timeout=None):  # noqa: ANN001
            self._wait_calls += 1
            if self._wait_calls == 1:
                raise subprocess.TimeoutExpired(cmd="python", timeout=timeout)
            return 0

        def kill(self) -> None:
            return None

    def fake_popen(*args, **kwargs):  # noqa: ANN001
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    env = object.__new__(BoxingEnv)
    result = BoxingEnv._execute_program_for_elpd(env, "print('hi')\n")

    assert result is None
    assert (999, signal.SIGTERM) in killpg_calls
    assert (999, signal.SIGKILL) in killpg_calls
