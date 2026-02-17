"""Program validity checks."""

from __future__ import annotations

from synthstats.core.types import Reward, Trajectory


class FormattingJudge:
    """Returns 1.0 if program is well-formed, 0.0 otherwise."""

    def __init__(self, forbidden_imports: list[str] | None = None):
        self.forbidden_imports = forbidden_imports or [
            "subprocess",
            "socket",
            "os.system",
        ]

    def score(self, *, task_name: str, trajectory: Trajectory, artifacts: dict) -> Reward:
        program = artifacts.get("program", "")
        is_valid = self._check_program(program)

        return Reward(
            total=1.0 if is_valid else 0.0,
            components={},
            info={"is_valid": is_valid},
        )

    def _check_program(self, code: str) -> bool:
        for forbidden in self.forbidden_imports:
            if forbidden in code:
                return False
        return True
