"""FormattingJudge - program validity checks.

Simple DSL checks for well-formed programs. Detects forbidden imports
and other safety violations.
"""

from synthstats.core.types import Reward, Trajectory


class FormattingJudge:
    """Checks if the output program is well-formed.

    Returns 1.0 if valid, 0.0 otherwise.
    Checks for forbidden imports that could be security risks.

    Args:
        forbidden_imports: List of import patterns to reject.
            Defaults to ["subprocess", "socket", "os.system"].
    """

    def __init__(self, forbidden_imports: list[str] | None = None):
        self.forbidden_imports = forbidden_imports or [
            "subprocess",
            "socket",
            "os.system",
        ]

    def score(
        self, *, task_name: str, trajectory: Trajectory, artifacts: dict
    ) -> Reward:
        """Check program validity.

        Args:
            task_name: Name of the task (unused in this judge).
            trajectory: Complete episode trajectory (unused in this judge).
            artifacts: Should contain "program" key with code to check.

        Returns:
            Reward with 1.0 if valid, 0.0 if invalid.
        """
        program = artifacts.get("program", "")
        is_valid = self._check_program(program)

        return Reward(
            total=1.0 if is_valid else 0.0,
            components={},
            info={"is_valid": is_valid},
        )

    def _check_program(self, code: str) -> bool:
        """Check program for forbidden patterns.

        Uses simple string matching for Phase 1.
        Could be extended to AST analysis for more robust checking.
        """
        for forbidden in self.forbidden_imports:
            if forbidden in code:
                return False
        return True
