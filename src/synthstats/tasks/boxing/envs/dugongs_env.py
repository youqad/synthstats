"""Dugongs environment for BoxingGym.

The true model: length = alpha - beta * |lambda|^age
where alpha=2.65, beta=0.97, lambda=-0.87

This is a simplified stub - the real BoxingGym env has more features.
"""

from __future__ import annotations

import random


class DugongsEnv:
    """Simple Dugongs environment.

    Models sea cow growth curves: length as a function of age.
    The underlying function is: length = alpha - beta * |lambda|^age
    """

    def __init__(self):
        self.alpha = 2.65
        self.beta = 0.97
        self.lam = -0.87
        self.rng = random.Random()

    def reset(self, seed: int | None = None) -> None:
        """Reset the environment with optional seed."""
        if seed is not None:
            self.rng.seed(seed)

    def query(self, query: str) -> str:
        """Query the environment with a simple query string.

        Args:
            query: Query string in format "age=<number>"

        Returns:
            Result string with observed length, or error message.
        """
        if "age=" in query:
            try:
                age = float(query.split("age=")[1].split()[0])
                length = self.alpha - self.beta * abs(self.lam) ** age
                noise = self.rng.gauss(0, 0.1)
                return f"length={length + noise:.3f}"
            except (ValueError, IndexError):
                pass
        return "Invalid query. Use format: age=<number>"
