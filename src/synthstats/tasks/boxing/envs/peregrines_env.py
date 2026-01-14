"""Peregrines environment for BoxingGym.

Models peregrine falcon population dynamics using discrete logistic growth:
N_t = N_{t-1} + r * N_{t-1} * (1 - N_{t-1}/K)

This is the standard discrete-time logistic growth model where:
- r: per-capita growth rate (~1.5-2.5)
- K: carrying capacity (~100-500)
- N_0: initial population

Observations include Gaussian noise. The agent queries years to observe counts.
"""

import random


class PeregrinesEnv:
    """Peregrine falcon population dynamics environment.

    Uses discrete logistic growth model with observation noise.
    Agent queries: year -> population count
    """

    def __init__(self):
        self.r = 2.0  # growth rate
        self.K = 200.0  # carrying capacity
        self.N_0 = 50.0  # initial population
        self.noise_std = 5.0
        self.rng = random.Random()
        self._population_cache: dict[int, float] = {}

    def reset(self, seed: int | None = None) -> None:
        """Reset the environment with optional seed.

        Randomizes parameters within reasonable ranges.
        """
        if seed is not None:
            self.rng.seed(seed)

        # randomize parameters
        self.r = self.rng.uniform(1.5, 2.5)
        self.K = self.rng.uniform(100, 500)
        self.N_0 = self.rng.uniform(20, 80)
        self.noise_std = self.rng.uniform(3.0, 8.0)
        self._population_cache = {0: self.N_0}

    def _get_population(self, year: int) -> float:
        """Compute population at given year using logistic growth."""
        if year in self._population_cache:
            return self._population_cache[year]

        if year < 0:
            return self.N_0

        # compute iteratively from the last cached year
        max_cached = max(self._population_cache.keys())
        N = self._population_cache[max_cached]

        for t in range(max_cached + 1, year + 1):
            # discrete logistic growth: N_t = N_{t-1} + r * N_{t-1} * (1 - N_{t-1}/K)
            N = N + self.r * N * (1 - N / self.K)
            N = max(0, N)  # population can't be negative
            self._population_cache[t] = N

        return N

    def query(self, query: str) -> str:
        """Query the environment with a year.

        Args:
            query: Query string in format "year=<number>"

        Returns:
            Result string with observed population count, or error message.
        """
        if "year=" in query:
            try:
                year = int(query.split("year=")[1].split()[0])
                if year < 0:
                    return "Invalid query. Year must be non-negative."

                true_pop = self._get_population(year)
                noise = self.rng.gauss(0, self.noise_std)
                observed = max(0, round(true_pop + noise))
                return f"population={observed}"
            except (ValueError, IndexError):
                pass
        return "Invalid query. Use format: year=<integer>"
