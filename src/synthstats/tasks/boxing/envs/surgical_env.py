"""Surgical environment for BoxingGym.

Models hospital survival/mortality data for CABG surgery.

True model:
- deaths_j ~ Poisson(exposure_j * hazard_j)
- log(hazard_j) ~ N(mu, sigma): log-hazard varies by hospital

Parameters:
- mu: baseline log-hazard rate
- sigma: between-hospital variation in log-hazard

The agent queries hospital IDs to observe (deaths, exposure) pairs.
"""

from __future__ import annotations

import math
import random

from synthstats.tasks.boxing.envs.query_parsing import parse_query_int


class SurgicalEnv:
    """Surgical mortality environment.

    Models hospital-level variation in surgical outcomes.
    Agent queries: hospital_id (1-12) -> (deaths, exposure)
    """

    # exposures (number of operations) - based on BUGS examples
    # these represent person-years at risk or procedure counts
    DEFAULT_EXPOSURES = [
        1767,
        2367,
        2145,
        1098,
        2389,
        1876,
        2134,
        1567,
        2897,
        1234,
        1876,
        2456,
    ]

    def __init__(self):
        self.n_hospitals = 12
        self.mu = -5.0  # baseline log-hazard (low mortality)
        self.sigma = 0.5  # between-hospital variation
        self.log_hazards: list[float] = []
        self.hazards: list[float] = []
        self.deaths: list[int] = []
        self.exposures: list[int] = list(self.DEFAULT_EXPOSURES)
        self.rng = random.Random()

    def reset(self, seed: int | None = None) -> None:
        """Reset the environment with optional seed.

        Generates new hospital-level hazard rates and death counts.
        """
        if seed is not None:
            self.rng.seed(seed)

        # randomize hyperparameters slightly
        self.mu = self.rng.uniform(-6.0, -4.0)  # log-hazard ~ 0.2% to 2% base rate
        self.sigma = self.rng.uniform(0.3, 0.8)

        # sample log-hazards for each hospital
        self.log_hazards = [self.rng.gauss(self.mu, self.sigma) for _ in range(self.n_hospitals)]
        self.hazards = [math.exp(lh) for lh in self.log_hazards]

        # randomize exposures slightly around defaults
        self.exposures = [
            max(100, int(e * self.rng.uniform(0.8, 1.2))) for e in self.DEFAULT_EXPOSURES
        ]

        # sample deaths from Poisson
        self.deaths = [
            self._poisson(self.exposures[j] * self.hazards[j]) for j in range(self.n_hospitals)
        ]

    def _poisson(self, lam: float) -> int:
        """Sample from Poisson distribution."""
        # Knuth algorithm for small lambda
        if lam < 30:
            L = math.exp(-lam)
            k = 0
            p = 1.0
            while p > L:
                k += 1
                p *= self.rng.random()
            return k - 1
        else:
            # normal approximation for large lambda
            return max(0, round(self.rng.gauss(lam, math.sqrt(lam))))

    def query(self, query: str) -> str:
        """Query the environment with a hospital ID.

        Args:
            query: Query string in format "hospital=<1-12>"

        Returns:
            Result string with deaths and exposure, or error message.
        """
        hospital_id = parse_query_int(query, "hospital")
        if hospital_id is None:
            return f"Invalid query. Use format: hospital=<1-{self.n_hospitals}>"
        if not 1 <= hospital_id <= self.n_hospitals:
            return f"Invalid hospital ID. Use 1-{self.n_hospitals}."

        idx = hospital_id - 1
        deaths = self.deaths[idx]
        exposure = self.exposures[idx]
        return f"deaths={deaths}, exposure={exposure}"
