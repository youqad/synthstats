"""Eight Schools environment for BoxingGym.

Classic hierarchical Bayesian model for SAT coaching effects.

True model:
- mu ~ N(0, 10): global mean effect
- tau ~ HalfNormal(10): between-school standard deviation
- theta_j ~ N(mu, tau): true effect for school j
- y_j ~ N(theta_j, sigma_j): observed effect with known standard error

The agent queries school IDs to observe effects with their standard errors.
"""

import random


class EightSchoolsEnv:
    """Eight Schools hierarchical model environment.

    Models SAT coaching effects across 8 schools.
    Agent queries: school_id (1-8) -> (effect, standard_error)
    """

    # known standard errors from the original Rubin (1981) study
    KNOWN_SES = [15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0]

    def __init__(self):
        self.mu = 0.0  # global mean
        self.tau = 5.0  # between-school sd
        self.theta = [0.0] * 8  # true effects
        self.observed_effects = [0.0] * 8  # observed with noise
        self.rng = random.Random()

    def reset(self, seed: int | None = None) -> None:
        """Reset the environment with optional seed.

        Generates new hierarchical parameters and school effects.
        """
        if seed is not None:
            self.rng.seed(seed)

        # sample hyperparameters
        self.mu = self.rng.gauss(0, 10)
        # half-normal: |Z| where Z ~ N(0, 10)
        self.tau = abs(self.rng.gauss(0, 10))

        # sample true effects for each school
        # floor tau at 0.1 to avoid degenerate sampling when tau ≈ 0
        self.theta = [self.rng.gauss(self.mu, max(0.1, self.tau)) for _ in range(8)]

        # generate observed effects (with known standard errors)
        self.observed_effects = [self.rng.gauss(self.theta[j], self.KNOWN_SES[j]) for j in range(8)]

    def query(self, query: str) -> str:
        """Query the environment with a school ID.

        Args:
            query: Query string in format "school=<1-8>"

        Returns:
            Result string with effect estimate and standard error, or error message.
        """
        if "school=" in query:
            try:
                school_id = int(query.split("school=")[1].split()[0])
                if 1 <= school_id <= 8:
                    idx = school_id - 1
                    effect = self.observed_effects[idx]
                    se = self.KNOWN_SES[idx]
                    return f"effect={effect:.1f}, se={se:.1f}"
                return "Invalid school ID. Use 1-8."
            except (ValueError, IndexError):
                pass
        return "Invalid query. Use format: school=<1-8>"
