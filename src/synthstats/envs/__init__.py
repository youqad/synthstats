"""SynthStats environment wrappers for external frameworks."""

from synthstats.envs.boxing_env import SKYRL_AVAILABLE, BoxingEnv, BoxingEnvConfig
from synthstats.envs.skyrl_text_env import SynthStatsTextEnv

__all__ = [
    "SynthStatsTextEnv",
    "BoxingEnv",
    "BoxingEnvConfig",
    "SKYRL_AVAILABLE",
]
