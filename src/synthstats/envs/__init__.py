"""SynthStats environment wrappers for external frameworks."""

from synthstats.envs.skyrl_text_env import SynthStatsTextEnv
from synthstats.envs.boxing_env import BoxingEnv, BoxingEnvConfig, SKYRL_AVAILABLE

__all__ = [
    "SynthStatsTextEnv",
    "BoxingEnv",
    "BoxingEnvConfig",
    "SKYRL_AVAILABLE",
]
