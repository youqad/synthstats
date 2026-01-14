"""Boxing environments package."""

from synthstats.tasks.boxing.envs.dugongs_env import DugongsEnv
from synthstats.tasks.boxing.envs.eight_schools_env import EightSchoolsEnv
from synthstats.tasks.boxing.envs.peregrines_env import PeregrinesEnv
from synthstats.tasks.boxing.envs.surgical_env import SurgicalEnv

__all__ = ["DugongsEnv", "EightSchoolsEnv", "PeregrinesEnv", "SurgicalEnv"]
