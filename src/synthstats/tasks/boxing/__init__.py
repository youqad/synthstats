"""Boxing task package - BoxingGym environment for probabilistic program synthesis."""

from synthstats.tasks.boxing.codecs import BoxingCodec
from synthstats.tasks.boxing.task import BoxingState, BoxingTask

__all__ = ["BoxingCodec", "BoxingState", "BoxingTask"]
