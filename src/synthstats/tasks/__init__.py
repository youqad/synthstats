"""Task plugins."""

from synthstats.tasks.boxing import BoxingCodec, BoxingState, BoxingTask
from synthstats.tasks.toy import ToyState, ToyTask

__all__ = ["BoxingCodec", "BoxingState", "BoxingTask", "ToyState", "ToyTask"]
