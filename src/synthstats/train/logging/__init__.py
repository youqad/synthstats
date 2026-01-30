"""Logging - metric sinks.

- LoggerSink: Protocol for logging implementations
- StdoutLogger: Console logging
- WandbLogger: Weights & Biases logging
"""

from synthstats.train.logging.base import LoggerSink
from synthstats.train.logging.stdout import StdoutLogger

__all__ = [
    "LoggerSink",
    "StdoutLogger",
]
