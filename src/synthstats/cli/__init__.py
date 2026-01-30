"""SynthStats CLI module.

Provides command-line interfaces for training and evaluation.

Console scripts:
    synthstats-train: Unified training entrypoint (dispatches to runners)
"""

from synthstats.cli.train import main

__all__ = ["main"]
