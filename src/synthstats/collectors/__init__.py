"""Trajectory collectors for SynthStats.

Provides SimpleCollector for on-policy trajectory collection and
batch building utilities for GFlowNet training.
"""

from synthstats.collectors.simple_collector import SimpleCollector, build_subtb_batch

__all__ = ["SimpleCollector", "build_subtb_batch"]
