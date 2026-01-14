"""Judge implementations for SynthStats.

Judges compute reward signals from trajectories. Available judges:

- CompositeJudge: Combines multiple judges with weights
- LikelihoodJudge: ELPD-based reward (stub for Phase 1)
- FormattingJudge: Program validity checks
- LLMCriticJudge: LLM-as-critic for process reward modeling
"""

from synthstats.judges.composite import CompositeJudge
from synthstats.judges.formatting import FormattingJudge
from synthstats.judges.likelihood import LikelihoodJudge
from synthstats.judges.llm_critic import LLMCriticJudge

__all__ = [
    "CompositeJudge",
    "FormattingJudge",
    "LikelihoodJudge",
    "LLMCriticJudge",
]
