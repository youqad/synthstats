"""HuggingFace Policy wrapper for LLM inference.

DEPRECATED: This module re-exports from synthstats.policies.hf_policy.
Use `from synthstats.policies import HFPolicy, MockPolicy` instead.

Provides two implementations:
- HFPolicy: Real HuggingFace model wrapper with GPU support
- MockPolicy: Deterministic mock for testing without loading models
"""

from __future__ import annotations

import warnings

# re-export from canonical location
from synthstats.policies.hf_policy import HFPolicy, MockPolicy

# emit deprecation warning on import
warnings.warn(
    "synthstats.modeling.hf_policy is deprecated. "
    "Use synthstats.policies.hf_policy instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["HFPolicy", "MockPolicy"]
