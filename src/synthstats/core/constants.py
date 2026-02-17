"""Shared constants for SynthStats."""

from __future__ import annotations

import math

# reward floor: minimum reward before log-transform to avoid log(0)
REWARD_FLOOR_DEFAULT: float = 1e-4

# TB loss: max absolute residual (clamp for numerical stability)
TB_MAX_RESIDUAL_DEFAULT: float = 100.0

# SubTB: lambda decay for sub-trajectory lengths
SUBTB_LAMBDA_DEFAULT: float = 0.9

# logZ learning rate (canonical: 10x base LR of 0.001)
# matches gfn-lm-tuning paper and configs/objective/tb_subtb.yaml
LOGZ_LR_DEFAULT: float = 0.01

# sparse reward for incomplete prefixes in endpoint SubTB
LOG_SPARSE_REWARD_DEFAULT: float = math.log(1e-4)

# common stop tokens by model family
STOP_TOKEN_IDS: dict[str, list[int]] = {
    "qwen3": [151643, 151645],  # <|endoftext|>, <|im_end|>
    "qwen2": [151643, 151645],
    "glm": [151329, 151336, 151338],  # <|endoftext|>, <|user|>, <|observation|>
    "llama": [2, 128001, 128009],  # </s>, Llama 3 <|end_of_text|>, <|eot_id|>
    "mistral": [2],
    "default": [2],
}
