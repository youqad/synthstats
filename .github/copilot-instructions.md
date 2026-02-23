SynthStats is a GFlowNet-steered probabilistic program synthesis system. LLMs are trained (via SubTB loss) to sample from distributions over PyMC programs proportional to their likelihood.

## Stack
- Python 3.11+, **uv** for dependency management; run tests with `uv run pytest tests/ -x`
- PyMC 5.x, GFlowNets (SubTB loss), Hydra for config, W&B for tracking

## Hard architecture rules

**Dependency inversion (critical — flag any violation as blocking):**
- `train/*` must NOT import from `tasks/*`
- `tasks/*` must NOT import from `train/*`
- `executors/*` must not depend on any specific task
Violations break the plugin architecture and must be caught at review time.

**No backward compatibility in runtime code.** When code moves, update ALL imports and delete the old location. No shim layers, re-export facades, or compatibility aliases. Missing required configs must fail fast at init with an explicit error.

**Plugin interface completeness.** Each benchmark is a Task plugin with four components: `Task` (state machine + episode termination), `ActionCodec` (text ↔ structured action), `Executor` (sandbox tool runtime, optional), `Judge` (reward computation). New tasks must implement the full interface.

**No reward leakage across abstraction boundaries.** Judge outputs must not be used directly as GFlowNet log-rewards without going through the defined reward interface.

## Code style
- No trivial comments — code should be self-documenting
- No single-use abstractions or Manager/Service/Helper layers
- Prefer deleting stale code over wrapping it in a compat shim
- Validate at system boundaries only; trust internal plugin contracts
