"""Architecture guardrails (dependency inversion).

Project rule: `synthstats.train.*` must not import task plugins directly.
"""

from __future__ import annotations

import inspect


def test_train_runners_do_not_import_tasks() -> None:
    """Runners must not directly import synthstats.tasks.*."""
    from synthstats.train.runners import local as local_runner
    from synthstats.train.runners import tinker as tinker_runner

    for module in (local_runner, tinker_runner):
        source = inspect.getsource(module)
        assert "from synthstats.tasks" not in source
        assert "import synthstats.tasks" not in source
