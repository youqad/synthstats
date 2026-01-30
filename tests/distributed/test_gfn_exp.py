"""Tests for GFlowNetExp experiment class."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


class TestGFlowNetExpWithoutSkyRL:
    """Tests when SkyRL is not available."""

    def test_raises_import_error_when_skyrl_unavailable(self) -> None:
        """GFlowNetExp init fails without SkyRL."""
        import synthstats.distributed.gfn_exp as gfn_exp_module

        original_available = gfn_exp_module.SKYRL_AVAILABLE
        try:
            gfn_exp_module.SKYRL_AVAILABLE = False
            cfg = MagicMock()
            with pytest.raises(ImportError, match="SkyRL is required"):
                gfn_exp_module.GFlowNetExp(cfg)
        finally:
            gfn_exp_module.SKYRL_AVAILABLE = original_available


class TestGFlowNetExpHelpers:
    """Tests for helper methods that don't require SkyRL."""

    def test_make_placeholder_prompts(self) -> None:
        """Placeholder prompts contain seed and PyMC."""
        from synthstats.distributed.gfn_exp import GFlowNetExp

        # call unbound (no self needed, it uses PromptDataset mock)
        exp = object.__new__(GFlowNetExp)
        # mock PromptDataset since skyrl may not be installed
        import sys

        mock_skyrl = MagicMock()
        mock_skyrl.data.prompt_dataset.PromptDataset = lambda prompts: prompts
        sys.modules["skyrl_train"] = mock_skyrl
        sys.modules["skyrl_train.data"] = mock_skyrl.data
        sys.modules["skyrl_train.data.prompt_dataset"] = mock_skyrl.data.prompt_dataset

        try:
            prompts = exp._make_placeholder_prompts(5)
            assert len(prompts) == 5
            assert "seed=0" in prompts[0]
            assert "seed=4" in prompts[4]
            assert "PyMC" in prompts[0]
        finally:
            sys.modules.pop("skyrl_train", None)
            sys.modules.pop("skyrl_train.data", None)
            sys.modules.pop("skyrl_train.data.prompt_dataset", None)

    def test_load_prompt_file(self, tmp_path) -> None:
        """Loads prompts from text file."""
        import sys

        from synthstats.distributed.gfn_exp import GFlowNetExp

        prompt_file = tmp_path / "prompts.txt"
        prompt_file.write_text("prompt one\nprompt two\n\nprompt three\n")

        mock_skyrl = MagicMock()
        mock_skyrl.data.prompt_dataset.PromptDataset = lambda prompts: prompts
        sys.modules["skyrl_train"] = mock_skyrl
        sys.modules["skyrl_train.data"] = mock_skyrl.data
        sys.modules["skyrl_train.data.prompt_dataset"] = mock_skyrl.data.prompt_dataset

        try:
            exp = object.__new__(GFlowNetExp)
            prompts = exp._load_prompt_file(str(prompt_file))
            assert prompts == ["prompt one", "prompt two", "prompt three"]
        finally:
            sys.modules.pop("skyrl_train", None)
            sys.modules.pop("skyrl_train.data", None)
            sys.modules.pop("skyrl_train.data.prompt_dataset", None)

    def test_load_prompt_file_missing(self, tmp_path) -> None:
        """Falls back to placeholders for missing file."""
        import sys

        from synthstats.distributed.gfn_exp import GFlowNetExp

        mock_skyrl = MagicMock()
        mock_skyrl.data.prompt_dataset.PromptDataset = lambda prompts: prompts
        sys.modules["skyrl_train"] = mock_skyrl
        sys.modules["skyrl_train.data"] = mock_skyrl.data
        sys.modules["skyrl_train.data.prompt_dataset"] = mock_skyrl.data.prompt_dataset

        try:
            exp = object.__new__(GFlowNetExp)
            prompts = exp._load_prompt_file("/nonexistent/path.txt")
            assert len(prompts) == 100
            assert "PyMC" in prompts[0]
        finally:
            sys.modules.pop("skyrl_train", None)
            sys.modules.pop("skyrl_train.data", None)
            sys.modules.pop("skyrl_train.data.prompt_dataset", None)

    def test_no_task_imports_in_module(self) -> None:
        """gfn_exp.py must not import task plugins (dependency inversion)."""
        import inspect

        from synthstats.distributed import gfn_exp

        source = inspect.getsource(gfn_exp)
        assert "from synthstats.tasks" not in source
        assert "import synthstats.tasks" not in source


class TestModuleStructure:
    """Tests for module-level structure."""

    def test_skyrl_available_flag_exists(self) -> None:
        """SKYRL_AVAILABLE flag exposed."""
        from synthstats.distributed.gfn_exp import SKYRL_AVAILABLE

        assert isinstance(SKYRL_AVAILABLE, bool)

    def test_main_function_exists(self) -> None:
        """main() entrypoint defined."""
        from synthstats.distributed.gfn_exp import main

        assert callable(main)

    def test_gflownet_exp_class_exists(self) -> None:
        """GFlowNetExp class exported."""
        from synthstats.distributed.gfn_exp import GFlowNetExp

        assert GFlowNetExp is not None
        assert hasattr(GFlowNetExp, "get_trainer")
        assert hasattr(GFlowNetExp, "_setup_trainer")
        assert hasattr(GFlowNetExp, "_build_models_no_critic")
        assert hasattr(GFlowNetExp, "get_train_dataset")
        assert hasattr(GFlowNetExp, "_load_prompt_file")
        assert hasattr(GFlowNetExp, "_make_placeholder_prompts")
