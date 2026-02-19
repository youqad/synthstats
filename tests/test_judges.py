"""Tests for Judge implementations - WRITTEN FIRST per TDD."""

import math
import sys
import types

import pytest

from synthstats.core.types import Message, Reward, Trajectory


def make_trajectory(messages: list[Message] | None = None) -> Trajectory:
    """Helper to create a test trajectory."""
    if messages is None:
        messages = [Message(role="user", content="test")]
    return Trajectory(
        messages=messages,
        token_ids=[],
        token_logprobs=[],
        loss_mask=[],
        reward=Reward(total=0.0, components={}, info={}),
    )


class TestLikelihoodJudge:
    """Tests for LikelihoodJudge - ELPD-based reward computation."""

    def test_likelihood_judge_exists(self):
        from synthstats.judges import LikelihoodJudge

        judge = LikelihoodJudge()
        assert judge is not None

    def test_likelihood_judge_with_zero_elpd(self):
        from synthstats.judges import LikelihoodJudge

        judge = LikelihoodJudge(beta=0.1)
        traj = make_trajectory()
        artifacts = {"elpd": 0.0}

        reward = judge.score(task_name="test", trajectory=traj, artifacts=artifacts)

        # exp(0.1 * 0) = exp(0) = 1.0
        assert reward.total == pytest.approx(1.0)
        assert "elpd" in reward.components
        assert reward.components["elpd"] == 0.0

    def test_likelihood_judge_with_positive_elpd(self):
        from synthstats.judges import LikelihoodJudge

        judge = LikelihoodJudge(beta=0.1)
        traj = make_trajectory()
        artifacts = {"elpd": 10.0}

        reward = judge.score(task_name="test", trajectory=traj, artifacts=artifacts)

        # exp(0.1 * 10) = exp(1.0) = e
        assert reward.total == pytest.approx(math.exp(1.0))

    def test_likelihood_judge_with_negative_elpd(self):
        from synthstats.judges import LikelihoodJudge

        judge = LikelihoodJudge(beta=0.1)
        traj = make_trajectory()
        artifacts = {"elpd": -10.0}

        reward = judge.score(task_name="test", trajectory=traj, artifacts=artifacts)

        # exp(0.1 * -10) = exp(-1.0)
        assert reward.total == pytest.approx(math.exp(-1.0))

    def test_likelihood_judge_clips_extreme_values(self):
        from synthstats.judges import LikelihoodJudge

        judge = LikelihoodJudge(beta=1.0, clip_range=(-10, 10))
        traj = make_trajectory()

        # very large positive elpd should be clipped
        artifacts = {"elpd": 1000.0}
        reward = judge.score(task_name="test", trajectory=traj, artifacts=artifacts)
        assert reward.total == pytest.approx(math.exp(10.0))

        # very large negative elpd should be clipped
        artifacts = {"elpd": -1000.0}
        reward = judge.score(task_name="test", trajectory=traj, artifacts=artifacts)
        assert reward.total == pytest.approx(math.exp(-10.0))

    def test_likelihood_judge_missing_elpd_returns_default(self):
        from synthstats.judges import LikelihoodJudge

        judge = LikelihoodJudge(beta=0.1)
        traj = make_trajectory()
        artifacts = {}  # no elpd

        reward = judge.score(task_name="test", trajectory=traj, artifacts=artifacts)

        # missing elpd defaults to 0 -> exp(0) = 1.0
        assert reward.total == pytest.approx(1.0)

    def test_likelihood_judge_info_contains_log_reward(self):
        from synthstats.judges import LikelihoodJudge

        judge = LikelihoodJudge(beta=0.1)
        traj = make_trajectory()
        artifacts = {"elpd": 5.0}

        reward = judge.score(task_name="test", trajectory=traj, artifacts=artifacts)

        assert "log_reward" in reward.info
        assert reward.info["log_reward"] == pytest.approx(0.5)


class TestFormattingJudge:
    """Tests for FormattingJudge - program validity checks."""

    def test_formatting_judge_exists(self):
        from synthstats.judges import FormattingJudge

        judge = FormattingJudge()
        assert judge is not None

    def test_formatting_judge_valid_program(self):
        from synthstats.judges import FormattingJudge

        judge = FormattingJudge()
        traj = make_trajectory()
        artifacts = {"program": "import pymc as pm\nx = pm.Normal('x', 0, 1)"}

        reward = judge.score(task_name="test", trajectory=traj, artifacts=artifacts)

        assert reward.total == 1.0
        assert reward.info["is_valid"] is True

    def test_formatting_judge_detects_subprocess(self):
        from synthstats.judges import FormattingJudge

        judge = FormattingJudge()
        traj = make_trajectory()
        artifacts = {"program": "import subprocess\nsubprocess.run(['ls'])"}

        reward = judge.score(task_name="test", trajectory=traj, artifacts=artifacts)

        assert reward.total == 0.0
        assert reward.info["is_valid"] is False

    def test_formatting_judge_detects_socket(self):
        from synthstats.judges import FormattingJudge

        judge = FormattingJudge()
        traj = make_trajectory()
        artifacts = {"program": "import socket\nsocket.socket()"}

        reward = judge.score(task_name="test", trajectory=traj, artifacts=artifacts)

        assert reward.total == 0.0

    def test_formatting_judge_detects_os_system(self):
        from synthstats.judges import FormattingJudge

        judge = FormattingJudge()
        traj = make_trajectory()
        artifacts = {"program": "import os\nos.system('rm -rf /')"}

        reward = judge.score(task_name="test", trajectory=traj, artifacts=artifacts)

        assert reward.total == 0.0

    def test_formatting_judge_custom_forbidden_imports(self):
        from synthstats.judges import FormattingJudge

        judge = FormattingJudge(forbidden_imports=["pandas", "numpy"])
        traj = make_trajectory()

        # pandas should be forbidden
        artifacts = {"program": "import pandas as pd"}
        reward = judge.score(task_name="test", trajectory=traj, artifacts=artifacts)
        assert reward.total == 0.0

        # pymc should be allowed
        artifacts = {"program": "import pymc as pm"}
        reward = judge.score(task_name="test", trajectory=traj, artifacts=artifacts)
        assert reward.total == 1.0

    def test_formatting_judge_empty_program(self):
        from synthstats.judges import FormattingJudge

        judge = FormattingJudge()
        traj = make_trajectory()
        artifacts = {"program": ""}

        reward = judge.score(task_name="test", trajectory=traj, artifacts=artifacts)

        # empty program is technically valid (no forbidden imports)
        assert reward.total == 1.0

    def test_formatting_judge_missing_program(self):
        from synthstats.judges import FormattingJudge

        judge = FormattingJudge()
        traj = make_trajectory()
        artifacts = {}

        reward = judge.score(task_name="test", trajectory=traj, artifacts=artifacts)

        # missing program treated as empty, which is valid
        assert reward.total == 1.0


class TestCompositeJudge:
    """Tests for CompositeJudge - combining multiple judges."""

    def test_composite_judge_exists(self):
        from synthstats.judges import CompositeJudge

        judge = CompositeJudge([])
        assert judge is not None

    def test_composite_judge_empty_list(self):
        from synthstats.judges import CompositeJudge

        judge = CompositeJudge([])
        traj = make_trajectory()

        reward = judge.score(task_name="test", trajectory=traj, artifacts={})

        assert reward.total == 0.0
        assert reward.components == {}

    def test_composite_judge_single_judge(self):
        from synthstats.judges import CompositeJudge, LikelihoodJudge

        likelihood = LikelihoodJudge(beta=0.1)
        judge = CompositeJudge([(likelihood, 1.0)])
        traj = make_trajectory()
        artifacts = {"elpd": 0.0}

        reward = judge.score(task_name="test", trajectory=traj, artifacts=artifacts)

        # weight 1.0 * exp(0) = 1.0
        assert reward.total == pytest.approx(1.0)
        assert "LikelihoodJudge" in reward.components

    def test_composite_judge_weighted_combination(self):
        from synthstats.judges import CompositeJudge, FormattingJudge, LikelihoodJudge

        likelihood = LikelihoodJudge(beta=0.1)
        formatting = FormattingJudge()

        judge = CompositeJudge(
            [
                (likelihood, 0.7),
                (formatting, 0.3),
            ],
            scalarization_mode="weighted_sum",
        )
        traj = make_trajectory()
        artifacts = {"elpd": 0.0, "program": "x = 1"}

        reward = judge.score(task_name="test", trajectory=traj, artifacts=artifacts)

        # 0.7 * 1.0 + 0.3 * 1.0 = 1.0
        expected = 0.7 * 1.0 + 0.3 * 1.0
        assert reward.total == pytest.approx(expected)
        assert "LikelihoodJudge" in reward.components
        assert "FormattingJudge" in reward.components

    def test_composite_judge_with_failing_format(self):
        from synthstats.judges import CompositeJudge, FormattingJudge, LikelihoodJudge

        likelihood = LikelihoodJudge(beta=0.1)
        formatting = FormattingJudge()

        judge = CompositeJudge(
            [
                (likelihood, 0.7),
                (formatting, 0.3),
            ],
            scalarization_mode="weighted_sum",
        )
        traj = make_trajectory()
        artifacts = {"elpd": 0.0, "program": "import subprocess"}

        reward = judge.score(task_name="test", trajectory=traj, artifacts=artifacts)

        # 0.7 * 1.0 + 0.3 * 0.0 = 0.7
        expected = 0.7 * 1.0 + 0.3 * 0.0
        assert reward.total == pytest.approx(expected)

    def test_composite_judge_default_uses_weighted_log_product(self):
        from synthstats.core.judge import Judge
        from synthstats.judges import CompositeJudge

        class _LowJudge:
            def score(self, *, task_name: str, trajectory: Trajectory, artifacts: dict) -> Reward:
                del task_name, trajectory, artifacts
                return Reward(total=0.25, components={}, info={})

        class _HighJudge:
            def score(self, *, task_name: str, trajectory: Trajectory, artifacts: dict) -> Reward:
                del task_name, trajectory, artifacts
                return Reward(total=4.0, components={}, info={})

        assert isinstance(_LowJudge(), Judge)
        assert isinstance(_HighJudge(), Judge)

        judge = CompositeJudge(
            [
                (_LowJudge(), 0.5),
                (_HighJudge(), 0.5),
            ]
        )
        traj = make_trajectory()

        reward = judge.score(task_name="test", trajectory=traj, artifacts={})

        assert reward.total == pytest.approx(1.0)
        assert reward.info["scalarization_mode"] == "weighted_log_product"
        assert reward.info["factor_weights"] == {
            "_LowJudge": 0.5,
            "_HighJudge": 0.5,
        }

    def test_composite_judge_components_track_individual_scores(self):
        from synthstats.judges import CompositeJudge, FormattingJudge, LikelihoodJudge

        likelihood = LikelihoodJudge(beta=0.1)
        formatting = FormattingJudge()

        judge = CompositeJudge(
            [
                (likelihood, 0.5),
                (formatting, 0.5),
            ]
        )
        traj = make_trajectory()
        artifacts = {"elpd": 10.0, "program": "x = 1"}

        reward = judge.score(task_name="test", trajectory=traj, artifacts=artifacts)

        # components should contain the raw scores (not weighted)
        assert reward.components["LikelihoodJudge"] == pytest.approx(math.exp(1.0))
        assert reward.components["FormattingJudge"] == 1.0

    def test_composite_judge_implements_protocol(self):
        from synthstats.core.judge import Judge
        from synthstats.judges import CompositeJudge

        judge = CompositeJudge([])
        assert isinstance(judge, Judge)


class TestLLMCriticJudge:
    """Tests for LLMCriticJudge - LLM-as-critic for process reward modeling."""

    def test_llm_critic_judge_initialization(self):
        from synthstats.judges import LLMCriticJudge

        judge = LLMCriticJudge()
        assert judge.model_name == "gpt-4o-mini"
        assert judge.temperature == 0.3
        assert judge.num_samples == 1
        assert judge.default_score == 0.5

    def test_llm_critic_judge_custom_initialization(self):
        from synthstats.judges import LLMCriticJudge

        judge = LLMCriticJudge(
            model_name="claude-3-haiku-20240307",
            temperature=0.5,
            num_samples=3,
            default_score=0.3,
            api_base="https://api.example.com/v1",
            api_key_env_var="FAKE_LLM_KEY",
            max_tokens=512,
            timeout_s=12.0,
            max_retries=4,
        )
        assert judge.model_name == "claude-3-haiku-20240307"
        assert judge.temperature == 0.5
        assert judge.num_samples == 3
        assert judge.default_score == 0.3
        assert judge.api_base == "https://api.example.com/v1"
        assert judge.api_key_env_var == "FAKE_LLM_KEY"
        assert judge.max_tokens == 512
        assert judge.timeout_s == 12.0
        assert judge.max_retries == 4

    def test_llm_critic_judge_builds_prompt(self):
        from synthstats.judges import LLMCriticJudge

        judge = LLMCriticJudge()
        program = "import pymc as pm\nx = pm.Normal('x', 0, 1)"
        task_name = "dugongs"

        prompt = judge._build_prompt(program, task_name)

        assert "dugongs" in prompt
        assert "import pymc as pm" in prompt
        assert "code_quality" in prompt
        assert "statistical_validity" in prompt
        assert "problem_alignment" in prompt

    def test_llm_critic_judge_builds_prompt_custom_template(self):
        from synthstats.judges import LLMCriticJudge

        custom_template = "Task: {task_name}\nCode:\n{program}\nRate it."
        judge = LLMCriticJudge(prompt_template=custom_template)

        prompt = judge._build_prompt("x = 1", "test_task")

        assert prompt == "Task: test_task\nCode:\nx = 1\nRate it."

    def test_llm_critic_judge_parses_critique_valid(self):
        from synthstats.judges import LLMCriticJudge

        judge = LLMCriticJudge()
        response = """SCORES:
code_quality: 0.8
statistical_validity: 0.7
problem_alignment: 0.9
RATIONALE: Good model structure with appropriate priors."""

        scores = judge._parse_critique(response)

        assert scores["code_quality"] == pytest.approx(0.8)
        assert scores["statistical_validity"] == pytest.approx(0.7)
        assert scores["problem_alignment"] == pytest.approx(0.9)

    def test_llm_critic_judge_parses_critique_clamps_values(self):
        from synthstats.judges import LLMCriticJudge

        judge = LLMCriticJudge()
        response = """SCORES:
code_quality: 1.5
statistical_validity: -0.2
problem_alignment: 0.5
RATIONALE: Testing edge cases."""

        scores = judge._parse_critique(response)

        assert scores["code_quality"] == 1.0  # clamped from 1.5
        assert scores["statistical_validity"] == 0.0  # clamped from -0.2
        assert scores["problem_alignment"] == 0.5

    def test_llm_critic_judge_parses_critique_missing_scores(self):
        from synthstats.judges import LLMCriticJudge

        judge = LLMCriticJudge(default_score=0.5)
        response = """SCORES:
code_quality: 0.8
RATIONALE: Partial response."""

        scores = judge._parse_critique(response)

        assert scores["code_quality"] == pytest.approx(0.8)
        assert scores["statistical_validity"] == 0.5  # default
        assert scores["problem_alignment"] == 0.5  # default

    def test_llm_critic_judge_parses_critique_malformed(self):
        from synthstats.judges import LLMCriticJudge

        judge = LLMCriticJudge(default_score=0.5)
        response = "This is not in the expected format at all."

        scores = judge._parse_critique(response)

        assert scores["code_quality"] == 0.5
        assert scores["statistical_validity"] == 0.5
        assert scores["problem_alignment"] == 0.5

    def test_llm_critic_judge_parses_critique_case_insensitive(self):
        from synthstats.judges import LLMCriticJudge

        judge = LLMCriticJudge()
        response = """scores:
CODE_QUALITY: 0.6
Statistical_Validity: 0.7
PROBLEM_ALIGNMENT: 0.8
rationale: Case variations."""

        scores = judge._parse_critique(response)

        assert scores["code_quality"] == pytest.approx(0.6)
        assert scores["statistical_validity"] == pytest.approx(0.7)
        assert scores["problem_alignment"] == pytest.approx(0.8)

    def test_llm_critic_judge_score_empty_program(self):
        from synthstats.judges import LLMCriticJudge

        judge = LLMCriticJudge()
        traj = make_trajectory()
        artifacts = {"program": ""}

        reward = judge.score(task_name="test", trajectory=traj, artifacts=artifacts)

        assert reward.total == 0.0
        assert reward.components["code_quality"] == 0.0
        assert reward.info.get("error") == "empty_program"

    def test_llm_critic_judge_score_missing_program(self):
        from synthstats.judges import LLMCriticJudge

        judge = LLMCriticJudge()
        traj = make_trajectory()
        artifacts = {}

        reward = judge.score(task_name="test", trajectory=traj, artifacts=artifacts)

        assert reward.total == 0.0
        assert reward.info.get("error") == "empty_program"

    def test_llm_critic_judge_score_without_api(self):
        """Test scoring with mocked LLM call."""
        from unittest.mock import patch

        from synthstats.judges import LLMCriticJudge

        judge = LLMCriticJudge()
        traj = make_trajectory()
        artifacts = {"program": "import pymc as pm\nx = pm.Normal('x', 0, 1)"}

        mock_response = """SCORES:
code_quality: 0.8
statistical_validity: 0.9
problem_alignment: 0.7
RATIONALE: Good basic model."""

        with patch.object(judge, "_call_llm", return_value=mock_response):
            reward = judge.score(task_name="dugongs", trajectory=traj, artifacts=artifacts)

        assert reward.total == pytest.approx((0.8 + 0.9 + 0.7) / 3)
        assert reward.components["code_quality"] == pytest.approx(0.8)
        assert reward.components["statistical_validity"] == pytest.approx(0.9)
        assert reward.components["problem_alignment"] == pytest.approx(0.7)
        assert reward.info["num_samples"] == 1

    def test_llm_critic_judge_score_api_failure(self):
        """Test graceful handling when LLM call fails."""
        from unittest.mock import patch

        from synthstats.judges import LLMCriticJudge

        judge = LLMCriticJudge(default_score=0.5)
        traj = make_trajectory()
        artifacts = {"program": "x = 1"}

        with patch.object(judge, "_call_llm", return_value=None):
            reward = judge.score(task_name="test", trajectory=traj, artifacts=artifacts)

        assert reward.total == 0.5
        assert reward.components["code_quality"] == 0.5
        assert reward.info.get("error") == "llm_call_failed"

    def test_llm_critic_judge_score_multiple_samples(self):
        """Test averaging across multiple samples."""
        from unittest.mock import patch

        from synthstats.judges import LLMCriticJudge

        judge = LLMCriticJudge(num_samples=2)
        traj = make_trajectory()
        artifacts = {"program": "x = 1"}

        responses = [
            "SCORES:\ncode_quality: 0.6\nstatistical_validity: 0.8\nproblem_alignment: 0.7",
            "SCORES:\ncode_quality: 0.8\nstatistical_validity: 0.6\nproblem_alignment: 0.9",
        ]
        call_count = [0]

        def mock_call_llm(prompt):
            idx = call_count[0]
            call_count[0] += 1
            return responses[idx] if idx < len(responses) else None

        with patch.object(judge, "_call_llm", side_effect=mock_call_llm):
            reward = judge.score(task_name="test", trajectory=traj, artifacts=artifacts)

        # averaged scores
        assert reward.components["code_quality"] == pytest.approx((0.6 + 0.8) / 2)
        assert reward.components["statistical_validity"] == pytest.approx((0.8 + 0.6) / 2)
        assert reward.components["problem_alignment"] == pytest.approx((0.7 + 0.9) / 2)
        assert reward.info["num_samples"] == 2

    def test_llm_critic_judge_call_llm_passes_provider_settings(self, monkeypatch):
        from synthstats.judges import LLMCriticJudge

        captured: dict[str, object] = {}

        def _completion(**kwargs):
            captured.update(kwargs)
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="ok"))]
            )

        monkeypatch.setenv("FAKE_LLM_KEY", "secret")
        monkeypatch.setitem(sys.modules, "litellm", types.SimpleNamespace(completion=_completion))

        judge = LLMCriticJudge(
            model_name="openai/gpt-5-mini",
            api_base="https://proxy.example.com/v1",
            api_key_env_var="FAKE_LLM_KEY",
            max_tokens=321,
            timeout_s=9.0,
            max_retries=2,
            temperature=0.1,
        )

        response = judge._call_llm("hello")

        assert response == "ok"
        assert captured["model"] == "openai/gpt-5-mini"
        assert captured["api_base"] == "https://proxy.example.com/v1"
        assert captured["api_key"] == "secret"
        assert captured["max_tokens"] == 321
        assert captured["timeout"] == 9.0
        assert captured["num_retries"] == 2

    def test_llm_critic_judge_num_samples_minimum(self):
        """Test that num_samples is at least 1."""
        from synthstats.judges import LLMCriticJudge

        judge = LLMCriticJudge(num_samples=0)
        assert judge.num_samples == 1

        judge = LLMCriticJudge(num_samples=-5)
        assert judge.num_samples == 1

    def test_llm_critic_judge_in_composite(self):
        """Test LLMCriticJudge works in CompositeJudge."""
        from unittest.mock import patch

        from synthstats.judges import CompositeJudge, FormattingJudge, LLMCriticJudge

        llm_critic = LLMCriticJudge()
        formatting = FormattingJudge()

        judge = CompositeJudge(
            [
                (llm_critic, 0.6),
                (formatting, 0.4),
            ],
            scalarization_mode="weighted_sum",
        )

        traj = make_trajectory()
        artifacts = {"program": "import pymc as pm\nx = pm.Normal('x', 0, 1)"}

        mock_response = """SCORES:
code_quality: 0.9
statistical_validity: 0.9
problem_alignment: 0.9
RATIONALE: Great model."""

        with patch.object(llm_critic, "_call_llm", return_value=mock_response):
            reward = judge.score(task_name="test", trajectory=traj, artifacts=artifacts)

        # 0.6 * 0.9 (llm avg) + 0.4 * 1.0 (formatting) = 0.54 + 0.4 = 0.94
        expected = 0.6 * 0.9 + 0.4 * 1.0
        assert reward.total == pytest.approx(expected)
        assert "LLMCriticJudge" in reward.components
        assert "FormattingJudge" in reward.components


class TestJudgeExports:
    """Tests that judges are properly exported from the package."""

    def test_all_judges_exported(self):
        from synthstats.judges import (
            CompositeJudge,
            FormattingJudge,
            LikelihoodJudge,
            LLMCriticJudge,
        )

        assert CompositeJudge is not None
        assert LikelihoodJudge is not None
        assert FormattingJudge is not None
        assert LLMCriticJudge is not None

    def test_judges_in_all(self):
        import synthstats.judges

        assert "CompositeJudge" in synthstats.judges.__all__
        assert "LikelihoodJudge" in synthstats.judges.__all__
        assert "FormattingJudge" in synthstats.judges.__all__
        assert "LLMCriticJudge" in synthstats.judges.__all__
