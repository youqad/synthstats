from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

from synthstats.policies.parsing import estimate_entropy, parse_action
from synthstats.policies.runpod_policy import RunPodConfig, RunPodPolicy


def _try_import_openai() -> bool:
    try:
        import openai  # noqa: F401

        return True
    except ImportError:
        return False


@dataclass
class _MockTopLogprob:
    token: str
    logprob: float


@dataclass
class _MockTokenLogprob:
    token: str
    logprob: float
    top_logprobs: list[_MockTopLogprob] = field(default_factory=list)


@dataclass
class _MockLogprobs:
    content: list[_MockTokenLogprob] = field(default_factory=list)


@dataclass
class _MockMessage:
    content: str = '{"type": "answer", "payload": "42"}'


@dataclass
class _MockChoice:
    message: _MockMessage = field(default_factory=_MockMessage)
    logprobs: _MockLogprobs | None = None
    finish_reason: str = "stop"


@dataclass
class _MockChatResponse:
    choices: list[_MockChoice] = field(default_factory=list)


@dataclass
class _MockCompletionLogprobs:
    token_logprobs: list[float | None] = field(default_factory=list)
    tokens: list[str] = field(default_factory=list)


@dataclass
class _MockCompletionChoice:
    text: str = ""
    logprobs: _MockCompletionLogprobs | None = None
    finish_reason: str = "stop"


@dataclass
class _MockCompletionResponse:
    choices: list[_MockCompletionChoice] = field(default_factory=list)


def _make_chat_response(
    text: str = '{"type": "answer", "payload": "42"}',
    token_logprobs: list[float] | None = None,
    eos_token_in_top: str | None = None,
    eos_logprob: float = -5.0,
) -> _MockChatResponse:
    if token_logprobs is None:
        token_logprobs = [-0.1, -0.2, -0.15, -0.1, -0.2]

    content = []
    for i, lp in enumerate(token_logprobs):
        top = [_MockTopLogprob(token=f"tok_{i}", logprob=lp)]
        if i == len(token_logprobs) - 1 and eos_token_in_top:
            top.append(_MockTopLogprob(token=eos_token_in_top, logprob=eos_logprob))
        content.append(_MockTokenLogprob(token=f"tok_{i}", logprob=lp, top_logprobs=top))

    return _MockChatResponse(
        choices=[_MockChoice(message=_MockMessage(content=text), logprobs=_MockLogprobs(content=content))]
    )


def _make_completion_response(
    prompt_logprobs: list[float | None],
    action_logprobs: list[float],
    prompt_tokens: list[str] | None = None,
    action_tokens: list[str] | None = None,
) -> _MockCompletionResponse:
    all_lps = prompt_logprobs + action_logprobs
    all_toks = (prompt_tokens or [f"p{i}" for i in range(len(prompt_logprobs))]) + (
        action_tokens or [f"a{i}" for i in range(len(action_logprobs))]
    )
    return _MockCompletionResponse(
        choices=[
            _MockCompletionChoice(
                text="",
                logprobs=_MockCompletionLogprobs(token_logprobs=all_lps, tokens=all_toks),
            )
        ]
    )


@pytest.fixture
def config() -> RunPodConfig:
    return RunPodConfig(
        endpoint_id="test-endpoint",
        api_key="rp_test_key",
        model="test-model",
        max_tokens=100,
        temperature=0.7,
        top_logprobs=20,
    )


@pytest.fixture
def policy(config: RunPodConfig) -> RunPodPolicy:
    return RunPodPolicy(config=config)


class TestSampleWithEos:
    def test_returns_action_logp_entropy(self, policy: RunPodPolicy) -> None:
        resp = _make_chat_response(token_logprobs=[-0.1, -0.2, -0.3])
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = resp
        policy._client = mock_client

        action, logp, entropy, eos = policy.sample_with_eos("test obs")

        assert action == {"type": "answer", "payload": "42"}
        assert logp == pytest.approx(-0.6)
        assert entropy == pytest.approx(0.2)
        assert eos is None  # no tokenizer

    def test_extracts_eos_from_top_logprobs(self, policy: RunPodPolicy) -> None:
        resp = _make_chat_response(
            token_logprobs=[-0.1, -0.2],
            eos_token_in_top="<|endoftext|>",
            eos_logprob=-3.5,
        )
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = resp
        policy._client = mock_client

        mock_tok = MagicMock()
        mock_tok.eos_token_id = 151643
        mock_tok.decode.return_value = "<|endoftext|>"
        policy._tokenizer = mock_tok

        _, _, _, eos = policy.sample_with_eos("test obs")

        assert eos == pytest.approx(-3.5)

    def test_returns_none_when_eos_absent(self, policy: RunPodPolicy) -> None:
        resp = _make_chat_response(token_logprobs=[-0.1, -0.2])
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = resp
        policy._client = mock_client

        mock_tok = MagicMock()
        mock_tok.eos_token_id = 151643
        mock_tok.decode.return_value = "<|endoftext|>"
        policy._tokenizer = mock_tok

        _, _, _, eos = policy.sample_with_eos("test obs")

        assert eos is None  # eos string not in top_logprobs

    def test_call_delegates_to_sample_with_eos(self, policy: RunPodPolicy) -> None:
        resp = _make_chat_response(token_logprobs=[-0.5])
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = resp
        policy._client = mock_client

        action, logp, entropy = policy("test obs")

        assert isinstance(action, dict)
        assert isinstance(logp, float)
        assert isinstance(entropy, float)

    def test_passes_temperature_override(self, policy: RunPodPolicy) -> None:
        resp = _make_chat_response()
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = resp
        policy._client = mock_client

        policy.sample_with_eos("obs", temperature=0.3)

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["temperature"] == 0.3


class TestScoreAction:
    def test_computes_action_logprobs(self, policy: RunPodPolicy) -> None:
        # Simulate chat-echo logprobs where the final 3 tokens correspond to action text
        resp = _make_chat_response(token_logprobs=[-0.8, -0.5, -0.3, -0.2, -0.1])
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = resp
        policy._client = mock_client

        mock_tok = MagicMock()
        mock_tok.encode.return_value = [11, 12, 13]
        policy._tokenizer = mock_tok

        obs = json.dumps([{"role": "user", "content": "obs"}])
        action = {"type": "answer", "payload": "42"}
        logp, ent = policy.score_action(obs, action)

        assert isinstance(logp, torch.Tensor)
        assert logp.item() == pytest.approx(-0.6)
        assert isinstance(ent, torch.Tensor)

        expected_messages = [{"role": "user", "content": "obs"}] + [
            {"role": "assistant", "content": '{"type": "answer", "payload": "42"}'}
        ]
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["messages"] == expected_messages
        assert call_kwargs["max_tokens"] == 0

    def test_score_action_with_eos_returns_none(self, policy: RunPodPolicy) -> None:
        resp = _make_chat_response(token_logprobs=[-0.5, -0.1])
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = resp
        policy._client = mock_client

        mock_tok = MagicMock()
        mock_tok.encode.return_value = [1]
        policy._tokenizer = mock_tok

        logp, ent, eos = policy.score_action_with_eos(
            "obs", {"type": "answer", "payload": "x"}
        )

        assert isinstance(logp, torch.Tensor)
        assert eos is None


class TestConfig:
    def test_api_key_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RUNPOD_API_KEY", "rp_env_key")
        cfg = RunPodConfig(endpoint_id="ep1")
        assert cfg.get_api_key() == "rp_env_key"

    def test_api_key_missing_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
        cfg = RunPodConfig(endpoint_id="ep1")
        with pytest.raises(ValueError, match="RUNPOD_API_KEY"):
            cfg.get_api_key()

    def test_endpoint_id_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RUNPOD_ENDPOINT_ID", "ep_from_env")
        cfg = RunPodConfig(api_key="key")
        assert cfg.get_endpoint_id() == "ep_from_env"

    def test_endpoint_id_missing_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("RUNPOD_ENDPOINT_ID", raising=False)
        cfg = RunPodConfig(api_key="key")
        with pytest.raises(ValueError, match="RUNPOD_ENDPOINT_ID"):
            cfg.get_endpoint_id()


class TestRetryAndErrors:
    @pytest.mark.skipif(
        not _try_import_openai(),
        reason="openai not installed",
    )
    def test_auth_error_propagates(self, policy: RunPodPolicy) -> None:
        from openai import AuthenticationError

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = AuthenticationError(
            message="invalid key",
            response=MagicMock(status_code=401, headers={}),
            body=None,
        )
        policy._client = mock_client

        with pytest.raises(AuthenticationError):
            policy.sample_with_eos("obs")


class TestEstimateEntropy:
    def test_empty_returns_zero(self) -> None:
        assert estimate_entropy([]) == 0.0

    def test_single_value(self) -> None:
        assert estimate_entropy([-0.5]) == pytest.approx(0.5)

    def test_multiple_values(self) -> None:
        assert estimate_entropy([-0.1, -0.3]) == pytest.approx(0.2)


class TestParseAction:
    def test_json_action(self) -> None:
        result = parse_action('{"type": "query", "payload": "hello"}')
        assert result == {"type": "query", "payload": "hello"}

    def test_plain_text_fallback(self) -> None:
        result = parse_action("some text")
        assert result == {"type": "answer", "payload": "some text"}

    def test_strips_think_tags(self) -> None:
        result = parse_action(
            '<think>reasoning here</think>{"type": "answer", "payload": "42"}'
        )
        assert result == {"type": "answer", "payload": "42"}


class TestBuildChatMessages:
    def test_plain_string_obs(self, policy: RunPodPolicy) -> None:
        msgs = policy._build_chat_messages("hello world")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[1]["content"] == "hello world"

    def test_json_messages_passthrough(self, policy: RunPodPolicy) -> None:
        chat = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hey"}]
        msgs = policy._build_chat_messages(json.dumps(chat))
        assert msgs == chat


class TestCollectorIntegration:
    def test_collector_probes_sample_with_eos(self, policy: RunPodPolicy) -> None:
        from synthstats.train.data.collectors import TrajectoryCollector

        resp = _make_chat_response(
            text='{"type": "answer", "payload": "done"}',
            token_logprobs=[-0.1, -0.2],
        )
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = resp
        policy._client = mock_client

        # minimal env stub
        env = MagicMock()
        env.init.return_value = ([{"role": "user", "content": "go"}], None)
        env.step.return_value = {"reward": 1.0, "done": True, "observations": []}

        collector = TrajectoryCollector(env=env, policy_fn=policy)
        trajectories = collector.collect(episodes=1, temperature=0.5)

        assert len(trajectories) == 1
        traj = trajectories[0]
        assert traj.reward == 1.0
        assert traj.log_probs.shape == (1,)
