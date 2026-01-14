"""Tests for runtime layer - WRITTEN FIRST per TDD.

Tests for:
- ActionCodec protocol and implementations (JSONToolCodec, XMLToolCodec)
- rollout_episode function
- RolloutConfig
"""

import json

import pytest

# --- ToolSpec Tests ---


class TestToolSpec:
    def test_tool_spec_creation(self):
        from synthstats.runtime.codecs import ToolSpec

        spec = ToolSpec(
            name="query",
            description="Run an experiment query",
            parameters={"x": {"type": "number", "description": "x coordinate"}},
        )
        assert spec.name == "query"
        assert spec.description == "Run an experiment query"
        assert "x" in spec.parameters

    def test_tool_spec_empty_parameters(self):
        from synthstats.runtime.codecs import ToolSpec

        spec = ToolSpec(name="noop", description="No-op tool", parameters={})
        assert spec.parameters == {}


# --- JSONToolCodec Tests ---


class TestJSONToolCodec:
    def test_parse_json_code_block_tool_call(self):
        from synthstats.core.types import ToolCall
        from synthstats.runtime.codecs import JSONToolCodec

        codec = JSONToolCodec()
        text = """Let me query the data.
```json
{"tool": "query", "input": {"x": 5, "y": 10}}
```
"""
        action = codec.parse(text)
        assert isinstance(action, ToolCall)
        assert action.name == "query"
        assert action.input == {"x": 5, "y": 10}

    def test_parse_inline_json_tool_call(self):
        from synthstats.core.types import ToolCall
        from synthstats.runtime.codecs import JSONToolCodec

        codec = JSONToolCodec()
        text = 'I will run {"tool": "compute", "input": {"formula": "a+b"}} now.'
        action = codec.parse(text)
        assert isinstance(action, ToolCall)
        assert action.name == "compute"

    def test_parse_final_answer(self):
        from synthstats.core.types import FinalAnswer
        from synthstats.runtime.codecs import JSONToolCodec

        codec = JSONToolCodec()
        text = """Here is my answer:
```json
{"answer": "The result is 42"}
```
"""
        action = codec.parse(text)
        assert isinstance(action, FinalAnswer)
        assert action.text == "The result is 42"

    def test_parse_program_submission(self):
        from synthstats.core.types import Program
        from synthstats.runtime.codecs import JSONToolCodec

        codec = JSONToolCodec()
        text = """```json
{"program": "import pymc as pm\\nwith pm.Model(): x = pm.Normal('x', 0, 1)", "language": "pymc"}
```"""
        action = codec.parse(text)
        assert isinstance(action, Program)
        assert "pm.Normal" in action.code
        assert action.language == "pymc"

    def test_parse_program_default_language(self):
        from synthstats.core.types import Program
        from synthstats.runtime.codecs import JSONToolCodec

        codec = JSONToolCodec()
        text = '{"program": "x = 1"}'
        action = codec.parse(text)
        assert isinstance(action, Program)
        assert action.language == "pymc"  # default

    def test_parse_no_action_raises(self):
        from synthstats.runtime.codecs import JSONToolCodec, ParseError

        codec = JSONToolCodec()
        text = "Just some text without any JSON action"
        with pytest.raises(ParseError):
            codec.parse(text)

    def test_parse_invalid_json_raises(self):
        from synthstats.runtime.codecs import JSONToolCodec, ParseError

        codec = JSONToolCodec()
        text = '{"tool": "query", "input": {invalid json}'
        with pytest.raises(ParseError):
            codec.parse(text)

    def test_format_action_spec(self):
        from synthstats.runtime.codecs import JSONToolCodec, ToolSpec

        codec = JSONToolCodec()
        tools = [
            ToolSpec(
                name="query",
                description="Query experiment",
                parameters={"x": {"type": "number"}},
            ),
            ToolSpec(name="submit", description="Submit answer", parameters={}),
        ]
        spec_text = codec.format_action_spec(tools)
        assert "query" in spec_text
        assert "submit" in spec_text
        assert "json" in spec_text.lower()

    def test_render_tool_call(self):
        from synthstats.core.types import ToolCall
        from synthstats.runtime.codecs import JSONToolCodec

        codec = JSONToolCodec()
        action = ToolCall(name="query", input={"x": 5}, raw="")
        rendered = codec.render(action)
        assert "query" in rendered
        # should be parseable JSON
        parsed = json.loads(rendered)
        assert parsed["tool"] == "query"

    def test_render_final_answer(self):
        from synthstats.core.types import FinalAnswer
        from synthstats.runtime.codecs import JSONToolCodec

        codec = JSONToolCodec()
        action = FinalAnswer(text="The answer is 42")
        rendered = codec.render(action)
        parsed = json.loads(rendered)
        assert parsed["answer"] == "The answer is 42"

    def test_render_program(self):
        from synthstats.core.types import Program
        from synthstats.runtime.codecs import JSONToolCodec

        codec = JSONToolCodec()
        action = Program(code="x = 1", language="pymc")
        rendered = codec.render(action)
        parsed = json.loads(rendered)
        assert parsed["program"] == "x = 1"
        assert parsed["language"] == "pymc"


# --- XMLToolCodec Tests ---


class TestXMLToolCodec:
    def test_parse_tool_call(self):
        from synthstats.core.types import ToolCall
        from synthstats.runtime.codecs import XMLToolCodec

        codec = XMLToolCodec()
        text = """I'll query the data.
<tool name="query">
{"x": 5, "y": 10}
</tool>
"""
        action = codec.parse(text)
        assert isinstance(action, ToolCall)
        assert action.name == "query"
        assert action.input == {"x": 5, "y": 10}

    def test_parse_final_answer(self):
        from synthstats.core.types import FinalAnswer
        from synthstats.runtime.codecs import XMLToolCodec

        codec = XMLToolCodec()
        text = """<answer>The result is 42</answer>"""
        action = codec.parse(text)
        assert isinstance(action, FinalAnswer)
        assert action.text == "The result is 42"

    def test_parse_program(self):
        from synthstats.core.types import Program
        from synthstats.runtime.codecs import XMLToolCodec

        codec = XMLToolCodec()
        text = """<program language="pymc">
import pymc as pm
with pm.Model():
    x = pm.Normal('x', 0, 1)
</program>"""
        action = codec.parse(text)
        assert isinstance(action, Program)
        assert "pm.Normal" in action.code
        assert action.language == "pymc"

    def test_parse_no_action_raises(self):
        from synthstats.runtime.codecs import ParseError, XMLToolCodec

        codec = XMLToolCodec()
        text = "Just some text without any XML tags"
        with pytest.raises(ParseError):
            codec.parse(text)

    def test_format_action_spec(self):
        from synthstats.runtime.codecs import ToolSpec, XMLToolCodec

        codec = XMLToolCodec()
        tools = [
            ToolSpec(name="query", description="Query experiment", parameters={}),
        ]
        spec_text = codec.format_action_spec(tools)
        assert "query" in spec_text
        assert "<tool" in spec_text.lower() or "tool" in spec_text.lower()

    def test_render_tool_call(self):
        from synthstats.core.types import ToolCall
        from synthstats.runtime.codecs import XMLToolCodec

        codec = XMLToolCodec()
        action = ToolCall(name="query", input={"x": 5}, raw="")
        rendered = codec.render(action)
        assert "<tool" in rendered
        assert "query" in rendered

    def test_render_final_answer(self):
        from synthstats.core.types import FinalAnswer
        from synthstats.runtime.codecs import XMLToolCodec

        codec = XMLToolCodec()
        action = FinalAnswer(text="The answer is 42")
        rendered = codec.render(action)
        assert "<answer>" in rendered
        assert "The answer is 42" in rendered


# --- RolloutConfig Tests ---


class TestRolloutConfig:
    def test_rollout_config_defaults(self):
        from synthstats.runtime.rollout import RolloutConfig

        cfg = RolloutConfig()
        assert cfg.max_steps > 0
        assert cfg.tool_timeout_s > 0

    def test_rollout_config_custom_values(self):
        from synthstats.runtime.rollout import RolloutConfig

        cfg = RolloutConfig(max_steps=5, tool_timeout_s=10.0, seed=42)
        assert cfg.max_steps == 5
        assert cfg.tool_timeout_s == 10.0
        assert cfg.seed == 42


# --- Rollout Episode Tests ---


class TestRolloutEpisode:
    """Tests for the main rollout loop."""

    def _make_dummy_task(self):
        """Create a minimal task for testing."""
        from synthstats.core.types import Action, FinalAnswer, Message, StepResult

        class DummyTask:
            name = "dummy"

            def reset(self, seed: int | None = None) -> dict:
                return {"step": 0, "seed": seed}

            def observe(self, state: dict) -> list[Message]:
                return [
                    Message(role="system", content="You are a test assistant."),
                    Message(role="user", content=f"Step {state['step']}"),
                ]

            def step(self, state: dict, action: Action) -> StepResult:
                new_step = state["step"] + 1
                done = isinstance(action, FinalAnswer) or new_step >= 3
                return StepResult(
                    next_state={"step": new_step, "seed": state["seed"]},
                    done=done,
                    artifacts={"last_action": str(type(action).__name__)},
                )

        return DummyTask()

    def _make_dummy_policy(self, responses: list[str]):
        """Create a minimal policy that returns predetermined responses."""
        from synthstats.core.policy import GenConfig, Generation
        from synthstats.core.types import Message

        class DummyPolicy:
            def __init__(self, responses: list[str]):
                self._responses = responses
                self._call_count = 0

            def generate(
                self, messages: list[Message], *, gen: GenConfig
            ) -> Generation:
                idx = min(self._call_count, len(self._responses) - 1)
                text = self._responses[idx]
                self._call_count += 1
                return Generation(
                    text=text,
                    token_ids=[1, 2, 3],
                    token_logprobs=[-0.1, -0.2, -0.3],
                    finish_reason="stop",
                )

            def logprobs(self, messages: list[Message], tokens: list[int]):
                from synthstats.core.policy import TokenLogProbs

                return TokenLogProbs(
                    token_ids=tokens, logprobs=[-0.1] * len(tokens)
                )

        return DummyPolicy(responses)

    def _make_dummy_executor(self):
        """Create a minimal executor for testing."""
        from synthstats.core.executor import ToolResult
        from synthstats.core.types import ToolCall

        class DummyExecutor:
            name = "dummy"

            def execute(self, payload: ToolCall, *, timeout_s: float) -> ToolResult:
                return ToolResult(
                    output=f"Executed {payload.name} with {payload.input}",
                    success=True,
                    error=None,
                )

        return DummyExecutor()

    def _make_dummy_judge(self, reward_value: float = 1.0):
        """Create a minimal judge for testing."""
        from synthstats.core.types import Reward, Trajectory

        class DummyJudge:
            def __init__(self, reward_value: float):
                self._reward_value = reward_value

            def score(
                self, *, task_name: str, trajectory: Trajectory, artifacts: dict
            ) -> Reward:
                return Reward(
                    total=self._reward_value,
                    components={"test": self._reward_value},
                    info={"artifacts": artifacts},
                )

        return DummyJudge(reward_value)

    def test_rollout_returns_trajectory(self):
        from synthstats.core.types import Trajectory
        from synthstats.runtime.codecs import JSONToolCodec
        from synthstats.runtime.rollout import RolloutConfig, rollout_episode

        task = self._make_dummy_task()
        policy = self._make_dummy_policy(['{"answer": "done"}'])
        codec = JSONToolCodec()
        judge = self._make_dummy_judge(0.9)
        cfg = RolloutConfig(max_steps=10, seed=42)

        traj = rollout_episode(
            task=task,
            policy=policy,
            codec=codec,
            executors={},
            judge=judge,
            cfg=cfg,
        )
        assert isinstance(traj, Trajectory)
        assert traj.reward.total == 0.9

    def test_rollout_terminates_on_final_answer(self):
        from synthstats.runtime.codecs import JSONToolCodec
        from synthstats.runtime.rollout import RolloutConfig, rollout_episode

        task = self._make_dummy_task()
        policy = self._make_dummy_policy(['{"answer": "The answer is 42"}'])
        codec = JSONToolCodec()
        judge = self._make_dummy_judge()
        cfg = RolloutConfig(max_steps=100)

        traj = rollout_episode(
            task=task,
            policy=policy,
            codec=codec,
            executors={},
            judge=judge,
            cfg=cfg,
        )
        # should terminate after first step due to FinalAnswer
        # messages: system, user, assistant (with answer)
        assert len(traj.messages) >= 3

    def test_rollout_executes_tool_calls(self):
        from synthstats.runtime.codecs import JSONToolCodec
        from synthstats.runtime.rollout import RolloutConfig, rollout_episode

        task = self._make_dummy_task()
        # first call returns tool call, second returns final answer
        policy = self._make_dummy_policy([
            '{"tool": "dummy", "input": {"x": 1}}',
            '{"answer": "done"}',
        ])
        codec = JSONToolCodec()
        executor = self._make_dummy_executor()
        judge = self._make_dummy_judge()
        cfg = RolloutConfig(max_steps=10)

        traj = rollout_episode(
            task=task,
            policy=policy,
            codec=codec,
            executors={"dummy": executor},
            judge=judge,
            cfg=cfg,
        )
        # should have tool result message in trajectory
        tool_messages = [m for m in traj.messages if m.role == "tool"]
        assert len(tool_messages) >= 1
        assert "Executed" in tool_messages[0].content

    def test_rollout_respects_max_steps(self):
        from synthstats.runtime.codecs import JSONToolCodec
        from synthstats.runtime.rollout import RolloutConfig, rollout_episode

        task = self._make_dummy_task()
        # policy always returns tool call, never final answer
        policy = self._make_dummy_policy(
            ['{"tool": "dummy", "input": {}}'] * 100
        )
        codec = JSONToolCodec()
        executor = self._make_dummy_executor()
        judge = self._make_dummy_judge()
        cfg = RolloutConfig(max_steps=3)

        traj = rollout_episode(
            task=task,
            policy=policy,
            codec=codec,
            executors={"dummy": executor},
            judge=judge,
            cfg=cfg,
        )
        # should terminate after max_steps
        # count assistant messages
        assistant_msgs = [m for m in traj.messages if m.role == "assistant"]
        assert len(assistant_msgs) <= 3

    def test_rollout_handles_parse_error_gracefully(self):
        from synthstats.runtime.codecs import JSONToolCodec
        from synthstats.runtime.rollout import RolloutConfig, rollout_episode

        task = self._make_dummy_task()
        # first call returns unparseable text, should trigger error handling
        policy = self._make_dummy_policy([
            "Invalid response with no JSON",
            '{"answer": "done"}',
        ])
        codec = JSONToolCodec()
        judge = self._make_dummy_judge()
        cfg = RolloutConfig(max_steps=10)

        # should not raise - parse errors are handled internally
        traj = rollout_episode(
            task=task,
            policy=policy,
            codec=codec,
            executors={},
            judge=judge,
            cfg=cfg,
        )
        assert traj is not None

    def test_rollout_handles_missing_executor(self):
        from synthstats.runtime.codecs import JSONToolCodec
        from synthstats.runtime.rollout import RolloutConfig, rollout_episode

        task = self._make_dummy_task()
        # policy calls unknown tool, then returns final answer
        policy = self._make_dummy_policy([
            '{"tool": "unknown_tool", "input": {}}',
            '{"answer": "done"}',
        ])
        codec = JSONToolCodec()
        judge = self._make_dummy_judge()
        cfg = RolloutConfig(max_steps=10)

        # should handle missing executor gracefully
        traj = rollout_episode(
            task=task,
            policy=policy,
            codec=codec,
            executors={},  # no executors
            judge=judge,
            cfg=cfg,
        )
        # check for error message in trajectory
        tool_messages = [m for m in traj.messages if m.role == "tool"]
        assert len(tool_messages) >= 1
        assert (
            "unknown" in tool_messages[0].content.lower()
            or "error" in tool_messages[0].content.lower()
        )

    def test_rollout_collects_token_data(self):
        from synthstats.runtime.codecs import JSONToolCodec
        from synthstats.runtime.rollout import RolloutConfig, rollout_episode

        task = self._make_dummy_task()
        policy = self._make_dummy_policy(['{"answer": "done"}'])
        codec = JSONToolCodec()
        judge = self._make_dummy_judge()
        cfg = RolloutConfig(max_steps=10)

        traj = rollout_episode(
            task=task,
            policy=policy,
            codec=codec,
            executors={},
            judge=judge,
            cfg=cfg,
        )
        # should have token data for assistant turns
        assert len(traj.token_ids) > 0
        assert len(traj.token_logprobs) > 0
        assert len(traj.loss_mask) > 0

    def test_rollout_passes_seed_to_task(self):
        from synthstats.core.types import Action, Message, StepResult
        from synthstats.runtime.codecs import JSONToolCodec
        from synthstats.runtime.rollout import RolloutConfig, rollout_episode

        received_seed = None

        class SeedCapturingTask:
            name = "seed_test"

            def reset(self, seed: int | None = None) -> dict:
                nonlocal received_seed
                received_seed = seed
                return {"seed": seed}

            def observe(self, state: dict) -> list[Message]:
                return [Message(role="user", content="test")]

            def step(self, state: dict, action: Action) -> StepResult:
                return StepResult(next_state=state, done=True, artifacts={})

        task = SeedCapturingTask()
        policy = self._make_dummy_policy(['{"answer": "done"}'])
        codec = JSONToolCodec()
        judge = self._make_dummy_judge()
        cfg = RolloutConfig(seed=12345)

        rollout_episode(
            task=task,
            policy=policy,
            codec=codec,
            executors={},
            judge=judge,
            cfg=cfg,
        )
        assert received_seed == 12345

    def test_rollout_accumulates_artifacts(self):
        from synthstats.core.types import Action, FinalAnswer, Message, StepResult
        from synthstats.runtime.codecs import JSONToolCodec
        from synthstats.runtime.rollout import RolloutConfig, rollout_episode

        class ArtifactTask:
            name = "artifact_test"
            _step = 0

            def reset(self, seed: int | None = None) -> dict:
                self._step = 0
                return {}

            def observe(self, state: dict) -> list[Message]:
                return [Message(role="user", content="test")]

            def step(self, state: dict, action: Action) -> StepResult:
                self._step += 1
                done = isinstance(action, FinalAnswer)
                return StepResult(
                    next_state=state,
                    done=done,
                    artifacts={f"artifact_{self._step}": f"value_{self._step}"},
                )

        task = ArtifactTask()
        policy = self._make_dummy_policy([
            '{"tool": "dummy", "input": {}}',
            '{"answer": "done"}',
        ])
        codec = JSONToolCodec()
        executor = self._make_dummy_executor()

        # judge captures artifacts
        captured_artifacts = {}

        class ArtifactCapturingJudge:
            def score(self, *, task_name, trajectory, artifacts):
                nonlocal captured_artifacts
                captured_artifacts = artifacts
                from synthstats.core.types import Reward
                return Reward(total=1.0, components={}, info={})

        judge = ArtifactCapturingJudge()
        cfg = RolloutConfig(max_steps=10)

        rollout_episode(
            task=task,
            policy=policy,
            codec=codec,
            executors={"dummy": executor},
            judge=judge,
            cfg=cfg,
        )
        # should have accumulated artifacts from both steps
        assert "artifact_1" in captured_artifacts
        assert "artifact_2" in captured_artifacts


# --- ActionCodec Protocol Tests ---


class TestActionCodecProtocol:
    def test_codec_protocol_has_parse_method(self):
        from synthstats.runtime.codecs import ActionCodec

        assert hasattr(ActionCodec, "parse")

    def test_codec_protocol_has_render_method(self):
        from synthstats.runtime.codecs import ActionCodec

        assert hasattr(ActionCodec, "render")

    def test_codec_protocol_has_format_action_spec_method(self):
        from synthstats.runtime.codecs import ActionCodec

        assert hasattr(ActionCodec, "format_action_spec")

    def test_json_codec_parse_render_roundtrip(self):
        """Behavioral test: parse(render(action)) should preserve action semantics."""
        from synthstats.core.types import FinalAnswer, Program, ToolCall
        from synthstats.runtime.codecs import JSONToolCodec

        codec = JSONToolCodec()

        # Test ToolCall roundtrip
        original_tool = ToolCall(name="query", input={"x": 10, "y": "test"}, raw="")
        rendered = codec.render(original_tool)
        parsed = codec.parse(rendered)
        assert isinstance(parsed, ToolCall)
        assert parsed.name == original_tool.name
        assert parsed.input == original_tool.input

        # Test FinalAnswer roundtrip
        original_answer = FinalAnswer(text="The answer is 42")
        rendered = codec.render(original_answer)
        parsed = codec.parse(rendered)
        assert isinstance(parsed, FinalAnswer)
        assert parsed.text == original_answer.text

        # Test Program roundtrip
        original_program = Program(code="x = pm.Normal('x', 0, 1)", language="pymc")
        rendered = codec.render(original_program)
        parsed = codec.parse(rendered)
        assert isinstance(parsed, Program)
        assert parsed.code == original_program.code
        assert parsed.language == original_program.language

    def test_xml_codec_parse_render_roundtrip(self):
        """Behavioral test: XML codec should also support roundtrip."""
        from synthstats.core.types import FinalAnswer, ToolCall
        from synthstats.runtime.codecs import XMLToolCodec

        codec = XMLToolCodec()

        # Test ToolCall roundtrip
        original_tool = ToolCall(name="compute", input={"formula": "a+b"}, raw="")
        rendered = codec.render(original_tool)
        parsed = codec.parse(rendered)
        assert isinstance(parsed, ToolCall)
        assert parsed.name == original_tool.name
        assert parsed.input == original_tool.input

        # Test FinalAnswer roundtrip
        original_answer = FinalAnswer(text="Result computed")
        rendered = codec.render(original_answer)
        parsed = codec.parse(rendered)
        assert isinstance(parsed, FinalAnswer)
        assert parsed.text == original_answer.text

    def test_codec_format_action_spec_includes_tool_info(self):
        """Behavioral test: format_action_spec should produce usable documentation."""
        from synthstats.runtime.codecs import JSONToolCodec, ToolSpec, XMLToolCodec

        tools = [
            ToolSpec(
                name="search",
                description="Search the database for records",
                parameters={
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer", "description": "Max results"},
                },
            ),
            ToolSpec(
                name="submit",
                description="Submit the final answer",
                parameters={},
            ),
        ]

        # JSON codec spec should mention tool names and be parseable guidance
        json_codec = JSONToolCodec()
        json_spec = json_codec.format_action_spec(tools)
        assert "search" in json_spec
        assert "submit" in json_spec
        assert "query" in json_spec  # parameter should be mentioned

        # XML codec spec should also include tool info
        xml_codec = XMLToolCodec()
        xml_spec = xml_codec.format_action_spec(tools)
        assert "search" in xml_spec
        assert "submit" in xml_spec

    def test_codec_parse_extracts_correct_action_type(self):
        """Behavioral test: parse should correctly identify action types."""
        from synthstats.core.types import FinalAnswer, Program, ToolCall
        from synthstats.runtime.codecs import JSONToolCodec

        codec = JSONToolCodec()

        # ToolCall detection
        tool_text = '{"tool": "query", "input": {"x": 1}}'
        action = codec.parse(tool_text)
        assert isinstance(action, ToolCall)
        assert not isinstance(action, FinalAnswer)
        assert not isinstance(action, Program)

        # FinalAnswer detection
        answer_text = '{"answer": "done"}'
        action = codec.parse(answer_text)
        assert isinstance(action, FinalAnswer)
        assert not isinstance(action, ToolCall)

        # Program detection
        program_text = '{"program": "x = 1"}'
        action = codec.parse(program_text)
        assert isinstance(action, Program)
        assert not isinstance(action, ToolCall)
