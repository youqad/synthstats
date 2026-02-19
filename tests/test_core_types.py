"""Tests for core types - WRITTEN FIRST per TDD."""


class TestMessage:
    def test_message_creation(self):
        from synthstats.core.types import Message

        msg = Message(role="user", content="hello")
        assert msg.role == "user"
        assert msg.content == "hello"
        assert msg.tool_call_id is None

    def test_message_with_tool_call_id(self):
        from synthstats.core.types import Message

        msg = Message(role="tool", content="result", tool_call_id="call_123")
        assert msg.tool_call_id == "call_123"

    def test_message_to_dict(self):
        from synthstats.core.types import Message

        msg = Message(role="user", content="hello", tool_call_id="call_1")
        d = msg.to_dict()
        assert d == {"role": "user", "content": "hello", "tool_call_id": "call_1"}

    def test_message_from_dict(self):
        from synthstats.core.types import Message

        d = {"role": "assistant", "content": "world", "tool_call_id": None}
        msg = Message.from_dict(d)
        assert msg.role == "assistant"
        assert msg.content == "world"
        assert msg.tool_call_id is None

    def test_message_roundtrip(self):
        from synthstats.core.types import Message

        original = Message(role="tool", content="result", tool_call_id="call_42")
        restored = Message.from_dict(original.to_dict())
        assert restored.role == original.role
        assert restored.content == original.content
        assert restored.tool_call_id == original.tool_call_id

    def test_message_roles(self):
        from synthstats.core.types import Message

        for role in ["system", "user", "assistant", "tool"]:
            msg = Message(role=role, content="test")
            assert msg.role == role


class TestAction:
    def test_final_answer_creation(self):
        from synthstats.core.types import FinalAnswer

        action = FinalAnswer(text="The answer is 42")
        assert action.text == "The answer is 42"

    def test_tool_call_creation(self):
        from synthstats.core.types import ToolCall

        action = ToolCall(name="query", input={"x": 1}, raw="query(x=1)")
        assert action.name == "query"
        assert action.input == {"x": 1}
        assert action.raw == "query(x=1)"

    def test_program_creation(self):
        from synthstats.core.types import Program

        action = Program(code="import pymc as pm", language="pymc")
        assert action.code == "import pymc as pm"
        assert action.language == "pymc"

    def test_program_default_language(self):
        from synthstats.core.types import Program

        action = Program(code="x = 1")
        assert action.language == "pymc"


class TestStepResult:
    def test_step_result_creation(self):
        from synthstats.core.types import StepResult

        result = StepResult(next_state={"step": 1}, done=False, artifacts={})
        assert result.next_state == {"step": 1}
        assert result.done is False
        assert result.artifacts == {}

    def test_step_result_done(self):
        from synthstats.core.types import StepResult

        result = StepResult(next_state=None, done=True, artifacts={"program": "x=1"})
        assert result.done is True
        assert "program" in result.artifacts


class TestReward:
    def test_reward_creation(self):
        from synthstats.core.types import Reward

        reward = Reward(
            total=0.95,
            components={"likelihood": 0.9, "formatting": 0.05},
            info={"elpd": -10.5},
            factors={"likelihood": 0.9, "formatting": 1.0},
            scalarization="weighted_log_product",
        )
        assert reward.total == 0.95
        assert reward.components["likelihood"] == 0.9
        assert reward.info["elpd"] == -10.5
        assert reward.factors["likelihood"] == 0.9
        assert reward.scalarization == "weighted_log_product"

    def test_reward_empty_components(self):
        from synthstats.core.types import Reward

        reward = Reward(total=1.0, components={}, info={})
        assert reward.total == 1.0

    def test_reward_to_dict(self):
        from synthstats.core.types import Reward

        reward = Reward(
            total=0.8,
            components={"a": 0.5},
            info={"k": "v"},
            factors={"a": 0.5},
            scalarization="weighted_sum",
        )
        d = reward.to_dict()
        assert d == {
            "total": 0.8,
            "components": {"a": 0.5},
            "info": {"k": "v"},
            "factors": {"a": 0.5},
            "scalarization": "weighted_sum",
        }

    def test_reward_from_dict(self):
        from synthstats.core.types import Reward

        d = {
            "total": 0.9,
            "components": {"b": 0.3},
            "info": {},
            "factors": {"b": 0.3},
            "scalarization": "weighted_log_product",
        }
        reward = Reward.from_dict(d)
        assert reward.total == 0.9
        assert reward.components == {"b": 0.3}
        assert reward.factors == {"b": 0.3}
        assert reward.scalarization == "weighted_log_product"

    def test_reward_from_dict_backward_compatible(self):
        from synthstats.core.types import Reward

        d = {"total": 0.9, "components": {"b": 0.3}, "info": {}}
        reward = Reward.from_dict(d)

        assert reward.factors == {}
        assert reward.scalarization is None

    def test_reward_roundtrip(self):
        from synthstats.core.types import Reward

        original = Reward(total=0.75, components={"x": 0.5, "y": 0.25}, info={"meta": 123})
        restored = Reward.from_dict(original.to_dict())
        assert restored.total == original.total
        assert restored.components == original.components
        assert restored.info == original.info


class TestTrajectory:
    def test_trajectory_has_required_fields(self):
        from synthstats.core.types import Message, Reward, Trajectory

        traj = Trajectory(
            messages=[Message(role="user", content="test")],
            token_ids=[[1, 2, 3]],
            token_logprobs=[[-0.1, -0.2, -0.3]],
            loss_mask=[[True, True, False]],
            reward=Reward(total=0.5, components={}, info={}),
        )
        assert len(traj.messages) == 1
        assert len(traj.token_ids) == 1
        assert len(traj.token_logprobs) == 1
        assert len(traj.loss_mask) == 1
        assert traj.reward.total == 0.5

    def test_trajectory_multiple_turns(self):
        from synthstats.core.types import Message, Reward, Trajectory

        messages = [
            Message(role="user", content="q1"),
            Message(role="assistant", content="a1"),
            Message(role="user", content="q2"),
            Message(role="assistant", content="a2"),
        ]
        traj = Trajectory(
            messages=messages,
            token_ids=[[1], [2], [3], [4]],
            token_logprobs=[[-0.1], [-0.2], [-0.3], [-0.4]],
            loss_mask=[[True], [True], [True], [True]],
            reward=Reward(total=1.0, components={}, info={}),
        )
        assert len(traj.messages) == 4

    def test_trajectory_to_dict(self):
        from synthstats.core.types import Message, Reward, Trajectory

        traj = Trajectory(
            messages=[Message(role="user", content="hi")],
            token_ids=[[1, 2]],
            token_logprobs=[[-0.1, -0.2]],
            loss_mask=[[True, False]],
            reward=Reward(total=0.5, components={}, info={}),
            eos_logprobs=[[-0.5, -0.6]],
        )
        d = traj.to_dict()
        assert d["token_ids"] == [[1, 2]]
        assert d["eos_logprobs"] == [[-0.5, -0.6]]
        assert len(d["messages"]) == 1
        assert d["messages"][0]["role"] == "user"

    def test_trajectory_from_dict(self):
        from synthstats.core.types import Trajectory

        d = {
            "messages": [{"role": "assistant", "content": "bye", "tool_call_id": None}],
            "token_ids": [[3]],
            "token_logprobs": [[-0.3]],
            "loss_mask": [[True]],
            "reward": {"total": 0.9, "components": {}, "info": {}},
            "eos_logprobs": [[-0.1]],
        }
        traj = Trajectory.from_dict(d)
        assert traj.messages[0].content == "bye"
        assert traj.reward.total == 0.9
        assert traj.eos_logprobs == [[-0.1]]

    def test_trajectory_from_dict_missing_eos_logprobs(self):
        from synthstats.core.types import Trajectory

        d = {
            "messages": [{"role": "user", "content": "test", "tool_call_id": None}],
            "token_ids": [[1]],
            "token_logprobs": [[-0.1]],
            "loss_mask": [[True]],
            "reward": {"total": 1.0, "components": {}, "info": {}},
        }
        traj = Trajectory.from_dict(d)
        assert traj.eos_logprobs == []

    def test_trajectory_roundtrip(self):
        from synthstats.core.types import Message, Reward, Trajectory

        original = Trajectory(
            messages=[
                Message(role="user", content="q"),
                Message(role="assistant", content="a"),
            ],
            token_ids=[[1, 2], [3, 4]],
            token_logprobs=[[-0.1, -0.2], [-0.3, -0.4]],
            loss_mask=[[True, True], [False, True]],
            reward=Reward(total=0.7, components={"c": 0.7}, info={"i": 1}),
            eos_logprobs=[[-0.5], [-0.6]],
        )
        restored = Trajectory.from_dict(original.to_dict())
        assert len(restored.messages) == len(original.messages)
        assert restored.token_ids == original.token_ids
        assert restored.token_logprobs == original.token_logprobs
        assert restored.loss_mask == original.loss_mask
        assert restored.reward.total == original.reward.total
        assert restored.eos_logprobs == original.eos_logprobs
