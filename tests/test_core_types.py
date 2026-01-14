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
        )
        assert reward.total == 0.95
        assert reward.components["likelihood"] == 0.9
        assert reward.info["elpd"] == -10.5

    def test_reward_empty_components(self):
        from synthstats.core.types import Reward

        reward = Reward(total=1.0, components={}, info={})
        assert reward.total == 1.0


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
