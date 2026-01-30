"""Tests for core protocols - WRITTEN FIRST per TDD.

Tests include both structural checks (protocol shape) and behavioral tests
(actual method invocation and return value verification).
"""


class TestTaskProtocol:
    def test_task_protocol_has_name_attribute(self):
        from synthstats.core.task import Task

        # protocol attributes are in __annotations__
        assert "name" in Task.__annotations__

    def test_task_protocol_has_reset_method(self):
        from synthstats.core.task import Task

        # check that reset is defined
        assert hasattr(Task, "reset")

    def test_task_protocol_has_observe_method(self):
        from synthstats.core.task import Task

        assert hasattr(Task, "observe")

    def test_task_protocol_has_step_method(self):
        from synthstats.core.task import Task

        assert hasattr(Task, "step")

    def test_dummy_task_implements_protocol(self):
        from synthstats.core.types import Action, Message, StepResult

        class DummyTask:
            name = "dummy"

            def reset(self, seed: int | None = None) -> dict:
                return {"step": 0}

            def observe(self, state: dict) -> list[Message]:
                return [Message(role="system", content="test")]

            def step(self, state: dict, action: Action) -> StepResult:
                return StepResult(next_state=state, done=True, artifacts={})

        task = DummyTask()
        assert task.name == "dummy"
        state = task.reset()
        messages = task.observe(state)
        assert len(messages) == 1

    def test_task_reset_returns_state_with_seed(self):
        """Behavioral test: reset should accept seed and return usable state."""
        from synthstats.core.types import Action, Message, StepResult

        class SeededTask:
            name = "seeded"

            def reset(self, seed: int | None = None) -> dict:
                return {"seed": seed, "data": [1, 2, 3]}

            def observe(self, state: dict) -> list[Message]:
                return [Message(role="user", content=f"seed={state['seed']}")]

            def step(self, state: dict, action: Action) -> StepResult:
                return StepResult(next_state=state, done=True, artifacts={})

        task = SeededTask()
        state = task.reset(seed=42)
        assert state["seed"] == 42
        assert "data" in state

    def test_task_observe_returns_messages_list(self):
        """Behavioral test: observe should return properly structured messages."""
        from synthstats.core.types import Action, Message, StepResult

        class MessageTask:
            name = "message_test"

            def reset(self, seed: int | None = None) -> dict:
                return {"turn": 0}

            def observe(self, state: dict) -> list[Message]:
                return [
                    Message(role="system", content="You are a helpful assistant."),
                    Message(role="user", content=f"Turn {state['turn']}"),
                ]

            def step(self, state: dict, action: Action) -> StepResult:
                return StepResult(next_state={"turn": state["turn"] + 1}, done=False, artifacts={})

        task = MessageTask()
        state = task.reset()
        messages = task.observe(state)

        assert len(messages) == 2
        assert messages[0].role == "system"
        assert messages[1].role == "user"
        assert "Turn 0" in messages[1].content

    def test_task_step_advances_state_and_returns_result(self):
        """Behavioral test: step should update state and return StepResult."""
        from synthstats.core.types import FinalAnswer, Message, StepResult

        class CountingTask:
            name = "counting"

            def reset(self, seed: int | None = None) -> dict:
                return {"count": 0}

            def observe(self, state: dict) -> list[Message]:
                return [Message(role="user", content=str(state["count"]))]

            def step(self, state: dict, action) -> StepResult:
                new_count = state["count"] + 1
                done = new_count >= 3
                return StepResult(
                    next_state={"count": new_count},
                    done=done,
                    artifacts={"last_count": new_count},
                )

        task = CountingTask()
        state = task.reset()
        assert state["count"] == 0

        result = task.step(state, FinalAnswer(text=""))
        assert result.next_state["count"] == 1
        assert result.done is False
        assert result.artifacts["last_count"] == 1

        result = task.step(result.next_state, FinalAnswer(text=""))
        result = task.step(result.next_state, FinalAnswer(text=""))
        assert result.done is True


class TestPolicyProtocol:
    def test_policy_protocol_has_generate_method(self):
        from synthstats.core.policy import Policy

        assert hasattr(Policy, "generate")

    def test_policy_protocol_has_logprobs_method(self):
        from synthstats.core.policy import Policy

        assert hasattr(Policy, "logprobs")

    def test_policy_generate_returns_generation(self):
        """Behavioral test: generate should return Generation with all fields."""
        from synthstats.core.policy import GenConfig, Generation
        from synthstats.core.types import Message

        class DummyPolicy:
            def generate(self, messages: list[Message], *, gen: GenConfig) -> Generation:
                return Generation(
                    text="Hello, world!",
                    token_ids=[1, 2, 3, 4],
                    token_logprobs=[-0.1, -0.2, -0.3, -0.4],
                    finish_reason="stop",
                )

            def logprobs(self, messages: list[Message], tokens: list[int]):
                from synthstats.core.policy import TokenLogProbs

                return TokenLogProbs(token_ids=tokens, logprobs=[-0.1] * len(tokens))

        policy = DummyPolicy()
        messages = [Message(role="user", content="Hi")]
        gen = policy.generate(messages, gen=GenConfig())

        assert gen.text == "Hello, world!"
        assert len(gen.token_ids) == 4
        assert len(gen.token_logprobs) == 4
        assert gen.finish_reason == "stop"

    def test_policy_logprobs_returns_token_logprobs(self):
        """Behavioral test: logprobs should return per-token log probabilities."""
        from synthstats.core.policy import GenConfig, Generation, TokenLogProbs
        from synthstats.core.types import Message

        class DummyPolicy:
            def generate(self, messages: list[Message], *, gen: GenConfig) -> Generation:
                return Generation(
                    text="test", token_ids=[1], token_logprobs=[-0.1], finish_reason="stop"
                )

            def logprobs(self, messages: list[Message], tokens: list[int]) -> TokenLogProbs:
                return TokenLogProbs(
                    token_ids=tokens, logprobs=[-0.5 * i for i in range(len(tokens))]
                )

        policy = DummyPolicy()
        messages = [Message(role="user", content="test")]
        result = policy.logprobs(messages, [10, 20, 30])

        assert result.token_ids == [10, 20, 30]
        assert len(result.logprobs) == 3
        assert result.logprobs[0] == 0.0
        assert result.logprobs[1] == -0.5


class TestExecutorProtocol:
    def test_executor_protocol_has_name_attribute(self):
        from synthstats.core.executor import Executor

        # protocol attributes are in __annotations__
        assert "name" in Executor.__annotations__

    def test_executor_protocol_has_execute_method(self):
        from synthstats.core.executor import Executor

        assert hasattr(Executor, "execute")

    def test_tool_result_creation(self):
        from synthstats.core.executor import ToolResult

        result = ToolResult(output="success", success=True, error=None)
        assert result.output == "success"
        assert result.success is True
        assert result.error is None

    def test_tool_result_with_error(self):
        from synthstats.core.executor import ToolResult

        result = ToolResult(output="", success=False, error="timeout")
        assert result.success is False
        assert result.error == "timeout"

    def test_executor_execute_returns_tool_result(self):
        """Behavioral test: execute should run tool and return ToolResult."""
        from synthstats.core.executor import ToolResult
        from synthstats.core.types import ToolCall

        class CalculatorExecutor:
            name = "calculator"

            def execute(self, payload: ToolCall, *, timeout_s: float) -> ToolResult:
                try:
                    expr = payload.input.get("expression", "0")
                    # safe eval for simple math
                    result = eval(expr, {"__builtins__": {}}, {})
                    return ToolResult(output=str(result), success=True, error=None)
                except Exception as e:
                    return ToolResult(output="", success=False, error=str(e))

        executor = CalculatorExecutor()
        assert executor.name == "calculator"

        call = ToolCall(name="calculator", input={"expression": "2 + 3"}, raw="")
        result = executor.execute(call, timeout_s=5.0)

        assert result.success is True
        assert result.output == "5"
        assert result.error is None

    def test_executor_handles_errors_gracefully(self):
        """Behavioral test: executor should handle errors without crashing."""
        from synthstats.core.executor import ToolResult
        from synthstats.core.types import ToolCall

        class FailingExecutor:
            name = "failing"

            def execute(self, payload: ToolCall, *, timeout_s: float) -> ToolResult:
                return ToolResult(output="", success=False, error="Intentional failure for testing")

        executor = FailingExecutor()
        call = ToolCall(name="test", input={}, raw="")
        result = executor.execute(call, timeout_s=1.0)

        assert result.success is False
        assert "failure" in result.error.lower()


class TestJudgeProtocol:
    def test_judge_protocol_has_score_method(self):
        from synthstats.core.judge import Judge

        assert hasattr(Judge, "score")

    def test_dummy_judge_implements_protocol(self):
        from synthstats.core.types import Message, Reward, Trajectory

        class DummyJudge:
            def score(self, *, task_name: str, trajectory: Trajectory, artifacts: dict) -> Reward:
                return Reward(total=1.0, components={}, info={})

        judge = DummyJudge()
        traj = Trajectory(
            messages=[Message(role="user", content="test")],
            token_ids=[],
            token_logprobs=[],
            loss_mask=[],
            reward=Reward(total=0.0, components={}, info={}),
        )
        reward = judge.score(task_name="test", trajectory=traj, artifacts={})
        assert reward.total == 1.0

    def test_judge_score_uses_trajectory_content(self):
        """Behavioral test: judge should analyze trajectory for scoring."""
        from synthstats.core.types import Message, Reward, Trajectory

        class LengthJudge:
            """Scores based on total message length."""

            def score(self, *, task_name: str, trajectory: Trajectory, artifacts: dict) -> Reward:
                total_chars = sum(len(m.content) for m in trajectory.messages)
                normalized = min(total_chars / 100.0, 1.0)
                return Reward(
                    total=normalized,
                    components={"length_score": normalized},
                    info={"total_chars": total_chars},
                )

        judge = LengthJudge()

        short_traj = Trajectory(
            messages=[Message(role="user", content="hi")],
            token_ids=[],
            token_logprobs=[],
            loss_mask=[],
            reward=Reward(total=0.0, components={}, info={}),
        )
        long_traj = Trajectory(
            messages=[Message(role="user", content="x" * 200)],
            token_ids=[],
            token_logprobs=[],
            loss_mask=[],
            reward=Reward(total=0.0, components={}, info={}),
        )

        short_reward = judge.score(task_name="test", trajectory=short_traj, artifacts={})
        long_reward = judge.score(task_name="test", trajectory=long_traj, artifacts={})

        assert short_reward.total < long_reward.total
        assert short_reward.info["total_chars"] == 2
        assert long_reward.total == 1.0  # capped at 1.0

    def test_judge_score_uses_artifacts(self):
        """Behavioral test: judge should use artifacts for scoring."""
        from synthstats.core.types import Message, Reward, Trajectory

        class ArtifactJudge:
            """Scores based on artifacts."""

            def score(self, *, task_name: str, trajectory: Trajectory, artifacts: dict) -> Reward:
                score = artifacts.get("quality_score", 0.0)
                return Reward(
                    total=score,
                    components={"quality": score},
                    info={"from_artifacts": True},
                )

        judge = ArtifactJudge()
        traj = Trajectory(
            messages=[Message(role="user", content="test")],
            token_ids=[],
            token_logprobs=[],
            loss_mask=[],
            reward=Reward(total=0.0, components={}, info={}),
        )

        reward_low = judge.score(task_name="t", trajectory=traj, artifacts={"quality_score": 0.3})
        reward_high = judge.score(task_name="t", trajectory=traj, artifacts={"quality_score": 0.9})

        assert reward_low.total == 0.3
        assert reward_high.total == 0.9
        assert reward_high.info["from_artifacts"] is True
