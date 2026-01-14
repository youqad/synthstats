"""Tests for SkyRL text environment wrapper - WRITTEN FIRST per TDD."""


from synthstats.core.types import FinalAnswer, Message, StepResult


class MockTask:
    """Mock task for testing."""

    name = "mock_task"

    def __init__(self):
        self.step_count = 0

    def reset(self, seed=None):
        self.step_count = 0
        return {"step": 0}

    def observe(self, state):
        return [
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content=f"Step {state['step']}. What is 2+2?"),
        ]

    def step(self, state, action):
        self.step_count += 1
        new_step = state["step"] + 1
        done = new_step >= 3 or isinstance(action, FinalAnswer)
        return StepResult(
            next_state={"step": new_step},
            done=done,
            artifacts={"last_action": action},
        )


class MockCodec:
    """Mock codec for testing."""

    def parse(self, text):
        if "answer:" in text.lower():
            return FinalAnswer(text=text.split("answer:")[-1].strip())
        return FinalAnswer(text=text)

    def render(self, action):
        if isinstance(action, FinalAnswer):
            return f"answer: {action.text}"
        return str(action)


class TestSynthStatsTextEnvImport:
    """Verify text env is importable."""

    def test_import_text_env_class(self):
        """SynthStatsTextEnv should be importable."""
        from synthstats.envs.skyrl_text_env import SynthStatsTextEnv

        assert SynthStatsTextEnv is not None


class TestSynthStatsTextEnvInit:
    """Verify init method works correctly."""

    def test_init_returns_chat_history_and_info(self):
        """init() should return (chat_history, info)."""
        from synthstats.envs.skyrl_text_env import SynthStatsTextEnv

        task = MockTask()
        codec = MockCodec()
        env = SynthStatsTextEnv(task=task, codec=codec)

        chat_history, info = env.init()

        assert isinstance(chat_history, list)
        assert isinstance(info, dict)
        assert len(chat_history) >= 1

    def test_init_chat_history_has_correct_format(self):
        """Chat history should be list of dicts with role/content."""
        from synthstats.envs.skyrl_text_env import SynthStatsTextEnv

        task = MockTask()
        codec = MockCodec()
        env = SynthStatsTextEnv(task=task, codec=codec)

        chat_history, _ = env.init()

        for msg in chat_history:
            assert "role" in msg
            assert "content" in msg
            assert msg["role"] in ["system", "user", "assistant", "tool"]

    def test_init_with_custom_prompt(self):
        """init() should accept custom prompt."""
        from synthstats.envs.skyrl_text_env import SynthStatsTextEnv

        task = MockTask()
        codec = MockCodec()
        env = SynthStatsTextEnv(task=task, codec=codec)

        custom_prompt = [{"role": "system", "content": "Custom system prompt."}]
        chat_history, _ = env.init(prompt=custom_prompt)

        assert chat_history[0]["content"] == "Custom system prompt."


class TestSynthStatsTextEnvStep:
    """Verify step method works correctly."""

    def test_step_returns_step_output_format(self):
        """step() should return dict with observations, reward, done, metadata."""
        from synthstats.envs.skyrl_text_env import SynthStatsTextEnv

        task = MockTask()
        codec = MockCodec()
        env = SynthStatsTextEnv(task=task, codec=codec)
        env.init()

        result = env.step("answer: 4")

        assert "observations" in result
        assert "reward" in result
        assert "done" in result
        assert "metadata" in result

    def test_step_observations_is_list(self):
        """observations should be a list."""
        from synthstats.envs.skyrl_text_env import SynthStatsTextEnv

        task = MockTask()
        codec = MockCodec()
        env = SynthStatsTextEnv(task=task, codec=codec)
        env.init()

        result = env.step("answer: 4")

        assert isinstance(result["observations"], list)

    def test_step_reward_is_float(self):
        """reward should be a float."""
        from synthstats.envs.skyrl_text_env import SynthStatsTextEnv

        task = MockTask()
        codec = MockCodec()
        env = SynthStatsTextEnv(task=task, codec=codec)
        env.init()

        result = env.step("answer: 4")

        assert isinstance(result["reward"], float)

    def test_step_done_is_bool(self):
        """done should be a bool."""
        from synthstats.envs.skyrl_text_env import SynthStatsTextEnv

        task = MockTask()
        codec = MockCodec()
        env = SynthStatsTextEnv(task=task, codec=codec)
        env.init()

        result = env.step("answer: 4")

        assert isinstance(result["done"], bool)

    def test_step_updates_chat_history(self):
        """step() should update chat_history with assistant message."""
        from synthstats.envs.skyrl_text_env import SynthStatsTextEnv

        task = MockTask()
        codec = MockCodec()
        env = SynthStatsTextEnv(task=task, codec=codec)
        env.init()

        initial_len = len(env.chat_history)
        env.step("answer: 4")

        assert len(env.chat_history) > initial_len

    def test_multiple_steps_until_done(self):
        """Multiple steps should eventually reach done=True."""
        from synthstats.envs.skyrl_text_env import SynthStatsTextEnv

        task = MockTask()
        codec = MockCodec()
        env = SynthStatsTextEnv(task=task, codec=codec)
        env.init()

        max_steps = 10
        done = False
        for i in range(max_steps):
            result = env.step(f"answer: {i}")
            done = result["done"]
            if done:
                break

        assert done, "Episode should terminate within max_steps"


class TestSynthStatsTextEnvWithExecutors:
    """Verify executor integration."""

    def test_env_accepts_executors(self):
        """Environment should accept executors dict."""
        from synthstats.envs.skyrl_text_env import SynthStatsTextEnv

        task = MockTask()
        codec = MockCodec()

        # empty executors should work
        env = SynthStatsTextEnv(task=task, codec=codec, executors={})
        assert env.executors == {}


class TestSynthStatsTextEnvClose:
    """Verify close method exists and works."""

    def test_close_exists(self):
        """close() method should exist."""
        from synthstats.envs.skyrl_text_env import SynthStatsTextEnv

        task = MockTask()
        codec = MockCodec()
        env = SynthStatsTextEnv(task=task, codec=codec)

        assert hasattr(env, "close")
        assert callable(env.close)

    def test_close_does_not_raise(self):
        """close() should not raise."""
        from synthstats.envs.skyrl_text_env import SynthStatsTextEnv

        task = MockTask()
        codec = MockCodec()
        env = SynthStatsTextEnv(task=task, codec=codec)
        env.init()

        # should not raise
        env.close()


class TestSynthStatsTextEnvGetMetrics:
    """Verify get_metrics method."""

    def test_get_metrics_returns_dict(self):
        """get_metrics() should return a dict."""
        from synthstats.envs.skyrl_text_env import SynthStatsTextEnv

        task = MockTask()
        codec = MockCodec()
        env = SynthStatsTextEnv(task=task, codec=codec)

        metrics = env.get_metrics()

        assert isinstance(metrics, dict)
