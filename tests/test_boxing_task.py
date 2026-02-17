"""Tests for BoxingTask - WRITTEN FIRST per TDD."""

import pytest


class TestDugongsEnv:
    """Test the Dugongs environment stub."""

    def test_dugongs_env_creation(self):
        from synthstats.tasks.boxing.envs.dugongs_env import DugongsEnv

        env = DugongsEnv()
        assert env.alpha == 2.65
        assert env.beta == 0.97
        assert env.lam == -0.87

    def test_dugongs_env_reset(self):
        from synthstats.tasks.boxing.envs.dugongs_env import DugongsEnv

        env = DugongsEnv()
        env.reset(seed=42)
        # should not raise

    def test_dugongs_env_query_valid(self):
        from synthstats.tasks.boxing.envs.dugongs_env import DugongsEnv

        env = DugongsEnv()
        env.reset(seed=42)
        result = env.query("age=5")
        assert "length=" in result
        # parse the length value
        length = float(result.split("length=")[1])
        # true value: 2.65 - 0.97 * 0.87^5 ~ 2.65 - 0.97 * 0.498 ~ 2.167
        assert 1.5 < length < 3.0  # reasonable range with noise

    def test_dugongs_env_query_invalid(self):
        from synthstats.tasks.boxing.envs.dugongs_env import DugongsEnv

        env = DugongsEnv()
        env.reset()
        result = env.query("invalid")
        assert "Invalid" in result

    def test_dugongs_env_deterministic_with_seed(self):
        from synthstats.tasks.boxing.envs.dugongs_env import DugongsEnv

        env1 = DugongsEnv()
        env1.reset(seed=123)
        result1 = env1.query("age=3")

        env2 = DugongsEnv()
        env2.reset(seed=123)
        result2 = env2.query("age=3")

        assert result1 == result2


class TestPeregrinesEnv:
    """Test the Peregrines falcon population environment."""

    def test_peregrines_env_creation(self):
        from synthstats.tasks.boxing.envs.peregrines_env import PeregrinesEnv

        env = PeregrinesEnv()
        assert env.r == 2.0  # default growth rate
        assert env.K == 200.0  # default carrying capacity
        assert env.N_0 == 50.0  # default initial population
        assert env.noise_std == 5.0

    def test_peregrines_env_reset_with_seed(self):
        from synthstats.tasks.boxing.envs.peregrines_env import PeregrinesEnv

        env1 = PeregrinesEnv()
        env2 = PeregrinesEnv()
        env1.reset(seed=42)
        env2.reset(seed=42)

        # parameters should be identical with same seed
        assert env1.r == env2.r
        assert env1.K == env2.K
        assert env1.N_0 == env2.N_0

    def test_peregrines_query_year(self):
        from synthstats.tasks.boxing.envs.peregrines_env import PeregrinesEnv

        env = PeregrinesEnv()
        env.reset(seed=123)
        result = env.query("year=5")
        assert "population=" in result
        pop = int(result.split("population=")[1])
        assert pop >= 0
        assert pop < 1000  # reasonable upper bound for early years

    def test_peregrines_query_invalid(self):
        from synthstats.tasks.boxing.envs.peregrines_env import PeregrinesEnv

        env = PeregrinesEnv()
        env.reset(seed=123)
        result = env.query("invalid query")
        assert "Invalid" in result

    def test_peregrines_query_negative_year(self):
        from synthstats.tasks.boxing.envs.peregrines_env import PeregrinesEnv

        env = PeregrinesEnv()
        env.reset(seed=123)
        result = env.query("year=-5")
        assert "Invalid" in result or "non-negative" in result.lower()

    def test_peregrines_population_growth(self):
        """Test that population follows logistic growth pattern."""
        from synthstats.tasks.boxing.envs.peregrines_env import PeregrinesEnv

        env = PeregrinesEnv()
        env.reset(seed=42)

        # get true population (no noise) at different years
        pop_0 = env._get_population(0)
        pop_5 = env._get_population(5)
        pop_50 = env._get_population(50)

        # initial population should match N_0
        assert pop_0 == env.N_0

        # population should grow initially (starting below K)
        assert pop_5 > pop_0

        # after many generations, population should stabilize near K
        # discrete logistic can overshoot, so just check it's bounded and positive
        assert pop_50 > 0
        assert pop_50 < env.K * 2  # shouldn't explode to infinity

    def test_peregrines_deterministic_with_seed(self):
        from synthstats.tasks.boxing.envs.peregrines_env import PeregrinesEnv

        env1 = PeregrinesEnv()
        env1.reset(seed=999)
        result1 = env1.query("year=10")

        env2 = PeregrinesEnv()
        env2.reset(seed=999)
        result2 = env2.query("year=10")

        assert result1 == result2

    def test_peregrines_logistic_growth_exact(self):
        """Verify exact logistic growth formula: N_t = N_{t-1} + r*N*(1 - N/K)."""
        from synthstats.tasks.boxing.envs.peregrines_env import PeregrinesEnv

        env = PeregrinesEnv()
        # set known fixed values (bypass reset randomization)
        env.r = 2.0
        env.K = 200.0
        env.N_0 = 50.0
        env._population_cache = {0: env.N_0}

        pop_1 = env._get_population(1)
        # N_1 = 50 + 2.0 * 50 * (1 - 50/200) = 50 + 100 * 0.75 = 125.0
        expected = 50 + 2.0 * 50 * (1 - 50 / 200)
        assert abs(pop_1 - expected) < 0.001, f"Expected {expected}, got {pop_1}"

    def test_peregrines_approaches_carrying_capacity(self):
        """Population should stabilize near K after many generations."""
        from synthstats.tasks.boxing.envs.peregrines_env import PeregrinesEnv

        env = PeregrinesEnv()
        env.r = 1.8  # moderate growth rate (stable regime)
        env.K = 200.0
        env.N_0 = 50.0
        env._population_cache = {0: env.N_0}

        pop_50 = env._get_population(50)
        # should be within 10% of carrying capacity
        assert abs(pop_50 - env.K) / env.K < 0.1, f"Expected ~{env.K}, got {pop_50}"


class TestBoxingState:
    """Test the BoxingState dataclass."""

    def test_boxing_state_creation(self):
        from synthstats.tasks.boxing.task import BoxingState

        state = BoxingState(observations=[], step=0)
        assert state.observations == []
        assert state.step == 0
        assert state.done is False

    def test_boxing_state_with_observations(self):
        from synthstats.tasks.boxing.task import BoxingState

        state = BoxingState(
            observations=["Query: age=5\nResult: length=2.1"],
            step=1,
            done=False,
        )
        assert len(state.observations) == 1
        assert state.step == 1

    def test_boxing_state_done(self):
        from synthstats.tasks.boxing.task import BoxingState

        state = BoxingState(observations=[], step=10, done=True)
        assert state.done is True


class TestBoxingTask:
    """Test the BoxingTask implementation."""

    def test_boxing_task_creation(self):
        from synthstats.tasks.boxing.task import BoxingTask

        task = BoxingTask(env_name="dugongs", max_steps=10)
        assert task.name == "boxing"
        assert task.env_name == "dugongs"
        assert task.max_steps == 10

    def test_boxing_task_default_args(self):
        from synthstats.tasks.boxing.task import BoxingTask

        task = BoxingTask()
        assert task.env_name == "dugongs"
        assert task.max_steps == 10

    def test_boxing_task_unknown_env_raises(self):
        from synthstats.tasks.boxing.task import BoxingTask

        with pytest.raises(ValueError, match="Unknown environment"):
            BoxingTask(env_name="unknown_env")

    def test_boxing_task_reset(self):
        from synthstats.tasks.boxing.task import BoxingState, BoxingTask

        task = BoxingTask()
        state = task.reset(seed=42)
        assert isinstance(state, BoxingState)
        assert state.observations == []
        assert state.step == 0
        assert state.done is False

    def test_boxing_task_observe_initial(self):
        from synthstats.tasks.boxing.task import BoxingTask

        task = BoxingTask()
        state = task.reset()
        messages = task.observe(state)
        assert len(messages) == 2
        assert messages[0].role == "system"
        assert "dugongs" in messages[0].content
        assert messages[1].role == "user"
        assert "No observations" in messages[1].content

    def test_boxing_task_observe_with_data(self):
        from synthstats.tasks.boxing.task import BoxingState, BoxingTask

        task = BoxingTask()
        state = BoxingState(
            observations=["Query: age=5\nResult: length=2.1"],
            step=1,
        )
        messages = task.observe(state)
        assert "age=5" in messages[1].content
        assert "length=2.1" in messages[1].content

    def test_boxing_task_step_with_query(self):
        from synthstats.core.types import StepResult, ToolCall
        from synthstats.tasks.boxing.task import BoxingTask

        task = BoxingTask()
        state = task.reset(seed=42)

        action = ToolCall(name="query", input={"query": "age=5"}, raw="query(age=5)")
        result = task.step(state, action)

        assert isinstance(result, StepResult)
        assert result.done is False
        assert result.next_state.step == 1
        assert len(result.next_state.observations) == 1
        assert "age=5" in result.next_state.observations[0]

    def test_boxing_task_step_with_program(self):
        from synthstats.core.types import Program
        from synthstats.tasks.boxing.task import BoxingTask

        task = BoxingTask()
        state = task.reset()

        program_code = """
import pymc as pm
with pm.Model() as model:
    alpha = pm.Normal('alpha', 2.5, 0.5)
"""
        action = Program(code=program_code)
        result = task.step(state, action)

        assert result.done is True
        assert result.next_state.done is True
        assert "program" in result.artifacts
        assert result.artifacts["program"] == program_code

    def test_boxing_task_step_max_steps(self):
        from synthstats.core.types import ToolCall
        from synthstats.tasks.boxing.task import BoxingState, BoxingTask

        task = BoxingTask(max_steps=3)
        state = BoxingState(observations=["obs1", "obs2"], step=2)

        action = ToolCall(name="query", input={"query": "age=1"}, raw="query(age=1)")
        result = task.step(state, action)

        # step 2 -> step 3, which equals max_steps, so done
        assert result.done is True
        assert result.next_state.step == 3

    def test_boxing_task_step_unknown_action(self):
        from synthstats.core.types import FinalAnswer
        from synthstats.tasks.boxing.task import BoxingTask

        task = BoxingTask()
        state = task.reset()

        # FinalAnswer is not expected in boxing task
        action = FinalAnswer(text="some answer")
        result = task.step(state, action)

        # unknown action: state unchanged, not done
        assert result.done is False
        assert result.next_state.step == state.step

    def test_boxing_task_implements_protocol(self):
        from synthstats.core.task import Task
        from synthstats.tasks.boxing.task import BoxingTask

        task = BoxingTask()
        # runtime_checkable protocol
        assert isinstance(task, Task)


class TestBoxingCodec:
    """Test the BoxingCodec for parsing actions."""

    def test_codec_parse_query_action(self):
        from synthstats.tasks.boxing.codecs import BoxingCodec

        codec = BoxingCodec()
        text = '<tool_call>{"name": "query", "input": {"query": "age=5"}}</tool_call>'
        action = codec.parse(text)

        from synthstats.core.types import ToolCall

        assert isinstance(action, ToolCall)
        assert action.name == "query"
        assert action.input["query"] == "age=5"

    def test_codec_parse_submit_program(self):
        from synthstats.tasks.boxing.codecs import BoxingCodec

        codec = BoxingCodec()
        program_code = "import pymc as pm"
        text = f"<submit_program>{program_code}</submit_program>"
        action = codec.parse(text)

        from synthstats.core.types import Program

        assert isinstance(action, Program)
        assert action.code == program_code

    def test_codec_parse_invalid_raises(self):
        from synthstats.runtime.codecs import ParseError
        from synthstats.tasks.boxing.codecs import BoxingCodec

        codec = BoxingCodec()
        text = "just some random text"
        with pytest.raises(ParseError):
            codec.parse(text)

    def test_codec_render_query(self):
        from synthstats.core.types import ToolCall
        from synthstats.tasks.boxing.codecs import BoxingCodec

        codec = BoxingCodec()
        action = ToolCall(name="query", input={"query": "age=5"}, raw="")
        text = codec.render(action)

        assert "query" in text
        assert "age=5" in text

    def test_codec_render_program(self):
        from synthstats.core.types import Program
        from synthstats.tasks.boxing.codecs import BoxingCodec

        codec = BoxingCodec()
        action = Program(code="import pymc as pm")
        text = codec.render(action)

        assert "submit_program" in text
        assert "import pymc as pm" in text


class TestBoxingTaskIntegration:
    """Integration tests for a complete episode."""

    def test_full_episode(self):
        from synthstats.core.types import Program, ToolCall
        from synthstats.tasks.boxing.task import BoxingTask

        task = BoxingTask(max_steps=5)
        state = task.reset(seed=42)

        # step 1: query
        action1 = ToolCall(name="query", input={"query": "age=1"}, raw="query(age=1)")
        result1 = task.step(state, action1)
        assert not result1.done
        state = result1.next_state

        # step 2: another query
        action2 = ToolCall(name="query", input={"query": "age=5"}, raw="query(age=5)")
        result2 = task.step(state, action2)
        assert not result2.done
        state = result2.next_state
        assert len(state.observations) == 2

        # step 3: submit program
        action3 = Program(code="model = ...")
        result3 = task.step(state, action3)
        assert result3.done
        assert "program" in result3.artifacts
