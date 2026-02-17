"""Checkpoint/resume functionality tests."""

import random
from pathlib import Path

import numpy as np
import pytest
import torch


class TestCheckpointState:
    """Test CheckpointState dataclass and serialization."""

    def test_checkpoint_state_creation(self):
        """Stores all required fields."""
        from synthstats.train.checkpointing.base import CheckpointState

        state = CheckpointState(
            step_count=100,
            logZ=1.5,
            model_state_dict={"weight": torch.randn(10, 10)},
            optimizer_state_dict={"state": {}},
            rng_states={"torch": torch.get_rng_state()},
            replay_buffer={"entries": []},
            config={"batch_size": 4},
            metrics_history=[{"loss": 0.5}],
        )

        assert state.step_count == 100
        assert state.logZ == 1.5
        assert state.config["batch_size"] == 4

    def test_checkpoint_state_to_dict(self):
        """Returns serializable dict."""
        from synthstats.train.checkpointing.base import CheckpointState

        state = CheckpointState(
            step_count=50,
            logZ=2.0,
            model_state_dict=None,
            optimizer_state_dict=None,
            rng_states={},
            replay_buffer=None,
            config={},
            metrics_history=[],
        )

        d = state.to_dict()
        assert isinstance(d, dict)
        assert d["step_count"] == 50
        assert d["logZ"] == 2.0

    def test_checkpoint_state_to_dict_is_shallow(self):
        """to_dict() must not deepcopy tensors (can OOM on large models)."""
        from synthstats.train.checkpointing.base import CheckpointState

        weight = torch.randn(2, 2)
        model_state_dict = {"weight": weight}

        state = CheckpointState(
            step_count=1,
            logZ=0.0,
            model_state_dict=model_state_dict,
            optimizer_state_dict=None,
            rng_states={},
            replay_buffer=None,
            config={},
            metrics_history=[],
        )

        d = state.to_dict()
        assert d["model_state_dict"] is model_state_dict
        assert d["model_state_dict"]["weight"] is weight

    def test_checkpoint_state_from_dict(self):
        """Reconstructs state from dict."""
        from synthstats.train.checkpointing.base import CheckpointState

        original = CheckpointState(
            step_count=75,
            logZ=-1.0,
            model_state_dict={"layer": torch.zeros(5)},
            optimizer_state_dict={"lr": 0.001},
            rng_states={"random": random.getstate()},
            replay_buffer={"size": 100},
            config={"device": "cpu"},
            metrics_history=[{"step": 1}, {"step": 2}],
        )

        d = original.to_dict()
        restored = CheckpointState.from_dict(d)

        assert restored.step_count == 75
        assert restored.logZ == -1.0
        assert len(restored.metrics_history) == 2


class TestRngStateFunctions:
    """Test RNG state capture and restoration."""

    def test_get_rng_states(self):
        """Captures all RNG sources."""
        from synthstats.train.utils.seeding import get_rng_states

        states = get_rng_states()

        assert "torch" in states
        assert "numpy" in states
        assert "random" in states

    def test_set_rng_states_restores_torch(self):
        """Restores torch RNG."""
        from synthstats.train.utils.seeding import get_rng_states, set_rng_states

        torch.manual_seed(42)
        states = get_rng_states()

        # generate some random numbers
        _ = torch.randn(10)

        # restore and generate again
        set_rng_states(states)
        after_restore = torch.randn(10)

        # generate fresh with same seed
        torch.manual_seed(42)
        expected = torch.randn(10)

        assert torch.allclose(after_restore, expected)

    def test_set_rng_states_restores_numpy(self):
        """Restores numpy RNG."""
        from synthstats.train.utils.seeding import get_rng_states, set_rng_states

        np.random.seed(42)
        states = get_rng_states()

        # generate some random numbers
        _ = np.random.randn(10)

        # restore and generate again
        set_rng_states(states)
        after_restore = np.random.randn(10)

        # generate fresh with same seed
        np.random.seed(42)
        expected = np.random.randn(10)

        assert np.allclose(after_restore, expected)

    def test_set_rng_states_restores_random(self):
        """Restores Python random module."""
        from synthstats.train.utils.seeding import get_rng_states, set_rng_states

        random.seed(42)
        states = get_rng_states()

        # generate some random numbers
        _ = [random.random() for _ in range(10)]

        # restore and generate again
        set_rng_states(states)
        after_restore = [random.random() for _ in range(10)]

        # generate fresh with same seed
        random.seed(42)
        expected = [random.random() for _ in range(10)]

        assert after_restore == expected


class TestSaveLoadCheckpoint:
    """Test save_checkpoint and load_checkpoint functions."""

    def test_save_checkpoint_creates_file(self, tmp_path: Path):
        """Creates checkpoint file."""
        from synthstats.train.checkpointing.base import CheckpointState, save_checkpoint

        path = tmp_path / "test_ckpt.pt"
        state = CheckpointState(
            step_count=10,
            logZ=0.0,
            model_state_dict=None,
            optimizer_state_dict=None,
            rng_states={},
            replay_buffer=None,
            config={},
            metrics_history=[],
        )

        save_checkpoint(path, state)

        assert path.exists()

    def test_save_checkpoint_creates_parent_dirs(self, tmp_path: Path):
        """Creates parent directories."""
        from synthstats.train.checkpointing.base import CheckpointState, save_checkpoint

        path = tmp_path / "nested" / "dir" / "checkpoint.pt"
        state = CheckpointState(
            step_count=1,
            logZ=0.0,
            model_state_dict=None,
            optimizer_state_dict=None,
            rng_states={},
            replay_buffer=None,
            config={},
            metrics_history=[],
        )

        save_checkpoint(path, state)

        assert path.exists()

    def test_load_checkpoint_raises_for_missing_file(self, tmp_path: Path):
        """Raises FileNotFoundError for missing file."""
        from synthstats.train.checkpointing.base import load_checkpoint

        with pytest.raises(FileNotFoundError):
            load_checkpoint(tmp_path / "nonexistent.pt")

    def test_checkpoint_roundtrip(self, tmp_path: Path):
        """Roundtrip preserves state."""
        from synthstats.train.checkpointing.base import (
            CheckpointState,
            load_checkpoint,
            save_checkpoint,
        )
        from synthstats.train.utils.seeding import get_rng_states

        path = tmp_path / "roundtrip.pt"

        original = CheckpointState(
            step_count=42,
            logZ=3.14,
            model_state_dict={"w": torch.randn(5, 5)},
            optimizer_state_dict={"param_groups": []},
            rng_states=get_rng_states(),
            replay_buffer={"entries": [{"a": 1}]},
            config={"lr": 0.001, "batch_size": 8},
            metrics_history=[{"loss": 1.0}, {"loss": 0.5}],
        )

        save_checkpoint(path, original)
        loaded = load_checkpoint(path)

        assert loaded.step_count == 42
        assert loaded.logZ == 3.14
        assert loaded.config["batch_size"] == 8
        assert len(loaded.metrics_history) == 2
        assert torch.allclose(loaded.model_state_dict["w"], original.model_state_dict["w"])


class TestCleanupOldCheckpoints:
    """Test checkpoint cleanup functionality."""

    def test_cleanup_keeps_recent_checkpoints(self, tmp_path: Path):
        """Keeps the most recent N checkpoints."""
        from synthstats.train.checkpointing.base import (
            CheckpointState,
            cleanup_old_checkpoints,
            save_checkpoint,
        )

        # create 5 checkpoints with different modification times
        for i in range(5):
            path = tmp_path / f"checkpoint_{i:06d}.pt"
            state = CheckpointState(
                step_count=i,
                logZ=0.0,
                model_state_dict=None,
                optimizer_state_dict=None,
                rng_states={},
                replay_buffer=None,
                config={},
                metrics_history=[],
            )
            save_checkpoint(path, state)

        # cleanup, keeping last 2
        removed = cleanup_old_checkpoints(tmp_path, keep_last_n=2)

        # should have removed 3 checkpoints
        assert len(removed) == 3

        # should have 2 remaining
        remaining = list(tmp_path.glob("checkpoint_*.pt"))
        assert len(remaining) == 2

    def test_cleanup_does_nothing_when_under_limit(self, tmp_path: Path):
        """No-op when count <= keep_last_n."""
        from synthstats.train.checkpointing.base import (
            CheckpointState,
            cleanup_old_checkpoints,
            save_checkpoint,
        )

        # create 2 checkpoints
        for i in range(2):
            path = tmp_path / f"checkpoint_{i:06d}.pt"
            state = CheckpointState(
                step_count=i,
                logZ=0.0,
                model_state_dict=None,
                optimizer_state_dict=None,
                rng_states={},
                replay_buffer=None,
                config={},
                metrics_history=[],
            )
            save_checkpoint(path, state)

        # cleanup with keep_last_n=5 (more than we have)
        removed = cleanup_old_checkpoints(tmp_path, keep_last_n=5)

        assert len(removed) == 0
        assert len(list(tmp_path.glob("checkpoint_*.pt"))) == 2


class TestBufferEntryStateDict:
    """Test BufferEntry serialization."""

    def test_buffer_entry_to_dict(self):
        """Returns dict with all fields."""
        from synthstats.train.loop.replay import BufferEntry

        entry = BufferEntry(
            actions=[{"type": "query"}, {"type": "submit"}],
            log_reward=-1.5,
            observations=["obs1", "obs2"],
            policy_version=3,
            temperature=0.8,
        )

        d = entry.to_dict()

        assert d["actions"] == [{"type": "query"}, {"type": "submit"}]
        assert d["log_reward"] == -1.5
        assert d["observations"] == ["obs1", "obs2"]
        assert d["policy_version"] == 3
        assert d["temperature"] == 0.8
        assert "timestamp" in d

    def test_buffer_entry_from_dict(self):
        """Reconstructs entry from dict."""
        from synthstats.train.loop.replay import BufferEntry

        d = {
            "actions": [{"type": "query"}],
            "log_reward": 0.0,
            "observations": ["obs"],
            "policy_version": 5,
            "temperature": 1.0,
            "timestamp": 1234567890.0,
        }

        entry = BufferEntry.from_dict(d)

        assert entry.actions == [{"type": "query"}]
        assert entry.policy_version == 5
        assert entry.timestamp == 1234567890.0

    def test_buffer_entry_roundtrip(self):
        """Roundtrip preserves all data."""
        from synthstats.train.loop.replay import BufferEntry

        original = BufferEntry(
            actions=[{"a": 1}, {"b": 2}],
            log_reward=-2.5,
            observations=["x", "y", "z"],
            policy_version=10,
            temperature=0.5,
        )

        d = original.to_dict()
        restored = BufferEntry.from_dict(d)

        assert restored.actions == original.actions
        assert restored.log_reward == original.log_reward
        assert restored.observations == original.observations
        assert restored.policy_version == original.policy_version
        assert restored.temperature == original.temperature


class TestGFNReplayBufferStateDict:
    """Test GFNReplayBuffer state_dict/load_state_dict."""

    def test_buffer_state_dict(self):
        """Returns complete state."""
        from synthstats.train.loop.replay import BufferEntry, GFNReplayBuffer

        buffer = GFNReplayBuffer(capacity=10, prioritized=True, alpha=0.5)
        buffer.add(BufferEntry(actions=[{}], log_reward=0.0, observations=["obs"]))
        buffer.increment_policy_version()

        state = buffer.state_dict()

        assert "entries" in state
        assert len(state["entries"]) == 1
        assert state["policy_version"] == 1
        assert state["capacity"] == 10
        assert state["prioritized"] is True
        assert state["alpha"] == 0.5

    def test_buffer_load_state_dict(self):
        """Restores state from dict."""
        from synthstats.train.loop.replay import BufferEntry, GFNReplayBuffer

        # create and populate original buffer
        buffer1 = GFNReplayBuffer(capacity=10, prioritized=True, alpha=0.8)
        buffer1.add(BufferEntry(actions=[{"a": 1}], log_reward=-1.0, observations=["o1"]))
        buffer1.add(BufferEntry(actions=[{"b": 2}], log_reward=-0.5, observations=["o2"]))
        buffer1.increment_policy_version()
        buffer1.increment_policy_version()

        state = buffer1.state_dict()

        # create new buffer and restore
        buffer2 = GFNReplayBuffer(capacity=5)  # different initial capacity
        buffer2.load_state_dict(state)

        assert len(buffer2) == 2
        assert buffer2._policy_version == 2
        assert buffer2._capacity == 10
        assert buffer2._prioritized is True
        assert buffer2._alpha == 0.8

    def test_buffer_state_dict_roundtrip(self):
        """Roundtrip preserves buffer contents."""
        from synthstats.train.loop.replay import BufferEntry, GFNReplayBuffer

        buffer1 = GFNReplayBuffer(capacity=5, prioritized=False)

        for i in range(3):
            buffer1.add(
                BufferEntry(
                    actions=[{"id": i}],
                    log_reward=float(i),
                    observations=[f"obs{i}"],
                    policy_version=i,
                )
            )

        state = buffer1.state_dict()
        buffer2 = GFNReplayBuffer(capacity=100)
        buffer2.load_state_dict(state)

        entries1 = list(buffer1)
        entries2 = list(buffer2)

        assert len(entries2) == len(entries1)
        for e1, e2 in zip(entries1, entries2, strict=True):
            assert e1.actions == e2.actions
            assert e1.log_reward == e2.log_reward
            assert e1.observations == e2.observations


class TestSkyRLSubTBTrainerStateDict:
    """Test SkyRLSubTBTrainer state_dict/load_state_dict."""

    def test_trainer_state_dict(self):
        """Returns logZ and config."""
        from synthstats.train.runners.skyrl_subtb import SkyRLSubTBConfig, SkyRLSubTBTrainer

        config = SkyRLSubTBConfig(logZ_init=2.5, beta=0.5)
        trainer = SkyRLSubTBTrainer(config=config, device="cpu")

        state = trainer.state_dict()

        assert "logZ" in state
        assert state["logZ"] == 2.5
        assert "config" in state
        assert state["config"]["beta"] == 0.5

    def test_trainer_load_state_dict(self):
        """Restores logZ."""
        from synthstats.train.runners.skyrl_subtb import SkyRLSubTBTrainer

        trainer = SkyRLSubTBTrainer(device="cpu")
        assert trainer.logZ.item() == 0.0  # default

        trainer.load_state_dict({"logZ": 5.5, "config": {}})

        assert trainer.logZ.item() == 5.5

    def test_trainer_state_dict_roundtrip(self):
        """Roundtrip preserves logZ."""
        from synthstats.train.runners.skyrl_subtb import SkyRLSubTBConfig, SkyRLSubTBTrainer

        config = SkyRLSubTBConfig(logZ_init=-1.5)
        trainer1 = SkyRLSubTBTrainer(config=config, device="cpu")

        # modify logZ
        with torch.no_grad():
            trainer1.logZ.fill_(3.14)

        state = trainer1.state_dict()

        trainer2 = SkyRLSubTBTrainer(device="cpu")
        trainer2.load_state_dict(state)

        assert abs(trainer2.logZ.item() - 3.14) < 1e-5


class TestTrainingLoopCheckpointing:
    """Test TrainingLoop checkpoint integration."""

    @pytest.fixture
    def mock_policy(self):
        """Create mock policy for testing."""
        from synthstats.policies.hf_policy import MockHFPolicy

        return MockHFPolicy(device="cpu")

    @pytest.fixture
    def mock_trainer(self):
        """Create trainer for testing."""
        from synthstats.train.runners.skyrl_subtb import SkyRLSubTBTrainer

        trainer = SkyRLSubTBTrainer(device="cpu")
        trainer.optimizer = torch.optim.Adam([trainer.logZ], lr=0.01)
        return trainer

    @pytest.fixture
    def mock_env(self):
        """Create mock environment for testing."""
        from unittest.mock import MagicMock

        from synthstats.envs.skyrl_text_env import SynthStatsTextEnv

        # create minimal mock
        env = MagicMock(spec=SynthStatsTextEnv)
        env.task = MagicMock()
        env.codec = MagicMock()
        env.executors = {}
        return env

    def test_training_loop_save_checkpoint(self, tmp_path, mock_policy, mock_trainer, mock_env):
        """Creates checkpoint file."""
        from synthstats.train.runners.skyrl_loop import TrainingConfig, TrainingLoop

        config = TrainingConfig(batch_size=2, device="cpu")
        loop = TrainingLoop(config=config)
        loop.setup(policy=mock_policy, trainer=mock_trainer, env=mock_env)

        path = tmp_path / "checkpoint.pt"
        result = loop.save_checkpoint(path)

        assert result == path
        assert path.exists()

    def test_training_loop_load_checkpoint(self, tmp_path, mock_policy, mock_trainer, mock_env):
        """Restores state from checkpoint."""
        from synthstats.train.runners.skyrl_loop import TrainingConfig, TrainingLoop

        config = TrainingConfig(batch_size=2, device="cpu")
        loop = TrainingLoop(config=config)
        loop.setup(policy=mock_policy, trainer=mock_trainer, env=mock_env)

        # modify state
        loop.step_count = 50
        loop._all_metrics = [{"loss": 1.0}, {"loss": 0.5}]
        with torch.no_grad():
            mock_trainer.logZ.fill_(7.5)

        # save and create new loop
        path = tmp_path / "checkpoint.pt"
        loop.save_checkpoint(path)

        # create fresh loop and load
        config2 = TrainingConfig(batch_size=2, device="cpu")
        loop2 = TrainingLoop(config=config2)
        loop2.setup(policy=mock_policy, trainer=mock_trainer, env=mock_env)

        # reset trainer logZ to verify it gets restored
        with torch.no_grad():
            mock_trainer.logZ.fill_(0.0)

        loop2.load_checkpoint(path)

        assert loop2.step_count == 50
        assert len(loop2._all_metrics) == 2
        assert abs(mock_trainer.logZ.item() - 7.5) < 1e-5

    def test_training_loop_from_checkpoint(self, tmp_path, mock_policy, mock_trainer, mock_env):
        """Creates configured loop from checkpoint."""
        from synthstats.train.runners.skyrl_loop import TrainingConfig, TrainingLoop

        config = TrainingConfig(batch_size=4, learning_rate=1e-3, device="cpu")
        loop1 = TrainingLoop(config=config)
        loop1.setup(policy=mock_policy, trainer=mock_trainer, env=mock_env)
        loop1.step_count = 25

        path = tmp_path / "checkpoint.pt"
        loop1.save_checkpoint(path)

        # create from checkpoint
        loop2 = TrainingLoop.from_checkpoint(
            path=path,
            policy=mock_policy,
            trainer=mock_trainer,
            env=mock_env,
        )

        assert loop2.step_count == 25
        assert loop2.config.batch_size == 4

    def test_training_loop_checkpoint_with_gfn_buffer(
        self, tmp_path, mock_policy, mock_trainer, mock_env
    ):
        """Includes GFN replay buffer."""
        from synthstats.train.loop.replay import BufferEntry
        from synthstats.train.runners.skyrl_loop import TrainingConfig, TrainingLoop

        config = TrainingConfig(
            batch_size=2,
            device="cpu",
            replay_buffer_size=100,
            use_gfn_replay=True,
        )
        loop = TrainingLoop(config=config)
        loop.setup(policy=mock_policy, trainer=mock_trainer, env=mock_env)

        # add entries to buffer
        loop._gfn_replay_buffer.add(
            BufferEntry(actions=[{"test": 1}], log_reward=0.5, observations=["obs"])
        )

        path = tmp_path / "checkpoint.pt"
        loop.save_checkpoint(path)

        # create new loop and load
        config2 = TrainingConfig(
            batch_size=2, device="cpu", replay_buffer_size=100, use_gfn_replay=True
        )
        loop2 = TrainingLoop(config=config2)
        loop2.setup(policy=mock_policy, trainer=mock_trainer, env=mock_env)
        loop2.load_checkpoint(path)

        assert len(loop2._gfn_replay_buffer) == 1

    def test_training_loop_requires_setup_before_save(self):
        """Raises RuntimeError if setup() not called."""
        from synthstats.train.runners.skyrl_loop import TrainingLoop

        loop = TrainingLoop()

        with pytest.raises(RuntimeError, match="setup"):
            loop.save_checkpoint("/tmp/test.pt")

    def test_training_loop_requires_setup_before_load(self, tmp_path):
        """Requires setup() first."""
        from synthstats.train.runners.skyrl_loop import TrainingLoop

        loop = TrainingLoop()

        with pytest.raises(RuntimeError, match="setup"):
            loop.load_checkpoint(tmp_path / "test.pt")
