import pytest
import torch

from synthstats.train.checkpointing.base import CheckpointState
from synthstats.train.checkpointing.torch_full import FullStateCheckpoint


class _MockLearner:
    def __init__(self, logz: float = 0.0):
        self.logZ = logz
        self.optimizer = None
        self._state = {"objective": {"logZ": logz}}

    def state_dict(self):
        return self._state

    def load_state_dict(self, state):
        self._state = state
        self.logZ = state["objective"]["logZ"]


class TestFullStateCheckpointSave:
    def test_save_creates_file(self, tmp_path):
        ckpt = FullStateCheckpoint(save_dir=tmp_path, every_steps=1)
        learner = _MockLearner(logz=1.5)
        path = ckpt.save(step=10, learner=learner)

        assert path.exists()
        assert "checkpoint_000010" in path.name

    def test_save_contains_expected_keys(self, tmp_path):
        ckpt = FullStateCheckpoint(save_dir=tmp_path, every_steps=1)
        learner = _MockLearner(logz=2.0)
        path = ckpt.save(step=5, learner=learner)

        data = torch.load(path, weights_only=False)
        assert "step_count" in data
        assert "logZ" in data
        assert data["step_count"] == 5
        assert data["logZ"] == pytest.approx(2.0)


class TestFullStateCheckpointLoadRoundtrip:
    def test_load_roundtrip(self, tmp_path):
        ckpt = FullStateCheckpoint(save_dir=tmp_path)
        learner = _MockLearner(logz=3.14)
        path = ckpt.save(step=42, learner=learner)

        state = ckpt.load(path)
        assert isinstance(state, CheckpointState)
        assert state.step_count == 42
        assert state.logZ == pytest.approx(3.14)

    def test_load_nonexistent_raises(self, tmp_path):
        ckpt = FullStateCheckpoint(save_dir=tmp_path)
        with pytest.raises(FileNotFoundError):
            ckpt.load(tmp_path / "nonexistent.pt")


class TestRestoreFallback:
    def test_restore_with_valid_learner(self, tmp_path):
        ckpt = FullStateCheckpoint(save_dir=tmp_path)
        learner = _MockLearner(logz=5.0)
        ckpt.save(step=10, learner=learner)

        new_learner = _MockLearner(logz=0.0)
        step = ckpt.restore(new_learner, path=ckpt.find_latest())
        assert step == 10
        assert new_learner.logZ == pytest.approx(5.0)

    def test_restore_returns_zero_when_no_checkpoint(self, tmp_path):
        ckpt = FullStateCheckpoint(save_dir=tmp_path)
        learner = _MockLearner()
        step = ckpt.restore(learner)
        assert step == 0


class TestCleanup:
    def test_keeps_only_last_n_checkpoints(self, tmp_path):
        ckpt = FullStateCheckpoint(save_dir=tmp_path, every_steps=1, keep_last=2)
        learner = _MockLearner()

        for i in range(5):
            ckpt.save(step=i, learner=learner)

        remaining = list(tmp_path.glob("checkpoint_*.pt"))
        assert len(remaining) == 2


class TestFindLatest:
    def test_returns_most_recent(self, tmp_path):
        ckpt = FullStateCheckpoint(save_dir=tmp_path, keep_last=10)
        learner = _MockLearner()
        ckpt.save(step=1, learner=learner)
        ckpt.save(step=5, learner=learner)
        ckpt.save(step=3, learner=learner)

        latest = ckpt.find_latest()
        assert latest is not None
        assert "000003" in latest.name  # step=3 was saved last

    def test_returns_none_when_empty(self, tmp_path):
        ckpt = FullStateCheckpoint(save_dir=tmp_path)
        assert ckpt.find_latest() is None


class TestMaybeSave:
    def test_saves_at_interval(self, tmp_path):
        ckpt = FullStateCheckpoint(save_dir=tmp_path, every_steps=5)
        learner = _MockLearner()
        assert ckpt.maybe_save(step=5, learner=learner) is not None

    def test_skips_between_intervals(self, tmp_path):
        ckpt = FullStateCheckpoint(save_dir=tmp_path, every_steps=5)
        learner = _MockLearner()
        assert ckpt.maybe_save(step=3, learner=learner) is None

    def test_never_saves_when_disabled(self, tmp_path):
        ckpt = FullStateCheckpoint(save_dir=tmp_path, every_steps=0)
        learner = _MockLearner()
        assert ckpt.maybe_save(step=100, learner=learner) is None
