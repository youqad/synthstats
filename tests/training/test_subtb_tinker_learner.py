import pytest
import torch

from synthstats.train.learners.subtb_tinker import SubTBTinkerLearner


class _MockTinkerTrainer:
    def __init__(self, logz_init: float = 0.0):
        self.logZ = torch.nn.Parameter(torch.tensor(logz_init))
        self._train_step_calls = 0

    def train_step(self, batch):
        self._train_step_calls += 1
        return {"loss": 0.42}


class TestLazyCreation:
    def test_trainer_is_none_until_first_use(self):
        learner = SubTBTinkerLearner()
        assert learner._trainer is None

    def test_get_trainer_creates_on_demand(self, monkeypatch):
        mock_trainer = _MockTinkerTrainer(logz_init=1.0)

        def _fake_get_trainer(self):
            if self._trainer is None:
                self._trainer = mock_trainer
            return self._trainer

        monkeypatch.setattr(SubTBTinkerLearner, "_get_trainer", _fake_get_trainer)

        learner = SubTBTinkerLearner()
        assert learner._trainer is None
        assert learner.logZ == pytest.approx(1.0)
        assert learner._trainer is mock_trainer


class TestUpdate:
    def test_delegates_to_trainer_train_step(self, monkeypatch):
        mock_trainer = _MockTinkerTrainer(logz_init=2.0)
        monkeypatch.setattr(SubTBTinkerLearner, "_get_trainer", lambda self: mock_trainer)

        learner = SubTBTinkerLearner()
        metrics = learner.update({"prompts": ["hi"], "completions": ["bye"]})

        assert mock_trainer._train_step_calls == 1
        assert "loss" in metrics
        assert metrics["loss"] == pytest.approx(0.42)
        assert "logZ" in metrics


class TestLogZProperty:
    def test_returns_float(self, monkeypatch):
        mock_trainer = _MockTinkerTrainer(logz_init=3.14)
        monkeypatch.setattr(SubTBTinkerLearner, "_get_trainer", lambda self: mock_trainer)

        learner = SubTBTinkerLearner()
        assert isinstance(learner.logZ, float)
        assert learner.logZ == pytest.approx(3.14)


class TestStateDictRoundtrip:
    def test_state_dict_contains_logz(self, monkeypatch):
        mock_trainer = _MockTinkerTrainer(logz_init=5.0)
        monkeypatch.setattr(SubTBTinkerLearner, "_get_trainer", lambda self: mock_trainer)

        learner = SubTBTinkerLearner()
        state = learner.state_dict()
        assert "logZ" in state
        assert state["logZ"] == pytest.approx(5.0)

    def test_load_state_dict_restores_logz(self, monkeypatch):
        mock_trainer = _MockTinkerTrainer(logz_init=0.0)
        monkeypatch.setattr(SubTBTinkerLearner, "_get_trainer", lambda self: mock_trainer)

        learner = SubTBTinkerLearner()
        learner.load_state_dict({"logZ": 7.5})
        assert learner.logZ == pytest.approx(7.5)
