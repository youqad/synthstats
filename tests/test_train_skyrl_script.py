import builtins

from omegaconf import OmegaConf

from scripts.train_skyrl import build_task


def test_build_task_toy_does_not_import_test_fixtures(monkeypatch):
    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "tests.fixtures" or name.startswith("tests.fixtures."):
            raise AssertionError("build_task imported tests.fixtures")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    cfg = OmegaConf.create({"task": {"name": "toy"}})

    task = build_task(cfg)

    assert task.name == "toy"
    assert task.__class__.__module__ != "tests.fixtures"
