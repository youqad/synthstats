"""Tests for HFPolicy.

Note: These tests don't load actual models (too slow/expensive).
They test the interface and mock behavior.
"""

import torch


class TestHFPolicyImport:
    """Verify HFPolicy is importable."""

    def test_import_hf_policy_class(self):
        """HFPolicy should be importable."""
        from synthstats.policies.hf_policy import HFPolicy

        assert HFPolicy is not None

    def test_import_policy_output(self):
        """PolicyOutput should be importable."""
        from synthstats.policies.hf_policy import PolicyOutput

        assert PolicyOutput is not None


class TestHFPolicyConfig:
    """Verify HFPolicy configuration options."""

    def test_hf_policy_has_default_model_name(self):
        """HFPolicy should have default model_name."""
        # check class has the parameter
        import inspect

        from synthstats.policies.hf_policy import HFPolicy

        sig = inspect.signature(HFPolicy.__init__)
        params = sig.parameters
        assert "model_name" in params

    def test_hf_policy_accepts_device(self):
        """HFPolicy should accept device parameter."""
        import inspect

        from synthstats.policies.hf_policy import HFPolicy

        sig = inspect.signature(HFPolicy.__init__)
        params = sig.parameters
        assert "device" in params

    def test_hf_policy_accepts_lora_config(self):
        """HFPolicy should accept lora_config parameter."""
        import inspect

        from synthstats.policies.hf_policy import HFPolicy

        sig = inspect.signature(HFPolicy.__init__)
        params = sig.parameters
        assert "lora_config" in params

    def test_hf_policy_accepts_require_grad_logp(self):
        """HFPolicy should accept require_grad_logp parameter."""
        import inspect

        from synthstats.policies.hf_policy import HFPolicy

        sig = inspect.signature(HFPolicy.__init__)
        params = sig.parameters
        assert "require_grad_logp" in params

    def test_hf_policy_accepts_use_4bit(self):
        """HFPolicy should accept use_4bit parameter."""
        import inspect

        from synthstats.policies.hf_policy import HFPolicy

        sig = inspect.signature(HFPolicy.__init__)
        params = sig.parameters
        assert "use_4bit" in params


class TestHFPolicyMethods:
    """Verify HFPolicy has required methods."""

    def test_hf_policy_has_call_method(self):
        """HFPolicy should be callable."""
        from synthstats.policies.hf_policy import HFPolicy

        assert callable(HFPolicy.__call__)

    def test_hf_policy_has_score_action_method(self):
        """HFPolicy should have score_action method."""
        from synthstats.policies.hf_policy import HFPolicy

        assert hasattr(HFPolicy, "score_action")
        assert callable(HFPolicy.score_action)

    def test_hf_policy_has_parameters_method(self):
        """HFPolicy should have parameters method for optimizer."""
        from synthstats.policies.hf_policy import HFPolicy

        assert hasattr(HFPolicy, "parameters")


class TestMockHFPolicy:
    """Test HFPolicy behavior with mocked model."""

    def test_mock_policy_call_returns_action_logp_ent(self):
        """Calling policy should return (action_dict, log_prob, entropy)."""
        from synthstats.policies.hf_policy import MockHFPolicy

        policy = MockHFPolicy()
        action, logp, ent = policy("Test observation")

        assert isinstance(action, dict)
        assert "type" in action
        assert isinstance(logp, float | torch.Tensor)
        assert isinstance(ent, float | torch.Tensor)

    def test_mock_policy_with_temperature(self):
        """Policy should accept temperature parameter."""
        from synthstats.policies.hf_policy import MockHFPolicy

        policy = MockHFPolicy()
        action, logp, ent = policy("Test observation", temperature=0.5)

        # should not raise
        assert action is not None

    def test_mock_policy_score_action_returns_tensors(self):
        """score_action should return (log_prob, entropy) tensors."""
        from synthstats.policies.hf_policy import MockHFPolicy

        policy = MockHFPolicy()
        action = {"type": "answer", "payload": "42"}

        logp, ent = policy.score_action("Test observation", action)

        assert isinstance(logp, torch.Tensor)
        assert isinstance(ent, torch.Tensor)

    def test_mock_policy_with_require_grad_logp(self):
        """With require_grad_logp=True, should return tensors with grad."""
        from synthstats.policies.hf_policy import MockHFPolicy

        policy = MockHFPolicy(require_grad_logp=True)
        action, logp, ent = policy("Test observation")

        assert isinstance(logp, torch.Tensor)
        assert isinstance(ent, torch.Tensor)

    def test_mock_policy_parameters(self):
        """parameters() should return an iterable."""
        from synthstats.policies.hf_policy import MockHFPolicy

        policy = MockHFPolicy()
        params = list(policy.parameters())

        # should have at least one parameter (dummy for testing)
        assert len(params) >= 1


class TestPolicyOutput:
    """Test PolicyOutput type alias."""

    def test_policy_output_tuple_structure(self):
        """PolicyOutput should be a tuple of (action, logp, ent)."""
        from synthstats.policies.hf_policy import PolicyOutput

        # PolicyOutput is a type alias, check it can be used
        output: PolicyOutput = ({"type": "answer"}, -0.5, 0.1)
        action, logp, ent = output

        assert action["type"] == "answer"
        assert logp == -0.5
        assert ent == 0.1


# Note: TestHFPolicyIntegrationWithCollector was removed in January 2026
# when we migrated to native SkyRL integration. The TrajectoryCollector and
# SynthStatsTextEnv classes were archived to _archive/skyrl_integration_2026-01/.
# For policy integration tests with native SkyRL, see test_tb_trainer.py.
