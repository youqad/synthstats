import torch

from synthstats.train.objectives.subtb_endpoint import (
    broadcast_terminal_reward,
    compute_endpoint_subtb_loss,
    create_eos_unavailable_mask,
)

B, T = 2, 4


def _make_inputs(*, requires_grad: bool = False):
    log_pf = torch.randn(B, T).abs().neg()
    if requires_grad:
        log_pf = log_pf.requires_grad_(True)
    log_reward = torch.rand(B, T + 1) * 0.5
    eos_logprob = torch.randn(B, T + 1).abs().neg()
    eos_available = torch.ones(B, T + 1, dtype=torch.bool)
    return log_pf, log_reward, eos_logprob, eos_available


def test_basic_loss_computation():
    log_pf, log_reward, eos_logprob, eos_available = _make_inputs()

    loss, metrics = compute_endpoint_subtb_loss(
        log_pf=log_pf,
        log_reward=log_reward,
        eos_logprob=eos_logprob,
        eos_available=eos_available,
    )

    assert loss.shape == ()
    assert torch.isfinite(loss)
    assert loss.item() >= 0
    assert metrics["subtb_coverage"] == 1.0
    assert metrics["subtb_valid_count"] > 0


def test_masked_transitions_excluded():
    log_pf, log_reward, eos_logprob, eos_available = _make_inputs()

    mask_all = torch.ones(B, T, dtype=torch.bool)
    loss_all, m_all = compute_endpoint_subtb_loss(
        log_pf=log_pf,
        log_reward=log_reward,
        eos_logprob=eos_logprob,
        eos_available=eos_available,
        loss_mask=mask_all,
    )

    # mask out middle transitions
    mask_partial = torch.ones(B, T, dtype=torch.bool)
    mask_partial[:, 1] = False
    mask_partial[:, 2] = False
    loss_partial, m_partial = compute_endpoint_subtb_loss(
        log_pf=log_pf,
        log_reward=log_reward,
        eos_logprob=eos_logprob,
        eos_available=eos_available,
        loss_mask=mask_partial,
    )

    # fewer valid subtrajectories when some transitions are masked
    assert m_partial["subtb_valid_count"] < m_all["subtb_valid_count"]
    assert loss_all.item() != loss_partial.item()


def test_min_valid_subtrajs_fallback():
    """When no EOS is available, fallback produces graph-connected zero."""
    log_pf = torch.randn(B, T, requires_grad=True)
    log_reward = torch.zeros(B, T + 1)
    eos_logprob, eos_available = create_eos_unavailable_mask(B, T)

    loss, metrics = compute_endpoint_subtb_loss(
        log_pf=log_pf,
        log_reward=log_reward,
        eos_logprob=eos_logprob,
        eos_available=eos_available,
    )

    assert loss.item() == 0.0
    assert loss.requires_grad
    assert metrics.get("subtb_warning", 0.0) == 1.0

    # backward still works (graph-connected zero)
    loss.backward()
    assert log_pf.grad is not None


def test_broadcast_terminal_reward_shape():
    log_reward = torch.tensor([1.5, -0.3])
    seq_len = T

    per_prefix = broadcast_terminal_reward(log_reward, seq_len)

    assert per_prefix.shape == (B, T + 1)
    assert per_prefix[0, -1].item() == 1.5
    assert abs(per_prefix[1, -1].item() - (-0.3)) < 1e-5
    # non-terminal positions get the sparse default
    assert per_prefix[0, 0].item() < 0


def test_create_eos_unavailable_mask_shape():
    eos_logprob, eos_available = create_eos_unavailable_mask(B, T)

    assert eos_logprob.shape == (B, T + 1)
    assert eos_available.shape == (B, T + 1)
    assert not eos_available.any(), "all positions should be unavailable"


def test_logZ_override():
    log_pf, log_reward, eos_logprob, eos_available = _make_inputs()
    logZ_val = 42.0

    loss_default, _ = compute_endpoint_subtb_loss(
        log_pf=log_pf,
        log_reward=log_reward,
        eos_logprob=eos_logprob,
        eos_available=eos_available,
    )

    loss_override, _ = compute_endpoint_subtb_loss(
        log_pf=log_pf,
        log_reward=log_reward,
        eos_logprob=eos_logprob,
        eos_available=eos_available,
        logZ=logZ_val,
    )

    assert loss_default.item() != loss_override.item()


def test_gradient_flows_through_loss():
    log_pf, log_reward, eos_logprob, eos_available = _make_inputs(
        requires_grad=True,
    )

    loss, _ = compute_endpoint_subtb_loss(
        log_pf=log_pf,
        log_reward=log_reward,
        eos_logprob=eos_logprob,
        eos_available=eos_available,
    )

    loss.backward()
    assert log_pf.grad is not None
    assert not torch.all(log_pf.grad == 0)
    assert torch.all(torch.isfinite(log_pf.grad))
