"""Model shape/parameter-budget and loss-function tests."""

from __future__ import annotations

import torch

from patron.models.losses import masked_cross_entropy, pinball_loss, total_loss
from patron.models.patron_net import build_model, count_parameters


def test_parameter_budget(cfg):
    model = build_model(cfg)
    n = count_parameters(model)
    low, high = cfg.model.target_trainable_parameters
    assert low <= n <= high, f"{n} params outside spec budget [{low}, {high}]"


def test_forward_shapes(cfg):
    model = build_model(cfg).eval()
    x = torch.randn(3, cfg.sequence_length, cfg.feature_count)
    with torch.no_grad():
        out = model(x)
    assert out["pattern"].shape == (3, len(cfg.pattern_tags))
    assert out["barrier"].shape == (3, 3)
    assert out["quantile"].shape == (3, cfg.horizon, 3, 3)
    assert out["momentum"].shape == (3, 5)
    assert torch.isfinite(out["barrier"]).all()


def test_quantiles_never_cross(cfg):
    model = build_model(cfg).eval()
    with torch.no_grad():
        q = model(torch.randn(16, cfg.sequence_length, cfg.feature_count))["quantile"]
    assert (q[..., 0] <= q[..., 1] + 1e-6).all()
    assert (q[..., 1] <= q[..., 2] + 1e-6).all()


def test_invalid_input_shape_rejected(cfg):
    model = build_model(cfg).eval()
    try:
        model(torch.randn(2, 95, 12))
    except ValueError as exc:
        assert "expected" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("wrong sequence length must be rejected")


def test_pinball_perfect_prediction_is_zero():
    target = torch.tensor([[[0.01, -0.01, 0.0]]], dtype=torch.float32)
    preds = target.unsqueeze(-1).repeat(1, 1, 1, 3)
    assert float(pinball_loss(preds, target)) == 0.0


def test_pinball_penalises_asymmetrically():
    target = torch.zeros(1, 1, 1)
    under = torch.full((1, 1, 1, 1), -0.01)  # predicting below the target
    over = torch.full((1, 1, 1, 1), 0.01)  # predicting above the target
    # at q=0.9 an under-prediction must hurt more than an over-prediction
    assert float(pinball_loss(under, target, quantiles=(0.9,))) > float(
        pinball_loss(over, target, quantiles=(0.9,))
    )


def test_masked_ce_ignores_masked_samples():
    logits = torch.tensor([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]], requires_grad=True)
    target = torch.tensor([0, 2])
    full = masked_cross_entropy(logits, target, torch.tensor([1.0, 1.0]))
    masked = masked_cross_entropy(logits, target, torch.tensor([1.0, 0.0]))
    only_first = torch.nn.functional.cross_entropy(logits[:1], target[:1])
    assert torch.isclose(masked, only_first)
    assert not torch.isclose(masked, full)


def test_masked_ce_all_masked_is_finite():
    logits = torch.zeros(2, 3, requires_grad=True)
    loss = masked_cross_entropy(logits, torch.tensor([0, 1]), torch.zeros(2))
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None


def test_total_loss_runs_and_backprops(cfg):
    model = build_model(cfg)
    x = torch.randn(4, cfg.sequence_length, cfg.feature_count)
    out = model(x)
    batch = {
        "barrier": torch.tensor([0, 1, 2, 0]),
        "barrier_mask": torch.tensor([1.0, 1.0, 0.0, 1.0]),
        "momentum": torch.tensor([0, 2, 4, 1]),
        "pattern": (torch.rand(4, len(cfg.pattern_tags)) > 0.7).float(),
        "pattern_mask": torch.ones(4, len(cfg.pattern_tags)),
        "range": torch.randn(4, cfg.horizon, 3) * 0.01,
    }
    total, terms = total_loss(out, batch, cfg.training.loss_weights.model_dump(), tuple(cfg.quantiles))
    total.backward()
    assert torch.isfinite(total)
    assert set(terms) == {"loss", "barrier_loss", "quantile_loss", "momentum_loss", "pattern_loss"}
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
