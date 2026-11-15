"""Diagnostics for the PATRON training audit (T1).

Reads the shipped v2 checkpoint and dataset, runs no training, writes
``reports/training_upgrade/audit_evidence.json`` with:

* split facts, label balance, loss term magnitudes and class weights
* decision-rule analysis for the barrier head on the validation split
  (plain argmax vs prior-corrected argmax)
* quantile calibration against a constant-zero predictor and the persistence
  baseline
* pattern-head output statistics
* per-head gradient norms and pairwise cosine similarity on the shared trunk
* CPU timing for forward/backward to size the experiment budget

Only validation data is used here; the test split is left untouched.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from patron.config import load_config
from patron.datasets.loader import prepare_data
from patron.evaluation.metrics import (
    classification_metrics,
    expected_calibration_error,
    pattern_metrics,
    pinball,
)
from patron.labels.barrier import BARRIER_CLASSES, barrier_class_weights
from patron.labels.momentum import MOMENTUM_CLASSES, momentum_class_weights
from patron.models.losses import (
    masked_bce_with_logits,
    masked_cross_entropy,
    pinball_loss,
    total_loss,
)
from patron.models.patron_net import build_model
from patron.paths import PROJECT_ROOT as ROOT_DIR
from patron.training.trainer import load_checkpoint
from patron.utils.io import write_json

OUT_DIR = ROOT_DIR / "reports" / "training_upgrade"


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


@torch.no_grad()
def collect(model, dataset, batch_size: int = 512) -> dict[str, np.ndarray]:
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    chunks: dict[str, list[np.ndarray]] = {}
    for batch in loader:
        out = model(batch["x"])
        for key in ("barrier", "momentum", "pattern", "quantile"):
            chunks.setdefault(key, []).append(out[key].numpy())
        for key in ("barrier", "barrier_mask", "momentum", "pattern", "pattern_mask", "range"):
            chunks.setdefault("y_" + key, []).append(batch[key].numpy())
    return {k: np.concatenate(v) for k, v in chunks.items()}


def decision_rule_analysis(
    probs: np.ndarray, labels: np.ndarray, weights: np.ndarray, names: list[str], valid: np.ndarray
) -> dict:
    """Compare plain argmax with prior-corrected argmax ``argmax(p / w**alpha)``."""
    out = {"plain": classification_metrics(probs, labels, names, valid), "grid": []}
    for alpha in (0.25, 0.5, 0.75, 1.0):
        adjusted = probs / (weights.astype(np.float64) ** alpha)
        adjusted = adjusted / adjusted.sum(axis=1, keepdims=True)
        metrics = classification_metrics(adjusted, labels, names, valid)
        out["grid"].append(
            {
                "alpha": alpha,
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "macro_f1": metrics["macro_f1"],
                "log_loss": metrics["log_loss"],
                "ece": metrics["ece"],
                "prediction_counts": metrics["prediction_distribution"],
            }
        )
    return out


def quantile_diagnostics(pred: np.ndarray, target: np.ndarray, quantiles: tuple[float, ...]) -> dict:
    """Location skill vs persistence, spread calibration, per-step quality."""
    q = np.asarray(quantiles, dtype=np.float64)
    loss = pinball(pred, target, quantiles)
    zero_loss = pinball(np.zeros_like(pred), target, quantiles)
    med = pred[..., 1]
    slope = {}
    per_step = []
    for field, name in enumerate(("high", "low", "close")):
        x = med[:, :, field].reshape(-1)
        y = target[:, :, field].reshape(-1)
        var = float(np.var(x))
        slope[name] = float(np.cov(x, y, bias=True)[0, 1] / var) if var > 0 else None
    for step in range(target.shape[1]):
        per_step.append(
            {
                "step": step + 1,
                "pinball": float(loss[:, step].mean()),
                "pinball_const_zero": float(zero_loss[:, step].mean()),
                "median_mae": float(np.abs(med[:, step] - target[:, step]).mean()),
                "coverage_q10_q90": float(
                    (
                        (target[:, step] >= pred[:, step, :, 0])
                        & (target[:, step] <= pred[:, step, :, 2])
                    ).mean()
                ),
            }
        )
    width = (pred[..., 2] - pred[..., 0])
    return {
        "pinball_mean": float(loss.mean()),
        "pinball_const_zero": float(zero_loss.mean()),
        "coverage_q10_q90": float(
            ((target >= pred[..., 0]) & (target <= pred[..., 2])).mean()
        ),
        "nominal_coverage": float(quantiles[-1] - quantiles[0]),
        "median_mae": float(np.abs(med - target).mean()),
        "interval_width_mean": float(width.mean()),
        "abs_target_mean": float(np.abs(target).mean()),
        "median_regression_slope": slope,
        "per_step": per_step,
    }


def gradient_conflict(model, dataset, weights: dict, quantiles, barrier_w, momentum_w, n_batches=6) -> dict:
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    trunk = list(model.cnn.parameters()) + list(model.bilstm.parameters()) + \
        list(model.attention.parameters()) + list(model.ffn.parameters()) + \
        list(model.shared.parameters())
    norms: dict[str, list[float]] = {}
    cosines: dict[str, list[float]] = {}
    order = ("barrier", "quantile", "momentum", "pattern")
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        out = model(batch["x"])
        terms = {
            "barrier": masked_cross_entropy(
                out["barrier"], batch["barrier"], batch["barrier_mask"], barrier_w
            ),
            "quantile": pinball_loss(out["quantile"], batch["range"], quantiles),
            "momentum": torch.nn.functional.cross_entropy(
                out["momentum"], batch["momentum"], weight=momentum_w
            ),
            "pattern": masked_bce_with_logits(out["pattern"], batch["pattern"], batch["pattern_mask"]),
        }
        grads: dict[str, list[torch.Tensor]] = {}
        for name, term in terms.items():
            model.zero_grad(set_to_none=True)
            term.backward(retain_graph=True)
            grads[name] = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in trunk]
            norms.setdefault(name, []).append(
                float(torch.sqrt(sum((g ** 2).sum() for g in grads[name])).item())
            )
        for a in range(len(order)):
            for b in range(a + 1, len(order)):
                ga, gb = grads[order[a]], grads[order[b]]
                dot = sum(float((x * y).sum()) for x, y in zip(ga, gb))
                na = float(torch.sqrt(sum((x ** 2).sum() for x in ga)).item())
                nb = float(torch.sqrt(sum((x ** 2).sum() for x in gb)).item())
                key = f"{order[a]}_vs_{order[b]}"
                cosines.setdefault(key, []).append(dot / max(na * nb, 1e-12))
    model.zero_grad(set_to_none=True)
    return {
        "grad_norm_mean": {k: float(np.mean(v)) for k, v in norms.items()},
        "cosine_mean": {k: float(np.mean(v)) for k, v in cosines.items()},
        "batches": n_batches,
        "batch_size": 64,
        "trunk_parameters": int(sum(p.numel() for p in trunk)),
    }


def timing(model, dataset, weights, quantiles, barrier_w, momentum_w, batch_size=128, repeats=5) -> dict:
    """Time a real training step (all four heads in the backward graph)."""
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    batches = [b for i, b in enumerate(loader) if i < repeats]
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()
    step_times = []
    for batch in batches:
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        out = model(batch["x"])
        loss, _ = total_loss(out, batch, weights, quantiles, barrier_w, momentum_w)
        loss.backward()
        optimizer.step()
        step_times.append(time.perf_counter() - start)
    model.eval()
    return {
        "seconds_per_step_batch128": float(np.median(step_times)),
        "train_windows": len(dataset),
        "epoch_seconds_lower_bound": float(
            np.median(step_times) * np.ceil(len(dataset) / batch_size)
        ),
        "note": (
            "5-step median on the full train split; excludes validation, logging and "
            "grad clipping, so treat it as a lower bound. Measured whole epoch including "
            "validation: 125.5 s at train-stride 2 (61,304 windows)."
        ),
        "threads": torch.get_num_threads(),
        "device": "cpu",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="experiments/patron_v2_20261114-203859/best.pt")
    parser.add_argument("--out", default=str(OUT_DIR / "audit_evidence.json"))
    args = parser.parse_args()

    torch.manual_seed(42)
    cfg = load_config()
    payload = load_checkpoint(args.checkpoint)
    model = build_model(cfg)
    model.load_state_dict(payload["model_state"])
    model.eval()

    data = prepare_data(cfg)
    train_ds, val_ds = data.bundles.train, data.bundles.val
    train_frame = data.frame.iloc[train_ds.positions]
    barrier_w = torch.tensor(
        barrier_class_weights(
            train_frame["barrier_label"].to_numpy(), train_frame["barrier_mask"].to_numpy().astype(bool)
        )
    )
    momentum_w = torch.tensor(
        momentum_class_weights(
            train_frame["momentum_class"].to_numpy(),
            train_frame["momentum_class"].to_numpy() >= 0,
        )
    )
    quantiles = tuple(cfg.quantiles)
    loss_weights = cfg.training.loss_weights.model_dump()

    preds = collect(model, val_ds)
    barrier_valid = preds["y_barrier_mask"] > 0
    momentum_valid = preds["y_momentum"] >= 0
    barrier_probs = softmax(preds["barrier"])
    momentum_probs = softmax(preds["momentum"])
    pattern_probs = 1.0 / (1.0 + np.exp(-preds["pattern"]))

    barrier_block = decision_rule_analysis(
        barrier_probs, preds["y_barrier"], barrier_w.numpy(), list(BARRIER_CLASSES), barrier_valid
    )
    momentum_block = decision_rule_analysis(
        momentum_probs, preds["y_momentum"], momentum_w.numpy(), list(MOMENTUM_CLASSES), momentum_valid
    )

    per_tag = pattern_metrics(
        pattern_probs, preds["y_pattern"], preds["y_pattern_mask"], list(cfg.pattern_tags)
    )["per_tag"]
    pattern_block = {
        "macro_average_precision": pattern_metrics(
            pattern_probs, preds["y_pattern"], preds["y_pattern_mask"], list(cfg.pattern_tags)
        )["macro_average_precision"],
        "mean_predicted_probability": {
            tag: float(pattern_probs[:, i].mean()) for i, tag in enumerate(cfg.pattern_tags)
        },
        "base_rate": {tag: info.get("positive_rate") for tag, info in per_tag.items()},
        "average_precision": {tag: info.get("average_precision") for tag, info in per_tag.items()},
        "max_predicted_probability": float(pattern_probs.max()),
    }

    terms = {}
    with torch.no_grad():
        for batch in torch.utils.data.DataLoader(val_ds, batch_size=512):
            out = model(batch["x"])
            values = {
                "barrier": float(masked_cross_entropy(out["barrier"], batch["barrier"], batch["barrier_mask"], barrier_w)),
                "quantile": float(pinball_loss(out["quantile"], batch["range"], quantiles)),
                "momentum": float(torch.nn.functional.cross_entropy(out["momentum"], batch["momentum"], weight=momentum_w)),
                "pattern": float(masked_bce_with_logits(out["pattern"], batch["pattern"], batch["pattern_mask"])),
            }
            for k, v in values.items():
                terms.setdefault(k, []).append(v)
    raw = {k: float(np.mean(v)) for k, v in terms.items()}
    effective = {k: raw[k] * loss_weights[k] for k in raw}
    total = sum(effective.values())

    evidence = {
        "checkpoint": args.checkpoint,
        "dataset": str(ROOT_DIR / "data" / "features" / "BTCUSDT_dataset.parquet"),
        "n_windows": {"train": len(train_ds), "val": len(val_ds), "test": len(data.bundles.test)},
        "sequence_overlap": {
            "window_length": cfg.sequence_length,
            "adjacent_overlap_fraction": (cfg.sequence_length - 1) / cfg.sequence_length,
            "independent_chunks_estimate": int(len(train_ds) / cfg.sequence_length),
        },
        "class_weights": {
            "barrier": dict(zip(BARRIER_CLASSES, barrier_w.tolist())),
            "momentum": dict(zip(MOMENTUM_CLASSES, momentum_w.tolist())),
        },
        "val_loss_terms": {
            "raw": raw,
            "weighted": effective,
            "weighted_share": {k: v / total for k, v in effective.items()},
            "total": total,
        },
        "barrier_decision_rule": barrier_block,
        "momentum_decision_rule": momentum_block,
        "quantile": quantile_diagnostics(preds["quantile"], preds["y_range"], quantiles),
        "pattern": pattern_block,
        "gradient_conflict": gradient_conflict(
            model, train_ds, loss_weights, quantiles, barrier_w, momentum_w
        ),
        "timing": timing(
            model, train_ds, loss_weights, quantiles, barrier_w, momentum_w
        ),
        "confidence": {
            "barrier_mean_top_probability": float(barrier_probs.max(axis=1).mean()),
            "momentum_mean_top_probability": float(momentum_probs.max(axis=1).mean()),
            "barrier_ece": float(
                expected_calibration_error(barrier_probs[barrier_valid], preds["y_barrier"][barrier_valid])
            ),
        },
    }
    write_json(evidence, Path(args.out))
    print(json.dumps({k: evidence[k] for k in ("val_loss_terms", "confidence", "timing")}, indent=1))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
