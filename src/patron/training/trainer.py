"""CPU training loop with early stopping, checkpointing and JSONL logs."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from patron.datasets.loader import PreparedData
from patron.labels.barrier import barrier_class_weights
from patron.labels.momentum import momentum_class_weights
from patron.logging_utils import JsonlLogger
from patron.models.losses import total_loss
from patron.models.patron_net import PatronNet, count_parameters
from patron.paths import CHECKPOINT_DIR, EXPERIMENT_DIR
from patron.utils.io import write_json
from patron.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


@dataclass
class EpochRecord:
    epoch: int
    train_loss: float
    train_terms: dict
    val_loss: float
    val_terms: dict
    val_barrier_accuracy: float
    val_momentum_accuracy: float
    lr: float
    seconds: float


@dataclass
class TrainingHistory:
    records: list[EpochRecord] = field(default_factory=list)
    best_epoch: int = -1
    best_val_loss: float = float("inf")
    stopped_early: bool = False
    run_dir: Path | None = None

    def to_dict(self) -> dict:
        return {
            "epochs_run": len(self.records),
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "stopped_early": self.stopped_early,
            "run_dir": str(self.run_dir) if self.run_dir else None,
            "history": [r.__dict__ for r in self.records],
        }


def _masked_accuracy(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    with torch.no_grad():
        valid = mask > 0
        if valid.sum() == 0:
            return float("nan")
        pred = logits.argmax(dim=-1)
        return float((pred[valid] == target[valid]).float().mean())


def evaluate(
    model: PatronNet,
    loader: DataLoader,
    loss_weights: dict,
    quantiles: tuple[float, ...],
    barrier_weight: torch.Tensor | None,
    momentum_weight: torch.Tensor | None,
    max_batches: int | None = None,
) -> tuple[dict, dict]:
    """Return (mean losses, extra metrics) over the loader."""
    model.eval()
    totals: dict[str, float] = {}
    n_batches = 0
    barrier_hits, barrier_n = 0.0, 0.0
    momentum_hits, momentum_n = 0.0, 0.0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            outputs = model(batch["x"])
            _, terms = total_loss(
                outputs, batch, loss_weights, quantiles, barrier_weight, momentum_weight
            )
            for k, v in terms.items():
                totals[k] = totals.get(k, 0.0) + v
            n_batches += 1

            valid = batch["barrier_mask"] > 0
            barrier_hits += float((outputs["barrier"].argmax(-1)[valid] == batch["barrier"][valid]).sum())
            barrier_n += float(valid.sum())
            momentum_hits += float((outputs["momentum"].argmax(-1) == batch["momentum"]).sum())
            momentum_n += float(len(batch["momentum"]))

    means = {k: v / max(1, n_batches) for k, v in totals.items()}
    metrics = {
        "barrier_accuracy": barrier_hits / barrier_n if barrier_n else float("nan"),
        "momentum_accuracy": momentum_hits / momentum_n if momentum_n else float("nan"),
    }
    return means, metrics


def train(
    model: PatronNet,
    data: PreparedData,
    cfg,
    run_name: str | None = None,
    max_epochs: int | None = None,
    batch_size: int | None = None,
    num_threads: int | None = None,
    log_every: int = 1,
) -> TrainingHistory:
    set_seed(cfg.training.random_seed)
    if num_threads:
        torch.set_num_threads(num_threads)

    train_ds = data.bundles.train
    val_ds = data.bundles.val
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError("empty train or validation split")

    batch_size = batch_size or cfg.training.batch_size
    max_epochs = max_epochs or cfg.training.max_epochs
    generator = torch.Generator().manual_seed(cfg.training.random_seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

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
    loss_weights = cfg.training.loss_weights.model_dump()
    quantiles = tuple(cfg.quantiles)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay
    )

    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = EXPERIMENT_DIR / f"{run_name or cfg.project.name.lower()}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.save_snapshot(run_dir)
    jsonl = JsonlLogger(run_dir / "train_log.jsonl")

    history = TrainingHistory(run_dir=run_dir)
    patience = cfg.training.early_stopping_patience
    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"

    LOGGER.info(
        "training: %d train / %d val windows, %d params, batch=%d, max_epochs=%d",
        len(train_ds), len(val_ds), count_parameters(model), batch_size, max_epochs,
    )
    jsonl.log(
        {
            "event": "start",
            "train_windows": len(train_ds),
            "val_windows": len(val_ds),
            "params": count_parameters(model),
            "batch_size": batch_size,
            "max_epochs": max_epochs,
            "barrier_weights": barrier_w.tolist(),
            "momentum_weights": momentum_w.tolist(),
        }
    )

    for epoch in range(1, max_epochs + 1):
        started = time.time()
        model.train()
        sums: dict[str, float] = {}
        n_batches = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch["x"])
            loss, terms = total_loss(
                outputs, batch, loss_weights, quantiles, barrier_w, momentum_w
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip_norm)
            optimizer.step()
            for k, v in terms.items():
                sums[k] = sums.get(k, 0.0) + v
            n_batches += 1
            if n_batches % 200 == 0:
                LOGGER.info("epoch %d step %d loss %.4f", epoch, n_batches, terms["loss"])

        train_terms = {k: v / max(1, n_batches) for k, v in sums.items()}
        val_terms, val_metrics = evaluate(
            model, val_loader, loss_weights, quantiles, barrier_w, momentum_w
        )
        elapsed = time.time() - started
        record = EpochRecord(
            epoch=epoch,
            train_loss=train_terms["loss"],
            train_terms=train_terms,
            val_loss=val_terms["loss"],
            val_terms=val_terms,
            val_barrier_accuracy=val_metrics["barrier_accuracy"],
            val_momentum_accuracy=val_metrics["momentum_accuracy"],
            lr=optimizer.param_groups[0]["lr"],
            seconds=elapsed,
        )
        history.records.append(record)
        if log_every and epoch % log_every == 0:
            jsonl.log({"event": "epoch", **record.__dict__})
        LOGGER.info(
            "epoch %d | train %.4f | val %.4f | barrier_acc %.3f | momentum_acc %.3f | %.1fs",
            epoch, record.train_loss, record.val_loss, record.val_barrier_accuracy,
            record.val_momentum_accuracy, elapsed,
        )

        if record.val_loss < history.best_val_loss - 1e-6:
            history.best_val_loss = record.val_loss
            history.best_epoch = epoch
            _save_checkpoint(best_path, model, cfg, data, epoch, record, history)
        _save_checkpoint(last_path, model, cfg, data, epoch, record, history)

        if epoch - history.best_epoch >= patience:
            history.stopped_early = True
            LOGGER.info("early stopping at epoch %d (best %d)", epoch, history.best_epoch)
            break

    jsonl.log({"event": "finish", **{k: v for k, v in history.to_dict().items() if k != "history"}})
    write_json(history.to_dict(), run_dir / "training_history.json")
    LOGGER.info("training complete: best epoch %d val %.4f", history.best_epoch, history.best_val_loss)
    return history


def _save_checkpoint(
    path: Path,
    model: PatronNet,
    cfg,
    data: PreparedData,
    epoch: int,
    record: EpochRecord,
    history: TrainingHistory,
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": cfg.to_dict(),
            "feature_order": list(cfg.feature_order),
            "sequence_length": cfg.sequence_length,
            "horizon": cfg.horizon,
            "quantiles": list(cfg.quantiles),
            "scaler": data.scaler.to_dict(),
            "epoch": epoch,
            "best_epoch": history.best_epoch,
            "val_loss": record.val_loss,
            "val_barrier_accuracy": record.val_barrier_accuracy,
            "val_momentum_accuracy": record.val_momentum_accuracy,
            "split": data.bundles.split.summary(),
        },
        path,
    )


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    required = {"model_state", "config", "feature_order", "scaler"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"checkpoint missing keys: {sorted(missing)}")
    return payload


def build_model_from_checkpoint(payload: dict) -> PatronNet:
    from patron.config import PatronConfig
    from patron.models.patron_net import build_model

    cfg = PatronConfig.model_validate(payload["config"])
    model = build_model(cfg)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model
