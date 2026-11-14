"""Non-neural baselines that every model must beat.

Three cheap references, all fitted on the training split only:
* majority class (barrier / momentum),
* persistence (future close == current close, future high/low == current),
* logistic regression on the last bar's 12 features.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

from patron.features.schema import FEATURE_ORDER
from patron.labels.barrier import BARRIER_CLASSES
from patron.labels.momentum import MOMENTUM_CLASSES
from patron.labels.range_labels import frame_to_targets

LOGGER = logging.getLogger(__name__)


@dataclass
class BaselineResult:
    name: str
    task: str
    metrics: dict
    n_samples: int

    def as_dict(self) -> dict:
        return {"name": self.name, "task": self.task, "n": self.n_samples, "metrics": self.metrics}


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int]) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_class_f1": [
            float(v)
            for v in f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
        ],
    }


def run_baselines(
    frame: pd.DataFrame,
    scaled_features: np.ndarray,
    positions: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    horizon: int = 6,
) -> dict:
    """Fit/evaluate all baselines on the given split indices (into `positions`)."""
    train_pos = positions[train_idx]
    test_pos = positions[test_idx]
    rows = frame.iloc[test_pos]
    train_rows = frame.iloc[train_pos]

    y_barrier_train = train_rows["barrier_label"].to_numpy()
    y_momentum_train = train_rows["momentum_class"].to_numpy()
    y_barrier_test = rows["barrier_label"].to_numpy()
    y_momentum_test = rows["momentum_class"].to_numpy()

    results: list[BaselineResult] = []

    # ---------------------------------------------------------- majority class
    majority_barrier = int(np.bincount(y_barrier_train, minlength=3).argmax())
    majority_momentum = int(np.bincount(y_momentum_train, minlength=5).argmax())
    results.append(
        BaselineResult(
            "majority_class",
            "barrier",
            {
                "predicted_class": BARRIER_CLASSES[majority_barrier],
                **_classification_metrics(y_barrier_test, np.full_like(y_barrier_test, majority_barrier), [0, 1, 2]),
            },
            len(test_pos),
        )
    )
    results.append(
        BaselineResult(
            "majority_class",
            "momentum",
            {
                "predicted_class": MOMENTUM_CLASSES[majority_momentum],
                **_classification_metrics(y_momentum_test, np.full_like(y_momentum_test, majority_momentum), [0, 1, 2, 3, 4]),
            },
            len(test_pos),
        )
    )

    # ------------------------------------------------------------- persistence
    targets = frame_to_targets(rows, horizon=horizon)
    zeros = np.zeros_like(targets)
    per_step = {}
    for step in range(horizon):
        per_step[f"step_{step + 1}"] = {
            "close_mae": float(mean_absolute_error(targets[:, step, 2], zeros[:, step, 2])),
            "high_mae": float(mean_absolute_error(targets[:, step, 0], zeros[:, step, 0])),
            "low_mae": float(mean_absolute_error(targets[:, step, 1], zeros[:, step, 1])),
        }
    results.append(
        BaselineResult(
            "persistence_flat",
            "range",
            {
                "close_mae_all_steps": float(mean_absolute_error(targets[:, :, 2], zeros[:, :, 2])),
                "per_step": per_step,
                "note": "flat price forecast in relative-return space",
            },
            len(test_pos),
        )
    )

    # --------------------------------------------------- logistic regression
    train_mask = y_barrier_train >= 0
    test_mask = y_barrier_test >= 0
    X_train = scaled_features[train_pos][train_mask]
    X_test = scaled_features[test_pos][test_mask]
    if X_train.shape[0] > 100 and X_test.shape[0] > 0:
        lr = LogisticRegression(max_iter=1000, multi_class="multinomial", n_jobs=-1)
        lr.fit(X_train, y_barrier_train[train_mask])
        pred = lr.predict(X_test)
        results.append(
            BaselineResult(
                "logistic_regression",
                "barrier",
                {
                    "features": "last bar, 12 scaled features",
                    **_classification_metrics(y_barrier_test[test_mask], pred, [0, 1, 2]),
                },
                int(test_mask.sum()),
            )
        )

        train_mask_m = y_momentum_train >= 0
        test_mask_m = y_momentum_test >= 0
        lr_m = LogisticRegression(max_iter=1000, multi_class="multinomial", n_jobs=-1)
        lr_m.fit(scaled_features[train_pos][train_mask_m], y_momentum_train[train_mask_m])
        pred_m = lr_m.predict(scaled_features[test_pos][test_mask_m])
        results.append(
            BaselineResult(
                "logistic_regression",
                "momentum",
                {
                    "features": "last bar, 12 scaled features",
                    **_classification_metrics(y_momentum_test[test_mask_m], pred_m, [0, 1, 2, 3, 4]),
                },
                int(test_mask_m.sum()),
            )
        )

    payload = {"baselines": [r.as_dict() for r in results], "feature_order": list(FEATURE_ORDER)}
    LOGGER.info("baselines: %s", [(r.name, r.task, r.metrics.get("accuracy")) for r in results])
    return payload
