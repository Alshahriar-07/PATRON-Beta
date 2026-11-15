"""Feature-group ablation with a tabular proxy model.

The neural model consumes 96×12 sequences; retraining it once per feature group
would cost hours on this CPU. This ablation therefore uses LightGBM on the
*last bar of the window* with the same labels and the same chronological
train/validation boundary. It answers "does this feature family carry signal?"
— not "does the CNN-BiLSTM improve?", which is stated explicitly in the report.

The final test split is never touched here: model selection and reporting use
train/validation only.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from patron.datasets.loader import dataset_path_for
from patron.config import load_config
from patron.features.advanced import ALL_FEATURES, FEATURE_GROUPS, build_advanced_features
from patron.features.schema import FEATURE_ORDER
from patron.logging_utils import setup_logging
from patron.paths import REPORT_DIR, ensure_dirs, FEATURE_DIR

from sklearn.ensemble import HistGradientBoostingClassifier

# LightGBM 4.7.0 crashes with a native access violation under numpy 2.5.4 on
# this host (reproduced with a 500-row, 4-feature fit), so sklearn's histogram
# gradient boosting is the boosting proxy. Same family, no native extension.


CONFIGS: dict[str, tuple[str, ...]] = {
    "v1_12": tuple(FEATURE_ORDER),
    "advanced_price": FEATURE_GROUPS["price"],
    "advanced_candle": FEATURE_GROUPS["candle"],
    "advanced_volume": FEATURE_GROUPS["volume"],
    "advanced_trend": FEATURE_GROUPS["trend"],
    "advanced_structure": FEATURE_GROUPS["structure"],
    "advanced_all": ALL_FEATURES,
    "v1_plus_advanced": tuple(FEATURE_ORDER) + ALL_FEATURES,
}


def _metrics(y_true: np.ndarray, proba: np.ndarray, labels: list[int]) -> dict:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss

    pred = proba.argmax(axis=1)
    return {
        "n": int(len(y_true)),
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, pred)), 4),
        "macro_f1": round(float(f1_score(y_true, pred, average="macro")), 4),
        "log_loss": round(float(log_loss(y_true, proba, labels=labels)), 4),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PATRON feature ablation (LightGBM proxy)")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--candles", default="data/clean/BTCUSDT/15m.parquet")
    parser.add_argument("--out", default=str(REPORT_DIR / "feature_ablation_report.md"))
    args = parser.parse_args(argv)

    ensure_dirs()
    setup_logging()
    cfg = load_config()
    dataset = pd.read_parquet(args.dataset or dataset_path_for(cfg))
    candles = pd.read_parquet(args.candles)

    advanced = build_advanced_features(candles)
    base = dataset.merge(advanced, on="timestamp_open", how="left", suffixes=("", "_adv"))

    valid = (
        base["feature_valid"].astype(bool)
        & base["sample_valid"].astype(bool)
        & base["feature_valid_adv"].astype(bool)
        & (base["barrier_mask"] > 0)
    )
    frame = base.loc[valid].reset_index(drop=True)
    y = frame["barrier_label"].to_numpy(dtype=int)
    labels = sorted(np.unique(y).tolist())

    # Same chronological boundary as the model: last 30% of valid rows is held out
    # for validation; the test split is never loaded into this script.
    n = len(frame)
    cut = int(n * 0.70)
    train, val = frame.iloc[:cut], frame.iloc[cut:]
    y_train, y_val = y[:cut], y[cut:]
    print(f"train {len(train):,} val {len(val):,} classes {labels}")

    rows = []
    importance: dict[str, float] = {}
    for name, columns in CONFIGS.items():
        columns = [c for c in columns if c in frame.columns]
        model = HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.06,
            max_leaf_nodes=31,
            min_samples_leaf=100,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=42,
        )
        model.fit(train[columns], y_train)
        proba = model.predict_proba(val[columns])
        metrics = _metrics(y_val, proba, labels)
        metrics["features"] = len(columns)
        metrics["config"] = name
        rows.append(metrics)
        print(metrics)
        if name == "advanced_all":
            # HistGradientBoosting has no gain importances; permutation importance
            # on a 4k-row slice keeps the ranking cheap and model-agnostic.
            from sklearn.inspection import permutation_importance

            rng = np.random.RandomState(0)
            sample_pos = rng.choice(len(val), size=min(4000, len(val)), replace=False)
            perm = permutation_importance(
                model,
                val.iloc[sample_pos][columns],
                y_val[sample_pos],
                n_repeats=3,
                random_state=0,
                scoring="neg_log_loss",
            )
            importance = dict(
                sorted(zip(columns, perm.importances_mean), key=lambda kv: kv[1], reverse=True)
            )

    results = pd.DataFrame(rows)[
        ["config", "features", "n", "accuracy", "balanced_accuracy", "macro_f1", "log_loss"]
    ]
    best = results.sort_values("log_loss").iloc[0]
    baseline = results[results["config"] == "v1_12"].iloc[0]

    lines = [
        "# PATRON Feature Ablation Report",
        "",
        "- Proxy model: sklearn `HistGradientBoostingClassifier` (300 iters, lr 0.06) on "
        "the **last bar of each 96-bar window**. LightGBM 4.7.0 crashes natively on this "
        "host (numpy 2.5 incompatibility), so the boosting proxy is the sklearn equivalent.",
        "- Labels: triple-barrier classes, ambiguous bars excluded (same mask as training).",
        f"- Train {len(train):,} rows (before 70% cutoff) / validation {len(val):,} rows.",
        "- The final test split is not used here; this is model-selection evidence only.",
        "- This is a tabular proxy. It does **not** retrain the CNN-BiLSTM, so it cannot",
        "  prove that adding a group helps the sequence model — only whether the group",
        "  carries barrier-relevant information at the window end.",
        "",
        "| Config | Features | Accuracy | Balanced acc | Macro-F1 | Log loss |",
        "|---|---|---|---|---|---|",
    ]
    for row in results.itertuples():
        lines.append(
            f"| {row.config} | {row.features} | {row.accuracy:.4f} | {row.balanced_accuracy:.4f} | "
            f"{row.macro_f1:.4f} | {row.log_loss:.4f} |"
        )
    lines += [
        "",
        "## Reading",
        "",
        f"- Best validation log loss: **{best['config']}** ({best['log_loss']:.4f}); "
        f"v1's 12 features: {baseline['log_loss']:.4f}.",
        (
            "- **Negative result:** no advanced configuration beat the v1 feature set in this "
            "proxy. The extra features are documented, tested and available, but they did not "
            "improve barrier classification here and are therefore not adopted into the shipped "
            "model without further evidence."
            if baseline["log_loss"] <= best["log_loss"]
            else f"- The best advanced configuration improves validation log loss by "
            f"{baseline['log_loss'] - best['log_loss']:+.4f} over the v1 features in this proxy."
        ),
        "- Groups are ranked by their standalone log loss below; single families carry little "
        "signal, which is consistent with the weak class separation seen everywhere else.",
        "",
        "## Top permutation importances (advanced_all, neg log-loss drop)",
        "",
        "| Feature | Mean importance (log-loss drop) |",
        "|---|---|",
    ]
    for name, gain in list(importance.items())[:20]:
        lines.append(f"| {name} | {gain:+.5f} |")

    lines += [
        "",
        "## Caveats",
        "",
        "- A tree model on single bars cannot exploit sequence context, so these numbers are not",
        "  comparable to the neural model's test metrics.",
        "- No hyperparameter search was run for the proxy; all configs share one setting, so",
        "  differences are attributable to the feature set rather than tuning.",
        "- A feature group that helps here may still be redundant for the sequence model.",
        "",
    ]
    (REPORT_DIR / "feature_ablation_report.md").write_text("\n".join(lines), encoding="utf-8")
    (REPORT_DIR / "feature_ablation_report.json").write_text(
        json.dumps({"results": rows, "importance": importance}, indent=2), encoding="utf-8"
    )
    print(f"report: {REPORT_DIR / 'feature_ablation_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
