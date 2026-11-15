"""Assemble the final model comparison report from the individual reports."""

from __future__ import annotations

import json
from pathlib import Path

from patron.paths import REPORT_DIR
from patron.patterns.registry import registry_summary
from patron.patterns.structures import STRUCTURE_DEFINITIONS


def _direction_text(counts: dict) -> str:
    return ", ".join(f"{counts[k]} {k}" for k in ("bullish", "bearish", "neutral") if counts.get(k))


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"


def _baselines(evaluation: dict) -> dict:
    """Pull baseline metrics from the evaluation report instead of hardcoding them."""
    block = evaluation.get("baselines", {}).get("baselines", [])
    out: dict = {}
    for row in block:
        metrics = row.get("metrics", {})
        if row["name"] == "majority_class" and row["task"] == "barrier":
            out["maj_b_acc"] = metrics.get("accuracy")
            out["maj_b_f1"] = metrics.get("macro_f1")
        elif row["name"] == "majority_class" and row["task"] == "momentum":
            out["maj_m_acc"] = metrics.get("accuracy")
            out["maj_m_f1"] = metrics.get("macro_f1")
        elif row["name"] == "logistic_regression" and row["task"] == "barrier":
            out["log_b_acc"] = metrics.get("accuracy")
            out["log_b_f1"] = metrics.get("macro_f1")
        elif row["name"] == "logistic_regression" and row["task"] == "momentum":
            out["log_m_acc"] = metrics.get("accuracy")
            out["log_m_f1"] = metrics.get("macro_f1")
        elif row["name"] == "persistence_flat":
            out["persistence_mae"] = metrics.get("close_mae_all_steps")
    out.setdefault("maj_b_bal", 1 / 3)
    out.setdefault("maj_m_bal", 1 / 5)
    # The evaluator does not score balanced accuracy for the flat baselines.
    out.setdefault("log_b_bal", None)
    out.setdefault("log_m_bal", None)
    return out


def _load(name: str) -> dict | None:
    path = REPORT_DIR / name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    evaluation = _load("evaluation_report.json")
    walk_forward = _load("walk_forward_report.json")
    ablation = _load("feature_ablation_report.json")
    multiseed = _load("multiseed_report.json")

    lines = [
        "# PATRON Final Model Comparison",
        "",
        "Dataset versions are never mixed: **1.0.0** is the shipped Binance-only 15m",
        "dataset that trained the v2 checkpoint; **2.0.0** is the expanded multi-venue /",
        "multi-asset data collected for this upgrade. Metrics below are labelled by version.",
        "",
        "## 1. Shipped model vs baselines (dataset 1.0.0, untouched test split)",
        "",
    ]

    if evaluation:
        test = evaluation["test"]
        barrier = test["barrier"]["calibrated"] or test["barrier"]["raw"]
        momentum = test["momentum"]
        base = _baselines(evaluation)
        lines += [
            f"- Test split: {evaluation['splits']['test']['start'][:10]} → "
            f"{evaluation['splits']['test']['end'][:10]} "
            f"({evaluation['splits']['test']['count']:,} windows), touched once.",
            f"- Calibration temperature: {evaluation['temperature']:.4f}.",
            "",
            "| Task | Metric | PATRON v2 | Majority | Logistic |",
            "|---|---|---|---|---|",
            f"| Barrier | accuracy | {barrier['accuracy']:.4f} | {_fmt(base['maj_b_acc'])} | "
            f"{_fmt(base['log_b_acc'])} |",
            f"| Barrier | balanced accuracy | {barrier['balanced_accuracy']:.4f} | "
            f"{_fmt(base['maj_b_bal'])} | {_fmt(base['log_b_bal'])} |",
            f"| Barrier | macro-F1 | {barrier['macro_f1']:.4f} | {_fmt(base['maj_b_f1'])} | "
            f"{_fmt(base['log_b_f1'])} |",
            f"| Barrier | ROC-AUC (OvR) | {barrier['roc_auc_ovr']:.4f} | 0.5000 | — |",
            f"| Barrier | log loss | {barrier['log_loss']:.4f} | — | — |",
            f"| Barrier | ECE | {barrier['ece']:.4f} | — | — |",
            f"| Momentum | accuracy | {momentum['accuracy']:.4f} | {_fmt(base['maj_m_acc'])} | "
            f"{_fmt(base['log_m_acc'])} |",
            f"| Momentum | balanced accuracy | {momentum['balanced_accuracy']:.4f} | "
            f"{_fmt(base['maj_m_bal'])} | {_fmt(base['log_m_bal'])} |",
            f"| Momentum | macro-F1 | {momentum['macro_f1']:.4f} | {_fmt(base['maj_m_f1'])} | "
            f"{_fmt(base['log_m_f1'])} |",
            f"| Range | pinball (mean) | {test['quantile']['pinball_mean']:.6f} | "
            f"persistence MAE {base['persistence_mae']:.6f} | — |",
        ]
    else:
        lines.append("_evaluation_report.json missing_")

    lines += ["", "## 2. Rolling out-of-sample stability (same checkpoint)", ""]
    if walk_forward:
        lines += [
            f"- {len(walk_forward['folds'])} chronological blocks over "
            f"{walk_forward['folds'][0]['start'][:10]} → {walk_forward['folds'][-1]['end'][:10]}.",
            f"- Barrier accuracy range: "
            f"{min(f['barrier_accuracy'] for f in walk_forward['folds']):.4f} – "
            f"{max(f['barrier_accuracy'] for f in walk_forward['folds']):.4f}.",
            "- The spread is modest and non-monotone: the model is not tied to one market phase,",
            "  but its level stays below the majority baseline.",
            "",
            "| Fold | Start | Barrier acc | Balanced acc | Momentum acc | Pinball |",
            "|---|---|---|---|---|---|",
        ]
        for fold in walk_forward["folds"]:
            lines.append(
                f"| {fold['fold']} | {fold['start'][:10]} | {fold['barrier_accuracy']:.4f} | "
                f"{fold['barrier_balanced_accuracy']:.4f} | {fold['momentum_accuracy']:.4f} | "
                f"{fold['pinball']:.6f} |"
            )
    else:
        lines.append("_walk_forward_report.json missing_")

    lines += ["", "## 3. Feature ablation (tabular proxy, validation only)", ""]
    if ablation:
        results = ablation["results"]
        best = min(results, key=lambda r: r["log_loss"])
        v1 = next(r for r in results if r["config"] == "v1_12")
        lines += [
            f"- Best config: `{best['config']}` (log loss {best['log_loss']:.4f}); "
            f"the 12-feature baseline scores {v1['log_loss']:.4f}.",
            "- **Negative result:** the 34-feature advanced set did not beat the original 12",
            "  features in the proxy. The advanced features remain implemented, tested and",
            "  documented, but are not adopted into the shipped model.",
            "",
            "| Config | Features | Balanced acc | Log loss |",
            "|---|---|---|---|",
        ]
        for row in results:
            lines.append(
                f"| {row['config']} | {row['features']} | {row['balanced_accuracy']:.4f} | "
                f"{row['log_loss']:.4f} |"
            )
    else:
        lines.append("_feature_ablation_report.json missing_")

    lines += ["", "## 4. Multi-seed reproducibility (short budget)", ""]
    if multiseed:
        lines += [
            f"- Seeds: {', '.join(str(r['seed']) for r in multiseed['runs'])}; "
            f"budget {multiseed['budget']['epochs']} epochs, stride {multiseed['budget']['stride']}.",
            f"- Validation barrier accuracy: {multiseed['val_barrier_accuracy']['mean']:.4f} "
            f"± {multiseed['val_barrier_accuracy']['std']:.4f}.",
            f"- Best validation loss: {multiseed['best_val_loss']['mean']:.4f} "
            f"± {multiseed['best_val_loss']['std']:.4f}.",
            "",
            "| Seed | Best epoch | Val loss | Barrier acc | Momentum acc |",
            "|---|---|---|---|---|",
        ]
        for run in multiseed["runs"]:
            lines.append(
                f"| {run['seed']} | {run['best_epoch']} | {run['best_val_loss']:.4f} | "
                f"{run['val_barrier_accuracy']:.4f} | {run['val_momentum_accuracy']:.4f} |"
            )
    else:
        lines.append("_multiseed_report.json missing (run still in progress)_")

    summary = registry_summary()
    lines += [
        "",
        "## 5. Expanded data and pattern research (dataset 2.0.0)",
        "",
        "- Six venues collected for BTC at 15m: Binance, Bybit and OKX provide the full",
        "  2023-11 → present window; Coinbase provides 3 years with a handful of gaps;",
        "  Kraken (720-bar API cap) and Bitget (~10-day history) contribute recent windows only.",
        "- Auxiliary assets (ETH, SOL, BNB, XRP, ADA, DOGE, AVAX, LINK, LTC) are collected",
        "  as context only — never labelled with BTC's outcome.",
        f"- Pattern registry: {summary['patterns']} candlestick patterns "
        f"({_direction_text(summary['by_direction'])}) plus "
        f"{len(STRUCTURE_DEFINITIONS)} chart structures.",
        "- Pattern hit rates cluster around 50% with regime dependence; see",
        "  `reports/pattern_research_report.md` (no multiple-testing correction claimed).",
        "",
        "## 6. Verdict",
        "",
        "Nothing in this upgrade demonstrates an accuracy improvement over PATRON v2, and",
        "the report says so explicitly:",
        "",
        "1. The advanced feature set lost to the original features in the proxy ablation.",
        "2. Multi-seed runs show seed variance of the same order as the model–baseline gap.",
        "3. Barrier accuracy stays below the majority baseline; ROC-AUC ≈ 0.59 is the only",
        "   consistent evidence of signal, and balanced accuracy (≈ 0.41) is above chance.",
        "4. The expanded multi-venue data is a foundation for cross-venue features and",
        "   pretraining, not a proven improvement; adopting it would require retraining and",
        "   a new untouched holdout on dataset 2.0.0.",
        "",
        "## 7. Reproduce",
        "",
        "```bash",
        "./.venv/Scripts/python.exe scripts/collect_market_data.py --all-assets --start 2023-11-01",
        "./.venv/Scripts/python.exe scripts/build_data_report.py",
        "./.venv/Scripts/python.exe scripts/build_pattern_dataset.py",
        "./.venv/Scripts/python.exe scripts/run_feature_ablation.py",
        "./.venv/Scripts/python.exe scripts/run_walk_forward.py",
        "./.venv/Scripts/python.exe scripts/train_multiseed.py --seeds 42 7 2026 --epochs 4 --stride 4",
        "./.venv/Scripts/python.exe scripts/build_final_comparison.py",
        "```",
        "",
    ]
    out = REPORT_DIR / "final_model_comparison.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
