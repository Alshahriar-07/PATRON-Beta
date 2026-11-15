"""Build the multi-venue BTC panel and the data-expansion report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from patron.data.align import align_venue_panel, assert_panel_integrity
from patron.data.collect import partition_path
from patron.data.markets import VENUE_ORDER, get_venue
from patron.data.quality import frame_checksum, validate_venue_frame
from patron.logging_utils import setup_logging
from patron.paths import MANIFEST_DIR, REPORT_DIR, UNIFIED_DIR, ensure_dirs


def _load_manifest(name: str) -> dict | None:
    path = MANIFEST_DIR / name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PATRON data expansion report")
    parser.add_argument("--timeframe", default="15m")
    parser.add_argument("--assets", nargs="+", default=["BTC"])
    args = parser.parse_args(argv)

    ensure_dirs()
    setup_logging()

    frames: dict[str, pd.DataFrame] = {}
    rows: list[dict] = []
    for slug in VENUE_ORDER:
        venue = get_venue(slug)
        path = partition_path(venue, "BTC", args.timeframe)
        if not path.exists():
            rows.append({"exchange": slug, "status": "missing"})
            continue
        frame = pd.read_parquet(path)
        frames[slug] = frame
        report = validate_venue_frame(frame, venue, "BTC", args.timeframe)
        rows.append(
            {
                "exchange": slug,
                "quote": venue.quote,
                "rows": int(len(frame)),
                "start": str(frame["timestamp_open"].min()),
                "end": str(frame["timestamp_open"].max()),
                "coverage": round(report.coverage_ratio, 4),
                "gaps": report.gap_count,
                "missing_bars": report.missing_bars,
                "outliers": report.outlier_count,
                "unclosed": report.unclosed_candles,
                "valid": report.valid,
                "checksum": frame_checksum(frame)[:16],
                "history_limit": venue.max_history_days,
            }
        )

    coverage = pd.DataFrame(rows)
    panel_stats: dict = {}
    if len(frames) >= 2:
        panel, panel_stats = align_venue_panel(frames, timeframe=args.timeframe)
        assert_panel_integrity(panel)
        UNIFIED_DIR.mkdir(parents=True, exist_ok=True)
        panel_path = UNIFIED_DIR / f"panel_BTC_{args.timeframe}.parquet"
        panel.to_parquet(panel_path, index=False)
        (UNIFIED_DIR / f"panel_BTC_{args.timeframe}.manifest.json").write_text(
            json.dumps(
                {
                    "path": str(panel_path),
                    "rows": int(len(panel)),
                    "checksum": frame_checksum(panel.rename(columns={"timestamp_open": "timestamp_open"})),
                    "stats": panel_stats,
                    "note": "Venue columns are never merged row-wise; USD-quoted venues "
                    "are compared on returns only.",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    else:
        panel_stats = {"warning": "fewer than two venue frames available"}

    asset_manifest = _load_manifest("assets_15m_binance_bybit.json")
    multi_venue_manifest = _load_manifest("btc_multivenue_15m.json")
    refresh_manifest = _load_manifest("btc_multivenue_15m_refresh.json")

    lines = [
        "# PATRON Data Expansion Report",
        "",
        "Dataset version **2.0.0** — multi-venue BTC/USDT (and USD) 15m plus a 10-asset",
        "auxiliary set. The original 5-year Binance-only dataset (`1.0.0`) is untouched",
        "and still backs the shipped v2 checkpoint.",
        "",
        "## BTC coverage by venue",
        "",
        "| Exchange | Quote | Rows | Start | End | Coverage | Gaps | Missing bars | Outliers | Valid | Checksum |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in coverage.itertuples():
        if getattr(row, "status", None) == "missing":
            lines.append(f"| {row.exchange} | — | 0 | — | — | — | — | — | — | missing | — |")
            continue
        lines.append(
            f"| {row.exchange} | {row.quote} | {row.rows:,} | {row.start[:10]} | {row.end[:10]} | "
            f"{row.coverage:.3f} | {row.gaps} | {row.missing_bars} | {row.outliers} | "
            f"{'yes' if row.valid else 'no'} | `{row.checksum}` |"
        )

    lines += [
        "",
        "### Venue-specific coverage limits (documented, not silently patched)",
        "",
        "- **Kraken** `/0/public/OHLC` returns at most 720 candles per pairing (~7.5 days of",
        "  15m). The collector clamps the request window and records the clamp in each",
        "  partition's metadata; Kraken therefore contributes a recent-window sanity series,",
        "  not a 3-year history.",
        "- **Bitget** v2 spot candles only reach back ~10 days for 15m, so it is likewise a",
        "  recent-window series.",
        "- **Coinbase** and **Kraken** quote BTC/USD, not USDT. Only returns and relative",
        "  measures are comparable to the USDT venues; raw closes differ by the USDT/USD basis.",
        "- Binance, Bybit and OKX serve the full 2023-11 → present window.",
        "",
        "## Cross-venue alignment (BTC, 15m)",
        "",
    ]
    if "overlap" in panel_stats:
        lines += [
            "| Pair | Overlapping bars | Return correlation | Mean abs log basis |",
            "|---|---|---|---|",
        ]
        for pair, stats in panel_stats["overlap"].items():
            if "return_corr" not in stats:
                continue
            lines.append(
                f"| {pair.replace('|', ' vs ')} | {stats['n']:,} | {stats['return_corr']:.6f} | "
                f"{stats['mean_abs_log_basis']:.6f} |"
            )
        lines += [
            "",
            f"- Panel rows (outer join): {panel_stats.get('rows', 0):,}",
            f"- Timestamps where every venue is present: {panel_stats.get('all_venues_present_share', 0):.1%}",
            "- Duplicate `(timestamp, venue, symbol)` observations: "
            f"{panel_stats.get('duplicate_timestamps', 0)}",
            "- Alignment is outer-joined per venue; candles are never averaged or overwritten.",
        ]
    else:
        lines.append("Not enough venue frames for alignment statistics.")

    lines += ["", "## Multi-asset set (Binance + Bybit, 15m)", ""]
    if asset_manifest:
        summary = asset_manifest["summary"]
        lines += [
            f"- Series: {summary['total']} (assets: "
            f"{', '.join(sorted({r['asset'] for r in asset_manifest['results'] if r['asset'] != 'BTC'}))}; "
            f"venues: {', '.join(sorted({r['exchange'] for r in asset_manifest['results']}))})",
            f"- OK: {summary['ok']} · skipped: {summary['skipped']} · empty: {summary['empty']} · failed: {summary['failed']}",
            f"- Total rows: {summary['rows']:,}",
            "",
            "| Asset | Exchange | Status | Rows | Start | End |",
            "|---|---|---|---|---|---|",
        ]
        for result in asset_manifest["results"]:
            if result["status"] != "ok":
                continue
            lines.append(
                f"| {result['asset']} | {result['exchange']} | {result['status']} | "
                f"{result['rows']:,} | {(result['start'] or '')[:10]} | {(result['end'] or '')[:10]} |"
            )
        lines += [
            "",
            "Assets other than BTC are auxiliary: they are never labelled with BTC's future",
            "outcome. Absolute price levels are never fed to the model — features are ratios.",
        ]
    else:
        lines.append("Asset manifest not found (collection still running or skipped).")

    if asset_manifest:
        coverage_rows = [
            {
                "asset": r["asset"],
                "exchange": r["exchange"],
                "rows": r["rows"],
                "start": r["start"],
                "end": r["end"],
                "checksum": r["checksum"],
                "status": r["status"],
            }
            for r in asset_manifest["results"]
        ]
        from patron.paths import ASSET_DIR

        ASSET_DIR.mkdir(parents=True, exist_ok=True)
        (ASSET_DIR / "coverage.json").write_text(
            json.dumps({"generated_at": asset_manifest["generated_at"], "series": coverage_rows}, indent=2),
            encoding="utf-8",
        )

    lines += [
        "",
        "## Storage layout note",
        "",
        "Validated exchange candles are stored under `data/raw/<exchange>/<market>/<asset>/<tf>.parquet`",
        "with a `.meta.json` sidecar (request window, checksum, coverage, full quality report).",
        "`data/processed/unified/` holds the aligned BTC panel and `data/processed/assets/coverage.json`",
        "summarises the auxiliary assets. Per-venue candles are deliberately not duplicated into",
        "`data/processed/exchange/`: one canonical copy with provenance is auditable, two copies invite drift.",
        "",
        "## Quality gate behaviour",
        "",
        "- Unclosed candles are dropped by the adapters and counted by the validator.",
        "- Duplicates are removed by `(timestamp)` within a venue and by",
        "  `(timestamp, venue, symbol)` across the panel; both counts are reported above.",
        "- Gaps are reported with missing-bar counts and are **not** forward-filled.",
        "- Return outliers (> 25% or robust z > 12) are flagged for review and kept.",
        "- Each partition's sidecar `.meta.json` stores the request window, checksum,",
        "  coverage and the full quality report.",
    ]
    (REPORT_DIR / "data_expansion_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"report: {REPORT_DIR / 'data_expansion_report.md'}")
    print(coverage.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
