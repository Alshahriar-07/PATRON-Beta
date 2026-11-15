"""Resumable multi-venue / multi-asset collection with provenance.

Each download is written as one parquet partition per
``exchange/market/asset/timeframe`` plus a sidecar metadata JSON containing the
request window, row counts, coverage, checksum and the full quality report.
Existing partitions are merged (never overwritten blindly) so re-running the
collector is cheap and safe.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from patron.data.markets import TIMEFRAME_MINUTES, VENUE_ORDER, Venue, get_venue
from patron.data.quality import VenueQualityReport, frame_checksum, validate_venue_frame
from patron.data.venues import VenueError, fetch_candles
from patron.paths import MANIFEST_DIR, RAW_DIR
from patron.utils.io import write_parquet

LOGGER = logging.getLogger(__name__)


@dataclass
class CollectResult:
    exchange: str
    market_type: str
    asset: str
    timeframe: str
    rows: int = 0
    start: str | None = None
    end: str | None = None
    path: str | None = None
    checksum: str | None = None
    status: str = "ok"  # ok | skipped | empty | failed
    error: str | None = None
    report: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def partition_path(venue: Venue, asset: str, timeframe: str) -> Path:
    return RAW_DIR / venue.slug / venue.market_type / asset.upper() / f"{timeframe}.parquet"


def metadata_path(venue: Venue, asset: str, timeframe: str) -> Path:
    return partition_path(venue, asset, timeframe).with_suffix(".meta.json")


def _load_existing(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception as exc:  # pragma: no cover - corrupted cache is rare
        LOGGER.warning("could not read %s (%s); refetching", path, exc)
        return pd.DataFrame()


def collect_venue(
    exchange: str,
    asset: str,
    timeframe: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp | None = None,
    pause: float | None = None,
    refresh: bool = False,
) -> CollectResult:
    """Download (or refresh) one venue/asset/timeframe partition."""
    venue = get_venue(exchange)
    asset = asset.upper()
    end_ts = pd.Timestamp(end or datetime.now(timezone.utc))
    end_ts = end_ts.tz_localize("UTC") if end_ts.tzinfo is None else end_ts.tz_convert("UTC")
    start_ts = pd.Timestamp(start)
    start_ts = start_ts.tz_localize("UTC") if start_ts.tzinfo is None else start_ts.tz_convert("UTC")

    clamped_note = None
    if venue.max_history_days is not None:
        floor_ts = pd.Timestamp.now(timezone.utc) - pd.Timedelta(days=venue.max_history_days)
        if start_ts < floor_ts:
            clamped_note = (
                f"start clamped from {start_ts.isoformat()} to {floor_ts.isoformat()} "
                f"(venue history depth = {venue.max_history_days} days)"
            )
            LOGGER.warning("%s: %s", venue.slug, clamped_note)
            start_ts = floor_ts

    existing = _load_existing(partition_path(venue, asset, timeframe))
    if not refresh and not existing.empty:
        have_start = pd.Timestamp(existing["timestamp_open"].min())
        have_end = pd.Timestamp(existing["timestamp_open"].max())
        step = pd.Timedelta(minutes=TIMEFRAME_MINUTES[timeframe])
        if have_start <= start_ts and have_end >= end_ts - step:
            report = validate_venue_frame(existing, venue, asset, timeframe)
            return CollectResult(
                exchange=venue.slug,
                market_type=venue.market_type,
                asset=asset,
                timeframe=timeframe,
                rows=int(len(existing)),
                start=pd.Timestamp(existing["timestamp_open"].iloc[0]).isoformat(),
                end=pd.Timestamp(existing["timestamp_open"].iloc[-1]).isoformat(),
                path=str(partition_path(venue, asset, timeframe)),
                checksum=frame_checksum(existing),
                status="skipped",
                report=report.to_dict(),
            )

    try:
        fresh = fetch_candles(venue, asset, timeframe, start_ts, end_ts, pause=pause)
    except (VenueError, KeyError) as exc:
        LOGGER.error("%s %s %s: %s", venue.slug, asset, timeframe, exc)
        return CollectResult(
            exchange=venue.slug,
            market_type=venue.market_type,
            asset=asset,
            timeframe=timeframe,
            status="failed",
            error=str(exc),
        )

    if fresh.empty:
        return CollectResult(
            exchange=venue.slug,
            market_type=venue.market_type,
            asset=asset,
            timeframe=timeframe,
            status="empty",
            error="venue returned no candles for the requested window",
        )

    merged = fresh
    if not existing.empty:
        common = [c for c in fresh.columns if c in existing.columns]
        merged = pd.concat([existing[common], fresh[common]], ignore_index=True)
        merged = merged.drop_duplicates(subset=["timestamp_open"], keep="last")
        merged = merged.sort_values("timestamp_open").reset_index(drop=True)

    report = validate_venue_frame(merged, venue, asset, timeframe)
    if report.duplicate_timestamps or report.non_monotonic:
        merged = merged.drop_duplicates(subset=["timestamp_open"], keep="last").sort_values(
            "timestamp_open"
        )

    path = partition_path(venue, asset, timeframe)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_parquet(merged, path)

    payload = {
        "exchange": venue.slug,
        "exchange_name": venue.name,
        "market_type": venue.market_type,
        "quote": venue.quote,
        "asset": asset,
        "symbol": f"{asset}/{venue.quote}",
        "venue_symbol": venue.venue_symbol(asset),
        "timeframe": timeframe,
        "requested": {"start": start_ts.isoformat(), "end": end_ts.isoformat()},
        "rows": int(len(merged)),
        "start": pd.Timestamp(merged["timestamp_open"].iloc[0]).isoformat(),
        "end": pd.Timestamp(merged["timestamp_open"].iloc[-1]).isoformat(),
        "checksum_sha256": frame_checksum(merged),
        "coverage_ratio": report.coverage_ratio,
        "gap_count": report.gap_count,
        "missing_bars": report.missing_bars,
        "outlier_count": report.outlier_count,
        "licence_note": venue.licence_note,
        "history_limit_note": clamped_note,
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "quality": report.to_dict(),
    }
    metadata_path(venue, asset, timeframe).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return CollectResult(
        exchange=venue.slug,
        market_type=venue.market_type,
        asset=asset,
        timeframe=timeframe,
        rows=int(len(merged)),
        start=payload["start"],
        end=payload["end"],
        path=str(path),
        checksum=payload["checksum_sha256"],
        status="ok",
        report=report.to_dict(),
    )


def _summarise(results: list[dict]) -> dict:
    return {
        "total": len(results),
        "ok": sum(1 for r in results if r.get("status") == "ok"),
        "skipped": sum(1 for r in results if r.get("status") == "skipped"),
        "empty": sum(1 for r in results if r.get("status") == "empty"),
        "failed": sum(1 for r in results if r.get("status") == "failed"),
        "rows": int(sum(r.get("rows") or 0 for r in results)),
    }


def _merge_results(old: list[dict], new: list[dict]) -> list[dict]:
    """Merge manifest rows by (exchange, asset, timeframe); a newer ok row wins."""
    merged: dict[tuple, dict] = {}
    for row in old + new:
        key = (row.get("exchange"), row.get("asset"), row.get("timeframe"))
        previous = merged.get(key)
        if previous is None or row.get("status") == "ok" or previous.get("status") != "ok":
            merged[key] = row
    order = {v: i for i, v in enumerate(VENUE_ORDER)}
    return sorted(merged.values(), key=lambda r: (order.get(r.get("exchange"), 99), r.get("asset")))


def rebuild_manifest(name: str, timeframe: str = "15m", assets: list[str] | None = None) -> dict:
    """Rebuild a manifest purely from on-disk partitions and their sidecar metadata."""
    wanted = {a.upper() for a in assets} if assets else None
    results: list[dict] = []
    for meta_path in sorted(RAW_DIR.glob(f"*/spot/*/{timeframe}.meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if wanted is not None and meta["asset"].upper() not in wanted:
            continue
        quality = meta.get("quality", {})
        results.append(
            CollectResult(
                exchange=meta["exchange"],
                market_type=meta.get("market_type", "spot"),
                asset=meta["asset"],
                timeframe=meta["timeframe"],
                rows=meta.get("rows", 0),
                start=meta.get("start"),
                end=meta.get("end"),
                path=str(meta_path.with_suffix(".parquet")),
                checksum=meta.get("checksum_sha256"),
                status="ok" if quality.get("valid", True) else "failed",
                error=None if quality.get("valid", True) else "; ".join(quality.get("errors", [])),
                report=quality,
            ).to_dict()
        )
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "requested": {
            "source": "rebuilt from partition metadata",
            "timeframe": timeframe,
            "assets": sorted({r["asset"] for r in results}),
            "exchanges": sorted({r["exchange"] for r in results}),
        },
        "summary": _summarise(results),
        "results": results,
    }
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    (MANIFEST_DIR / name).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    LOGGER.info("manifest rebuilt from sidecars: %s", MANIFEST_DIR / name)
    return manifest


def collect_many(
    exchanges: list[str],
    assets: list[str],
    timeframe: str,
    start: str,
    end: str | None = None,
    pause: float | None = None,
    manifest_name: str | None = None,
    refresh: bool = False,
) -> dict:
    """Collect a venue × asset grid and write one manifest for the batch."""
    started = datetime.now(timezone.utc)
    results: list[CollectResult] = []
    for exchange in exchanges:
        venue = get_venue(exchange)
        for asset in assets:
            if asset.upper() not in venue.symbols or not venue.supports(timeframe):
                LOGGER.info("%s does not list %s %s; skipped", exchange, asset, timeframe)
                results.append(
                    CollectResult(
                        exchange=exchange,
                        market_type=venue.market_type,
                        asset=asset.upper(),
                        timeframe=timeframe,
                        status="skipped",
                        error="asset or timeframe not listed on this venue",
                    )
                )
                continue
            result = collect_venue(exchange, asset, timeframe, start, end, pause=pause, refresh=refresh)
            results.append(result)
            time.sleep(0.05)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "requested": {
            "exchanges": exchanges,
            "assets": assets,
            "timeframe": timeframe,
            "start": start,
            "end": end,
        },
        "duration_seconds": (datetime.now(timezone.utc) - started).total_seconds(),
        "summary": {
            "total": len(results),
            "ok": sum(1 for r in results if r.status == "ok"),
            "skipped": sum(1 for r in results if r.status == "skipped"),
            "empty": sum(1 for r in results if r.status == "empty"),
            "failed": sum(1 for r in results if r.status == "failed"),
            "rows": int(sum(r.rows for r in results)),
        },
        "results": [r.to_dict() for r in results],
    }
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    name = manifest_name or f"collect_{timeframe}_{started.strftime('%Y%m%dT%H%M%SZ')}.json"
    path = MANIFEST_DIR / name
    if manifest_name and path.exists():
        manifest["results"] = _merge_results(
            json.loads(path.read_text(encoding="utf-8")).get("results", []), manifest["results"]
        )
        manifest["summary"] = _summarise(manifest["results"])
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    LOGGER.info("manifest written: %s", MANIFEST_DIR / name)
    return manifest
