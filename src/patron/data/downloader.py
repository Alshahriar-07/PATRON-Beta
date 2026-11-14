"""Download BTC/USDT OHLCV from Binance public data.

Two sources are combined:

* ``data.binance.vision`` monthly kline archives (fast, immutable, closed
  candles only) for complete months;
* the public REST ``/api/v3/klines`` endpoint for the current month / tail.

No synthetic candles are ever produced. Months that are unavailable in both
sources are reported as missing and left out of the dataset.
"""

from __future__ import annotations

import io
import logging
import time
import zipfile
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

from patron.config import PatronConfig
from patron.paths import RAW_DIR, symbol_slug
from patron.utils.io import write_parquet

LOGGER = logging.getLogger(__name__)

TIMEFRAME_MINUTES: dict[str, int] = {"15m": 15, "1h": 60, "4h": 240}

RAW_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trade_count",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore",
]

CANONICAL_COLUMNS = [
    "timestamp_open",
    "timestamp_close",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "trade_count",
]


@dataclass
class DownloadResult:
    timeframe: str
    frame: pd.DataFrame
    partitions: list[dict] = field(default_factory=list)
    missing_months: list[str] = field(default_factory=list)


def timeframe_delta(timeframe: str) -> pd.Timedelta:
    if timeframe not in TIMEFRAME_MINUTES:
        raise ValueError(f"unsupported timeframe: {timeframe}")
    return pd.Timedelta(minutes=TIMEFRAME_MINUTES[timeframe])


def iter_months(start: date, end: date) -> list[tuple[int, int]]:
    months: list[tuple[int, int]] = []
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        months.append((year, month))
        month += 1
        if month == 13:
            year, month = year + 1, 1
    return months


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": "patron-research/1.0"})
    return session


def _get_with_retries(
    session: requests.Session, url: str, *, params: dict | None = None, attempts: int = 4
) -> requests.Response | None:
    """GET with backoff. Returns None on 404 (data genuinely absent)."""
    delay = 1.0
    for attempt in range(attempts):
        try:
            resp = session.get(url, params=params, timeout=30)
        except requests.RequestException as exc:  # network hiccup
            LOGGER.warning("request failed (%s/%s): %s", attempt + 1, attempts, exc)
            time.sleep(delay)
            delay *= 2
            continue
        if resp.status_code == 200:
            return resp
        if resp.status_code == 404:
            return None
        if resp.status_code in (418, 429, 500, 502, 503, 504):
            LOGGER.warning("HTTP %s for %s, retrying", resp.status_code, url)
            time.sleep(delay)
            delay *= 2
            continue
        resp.raise_for_status()
    return None


def _standardize(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert Binance kline columns into PATRON's canonical candle schema."""
    df = raw.copy()
    if len(df.columns) > len(RAW_COLUMNS):
        df = df.iloc[:, : len(RAW_COLUMNS)]
    if len(df.columns) < len(RAW_COLUMNS):
        # older archives ship 12 columns but with an extra ignored column
        df = df.reindex(columns=RAW_COLUMNS[: len(df.columns)])
        df.columns = RAW_COLUMNS[: len(df.columns)]
    else:
        df.columns = RAW_COLUMNS

    for col in ("open", "high", "low", "close", "volume", "quote_volume", "trade_count"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce").astype("int64")
    # Binance archives use either ms or (rarely) microseconds; normalise to ms.
    max_ts = int(df["open_time"].max()) if len(df) else 0
    if max_ts > 10**14:
        df["open_time"] = df["open_time"] // 1000

    out = pd.DataFrame(
        {
            "timestamp_open": pd.to_datetime(df["open_time"], unit="ms", utc=True),
            "open": df["open"].astype("float64"),
            "high": df["high"].astype("float64"),
            "low": df["low"].astype("float64"),
            "close": df["close"].astype("float64"),
            "volume": df["volume"].astype("float64"),
            "quote_volume": df["quote_volume"].astype("float64"),
            "trade_count": df["trade_count"].fillna(0).astype("int64"),
        }
    )
    return out


def _read_zip_csv(content: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise ValueError("archive contains no CSV")
        with archive.open(names[0]) as fh:
            probe = pd.read_csv(fh, header=None, nrows=1)
    has_header = str(probe.iloc[0, 0]).strip().lower().startswith("open")
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        with archive.open(names[0]) as fh:
            return pd.read_csv(fh, header=0 if has_header else None)


def download_month_vision(
    session: requests.Session,
    cfg: PatronConfig,
    symbol: str,
    timeframe: str,
    year: int,
    month: int,
) -> pd.DataFrame | None:
    """Download one monthly archive. Returns None when the month does not exist."""
    url = (
        f"{cfg.data.base_url}/data/spot/monthly/klines/{symbol}/{timeframe}/"
        f"{symbol}-{timeframe}-{year}-{month:02d}.zip"
    )
    resp = _get_with_retries(session, url)
    if resp is None:
        LOGGER.warning("monthly archive missing: %s-%02d %s", year, month, timeframe)
        return None
    raw = _read_zip_csv(resp.content)
    if raw.empty:
        return None
    return _standardize(raw)


def download_range_rest(
    session: requests.Session,
    cfg: PatronConfig,
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """Page through the REST kline endpoint for ``[start, end)``."""
    delta = timeframe_delta(timeframe)
    limit = 1000
    cursor = start
    frames: list[pd.DataFrame] = []
    url = f"{cfg.data.rest_base_url}/api/v3/klines"

    while cursor < end:
        params = {
            "symbol": symbol,
            "interval": timeframe,
            "startTime": int(cursor.timestamp() * 1000),
            "endTime": int(end.timestamp() * 1000),
            "limit": limit,
        }
        resp = _get_with_retries(session, url, params=params)
        if resp is None:
            LOGGER.warning("REST returned no data for %s from %s", timeframe, cursor)
            break
        rows = resp.json()
        if not rows:
            break
        frame = _standardize(pd.DataFrame(rows))
        frames.append(frame)
        last_open = frame["timestamp_open"].iloc[-1]
        if len(rows) < limit:
            break
        cursor = (last_open + delta).to_pydatetime()
        time.sleep(cfg.data.request_pause_seconds)

    if not frames:
        return pd.DataFrame(columns=list(CANONICAL_COLUMNS))
    return pd.concat(frames, ignore_index=True)


def download_timeframe(
    cfg: PatronConfig,
    timeframe: str,
    start: date | None = None,
    end: date | None = None,
    raw_root: Path | None = None,
    overwrite: bool | None = None,
    verbose: bool = True,
) -> DownloadResult:
    """Download (or reuse) all raw partitions for a timeframe and return them."""
    symbol = symbol_slug(cfg.symbol)
    start_date = start or datetime.fromisoformat(cfg.data.start).date()
    now = datetime.now(timezone.utc)
    end_date = end or (datetime.fromisoformat(cfg.data.end).date() if cfg.data.end else now.date())
    delta = timeframe_delta(timeframe)

    root = raw_root or RAW_DIR
    out_dir = Path(root) / symbol / timeframe
    out_dir.mkdir(parents=True, exist_ok=True)
    overwrite = cfg.data.raw_overwrite if overwrite is None else overwrite

    session = _session()
    frames: list[pd.DataFrame] = []
    partitions: list[dict] = []
    missing_months: list[str] = []
    current_month = (now.year, now.month)

    for year, month in iter_months(start_date, end_date):
        path = out_dir / f"{year}-{month:02d}.parquet"
        is_current = (year, month) >= current_month
        if path.exists() and not overwrite and not is_current:
            frame = pd.read_parquet(path, engine="pyarrow")
            source = "cache"
        elif is_current:
            month_start = datetime(year, month, 1, tzinfo=timezone.utc)
            frame = download_range_rest(session, cfg, symbol, timeframe, month_start, now)
            source = "rest"
            if frame.empty:
                missing_months.append(f"{year}-{month:02d}")
                continue
            write_parquet(frame, path)
        else:
            frame = download_month_vision(session, cfg, symbol, timeframe, year, month)
            source = "vision"
            if frame is None or frame.empty:
                # fall back to REST for this month before declaring it missing
                month_start = datetime(year, month, 1, tzinfo=timezone.utc)
                next_month = (month_start + pd.Timedelta(days=32)).replace(day=1)
                frame = download_range_rest(
                    session, cfg, symbol, timeframe, month_start, min(next_month, now)
                )
                source = "rest_fallback"
            if frame is None or frame.empty:
                missing_months.append(f"{year}-{month:02d}")
                continue
            write_parquet(frame, path)

        frame = frame.copy()
        frame["timestamp_close"] = frame["timestamp_open"] + delta
        frames.append(frame)
        partitions.append(
            {
                "month": f"{year}-{month:02d}",
                "source": source,
                "rows": int(len(frame)),
                "path": str(path),
            }
        )
        if verbose:
            LOGGER.info("%s %s-%02d: %s rows (%s)", timeframe, year, month, len(frame), source)

    if not frames:
        return DownloadResult(timeframe, pd.DataFrame(columns=list(CANONICAL_COLUMNS)), [], missing_months)

    combined = pd.concat(frames, ignore_index=True)
    combined = (
        combined.drop_duplicates(subset=["timestamp_open"], keep="last")
        .sort_values("timestamp_open")
        .reset_index(drop=True)
    )
    # Only closed candles: a candle is closed at timestamp_close (exclusive end).
    combined = combined[combined["timestamp_close"] <= pd.Timestamp(now)].reset_index(drop=True)
    combined = combined[combined["timestamp_open"] >= pd.Timestamp(start_date, tz="UTC")]
    return DownloadResult(timeframe, combined[list(CANONICAL_COLUMNS)], partitions, missing_months)
