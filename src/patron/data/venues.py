"""Public REST adapters that normalise exchange klines to one canonical schema.

Each adapter pages the venue's public endpoint forward/backward in time and
returns the same columns:

``timestamp_open, timestamp_close, open, high, low, close, volume,
quote_volume, trade_count``

Auxiliary columns a venue does not publish stay ``NaN`` — the venue-aware
validator reports them as unavailable rather than inventing numbers. Only closed
candles are returned: the adapter drops any bar whose close time is in the future.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

from patron.data.markets import CANONICAL_COLUMNS, TIMEFRAME_MINUTES, Venue

LOGGER = logging.getLogger(__name__)

USER_AGENT = "patron-research/2.0 (+public market data collection)"
MAX_RETRIES = 4
RETRY_BACKOFF = 1.8


class VenueError(RuntimeError):
    """Raised when a venue rejects a request or returns an unusable payload."""


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
    return session


def _get_json(
    session: requests.Session, url: str, params: dict, pause: float
) -> dict:
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            response = session.get(url, params=params, timeout=30)
            if response.status_code == 429 or response.status_code >= 500:
                raise VenueError(f"HTTP {response.status_code}: {response.text[:200]}")
            if response.status_code != 200:
                raise VenueError(f"HTTP {response.status_code}: {response.text[:200]}")
            time.sleep(pause)
            return response.json()
        except (requests.RequestException, ValueError, VenueError) as exc:
            last_error = exc
            if isinstance(exc, (requests.ConnectionError, requests.exceptions.ChunkedEncodingError)):
                # A reset connection can poison the keep-alive pool: drop it so the
                # retry opens a fresh socket instead of replaying the broken one.
                try:
                    session.close()
                except Exception:  # pragma: no cover - best effort
                    pass
            backoff = RETRY_BACKOFF**attempt
            LOGGER.warning("request failed (%s), retrying in %.1fs", exc, backoff)
            time.sleep(backoff)
    raise VenueError(f"request failed after {MAX_RETRIES} attempts: {last_error}")


def _frame(rows: list[dict], timeframe: str, venue: Venue, symbol: str) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=list(CANONICAL_COLUMNS))
    if frame.empty:
        return frame
    frame["timestamp_open"] = pd.to_datetime(frame["timestamp_open"], utc=True)
    frame["timestamp_close"] = pd.to_datetime(frame["timestamp_close"], utc=True)
    frame = frame.drop_duplicates(subset=["timestamp_open"], keep="last")
    frame = frame.sort_values("timestamp_open").reset_index(drop=True)
    now = pd.Timestamp(datetime.now(timezone.utc))
    frame = frame[frame["timestamp_close"] <= now].reset_index(drop=True)
    for column in ("open", "high", "low", "close", "volume", "quote_volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
    frame["trade_count"] = pd.to_numeric(frame["trade_count"], errors="coerce").astype("float64")
    frame["timeframe"] = timeframe
    frame["exchange"] = venue.slug
    frame["market_type"] = venue.market_type
    frame["symbol"] = symbol
    return frame


def _row(ts_ms: int, tf: str, o, h, l, c, volume, quote_volume=None, trade_count=None) -> dict:
    open_ts = pd.Timestamp(ts_ms, unit="ms", tz="UTC")
    close_ts = open_ts + pd.Timedelta(minutes=TIMEFRAME_MINUTES[tf])
    return {
        "timestamp_open": open_ts,
        "timestamp_close": close_ts,
        "open": o,
        "high": h,
        "low": l,
        "close": c,
        "volume": volume,
        "quote_volume": quote_volume,
        "trade_count": trade_count,
    }


def _ms(ts: pd.Timestamp) -> int:
    return int(ts.timestamp() * 1000)


def _fetch_binance(session, venue, symbol, timeframe, start, end, pause) -> list[dict]:
    rows: list[dict] = []
    cursor = start
    limit = venue.max_limit
    while cursor < end:
        payload = _get_json(
            session,
            f"{venue.base_url}/api/v3/klines",
            {
                "symbol": symbol,
                "interval": venue.interval(timeframe),
                "startTime": _ms(cursor),
                "endTime": _ms(end),
                "limit": limit,
            },
            pause,
        )
        if not payload:
            break
        for item in payload:
            rows.append(
                _row(
                    int(item[0]),
                    timeframe,
                    float(item[1]),
                    float(item[2]),
                    float(item[3]),
                    float(item[4]),
                    float(item[5]),
                    float(item[7]) if item[7] is not None else None,
                    float(item[8]) if item[8] is not None else None,
                )
            )
        last_open = pd.Timestamp(int(payload[-1][0]), unit="ms", tz="UTC")
        next_cursor = last_open + pd.Timedelta(minutes=TIMEFRAME_MINUTES[timeframe])
        if next_cursor <= cursor:
            break
        cursor = next_cursor
    return rows


def _fetch_bybit(session, venue, symbol, timeframe, start, end, pause) -> list[dict]:
    rows: list[dict] = []
    # Bybit returns the newest `limit` bars ending at `end` and ignores `start`
    # when the window is wider than `limit`, so page backwards from the end.
    cursor = end
    while cursor > start:
        payload = _get_json(
            session,
            f"{venue.base_url}/v5/market/kline",
            {
                "category": venue.market_type,
                "symbol": symbol,
                "interval": venue.interval(timeframe),
                "end": _ms(cursor),
                "limit": venue.max_limit,
            },
            pause,
        )
        if payload.get("retCode") not in (0, None):
            raise VenueError(f"bybit retCode {payload.get('retCode')}: {payload.get('retMsg')}")
        batch = (payload.get("result") or {}).get("list") or []
        if not batch:
            break
        batch = sorted(batch, key=lambda r: int(r[0]))
        for item in batch:
            rows.append(
                _row(
                    int(item[0]),
                    timeframe,
                    float(item[1]),
                    float(item[2]),
                    float(item[3]),
                    float(item[4]),
                    float(item[5]),
                    float(item[6]) if len(item) > 6 else None,
                )
            )
        oldest = pd.Timestamp(int(batch[0][0]), unit="ms", tz="UTC")
        if oldest >= cursor:
            break
        cursor = oldest
    return rows


def _fetch_okx(session, venue, symbol, timeframe, start, end, pause) -> list[dict]:
    rows: list[dict] = []
    cursor = end
    while cursor > start:
        payload = _get_json(
            session,
            f"{venue.base_url}/api/v5/market/history-candles",
            {
                "instId": symbol,
                "bar": venue.interval(timeframe),
                "after": _ms(cursor),
                "limit": venue.max_limit,
            },
            pause,
        )
        if payload.get("code") not in ("0", 0, None):
            raise VenueError(f"okx code {payload.get('code')}: {payload.get('msg')}")
        batch = payload.get("data") or []
        if not batch:
            break
        batch = sorted(batch, key=lambda r: int(r[0]))
        for item in batch:
            rows.append(
                _row(
                    int(item[0]),
                    timeframe,
                    float(item[1]),
                    float(item[2]),
                    float(item[3]),
                    float(item[4]),
                    float(item[5]),
                    float(item[7]) if len(item) > 7 and item[7] else None,
                )
            )
        oldest = pd.Timestamp(int(batch[0][0]), unit="ms", tz="UTC")
        if oldest >= cursor:
            break
        cursor = oldest
    return rows


def _fetch_coinbase(session, venue, symbol, timeframe, start, end, pause) -> list[dict]:
    rows: list[dict] = []
    step = pd.Timedelta(minutes=TIMEFRAME_MINUTES[timeframe] * venue.max_limit)
    cursor = start
    while cursor < end:
        window_end = min(cursor + step, end)
        payload = _get_json(
            session,
            f"{venue.base_url}/products/{symbol}/candles",
            {
                "granularity": venue.interval(timeframe),
                "start": cursor.isoformat().replace("+00:00", "Z"),
                "end": window_end.isoformat().replace("+00:00", "Z"),
            },
            pause,
        )
        if not isinstance(payload, list):
            raise VenueError(f"coinbase unexpected payload: {str(payload)[:200]}")
        for item in payload:
            # [time, low, high, open, close, volume]
            rows.append(
                _row(int(item[0]) * 1000, timeframe, item[3], item[2], item[1], item[4], item[5])
            )
        cursor = window_end
    return rows


def _fetch_kraken(session, venue, symbol, timeframe, start, end, pause) -> list[dict]:
    rows: list[dict] = []
    cursor = start
    while cursor < end:
        payload = _get_json(
            session,
            f"{venue.base_url}/0/public/OHLC",
            {
                "pair": symbol,
                "interval": venue.interval(timeframe),
                "since": int(cursor.timestamp()),
            },
            pause,
        )
        if payload.get("error"):
            raise VenueError(f"kraken error: {payload['error']}")
        result = payload.get("result") or {}
        batch = next((v for k, v in result.items() if k != "last" and isinstance(v, list)), [])
        if not batch:
            break
        batch = sorted(batch, key=lambda r: int(r[0]))
        for item in batch:
            rows.append(
                _row(
                    int(item[0]) * 1000,
                    timeframe,
                    float(item[1]),
                    float(item[2]),
                    float(item[3]),
                    float(item[4]),
                    float(item[6]),
                    trade_count=float(item[7]) if len(item) > 7 else None,
                )
            )
        last_open = pd.Timestamp(int(batch[-1][0]), unit="s", tz="UTC")
        next_cursor = last_open + pd.Timedelta(minutes=TIMEFRAME_MINUTES[timeframe])
        if next_cursor <= cursor:
            break
        cursor = next_cursor
    return rows


def _fetch_bitget(session, venue, symbol, timeframe, start, end, pause) -> list[dict]:
    rows: list[dict] = []
    cursor = start
    step = pd.Timedelta(minutes=TIMEFRAME_MINUTES[timeframe] * venue.max_limit)
    while cursor < end:
        # Bitget ignores startTime when the window is far in the future and just
        # returns the newest `limit` rows, so page in bounded windows instead.
        window_end = min(cursor + step, end)
        payload = _get_json(
            session,
            f"{venue.base_url}/api/v2/spot/market/candles",
            {
                "symbol": symbol,
                "granularity": venue.interval(timeframe),
                "startTime": _ms(cursor),
                "endTime": _ms(window_end),
                "limit": venue.max_limit,
            },
            pause,
        )
        if str(payload.get("code")) not in ("00000", "0", "None"):
            raise VenueError(f"bitget code {payload.get('code')}: {payload.get('msg')}")
        batch = payload.get("data") or []
        if not batch:
            break
        batch = sorted(batch, key=lambda r: int(r[0]))
        for item in batch:
            rows.append(
                _row(
                    int(item[0]),
                    timeframe,
                    float(item[1]),
                    float(item[2]),
                    float(item[3]),
                    float(item[4]),
                    float(item[5]),
                    float(item[6]) if len(item) > 6 and item[6] else None,
                )
            )
        cursor = window_end
    return rows


_FETCHERS = {
    "binance": _fetch_binance,
    "bybit": _fetch_bybit,
    "okx": _fetch_okx,
    "coinbase": _fetch_coinbase,
    "kraken": _fetch_kraken,
    "bitget": _fetch_bitget,
}


def fetch_candles(
    venue: Venue,
    base_asset: str,
    timeframe: str,
    start: pd.Timestamp | str,
    end: pd.Timestamp | str,
    pause: float | None = None,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Download closed candles for one venue/asset/timeframe range.

    Returns the canonical schema even when the venue returns nothing (empty
    frame with the right columns). Raises :class:`VenueError` on a rejected
    request so callers can record the failure instead of fabricating rows.
    """
    symbol = venue.venue_symbol(base_asset)
    if not venue.supports(timeframe):
        raise VenueError(f"{venue.slug} does not serve {timeframe}")
    start_ts = pd.Timestamp(start, tz="UTC") if pd.Timestamp(start).tzinfo is None else pd.Timestamp(start).tz_convert("UTC")
    end_ts = pd.Timestamp(end, tz="UTC") if pd.Timestamp(end).tzinfo is None else pd.Timestamp(end).tz_convert("UTC")
    # Align the request start to the timeframe grid.
    tf = pd.Timedelta(minutes=TIMEFRAME_MINUTES[timeframe])
    start_ts = start_ts.floor(tf)
    if end_ts <= start_ts:
        return pd.DataFrame(columns=[*CANONICAL_COLUMNS, "timeframe", "exchange", "market_type", "symbol"])

    owns_session = session is None
    session = session or _session()
    try:
        rows = _FETCHERS[venue.slug](
            session, venue, symbol, timeframe, start_ts, end_ts, pause or venue.default_pause
        )
    finally:
        if owns_session:
            session.close()

    frame = _frame(rows, timeframe, venue, symbol)
    if frame.empty:
        return frame
    # Clip to the requested window; the last requested bar may not be closed yet.
    frame = frame[(frame["timestamp_open"] >= start_ts) & (frame["timestamp_open"] < end_ts)]
    LOGGER.info(
        "%s %s %s: %d candles %s -> %s",
        venue.slug,
        symbol,
        timeframe,
        len(frame),
        frame["timestamp_open"].iloc[0] if len(frame) else None,
        frame["timestamp_open"].iloc[-1] if len(frame) else None,
    )
    return frame.reset_index(drop=True)


def canonicalise_external(
    rows: list[dict],
    venue: Venue,
    base_asset: str,
    timeframe: str,
) -> pd.DataFrame:
    """Canonicalise already-fetched rows (used by tests and offline imports)."""
    frame = _frame(rows, timeframe, venue, venue.venue_symbol(base_asset))
    if frame.empty:
        return frame
    return frame.replace({np.nan: np.nan}).reset_index(drop=True)
