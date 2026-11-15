"""Venue registry for multi-exchange market data.

Every venue is described declaratively so collection, validation and symbol
mapping are testable without network access. Symbols are mapped per venue
explicitly — ``BTC/USDT`` never silently becomes another pair.

Licence note: each entry records the public-data endpoint and any known coverage
restriction. Downloading implies acceptance of that venue's terms of use.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Canonical assets tracked in the multi-asset dataset (BTC is the target).
AUXILIARY_ASSETS: tuple[str, ...] = (
    "BTC",
    "ETH",
    "SOL",
    "BNB",
    "XRP",
    "ADA",
    "DOGE",
    "AVAX",
    "LINK",
    "LTC",
)

#: Supported canonical timeframes (minutes).
TIMEFRAME_MINUTES: dict[str, int] = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "4h": 240,
    "1d": 1440,
}

#: Canonical candle columns produced by every adapter. Auxiliary columns are
#: nullable: a venue that does not publish them leaves NaN and the venue-aware
#: validator records that instead of inventing values.
CANONICAL_COLUMNS: tuple[str, ...] = (
    "timestamp_open",
    "timestamp_close",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "trade_count",
)


@dataclass(frozen=True)
class Venue:
    """Declarative description of a public market-data venue."""

    slug: str
    name: str
    market_type: str
    quote: str
    base_url: str
    max_limit: int
    symbol_template: str
    interval_map: dict[str, str]
    supports_history: bool = True
    provides_quote_volume: bool = True
    provides_trade_count: bool = True
    licence_note: str = ""
    #: canonical base asset -> venue-native symbol
    symbols: dict[str, str] = field(default_factory=dict)
    default_pause: float = 0.15
    #: Depth of the public history endpoint, in days (None = full history).
    #: Kraken's OHLC endpoint returns at most 720 bars and Bitget's v2 spot
    #: candles only reach back ~10 days for 15m, so the collector clamps to this.
    max_history_days: int | None = None

    def venue_symbol(self, base_asset: str) -> str:
        """Map a canonical asset to this venue's symbol, failing loudly."""
        try:
            return self.symbols[base_asset.upper()]
        except KeyError as exc:
            raise KeyError(
                f"{self.slug} has no mapping for asset {base_asset!r}; "
                f"known: {sorted(self.symbols)}"
            ) from exc

    def interval(self, timeframe: str) -> str:
        try:
            return self.interval_map[timeframe]
        except KeyError as exc:
            raise KeyError(
                f"{self.slug} does not support timeframe {timeframe!r}; "
                f"supported: {sorted(self.interval_map)}"
            ) from exc

    def supports(self, timeframe: str) -> bool:
        return timeframe in self.interval_map


BINANCE = Venue(
    slug="binance",
    name="Binance",
    market_type="spot",
    quote="USDT",
    base_url="https://api.binance.com",
    max_limit=1000,
    symbol_template="{base}{quote}",
    interval_map={"1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m", "1h": "1h", "4h": "4h", "1d": "1d"},
    licence_note="Public REST /api/v3/klines and data.binance.vision archives; no API key required.",
    symbols={asset: f"{asset}USDT" for asset in AUXILIARY_ASSETS},
    default_pause=0.15,
)

BYBIT = Venue(
    slug="bybit",
    name="Bybit",
    market_type="spot",
    quote="USDT",
    base_url="https://api.bybit.com",
    max_limit=1000,
    symbol_template="{base}{quote}",
    interval_map={"1m": "1", "5m": "5", "15m": "15", "30m": "30", "1h": "60", "4h": "240", "1d": "D"},
    licence_note="Public v5 market kline endpoint; 1000 candles per request, paged by start/end.",
    symbols={asset: f"{asset}USDT" for asset in AUXILIARY_ASSETS},
    default_pause=0.12,
)

OKX = Venue(
    slug="okx",
    name="OKX",
    market_type="spot",
    quote="USDT",
    base_url="https://www.okx.com",
    max_limit=300,
    symbol_template="{base}-{quote}",
    interval_map={"1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m", "1h": "1H", "4h": "4H", "1d": "1D"},
    licence_note="Public v5 market history-candles; 300 per request, paged backwards via `after`.",
    symbols={asset: f"{asset}-USDT" for asset in AUXILIARY_ASSETS},
    default_pause=0.11,
)

COINBASE = Venue(
    slug="coinbase",
    name="Coinbase",
    market_type="spot",
    quote="USD",  # USD, not USDT — recorded explicitly, never treated as USDT
    base_url="https://api.exchange.coinbase.com",
    max_limit=300,
    symbol_template="{base}-{quote}",
    interval_map={"1m": "60", "5m": "300", "15m": "900", "1h": "3600", "1d": "86400"},
    provides_quote_volume=False,
    provides_trade_count=False,
    licence_note="Public Exchange candles; 300 candles per request, USD-quoted so volume is base units.",
    symbols={
        "BTC": "BTC-USD",
        "ETH": "ETH-USD",
        "SOL": "SOL-USD",
        "XRP": "XRP-USD",
        "ADA": "ADA-USD",
        "DOGE": "DOGE-USD",
        "AVAX": "AVAX-USD",
        "LINK": "LINK-USD",
        "LTC": "LTC-USD",
    },
    default_pause=0.34,
)

KRAKEN = Venue(
    slug="kraken",
    name="Kraken",
    market_type="spot",
    quote="USD",
    base_url="https://api.kraken.com",
    max_limit=720,
    symbol_template="{base}{quote}",
    interval_map={"1m": "1", "5m": "5", "15m": "15", "30m": "30", "1h": "60", "4h": "240", "1d": "1440"},
    provides_quote_volume=False,
    provides_trade_count=False,
    licence_note="Public /0/public/OHLC; max 720 candles per request, USD-quoted, XBT legacy symbol for BTC.",
    max_history_days=7,
    symbols={
        "BTC": "XBTUSD",
        "ETH": "ETHUSD",
        "SOL": "SOLUSD",
        "XRP": "XRPUSD",
        "ADA": "ADAUSD",
        "DOGE": "XDGUSD",
        "AVAX": "AVAXUSD",
        "LINK": "LINKUSD",
        "LTC": "LTCUSD",
    },
    default_pause=1.1,
)

BITGET = Venue(
    slug="bitget",
    name="Bitget",
    market_type="spot",
    quote="USDT",
    base_url="https://api.bitget.com",
    max_limit=200,
    symbol_template="{base}{quote}",
    interval_map={"1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min", "1h": "1h", "4h": "4h", "1d": "1day"},
    provides_quote_volume=False,
    provides_trade_count=False,
    licence_note="Public v2 spot market candles; 200 per request, history limited to recent days.",
    max_history_days=10,
    symbols={asset: f"{asset}USDT" for asset in AUXILIARY_ASSETS},
    default_pause=0.2,
)

VENUES: dict[str, Venue] = {
    v.slug: v for v in (BINANCE, BYBIT, OKX, COINBASE, KRAKEN, BITGET)
}

#: Preferred ordering when presenting multi-venue coverage.
VENUE_ORDER: tuple[str, ...] = ("binance", "bybit", "okx", "coinbase", "kraken", "bitget")


def get_venue(slug: str) -> Venue:
    try:
        return VENUES[slug]
    except KeyError as exc:
        raise KeyError(f"unknown venue {slug!r}; known: {sorted(VENUES)}") from exc


def venues_for_asset(asset: str, timeframe: str = "15m") -> list[Venue]:
    """Venues that both list the asset and support the timeframe."""
    asset = asset.upper()
    return [
        VENUES[slug]
        for slug in VENUE_ORDER
        if asset in VENUES[slug].symbols and VENUES[slug].supports(timeframe)
    ]


def all_timeframes() -> tuple[str, ...]:
    return tuple(TIMEFRAME_MINUTES)
