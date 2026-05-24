"""
Binance Futures historical data fetcher with parquet caching.
Cache key: data/cache/<symbol>_<timeframe>_<start>_<end>.parquet
"""
from __future__ import annotations
import time
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from binance.client import Client
from crypto_bot.core.data.models import Candle, FundingRate

CACHE_DIR = Path("data/cache")
TIMEFRAME_MAP = {
    "1m": Client.KLINE_INTERVAL_1MINUTE,
    "5m": Client.KLINE_INTERVAL_5MINUTE,
    "15m": Client.KLINE_INTERVAL_15MINUTE,
    "1h": Client.KLINE_INTERVAL_1HOUR,
    "4h": Client.KLINE_INTERVAL_4HOUR,
    "1d": Client.KLINE_INTERVAL_1DAY,
}


def _cache_path(symbol: str, timeframe: str, start: str, end: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{symbol}_{timeframe}_{start}_{end}.parquet"


def fetch_candles(
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
    api_key: str = "",
    api_secret: str = "",
) -> pd.DataFrame:
    """
    Fetch OHLCV candles for a symbol. Returns DataFrame with columns:
    symbol, timestamp, open, high, low, close, volume, is_clean.
    Uses disk cache — re-fetches only if cache file is absent.
    """
    cache = _cache_path(symbol, timeframe, start_date, end_date)
    if cache.exists():
        df = pd.read_parquet(cache)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    client = Client(api_key, api_secret)
    interval = TIMEFRAME_MAP.get(timeframe, Client.KLINE_INTERVAL_4HOUR)

    klines = client.futures_klines(
        symbol=symbol,
        interval=interval,
        startTime=_to_ms(start_date),
        endTime=_to_ms(end_date),
        limit=1500,
    )

    # Paginate if needed
    while True:
        if len(klines) == 0:
            break
        last_ts = klines[-1][0]
        end_ms = _to_ms(end_date)
        if last_ts >= end_ms:
            break
        next_batch = client.futures_klines(
            symbol=symbol,
            interval=interval,
            startTime=last_ts + 1,
            endTime=end_ms,
            limit=1500,
        )
        if not next_batch:
            break
        klines.extend(next_batch)
        time.sleep(0.1)

    records = []
    for k in klines:
        ts = datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc).replace(tzinfo=None)
        records.append({
            "symbol": symbol,
            "timestamp": ts,
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5]),
            "is_clean": True,
        })

    df = pd.DataFrame(records)
    df.to_parquet(cache, index=False)
    return df


def fetch_funding_rates(
    symbol: str,
    start_date: str,
    end_date: str,
    api_key: str = "",
    api_secret: str = "",
) -> pd.DataFrame:
    """
    Fetch funding rates for a symbol. Returns DataFrame with columns:
    symbol, timestamp, rate.
    Cached to data/cache/<symbol>_funding_<start>_<end>.parquet.
    """
    cache = CACHE_DIR / f"{symbol}_funding_{start_date}_{end_date}.parquet"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        df = pd.read_parquet(cache)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    client = Client(api_key, api_secret)
    rates = client.futures_funding_rate(
        symbol=symbol,
        startTime=_to_ms(start_date),
        endTime=_to_ms(end_date),
        limit=1000,
    )

    records = []
    for r in rates:
        ts = datetime.fromtimestamp(r["fundingTime"] / 1000, tz=timezone.utc).replace(tzinfo=None)
        records.append({
            "symbol": symbol,
            "timestamp": ts,
            "rate": float(r["fundingRate"]),
        })

    df = pd.DataFrame(records) if records else pd.DataFrame(columns=["symbol", "timestamp", "rate"])
    df.to_parquet(cache, index=False)
    return df


def _to_ms(date_str: str) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)
