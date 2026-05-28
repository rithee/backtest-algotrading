"""
CoinMarketCap top-N coins by market cap.
Requires CMC_API_KEY environment variable.
Falls back to a hardcoded top-10 default if key is missing.
Caches monthly (same month = same cache file).
"""
from __future__ import annotations
import json
import os
import warnings
from pathlib import Path

import requests

CACHE_DIR = Path("data/external/coinmarketcap")
_CMC_URL = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"

_DEFAULT_TOP10 = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "ADAUSDT", "AVAXUSDT", "DOGEUSDT", "DOTUSDT", "MATICUSDT",
]


def fetch_top_n(n: int = 10, date: str | None = None) -> list[str]:
    """
    Return list of top-N symbols by market cap with USDT suffix (e.g. 'BTCUSDT').
    Falls back to hardcoded default list if CMC_API_KEY is not set or fetch fails.
    Cache key is monthly: two calls in the same calendar month return the same data.

    Args:
        n:    number of symbols to return
        date: 'YYYY-MM-DD' (used for monthly cache key; defaults to today)

    Returns:
        List of symbol strings e.g. ['BTCUSDT', 'ETHUSDT', ...]
    """
    api_key = os.getenv("CMC_API_KEY")
    if not api_key:
        warnings.warn("CMC_API_KEY not set — using default top-10 list")
        return _DEFAULT_TOP10[:n]

    if date is None:
        from datetime import date as dt_date
        date = dt_date.today().isoformat()

    month_key = date[:7]   # 'YYYY-MM'
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"top{n}_{month_key}.json"

    if cache_path.exists():
        return json.loads(cache_path.read_text())

    try:
        resp = requests.get(
            _CMC_URL,
            headers={"X-CMC_PRO_API_KEY": api_key},
            params={"limit": n, "convert": "USDT", "sort": "market_cap"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])
        symbols = [f"{coin['symbol']}USDT" for coin in data[:n]]
        cache_path.write_text(json.dumps(symbols))
        return symbols
    except Exception as exc:
        warnings.warn(f"CoinMarketCap fetch failed: {exc}. Using default list.")
        return _DEFAULT_TOP10[:n]
