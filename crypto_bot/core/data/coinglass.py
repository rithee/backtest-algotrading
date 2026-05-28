"""
CoinGlass liquidation data fetcher.
Requires COINGLASS_API_KEY environment variable.
Degrades gracefully to None if key missing or API unavailable.
"""
from __future__ import annotations
import os
import warnings
from pathlib import Path

import pandas as pd
import requests

CACHE_DIR = Path("data/external/coinglass")
_BASE_URL = "https://open-api.coinglass.com/public/v2"


def fetch_liquidations(
    symbol: str,
    start: str,
    end: str,
) -> pd.DataFrame | None:
    """
    Fetch liquidation history for a symbol.

    Args:
        symbol: e.g. 'BTCUSDT' (USDT suffix is stripped for the API)
        start:  'YYYY-MM-DD'
        end:    'YYYY-MM-DD'

    Returns:
        DataFrame indexed by datetime with columns: longLiquidationUsd, shortLiquidationUsd.
        Returns None if API key missing or fetch fails.
    """
    api_key = os.getenv("COINGLASS_API_KEY")
    if not api_key:
        warnings.warn("COINGLASS_API_KEY not set — skipping liquidation data")
        return None

    coin = symbol.replace("USDT", "").replace("BUSD", "")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{coin}_liquidations_{start}_{end}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    try:
        resp = requests.get(
            f"{_BASE_URL}/liquidation_history",
            headers={"coinglassSecret": api_key},
            params={"symbol": coin, "interval": "4h", "limit": 2000},
            timeout=30,
        )
        resp.raise_for_status()
        payload = resp.json()
        rows = payload.get("data", [])
        if not rows:
            return None

        df = pd.DataFrame(rows)
        df["createTime"] = pd.to_datetime(df["createTime"], unit="ms")
        df = df.set_index("createTime")
        df.index.name = "timestamp"
        df = df[start:end]
        df.to_parquet(cache_path)
        return df
    except Exception as exc:
        warnings.warn(f"CoinGlass fetch failed for {symbol}: {exc}")
        return None
