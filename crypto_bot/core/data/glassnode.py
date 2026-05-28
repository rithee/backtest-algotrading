"""
Glassnode on-chain metrics fetcher.
Requires GLASSNODE_API_KEY environment variable.
Degrades gracefully to None if key missing or API unavailable.
"""
from __future__ import annotations
import os
import warnings
from pathlib import Path

import pandas as pd
import requests

CACHE_DIR = Path("data/external/glassnode")
_BASE_URL = "https://api.glassnode.com/v1/metrics"

_METRIC_ENDPOINTS: dict[str, str] = {
    "mvrv":             "/market/mvrv",
    "mvrv_zscore":      "/market/mvrv_z_score",
    "sopr":             "/indicators/sopr",
    "nupl":             "/indicators/nupl",
    "nvt":              "/indicators/nvt",
    "exchange_balance": "/distribution/balance_exchanges",
    "lth_supply":       "/supply/lth_supply",
    "miner_outflow":    "/mining/hash_rate_mean",
}


def fetch_glassnode(
    metric: str,
    symbol: str,
    start: str,
    end: str,
) -> pd.DataFrame | None:
    """
    Fetch a single on-chain metric for one symbol.

    Args:
        metric: one of the keys in _METRIC_ENDPOINTS
        symbol: 'BTC', 'ETH', etc. (no USDT suffix)
        start:  'YYYY-MM-DD'
        end:    'YYYY-MM-DD'

    Returns:
        DataFrame with DatetimeIndex and one column named `metric`.
        Returns None if API key missing or fetch fails.
    """
    api_key = os.getenv("GLASSNODE_API_KEY")
    if not api_key:
        warnings.warn("GLASSNODE_API_KEY not set — skipping on-chain data")
        return None

    endpoint = _METRIC_ENDPOINTS.get(metric)
    if not endpoint:
        warnings.warn(f"Unknown glassnode metric: {metric!r}. Valid: {list(_METRIC_ENDPOINTS)}")
        return None

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{symbol}_{metric}_{start}_{end}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    try:
        resp = requests.get(
            f"{_BASE_URL}{endpoint}",
            params={
                "a": symbol,
                "api_key": api_key,
                "s": start,
                "u": end,
                "i": "24h",
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return None
        df = pd.DataFrame(data)
        df["t"] = pd.to_datetime(df["t"], unit="s")
        df = df.set_index("t").rename(columns={"v": metric})
        df.index.name = "timestamp"
        df.to_parquet(cache_path)
        return df
    except Exception as exc:
        warnings.warn(f"Glassnode fetch failed for {metric}/{symbol}: {exc}")
        return None
