"""
Deribit options data fetcher: instruments OI, max pain, gamma exposure.
Uses Deribit's public REST API — no authentication required for market data.

Historical limitation: Deribit's /get_instruments returns current/live instruments only.
For live/paper trading this works correctly. For historical backtests, strategies
that use this data should detect None and fall back to technical-only signals.
"""
from __future__ import annotations
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

CACHE_DIR = Path("data/external/deribit")
_BASE_URL = "https://www.deribit.com/api/v2/public"
_WEEK_MS = 7 * 24 * 60 * 60 * 1000


def fetch_options_summary(symbol: str, date: str) -> pd.DataFrame | None:
    """
    Fetch options OI by strike for the nearest weekly expiry to `date`.

    Args:
        symbol: 'BTC' or 'ETH'
        date:   'YYYY-MM-DD'

    Returns:
        DataFrame with columns: strike, option_type ('C'/'P'), open_interest, expiry.
        Returns None if fetch fails.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{symbol}_options_{date}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    try:
        resp = requests.get(
            f"{_BASE_URL}/get_instruments",
            params={"currency": symbol, "kind": "option", "expired": "false"},
            timeout=30,
        )
        resp.raise_for_status()
        instruments = resp.json().get("result", [])

        target_ts = int(datetime.strptime(date, "%Y-%m-%d").timestamp() * 1000)
        nearby = [
            inst for inst in instruments
            if abs(inst.get("expiration_timestamp", 0) - target_ts) <= _WEEK_MS
        ]
        if not nearby:
            return None

        rows = []
        for inst in nearby[:50]:   # cap to avoid rate limits
            name = inst["instrument_name"]
            ob_resp = requests.get(
                f"{_BASE_URL}/get_order_book",
                params={"instrument_name": name, "depth": 1},
                timeout=10,
            )
            if ob_resp.ok:
                ob = ob_resp.json().get("result", {})
                rows.append({
                    "strike": float(inst["strike"]),
                    "option_type": "C" if name.endswith("-C") else "P",
                    "open_interest": float(ob.get("open_interest", 0)),
                    "expiry": inst["expiration_timestamp"],
                })

        if not rows:
            return None

        df = pd.DataFrame(rows)
        df.to_parquet(cache_path)
        return df
    except Exception as exc:
        warnings.warn(f"Deribit fetch failed for {symbol} on {date}: {exc}")
        return None


def compute_max_pain(options_df: pd.DataFrame | None) -> float | None:
    """
    Compute the Max Pain price: the strike where total dollar loss for ALL option
    holders is maximized (dealers profit most → price gravitates here pre-expiry).

    Args:
        options_df: DataFrame from fetch_options_summary

    Returns:
        Max Pain price as float, or None if data unavailable.
    """
    if options_df is None or options_df.empty:
        return None

    strikes = sorted(options_df["strike"].unique())
    pain: dict[float, float] = {}

    for s in strikes:
        # Call holders lose when price < strike
        call_rows = options_df[(options_df["option_type"] == "C") & (options_df["strike"] > s)]
        call_pain = float(((call_rows["strike"] - s) * call_rows["open_interest"]).sum())
        # Put holders lose when price > strike
        put_rows = options_df[(options_df["option_type"] == "P") & (options_df["strike"] < s)]
        put_pain = float(((s - put_rows["strike"]) * put_rows["open_interest"]).sum())
        pain[s] = call_pain + put_pain

    # Max pain = strike with MINIMUM total pain for option holders
    return min(pain, key=pain.get) if pain else None
