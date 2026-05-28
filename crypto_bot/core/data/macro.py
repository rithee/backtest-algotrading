"""
Macro market data fetcher: SPX, DXY, Gold (yfinance) + 10Y real yield (FRED via fredapi).
All data cached as Parquet. Missing sources degrade gracefully.
"""
from __future__ import annotations
import os
import warnings
from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE_DIR = Path("data/external/macro")

_YFINANCE_TICKERS: dict[str, str] = {
    "spx": "^GSPC",
    "dxy": "DX-Y.NYB",
    "gold": "GC=F",
}
_FRED_SERIES = "DFII10"   # 10Y Treasury Inflation-Indexed Security (real yield)


def fetch_macro(start: str, end: str) -> pd.DataFrame | None:
    """
    Returns DataFrame with columns: spx, dxy, gold, real_yield_10y.
    Index: DatetimeIndex (daily). Forward-fills gaps. Returns None if all sources fail.
    Caches result to CACHE_DIR/macro_{start}_{end}.parquet.

    Args:
        start: 'YYYY-MM-DD'
        end:   'YYYY-MM-DD'
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"macro_{start}_{end}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    frames: dict[str, pd.Series] = {}

    # yfinance: SPX, DXY, Gold
    for col_name, ticker in _YFINANCE_TICKERS.items():
        try:
            raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if not raw.empty:
                close = raw["Close"]
                if isinstance(close, pd.DataFrame):
                    close = close.squeeze()
                frames[col_name] = close.rename(col_name)
        except Exception as exc:
            warnings.warn(f"yfinance fetch failed for {ticker}: {exc}")

    # FRED: 10Y real yield via fredapi (requires FRED_API_KEY env var)
    fred_api_key = os.getenv("FRED_API_KEY")
    if fred_api_key:
        try:
            import fredapi  # type: ignore
            fred = fredapi.Fred(api_key=fred_api_key)
            series = fred.get_series(_FRED_SERIES, observation_start=start, observation_end=end)
            if series is not None and len(series) > 0:
                frames["real_yield_10y"] = series.rename("real_yield_10y")
        except Exception as exc:
            warnings.warn(f"FRED fetch failed ({_FRED_SERIES}): {exc}")
    else:
        warnings.warn("FRED_API_KEY not set — skipping real yield data")

    if not frames:
        return None

    result = pd.DataFrame(frames)
    result.index = pd.to_datetime(result.index)
    result = result.ffill().dropna(how="all")
    result.to_parquet(cache_path)
    return result
