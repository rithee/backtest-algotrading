"""
Central data loader for all backtest modes.
Builds the aux_data dict and loads multi-timeframe candles.
All sources degrade gracefully — a missing API key gives None, not a crash.
"""
from __future__ import annotations
from typing import Any

import pandas as pd

from crypto_bot.core.config import Config

# Import fetchers (each degrades gracefully on missing keys)
from .macro import fetch_macro
from .glassnode import fetch_glassnode
from .coinglass import fetch_liquidations
from .deribit import fetch_options_summary
from .coinmarketcap import fetch_top_n

# Existing Binance fetchers
from .binance_history import fetch_candles, fetch_open_interest, fetch_funding_rates

_GLASSNODE_SPOT_METRICS = [
    "mvrv", "mvrv_zscore", "sopr", "nupl", "nvt", "exchange_balance", "lth_supply",
]

# Binance OI default interval (4h aligns with swing/intraday primary timeframes)
_OI_DEFAULT_INTERVAL = "4h"


def load_aux_data(config: Config, mode: str) -> dict[str, Any]:
    """
    Build the aux_data dict for a given mode.
    Keys that fail or have no API key are set to None — strategies handle this gracefully.

    Args:
        config: loaded Config object
        mode:   'intraday' | 'swing' | 'swing_new' | 'spot_longterm'

    Returns dict with keys:
        open_interest:   {symbol: pd.Series | None}
        funding_rate:    {symbol: pd.DataFrame | None}
        liquidations:    {symbol: pd.DataFrame | None}  (intraday only)
        exchange_netflow:{symbol: pd.DataFrame | None}  (swing_new only)
        options_gamma:   {symbol: pd.DataFrame | None}  (swing_new only)
        macro:           pd.DataFrame | None             (swing_new + spot)
        onchain:         {f"{coin}_{metric}": df | None} (spot only)
        top10_mcap:      list[str]                        (spot only)
    """
    start = config.backtest.start_date
    end = config.backtest.end_date
    symbols = config.backtest.symbols

    aux: dict[str, Any] = {
        "open_interest": {
            s: fetch_open_interest(s, _OI_DEFAULT_INTERVAL, start, end)
            for s in symbols
        },
        "funding_rate": {
            s: fetch_funding_rates(s, start, end)
            for s in symbols
        },
    }

    if mode == "intraday":
        aux["liquidations"] = {s: fetch_liquidations(s, start, end) for s in symbols}

    if mode in ("swing", "swing_new", "spot_longterm"):
        aux["macro"] = fetch_macro(start, end)

    if mode == "swing_new":
        aux["exchange_netflow"] = {
            s: fetch_glassnode("exchange_balance", s.replace("USDT", ""), start, end)
            for s in symbols
        }
        aux["options_gamma"] = {
            s: fetch_options_summary(s.replace("USDT", ""), start)
            for s in symbols
        }

    if mode == "spot_longterm":
        onchain: dict[str, pd.DataFrame | None] = {}
        for s in symbols:
            coin = s.replace("USDT", "")
            for metric in _GLASSNODE_SPOT_METRICS:
                onchain[f"{coin}_{metric}"] = fetch_glassnode(metric, coin, start, end)
        aux["onchain"] = onchain
        aux["top10_mcap"] = fetch_top_n(10)
        aux["macro"] = fetch_macro(start, end)

    return aux


def load_mtf_candles(
    config: Config,
    symbols: list[str] | None = None,
) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Load candles for all timeframes specified in config.

    Args:
        config:  Config with backtest.active_timeframes and backtest.symbols
        symbols: override symbol list (defaults to config.backtest.symbols)

    Returns:
        {symbol: {timeframe: candle_df}}
        Missing fetches are silently omitted from the inner dict.
    """
    symbols = symbols or config.backtest.symbols
    timeframes = config.backtest.active_timeframes
    start = config.backtest.start_date
    end = config.backtest.end_date

    result: dict[str, dict[str, pd.DataFrame]] = {}
    for symbol in symbols:
        result[symbol] = {}
        for tf in timeframes:
            df = fetch_candles(symbol, tf, start, end)
            if df is not None and not df.empty:
                result[symbol][tf] = df
    return result
