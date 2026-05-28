"""Tests for the aux_data and MTF candle loader."""
import pandas as pd
import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch
from crypto_bot.core.config import Config


def _make_candle_df(n: int = 10, symbol: str = "BTCUSDT") -> pd.DataFrame:
    ts = [datetime(2023, 1, 1) + timedelta(hours=i * 4) for i in range(n)]
    close = np.linspace(20000, 21000, n)
    return pd.DataFrame({
        "timestamp": ts, "open": close * 0.99, "high": close * 1.01,
        "low": close * 0.98, "close": close,
        "volume": np.ones(n) * 500, "is_clean": [True] * n, "symbol": [symbol] * n,
    })


def test_load_aux_data_intraday_includes_liquidations(monkeypatch):
    from crypto_bot.core.data import loader as loader_mod
    monkeypatch.setattr(loader_mod, "fetch_liquidations",
                        lambda symbol, start, end: pd.DataFrame({"val": [1.0]}))
    monkeypatch.setattr(loader_mod, "fetch_open_interest",
                        lambda symbol, interval, start, end: pd.Series([1.0], name="open_interest"))
    monkeypatch.setattr(loader_mod, "fetch_funding_rates",
                        lambda symbol, start, end: pd.DataFrame({"rate": [0.001]}))

    cfg = Config.from_yaml("config/intraday.yaml")
    from crypto_bot.core.data.loader import load_aux_data
    aux = load_aux_data(cfg, mode="intraday")

    assert "liquidations" in aux
    assert "open_interest" in aux
    assert "funding_rate" in aux


def test_load_aux_data_spot_includes_macro_and_onchain(monkeypatch):
    from crypto_bot.core.data import loader as loader_mod
    monkeypatch.setattr(loader_mod, "fetch_macro",
                        lambda start, end: pd.DataFrame({"spx": [100.0]}))
    monkeypatch.setattr(loader_mod, "fetch_glassnode",
                        lambda metric, symbol, start, end: pd.DataFrame({"v": [1.0]}))
    monkeypatch.setattr(loader_mod, "fetch_top_n",
                        lambda n=10, date=None: ["BTCUSDT", "ETHUSDT"])
    monkeypatch.setattr(loader_mod, "fetch_open_interest",
                        lambda symbol, interval, start, end: None)
    monkeypatch.setattr(loader_mod, "fetch_funding_rates",
                        lambda symbol, start, end: None)

    cfg = Config.from_yaml("config/spot_longterm.yaml")
    from crypto_bot.core.data.loader import load_aux_data
    aux = load_aux_data(cfg, mode="spot_longterm")

    assert "macro" in aux
    assert "onchain" in aux
    assert "top10_mcap" in aux


def test_load_mtf_candles_returns_nested_dict(monkeypatch):
    from crypto_bot.core.data import loader as loader_mod
    monkeypatch.setattr(loader_mod, "fetch_candles",
                        lambda symbol, tf, start, end: _make_candle_df(symbol=symbol))

    cfg = Config.from_yaml("config/intraday.yaml")
    from crypto_bot.core.data.loader import load_mtf_candles
    result = load_mtf_candles(cfg)

    assert isinstance(result, dict)
    for symbol in cfg.backtest.symbols:
        assert symbol in result
        assert isinstance(result[symbol], dict)
        for tf in cfg.backtest.active_timeframes:
            assert tf in result[symbol]


def test_load_mtf_candles_skips_failed_fetches(monkeypatch):
    from crypto_bot.core.data import loader as loader_mod

    def partial_fetch(symbol, tf, start, end):
        return None if tf == "1m" else _make_candle_df(symbol=symbol)

    monkeypatch.setattr(loader_mod, "fetch_candles", partial_fetch)

    cfg = Config.from_yaml("config/intraday.yaml")
    from crypto_bot.core.data.loader import load_mtf_candles
    result = load_mtf_candles(cfg)

    for symbol in cfg.backtest.symbols:
        assert "1m" not in result[symbol]   # failed fetches are omitted
        assert "5m" in result[symbol]        # others still present
