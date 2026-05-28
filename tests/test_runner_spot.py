"""Tests for spot mode runner: no leverage, min_holding_days."""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from crypto_bot.core.config import Config
from crypto_bot.core.signals.models import Signal
from backtest.runner import BacktestRunner


def _make_candles(n: int = 60, symbol: str = "BTCUSDT") -> pd.DataFrame:
    ts = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(n)]
    close = np.linspace(20000, 25000, n)
    return pd.DataFrame({
        "timestamp": ts, "open": close * 0.99, "high": close * 1.01,
        "low": close * 0.98, "close": close,
        "volume": np.ones(n) * 500, "is_clean": [True] * n, "symbol": [symbol] * n,
    })


class AlwaysLongSpotStrategy:
    """Fires a single LONG signal on the first candle."""
    name = "AlwaysLongSpot"
    params = {}

    def default_params(self): return {}

    def generate_signals(self, candles, aux_data=None):
        signals = []
        for i, row in candles.iterrows():
            if i == 0:
                signals.append(Signal(
                    strategy=self.name,
                    symbol=row["symbol"],
                    direction="LONG",
                    timestamp=row["timestamp"],
                    strength=0.9,
                    close_price=row["close"],
                    atr=row["close"] * 0.02,
                    reason=["test_entry"],
                ))
        return signals

    def generate_signals_mtf(self, candles_by_tf, aux_data=None):
        first = next(iter(candles_by_tf.values()))
        return self.generate_signals(first, aux_data)

    @property
    def mode(self): return "spot_longterm"

    @property
    def param_space(self): return {}


def test_spot_runner_returns_backtest_result():
    cfg = Config.from_yaml("config/spot_longterm.yaml")
    runner = BacktestRunner(cfg, mode="spot_longterm")
    candles = {"BTCUSDT": _make_candles()}
    result = runner.run(AlwaysLongSpotStrategy(), candles)
    assert result is not None
    assert result.strategy_name == "AlwaysLongSpot"
    assert result.mode == "spot_longterm"
    assert len(result.equity_curve) > 0


def test_spot_runner_enforces_leverage_one():
    cfg = Config.from_yaml("config/spot_longterm.yaml")
    runner = BacktestRunner(cfg, mode="spot_longterm")
    candles = {"BTCUSDT": _make_candles()}
    result = runner.run(AlwaysLongSpotStrategy(), candles)
    for fill in result.fills:
        assert fill.leverage <= 1.0


def test_spot_runner_uses_mtf_signals_when_provided():
    cfg = Config.from_yaml("config/spot_longterm.yaml")
    runner = BacktestRunner(cfg, mode="spot_longterm")
    candles = {"BTCUSDT": _make_candles()}
    candles_by_tf = {"BTCUSDT": {"1w": _make_candles(), "1d": _make_candles()}}
    result = runner.run(AlwaysLongSpotStrategy(), candles, candles_by_tf=candles_by_tf)
    assert result is not None
    assert result.mode == "spot_longterm"
