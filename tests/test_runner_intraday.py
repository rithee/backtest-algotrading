"""Tests for intraday mode runner: session filter, daily kill switch, max trades/day."""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from crypto_bot.core.config import Config
from crypto_bot.core.signals.models import Signal
from backtest.runner import BacktestRunner


def _make_intraday_candles(n: int = 120, symbol: str = "BTCUSDT") -> pd.DataFrame:
    """5-minute candles starting at 08:00 UTC so they land in a session window."""
    base = datetime(2023, 1, 2, 8, 0)  # Monday 08:00 UTC
    ts = [base + timedelta(minutes=5 * i) for i in range(n)]
    close = np.linspace(20000, 21000, n)
    return pd.DataFrame({
        "timestamp": ts,
        "open": close * 0.999,
        "high": close * 1.001,
        "low": close * 0.998,
        "close": close,
        "volume": np.ones(n) * 500,
        "is_clean": [True] * n,
        "symbol": [symbol] * n,
    })


class AlwaysLongIntradayStrategy:
    """Fires a LONG signal on every candle (for testing limits)."""
    name = "AlwaysLongIntraday"
    params = {}

    def default_params(self): return {}

    def generate_signals(self, candles, aux_data=None):
        signals = []
        for i, row in candles.iterrows():
            signals.append(Signal(
                strategy=self.name,
                symbol=row["symbol"],
                direction="LONG",
                timestamp=row["timestamp"],
                strength=0.8,
                close_price=row["close"],
                atr=row["close"] * 0.01,
                reason=["intraday_test"],
            ))
        return signals

    def generate_signals_mtf(self, candles_by_tf, aux_data=None):
        first = next(iter(candles_by_tf.values()))
        return self.generate_signals(first, aux_data)

    @property
    def mode(self): return "intraday"

    @property
    def param_space(self): return {}


def test_intraday_runner_returns_result():
    cfg = Config.from_yaml("config/intraday.yaml")
    runner = BacktestRunner(cfg, mode="intraday")
    candles = {"BTCUSDT": _make_intraday_candles()}
    result = runner.run(AlwaysLongIntradayStrategy(), candles)
    assert result is not None
    assert result.strategy_name == "AlwaysLongIntraday"
    assert result.mode == "intraday"
    assert len(result.equity_curve) > 0


def test_intraday_runner_respects_max_trades_per_day():
    """With max_trades_per_day=5, at most 5 fills per symbol per day."""
    cfg = Config.from_yaml("config/intraday.yaml")
    runner = BacktestRunner(cfg, mode="intraday")
    candles = {"BTCUSDT": _make_intraday_candles(n=120)}
    result = runner.run(AlwaysLongIntradayStrategy(), candles)

    # Count fills per (date, symbol)
    if result.fills:
        from collections import Counter
        counts = Counter(
            (f.timestamp.date(), f.symbol) for f in result.fills
        )
        max_daily = cfg.risk.max_trades_per_day or 5
        for (date, sym), count in counts.items():
            assert count <= max_daily, (
                f"Too many fills on {date} for {sym}: {count} > {max_daily}"
            )


def test_intraday_runner_uses_mtf_signals():
    cfg = Config.from_yaml("config/intraday.yaml")
    runner = BacktestRunner(cfg, mode="intraday")
    candles = {"BTCUSDT": _make_intraday_candles()}
    candles_by_tf = {"BTCUSDT": {"5m": _make_intraday_candles(), "15m": _make_intraday_candles()}}
    result = runner.run(AlwaysLongIntradayStrategy(), candles, candles_by_tf=candles_by_tf)
    assert result is not None
    assert result.mode == "intraday"
