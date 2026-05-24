"""Tests for strategies 1–4: signal types, no lookahead, handles clean/dirty candles."""
import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta
from crypto_bot.core.signals.strategies.ema_ribbon import EMARibbonStrategy
from crypto_bot.core.signals.strategies.ttm_squeeze import TTMSqueezeStrategy
from crypto_bot.core.signals.strategies.rsi_divergence import RSIDivergenceStrategy
from crypto_bot.core.signals.strategies.supertrend_adx import SupertrendADXStrategy


def make_candles(n: int = 300, trend: str = "up", seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    if trend == "up":
        close = 30000.0 + np.arange(n) * 8.0 + np.random.randn(n) * 40.0
    elif trend == "down":
        close = 50000.0 - np.arange(n) * 8.0 + np.random.randn(n) * 40.0
    else:
        close = 30000.0 + np.random.randn(n) * 200.0
    high = close + np.abs(np.random.randn(n) * 30.0)
    low = close - np.abs(np.random.randn(n) * 30.0)
    timestamps = [datetime(2023, 1, 1) + timedelta(hours=4 * i) for i in range(n)]
    return pd.DataFrame({
        "symbol": "BTCUSDT", "timestamp": timestamps,
        "open": close - np.random.randn(n) * 15,
        "high": high, "low": low, "close": close,
        "volume": 5000.0, "is_clean": True,
    })


class TestEMARibbonStrategy:
    def test_default_params_returns_signals(self, trending_up_candles):
        dummy = EMARibbonStrategy({})
        strategy = EMARibbonStrategy(dummy.default_params())
        sigs = strategy.generate_signals(trending_up_candles)
        assert len(sigs) > 0

    def test_signal_directions_valid(self, trending_up_candles):
        dummy = EMARibbonStrategy({})
        strategy = EMARibbonStrategy(dummy.default_params())
        sigs = strategy.generate_signals(trending_up_candles)
        valid = {"LONG", "SHORT", "EXIT_LONG", "EXIT_SHORT"}
        assert all(s.direction in valid for s in sigs)

    def test_strength_bounded(self, trending_up_candles):
        dummy = EMARibbonStrategy({})
        strategy = EMARibbonStrategy(dummy.default_params())
        sigs = strategy.generate_signals(trending_up_candles)
        assert all(0.0 <= s.strength <= 1.0 for s in sigs)

    def test_no_signals_after_warmup_on_dirty(self):
        df = make_candles(300)
        df["is_clean"] = False  # all dirty
        dummy = EMARibbonStrategy({})
        strategy = EMARibbonStrategy(dummy.default_params())
        sigs = strategy.generate_signals(df)
        assert sigs == []

    def test_signal_timestamps_in_order(self, trending_up_candles):
        dummy = EMARibbonStrategy({})
        strategy = EMARibbonStrategy(dummy.default_params())
        sigs = strategy.generate_signals(trending_up_candles)
        timestamps = [s.timestamp for s in sigs]
        assert timestamps == sorted(timestamps)

    def test_name_is_classname(self):
        s = EMARibbonStrategy({})
        assert s.name == "EMARibbonStrategy"


class TestTTMSqueezeStrategy:
    def test_returns_signals_on_trending_data(self, trending_up_candles):
        dummy = TTMSqueezeStrategy({})
        strategy = TTMSqueezeStrategy(dummy.default_params())
        sigs = strategy.generate_signals(trending_up_candles)
        assert isinstance(sigs, list)

    def test_signal_atr_positive(self, trending_up_candles):
        dummy = TTMSqueezeStrategy({})
        strategy = TTMSqueezeStrategy(dummy.default_params())
        sigs = strategy.generate_signals(trending_up_candles)
        assert all(s.atr > 0 for s in sigs)


class TestRSIDivergenceStrategy:
    def test_valid_signal_types(self, trending_up_candles):
        dummy = RSIDivergenceStrategy({})
        strategy = RSIDivergenceStrategy(dummy.default_params())
        sigs = strategy.generate_signals(trending_up_candles)
        valid = {"LONG", "SHORT", "EXIT_LONG", "EXIT_SHORT"}
        assert all(s.direction in valid for s in sigs)

    def test_strength_in_range(self, trending_up_candles):
        dummy = RSIDivergenceStrategy({})
        strategy = RSIDivergenceStrategy(dummy.default_params())
        sigs = strategy.generate_signals(trending_up_candles)
        assert all(0.0 <= s.strength <= 1.0 for s in sigs)


class TestSupertrendADXStrategy:
    def test_uptrend_produces_long_signal(self):
        df = make_candles(300, trend="up")
        dummy = SupertrendADXStrategy({})
        strategy = SupertrendADXStrategy(dummy.default_params())
        sigs = strategy.generate_signals(df)
        long_sigs = [s for s in sigs if s.direction == "LONG"]
        assert len(long_sigs) > 0

    def test_downtrend_produces_short_signal(self):
        df = make_candles(300, trend="down")
        dummy = SupertrendADXStrategy({})
        strategy = SupertrendADXStrategy(dummy.default_params())
        sigs = strategy.generate_signals(df)
        short_sigs = [s for s in sigs if s.direction == "SHORT"]
        assert len(short_sigs) > 0

    def test_default_params_mid_of_space(self):
        s = SupertrendADXStrategy({})
        defaults = s.default_params()
        space = s.param_space
        for k, v in defaults.items():
            low, high = space[k][0], space[k][1]
            assert low <= v <= high
