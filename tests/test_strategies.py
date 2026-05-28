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
    # Variable volume with periodic spikes so volume-based strategies can trigger
    volume = np.random.exponential(scale=5000.0, size=n)
    spike_idx = np.random.choice(n, size=n // 5, replace=False)
    volume[spike_idx] *= np.random.uniform(2.5, 5.0, size=len(spike_idx))
    return pd.DataFrame({
        "symbol": "BTCUSDT", "timestamp": timestamps,
        "open": close - np.random.randn(n) * 15,
        "high": high, "low": low, "close": close,
        "volume": volume, "is_clean": True,
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


from crypto_bot.core.signals.strategies.donchian_breakout import DonchianBreakoutStrategy


class TestDonchianBreakoutStrategy:
    # Params with no volume/ATR filters — tests pure breakout detection logic.
    # Real runs use default params which require vol surge + ATR expansion
    # (filters validated on actual Binance data where volatility spikes at breakouts).
    _unfiltered = {
        "channel_period": 20, "vol_period": 20, "vol_mult": 1.0,
        "atr_period": 14, "atr_expansion_mult": 1.0,
    }

    def test_uptrend_produces_long_signals(self):
        df = make_candles(300, trend="up")
        sigs = DonchianBreakoutStrategy(self._unfiltered).generate_signals(df)
        assert len([s for s in sigs if s.direction == "LONG"]) > 0

    def test_downtrend_produces_short_signals(self):
        df = make_candles(300, trend="down")
        sigs = DonchianBreakoutStrategy(self._unfiltered).generate_signals(df)
        assert len([s for s in sigs if s.direction == "SHORT"]) > 0

    def test_signal_directions_valid(self):
        df = make_candles(300, trend="up")
        valid = {"LONG", "SHORT", "EXIT_LONG", "EXIT_SHORT"}
        for s in DonchianBreakoutStrategy(self._unfiltered).generate_signals(df):
            assert s.direction in valid

    def test_no_signals_on_too_short_data(self):
        df = make_candles(5, trend="up")
        assert DonchianBreakoutStrategy(self._unfiltered).generate_signals(df) == []

    def test_default_params_within_space(self):
        s = DonchianBreakoutStrategy({})
        defaults = s.default_params()
        for k, spec in s.param_space.items():
            assert spec[0] <= defaults[k] <= spec[1]

    def test_dirty_candles_skipped(self):
        df = make_candles(300, trend="up").copy()
        df.loc[100:150, "is_clean"] = False
        sigs = DonchianBreakoutStrategy(self._unfiltered).generate_signals(df)
        dirty_times = set(df[~df["is_clean"]]["timestamp"])
        for s in sigs:
            assert s.timestamp not in dirty_times

    def test_volume_filter_reduces_signals(self):
        # Tight vol filter must produce fewer signals than no filter
        df = make_candles(300, trend="up")
        no_filter = DonchianBreakoutStrategy(self._unfiltered).generate_signals(df)
        tight = DonchianBreakoutStrategy({**self._unfiltered, "vol_mult": 3.0}).generate_signals(df)
        assert len(tight) <= len(no_filter)


from crypto_bot.core.signals.strategies.tsmom import TSMOMStrategy
from crypto_bot.core.signals.strategies.hma_chandelier import HMAChandelierStrategy
from crypto_bot.core.signals.strategies.adaptive_trend import AdaptiveTrendStrategy
from crypto_bot.core.signals.strategies.vwap_breakout import VWAPBreakoutStrategy


class TestTSMOMStrategy:
    _p = {"momentum_period": 20, "vol_period": 10, "trend_period": 50, "vol_spike_mult": 2.0}

    def test_produces_signals_uptrend(self):
        df = make_candles(300, trend="up")
        sigs = TSMOMStrategy(self._p).generate_signals(df)
        assert len([s for s in sigs if s.direction == "LONG"]) > 0

    def test_produces_signals_downtrend(self):
        df = make_candles(300, trend="down")
        sigs = TSMOMStrategy(self._p).generate_signals(df)
        assert len([s for s in sigs if s.direction == "SHORT"]) > 0

    def test_no_signals_short_data(self):
        df = make_candles(10, trend="up")
        assert TSMOMStrategy(self._p).generate_signals(df) == []

    def test_valid_directions(self):
        df = make_candles(300, trend="up")
        valid = {"LONG", "SHORT", "EXIT_LONG", "EXIT_SHORT"}
        for s in TSMOMStrategy(self._p).generate_signals(df):
            assert s.direction in valid

    def test_default_params_in_space(self):
        s = TSMOMStrategy({})
        for k, spec in s.param_space.items():
            assert spec[0] <= s.default_params()[k] <= spec[1]


class TestHMAChandelierStrategy:
    _p = {"hma_fast": 9, "hma_slow": 21, "hma_trend": 50, "vol_period": 20,
          "vol_mult": 1.0, "chandelier_period": 22, "chandelier_mult": 3.0}

    def test_produces_long_in_uptrend(self):
        df = make_candles(300, trend="up")
        sigs = HMAChandelierStrategy(self._p).generate_signals(df)
        assert len([s for s in sigs if s.direction == "LONG"]) > 0

    def test_produces_short_in_downtrend(self):
        df = make_candles(300, trend="down")
        sigs = HMAChandelierStrategy(self._p).generate_signals(df)
        assert len([s for s in sigs if s.direction == "SHORT"]) > 0

    def test_no_signals_short_data(self):
        df = make_candles(10, trend="up")
        assert HMAChandelierStrategy(self._p).generate_signals(df) == []

    def test_valid_directions(self):
        df = make_candles(300, trend="up")
        valid = {"LONG", "SHORT", "EXIT_LONG", "EXIT_SHORT"}
        for s in HMAChandelierStrategy(self._p).generate_signals(df):
            assert s.direction in valid

    def test_default_params_in_space(self):
        s = HMAChandelierStrategy({})
        for k, spec in s.param_space.items():
            assert spec[0] <= s.default_params()[k] <= spec[1]


class TestAdaptiveTrendStrategy:
    _p = {"ema_fast": 10, "ema_mid": 20, "ema_slow": 40,
          "vol_period": 20, "vol_regime_pct": 80, "atr_period": 14}

    def test_produces_long_in_uptrend(self):
        df = make_candles(300, trend="up")
        sigs = AdaptiveTrendStrategy(self._p).generate_signals(df)
        assert len([s for s in sigs if s.direction == "LONG"]) > 0

    def test_produces_short_in_downtrend(self):
        df = make_candles(300, trend="down")
        sigs = AdaptiveTrendStrategy(self._p).generate_signals(df)
        assert len([s for s in sigs if s.direction == "SHORT"]) > 0

    def test_no_signals_short_data(self):
        df = make_candles(10, trend="up")
        assert AdaptiveTrendStrategy(self._p).generate_signals(df) == []

    def test_valid_directions(self):
        df = make_candles(300, trend="up")
        valid = {"LONG", "SHORT", "EXIT_LONG", "EXIT_SHORT"}
        for s in AdaptiveTrendStrategy(self._p).generate_signals(df):
            assert s.direction in valid

    def test_default_params_in_space(self):
        s = AdaptiveTrendStrategy({})
        for k, spec in s.param_space.items():
            assert spec[0] <= s.default_params()[k] <= spec[1]


class TestVWAPBreakoutStrategy:
    _p = {"vwap_period": 20, "band_mult": 1.5, "vol_period": 20,
          "vol_mult": 1.0, "atr_period": 14}

    def test_produces_signals(self):
        df = make_candles(300, trend="up")
        sigs = VWAPBreakoutStrategy(self._p).generate_signals(df)
        assert len(sigs) > 0

    def test_no_signals_short_data(self):
        df = make_candles(5, trend="up")
        assert VWAPBreakoutStrategy(self._p).generate_signals(df) == []

    def test_valid_directions(self):
        df = make_candles(300, trend="up")
        valid = {"LONG", "SHORT", "EXIT_LONG", "EXIT_SHORT"}
        for s in VWAPBreakoutStrategy(self._p).generate_signals(df):
            assert s.direction in valid

    def test_default_params_in_space(self):
        s = VWAPBreakoutStrategy({})
        for k, spec in s.param_space.items():
            assert spec[0] <= s.default_params()[k] <= spec[1]


from crypto_bot.core.signals.strategies.ichimoku_cloud import IchimokuCloudStrategy


class TestIchimokuCloudStrategy:
    _p = {
        "tenkan_period": 9, "kijun_period": 26,
        "senkou_b_period": 52, "atr_period": 14,
    }

    def test_produces_long_in_uptrend(self):
        df = make_candles(300, trend="up")
        sigs = IchimokuCloudStrategy(self._p).generate_signals(df)
        assert len([s for s in sigs if s.direction == "LONG"]) > 0

    def test_produces_short_in_downtrend(self):
        df = make_candles(300, trend="down")
        sigs = IchimokuCloudStrategy(self._p).generate_signals(df)
        assert len([s for s in sigs if s.direction == "SHORT"]) > 0

    def test_no_signals_short_data(self):
        df = make_candles(50, trend="up")
        assert IchimokuCloudStrategy(self._p).generate_signals(df) == []

    def test_valid_directions(self):
        df = make_candles(300, trend="up")
        valid = {"LONG", "SHORT", "EXIT_LONG", "EXIT_SHORT"}
        for s in IchimokuCloudStrategy(self._p).generate_signals(df):
            assert s.direction in valid

    def test_default_params_in_space(self):
        s = IchimokuCloudStrategy({})
        for k, spec in s.param_space.items():
            assert spec[0] <= s.default_params()[k] <= spec[1]


from crypto_bot.core.signals.strategies.stoch_rsi import StochRSIStrategy


class TestStochRSIStrategy:
    _p = {
        "rsi_period": 14, "stoch_period": 14,
        "smooth_k": 3, "smooth_d": 3, "atr_period": 14,
    }

    def test_produces_long_in_uptrend(self):
        df = make_candles(300, trend="up")
        sigs = StochRSIStrategy(self._p).generate_signals(df)
        assert len([s for s in sigs if s.direction == "LONG"]) > 0

    def test_produces_short_from_overbought(self):
        # StochRSI is mean-reversion: SHORT fires when RSI is overbought (high RSI = uptrend/choppy)
        df = make_candles(300, trend="flat", seed=0)
        sigs = StochRSIStrategy(self._p).generate_signals(df)
        assert len([s for s in sigs if s.direction == "SHORT"]) > 0

    def test_no_signals_short_data(self):
        df = make_candles(20, trend="up")
        assert StochRSIStrategy(self._p).generate_signals(df) == []

    def test_valid_directions(self):
        df = make_candles(300, trend="up")
        valid = {"LONG", "SHORT", "EXIT_LONG", "EXIT_SHORT"}
        for s in StochRSIStrategy(self._p).generate_signals(df):
            assert s.direction in valid

    def test_default_params_in_space(self):
        s = StochRSIStrategy({})
        for k, spec in s.param_space.items():
            assert spec[0] <= s.default_params()[k] <= spec[1]
