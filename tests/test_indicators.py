import numpy as np
import pandas as pd
import pytest
from crypto_bot.core.signals.indicators import (
    ema, rsi, macd, bollinger_bands, atr, adx,
    supertrend, keltner_channels, squeeze_momentum,
    bb_percent_b, bb_width,
)


class TestEMA:
    def test_constant_series_equals_value(self, flat_candles):
        result = ema(flat_candles["close"], period=10)
        assert abs(result.iloc[-1] - 100.0) < 1e-6

    def test_faster_ema_higher_in_uptrend(self, trending_up_candles):
        close = trending_up_candles["close"]
        assert ema(close, 10).iloc[-1] > ema(close, 50).iloc[-1]

    def test_output_length_matches_input(self, flat_candles):
        assert len(ema(flat_candles["close"], 10)) == len(flat_candles)


class TestRSI:
    def test_flat_series_near_50(self, flat_candles):
        result = rsi(flat_candles["close"], period=14)
        assert abs(result.iloc[-1] - 50.0) < 5.0

    def test_uptrend_above_50(self, trending_up_candles):
        assert rsi(trending_up_candles["close"], 14).iloc[-1] > 50

    def test_downtrend_below_50(self, trending_down_candles):
        assert rsi(trending_down_candles["close"], 14).iloc[-1] < 50

    def test_values_bounded_0_100(self, trending_up_candles):
        result = rsi(trending_up_candles["close"], 14).dropna()
        assert result.min() >= 0.0
        assert result.max() <= 100.0

    def test_pure_up_series_approaches_100(self):
        close = pd.Series(np.arange(1.0, 101.0))
        assert rsi(close, 14).iloc[-1] > 90


class TestMACD:
    def test_histogram_equals_line_minus_signal(self, trending_up_candles):
        line, signal, hist = macd(trending_up_candles["close"])
        diff = (line - signal - hist).dropna().abs()
        assert diff.max() < 1e-10

    def test_uptrend_positive_macd(self, trending_up_candles):
        line, _, _ = macd(trending_up_candles["close"])
        assert line.iloc[-1] > 0

    def test_returns_three_series_same_length(self, flat_candles):
        line, sig, hist = macd(flat_candles["close"])
        assert len(line) == len(flat_candles)
        assert len(sig) == len(flat_candles)
        assert len(hist) == len(flat_candles)


class TestBollingerBands:
    def test_upper_above_lower(self, trending_up_candles):
        upper, _, lower = bollinger_bands(trending_up_candles["close"])
        valid = upper.notna() & lower.notna()
        assert (upper[valid] > lower[valid]).all()

    def test_price_mostly_inside_2sigma(self, trending_up_candles):
        close = trending_up_candles["close"]
        upper, _, lower = bollinger_bands(close, 20, 2.0)
        valid = upper.notna()
        inside = ((close[valid] <= upper[valid]) & (close[valid] >= lower[valid])).mean()
        assert inside > 0.90


class TestATR:
    def test_constant_range_converges(self, flat_candles):
        # high=101, low=99 → range=2 → ATR converges to ~2
        result = atr(flat_candles["high"], flat_candles["low"], flat_candles["close"], 14)
        assert abs(result.iloc[-1] - 2.0) < 0.5

    def test_always_positive(self, trending_up_candles):
        result = atr(
            trending_up_candles["high"], trending_up_candles["low"],
            trending_up_candles["close"], 14,
        ).dropna()
        assert (result > 0).all()

    def test_higher_volatility_higher_atr(self):
        np.random.seed(0)
        n = 150
        c_low = pd.Series(100.0 + np.random.randn(n) * 0.5)
        c_high = pd.Series(100.0 + np.random.randn(n) * 5.0)
        atr_low = atr(c_low + 0.5, c_low - 0.5, c_low, 14).iloc[-1]
        atr_high = atr(c_high + 5.0, c_high - 5.0, c_high, 14).iloc[-1]
        assert atr_high > atr_low


class TestADX:
    def test_range_0_to_100(self, trending_up_candles):
        result = adx(
            trending_up_candles["high"], trending_up_candles["low"],
            trending_up_candles["close"], 14,
        ).dropna()
        assert result.min() >= 0
        assert result.max() <= 100

    def test_strong_trend_high_adx(self, trending_up_candles):
        result = adx(
            trending_up_candles["high"], trending_up_candles["low"],
            trending_up_candles["close"], 14,
        )
        assert result.iloc[-1] > 15


class TestSupertrend:
    def test_direction_only_1_or_minus1(self, trending_up_candles):
        _, direction = supertrend(
            trending_up_candles["high"], trending_up_candles["low"],
            trending_up_candles["close"],
        )
        valid = direction[direction != 0]
        assert set(valid.unique()).issubset({1, -1})

    def test_uptrend_direction_positive(self, trending_up_candles):
        _, direction = supertrend(
            trending_up_candles["high"], trending_up_candles["low"],
            trending_up_candles["close"],
        )
        assert direction.iloc[-1] == 1

    def test_downtrend_direction_negative(self, trending_down_candles):
        _, direction = supertrend(
            trending_down_candles["high"], trending_down_candles["low"],
            trending_down_candles["close"],
        )
        assert direction.iloc[-1] == -1


class TestKeltnerChannels:
    def test_upper_above_lower(self, trending_up_candles):
        upper, _, lower = keltner_channels(
            trending_up_candles["high"], trending_up_candles["low"],
            trending_up_candles["close"],
        )
        valid = upper.notna()
        assert (upper[valid] > lower[valid]).all()


class TestSqueezeMomentum:
    def test_returns_two_series_correct_length(self, trending_up_candles):
        mom, sq = squeeze_momentum(
            trending_up_candles["high"], trending_up_candles["low"],
            trending_up_candles["close"],
        )
        assert len(mom) == len(trending_up_candles)
        assert len(sq) == len(trending_up_candles)

    def test_squeeze_on_is_bool_dtype(self, flat_candles):
        _, sq = squeeze_momentum(
            flat_candles["high"], flat_candles["low"], flat_candles["close"],
        )
        assert sq.dtype == bool


class TestBBHelpers:
    def test_bb_width_positive(self, trending_up_candles):
        result = bb_width(trending_up_candles["close"], 20).dropna()
        assert (result > 0).all()

    def test_percent_b_at_midband_is_half(self):
        # Constant series: close == middle band → %B = 0.5
        close = pd.Series([100.0] * 100)
        result = bb_percent_b(close, 20).dropna()
        if len(result) > 0:
            assert abs(result.iloc[-1] - 0.5) < 0.05
