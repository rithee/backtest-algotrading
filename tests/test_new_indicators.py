"""Tests for new indicators: ichimoku, stoch_rsi, ema_slope."""
import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta
from crypto_bot.core.signals.indicators import ichimoku, stoch_rsi, ema_slope


def _make_series(n=300, trend="up", seed=42):
    np.random.seed(seed)
    if trend == "up":
        close = 30000.0 + np.arange(n) * 8.0 + np.random.randn(n) * 40.0
    else:
        close = 50000.0 - np.arange(n) * 8.0 + np.random.randn(n) * 40.0
    high = close + np.abs(np.random.randn(n) * 30.0)
    low  = close - np.abs(np.random.randn(n) * 30.0)
    idx  = [datetime(2023, 1, 1) + timedelta(hours=4 * i) for i in range(n)]
    return (
        pd.Series(high, index=idx),
        pd.Series(low,  index=idx),
        pd.Series(close, index=idx),
    )


class TestIchimoku:
    def test_returns_four_series(self):
        high, low, close = _make_series()
        result = ichimoku(high, low)
        assert len(result) == 4
        tenkan, kijun, span_a, span_b = result
        assert isinstance(tenkan, pd.Series)
        assert isinstance(span_b, pd.Series)

    def test_length_matches_input(self):
        high, low, close = _make_series()
        tenkan, kijun, span_a, span_b = ichimoku(high, low)
        assert len(tenkan) == len(high)
        assert len(span_b) == len(high)

    def test_tenkan_shorter_warmup_than_kijun(self):
        high, low, close = _make_series()
        tenkan, kijun, _, _ = ichimoku(high, low, tenkan_period=9, kijun_period=26)
        assert tenkan.first_valid_index() <= kijun.first_valid_index()

    def test_span_a_is_mean_of_tenkan_kijun(self):
        high, low, close = _make_series(200)
        tenkan, kijun, span_a, _ = ichimoku(high, low, tenkan_period=9, kijun_period=26, displacement=0)
        expected = (tenkan + kijun) / 2
        pd.testing.assert_series_equal(span_a, expected, check_names=False)


class TestStochRSI:
    def test_returns_two_series(self):
        _, _, close = _make_series()
        k, d = stoch_rsi(close)
        assert isinstance(k, pd.Series)
        assert isinstance(d, pd.Series)

    def test_values_bounded_0_100(self):
        _, _, close = _make_series()
        k, d = stoch_rsi(close)
        valid_k = k.dropna()
        valid_d = d.dropna()
        assert (valid_k >= 0).all() and (valid_k <= 100).all()
        assert (valid_d >= 0).all() and (valid_d <= 100).all()

    def test_d_is_smoother_than_k(self):
        _, _, close = _make_series()
        k, d = stoch_rsi(close, smooth_k=3, smooth_d=3)
        assert d.dropna().std() <= k.dropna().std()

    def test_length_matches_input(self):
        _, _, close = _make_series()
        k, d = stoch_rsi(close)
        assert len(k) == len(close)


class TestEMASlope:
    def test_positive_on_uptrend(self):
        _, _, close = _make_series(trend="up")
        slope = ema_slope(close, period=20, lookback=5)
        assert slope.dropna().mean() > 0

    def test_negative_on_downtrend(self):
        _, _, close = _make_series(trend="down")
        slope = ema_slope(close, period=20, lookback=5)
        assert slope.dropna().mean() < 0

    def test_length_matches_input(self):
        _, _, close = _make_series()
        slope = ema_slope(close, period=20, lookback=5)
        assert len(slope) == len(close)
