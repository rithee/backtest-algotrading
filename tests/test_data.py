"""Tests for data validator. Fetcher is not tested (requires network)."""
import numpy as np
import pandas as pd
import pytest
from crypto_bot.core.data.validator import validate, quality_score


def make_candles(n: int = 100, seed: int = 0) -> pd.DataFrame:
    np.random.seed(seed)
    close = 30000.0 + np.random.randn(n) * 200
    high = close + np.abs(np.random.randn(n) * 50)
    low = close - np.abs(np.random.randn(n) * 50)
    return pd.DataFrame({
        "symbol": "BTCUSDT",
        "timestamp": pd.date_range("2023-01-01", periods=n, freq="4h"),
        "open": close - np.random.randn(n) * 20,
        "high": high, "low": low, "close": close,
        "volume": 5000.0,
        "is_clean": True,
    })


class TestValidator:
    def test_clean_data_stays_clean(self):
        df = make_candles(100)
        result = validate(df)
        assert quality_score(result) > 0.90

    def test_zero_volume_flagged(self):
        df = make_candles(100)
        df.loc[5, "volume"] = 0
        result = validate(df)
        assert not result.loc[result["timestamp"] == df.loc[5, "timestamp"], "is_clean"].values[0]

    def test_anomalous_range_flagged(self):
        df = make_candles(100)
        # Create a candle with a massive range (100× normal)
        df.loc[50, "high"] = df.loc[50, "low"] + 50000.0
        result = validate(df)
        assert not result.loc[50, "is_clean"]

    def test_duplicate_timestamps_removed(self):
        df = make_candles(50)
        df_dup = pd.concat([df, df.iloc[:5]], ignore_index=True)
        result = validate(df_dup)
        assert len(result) == 50

    def test_quality_score_range(self):
        df = make_candles(100)
        score = quality_score(validate(df))
        assert 0.0 <= score <= 1.0

    def test_quality_score_empty_df(self):
        df = pd.DataFrame(columns=["symbol", "timestamp", "open", "high", "low", "close", "volume", "is_clean"])
        assert quality_score(df) == 0.0
