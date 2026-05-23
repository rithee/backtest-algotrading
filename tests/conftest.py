import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta


@pytest.fixture
def flat_candles():
    """200 4h candles at constant close=100."""
    n = 200
    timestamps = [datetime(2023, 1, 1) + timedelta(hours=4 * i) for i in range(n)]
    return pd.DataFrame({
        "symbol": "BTCUSDT",
        "timestamp": timestamps,
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.0,
        "volume": 1000.0,
        "is_clean": True,
    })


@pytest.fixture
def trending_up_candles():
    """250 4h candles with a clear uptrend (+10 per candle + noise)."""
    n = 250
    np.random.seed(42)
    timestamps = [datetime(2023, 1, 1) + timedelta(hours=4 * i) for i in range(n)]
    close = 30000.0 + np.arange(n) * 10.0 + np.random.randn(n) * 50.0
    high = close + np.abs(np.random.randn(n) * 30.0)
    low = close - np.abs(np.random.randn(n) * 30.0)
    return pd.DataFrame({
        "symbol": "BTCUSDT",
        "timestamp": timestamps,
        "open": close - np.random.randn(n) * 20.0,
        "high": high,
        "low": low,
        "close": close,
        "volume": np.abs(np.random.randn(n) * 1000.0 + 5000.0),
        "is_clean": True,
    })


@pytest.fixture
def trending_down_candles():
    """250 4h candles with a clear downtrend (-10 per candle + noise)."""
    n = 250
    np.random.seed(99)
    timestamps = [datetime(2023, 1, 1) + timedelta(hours=4 * i) for i in range(n)]
    close = 50000.0 - np.arange(n) * 10.0 + np.random.randn(n) * 50.0
    high = close + np.abs(np.random.randn(n) * 30.0)
    low = close - np.abs(np.random.randn(n) * 30.0)
    return pd.DataFrame({
        "symbol": "BTCUSDT",
        "timestamp": timestamps,
        "open": close - np.random.randn(n) * 20.0,
        "high": high,
        "low": low,
        "close": close,
        "volume": np.abs(np.random.randn(n) * 1000.0 + 5000.0),
        "is_clean": True,
    })
