"""All TA indicators as pure functions. No side effects, fully vectorized."""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    # When avg_loss == 0 and avg_gain == 0: RSI = 50 (flat market)
    # When avg_loss == 0 and avg_gain > 0:  RSI = 100 (pure uptrend)
    result = pd.Series(np.where(
        avg_loss == 0,
        np.where(avg_gain == 0, 50.0, 100.0),
        100.0 - (100.0 / (1.0 + avg_gain / avg_loss)),
    ), index=close.index)
    # Preserve NaN where avg_gain is NaN (first row after diff)
    result[avg_gain.isna()] = np.nan
    return result


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal_period: int = 9,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (macd_line, signal_line, histogram)."""
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal_period)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger_bands(
    close: pd.Series, period: int = 20, std_dev: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (upper, middle, lower)."""
    middle = close.rolling(period).mean()
    std = close.rolling(period).std(ddof=0)
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    return upper, middle, lower


def atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


def adx(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """Average Directional Index — trend strength 0–100."""
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=close.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=close.index,
    )
    atr_val = atr(high, low, close, period)
    plus_di = 100.0 * plus_dm.ewm(com=period - 1, adjust=False).mean() / atr_val
    minus_di = 100.0 * minus_dm.ewm(com=period - 1, adjust=False).mean() / atr_val
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(com=period - 1, adjust=False).mean()


def supertrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 10,
    multiplier: float = 3.0,
) -> Tuple[pd.Series, pd.Series]:
    """Returns (supertrend_line, direction) where direction 1=up, -1=down."""
    atr_val = atr(high, low, close, period)
    hl2 = (high + low) / 2.0
    raw_upper = (hl2 + multiplier * atr_val).values.copy()
    raw_lower = (hl2 - multiplier * atr_val).values.copy()
    close_arr = close.values
    atr_arr = atr_val.values

    n = len(close_arr)
    st = np.full(n, np.nan)
    direction = np.zeros(n, dtype=int)
    upper = raw_upper.copy()
    lower = raw_lower.copy()

    for i in range(1, n):
        if np.isnan(atr_arr[i]):
            continue
        upper[i] = (
            raw_upper[i]
            if raw_upper[i] < upper[i - 1] or close_arr[i - 1] > upper[i - 1]
            else upper[i - 1]
        )
        lower[i] = (
            raw_lower[i]
            if raw_lower[i] > lower[i - 1] or close_arr[i - 1] < lower[i - 1]
            else lower[i - 1]
        )
        prev_dir = direction[i - 1]
        if prev_dir >= 0:
            if close_arr[i] < lower[i]:
                direction[i] = -1
                st[i] = upper[i]
            else:
                direction[i] = 1
                st[i] = lower[i]
        else:
            if close_arr[i] > upper[i]:
                direction[i] = 1
                st[i] = lower[i]
            else:
                direction[i] = -1
                st[i] = upper[i]

    return pd.Series(st, index=close.index), pd.Series(direction, index=close.index)


def keltner_channels(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 20,
    multiplier: float = 1.5,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (upper, middle, lower)."""
    middle = ema(close, period)
    atr_val = atr(high, low, close, period)
    return middle + multiplier * atr_val, middle, middle - multiplier * atr_val


def squeeze_momentum(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    bb_period: int = 20,
    bb_std: float = 2.0,
    kc_period: int = 20,
    kc_mult: float = 1.5,
    mom_period: int = 12,
) -> Tuple[pd.Series, pd.Series]:
    """TTM Squeeze. Returns (momentum, squeeze_on)."""
    bb_upper, _, bb_lower = bollinger_bands(close, bb_period, bb_std)
    kc_upper, kc_mid, kc_lower = keltner_channels(high, low, close, kc_period, kc_mult)
    squeeze_on = (bb_upper < kc_upper) & (bb_lower > kc_lower)

    highest = close.rolling(bb_period).max()
    lowest = close.rolling(bb_period).min()
    midpoint = (highest + lowest) / 2.0
    delta = close - (midpoint + kc_mid) / 2.0

    def _linreg_last(x: np.ndarray) -> float:
        n = len(x)
        if n < 2:
            return float("nan")
        m, b = np.polyfit(np.arange(n, dtype=float), x, 1)
        return float(m * (n - 1) + b)

    momentum = delta.rolling(mom_period).apply(_linreg_last, raw=True)
    return momentum, squeeze_on.astype(bool)


def bb_percent_b(close: pd.Series, period: int = 20, std_dev: float = 2.0) -> pd.Series:
    """Bollinger %B — 0 = lower band, 1 = upper band."""
    upper, _, lower = bollinger_bands(close, period, std_dev)
    band_width = (upper - lower).replace(0, np.nan)
    return (close - lower) / band_width


def bb_width(close: pd.Series, period: int = 20, std_dev: float = 2.0) -> pd.Series:
    """Bollinger Band Width normalised by middle band."""
    upper, middle, lower = bollinger_bands(close, period, std_dev)
    return (upper - lower) / middle.replace(0, np.nan)
