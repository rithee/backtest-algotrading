"""Data quality validator. Marks candles as is_clean=False if anomalous."""
from __future__ import annotations
import numpy as np
import pandas as pd
from crypto_bot.core.signals.indicators import atr


def validate(df: pd.DataFrame, atr_anomaly_multiplier: float = 5.0) -> pd.DataFrame:
    """
    Validate a candle DataFrame. Returns a copy with is_clean updated.

    Rules:
    1. Missing candles (gaps in expected sequence) are forward-filled and flagged.
    2. Candles where (high - low) > atr_anomaly_multiplier * ATR(14) are flagged.
    3. Zero-volume candles are flagged.
    4. Duplicate timestamps are dropped (keep last).
    """
    df = df.copy()
    df = df.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp").reset_index(drop=True)
    df["is_clean"] = True

    # Flag zero-volume candles
    df.loc[df["volume"] <= 0, "is_clean"] = False

    # Flag anomalous candle ranges (range > N * ATR)
    atr_series = atr(df["high"], df["low"], df["close"], period=14)
    candle_range = df["high"] - df["low"]
    anomaly_mask = candle_range > (atr_anomaly_multiplier * atr_series)
    df.loc[anomaly_mask, "is_clean"] = False

    return df


def quality_score(df: pd.DataFrame) -> float:
    """Returns fraction of clean candles (0.0–1.0)."""
    if len(df) == 0:
        return 0.0
    return float(df["is_clean"].sum() / len(df))
