"""
RSI Hidden Divergence Strategy.
Hidden bullish: price higher low + RSI lower low → long continuation.
Hidden bearish: price lower high + RSI higher high → short continuation.
MACD direction used as confirmation.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from crypto_bot.core.signals.base import BaseStrategy
from crypto_bot.core.signals.models import Signal
from crypto_bot.core.signals.indicators import rsi, macd, atr


class RSIDivergenceStrategy(BaseStrategy):
    @property
    def param_space(self) -> dict:
        return {
            "rsi_period":         (8,  21, "int"),
            "divergence_lookback": (3,  10, "int"),
            "macd_fast":          (8,  16, "int"),
            "macd_slow":          (20, 30, "int"),
            "macd_signal":        (7,  12, "int"),
            "atr_period":         (10, 20, "int"),
        }

    def generate_signals(self, candles: pd.DataFrame, aux_data=None) -> list[Signal]:
        p = self.params
        close  = candles["close"]
        high   = candles["high"]
        low    = candles["low"]
        symbol = str(candles["symbol"].iloc[0])

        rsi_v = rsi(close, int(p["rsi_period"]))
        _, _, macd_hist = macd(
            close,
            fast=int(p["macd_fast"]),
            slow=int(p["macd_slow"]),
            signal_period=int(p["macd_signal"]),
        )
        atr_v = atr(high, low, close, int(p.get("atr_period", 14)))
        lb = int(p["divergence_lookback"])

        signals: list[Signal] = []
        open_pos: str | None = None
        warmup = int(p["macd_slow"]) + int(p["macd_signal"]) + lb + 1

        # pre-convert to numpy — eliminates pandas .iloc overhead in hot loop
        is_clean_arr   = candles["is_clean"].values
        timestamps_arr = candles["timestamp"].dt.to_pydatetime()
        close_arr      = close.values
        high_arr       = high.values
        low_arr        = low.values
        rsi_arr        = rsi_v.values
        hist_arr       = macd_hist.values
        atr_arr        = atr_v.values

        for i in range(warmup, len(candles)):
            if not is_clean_arr[i]:
                continue
            rsi_i  = rsi_arr[i]
            atr_i  = atr_arr[i]
            hist_i = hist_arr[i]
            if np.isnan(rsi_i) or np.isnan(atr_i) or np.isnan(hist_i):
                continue

            # Lookback window (numpy slices — fast)
            low_window  = low_arr[i - lb:i]
            high_window = high_arr[i - lb:i]
            rsi_window  = rsi_arr[i - lb:i]
            if np.isnan(rsi_window).any():
                continue

            lo_i = low_arr[i]; hi_i = high_arr[i]

            # Hidden bullish: price makes higher low, RSI makes lower low
            hidden_bull = (
                lo_i > low_window.min()
                and rsi_i < rsi_window.min()
                and hist_i > 0
            )
            # Hidden bearish: price makes lower high, RSI makes higher high
            hidden_bear = (
                hi_i < high_window.max()
                and rsi_i > rsi_window.max()
                and hist_i < 0
            )

            direction: str | None = None
            reason: list[str] = []
            confirming = 0

            if hidden_bull and open_pos != "LONG":
                direction = "LONG"
                confirming = sum([lo_i > low_window.min(), rsi_i < rsi_window.min(), hist_i > 0])
                reason = ["hidden_bull_div", f"RSI={rsi_i:.1f}", "MACD_pos"]
                open_pos = "LONG"
            elif hidden_bear and open_pos != "SHORT":
                direction = "SHORT"
                confirming = 3
                reason = ["hidden_bear_div", f"RSI={rsi_i:.1f}", "MACD_neg"]
                open_pos = "SHORT"
            elif open_pos == "LONG" and hist_i < 0:
                direction = "EXIT_LONG"
                reason = ["MACD_turned_neg"]
                open_pos = None
            elif open_pos == "SHORT" and hist_i > 0:
                direction = "EXIT_SHORT"
                reason = ["MACD_turned_pos"]
                open_pos = None

            if direction:
                signals.append(Signal(
                    strategy=self.name,
                    symbol=symbol,
                    timestamp=timestamps_arr[i],
                    direction=direction,
                    strength=confirming / 3.0 if confirming else 0.5,
                    close_price=float(close_arr[i]),
                    atr=float(atr_i),
                    reason=reason,
                ))

        return signals
