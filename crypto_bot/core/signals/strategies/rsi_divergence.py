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

        for i in range(warmup, len(candles)):
            if not candles["is_clean"].iloc[i]:
                continue
            rsi_i = rsi_v.iloc[i]
            atr_i = atr_v.iloc[i]
            hist_i = macd_hist.iloc[i]
            if np.isnan(rsi_i) or np.isnan(atr_i) or np.isnan(hist_i):
                continue

            # Lookback window
            low_window   = low.iloc[i - lb:i]
            high_window  = high.iloc[i - lb:i]
            rsi_window   = rsi_v.iloc[i - lb:i]
            if rsi_window.isna().any():
                continue

            # Hidden bullish: price makes higher low, RSI makes lower low
            hidden_bull = (
                float(low.iloc[i]) > float(low_window.min())
                and rsi_i < float(rsi_window.min())
                and hist_i > 0  # MACD confirms upward momentum
            )
            # Hidden bearish: price makes lower high, RSI makes higher high
            hidden_bear = (
                float(high.iloc[i]) < float(high_window.max())
                and rsi_i > float(rsi_window.max())
                and hist_i < 0
            )

            direction: str | None = None
            reason: list[str] = []
            confirming = 0

            if hidden_bull and open_pos != "LONG":
                direction = "LONG"
                confirming = sum([
                    float(low.iloc[i]) > float(low_window.min()),
                    rsi_i < float(rsi_window.min()),
                    hist_i > 0,
                ])
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
                    timestamp=candles["timestamp"].iloc[i],
                    direction=direction,
                    strength=confirming / 3.0 if confirming else 0.5,
                    close_price=float(close.iloc[i]),
                    atr=float(atr_i),
                    reason=reason,
                ))

        return signals
