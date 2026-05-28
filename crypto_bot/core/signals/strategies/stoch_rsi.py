"""
Stochastic RSI Strategy.
Entry LONG : %K crosses above %D while both below 20 (oversold zone).
Entry SHORT: %K crosses below %D while both above 80 (overbought zone).
Exit       : %K/%D cross reverses OR both enter neutral zone (40-60).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from crypto_bot.core.signals.base import BaseStrategy
from crypto_bot.core.signals.models import Signal
from crypto_bot.core.signals.indicators import stoch_rsi, atr


class StochRSIStrategy(BaseStrategy):
    @property
    def param_space(self) -> dict:
        return {
            "rsi_period":   (10, 21, "int"),
            "stoch_period": (10, 21, "int"),
            "smooth_k":     (2,  5,  "int"),
            "smooth_d":     (2,  5,  "int"),
            "atr_period":   (10, 20, "int"),
        }

    def generate_signals(self, candles: pd.DataFrame, aux_data=None) -> list[Signal]:
        p      = self.params
        close  = candles["close"]
        high   = candles["high"]
        low    = candles["low"]
        symbol = str(candles["symbol"].iloc[0])

        rsi_p   = int(p["rsi_period"])
        stoch_p = int(p["stoch_period"])
        sk      = int(p["smooth_k"])
        sd      = int(p["smooth_d"])
        atr_p   = int(p["atr_period"])

        k_series, d_series = stoch_rsi(close, rsi_p, stoch_p, sk, sd)
        atr_v  = atr(high, low, close, atr_p)
        warmup = rsi_p + stoch_p + sk + sd + 1

        signals: list[Signal] = []
        open_pos: str | None  = None

        # pre-convert to numpy — eliminates pandas .iloc overhead in hot loop
        is_clean_arr   = candles["is_clean"].values
        timestamps_arr = candles["timestamp"].dt.to_pydatetime()
        close_arr      = close.values
        k_arr          = k_series.values
        d_arr          = d_series.values
        atr_arr        = atr_v.values

        for i in range(warmup, len(candles)):
            if not is_clean_arr[i]:
                continue

            k_i   = k_arr[i];  k_p = k_arr[i - 1]
            d_i   = d_arr[i];  d_p = d_arr[i - 1]
            atr_i = atr_arr[i]

            if any(np.isnan(v) for v in (k_i, d_i, k_p, d_p, atr_i)):
                continue

            k_cross_up   = (k_i > d_i)  and (k_p <= d_p)
            k_cross_down = (k_i < d_i)  and (k_p >= d_p)
            oversold     = k_i < 20     and d_i < 20
            overbought   = k_i > 80     and d_i > 80
            neutral      = 40.0 <= k_i  <= 60.0

            direction: str | None = None
            reason: list[str]     = []

            if k_cross_up and oversold and open_pos != "LONG":
                direction = "LONG"
                reason    = ["k_cross_up", f"k={k_i:.1f}<20", f"d={d_i:.1f}<20"]
                open_pos  = "LONG"
            elif k_cross_down and overbought and open_pos != "SHORT":
                direction = "SHORT"
                reason    = ["k_cross_down", f"k={k_i:.1f}>80", f"d={d_i:.1f}>80"]
                open_pos  = "SHORT"
            elif open_pos == "LONG" and (k_cross_down or neutral):
                direction = "EXIT_LONG"
                reason    = ["k_cross_down" if k_cross_down else "neutral_zone"]
                open_pos  = None
            elif open_pos == "SHORT" and (k_cross_up or neutral):
                direction = "EXIT_SHORT"
                reason    = ["k_cross_up" if k_cross_up else "neutral_zone"]
                open_pos  = None

            if direction:
                if direction == "LONG":
                    strength = min((20 - min(k_i, d_i)) / 20.0, 1.0)
                elif direction == "SHORT":
                    strength = min((max(k_i, d_i) - 80) / 20.0, 1.0)
                else:
                    strength = 0.5
                signals.append(Signal(
                    strategy=self.name,
                    symbol=symbol,
                    timestamp=timestamps_arr[i],
                    direction=direction,
                    strength=max(0.0, min(strength, 1.0)),
                    close_price=float(close_arr[i]),
                    atr=float(atr_i),
                    reason=reason,
                ))

        return signals
