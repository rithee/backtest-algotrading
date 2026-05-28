"""
MACD Histogram Divergence Strategy.
Bullish divergence: price lower low + histogram higher low + histogram turning positive.
Bearish divergence: price higher high + histogram lower high + histogram turning negative.
Exit: histogram crosses zero in opposite direction.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from crypto_bot.core.signals.base import BaseStrategy
from crypto_bot.core.signals.models import Signal
from crypto_bot.core.signals.indicators import macd, atr


class MACDHistDivergenceStrategy(BaseStrategy):
    @property
    def param_space(self) -> dict:
        return {
            "macd_fast":           (8,  16, "int"),
            "macd_slow":           (20, 30, "int"),
            "macd_signal":         (7,  12, "int"),
            "divergence_lookback": (3,  10, "int"),
            "atr_period":          (10, 20, "int"),
        }

    def generate_signals(self, candles: pd.DataFrame, aux_data=None) -> list[Signal]:
        p      = self.params
        close  = candles["close"]
        high   = candles["high"]
        low    = candles["low"]
        symbol = str(candles["symbol"].iloc[0])

        fast_p = int(p["macd_fast"])
        slow_p = int(p["macd_slow"])
        sig_p  = int(p["macd_signal"])
        lb     = int(p["divergence_lookback"])
        atr_p  = int(p["atr_period"])

        _, _, hist = macd(close, fast=fast_p, slow=slow_p, signal_period=sig_p)
        atr_v      = atr(high, low, close, atr_p)
        warmup     = slow_p + sig_p + lb + 1

        signals: list[Signal] = []
        open_pos: str | None  = None

        # pre-convert to numpy — eliminates pandas .iloc overhead in hot loop
        is_clean_arr   = candles["is_clean"].values
        timestamps_arr = candles["timestamp"].dt.to_pydatetime()
        close_arr      = close.values
        hist_arr       = hist.values
        atr_arr        = atr_v.values

        for i in range(warmup, len(candles)):
            if not is_clean_arr[i]:
                continue

            hist_i  = hist_arr[i]
            hist_p  = hist_arr[i - 1]
            atr_i   = atr_arr[i]
            cl_i    = close_arr[i]
            hist_lb = hist_arr[i - lb]
            cl_lb   = close_arr[i - lb]

            if any(np.isnan(v) for v in (hist_i, hist_p, atr_i, hist_lb)):
                continue

            bullish_div = (
                cl_i   < cl_lb    and
                hist_i > hist_lb  and
                hist_i < 0        and
                hist_i > hist_p
            )
            bearish_div = (
                cl_i   > cl_lb    and
                hist_i < hist_lb  and
                hist_i > 0        and
                hist_i < hist_p
            )

            direction: str | None = None
            reason: list[str]     = []

            if bullish_div and open_pos != "LONG":
                direction = "LONG"
                reason    = ["bullish_div", f"hist_lb={hist_lb:.4f}", f"hist={hist_i:.4f}"]
                open_pos  = "LONG"
            elif bearish_div and open_pos != "SHORT":
                direction = "SHORT"
                reason    = ["bearish_div", f"hist_lb={hist_lb:.4f}", f"hist={hist_i:.4f}"]
                open_pos  = "SHORT"
            elif open_pos == "LONG" and hist_i >= 0 and hist_p < 0:
                direction = "EXIT_LONG"
                reason    = ["hist_zero_cross_up"]
                open_pos  = None
            elif open_pos == "SHORT" and hist_i <= 0 and hist_p > 0:
                direction = "EXIT_SHORT"
                reason    = ["hist_zero_cross_down"]
                open_pos  = None

            if direction:
                strength = min(abs(hist_i) / max(atr_i * 0.01, 1e-9), 1.0)
                signals.append(Signal(
                    strategy=self.name,
                    symbol=symbol,
                    timestamp=timestamps_arr[i],
                    direction=direction,
                    strength=min(strength, 1.0),
                    close_price=float(cl_i),
                    atr=float(atr_i),
                    reason=reason,
                ))

        return signals
