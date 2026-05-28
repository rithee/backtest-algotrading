"""
Open Interest Divergence Strategy.
LONG : OI rising + price falling (shorts trapped → squeeze incoming) + RSI < 45.
SHORT: OI rising + price rising  (longs trapped → unwind incoming)  + RSI > 55.
Exit : OI drops sharply (positions unwinding).
Requires aux_data['open_interest'] — pd.Series with timestamp index.
Returns [] gracefully if aux_data is absent.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from crypto_bot.core.signals.base import BaseStrategy
from crypto_bot.core.signals.models import Signal
from crypto_bot.core.signals.indicators import rsi, atr


class OpenInterestDivergenceStrategy(BaseStrategy):
    @property
    def param_space(self) -> dict:
        return {
            "oi_change_period": (3,   10,  "int"),
            "oi_surge_pct":     (0.5, 3.0),
            "rsi_period":       (10,  21,  "int"),
            "atr_period":       (10,  20,  "int"),
        }

    def generate_signals(self, candles: pd.DataFrame, aux_data=None) -> list[Signal]:
        # Graceful degradation — no OI data, no signals
        if aux_data is None or "open_interest" not in aux_data:
            return []

        p      = self.params
        close  = candles["close"]
        high   = candles["high"]
        low    = candles["low"]
        symbol = str(candles["symbol"].iloc[0])

        oi_period = int(p["oi_change_period"])
        surge_pct = float(p["oi_surge_pct"]) / 100.0
        rsi_p     = int(p["rsi_period"])
        atr_p     = int(p["atr_period"])

        rsi_v = rsi(close, rsi_p)
        atr_v = atr(high, low, close, atr_p)

        # Align OI to candle timestamps via forward-fill
        oi_raw: pd.Series = aux_data["open_interest"]
        oi_aligned = oi_raw.reindex(candles["timestamp"].tolist(), method="ffill")

        warmup = oi_period + rsi_p + 1
        signals: list[Signal] = []
        open_pos: str | None  = None

        # pre-convert to numpy — eliminates pandas .iloc overhead in hot loop
        is_clean_arr   = candles["is_clean"].values
        timestamps_arr = candles["timestamp"].dt.to_pydatetime()
        close_arr      = close.values
        rsi_arr        = rsi_v.values
        atr_arr        = atr_v.values
        oi_arr         = oi_aligned.values

        for i in range(warmup, len(candles)):
            if not is_clean_arr[i]:
                continue

            rsi_i   = rsi_arr[i]
            atr_i   = atr_arr[i]
            cl_i    = close_arr[i]
            cl_lb   = close_arr[i - oi_period]
            oi_curr = oi_arr[i]
            oi_prev = oi_arr[i - oi_period]

            if any(np.isnan(v) for v in (rsi_i, atr_i, oi_curr, oi_prev)):
                continue
            if oi_prev == 0:
                continue

            oi_change    = (oi_curr - oi_prev) / oi_prev
            price_change = (cl_i - cl_lb) / max(cl_lb, 1e-9)

            oi_rising     = oi_change >  surge_pct
            oi_unwinding  = oi_change < -surge_pct
            price_falling = price_change < -0.001
            price_rising  = price_change >  0.001

            direction: str | None = None
            reason: list[str]     = []

            if oi_rising and price_falling and rsi_i < 45 and open_pos != "LONG":
                direction = "LONG"
                reason    = ["OI_rising", "price_falling", f"RSI={rsi_i:.1f}"]
                open_pos  = "LONG"
            elif oi_rising and price_rising and rsi_i > 55 and open_pos != "SHORT":
                direction = "SHORT"
                reason    = ["OI_rising", "price_rising", f"RSI={rsi_i:.1f}"]
                open_pos  = "SHORT"
            elif open_pos == "LONG" and oi_unwinding:
                direction = "EXIT_LONG"
                reason    = ["OI_unwinding"]
                open_pos  = None
            elif open_pos == "SHORT" and oi_unwinding:
                direction = "EXIT_SHORT"
                reason    = ["OI_unwinding"]
                open_pos  = None

            if direction:
                signals.append(Signal(
                    strategy=self.name,
                    symbol=symbol,
                    timestamp=timestamps_arr[i],
                    direction=direction,
                    strength=min(abs(oi_change) / max(surge_pct * 3, 1e-9), 1.0),
                    close_price=float(cl_i),
                    atr=float(atr_i),
                    reason=reason,
                ))

        return signals
