"""
TTM Squeeze (LazyBear) Breakout Strategy.
Squeeze ON = BB inside Keltner = compression.
Squeeze OFF + momentum direction = entry signal.
Exit when momentum histogram reverses.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from crypto_bot.core.signals.base import BaseStrategy
from crypto_bot.core.signals.models import Signal
from crypto_bot.core.signals.indicators import squeeze_momentum, atr


class TTMSqueezeStrategy(BaseStrategy):
    @property
    def param_space(self) -> dict:
        return {
            "bb_period":  (15, 25, "int"),
            "bb_std":     (1.5, 2.5),
            "kc_period":  (15, 25, "int"),
            "kc_mult":    (1.0, 2.0),
            "mom_period": (10, 20, "int"),
            "atr_period": (10, 20, "int"),
        }

    def generate_signals(self, candles: pd.DataFrame, aux_data=None) -> list[Signal]:
        p = self.params
        close  = candles["close"]
        high   = candles["high"]
        low    = candles["low"]
        symbol = str(candles["symbol"].iloc[0])

        momentum, squeeze_on = squeeze_momentum(
            high, low, close,
            bb_period=int(p["bb_period"]),
            bb_std=float(p["bb_std"]),
            kc_period=int(p["kc_period"]),
            kc_mult=float(p["kc_mult"]),
            mom_period=int(p["mom_period"]),
        )
        atr_v = atr(high, low, close, int(p.get("atr_period", 14)))

        signals: list[Signal] = []
        open_pos: str | None = None
        warmup = int(p["kc_period"]) + int(p["mom_period"]) + 1

        for i in range(warmup, len(candles)):
            if not candles["is_clean"].iloc[i]:
                continue
            mom_i   = momentum.iloc[i]
            mom_prev = momentum.iloc[i - 1]
            sq_i    = squeeze_on.iloc[i]
            atr_i   = atr_v.iloc[i]
            if np.isnan(mom_i) or np.isnan(mom_prev) or np.isnan(atr_i):
                continue

            # Squeeze fired: was ON, now OFF
            squeeze_fired = squeeze_on.iloc[i - 1] and not sq_i
            direction: str | None = None
            reason: list[str] = []

            if squeeze_fired and mom_i > 0 and mom_i > mom_prev and open_pos != "LONG":
                direction = "LONG"
                reason = ["squeeze_fire_up", f"mom={mom_i:.2f}"]
                open_pos = "LONG"
            elif squeeze_fired and mom_i < 0 and mom_i < mom_prev and open_pos != "SHORT":
                direction = "SHORT"
                reason = ["squeeze_fire_down", f"mom={mom_i:.2f}"]
                open_pos = "SHORT"
            elif open_pos == "LONG" and mom_i < 0:
                direction = "EXIT_LONG"
                reason = ["mom_turned_neg"]
                open_pos = None
            elif open_pos == "SHORT" and mom_i > 0:
                direction = "EXIT_SHORT"
                reason = ["mom_turned_pos"]
                open_pos = None

            if direction:
                signals.append(Signal(
                    strategy=self.name,
                    symbol=symbol,
                    timestamp=candles["timestamp"].iloc[i],
                    direction=direction,
                    strength=min(abs(mom_i) / (atr_i + 1e-9), 1.0),
                    close_price=float(close.iloc[i]),
                    atr=float(atr_i),
                    reason=reason,
                ))

        return signals
