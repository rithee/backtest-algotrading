"""
EMA Ribbon Trend Follow Strategy.
Entry: all 4 EMAs aligned in bull/bear order + ADX confirms trend strength.
Exit: fastest EMA crosses back through second EMA.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from crypto_bot.core.signals.base import BaseStrategy
from crypto_bot.core.signals.models import Signal
from crypto_bot.core.signals.indicators import ema, adx, atr


class EMARibbonStrategy(BaseStrategy):
    @property
    def param_space(self) -> dict:
        return {
            "ema_fast":      (5,   15,  "int"),
            "ema_medium":    (15,  30,  "int"),
            "ema_slow":      (30,  60,  "int"),
            "ema_trend":     (100, 250, "int"),
            "adx_period":    (10,  20,  "int"),
            "adx_threshold": (15.0, 30.0),
            "atr_period":    (10,  20,  "int"),
        }

    def generate_signals(self, candles: pd.DataFrame, aux_data=None) -> list[Signal]:
        p = self.params
        close = candles["close"]
        high  = candles["high"]
        low   = candles["low"]
        symbol = str(candles["symbol"].iloc[0])

        ema_f = ema(close, int(p["ema_fast"]))
        ema_m = ema(close, int(p["ema_medium"]))
        ema_s = ema(close, int(p["ema_slow"]))
        ema_t = ema(close, int(p["ema_trend"]))
        adx_v = adx(high, low, close, int(p["adx_period"]))
        atr_v = atr(high, low, close, int(p.get("atr_period", 14)))

        signals: list[Signal] = []
        open_pos: str | None = None  # "LONG" | "SHORT" | None
        adx_thresh = float(p["adx_threshold"])
        warmup = int(p["ema_trend"]) + 1

        for i in range(warmup, len(candles)):
            if not candles["is_clean"].iloc[i]:
                continue
            ef = ema_f.iloc[i]; em = ema_m.iloc[i]
            es = ema_s.iloc[i]; et = ema_t.iloc[i]
            adx_i = adx_v.iloc[i]; atr_i = atr_v.iloc[i]
            if np.isnan(ef) or np.isnan(et) or np.isnan(adx_i) or np.isnan(atr_i):
                continue

            bull = ef > em > es > et
            bear = ef < em < es < et
            strong = adx_i > adx_thresh

            direction: str | None = None
            reason: list[str] = []

            if bull and strong and open_pos != "LONG":
                direction = "LONG"
                reason = [f"EMA_bull_ribbon", f"ADX={adx_i:.1f}>{adx_thresh}"]
                open_pos = "LONG"
            elif bear and strong and open_pos != "SHORT":
                direction = "SHORT"
                reason = [f"EMA_bear_ribbon", f"ADX={adx_i:.1f}>{adx_thresh}"]
                open_pos = "SHORT"
            elif open_pos == "LONG" and not (ef > em):
                direction = "EXIT_LONG"
                reason = ["EMA_fast_cross_down"]
                open_pos = None
            elif open_pos == "SHORT" and not (ef < em):
                direction = "EXIT_SHORT"
                reason = ["EMA_fast_cross_up"]
                open_pos = None

            if direction:
                signals.append(Signal(
                    strategy=self.name,
                    symbol=symbol,
                    timestamp=candles["timestamp"].iloc[i],
                    direction=direction,
                    strength=min(adx_i / 50.0, 1.0),
                    close_price=float(close.iloc[i]),
                    atr=float(atr_i),
                    reason=reason,
                ))

        return signals
