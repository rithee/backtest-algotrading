"""
Bollinger Band Mean Reversion Strategy.
Entry LONG: price touches/crosses below lower BB + RSI oversold + BB width (squeeze) below threshold.
Entry SHORT: price touches/crosses above upper BB + RSI overbought + BB width below threshold.
Target: middle band. Strategy signals exit at middle band cross.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from crypto_bot.core.signals.base import BaseStrategy
from crypto_bot.core.signals.models import Signal
from crypto_bot.core.signals.indicators import bollinger_bands, rsi, atr, bb_width


class BBMeanReversionStrategy(BaseStrategy):
    @property
    def param_space(self) -> dict:
        return {
            "bb_period":           (15, 30, "int"),
            "bb_std":              (1.8, 2.5),
            "rsi_period":          (8,  21, "int"),
            "rsi_oversold":        (25.0, 40.0),
            "rsi_overbought":      (60.0, 75.0),
            "bb_width_threshold":  (0.02, 0.08),
            "atr_period":          (10,  20, "int"),
        }

    def generate_signals(self, candles: pd.DataFrame, aux_data=None) -> list[Signal]:
        p = self.params
        close  = candles["close"]
        high   = candles["high"]
        low    = candles["low"]
        symbol = str(candles["symbol"].iloc[0])

        bb_upper, bb_mid, bb_lower = bollinger_bands(close, int(p["bb_period"]), float(p["bb_std"]))
        rsi_v  = rsi(close, int(p["rsi_period"]))
        bbw    = bb_width(close, int(p["bb_period"]), float(p["bb_std"]))
        atr_v  = atr(high, low, close, int(p.get("atr_period", 14)))

        signals: list[Signal] = []
        open_pos: str | None = None
        warmup = int(p["bb_period"]) + 1

        for i in range(warmup, len(candles)):
            if not candles["is_clean"].iloc[i]:
                continue
            c     = float(close.iloc[i])
            c_prev = float(close.iloc[i - 1])
            rsi_i  = rsi_v.iloc[i]
            bbw_i  = bbw.iloc[i]
            atr_i  = atr_v.iloc[i]
            ub     = float(bb_upper.iloc[i])
            lb     = float(bb_lower.iloc[i])
            mid    = float(bb_mid.iloc[i])

            if np.isnan(rsi_i) or np.isnan(bbw_i) or np.isnan(atr_i):
                continue

            squeezed = bbw_i < float(p["bb_width_threshold"])

            direction: str | None = None
            reason: list[str] = []
            confirming = 0

            if c <= lb and rsi_i < float(p["rsi_oversold"]) and squeezed and open_pos != "LONG":
                direction = "LONG"
                confirming = sum([c <= lb, rsi_i < float(p["rsi_oversold"]), squeezed])
                reason = [f"price_at_BB_lower", f"RSI={rsi_i:.1f}<{p['rsi_oversold']}", f"BBW={bbw_i:.3f}"]
                open_pos = "LONG"
            elif c >= ub and rsi_i > float(p["rsi_overbought"]) and squeezed and open_pos != "SHORT":
                direction = "SHORT"
                confirming = 3
                reason = [f"price_at_BB_upper", f"RSI={rsi_i:.1f}>{p['rsi_overbought']}", f"BBW={bbw_i:.3f}"]
                open_pos = "SHORT"
            elif open_pos == "LONG" and c_prev < mid <= c:
                direction = "EXIT_LONG"
                reason = ["price_cross_BB_mid"]
                open_pos = None
            elif open_pos == "SHORT" and c_prev > mid >= c:
                direction = "EXIT_SHORT"
                reason = ["price_cross_BB_mid"]
                open_pos = None

            if direction:
                signals.append(Signal(
                    strategy=self.name,
                    symbol=symbol,
                    timestamp=candles["timestamp"].iloc[i],
                    direction=direction,
                    strength=confirming / 3.0 if confirming else 0.5,
                    close_price=c,
                    atr=float(atr_i),
                    reason=reason,
                ))

        return signals
