"""
Market Regime Strategy.
Detects market regime from ADX + EMA slope + ATR volatility.
UPTREND   (ADX > trend_thresh AND EMA slope > 0) → LONG on regime entry.
DOWNTREND (ADX > trend_thresh AND EMA slope < 0) → SHORT on regime entry.
RANGING   (ADX < range_thresh)                   → EXIT open position.
VOLATILE  (ATR > vol_thresh × ATR rolling mean)  → EXIT open position.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from crypto_bot.core.signals.base import BaseStrategy
from crypto_bot.core.signals.models import Signal
from crypto_bot.core.signals.indicators import adx, ema_slope, atr


class MarketRegimeStrategy(BaseStrategy):
    @property
    def param_space(self) -> dict:
        return {
            "adx_period":           (10,  20,  "int"),
            "adx_trend_threshold":  (20.0, 30.0),
            "adx_range_threshold":  (15.0, 25.0),
            "ema_period":           (50,  200, "int"),
            "slope_lookback":       (3,   10,  "int"),
            "atr_vol_threshold":    (1.5, 3.0),
            "atr_period":           (10,  20,  "int"),
        }

    def generate_signals(self, candles: pd.DataFrame, aux_data=None) -> list[Signal]:
        p      = self.params
        close  = candles["close"]
        high   = candles["high"]
        low    = candles["low"]
        symbol = str(candles["symbol"].iloc[0])

        adx_p        = int(p["adx_period"])
        trend_thresh = float(p["adx_trend_threshold"])
        range_thresh = float(p["adx_range_threshold"])
        ema_p        = int(p["ema_period"])
        slope_lb     = int(p["slope_lookback"])
        vol_thresh   = float(p["atr_vol_threshold"])
        atr_p        = int(p["atr_period"])

        adx_v    = adx(high, low, close, adx_p)
        slope_v  = ema_slope(close, ema_p, slope_lb)
        atr_v    = atr(high, low, close, atr_p)
        atr_mean = atr_v.rolling(20).mean()

        warmup = ema_p + slope_lb + 1
        signals: list[Signal] = []
        open_pos: str | None  = None
        prev_regime: str      = "UNKNOWN"

        # pre-convert to numpy — eliminates pandas .iloc overhead in hot loop
        is_clean_arr   = candles["is_clean"].values
        timestamps_arr = candles["timestamp"].dt.to_pydatetime()
        close_arr      = close.values
        adx_arr        = adx_v.values
        slope_arr      = slope_v.values
        atr_arr        = atr_v.values
        atr_mean_arr   = atr_mean.values

        for i in range(warmup, len(candles)):
            if not is_clean_arr[i]:
                continue

            adx_i   = adx_arr[i]
            slope_i = slope_arr[i]
            atr_i   = atr_arr[i]
            atr_m   = atr_mean_arr[i]

            if any(np.isnan(v) for v in (adx_i, slope_i, atr_i, atr_m)):
                continue

            if atr_i > vol_thresh * atr_m:
                regime = "VOLATILE"
            elif adx_i > trend_thresh:
                regime = "UPTREND" if slope_i > 0 else "DOWNTREND"
            elif adx_i < range_thresh:
                regime = "RANGING"
            else:
                regime = "TRANSITION"

            direction: str | None = None
            reason: list[str]     = []

            if regime != prev_regime:
                if regime == "UPTREND" and open_pos != "LONG":
                    direction = "LONG"
                    reason    = ["regime=UPTREND", f"ADX={adx_i:.1f}", f"slope={slope_i:.4f}"]
                    open_pos  = "LONG"
                elif regime == "DOWNTREND" and open_pos != "SHORT":
                    direction = "SHORT"
                    reason    = ["regime=DOWNTREND", f"ADX={adx_i:.1f}", f"slope={slope_i:.4f}"]
                    open_pos  = "SHORT"
                elif regime in ("RANGING", "VOLATILE"):
                    if open_pos == "LONG":
                        direction = "EXIT_LONG"
                        reason    = [f"regime={regime}"]
                        open_pos  = None
                    elif open_pos == "SHORT":
                        direction = "EXIT_SHORT"
                        reason    = [f"regime={regime}"]
                        open_pos  = None

            prev_regime = regime

            if direction:
                signals.append(Signal(
                    strategy=self.name,
                    symbol=symbol,
                    timestamp=timestamps_arr[i],
                    direction=direction,
                    strength=min(adx_i / 50.0, 1.0),
                    close_price=float(close_arr[i]),
                    atr=float(atr_i),
                    reason=reason,
                ))

        return signals
