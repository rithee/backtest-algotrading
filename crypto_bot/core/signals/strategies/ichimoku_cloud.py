"""
Ichimoku Cloud Strategy.
Entry LONG : transition into full bullish state:
             price above cloud AND tenkan >= kijun AND chikou above price[i-26].
Entry SHORT: transition into full bearish state:
             price below cloud AND tenkan <= kijun AND chikou below price[i-26].
Exit       : tenkan/kijun cross reverses OR price closes inside cloud.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from crypto_bot.core.signals.base import BaseStrategy
from crypto_bot.core.signals.models import Signal
from crypto_bot.core.signals.indicators import ichimoku, atr


class IchimokuCloudStrategy(BaseStrategy):
    @property
    def param_space(self) -> dict:
        return {
            "tenkan_period":   (7,  12, "int"),
            "kijun_period":    (22, 30, "int"),
            "senkou_b_period": (44, 60, "int"),
            "atr_period":      (10, 20, "int"),
        }

    def generate_signals(self, candles: pd.DataFrame, aux_data=None) -> list[Signal]:
        p      = self.params
        close  = candles["close"]
        high   = candles["high"]
        low    = candles["low"]
        symbol = str(candles["symbol"].iloc[0])

        displacement = 26  # standard Ichimoku constant
        tenkan_p     = int(p["tenkan_period"])
        kijun_p      = int(p["kijun_period"])
        senkou_b_p   = int(p["senkou_b_period"])
        atr_p        = int(p["atr_period"])

        tenkan, kijun, span_a, span_b = ichimoku(
            high, low,
            tenkan_period=tenkan_p,
            kijun_period=kijun_p,
            senkou_b_period=senkou_b_p,
            displacement=displacement,
        )
        atr_v  = atr(high, low, close, atr_p)
        warmup = senkou_b_p + displacement + 1

        signals: list[Signal] = []
        open_pos: str | None  = None
        prev_bullish           = False
        prev_bearish           = False

        for i in range(warmup, len(candles)):
            if not candles["is_clean"].iloc[i]:
                continue

            t_i   = tenkan.iloc[i]
            k_i   = kijun.iloc[i]
            sa_i  = span_a.iloc[i]
            sb_i  = span_b.iloc[i]
            atr_i = atr_v.iloc[i]
            cl_i  = close.iloc[i]

            if any(np.isnan(v) for v in (t_i, k_i, sa_i, sb_i, atr_i)):
                continue

            cloud_top    = max(sa_i, sb_i)
            cloud_bottom = min(sa_i, sb_i)
            above_cloud  = cl_i > cloud_top
            below_cloud  = cl_i < cloud_bottom
            inside_cloud = cloud_bottom <= cl_i <= cloud_top

            tk_bull = t_i >= k_i   # tenkan at or above kijun
            tk_bear = t_i <= k_i   # tenkan at or below kijun

            chikou_above = cl_i > close.iloc[i - displacement]
            chikou_below = cl_i < close.iloc[i - displacement]

            # Full bullish/bearish state (all three Ichimoku pillars aligned)
            bullish = above_cloud and tk_bull and chikou_above
            bearish = below_cloud and tk_bear and chikou_below

            direction: str | None = None
            reason: list[str]     = []

            # Entry on transition INTO bullish/bearish state
            if bullish and not prev_bullish and open_pos != "LONG":
                direction = "LONG"
                reason    = ["above_cloud", "tenkan>=kijun", "chikou_above"]
                open_pos  = "LONG"
            elif bearish and not prev_bearish and open_pos != "SHORT":
                direction = "SHORT"
                reason    = ["below_cloud", "tenkan<=kijun", "chikou_below"]
                open_pos  = "SHORT"
            elif open_pos == "LONG" and (tk_bear or inside_cloud):
                direction = "EXIT_LONG"
                reason    = ["tenkan_below_kijun" if tk_bear else "price_in_cloud"]
                open_pos  = None
            elif open_pos == "SHORT" and (tk_bull or inside_cloud):
                direction = "EXIT_SHORT"
                reason    = ["tenkan_above_kijun" if tk_bull else "price_in_cloud"]
                open_pos  = None

            prev_bullish = bullish
            prev_bearish = bearish

            if direction:
                dist     = abs(cl_i - (cloud_top if above_cloud else cloud_bottom))
                strength = min(dist / max(atr_i, 1e-9), 1.0)
                signals.append(Signal(
                    strategy=self.name,
                    symbol=symbol,
                    timestamp=candles["timestamp"].iloc[i],
                    direction=direction,
                    strength=strength,
                    close_price=float(cl_i),
                    atr=float(atr_i),
                    reason=reason,
                ))

        return signals
