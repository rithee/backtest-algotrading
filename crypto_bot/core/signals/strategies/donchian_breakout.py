"""
Donchian Channel Breakout with Volume + ATR Expansion confirmation.

Entry logic:
  LONG  — close breaks above the Donchian upper band AND
           volume > rolling_avg_volume × vol_mult AND
           current ATR > rolling_avg_ATR × atr_expansion_mult
  SHORT — close breaks below the Donchian lower band (same filters)

Exit logic:
  LONG  — close crosses back below the Donchian midline
  SHORT — close crosses back above the Donchian midline

Why it's explosive:
  Crypto's largest directional moves are breakouts — BTC above $20k (Nov 2020),
  $60k (Mar 2021), ETH DeFi summer, 2022 bear legs. This strategy fires only when
  a genuine breakout has volume behind it and volatility is expanding, so it catches
  the fat-tail moves while skipping false breakouts.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from crypto_bot.core.signals.base import BaseStrategy
from crypto_bot.core.signals.models import Signal
from crypto_bot.core.signals.indicators import donchian_channels, atr


class DonchianBreakoutStrategy(BaseStrategy):
    """Donchian channel breakout filtered by volume surge and ATR expansion."""

    @property
    def param_space(self) -> dict[str, tuple]:
        return {
            "channel_period":      (10, 50, "int"),
            "vol_period":          (10, 30, "int"),
            "vol_mult":            (1.2, 2.5),
            "atr_period":          (10, 21, "int"),
            "atr_expansion_mult":  (1.1, 1.8),
        }

    def generate_signals(self, candles: pd.DataFrame, aux_data: dict | None = None) -> list[Signal]:
        p = self.params
        channel_period     = int(p.get("channel_period",     20))
        vol_period         = int(p.get("vol_period",         20))
        vol_mult           = float(p.get("vol_mult",          1.5))
        atr_period         = int(p.get("atr_period",         14))
        atr_expansion_mult = float(p.get("atr_expansion_mult", 1.3))

        warmup = max(channel_period, vol_period, atr_period) + 1
        if len(candles) < warmup:
            return []

        close  = candles["close"]
        high   = candles["high"]
        low    = candles["low"]
        volume = candles["volume"]

        upper, middle, lower = donchian_channels(high, low, channel_period)
        atr_vals    = atr(high, low, close, atr_period)
        avg_atr     = atr_vals.rolling(vol_period).mean()
        avg_volume  = volume.rolling(vol_period).mean()

        symbol = str(candles["symbol"].iloc[0])
        signals: list[Signal] = []

        # pre-convert to numpy — eliminates pandas .iloc overhead in hot loop
        is_clean_arr   = candles["is_clean"].values
        timestamps_arr = candles["timestamp"].dt.to_pydatetime()
        close_arr      = close.values
        volume_arr     = volume.values
        upper_arr      = upper.values
        lower_arr      = lower.values
        middle_arr     = middle.values
        atr_arr        = atr_vals.values
        avg_atr_arr    = avg_atr.values
        avg_vol_arr    = avg_volume.values

        for i in range(warmup, len(candles)):
            if not is_clean_arr[i]:
                continue

            c      = close_arr[i]
            prev_c = close_arr[i - 1]
            u      = upper_arr[i - 1]   # previous bar's channel (no lookahead)
            l      = lower_arr[i - 1]
            mid    = middle_arr[i - 1]
            vol    = volume_arr[i]
            avg_vol  = avg_vol_arr[i]
            cur_atr  = atr_arr[i]
            mean_atr = avg_atr_arr[i]

            if any(np.isnan(x) for x in [u, l, mid, avg_vol, mean_atr]):
                continue

            volume_surge = avg_vol > 0 and vol > avg_vol * vol_mult
            atr_expand   = mean_atr > 0 and cur_atr > mean_atr * atr_expansion_mult

            # Long breakout
            if prev_c <= u and c > u and volume_surge and atr_expand:
                signals.append(Signal(
                    strategy=self.name,
                    symbol=symbol,
                    timestamp=timestamps_arr[i],
                    direction="LONG",
                    strength=min(1.0, (c - u) / (u * 0.005 + 1e-9)),
                    close_price=float(c),
                    atr=float(cur_atr),
                    reason=["donchian_upper_break", "volume_surge", "atr_expansion"],
                ))

            # Short breakout
            elif prev_c >= l and c < l and volume_surge and atr_expand:
                signals.append(Signal(
                    strategy=self.name,
                    symbol=symbol,
                    timestamp=timestamps_arr[i],
                    direction="SHORT",
                    strength=min(1.0, (l - c) / (l * 0.005 + 1e-9)),
                    close_price=float(c),
                    atr=float(cur_atr),
                    reason=["donchian_lower_break", "volume_surge", "atr_expansion"],
                ))

            # Long exit: close crossed back below midline
            elif c < mid and prev_c >= mid:
                signals.append(Signal(
                    strategy=self.name,
                    symbol=symbol,
                    timestamp=timestamps_arr[i],
                    direction="EXIT_LONG",
                    strength=1.0,
                    close_price=float(c),
                    atr=float(cur_atr),
                    reason=["price_below_midline"],
                ))

            # Short exit: close crossed back above midline
            elif c > mid and prev_c <= mid:
                signals.append(Signal(
                    strategy=self.name,
                    symbol=symbol,
                    timestamp=timestamps_arr[i],
                    direction="EXIT_SHORT",
                    strength=1.0,
                    close_price=float(c),
                    atr=float(cur_atr),
                    reason=["price_above_midline"],
                ))

        return signals
