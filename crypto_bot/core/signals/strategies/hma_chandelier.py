"""
Hull Moving Average Crossover with Chandelier Exit trailing stop.

The Hull MA (Alan Hull, 2005) uses weighted MAs to achieve near-zero lag —
it actually leads price slightly rather than lagging it. This means crossovers
signal trend changes at the inflection point rather than 5-8 bars later,
which is the main failure mode of standard EMA ribbon strategies.

Chandelier Exit (Chuck LeBeau) adapts the trailing stop to volatility:
  Long stop  = highest_high(period) − k × ATR(period)
  Short stop = lowest_low(period)   + k × ATR(period)
This lets winners run in trending markets while cutting losers quickly in chop.

Reported: Sharpe ~1.0–1.5 on BTC/ETH 4H, profit factor 1.61 in backtests.

Entry:
  LONG  — HMA(fast) crosses above HMA(slow) AND close > HMA(trend) AND volume > avg
  SHORT — HMA(fast) crosses below HMA(slow) AND close < HMA(trend) AND volume > avg

Exit:
  Chandelier stop hit (trailing) OR HMA crossover reverses
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from crypto_bot.core.signals.base import BaseStrategy
from crypto_bot.core.signals.models import Signal
from crypto_bot.core.signals.indicators import hma, chandelier_exit, atr


class HMAChandelierStrategy(BaseStrategy):
    """Hull MA crossover with Chandelier Exit adaptive trailing stop."""

    @property
    def param_space(self) -> dict[str, tuple]:
        return {
            "hma_fast":       (5, 15, "int"),
            "hma_slow":       (16, 35, "int"),
            "hma_trend":      (40, 80, "int"),
            "vol_period":     (15, 30, "int"),   # volume average period
            "vol_mult":       (1.0, 2.0),         # volume confirmation threshold
            "chandelier_period": (14, 30, "int"),
            "chandelier_mult":   (2.0, 4.0),
        }

    def generate_signals(self, candles: pd.DataFrame, aux_data: dict | None = None) -> list[Signal]:
        p = self.params
        fast    = int(p.get("hma_fast",           9))
        slow    = int(p.get("hma_slow",           21))
        trend   = int(p.get("hma_trend",          50))
        vp      = int(p.get("vol_period",         20))
        vm      = float(p.get("vol_mult",         1.2))
        cp      = int(p.get("chandelier_period",  22))
        cm      = float(p.get("chandelier_mult",  3.0))

        warmup = max(slow, trend, cp, vp) + int(np.sqrt(trend)) + 5
        if len(candles) < warmup:
            return []

        close  = candles["close"]
        high   = candles["high"]
        low    = candles["low"]
        volume = candles["volume"]

        hma_f  = hma(close, fast)
        hma_s  = hma(close, slow)
        hma_t  = hma(close, trend)
        avg_vol = volume.rolling(vp).mean()
        chan_long, chan_short = chandelier_exit(high, low, close, cp, cm)
        atr_vals = atr(high, low, close, cp)

        symbol = str(candles["symbol"].iloc[0])
        signals: list[Signal] = []
        in_long  = False
        in_short = False
        chan_long_level  = np.nan
        chan_short_level = np.nan

        # pre-convert to numpy — eliminates pandas .iloc overhead in hot loop
        is_clean_arr   = candles["is_clean"].values
        timestamps_arr = candles["timestamp"].dt.to_pydatetime()
        close_arr      = close.values
        hma_f_arr      = hma_f.values
        hma_s_arr      = hma_s.values
        hma_t_arr      = hma_t.values
        volume_arr     = volume.values
        avg_vol_arr    = avg_vol.values
        chan_long_arr  = chan_long.values
        chan_short_arr = chan_short.values
        atr_arr        = atr_vals.values

        for i in range(warmup, len(candles)):
            if not is_clean_arr[i]:
                continue

            c       = close_arr[i]
            hf      = hma_f_arr[i]
            hs      = hma_s_arr[i]
            ht      = hma_t_arr[i]
            hf_p    = hma_f_arr[i - 1]
            hs_p    = hma_s_arr[i - 1]
            vol     = volume_arr[i]
            avol    = avg_vol_arr[i]
            cl      = chan_long_arr[i]
            cs      = chan_short_arr[i]
            cur_atr = atr_arr[i]
            ts      = timestamps_arr[i]

            if any(np.isnan(x) for x in [hf, hs, ht, avol, cl, cs]):
                continue

            vol_ok = avol > 0 and vol > avol * vm

            # Update trailing chandelier levels (ratchet only)
            if in_long:
                chan_long_level = max(chan_long_level, cl) if not np.isnan(chan_long_level) else cl
            if in_short:
                chan_short_level = min(chan_short_level, cs) if not np.isnan(chan_short_level) else cs

            # Chandelier stop hit
            if in_long and c < chan_long_level:
                signals.append(Signal(
                    strategy=self.name, symbol=symbol,
                    timestamp=ts, direction="EXIT_LONG",
                    strength=1.0, close_price=float(c), atr=float(cur_atr),
                    reason=["chandelier_stop"],
                ))
                in_long = False
                chan_long_level = np.nan

            elif in_short and c > chan_short_level:
                signals.append(Signal(
                    strategy=self.name, symbol=symbol,
                    timestamp=ts, direction="EXIT_SHORT",
                    strength=1.0, close_price=float(c), atr=float(cur_atr),
                    reason=["chandelier_stop"],
                ))
                in_short = False
                chan_short_level = np.nan

            # Cross detection
            cross_up   = hf_p <= hs_p and hf > hs
            cross_down = hf_p >= hs_p and hf < hs

            # Long entry
            if cross_up and c > ht and vol_ok and not in_long:
                if in_short:
                    signals.append(Signal(
                        strategy=self.name, symbol=symbol,
                        timestamp=ts, direction="EXIT_SHORT",
                        strength=1.0, close_price=float(c), atr=float(cur_atr),
                        reason=["hma_cross_exit"],
                    ))
                    in_short = False
                signals.append(Signal(
                    strategy=self.name, symbol=symbol,
                    timestamp=ts, direction="LONG",
                    strength=min(1.0, abs(hf - hs) / (c * 0.005 + 1e-9)),
                    close_price=float(c), atr=float(cur_atr),
                    reason=["hma_cross_up", "above_trend_hma", "vol_confirm"],
                ))
                in_long = True
                chan_long_level = cl

            # Short entry
            elif cross_down and c < ht and vol_ok and not in_short:
                if in_long:
                    signals.append(Signal(
                        strategy=self.name, symbol=symbol,
                        timestamp=ts, direction="EXIT_LONG",
                        strength=1.0, close_price=float(c), atr=float(cur_atr),
                        reason=["hma_cross_exit"],
                    ))
                    in_long = False
                signals.append(Signal(
                    strategy=self.name, symbol=symbol,
                    timestamp=ts, direction="SHORT",
                    strength=min(1.0, abs(hf - hs) / (c * 0.005 + 1e-9)),
                    close_price=float(c), atr=float(cur_atr),
                    reason=["hma_cross_down", "below_trend_hma", "vol_confirm"],
                ))
                in_short = True
                chan_short_level = cs

        return signals
