"""
Rolling VWAP Bands Breakout Strategy.

VWAP (Volume-Weighted Average Price) is the benchmark institutional traders
use for execution quality. Large participants measure performance vs VWAP,
creating self-fulfilling price reactions at VWAP levels.

Rolling VWAP over N bars captures the volume-weighted mean price for a recent
window, while the band (± k × volume-weighted std) defines a "fair value range."

Breakout above the upper band = price leaving institutional fair value upward
with volume conviction → momentum trade.
Breakout below lower band = downside departure → short.
Return to VWAP midline = trade over.

This strategy is genuinely uncorrelated with price-indicator strategies because
it weights by volume, not just price, and the fair-value reference shifts
continuously with where money has actually transacted.

Entry:
  LONG  — close > VWAP + k × std AND volume > avg volume × vol_mult
  SHORT — close < VWAP - k × std AND volume > avg volume × vol_mult

Exit:
  Close crosses back through VWAP midline
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from crypto_bot.core.signals.base import BaseStrategy
from crypto_bot.core.signals.models import Signal
from crypto_bot.core.signals.indicators import rolling_vwap, atr


class VWAPBreakoutStrategy(BaseStrategy):
    """Rolling VWAP bands breakout with volume confirmation."""

    @property
    def param_space(self) -> dict[str, tuple]:
        return {
            "vwap_period":  (10, 40, "int"),   # rolling window for VWAP
            "band_mult":    (1.0, 3.0),         # std deviations for bands
            "vol_period":   (10, 30, "int"),   # volume average period
            "vol_mult":     (1.0, 2.5),         # volume confirmation threshold
            "atr_period":   (10, 21, "int"),
        }

    def generate_signals(self, candles: pd.DataFrame, aux_data: dict | None = None) -> list[Signal]:
        p = self.params
        vwap_period = int(p.get("vwap_period",  20))
        band_mult   = float(p.get("band_mult",  2.0))
        vol_period  = int(p.get("vol_period",   20))
        vol_mult    = float(p.get("vol_mult",   1.5))
        atr_period  = int(p.get("atr_period",  14))

        warmup = max(vwap_period, vol_period, atr_period) + 2
        if len(candles) < warmup:
            return []

        close  = candles["close"]
        high   = candles["high"]
        low    = candles["low"]
        volume = candles["volume"]

        vwap_, vwap_std = rolling_vwap(high, low, close, volume, vwap_period)
        upper = vwap_ + band_mult * vwap_std
        lower = vwap_ - band_mult * vwap_std
        avg_vol = volume.rolling(vol_period).mean()
        atr_v   = atr(high, low, close, atr_period)

        symbol = str(candles["symbol"].iloc[0])
        signals: list[Signal] = []
        in_long  = False
        in_short = False

        # pre-convert to numpy — eliminates pandas .iloc overhead in hot loop
        is_clean_arr   = candles["is_clean"].values
        timestamps_arr = candles["timestamp"].dt.to_pydatetime()
        close_arr      = close.values
        vwap_arr       = vwap_.values
        upper_arr      = upper.values
        lower_arr      = lower.values
        volume_arr     = volume.values
        avg_vol_arr    = avg_vol.values
        atr_arr        = atr_v.values
        vwap_std_arr   = vwap_std.values

        for i in range(warmup, len(candles)):
            if not is_clean_arr[i]:
                continue

            c       = close_arr[i]
            pc      = close_arr[i - 1]
            vw      = vwap_arr[i]
            up      = upper_arr[i]
            lo      = lower_arr[i]
            pvw     = vwap_arr[i - 1]
            vol     = volume_arr[i]
            avol    = avg_vol_arr[i]
            cur_atr = atr_arr[i]
            ts      = timestamps_arr[i]

            if any(np.isnan(x) for x in [vw, up, lo, avol]):
                continue

            vol_ok = avol > 0 and vol > avol * vol_mult

            # Exit: price crosses back through VWAP midline
            if in_long and c < vw and pc >= pvw:
                signals.append(Signal(
                    strategy=self.name, symbol=symbol,
                    timestamp=ts, direction="EXIT_LONG",
                    strength=1.0, close_price=float(c), atr=float(cur_atr),
                    reason=["price_below_vwap"],
                ))
                in_long = False

            elif in_short and c > vw and pc <= pvw:
                signals.append(Signal(
                    strategy=self.name, symbol=symbol,
                    timestamp=ts, direction="EXIT_SHORT",
                    strength=1.0, close_price=float(c), atr=float(cur_atr),
                    reason=["price_above_vwap"],
                ))
                in_short = False

            # Entry: breakout above upper band
            if not in_long and c > up and pc <= upper_arr[i - 1] and vol_ok:
                if in_short:
                    signals.append(Signal(
                        strategy=self.name, symbol=symbol,
                        timestamp=ts, direction="EXIT_SHORT",
                        strength=1.0, close_price=float(c), atr=float(cur_atr),
                        reason=["vwap_band_flip"],
                    ))
                    in_short = False
                signals.append(Signal(
                    strategy=self.name, symbol=symbol,
                    timestamp=ts, direction="LONG",
                    strength=min(1.0, (c - up) / (vwap_std_arr[i] + 1e-9)),
                    close_price=float(c), atr=float(cur_atr),
                    reason=["vwap_upper_break", "vol_confirm"],
                ))
                in_long = True

            # Entry: breakout below lower band
            elif not in_short and c < lo and pc >= lower_arr[i - 1] and vol_ok:
                if in_long:
                    signals.append(Signal(
                        strategy=self.name, symbol=symbol,
                        timestamp=ts, direction="EXIT_LONG",
                        strength=1.0, close_price=float(c), atr=float(cur_atr),
                        reason=["vwap_band_flip"],
                    ))
                    in_long = False
                signals.append(Signal(
                    strategy=self.name, symbol=symbol,
                    timestamp=ts, direction="SHORT",
                    strength=min(1.0, (lo - c) / (vwap_std_arr[i] + 1e-9)),
                    close_price=float(c), atr=float(cur_atr),
                    reason=["vwap_lower_break", "vol_confirm"],
                ))
                in_short = True

        return signals
