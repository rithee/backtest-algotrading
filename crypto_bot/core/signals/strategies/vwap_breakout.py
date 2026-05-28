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

        signals: list[Signal] = []
        in_long  = False
        in_short = False

        for i in range(warmup, len(candles)):
            row  = candles.iloc[i]
            if not row.get("is_clean", True):
                continue

            c    = float(close.iloc[i])
            pc   = float(close.iloc[i - 1])
            vw   = float(vwap_.iloc[i])
            up   = float(upper.iloc[i])
            lo   = float(lower.iloc[i])
            pvw  = float(vwap_.iloc[i - 1])
            vol  = float(volume.iloc[i])
            avol = float(avg_vol.iloc[i])
            cur_atr = float(atr_v.iloc[i])

            if any(np.isnan(x) for x in [vw, up, lo, avol]):
                continue

            vol_ok = avol > 0 and vol > avol * vol_mult

            # Exit: price crosses back through VWAP midline
            if in_long and c < vw and pc >= pvw:
                signals.append(Signal(
                    strategy=self.name, symbol=row["symbol"],
                    timestamp=row["timestamp"], direction="EXIT_LONG",
                    strength=1.0, close_price=c, atr=cur_atr,
                    reason=["price_below_vwap"],
                ))
                in_long = False

            elif in_short and c > vw and pc <= pvw:
                signals.append(Signal(
                    strategy=self.name, symbol=row["symbol"],
                    timestamp=row["timestamp"], direction="EXIT_SHORT",
                    strength=1.0, close_price=c, atr=cur_atr,
                    reason=["price_above_vwap"],
                ))
                in_short = False

            # Entry: breakout above upper band
            if not in_long and c > up and pc <= float(upper.iloc[i - 1]) and vol_ok:
                if in_short:
                    signals.append(Signal(
                        strategy=self.name, symbol=row["symbol"],
                        timestamp=row["timestamp"], direction="EXIT_SHORT",
                        strength=1.0, close_price=c, atr=cur_atr,
                        reason=["vwap_band_flip"],
                    ))
                    in_short = False
                signals.append(Signal(
                    strategy=self.name, symbol=row["symbol"],
                    timestamp=row["timestamp"], direction="LONG",
                    strength=min(1.0, (c - up) / (vwap_std.iloc[i] + 1e-9)),
                    close_price=c, atr=cur_atr,
                    reason=["vwap_upper_break", "vol_confirm"],
                ))
                in_long = True

            # Entry: breakout below lower band
            elif not in_short and c < lo and pc >= float(lower.iloc[i - 1]) and vol_ok:
                if in_long:
                    signals.append(Signal(
                        strategy=self.name, symbol=row["symbol"],
                        timestamp=row["timestamp"], direction="EXIT_LONG",
                        strength=1.0, close_price=c, atr=cur_atr,
                        reason=["vwap_band_flip"],
                    ))
                    in_long = False
                signals.append(Signal(
                    strategy=self.name, symbol=row["symbol"],
                    timestamp=row["timestamp"], direction="SHORT",
                    strength=min(1.0, (lo - c) / (vwap_std.iloc[i] + 1e-9)),
                    close_price=c, atr=cur_atr,
                    reason=["vwap_lower_break", "vol_confirm"],
                ))
                in_short = True

        return signals
