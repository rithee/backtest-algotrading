"""
LiquiditySweepReversal — Intraday futures strategy.

Logic:
  - Price sweeps above a recent swing high (stop hunt on longs) with a volume spike
    and leaves a large upper wick → price likely to reverse lower (bearish).
  - Price sweeps below a recent swing low (stop hunt on shorts) with a volume spike
    and leaves a large lower wick → price likely to reverse higher (bullish).
  - Entry: candle close confirms reversal direction.
  - Exit: price breaks back through the swept level.

Degrades gracefully without aux_data (uses only OHLCV).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_bot.core.signals.base import BaseStrategy
from crypto_bot.core.signals.models import Signal
from crypto_bot.core.signals.indicators import atr


class LiquiditySweepReversalStrategy(BaseStrategy):
    """Stop-hunt reversal on volume spike + wick confirmation."""

    @property
    def mode(self) -> str:
        return "intraday"

    @property
    def param_space(self) -> dict:
        return {
            "swing_lookback": (5,  20,   "int"),  # bars for swing high/low detection
            "vol_spike_mult": (1.5, 3.5),          # volume vs avg to qualify as spike
            "vol_avg_period": (10,  30,  "int"),   # averaging window for volume baseline
            "wick_ratio":     (0.45, 0.80),        # min wick / candle-range to confirm
            "atr_period":     (10,  20,  "int"),
        }

    def generate_signals(
        self, candles: pd.DataFrame, aux_data=None
    ) -> list[Signal]:
        p = self.params
        close  = candles["close"]
        high   = candles["high"]
        low    = candles["low"]
        volume = candles["volume"]
        symbol = str(candles["symbol"].iloc[0])

        swing_lb  = int(p["swing_lookback"])
        vol_mult  = float(p["vol_spike_mult"])
        vol_avg_p = int(p["vol_avg_period"])
        wick_r    = float(p["wick_ratio"])
        atr_p     = int(p["atr_period"])

        atr_vals   = atr(high, low, close, atr_p)
        swing_high = high.shift(1).rolling(swing_lb).max()
        swing_low  = low.shift(1).rolling(swing_lb).min()
        vol_avg    = volume.rolling(vol_avg_p).mean()

        # ── numpy hot path ────────────────────────────────────────────────
        close_arr = close.values
        high_arr  = high.values
        low_arr   = low.values
        vol_arr   = volume.values
        va_arr    = vol_avg.values
        sh_arr    = swing_high.values
        sl_arr    = swing_low.values
        atr_arr   = atr_vals.values
        ts_arr    = candles["timestamp"].dt.to_pydatetime()
        is_clean  = candles["is_clean"].values

        warmup  = swing_lb + vol_avg_p + atr_p
        signals: list[Signal] = []
        open_pos: str | None  = None

        for i in range(warmup, len(candles)):
            if not is_clean[i]:
                continue
            c  = close_arr[i]
            h  = high_arr[i]
            l  = low_arr[i]
            v  = vol_arr[i]
            va = va_arr[i]
            sh = sh_arr[i]
            sl = sl_arr[i]
            at = atr_arr[i]

            if np.isnan(sh) or np.isnan(sl) or np.isnan(at) or np.isnan(va) or va == 0:
                continue

            rng = h - l
            if rng <= 0:
                continue

            upper_wick = h - max(c, close_arr[i - 1])
            lower_wick = min(c, close_arr[i - 1]) - l
            vol_spike  = v > vol_mult * va

            # Bearish sweep: wick above swing high, close back below it
            bearish = (
                h > sh
                and c < sh
                and vol_spike
                and upper_wick / rng >= wick_r
            )
            # Bullish sweep: wick below swing low, close back above it
            bullish = (
                l < sl
                and c > sl
                and vol_spike
                and lower_wick / rng >= wick_r
            )

            if bearish and open_pos != "SHORT":
                open_pos = "SHORT"
                signals.append(Signal(
                    strategy=self.name, symbol=symbol, direction="SHORT",
                    timestamp=ts_arr[i],
                    strength=min(upper_wick / rng, 1.0),
                    close_price=c, atr=at,
                    reason=["lsr_sweep_bearish", f"vol_x{v/va:.1f}"],
                ))
            elif bullish and open_pos != "LONG":
                open_pos = "LONG"
                signals.append(Signal(
                    strategy=self.name, symbol=symbol, direction="LONG",
                    timestamp=ts_arr[i],
                    strength=min(lower_wick / rng, 1.0),
                    close_price=c, atr=at,
                    reason=["lsr_sweep_bullish", f"vol_x{v/va:.1f}"],
                ))
            elif open_pos == "LONG" and c < sl:
                open_pos = None
                signals.append(Signal(
                    strategy=self.name, symbol=symbol, direction="EXIT_LONG",
                    timestamp=ts_arr[i], strength=0.5,
                    close_price=c, atr=at, reason=["lsr_stop"],
                ))
            elif open_pos == "SHORT" and c > sh:
                open_pos = None
                signals.append(Signal(
                    strategy=self.name, symbol=symbol, direction="EXIT_SHORT",
                    timestamp=ts_arr[i], strength=0.5,
                    close_price=c, atr=at, reason=["lsr_stop"],
                ))

        return signals

    def generate_signals_mtf(self, candles_by_tf: dict, aux_data=None) -> list[Signal]:
        primary = candles_by_tf.get("5m")
        if primary is None:
            primary = next(iter(candles_by_tf.values()))
        return self.generate_signals(primary, aux_data)
