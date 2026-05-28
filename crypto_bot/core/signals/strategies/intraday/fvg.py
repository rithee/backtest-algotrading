"""
FairValueGap (FVG) — Intraday futures strategy.

Logic:
  An FVG is a 3-candle imbalance where price moves so aggressively that
  a gap forms between candle[i-2] and candle[i]:

    Bullish FVG: candle[i-2].high < candle[i].low
      → price left an unfilled gap below; when price pulls back into it,
        expect continuation upward.

    Bearish FVG: candle[i-2].low > candle[i].high
      → price left an unfilled gap above; when price fills it,
        expect continuation downward.

  Gap must be at least `min_gap_atr` × ATR wide (filters noise).
  RSI + trend EMA provide momentum confirmation before entry.
  Exit when RSI reaches overbought/oversold extreme.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from crypto_bot.core.signals.base import BaseStrategy
from crypto_bot.core.signals.models import Signal
from crypto_bot.core.signals.indicators import atr, ema, rsi


class FairValueGapStrategy(BaseStrategy):
    """3-candle imbalance fill with momentum confirmation."""

    @property
    def mode(self) -> str:
        return "intraday"

    @property
    def param_space(self) -> dict:
        return {
            "min_gap_atr":  (0.10, 0.50),         # gap size in ATR multiples
            "rsi_period":   (10,   20,   "int"),
            "rsi_momentum": (45.0, 58.0),          # RSI must be above this for LONG fills
            "trend_ema":    (20,   60,   "int"),
            "atr_period":   (10,   20,   "int"),
            "max_gaps":     (5,    30,   "int"),   # how many unfilled gaps to track
        }

    def generate_signals(
        self, candles: pd.DataFrame, aux_data: Any = None
    ) -> list[Signal]:
        p = self.params
        close  = candles["close"]
        high   = candles["high"]
        low    = candles["low"]
        symbol = str(candles["symbol"].iloc[0])

        min_gap = float(p["min_gap_atr"])
        rsi_p   = int(p["rsi_period"])
        rsi_mom = float(p["rsi_momentum"])
        trend_p = int(p["trend_ema"])
        atr_p   = int(p["atr_period"])
        max_g   = int(p["max_gaps"])

        atr_vals = atr(high, low, close, atr_p)
        rsi_vals = rsi(close, rsi_p)
        trend    = ema(close, trend_p)

        close_arr = close.values
        high_arr  = high.values
        low_arr   = low.values
        rsi_arr   = rsi_vals.values
        trend_arr = trend.values
        atr_arr   = atr_vals.values
        ts_arr    = candles["timestamp"].dt.to_pydatetime()
        is_clean  = candles["is_clean"].values

        warmup   = max(rsi_p, trend_p, atr_p) + 3
        signals: list[Signal] = []
        open_pos: str | None  = None

        # Active FVGs: list of {"low": float, "high": float, "dir": "LONG"|"SHORT"}
        active_fvgs: list[dict] = []

        for i in range(warmup, len(candles)):
            if not is_clean[i]:
                continue
            c  = close_arr[i]
            h  = high_arr[i]
            l  = low_arr[i]
            at = atr_arr[i]
            rs = rsi_arr[i]
            tr = trend_arr[i]

            if np.isnan(at) or np.isnan(rs) or np.isnan(tr):
                continue

            h2 = high_arr[i - 2]
            l2 = low_arr[i - 2]

            # Detect new FVGs on this candle
            if l > h2 and (l - h2) >= min_gap * at:
                # Bullish FVG: gap between C[i-2].high and C[i].low
                active_fvgs.append({"low": h2, "high": l, "dir": "LONG"})

            if h < l2 and (l2 - h) >= min_gap * at:
                # Bearish FVG: gap between C[i].high and C[i-2].low
                active_fvgs.append({"low": h, "high": l2, "dir": "SHORT"})

            # Limit gap tracking depth
            if len(active_fvgs) > max_g:
                active_fvgs = active_fvgs[-max_g:]

            # Check whether current price fills any active FVG
            remaining: list[dict] = []
            for fvg in active_fvgs:
                filled = fvg["low"] <= c <= fvg["high"]
                if filled:
                    if (
                        fvg["dir"] == "LONG"
                        and rs > rsi_mom
                        and c > tr
                        and open_pos != "LONG"
                    ):
                        open_pos = "LONG"
                        signals.append(Signal(
                            strategy=self.name, symbol=symbol, direction="LONG",
                            timestamp=ts_arr[i], strength=0.7,
                            close_price=c, atr=at,
                            reason=["FVG_bullish_fill", f"rsi={rs:.0f}"],
                        ))
                    elif (
                        fvg["dir"] == "SHORT"
                        and rs < (100 - rsi_mom)
                        and c < tr
                        and open_pos != "SHORT"
                    ):
                        open_pos = "SHORT"
                        signals.append(Signal(
                            strategy=self.name, symbol=symbol, direction="SHORT",
                            timestamp=ts_arr[i], strength=0.7,
                            close_price=c, atr=at,
                            reason=["FVG_bearish_fill", f"rsi={rs:.0f}"],
                        ))
                    # gap filled — drop it
                else:
                    remaining.append(fvg)
            active_fvgs = remaining

            # Exit on RSI extreme
            if open_pos == "LONG" and rs > 78:
                open_pos = None
                signals.append(Signal(
                    strategy=self.name, symbol=symbol, direction="EXIT_LONG",
                    timestamp=ts_arr[i], strength=0.5,
                    close_price=c, atr=at, reason=["FVG_rsi_overbought"],
                ))
            elif open_pos == "SHORT" and rs < 22:
                open_pos = None
                signals.append(Signal(
                    strategy=self.name, symbol=symbol, direction="EXIT_SHORT",
                    timestamp=ts_arr[i], strength=0.5,
                    close_price=c, atr=at, reason=["FVG_rsi_oversold"],
                ))

        return signals

    def generate_signals_mtf(self, candles_by_tf: dict, aux_data=None) -> list[Signal]:
        primary = candles_by_tf.get("5m")
        if primary is None:
            primary = next(iter(candles_by_tf.values()))
        return self.generate_signals(primary, aux_data)
