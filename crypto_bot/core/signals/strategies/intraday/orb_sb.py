"""
OpeningRangeBreakout + SessionBias — Intraday futures strategy.

Logic:
  Four crypto trading sessions (UTC):
    - Asia:     00–08
    - London:   08–12
    - New York: 13–17
    - NY Close: 17–22

  For each session: first `or_bars` candles form the Opening Range (OR).
  Once the OR is formed, trade a confirmed close outside the range:
    - Close > OR high + volume > avg + uptrend → LONG
    - Close < OR low  + volume > avg + downtrend → SHORT
  Exit when price closes back inside the range.

  Handles 1m, 5m, 15m candle sizes (detects session boundaries from timestamp).
"""
from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from crypto_bot.core.signals.base import BaseStrategy
from crypto_bot.core.signals.models import Signal
from crypto_bot.core.signals.indicators import atr, ema

# Session windows (UTC): name → (start_hour_inclusive, end_hour_exclusive)
_SESSIONS: dict[str, tuple[int, int]] = {
    "asia":     (0,  8),
    "london":   (8,  12),
    "ny":       (13, 17),
    "ny_close": (17, 22),
}


def _session_name(hour: int) -> str | None:
    for name, (start, end) in _SESSIONS.items():
        if start <= hour < end:
            return name
    return None


class OpeningRangeBreakoutStrategy(BaseStrategy):
    """Session Opening Range Breakout with volume + trend confirmation."""

    @property
    def mode(self) -> str:
        return "intraday"

    @property
    def param_space(self) -> dict:
        return {
            "or_bars":    (4,   12,  "int"),  # bars in opening range (4×5m = 20min)
            "vol_mult":   (1.2,  2.5),        # volume spike multiplier
            "vol_period": (10,  30,  "int"),  # volume baseline window
            "trend_ema":  (20,  60,  "int"),  # short-term trend EMA period
            "atr_period": (10,  20,  "int"),
        }

    def generate_signals(
        self, candles: pd.DataFrame, aux_data: Any = None
    ) -> list[Signal]:
        p = self.params
        close  = candles["close"]
        high   = candles["high"]
        low    = candles["low"]
        volume = candles["volume"]
        symbol = str(candles["symbol"].iloc[0])

        or_bars    = int(p["or_bars"])
        vol_mult   = float(p["vol_mult"])
        vol_period = int(p["vol_period"])
        trend_p    = int(p["trend_ema"])
        atr_p      = int(p["atr_period"])

        atr_vals  = atr(high, low, close, atr_p)
        trend_ema = ema(close, trend_p)
        vol_avg   = volume.rolling(vol_period).mean()

        # ── numpy arrays ─────────────────────────────────────────────────
        close_arr = close.values
        high_arr  = high.values
        low_arr   = low.values
        vol_arr   = volume.values
        va_arr    = vol_avg.values
        trend_arr = trend_ema.values
        atr_arr   = atr_vals.values
        ts_arr    = candles["timestamp"].dt.to_pydatetime()
        is_clean  = candles["is_clean"].values

        warmup   = max(vol_period, trend_p, atr_p) + 1
        signals: list[Signal] = []
        open_pos: str | None  = None

        # Per-(date, session) opening-range state
        or_state: dict[tuple[date, str], dict] = {}

        for i in range(warmup, len(candles)):
            if not is_clean[i]:
                continue
            ts      = ts_arr[i]
            session = _session_name(ts.hour)
            if session is None:
                continue

            key = (ts.date(), session)
            if key not in or_state:
                or_state[key] = {
                    "high":   float("-inf"),
                    "low":    float("inf"),
                    "bars":   0,
                    "formed": False,
                }

            state = or_state[key]
            if not state["formed"]:
                # Building the opening range
                state["high"] = max(state["high"], high_arr[i])
                state["low"]  = min(state["low"],  low_arr[i])
                state["bars"] += 1
                if state["bars"] >= or_bars:
                    state["formed"] = True
                continue

            c  = close_arr[i]
            v  = vol_arr[i]
            va = va_arr[i]
            at = atr_arr[i]
            te = trend_arr[i]
            or_h = state["high"]
            or_l = state["low"]

            if np.isnan(at) or np.isnan(te) or np.isnan(va) or va == 0:
                continue

            vol_ok = v > vol_mult * va

            if c > or_h and vol_ok and c > te and open_pos != "LONG":
                open_pos = "LONG"
                signals.append(Signal(
                    strategy=self.name, symbol=symbol, direction="LONG",
                    timestamp=ts,
                    strength=min((c - or_h) / max(at, 1e-8), 1.0),
                    close_price=c, atr=at,
                    reason=[f"ORB_{session}_long", f"range_high={or_h:.0f}"],
                ))
            elif c < or_l and vol_ok and c < te and open_pos != "SHORT":
                open_pos = "SHORT"
                signals.append(Signal(
                    strategy=self.name, symbol=symbol, direction="SHORT",
                    timestamp=ts,
                    strength=min((or_l - c) / max(at, 1e-8), 1.0),
                    close_price=c, atr=at,
                    reason=[f"ORB_{session}_short", f"range_low={or_l:.0f}"],
                ))
            elif open_pos == "LONG" and c < or_l:
                open_pos = None
                signals.append(Signal(
                    strategy=self.name, symbol=symbol, direction="EXIT_LONG",
                    timestamp=ts, strength=0.5,
                    close_price=c, atr=at, reason=["ORB_range_revert"],
                ))
            elif open_pos == "SHORT" and c > or_h:
                open_pos = None
                signals.append(Signal(
                    strategy=self.name, symbol=symbol, direction="EXIT_SHORT",
                    timestamp=ts, strength=0.5,
                    close_price=c, atr=at, reason=["ORB_range_revert"],
                ))

        return signals

    def generate_signals_mtf(self, candles_by_tf: dict, aux_data=None) -> list[Signal]:
        # Prefer 5m for precise opening range; fall back to 1m, then first available
        primary = candles_by_tf.get("5m")
        if primary is None:
            primary = candles_by_tf.get("1m")
        if primary is None:
            primary = next(iter(candles_by_tf.values()))
        return self.generate_signals(primary, aux_data)
