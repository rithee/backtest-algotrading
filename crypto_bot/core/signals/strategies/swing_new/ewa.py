"""
ElliottWaveAutomator — Swing futures strategy.

Logic:
  Algorithmic Elliott Wave counting via ZigZag pivot detection.

  ZigZag: identifies significant swing highs/lows by finding price
  reversals greater than `zz_pct` percent.

  Wave counting heuristic:
    - After 2 confirmed ZigZag legs, the 3rd leg (Wave 3) is typically
      the most powerful. Enter in the direction of Wave 3.
    - Wave 3 target: Wave 1 length × 1.618 (Fibonacci extension).
    - Wave 2 retracement must be 50–78.6% of Wave 1 (Fib validity check).
    - Stop: below Wave 2 low (long) / above Wave 2 high (short).

  Fibonacci extension levels: 1.618×, 2.618×
  Exit: price reaches 1.618 extension or trend breaks.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from crypto_bot.core.signals.base import BaseStrategy
from crypto_bot.core.signals.models import Signal
from crypto_bot.core.signals.indicators import atr, ema


def _find_zigzag_pivots(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    deviation_pct: float,
) -> list[dict]:
    """
    ZigZag pivot detector. Returns a strictly alternating list of pivots:
      {"idx": int, "price": float, "type": "H"|"L"}

    Algorithm:
      - Tracks the running extreme in the current trend direction.
      - A reversal is confirmed when close crosses `deviation_pct`% past the extreme.
      - Guarantees H and L strictly alternate.
    """
    pivots: list[dict] = []
    if len(close) < 3:
        return pivots

    # Start by deciding initial direction from first two bars
    direction  = 1 if close[1] >= close[0] else -1   # 1=looking for high, -1=looking for low
    ext_price  = high[0] if direction == 1 else low[0]
    ext_idx    = 0

    for i in range(1, len(close)):
        if direction == 1:
            # Tracking upward extreme
            if high[i] > ext_price:
                ext_price = high[i]
                ext_idx   = i
            # Reversal: close drops deviation_pct below the running high
            if close[i] <= ext_price * (1.0 - deviation_pct):
                pivots.append({"idx": ext_idx, "price": ext_price, "type": "H"})
                direction = -1
                ext_price = low[i]
                ext_idx   = i
        else:
            # Tracking downward extreme
            if low[i] < ext_price:
                ext_price = low[i]
                ext_idx   = i
            # Reversal: close rises deviation_pct above the running low
            if close[i] >= ext_price * (1.0 + deviation_pct):
                pivots.append({"idx": ext_idx, "price": ext_price, "type": "L"})
                direction = 1
                ext_price = high[i]
                ext_idx   = i

    return pivots


class ElliottWaveAutomatorStrategy(BaseStrategy):
    """ZigZag-based Wave 3 entry with Fibonacci extension targets."""

    @property
    def mode(self) -> str:
        return "swing"

    @property
    def param_space(self) -> dict:
        return {
            "zz_pct":         (0.03, 0.12),         # ZigZag deviation (3–12%)
            "fib_retr_min":   (0.40, 0.60),          # min Wave 2 retracement of Wave 1
            "fib_retr_max":   (0.70, 0.90),          # max Wave 2 retracement (else invalid)
            "fib_ext":        (1.4,  1.8),            # Wave 3 target extension multiple
            "trend_ema":      (50,  200, "int"),
            "atr_period":     (10,  20,  "int"),
        }

    def generate_signals(
        self, candles: pd.DataFrame, aux_data: Any = None
    ) -> list[Signal]:
        p = self.params
        close  = candles["close"]
        high   = candles["high"]
        low    = candles["low"]
        symbol = str(candles["symbol"].iloc[0])

        zz_pct    = float(p["zz_pct"])
        fib_min   = float(p["fib_retr_min"])
        fib_max   = float(p["fib_retr_max"])
        fib_ext   = float(p["fib_ext"])
        trend_p   = int(p["trend_ema"])
        atr_p     = int(p["atr_period"])

        atr_vals = atr(high, low, close, atr_p)
        trend    = ema(close, trend_p)

        high_arr  = high.values
        low_arr   = low.values
        close_arr = close.values
        atr_arr   = atr_vals.values
        trend_arr = trend.values
        ts_arr    = candles["timestamp"].dt.to_pydatetime()
        is_clean  = candles["is_clean"].values

        warmup   = max(trend_p, atr_p) + 5
        signals: list[Signal] = []
        open_pos: str | None  = None

        # We recompute ZigZag on expanding window every N bars (performance)
        # Re-detect every 10 bars
        pivots: list[dict] = []
        last_pivot_count = 0
        recompute_every  = 10
        w3_target: float | None = None

        for i in range(warmup, len(candles)):
            if not is_clean[i]:
                continue
            c  = close_arr[i]
            at = atr_arr[i]
            tr = trend_arr[i]

            if np.isnan(at) or np.isnan(tr):
                continue

            # Recompute pivots periodically
            if i % recompute_every == 0 or len(pivots) != last_pivot_count:
                pivots = _find_zigzag_pivots(
                    high_arr[:i+1], low_arr[:i+1], close_arr[:i+1], zz_pct
                )
                last_pivot_count = len(pivots)

            # Need at least 3 pivots to count 2 waves
            if len(pivots) < 3:
                continue

            # Most recent 3 pivots: W0 → W1 → W2
            w0 = pivots[-3]
            w1 = pivots[-2]
            w2 = pivots[-1]

            wave1_len = abs(w1["price"] - w0["price"])
            if wave1_len <= 0:
                continue

            wave2_retr = abs(w2["price"] - w1["price"]) / wave1_len

            # Fibonacci retracement validity
            if not (fib_min <= wave2_retr <= fib_max):
                continue

            # Bullish Wave 3: W0=L, W1=H, W2=L (two lows, one high between)
            if w0["type"] == "L" and w1["type"] == "H" and w2["type"] == "L":
                # Current candle is near Wave 2 low and should break above W1 high
                w3_entry_zone = w2["price"] + at * 0.5
                if c >= w3_entry_zone and c > tr and open_pos != "LONG":
                    w3_target = w1["price"] + wave1_len * fib_ext
                    open_pos  = "LONG"
                    signals.append(Signal(
                        strategy=self.name, symbol=symbol, direction="LONG",
                        timestamp=ts_arr[i], strength=0.8,
                        close_price=c, atr=at,
                        reason=["ewa_wave3_bull", f"W2={w2['price']:.0f}", f"target={w3_target:.0f}"],
                    ))

            # Bearish Wave 3: W0=H, W1=L, W2=H (two highs, one low between)
            elif w0["type"] == "H" and w1["type"] == "L" and w2["type"] == "H":
                w3_entry_zone = w2["price"] - at * 0.5
                if c <= w3_entry_zone and c < tr and open_pos != "SHORT":
                    w3_target = w1["price"] - wave1_len * fib_ext
                    open_pos  = "SHORT"
                    signals.append(Signal(
                        strategy=self.name, symbol=symbol, direction="SHORT",
                        timestamp=ts_arr[i], strength=0.8,
                        close_price=c, atr=at,
                        reason=["ewa_wave3_bear", f"W2={w2['price']:.0f}", f"target={w3_target:.0f}"],
                    ))

            # Exit: reached Fibonacci extension target or trend break
            if open_pos == "LONG" and w3_target is not None:
                if c >= w3_target or c < tr:
                    open_pos  = None
                    w3_target = None
                    signals.append(Signal(
                        strategy=self.name, symbol=symbol, direction="EXIT_LONG",
                        timestamp=ts_arr[i], strength=0.6,
                        close_price=c, atr=at, reason=["ewa_fib_target_or_trend"],
                    ))
            elif open_pos == "SHORT" and w3_target is not None:
                if c <= w3_target or c > tr:
                    open_pos  = None
                    w3_target = None
                    signals.append(Signal(
                        strategy=self.name, symbol=symbol, direction="EXIT_SHORT",
                        timestamp=ts_arr[i], strength=0.6,
                        close_price=c, atr=at, reason=["ewa_fib_target_or_trend"],
                    ))

        return signals

    def generate_signals_mtf(
        self, candles_by_tf: dict, aux_data: Any = None
    ) -> list[Signal]:
        """Use 4h for wave detection; 1d trend filter via trend_ema."""
        primary = candles_by_tf.get("4h")
        if primary is None:
            primary = next(iter(candles_by_tf.values()))
        return self.generate_signals(primary, aux_data)
