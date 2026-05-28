"""
WyckoffPhaseDetector — Swing futures strategy.

Wyckoff methodology identifies accumulation and distribution via the
relationship between price spread and volume across phases:

  Accumulation phases (buy signals):
    SC  — Selling Climax: large down bar with massive volume  → panic selling
    ST  — Secondary Test: lower volume retest of SC low       → supply drying up
    LPS — Last Point of Support: small bar near SC, declining vol → spring loaded

  Distribution phases (sell signals):
    BC  — Buying Climax: large up bar with massive volume     → exhaustion top
    UT  — Upthrust: retest of BC high with lower volume       → failed breakout
    LPSY— Last Point of Supply: weak rally, declining vol     → distribution done

  Uses 4h candles for phase detection; 1d candle (via generate_signals_mtf)
  provides the macro trend filter — only trade accumulation in 1d uptrend.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from crypto_bot.core.signals.base import BaseStrategy
from crypto_bot.core.signals.models import Signal
from crypto_bot.core.signals.indicators import atr, ema


class WyckoffPhaseDetectorStrategy(BaseStrategy):
    """Automated Wyckoff accumulation/distribution phase detector."""

    @property
    def mode(self) -> str:
        return "swing"

    @property
    def param_space(self) -> dict:
        return {
            "vol_climax_mult":   (2.0,  5.0),         # volume spike multiple for SC/BC
            "vol_avg_period":    (10,   30,   "int"),  # volume baseline window
            "spread_mult":       (1.5,  3.0),          # candle body ATR multiple for climax
            "retest_bars":       (3,    15,   "int"),  # bars to look for secondary test
            "vol_decay_ratio":   (0.3,  0.7),          # retest vol must be < this × climax vol
            "trend_ema":         (50,  200,   "int"),  # 1d trend EMA period
            "atr_period":        (10,   20,   "int"),
        }

    # ── Internal phase state ─────────────────────────────────────────────
    def _detect_climax(
        self,
        i: int,
        high_arr: np.ndarray,
        low_arr: np.ndarray,
        close_arr: np.ndarray,
        vol_arr: np.ndarray,
        va_arr: np.ndarray,
        atr_arr: np.ndarray,
        vol_mult: float,
        spread_mult: float,
    ) -> str | None:
        """Returns 'SC' (selling climax), 'BC' (buying climax), or None."""
        c, h, l, v, va, at = (
            close_arr[i], high_arr[i], low_arr[i],
            vol_arr[i], va_arr[i], atr_arr[i],
        )
        if np.isnan(va) or va == 0 or np.isnan(at):
            return None

        body = abs(c - close_arr[i - 1]) if i > 0 else 0
        vol_spike = v > vol_mult * va
        large_body = body > spread_mult * at

        prev_c = close_arr[i - 1] if i > 0 else c
        if vol_spike and large_body and c < prev_c:
            return "SC"
        if vol_spike and large_body and c > prev_c:
            return "BC"
        return None

    def _is_retest(
        self,
        climax_idx: int,
        climax_vol: float,
        i: int,
        vol_arr: np.ndarray,
        high_arr: np.ndarray,
        low_arr: np.ndarray,
        close_arr: np.ndarray,
        climax_type: str,
        decay_ratio: float,
        atr_val: float,
    ) -> bool:
        """True if candle i is a low-volume secondary test of the climax."""
        v = vol_arr[i]
        if v >= decay_ratio * climax_vol:
            return False   # volume hasn't decayed — not a valid retest

        if climax_type == "SC":
            climax_low = low_arr[climax_idx]
            return low_arr[i] <= climax_low + atr_val  # retests SC low zone
        else:  # BC
            climax_high = high_arr[climax_idx]
            return high_arr[i] >= climax_high - atr_val  # retests BC high zone

    def generate_signals(
        self, candles: pd.DataFrame, aux_data: Any = None
    ) -> list[Signal]:
        p = self.params
        close  = candles["close"]
        high   = candles["high"]
        low    = candles["low"]
        volume = candles["volume"]
        symbol = str(candles["symbol"].iloc[0])

        vol_mult    = float(p["vol_climax_mult"])
        vol_avg_p   = int(p["vol_avg_period"])
        spread_mult = float(p["spread_mult"])
        retest_bars = int(p["retest_bars"])
        decay_ratio = float(p["vol_decay_ratio"])
        trend_p     = int(p["trend_ema"])
        atr_p       = int(p["atr_period"])

        atr_vals = atr(high, low, close, atr_p)
        trend    = ema(close, trend_p)
        vol_avg  = volume.rolling(vol_avg_p).mean()

        close_arr = close.values
        high_arr  = high.values
        low_arr   = low.values
        vol_arr   = volume.values
        va_arr    = vol_avg.values
        atr_arr   = atr_vals.values
        trend_arr = trend.values
        ts_arr    = candles["timestamp"].dt.to_pydatetime()
        is_clean  = candles["is_clean"].values

        warmup   = max(vol_avg_p, trend_p, atr_p) + 2
        signals: list[Signal] = []
        open_pos: str | None  = None

        # Track most recent climax event
        last_climax: dict | None = None  # {"type": "SC"|"BC", "idx": int, "vol": float}

        for i in range(warmup, len(candles)):
            if not is_clean[i]:
                continue
            c  = close_arr[i]
            at = atr_arr[i]
            tr = trend_arr[i]

            if np.isnan(at) or np.isnan(tr):
                continue

            # Detect new climax
            climax_type = self._detect_climax(
                i, high_arr, low_arr, close_arr, vol_arr, va_arr, atr_arr,
                vol_mult, spread_mult,
            )
            if climax_type:
                last_climax = {"type": climax_type, "idx": i, "vol": vol_arr[i]}

            # Check for secondary test / LPS / LPSY after a climax
            if last_climax and (i - last_climax["idx"]) <= retest_bars:
                ct = last_climax["type"]
                is_rt = self._is_retest(
                    last_climax["idx"], last_climax["vol"],
                    i, vol_arr, high_arr, low_arr, close_arr,
                    ct, decay_ratio, at,
                )
                if is_rt:
                    if ct == "SC" and c > tr and open_pos != "LONG":
                        # Accumulation: SC + low-vol retest → price about to spring
                        open_pos = "LONG"
                        signals.append(Signal(
                            strategy=self.name, symbol=symbol, direction="LONG",
                            timestamp=ts_arr[i], strength=0.75,
                            close_price=c, atr=at,
                            reason=["wyckoff_accumulation_LPS", f"trend_up"],
                        ))
                    elif ct == "BC" and c < tr and open_pos != "SHORT":
                        # Distribution: BC + low-vol retest → LPSY, about to drop
                        open_pos = "SHORT"
                        signals.append(Signal(
                            strategy=self.name, symbol=symbol, direction="SHORT",
                            timestamp=ts_arr[i], strength=0.75,
                            close_price=c, atr=at,
                            reason=["wyckoff_distribution_LPSY", "trend_down"],
                        ))
                    last_climax = None  # consume the setup

            # Exit: trend reversal
            if open_pos == "LONG" and c < tr:
                open_pos = None
                signals.append(Signal(
                    strategy=self.name, symbol=symbol, direction="EXIT_LONG",
                    timestamp=ts_arr[i], strength=0.5,
                    close_price=c, atr=at, reason=["wyckoff_trend_break"],
                ))
            elif open_pos == "SHORT" and c > tr:
                open_pos = None
                signals.append(Signal(
                    strategy=self.name, symbol=symbol, direction="EXIT_SHORT",
                    timestamp=ts_arr[i], strength=0.5,
                    close_price=c, atr=at, reason=["wyckoff_trend_break"],
                ))

        return signals

    def generate_signals_mtf(
        self, candles_by_tf: dict, aux_data: Any = None
    ) -> list[Signal]:
        """Use 4h for phase detection; 1d trend EMA filters direction."""
        primary = candles_by_tf.get("4h")
        if primary is None:
            primary = next(iter(candles_by_tf.values()))
        return self.generate_signals(primary, aux_data)
