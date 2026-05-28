"""
MicrostructureConsolidationBreakout — Intraday futures strategy.

Logic:
  Markets "coil" before explosive moves — ATR contracts, OI builds, then
  price breaks out sharply as accumulated positions get triggered.

  Consolidation: current ATR ≤ rolling ATR percentile threshold (market is quiet).
  Breakout: price closes above/below the consolidation range with a volume spike.
  OI confirmation (optional): OI increasing during consolidation indicates new
    positions being built, not just short covering.

  Entry: first candle after a consolidation period breaks out.
  Exit: breakout fails — price pulls back more than `breakout_mult` × ATR.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from crypto_bot.core.signals.base import BaseStrategy
from crypto_bot.core.signals.models import Signal
from crypto_bot.core.signals.indicators import atr


class MicrostructureConsolidationBreakoutStrategy(BaseStrategy):
    """ATR-consolidation breakout with volume + optional OI confirmation."""

    @property
    def mode(self) -> str:
        return "intraday"

    @property
    def param_space(self) -> dict:
        return {
            "consol_period": (10, 30,  "int"),  # bars to measure consolidation
            "atr_percentile":(0.2, 0.5),         # ATR quantile below which = coiling
            "breakout_mult": (0.5, 2.0),         # ATR × this = breakout confirmation size
            "vol_mult":      (1.2, 2.5),         # volume confirmation
            "atr_period":    (10, 20,  "int"),
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

        consol_p  = int(p["consol_period"])
        atr_pct   = float(p["atr_percentile"])
        bk_mult   = float(p["breakout_mult"])
        vol_mult  = float(p["vol_mult"])
        atr_p     = int(p["atr_period"])

        atr_vals   = atr(high, low, close, atr_p)
        # Rolling quantile of ATR (twice the consol window for stability)
        atr_thresh = atr_vals.rolling(consol_p * 2).quantile(atr_pct)
        vol_avg    = volume.rolling(consol_p).mean()

        # Consolidation range boundaries
        consol_high = high.rolling(consol_p).max()
        consol_low  = low.rolling(consol_p).min()

        # ── Try to extract OI data ────────────────────────────────────────
        oi_arr: np.ndarray | None = None
        if aux_data and "open_interest" in aux_data:
            oi_map = aux_data["open_interest"]
            if isinstance(oi_map, dict) and symbol in oi_map:
                oi_df = oi_map[symbol]
                if oi_df is not None and not oi_df.empty:
                    try:
                        oi_col = next(
                            c for c in ("open_interest", "oi", "sumOpenInterest")
                            if c in oi_df.columns
                        )
                        oi_indexed = oi_df.set_index("timestamp")[oi_col]
                        aligned = oi_indexed.reindex(
                            candles["timestamp"].values,
                            method="nearest",
                            tolerance=pd.Timedelta("5min"),
                        ).fillna(method="ffill")
                        oi_arr = aligned.values.astype(float)
                    except Exception:
                        oi_arr = None

        # ── numpy arrays ─────────────────────────────────────────────────
        close_arr = close.values
        vol_arr   = volume.values
        va_arr    = vol_avg.values
        atr_arr   = atr_vals.values
        thr_arr   = atr_thresh.values
        ch_arr    = consol_high.values
        cl_arr    = consol_low.values
        ts_arr    = candles["timestamp"].dt.to_pydatetime()
        is_clean  = candles["is_clean"].values

        warmup   = consol_p * 2 + atr_p + 2
        signals: list[Signal] = []
        open_pos: str | None  = None

        for i in range(warmup, len(candles)):
            if not is_clean[i]:
                continue
            c  = close_arr[i]
            at = atr_arr[i]
            thr = thr_arr[i]
            v  = vol_arr[i]
            va = va_arr[i]
            ch = ch_arr[i]
            cl = cl_arr[i]

            if np.isnan(at) or np.isnan(thr) or np.isnan(va) or va == 0:
                continue

            in_consol = at <= thr
            vol_ok    = v > vol_mult * va

            if not in_consol:
                # Check whether previous bars were consolidating
                look_back = min(4, i)
                prev_atr  = atr_arr[i - look_back : i]
                prev_thr  = thr_arr[i - look_back : i]
                was_consol = np.any(prev_atr <= prev_thr)

                if was_consol and vol_ok:
                    # OI confirmation: OI was rising during consolidation
                    oi_rising = True  # default if no data
                    if oi_arr is not None:
                        oi_window = oi_arr[max(0, i - consol_p) : i]
                        valid_oi  = oi_window[~np.isnan(oi_window)]
                        if len(valid_oi) >= 3:
                            oi_rising = float(valid_oi[-1]) >= float(valid_oi[0])

                    if c > ch and oi_rising and open_pos != "LONG":
                        open_pos = "LONG"
                        signals.append(Signal(
                            strategy=self.name, symbol=symbol, direction="LONG",
                            timestamp=ts_arr[i],
                            strength=min((c - ch) / max(at, 1e-8), 1.0),
                            close_price=c, atr=at,
                            reason=["MCB_breakout_long", f"range={cl:.0f}-{ch:.0f}"],
                        ))
                    elif c < cl and oi_rising and open_pos != "SHORT":
                        open_pos = "SHORT"
                        signals.append(Signal(
                            strategy=self.name, symbol=symbol, direction="SHORT",
                            timestamp=ts_arr[i],
                            strength=min((cl - c) / max(at, 1e-8), 1.0),
                            close_price=c, atr=at,
                            reason=["MCB_breakout_short", f"range={cl:.0f}-{ch:.0f}"],
                        ))

            # Exit: breakout failed (price reverts bk_mult × ATR back into range)
            if open_pos == "LONG" and c < ch - bk_mult * at:
                open_pos = None
                signals.append(Signal(
                    strategy=self.name, symbol=symbol, direction="EXIT_LONG",
                    timestamp=ts_arr[i], strength=0.5,
                    close_price=c, atr=at, reason=["MCB_breakout_failed"],
                ))
            elif open_pos == "SHORT" and c > cl + bk_mult * at:
                open_pos = None
                signals.append(Signal(
                    strategy=self.name, symbol=symbol, direction="EXIT_SHORT",
                    timestamp=ts_arr[i], strength=0.5,
                    close_price=c, atr=at, reason=["MCB_breakout_failed"],
                ))

        return signals

    def generate_signals_mtf(self, candles_by_tf: dict, aux_data=None) -> list[Signal]:
        primary = candles_by_tf.get("5m")
        if primary is None:
            primary = next(iter(candles_by_tf.values()))
        return self.generate_signals(primary, aux_data)
