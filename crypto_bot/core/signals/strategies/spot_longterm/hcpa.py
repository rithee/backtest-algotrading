"""
HalvingCyclePhaseAllocator — Long-term spot strategy.

BTC operates on ~1460-day (4-year) halving cycles.
Four phases after each halving date:

  Phase 1 — Accumulation (0–365 days post-halving):
    Historically post-halving bear/recovery.  Accumulate steadily.

  Phase 2 — Bull Run (365–548 days):
    Strong uptrend.  Maximum exposure.

  Phase 3 — Distribution (548–730 days):
    Approaching cycle top.  Reduce position gradually, rotate to BTC-only.

  Phase 4 — Bear Market (730–1460 days):
    Defensive.  Hold minimal position or cash.

On-chain confirmation (optional):
  If Glassnode MVRV or NUPL data is available, adjust phase timing.
  High MVRV/NUPL in Phase 1 → already late → skip to Phase 3 allocation.

Known BTC halving dates (UTC):
  2012-11-28, 2016-07-09, 2020-05-11, 2024-04-19, 2028-03-~(estimated)
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from crypto_bot.core.signals.base import BaseStrategy
from crypto_bot.core.signals.models import Signal
from crypto_bot.core.signals.indicators import atr, ema

# Known BTC halving timestamps (UTC)
_HALVING_DATES = [
    datetime(2012, 11, 28, tzinfo=timezone.utc),
    datetime(2016,  7,  9, tzinfo=timezone.utc),
    datetime(2020,  5, 11, tzinfo=timezone.utc),
    datetime(2024,  4, 19, tzinfo=timezone.utc),
    datetime(2028,  3, 21, tzinfo=timezone.utc),  # estimated
]

# Phase durations in days after halving
_PHASE_DAYS = {1: (0, 365), 2: (365, 548), 3: (548, 730), 4: (730, 1460)}

# Target allocation by phase (0 = 0%, 1 = full)
_PHASE_ALLOCATION = {1: 0.70, 2: 1.00, 3: 0.40, 4: 0.15}


def _phase_from_days(days_since_halving: float) -> int:
    for phase, (start, end) in _PHASE_DAYS.items():
        if start <= days_since_halving < end:
            return phase
    return 4  # default defensive


def _days_since_last_halving(ts: datetime) -> float:
    """Days elapsed since the most recent halving before `ts`."""
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    past = [h for h in _HALVING_DATES if h <= ts]
    if not past:
        return 0.0
    return (ts - past[-1]).total_seconds() / 86400.0


class HalvingCyclePhaseAllocatorStrategy(BaseStrategy):
    """4-phase halving cycle allocator with optional on-chain confirmation."""

    @property
    def mode(self) -> str:
        return "spot_longterm"

    @property
    def param_space(self) -> dict:
        return {
            "phase2_start_days":  (300, 420, "int"),  # when to go max exposure
            "phase3_start_days":  (500, 600, "int"),  # when to start reducing
            "phase4_start_days":  (700, 800, "int"),  # when to go defensive
            "onchain_mvrv_adjust":(3.0,  7.0),         # MVRV above this → jump to phase 3
            "trend_ema":          (20,  100, "int"),   # short-term trend filter
            "atr_period":         (10,  20,  "int"),
        }

    def _get_onchain_value(
        self, candles: pd.DataFrame, aux_data: Any, coin: str, metric: str, idx: int
    ) -> float | None:
        """Attempt to read one on-chain value at candle index `idx`."""
        if aux_data is None:
            return None
        onchain = aux_data.get("onchain")
        if not isinstance(onchain, dict):
            return None
        df = onchain.get(f"{coin}_{metric}")
        if df is None or (hasattr(df, "empty") and df.empty):
            return None
        try:
            val_col = next(
                c for c in ("value", "v", metric, "mvrv", "nupl") if c in df.columns
            )
            ts_col = "timestamp" if "timestamp" in df.columns else None
            if ts_col:
                series = df.set_index(ts_col)[val_col]
            else:
                series = df[val_col]
            aligned = series.reindex(
                candles["timestamp"].values,
                method="nearest",
                tolerance=pd.Timedelta("14D"),
            )
            val = float(aligned.iloc[idx])
            return None if np.isnan(val) else val
        except Exception:
            return None

    def generate_signals(
        self, candles: pd.DataFrame, aux_data: Any = None
    ) -> list[Signal]:
        p = self.params
        close  = candles["close"]
        high   = candles["high"]
        low    = candles["low"]
        symbol = str(candles["symbol"].iloc[0])
        coin   = symbol.replace("USDT", "")

        p2_start    = int(p["phase2_start_days"])
        p3_start    = int(p["phase3_start_days"])
        p4_start    = int(p["phase4_start_days"])
        mvrv_thresh = float(p["onchain_mvrv_adjust"])
        trend_p     = int(p["trend_ema"])
        atr_p       = int(p["atr_period"])

        atr_vals = atr(high, low, close, atr_p)
        trend    = ema(close, trend_p)

        close_arr = close.values
        trend_arr = trend.values
        atr_arr   = atr_vals.values
        ts_arr    = candles["timestamp"].dt.to_pydatetime()
        is_clean  = candles["is_clean"].values

        warmup   = max(trend_p, atr_p) + 2
        signals: list[Signal] = []
        open_pos: str | None  = None
        last_phase = -1

        for i in range(warmup, len(candles)):
            if not is_clean[i]:
                continue
            c  = close_arr[i]
            at = atr_arr[i]
            tr = trend_arr[i]
            ts = ts_arr[i]

            if np.isnan(at) or np.isnan(tr):
                continue

            days = _days_since_last_halving(ts)

            # Dynamic phase thresholds from params
            if days < p2_start:
                phase = 1
            elif days < p3_start:
                phase = 2
            elif days < p4_start:
                phase = 3
            else:
                phase = 4

            # On-chain override: if MVRV already elevated in phase 1 or 2 → phase 3
            if phase <= 2 and coin == "BTC":
                mvrv = self._get_onchain_value(candles, aux_data, coin, "mvrv", i)
                if mvrv is not None and mvrv > mvrv_thresh:
                    phase = 3

            target_alloc = _PHASE_ALLOCATION.get(phase, 0.15)
            phase_changed = phase != last_phase
            last_phase = phase

            # Generate signals on phase transitions
            if phase_changed or (phase in (1, 2) and open_pos is None):
                if target_alloc >= 0.6 and open_pos != "LONG":
                    open_pos = "LONG"
                    signals.append(Signal(
                        strategy=self.name, symbol=symbol, direction="LONG",
                        timestamp=ts, strength=target_alloc,
                        close_price=c, atr=at,
                        reason=[f"halving_phase{phase}_accumulate", f"days={days:.0f}"],
                    ))
                elif target_alloc < 0.3 and open_pos == "LONG":
                    open_pos = None
                    signals.append(Signal(
                        strategy=self.name, symbol=symbol, direction="EXIT_LONG",
                        timestamp=ts, strength=1.0 - target_alloc,
                        close_price=c, atr=at,
                        reason=[f"halving_phase{phase}_reduce", f"days={days:.0f}"],
                    ))

        return signals

    def generate_signals_mtf(
        self, candles_by_tf: dict, aux_data: Any = None
    ) -> list[Signal]:
        primary = candles_by_tf.get("1w")
        if primary is None:
            primary = candles_by_tf.get("1d")
        if primary is None:
            primary = next(iter(candles_by_tf.values()))
        return self.generate_signals(primary, aux_data)
