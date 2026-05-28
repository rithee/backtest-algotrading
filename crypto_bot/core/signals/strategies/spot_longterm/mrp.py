"""
MacroRegimePortfolio — Long-term spot strategy.

Logic:
  Score 4 macro indicators — each adds 1 point if bullish for crypto:

    1. DXY Trend:    DXY falling (dollar weakening) → +1  (good for BTC)
    2. SPX Momentum: SPX positive 30d return        → +1  (risk-on)
    3. Real Yields:  10Y TIPS yield falling          → +1  (low real rates = bullish)
    4. Gold Trend:   Gold rising                     → +1  (inflation hedge demand)

  Score ≥ 3 → full position (LONG)
  Score 2   → hold (no change)
  Score ≤ 1 → exit to cash (EXIT_LONG)

  Requires aux_data["macro"] DataFrame with columns: spx, dxy, gold, real_yield
  from yfinance + FRED.  Degrades to BTC-momentum only when unavailable.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from crypto_bot.core.signals.base import BaseStrategy
from crypto_bot.core.signals.models import Signal
from crypto_bot.core.signals.indicators import atr, ema


class MacroRegimePortfolioStrategy(BaseStrategy):
    """4-signal macro score → position sizing (long-only spot)."""

    @property
    def mode(self) -> str:
        return "spot_longterm"

    @property
    def param_space(self) -> dict:
        return {
            "macro_window":   (20,  60,  "int"),   # lookback for macro momentum
            "entry_score":    (2,    4,   "int"),  # min macro score to buy
            "exit_score":     (0,    2,   "int"),  # max macro score before exit
            "trend_ema":      (50,  200, "int"),   # crypto price trend
            "recheck_days":   (14,   45, "int"),   # how often to re-evaluate macro
            "atr_period":     (10,   20, "int"),
        }

    def _align_macro(
        self, candles: pd.DataFrame, aux_data: Any
    ) -> dict[str, np.ndarray | None]:
        """Align macro DataFrame columns to candle timestamps."""
        result: dict[str, np.ndarray | None] = {
            "spx": None, "dxy": None, "gold": None, "real_yield": None,
        }
        if aux_data is None:
            return result
        macro_df = aux_data.get("macro")
        if macro_df is None or (hasattr(macro_df, "empty") and macro_df.empty):
            return result
        try:
            candle_dates = pd.to_datetime(candles["timestamp"].values).normalize()
            for col in result:
                if col not in macro_df.columns:
                    continue
                series = macro_df[col]
                aligned = series.reindex(candle_dates, method="ffill")
                aligned.index = pd.RangeIndex(len(aligned))
                result[col] = aligned.values.astype(float)
        except Exception:
            pass
        return result

    def _macro_score(
        self,
        i: int,
        macro: dict[str, np.ndarray | None],
        window: int,
    ) -> tuple[int, list[str]]:
        """Compute macro score at candle index i. Returns (score, reasons)."""
        score   = 0
        reasons = []

        def _momentum(arr: np.ndarray | None) -> float | None:
            if arr is None:
                return None
            j = max(0, i - window)
            if np.isnan(arr[i]) or np.isnan(arr[j]):
                return None
            base = arr[j]
            if base == 0:
                return None
            return (arr[i] - base) / abs(base)

        # 1. DXY falling → good for BTC
        dxy_mom = _momentum(macro["dxy"])
        if dxy_mom is not None and dxy_mom < -0.01:
            score += 1
            reasons.append("dxy_falling")

        # 2. SPX positive momentum → risk-on
        spx_mom = _momentum(macro["spx"])
        if spx_mom is not None and spx_mom > 0.02:
            score += 1
            reasons.append("spx_bullish")

        # 3. Real yield falling → expansionary
        ry_arr = macro["real_yield"]
        if ry_arr is not None:
            j = max(0, i - window)
            if not (np.isnan(ry_arr[i]) or np.isnan(ry_arr[j])):
                if ry_arr[i] < ry_arr[j]:
                    score += 1
                    reasons.append("real_yield_falling")

        # 4. Gold rising → inflation hedge demand
        gold_mom = _momentum(macro["gold"])
        if gold_mom is not None and gold_mom > 0.01:
            score += 1
            reasons.append("gold_rising")

        return score, reasons

    def generate_signals(
        self, candles: pd.DataFrame, aux_data: Any = None
    ) -> list[Signal]:
        p = self.params
        close  = candles["close"]
        high   = candles["high"]
        low    = candles["low"]
        symbol = str(candles["symbol"].iloc[0])

        macro_win   = int(p["macro_window"])
        entry_score = int(p["entry_score"])
        exit_score  = int(p["exit_score"])
        trend_p     = int(p["trend_ema"])
        recheck     = int(p["recheck_days"])
        atr_p       = int(p["atr_period"])

        atr_vals = atr(high, low, close, atr_p)
        trend    = ema(close, trend_p)

        macro    = self._align_macro(candles, aux_data)
        has_macro = any(v is not None for v in macro.values())

        close_arr = close.values
        trend_arr = trend.values
        atr_arr   = atr_vals.values
        ts_arr    = candles["timestamp"].dt.to_pydatetime()
        is_clean  = candles["is_clean"].values

        warmup   = max(trend_p, macro_win, atr_p) + 2
        signals: list[Signal] = []
        open_pos: str | None  = None
        last_check_ts = None

        for i in range(warmup, len(candles)):
            if not is_clean[i]:
                continue
            c  = close_arr[i]
            at = atr_arr[i]
            tr = trend_arr[i]
            ts = ts_arr[i]

            if np.isnan(at) or np.isnan(tr):
                continue

            # Recheck gate
            recheck_due = (
                last_check_ts is None
                or (ts - last_check_ts).days >= recheck
            )
            if not recheck_due:
                continue
            last_check_ts = ts

            if has_macro:
                score, reasons = self._macro_score(i, macro, macro_win)
                strength = score / 4.0  # normalise 0–1

                if score >= entry_score and c > tr and open_pos != "LONG":
                    open_pos = "LONG"
                    signals.append(Signal(
                        strategy=self.name, symbol=symbol, direction="LONG",
                        timestamp=ts, strength=strength,
                        close_price=c, atr=at,
                        reason=["mrp_macro_buy", f"score={score}/4"] + reasons,
                    ))
                elif score <= exit_score and open_pos == "LONG":
                    open_pos = None
                    signals.append(Signal(
                        strategy=self.name, symbol=symbol, direction="EXIT_LONG",
                        timestamp=ts, strength=1.0 - strength,
                        close_price=c, atr=at,
                        reason=["mrp_macro_exit", f"score={score}/4"],
                    ))
            else:
                # Fallback: simple BTC trend-following
                if c > tr and open_pos != "LONG":
                    open_pos = "LONG"
                    signals.append(Signal(
                        strategy=self.name, symbol=symbol, direction="LONG",
                        timestamp=ts, strength=0.5,
                        close_price=c, atr=at,
                        reason=["mrp_trend_fallback_long"],
                    ))
                elif c < tr and open_pos == "LONG":
                    open_pos = None
                    signals.append(Signal(
                        strategy=self.name, symbol=symbol, direction="EXIT_LONG",
                        timestamp=ts, strength=0.5,
                        close_price=c, atr=at,
                        reason=["mrp_trend_fallback_exit"],
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
