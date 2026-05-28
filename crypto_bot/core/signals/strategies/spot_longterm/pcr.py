"""
PiCycleRainbowComposite — Long-term spot strategy.

Two independent cycle indicators combined into a composite score:

  Pi Cycle Top indicator:
    - 111-day SMA and 350-day SMA (×2)
    - When 111d SMA crosses ABOVE 2×350d SMA → cycle top → EXIT
    - When 111d SMA well below 2×350d SMA → bottom zone → LONG

  Rainbow Chart (logarithmic regression bands):
    - Fit log-linear regression: log(price) = a + b×t
    - Compute residual standard deviation
    - Price in bottom 2 bands (low residual) → undervalued → LONG
    - Price in top 2 bands (high residual) → overvalued → EXIT

  Composite: average of both scores.
    > 0.6 → LONG,  < 0.3 → EXIT_LONG

  Works on daily candles (uses 111d/350d periods); auto-scales for weekly
  candles by dividing by 7.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from crypto_bot.core.signals.base import BaseStrategy
from crypto_bot.core.signals.models import Signal
from crypto_bot.core.signals.indicators import atr


def _log_regression_residual(log_prices: np.ndarray, window: int) -> np.ndarray:
    """
    Rolling log-linear regression residual.
    residual[i] = log_price[i] - predicted from regression over [i-window+1, i]
    Normalised by residual std to give z-score.
    Returns array of same length (NaN for warmup).
    """
    n = len(log_prices)
    result = np.full(n, np.nan)
    t = np.arange(window, dtype=float)
    for i in range(window - 1, n):
        y = log_prices[i - window + 1 : i + 1]
        if np.any(np.isnan(y)):
            continue
        slope, intercept = np.polyfit(t, y, 1)
        pred   = slope * (window - 1) + intercept
        resids = y - (slope * t + intercept)
        std    = resids.std()
        if std > 0:
            result[i] = (y[-1] - pred) / std
    return result


class PiCycleRainbowCompositeStrategy(BaseStrategy):
    """Pi Cycle top indicator + log-regression rainbow composite."""

    @property
    def mode(self) -> str:
        return "spot_longterm"

    @property
    def param_space(self) -> dict:
        return {
            "pi_fast":       (90,  130, "int"),   # Pi Cycle fast MA (default 111d)
            "pi_slow":       (300, 400, "int"),   # Pi Cycle slow MA (default 350d)
            "pi_slow_mult":  (1.8, 2.2),           # multiplier for slow MA (default 2×)
            "rainbow_win":   (200, 400, "int"),   # log regression window
            "buy_threshold": (0.5, 0.75),          # composite score to go long
            "exit_threshold":(0.2, 0.45),          # composite score to exit
            "atr_period":    (10,  20,  "int"),
        }

    def _candle_period_days(self, candles: pd.DataFrame) -> float:
        """Estimate candle period in days from timestamps."""
        if len(candles) < 2:
            return 1.0
        ts = candles["timestamp"].values
        delta = (pd.Timestamp(ts[1]) - pd.Timestamp(ts[0])).total_seconds()
        return delta / 86400.0

    def generate_signals(
        self, candles: pd.DataFrame, aux_data: Any = None
    ) -> list[Signal]:
        p = self.params
        close  = candles["close"]
        high   = candles["high"]
        low    = candles["low"]
        symbol = str(candles["symbol"].iloc[0])

        # Scale periods based on candle frequency
        period_days = self._candle_period_days(candles)
        scale = max(period_days, 0.5)   # 1.0 for daily, ~7.0 for weekly

        pi_fast    = max(3, int(p["pi_fast"]   / scale))
        pi_slow    = max(5, int(p["pi_slow"]   / scale))
        pi_mult    = float(p["pi_slow_mult"])
        rb_win     = max(10, int(p["rainbow_win"] / scale))
        buy_thr    = float(p["buy_threshold"])
        exit_thr   = float(p["exit_threshold"])
        atr_p      = int(p["atr_period"])

        atr_vals = atr(high, low, close, atr_p)

        # Pi Cycle SMAs
        sma_fast = close.rolling(pi_fast).mean()
        sma_slow = close.rolling(pi_slow).mean()

        # Log-regression rainbow
        log_px = np.log(close.values.clip(1e-8))
        rainbow_z = _log_regression_residual(log_px, rb_win)

        close_arr = close.values
        fast_arr  = sma_fast.values
        slow_arr  = sma_slow.values
        atr_arr   = atr_vals.values
        ts_arr    = candles["timestamp"].dt.to_pydatetime()
        is_clean  = candles["is_clean"].values

        warmup   = max(pi_slow, rb_win, atr_p) + 2
        signals: list[Signal] = []
        open_pos: str | None  = None

        for i in range(warmup, len(candles)):
            if not is_clean[i]:
                continue
            c  = close_arr[i]
            at = atr_arr[i]
            ts = ts_arr[i]
            sf = fast_arr[i]
            ss = slow_arr[i]
            rz = rainbow_z[i]

            if np.isnan(at) or np.isnan(sf) or np.isnan(ss) or np.isnan(rz):
                continue

            slow_target = ss * pi_mult

            # Pi Cycle signal: 0=top (fast above 2×slow), 1=bottom (large gap below)
            pi_gap_pct = (slow_target - sf) / max(slow_target, 1e-8)
            pi_score   = max(min(pi_gap_pct * 3, 1.0), 0.0)   # 1 = very far below → buy

            # Rainbow signal: rz < -1 = undervalued (buy), rz > 2 = overvalued (sell)
            rainbow_score = max(min((-rz + 1) / 3.0, 1.0), 0.0)   # 1 = cheap, 0 = expensive

            composite = (pi_score + rainbow_score) / 2.0

            # Pi Cycle top: fast crosses ABOVE 2×slow → confirmed sell
            pi_cross_up = (
                i > 0
                and fast_arr[i - 1] <= slow_arr[i - 1] * pi_mult
                and sf > slow_target
            )

            if pi_cross_up and open_pos == "LONG":
                open_pos = None
                signals.append(Signal(
                    strategy=self.name, symbol=symbol, direction="EXIT_LONG",
                    timestamp=ts, strength=1.0,
                    close_price=c, atr=at,
                    reason=["pi_cycle_top_cross"],
                ))
            elif composite >= buy_thr and open_pos != "LONG":
                open_pos = "LONG"
                signals.append(Signal(
                    strategy=self.name, symbol=symbol, direction="LONG",
                    timestamp=ts, strength=composite,
                    close_price=c, atr=at,
                    reason=["pi_rainbow_composite_buy", f"score={composite:.2f}"],
                ))
            elif composite <= exit_thr and open_pos == "LONG":
                open_pos = None
                signals.append(Signal(
                    strategy=self.name, symbol=symbol, direction="EXIT_LONG",
                    timestamp=ts, strength=1.0 - composite,
                    close_price=c, atr=at,
                    reason=["pi_rainbow_composite_exit", f"score={composite:.2f}"],
                ))

        return signals

    def generate_signals_mtf(
        self, candles_by_tf: dict, aux_data: Any = None
    ) -> list[Signal]:
        # Prefer 1d for period accuracy; fall back to 1w
        primary = candles_by_tf.get("1d")
        if primary is None:
            primary = candles_by_tf.get("1w")
        if primary is None:
            primary = next(iter(candles_by_tf.values()))
        return self.generate_signals(primary, aux_data)
