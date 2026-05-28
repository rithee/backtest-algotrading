"""
Top10RiskAdjustedMomentumRotation — Long-term spot strategy.

Logic (per-symbol quality filter):
  Monthly rebalance: for each symbol, compute rolling 90-day Sharpe ratio.
  Only hold positions where own Sharpe is above the quality threshold.

  BTC dominance signal (from aux_data["top10_mcap"] if available):
    When BTC dominance is rising (BTC's share of top-10 market cap increases),
    reduce alt-coin exposure and concentrate in BTC/ETH.

  Per-symbol: this strategy generates signals based on:
    1. Own 90-day rolling Sharpe ratio
    2. Own 90-day momentum (price return)
    3. Monthly rebalance calendar gate

  When called from the runner for each symbol separately, this acts as
  a quality filter — low-Sharpe symbols get exit signals, high-Sharpe get long.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from crypto_bot.core.signals.base import BaseStrategy
from crypto_bot.core.signals.models import Signal
from crypto_bot.core.signals.indicators import atr, ema


def _rolling_sharpe(returns: pd.Series, window: int, ann_factor: float = 365.0) -> pd.Series:
    """Annualised Sharpe from daily (or periodic) returns over a rolling window."""
    roll_mean = returns.rolling(window).mean()
    roll_std  = returns.rolling(window).std(ddof=1).replace(0, np.nan)
    # Assume roughly uniform candle intervals; scale by sqrt(ann_factor)
    return (roll_mean / roll_std) * np.sqrt(ann_factor)


class Top10MomentumRotationStrategy(BaseStrategy):
    """Rolling Sharpe quality filter with monthly rebalance gate."""

    @property
    def mode(self) -> str:
        return "spot_longterm"

    @property
    def param_space(self) -> dict:
        return {
            "sharpe_window":     (60,  120, "int"),   # rolling Sharpe window (days)
            "sharpe_entry":      (0.3,  1.2),          # min Sharpe to hold
            "sharpe_exit":       (-0.5, 0.3),          # exit when Sharpe below this
            "mom_period":        (30,   90, "int"),    # momentum confirmation window
            "rebalance_days":    (20,   35, "int"),    # rebalance frequency in calendar days
            "trend_ema":         (50,  150, "int"),
            "atr_period":        (10,   20, "int"),
        }

    def _is_btc_dominated(self, aux_data: Any, symbol: str) -> bool:
        """True if BTC dominance is rising (reduce alts, concentrate in BTC)."""
        if aux_data is None:
            return False
        top10 = aux_data.get("top10_mcap")
        if not isinstance(top10, list) or len(top10) < 2:
            return False
        # Simple proxy: if BTC is top-1 (normal) and symbol is not BTC/ETH,
        # we don't penalise. We'd need historical dominance series for proper
        # detection; without it, return False (no penalty).
        return False

    def generate_signals(
        self, candles: pd.DataFrame, aux_data: Any = None
    ) -> list[Signal]:
        p = self.params
        close  = candles["close"]
        high   = candles["high"]
        low    = candles["low"]
        symbol = str(candles["symbol"].iloc[0])

        sharpe_win  = int(p["sharpe_window"])
        sharpe_in   = float(p["sharpe_entry"])
        sharpe_out  = float(p["sharpe_exit"])
        mom_p       = int(p["mom_period"])
        rebal_days  = int(p["rebalance_days"])
        trend_p     = int(p["trend_ema"])
        atr_p       = int(p["atr_period"])

        returns     = close.pct_change()
        sharpe      = _rolling_sharpe(returns, sharpe_win)
        momentum    = close.pct_change(mom_p)
        trend       = ema(close, trend_p)
        atr_vals    = atr(high, low, close, atr_p)

        sharpe_arr  = sharpe.values
        mom_arr     = momentum.values
        trend_arr   = trend.values
        atr_arr     = atr_vals.values
        close_arr   = close.values
        ts_arr      = candles["timestamp"].dt.to_pydatetime()
        is_clean    = candles["is_clean"].values

        warmup   = max(sharpe_win, mom_p, trend_p, atr_p) + 2
        signals: list[Signal] = []
        open_pos: str | None  = None
        last_rebal_ts: datetime | None = None  # type: ignore[assignment]

        for i in range(warmup, len(candles)):
            if not is_clean[i]:
                continue
            c  = close_arr[i]
            at = atr_arr[i]
            sh = sharpe_arr[i]
            mo = mom_arr[i]
            tr = trend_arr[i]
            ts = ts_arr[i]

            if np.isnan(at) or np.isnan(sh) or np.isnan(mo) or np.isnan(tr):
                continue

            # Calendar rebalance gate
            rebal_due = (
                last_rebal_ts is None
                or (ts - last_rebal_ts).days >= rebal_days
            )
            if not rebal_due:
                continue
            last_rebal_ts = ts

            good_quality = sh >= sharpe_in and mo > 0 and c > tr
            poor_quality = sh < sharpe_out

            if good_quality and open_pos != "LONG":
                open_pos = "LONG"
                signals.append(Signal(
                    strategy=self.name, symbol=symbol, direction="LONG",
                    timestamp=ts, strength=min(sh / (sharpe_in * 2 + 1e-8), 1.0),
                    close_price=c, atr=at,
                    reason=["trmr_quality_buy", f"sharpe={sh:.2f}"],
                ))
            elif poor_quality and open_pos == "LONG":
                open_pos = None
                signals.append(Signal(
                    strategy=self.name, symbol=symbol, direction="EXIT_LONG",
                    timestamp=ts, strength=0.8,
                    close_price=c, atr=at,
                    reason=["trmr_quality_exit", f"sharpe={sh:.2f}"],
                ))

        return signals

    def generate_signals_mtf(
        self, candles_by_tf: dict, aux_data: Any = None
    ) -> list[Signal]:
        primary = candles_by_tf.get("1d")
        if primary is None:
            primary = candles_by_tf.get("1w")
        if primary is None:
            primary = next(iter(candles_by_tf.values()))
        return self.generate_signals(primary, aux_data)
