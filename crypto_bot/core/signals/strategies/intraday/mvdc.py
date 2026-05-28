"""
VWAPDeltaConfluence — Intraday futures strategy.

Logic:
  Rolling VWAP ± std_mult × σ acts as dynamic support/resistance.
  Volume delta (cumulative buy pressure − sell pressure, approximated from
  OHLCV) confirms whether order flow supports the trade direction.

  Entry:
    - LONG:  close < VWAP − std_mult*σ  AND  delta improving AND  close > trend EMA
    - SHORT: close > VWAP + std_mult*σ  AND  delta weakening  AND  close < trend EMA

  Exit: price mean-reverts back through VWAP.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from crypto_bot.core.signals.base import BaseStrategy
from crypto_bot.core.signals.models import Signal
from crypto_bot.core.signals.indicators import atr, ema, rolling_vwap


class VWAPDeltaConfluenceStrategy(BaseStrategy):
    """Rolling-VWAP deviation trade with volume-delta order-flow filter."""

    @property
    def mode(self) -> str:
        return "intraday"

    @property
    def param_space(self) -> dict:
        return {
            "vwap_period":  (10, 40,  "int"),  # rolling VWAP window
            "std_mult":     (1.0, 2.5),         # deviation threshold in σ
            "delta_period": (5,  20,  "int"),   # volume-delta smoothing window
            "trend_ema":    (20, 60,  "int"),   # trend direction EMA
            "atr_period":   (10, 20,  "int"),
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

        vwap_p    = int(p["vwap_period"])
        std_mult  = float(p["std_mult"])
        delta_p   = int(p["delta_period"])
        trend_p   = int(p["trend_ema"])
        atr_p     = int(p["atr_period"])

        atr_vals     = atr(high, low, close, atr_p)
        vwap_, vstd  = rolling_vwap(high, low, close, volume, vwap_p)
        trend        = ema(close, trend_p)

        # Volume delta: positive when candle closes higher than previous close
        prev_close = close.shift(1).fillna(close)
        price_dir  = pd.Series(
            np.where(close.values > prev_close.values, 1.0, -1.0),
            index=close.index,
        )
        delta = (volume * price_dir).rolling(delta_p).sum()

        # ── numpy arrays ─────────────────────────────────────────────────
        close_arr = close.values
        vwap_arr  = vwap_.values
        vstd_arr  = vstd.values
        delta_arr = delta.values
        trend_arr = trend.values
        atr_arr   = atr_vals.values
        ts_arr    = candles["timestamp"].dt.to_pydatetime()
        is_clean  = candles["is_clean"].values

        warmup   = max(vwap_p, trend_p, atr_p, delta_p) + 2
        signals: list[Signal] = []
        open_pos: str | None  = None

        for i in range(warmup, len(candles)):
            if not is_clean[i]:
                continue
            c  = close_arr[i]
            vw = vwap_arr[i]
            vs = vstd_arr[i]
            d  = delta_arr[i]
            tr = trend_arr[i]
            at = atr_arr[i]

            if (
                np.isnan(vw) or np.isnan(vs) or vs <= 0
                or np.isnan(d) or np.isnan(tr) or np.isnan(at)
            ):
                continue

            # Compare delta to median of prior 5 bars (direction of order flow)
            lookback = delta_arr[max(0, i - 5) : i]
            med = np.nanmedian(lookback) if len(lookback) > 0 else 0.0
            delta_improving = d > med
            delta_weakening  = d < med

            below_band = c < vw - std_mult * vs
            above_band = c > vw + std_mult * vs

            if below_band and delta_improving and c > tr and open_pos != "LONG":
                open_pos = "LONG"
                signals.append(Signal(
                    strategy=self.name, symbol=symbol, direction="LONG",
                    timestamp=ts_arr[i],
                    strength=min(abs(c - vw) / (vs + 1e-8), 2.0) / 2.0,
                    close_price=c, atr=at,
                    reason=["VWAP_dev_long", f"dev={(c - vw) / vs:.2f}σ"],
                ))
            elif above_band and delta_weakening and c < tr and open_pos != "SHORT":
                open_pos = "SHORT"
                signals.append(Signal(
                    strategy=self.name, symbol=symbol, direction="SHORT",
                    timestamp=ts_arr[i],
                    strength=min(abs(c - vw) / (vs + 1e-8), 2.0) / 2.0,
                    close_price=c, atr=at,
                    reason=["VWAP_dev_short", f"dev={(c - vw) / vs:.2f}σ"],
                ))
            elif open_pos == "LONG" and c > vw:
                open_pos = None
                signals.append(Signal(
                    strategy=self.name, symbol=symbol, direction="EXIT_LONG",
                    timestamp=ts_arr[i], strength=0.5,
                    close_price=c, atr=at, reason=["VWAP_mean_revert"],
                ))
            elif open_pos == "SHORT" and c < vw:
                open_pos = None
                signals.append(Signal(
                    strategy=self.name, symbol=symbol, direction="EXIT_SHORT",
                    timestamp=ts_arr[i], strength=0.5,
                    close_price=c, atr=at, reason=["VWAP_mean_revert"],
                ))

        return signals

    def generate_signals_mtf(self, candles_by_tf: dict, aux_data=None) -> list[Signal]:
        primary = candles_by_tf.get("5m")
        if primary is None:
            primary = next(iter(candles_by_tf.values()))
        return self.generate_signals(primary, aux_data)
