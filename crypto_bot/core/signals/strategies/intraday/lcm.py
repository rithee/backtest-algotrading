"""
LiquidationCascadeMomentum — Intraday futures strategy.

Logic:
  When a large liquidation cluster is swept, forced buy/sell orders
  create a momentum burst in the direction of the cascade:

    - Long cascade:  large liquidations detected + positive momentum + price above trend
    - Short cascade: large liquidations detected + negative momentum + price below trend

  If CoinGlass liquidation data is available (aux_data["liquidations"][symbol]),
  uses actual liquidation volume to detect spikes.
  Without it, degrades gracefully to pure volume-spike + momentum signals.

  Exit: momentum reverses (price pulls back against the cascade).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from crypto_bot.core.signals.base import BaseStrategy
from crypto_bot.core.signals.models import Signal
from crypto_bot.core.signals.indicators import atr, ema


class LiquidationCascadeMomentumStrategy(BaseStrategy):
    """Liquidation-driven momentum — with graceful fallback to volume+momentum."""

    @property
    def mode(self) -> str:
        return "intraday"

    @property
    def param_space(self) -> dict:
        return {
            "liq_thresh_mult": (0.5,  2.0),         # liq spike vs rolling avg
            "liq_avg_period":  (5,    20,  "int"),   # rolling avg window for liq
            "mom_period":      (5,    20,  "int"),   # price momentum lookback (bars)
            "mom_threshold":   (0.003, 0.015),       # min momentum % to qualify
            "vol_mult":        (1.5,  3.0),          # fallback volume threshold
            "atr_period":      (10,   20,  "int"),
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

        liq_mult  = float(p["liq_thresh_mult"])
        liq_avg_p = int(p["liq_avg_period"])
        mom_p     = int(p["mom_period"])
        mom_thr   = float(p["mom_threshold"])
        vol_mult  = float(p["vol_mult"])
        atr_p     = int(p["atr_period"])

        atr_vals = atr(high, low, close, atr_p)
        momentum = close.pct_change(mom_p)
        trend    = ema(close, 20)
        vol_avg  = volume.rolling(10).mean()

        # ── Try to extract and align liquidation data ─────────────────────
        liq_arr: np.ndarray | None = None
        if aux_data and "liquidations" in aux_data:
            liq_map = aux_data["liquidations"]
            liq_df: pd.DataFrame | None = None
            if isinstance(liq_map, dict):
                liq_df = liq_map.get(symbol)

            if liq_df is not None and not liq_df.empty:
                try:
                    liq_col = next(
                        c for c in ("liq_usd", "liquidation_usd", "value")
                        if c in liq_df.columns
                    )
                    liq_indexed = liq_df.set_index("timestamp")[liq_col]
                    aligned = liq_indexed.reindex(
                        candles["timestamp"].values,
                        method="nearest",
                        tolerance=pd.Timedelta("5min"),
                    ).fillna(0.0)
                    liq_arr = aligned.values
                except Exception:
                    liq_arr = None

        # ── numpy arrays ─────────────────────────────────────────────────
        close_arr = close.values
        mom_arr   = momentum.values
        trend_arr = trend.values
        atr_arr   = atr_vals.values
        vol_arr   = volume.values
        va_arr    = vol_avg.values
        ts_arr    = candles["timestamp"].dt.to_pydatetime()
        is_clean  = candles["is_clean"].values

        warmup   = max(liq_avg_p, mom_p, atr_p, 10) + 2
        signals: list[Signal] = []
        open_pos: str | None  = None

        for i in range(warmup, len(candles)):
            if not is_clean[i]:
                continue
            c  = close_arr[i]
            mo = mom_arr[i]
            tr = trend_arr[i]
            at = atr_arr[i]
            v  = vol_arr[i]
            va = va_arr[i]

            if np.isnan(mo) or np.isnan(tr) or np.isnan(at) or np.isnan(va) or va == 0:
                continue

            # Cascade detection: prefer liquidation data, fall back to volume
            if liq_arr is not None:
                liq_window = liq_arr[max(0, i - liq_avg_p) : i]
                liq_avg = np.nanmean(liq_window) if len(liq_window) > 0 else 0.0
                liq_spike = (
                    liq_arr[i] > liq_mult * liq_avg
                    if (liq_avg > 0 and not np.isnan(liq_arr[i]))
                    else False
                )
                cascade_up   = liq_spike and mo >  mom_thr and c > tr
                cascade_down = liq_spike and mo < -mom_thr and c < tr
            else:
                # Fallback: volume spike + strong momentum
                vol_spike    = v > vol_mult * va
                cascade_up   = vol_spike and mo >  mom_thr * 1.5 and c > tr
                cascade_down = vol_spike and mo < -mom_thr * 1.5 and c < tr

            if cascade_up and open_pos != "LONG":
                open_pos = "LONG"
                signals.append(Signal(
                    strategy=self.name, symbol=symbol, direction="LONG",
                    timestamp=ts_arr[i],
                    strength=min(abs(mo) / (mom_thr + 1e-8), 1.0),
                    close_price=c, atr=at,
                    reason=["lcm_cascade_long", f"mom={mo:.3f}"],
                ))
            elif cascade_down and open_pos != "SHORT":
                open_pos = "SHORT"
                signals.append(Signal(
                    strategy=self.name, symbol=symbol, direction="SHORT",
                    timestamp=ts_arr[i],
                    strength=min(abs(mo) / (mom_thr + 1e-8), 1.0),
                    close_price=c, atr=at,
                    reason=["lcm_cascade_short", f"mom={mo:.3f}"],
                ))
            elif open_pos == "LONG" and mo < -mom_thr * 0.5:
                open_pos = None
                signals.append(Signal(
                    strategy=self.name, symbol=symbol, direction="EXIT_LONG",
                    timestamp=ts_arr[i], strength=0.5,
                    close_price=c, atr=at, reason=["lcm_momentum_flip"],
                ))
            elif open_pos == "SHORT" and mo > mom_thr * 0.5:
                open_pos = None
                signals.append(Signal(
                    strategy=self.name, symbol=symbol, direction="EXIT_SHORT",
                    timestamp=ts_arr[i], strength=0.5,
                    close_price=c, atr=at, reason=["lcm_momentum_flip"],
                ))

        return signals

    def generate_signals_mtf(self, candles_by_tf: dict, aux_data=None) -> list[Signal]:
        primary = candles_by_tf.get("5m")
        if primary is None:
            primary = next(iter(candles_by_tf.values()))
        return self.generate_signals(primary, aux_data)
