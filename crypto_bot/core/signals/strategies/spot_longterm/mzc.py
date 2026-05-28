"""
MVRV_ZScore_Cycle — Long-term spot strategy.

Market Value to Realised Value Z-Score measures whether BTC is statistically
over- or under-valued relative to its 'fair value' (realised price):

  Z-Score = (MV - RV) / σ(MV)

  Zones:
    Z < 0     → historical price below realised value → max accumulate
    0 ≤ Z < 3 → neutral / hold (already own coins)
    3 ≤ Z < 7 → overbought, reduce position in tranches
    Z ≥ 7     → extreme bubble → full exit

  Position sizing is proportional to zone (not binary):
    1.0 at Z < 0, scaling to 0.0 at Z ≥ 7

  Degrades to 200-week SMA model when no Glassnode data:
    price < 200w SMA → buy / price > 2× 200w SMA → sell
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from crypto_bot.core.signals.base import BaseStrategy
from crypto_bot.core.signals.models import Signal
from crypto_bot.core.signals.indicators import atr, ema


# MVRV Z-Score threshold table
_ZONE_STRONG_BUY  = 0.0
_ZONE_REDUCE_FROM = 3.0
_ZONE_EXIT_FROM   = 7.0


class MVRVZScoreCycleStrategy(BaseStrategy):
    """MVRV Z-Score cycle allocator — max buy at lows, exit at bubble extremes."""

    @property
    def mode(self) -> str:
        return "spot_longterm"

    @property
    def param_space(self) -> dict:
        return {
            "z_buy_threshold":    (-1.0, 1.0),         # buy when Z below this
            "z_reduce_from":      (2.5,  5.0),          # start reducing above this
            "z_exit_threshold":   (6.0,  9.0),          # full exit above this
            "smoothing":          (3,    14,  "int"),    # EMA smoothing for Z-Score
            "ma_period_fallback": (150,  250, "int"),    # 200w fallback MA period
            "atr_period":         (10,   20,  "int"),
        }

    def _get_zscore_series(
        self, candles: pd.DataFrame, aux_data: Any, coin: str
    ) -> pd.Series | None:
        """Extract MVRV Z-Score from onchain data, aligned to candle timestamps."""
        if aux_data is None:
            return None
        onchain = aux_data.get("onchain")
        if not isinstance(onchain, dict):
            return None
        for key in (f"{coin}_mvrv_zscore", f"{coin}_mvrv"):
            df = onchain.get(key)
            if df is None or (hasattr(df, "empty") and df.empty):
                continue
            try:
                val_col = next(
                    c for c in ("value", "v", "zscore", "mvrv_zscore", "mvrv")
                    if c in df.columns
                )
                ts_col = "timestamp" if "timestamp" in df.columns else df.index.name
                if ts_col and ts_col != df.index.name:
                    series = df.set_index(ts_col)[val_col]
                else:
                    series = df[val_col]
                aligned = series.reindex(
                    candles["timestamp"].values,
                    method="nearest",
                    tolerance=pd.Timedelta("7D"),
                ).interpolate(limit=5)
                return aligned
            except Exception:
                continue
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

        z_buy     = float(p["z_buy_threshold"])
        z_reduce  = float(p["z_reduce_from"])
        z_exit    = float(p["z_exit_threshold"])
        smoothing = int(p["smoothing"])
        ma_p      = int(p["ma_period_fallback"])
        atr_p     = int(p["atr_period"])

        atr_vals = atr(high, low, close, atr_p)
        trend_ma = close.rolling(ma_p).mean()   # fallback 200w SMA

        z_raw    = self._get_zscore_series(candles, aux_data, coin)
        has_z    = z_raw is not None and not z_raw.isna().all()

        if has_z:
            z_smooth = z_raw.ewm(span=smoothing, adjust=False).mean()
            z_arr    = z_smooth.values
        else:
            z_arr    = np.full(len(candles), np.nan)

        close_arr = close.values
        ma_arr    = trend_ma.values
        atr_arr   = atr_vals.values
        ts_arr    = candles["timestamp"].dt.to_pydatetime()
        is_clean  = candles["is_clean"].values

        warmup   = max(ma_p, atr_p, smoothing) + 2
        signals: list[Signal] = []
        open_pos: str | None  = None

        for i in range(warmup, len(candles)):
            if not is_clean[i]:
                continue
            c  = close_arr[i]
            at = atr_arr[i]
            ts = ts_arr[i]

            if np.isnan(at):
                continue

            if has_z and not np.isnan(z_arr[i]):
                z = z_arr[i]
                # Strong buy zone
                if z < z_buy and open_pos != "LONG":
                    open_pos = "LONG"
                    strength = min(max((z_buy - z) / max(abs(z_buy) + 1, 1), 0.0), 1.0)
                    signals.append(Signal(
                        strategy=self.name, symbol=symbol, direction="LONG",
                        timestamp=ts, strength=max(strength, 0.5),
                        close_price=c, atr=at,
                        reason=["mvrv_z_strong_buy", f"Z={z:.2f}"],
                    ))
                # Reduce zone — only exit if we're in a position
                elif z >= z_exit and open_pos == "LONG":
                    open_pos = None
                    signals.append(Signal(
                        strategy=self.name, symbol=symbol, direction="EXIT_LONG",
                        timestamp=ts, strength=1.0,
                        close_price=c, atr=at,
                        reason=["mvrv_z_bubble_exit", f"Z={z:.2f}"],
                    ))
                elif z >= z_reduce and open_pos == "LONG":
                    # Gradual reduce signal (re-enter only if Z drops again)
                    open_pos = None
                    strength = min((z - z_reduce) / max(z_exit - z_reduce, 1), 1.0)
                    signals.append(Signal(
                        strategy=self.name, symbol=symbol, direction="EXIT_LONG",
                        timestamp=ts, strength=strength,
                        close_price=c, atr=at,
                        reason=["mvrv_z_reduce", f"Z={z:.2f}"],
                    ))
            else:
                # Fallback: price vs 200-week MA
                ma = ma_arr[i]
                if np.isnan(ma):
                    continue
                if c < ma and open_pos != "LONG":
                    open_pos = "LONG"
                    signals.append(Signal(
                        strategy=self.name, symbol=symbol, direction="LONG",
                        timestamp=ts, strength=0.6,
                        close_price=c, atr=at,
                        reason=["mvrv_200ma_fallback_buy"],
                    ))
                elif c > ma * 2.0 and open_pos == "LONG":
                    open_pos = None
                    signals.append(Signal(
                        strategy=self.name, symbol=symbol, direction="EXIT_LONG",
                        timestamp=ts, strength=0.8,
                        close_price=c, atr=at,
                        reason=["mvrv_200ma_2x_exit"],
                    ))

        return signals

    def generate_signals_mtf(
        self, candles_by_tf: dict, aux_data: Any = None
    ) -> list[Signal]:
        # Prefer 1w for cycle detection; fall back to 1d
        primary = candles_by_tf.get("1w")
        if primary is None:
            primary = candles_by_tf.get("1d")
        if primary is None:
            primary = next(iter(candles_by_tf.values()))
        return self.generate_signals(primary, aux_data)
