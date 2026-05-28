"""
NVT_SignalValuation — Long-term spot strategy.

Network Value to Transactions ratio — the crypto equivalent of the P/E ratio:

  NVT = Market Cap / On-Chain Transaction Volume (30d smoothed)

  When NVT is low:  the network is being heavily used relative to its value
                   → fundamentally undervalued → accumulate
  When NVT is high: market cap is high relative to on-chain utility
                   → overvalued, price driven by speculation → reduce

  Thresholds (from historical BTC NVT):
    NVT < 45  → deeply undervalued → LONG
    45–100    → fair value zone    → hold
    > 100     → overvalued         → neutral
    > 150     → bubble territory   → EXIT_LONG

  For non-BTC coins with no NVT data, fallback to price / volume ratio
  as a crude on-chain-value proxy.

  Requires aux_data["onchain"]["{COIN}_nvt"] from Glassnode.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from crypto_bot.core.signals.base import BaseStrategy
from crypto_bot.core.signals.models import Signal
from crypto_bot.core.signals.indicators import atr, ema


class NVTSignalValuationStrategy(BaseStrategy):
    """NVT ratio cycle allocator — buy cheap networks, exit expensive ones."""

    @property
    def mode(self) -> str:
        return "spot_longterm"

    @property
    def param_space(self) -> dict:
        return {
            "nvt_buy":       (30.0,  65.0),          # NVT below → accumulate
            "nvt_sell":      (100.0, 200.0),           # NVT above → exit
            "smoothing":     (14,    60,   "int"),    # EMA smoothing period
            "trend_ema":     (50,   200,   "int"),    # price trend filter
            "recheck_days":  (14,    45,   "int"),    # re-evaluate frequency
            "atr_period":    (10,    20,   "int"),
        }

    def _get_nvt_series(
        self, candles: pd.DataFrame, aux_data: Any, coin: str, smoothing: int
    ) -> np.ndarray | None:
        """Fetch NVT from onchain dict, aligned and smoothed."""
        if aux_data is None:
            return None
        onchain = aux_data.get("onchain")
        if not isinstance(onchain, dict):
            return None
        df = onchain.get(f"{coin}_nvt")
        if df is None or (hasattr(df, "empty") and df.empty):
            return None
        try:
            val_col = next(
                c for c in ("value", "v", "nvt", "NVT") if c in df.columns
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
            ).interpolate(limit=7)
            smoothed = aligned.ewm(span=smoothing, adjust=False).mean()
            return smoothed.values.astype(float)
        except Exception:
            return None

    def _price_volume_nvt_proxy(
        self, close: pd.Series, volume: pd.Series, smoothing: int
    ) -> pd.Series:
        """
        Crude fallback NVT proxy: price / 30d avg volume × constant.
        Not equivalent to real NVT but captures the same relative concept.
        """
        vol_avg = volume.rolling(30).mean().replace(0, np.nan)
        ratio   = close / vol_avg
        # Normalise to roughly NVT scale: empirical scale factor
        scaled  = ratio.ewm(span=smoothing, adjust=False).mean()
        return scaled / scaled.rolling(200).median().replace(0, np.nan) * 65.0

    def generate_signals(
        self, candles: pd.DataFrame, aux_data: Any = None
    ) -> list[Signal]:
        p = self.params
        close   = candles["close"]
        high    = candles["high"]
        low     = candles["low"]
        volume  = candles["volume"]
        symbol  = str(candles["symbol"].iloc[0])
        coin    = symbol.replace("USDT", "")

        nvt_buy    = float(p["nvt_buy"])
        nvt_sell   = float(p["nvt_sell"])
        smoothing  = int(p["smoothing"])
        trend_p    = int(p["trend_ema"])
        recheck    = int(p["recheck_days"])
        atr_p      = int(p["atr_period"])

        atr_vals = atr(high, low, close, atr_p)
        trend    = ema(close, trend_p)

        nvt_raw = self._get_nvt_series(candles, aux_data, coin, smoothing)
        if nvt_raw is not None:
            nvt_arr = nvt_raw
            has_nvt = True
        else:
            proxy   = self._price_volume_nvt_proxy(close, volume, smoothing)
            nvt_arr = proxy.values.astype(float)
            has_nvt = False   # degraded mode

        close_arr = close.values
        trend_arr = trend.values
        atr_arr   = atr_vals.values
        ts_arr    = candles["timestamp"].dt.to_pydatetime()
        is_clean  = candles["is_clean"].values

        warmup   = max(trend_p, smoothing, atr_p, 30) + 2
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

            recheck_due = (
                last_check_ts is None
                or (ts - last_check_ts).days >= recheck
            )
            if not recheck_due:
                continue
            last_check_ts = ts

            nvt = nvt_arr[i]
            if np.isnan(nvt):
                continue

            label = "nvt" if has_nvt else "nvt_proxy"

            if nvt < nvt_buy and c > tr * 0.9 and open_pos != "LONG":
                strength = min((nvt_buy - nvt) / max(nvt_buy, 1e-8), 1.0)
                open_pos = "LONG"
                signals.append(Signal(
                    strategy=self.name, symbol=symbol, direction="LONG",
                    timestamp=ts, strength=max(strength, 0.5),
                    close_price=c, atr=at,
                    reason=[f"{label}_undervalued_buy", f"NVT={nvt:.0f}"],
                ))
            elif nvt > nvt_sell and open_pos == "LONG":
                strength = min((nvt - nvt_sell) / max(nvt_sell, 1e-8), 1.0)
                open_pos = None
                signals.append(Signal(
                    strategy=self.name, symbol=symbol, direction="EXIT_LONG",
                    timestamp=ts, strength=min(strength, 1.0),
                    close_price=c, atr=at,
                    reason=[f"{label}_overvalued_exit", f"NVT={nvt:.0f}"],
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
