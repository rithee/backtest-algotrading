"""
OnChainAccumulationComposite — Long-term spot strategy.

Scores 6 on-chain signals from Glassnode; each adds 1 point when bullish:

  1. SOPR < 1.0:              Coins sold at a loss → capitulation / accumulation zone
  2. LTH supply increasing:   Long-term holders accumulating (not distributing)
  3. Exchange balance falling: Net outflows → coins leaving exchanges (buy & hold)
  4. NUPL < 0:                 Net Unrealised Profit/Loss negative → fear/capitulation
  5. MVRV < 1.0:              Market cap below realised cap → fundamentally cheap
  6. NVT low (< 65):           Network activity high relative to valuation → fair price

  Composite score:
    5–6 → maximum accumulation (LONG, strength=1.0)
    3–4 → moderate accumulation (LONG, strength=0.6)
    1–2 → hold
    0   → exit (everything bearish)

  Degrades to pure price-based accumulation model when no Glassnode data.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from crypto_bot.core.signals.base import BaseStrategy
from crypto_bot.core.signals.models import Signal
from crypto_bot.core.signals.indicators import atr, ema


class OnChainAccumulationCompositeStrategy(BaseStrategy):
    """6-signal on-chain accumulation composite score."""

    @property
    def mode(self) -> str:
        return "spot_longterm"

    @property
    def param_space(self) -> dict:
        return {
            "smoothing":       (3,   14,  "int"),   # EMA smoothing for on-chain series
            "sopr_threshold":  (0.95, 1.02),         # SOPR below this → bearish / accumulate
            "nupl_threshold":  (-0.1, 0.2),           # NUPL below this → accumulate
            "mvrv_threshold":  (0.8,  1.2),           # MVRV below this → cheap
            "nvt_buy_thresh":  (40.0, 75.0),           # NVT below this → buy
            "nvt_sell_thresh": (100.0, 180.0),         # NVT above this → sell
            "recheck_days":    (14,   45,  "int"),   # re-evaluate frequency
            "atr_period":      (10,   20,  "int"),
        }

    def _get_metric(
        self,
        candles: pd.DataFrame,
        aux_data: Any,
        coin: str,
        metric: str,
        smoothing: int,
    ) -> np.ndarray | None:
        """Fetch, align and smooth one on-chain metric. Returns array or None."""
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
                c for c in ("value", "v", metric, "sopr", "nupl", "mvrv", "nvt",
                             "lth_supply", "exchange_balance")
                if c in df.columns
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

    def generate_signals(
        self, candles: pd.DataFrame, aux_data: Any = None
    ) -> list[Signal]:
        p = self.params
        close  = candles["close"]
        high   = candles["high"]
        low    = candles["low"]
        symbol = str(candles["symbol"].iloc[0])
        coin   = symbol.replace("USDT", "")

        smoothing   = int(p["smoothing"])
        sopr_thr    = float(p["sopr_threshold"])
        nupl_thr    = float(p["nupl_threshold"])
        mvrv_thr    = float(p["mvrv_threshold"])
        nvt_buy     = float(p["nvt_buy_thresh"])
        nvt_sell    = float(p["nvt_sell_thresh"])
        recheck     = int(p["recheck_days"])
        atr_p       = int(p["atr_period"])

        atr_vals = atr(high, low, close, atr_p)
        trend    = ema(close, 100)

        # Fetch on-chain arrays (each may be None)
        sopr_arr = self._get_metric(candles, aux_data, coin, "sopr",             smoothing)
        lth_arr  = self._get_metric(candles, aux_data, coin, "lth_supply",       smoothing)
        exb_arr  = self._get_metric(candles, aux_data, coin, "exchange_balance",  smoothing)
        nupl_arr = self._get_metric(candles, aux_data, coin, "nupl",             smoothing)
        mvrv_arr = self._get_metric(candles, aux_data, coin, "mvrv",             smoothing)
        nvt_arr  = self._get_metric(candles, aux_data, coin, "nvt",              smoothing)

        has_onchain = any(a is not None for a in (sopr_arr, lth_arr, exb_arr, nupl_arr, mvrv_arr, nvt_arr))

        close_arr = close.values
        trend_arr = trend.values
        atr_arr   = atr_vals.values
        ts_arr    = candles["timestamp"].dt.to_pydatetime()
        is_clean  = candles["is_clean"].values

        warmup = max(100, atr_p) + 2
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

            if has_onchain:
                score   = 0
                reasons = []

                def _val(arr: np.ndarray | None) -> float | None:
                    if arr is None or np.isnan(arr[i]):
                        return None
                    return float(arr[i])

                sopr = _val(sopr_arr)
                if sopr is not None and sopr < sopr_thr:
                    score += 1; reasons.append(f"sopr={sopr:.3f}<{sopr_thr}")

                lth = _val(lth_arr)
                if lth is not None and i > 0 and lth_arr is not None:
                    prev = float(lth_arr[i - 1]) if not np.isnan(lth_arr[i - 1]) else lth
                    if lth > prev:
                        score += 1; reasons.append("lth_accumulating")

                exb = _val(exb_arr)
                if exb is not None and i > 0 and exb_arr is not None:
                    prev = float(exb_arr[i - 1]) if not np.isnan(exb_arr[i - 1]) else exb
                    if exb < prev:
                        score += 1; reasons.append("exchange_outflow")

                nupl = _val(nupl_arr)
                if nupl is not None and nupl < nupl_thr:
                    score += 1; reasons.append(f"nupl={nupl:.3f}<{nupl_thr}")

                mvrv = _val(mvrv_arr)
                if mvrv is not None and mvrv < mvrv_thr:
                    score += 1; reasons.append(f"mvrv={mvrv:.2f}<{mvrv_thr}")

                nvt = _val(nvt_arr)
                if nvt is not None and nvt < nvt_buy:
                    score += 1; reasons.append(f"nvt={nvt:.0f}<{nvt_buy}")

                strength = score / 6.0

                if score >= 4 and open_pos != "LONG":
                    open_pos = "LONG"
                    signals.append(Signal(
                        strategy=self.name, symbol=symbol, direction="LONG",
                        timestamp=ts, strength=min(strength, 1.0),
                        close_price=c, atr=at,
                        reason=[f"oac_score={score}/6"] + reasons[:3],
                    ))
                elif score == 0 and nvt is not None and nvt > nvt_sell and open_pos == "LONG":
                    open_pos = None
                    signals.append(Signal(
                        strategy=self.name, symbol=symbol, direction="EXIT_LONG",
                        timestamp=ts, strength=0.9,
                        close_price=c, atr=at,
                        reason=[f"oac_score={score}/6", f"nvt={nvt:.0f}>{nvt_sell}"],
                    ))
            else:
                # Fallback: oversold RSI + below 100-week MA
                if c < tr and open_pos != "LONG":
                    open_pos = "LONG"
                    signals.append(Signal(
                        strategy=self.name, symbol=symbol, direction="LONG",
                        timestamp=ts, strength=0.5,
                        close_price=c, atr=at,
                        reason=["oac_price_below_ma_fallback"],
                    ))
                elif c > tr * 2.5 and open_pos == "LONG":
                    open_pos = None
                    signals.append(Signal(
                        strategy=self.name, symbol=symbol, direction="EXIT_LONG",
                        timestamp=ts, strength=0.8,
                        close_price=c, atr=at,
                        reason=["oac_price_extreme_fallback"],
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
