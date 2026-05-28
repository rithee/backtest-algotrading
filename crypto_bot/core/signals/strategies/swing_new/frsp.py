"""
FundingRateSqueezePredIctor — Swing futures strategy.

Logic (enhanced funding + OI squeeze detector):
  The existing FundingRateReversionStrategy looks at funding level.
  This strategy uses the RATE OF CHANGE of both funding AND OI together:

  Short Squeeze Setup (LONG signal):
    - Funding rate is deeply negative (shorts paying longs)   AND
    - OI is rising (more shorts being opened / piling in)    AND
    - Funding's rate-of-change is accelerating more negative  →
      Short squeeze imminent: any catalyst = violent up move

  Long Squeeze Setup (SHORT signal):
    - Funding rate is highly positive (longs paying shorts)  AND
    - OI is rising (more longs piling in)                    AND
    - Funding's rate-of-change is accelerating more positive →
      Long squeeze imminent: any catalyst = violent down move

  When OI FALLS while funding is extreme → deleveraging, not squeeze setup.
  Uses EMA of funding change and OI change to reduce noise.

  Degrades to funding-level reversion when OI data is unavailable.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from crypto_bot.core.signals.base import BaseStrategy
from crypto_bot.core.signals.models import Signal
from crypto_bot.core.signals.indicators import atr, ema, rsi


class FundingRateSqueezePredictorStrategy(BaseStrategy):
    """OI + funding rate-of-change squeeze predictor."""

    @property
    def mode(self) -> str:
        return "swing"

    @property
    def param_space(self) -> dict:
        return {
            "funding_threshold":   (0.0003, 0.002),    # |funding| extreme level
            "funding_roc_period":  (3,  10,  "int"),   # funding ROC window
            "funding_roc_min":     (0.0001, 0.001),    # min ROC acceleration
            "oi_roc_period":       (3,  10,  "int"),   # OI ROC window
            "oi_roc_min":          (0.005, 0.03),       # min OI ROC for buildup
            "rsi_period":          (10, 20,  "int"),
            "atr_period":          (10, 20,  "int"),
            "smoothing":           (2,   8,  "int"),    # EMA smoothing for funding/OI
        }

    def _align_series(
        self,
        candles: pd.DataFrame,
        series: pd.Series | None,
        val_col: str,
        tolerance: str = "4h",
    ) -> pd.Series:
        """Align external time series to candle timestamps."""
        if series is None or (hasattr(series, "empty") and series.empty):
            return pd.Series(np.nan, index=range(len(candles)))
        try:
            aligned = series.reindex(
                candles["timestamp"].values,
                method="nearest",
                tolerance=pd.Timedelta(tolerance),
            ).fillna(method="ffill")
            return aligned
        except Exception:
            return pd.Series(np.nan, index=range(len(candles)))

    def generate_signals(
        self, candles: pd.DataFrame, aux_data: Any = None
    ) -> list[Signal]:
        p = self.params
        close  = candles["close"]
        high   = candles["high"]
        low    = candles["low"]
        symbol = str(candles["symbol"].iloc[0])

        fund_thr   = float(p["funding_threshold"])
        fund_roc_p = int(p["funding_roc_period"])
        fund_roc_m = float(p["funding_roc_min"])
        oi_roc_p   = int(p["oi_roc_period"])
        oi_roc_m   = float(p["oi_roc_min"])
        rsi_p      = int(p["rsi_period"])
        atr_p      = int(p["atr_period"])
        smooth     = int(p["smoothing"])

        atr_vals = atr(high, low, close, atr_p)
        rsi_vals = rsi(close, rsi_p)

        # ── Extract funding rate series ────────────────────────────────────
        funding_raw: pd.Series | None = None
        if aux_data and "funding_rate" in aux_data:
            fr_map = aux_data["funding_rate"]
            if isinstance(fr_map, dict):
                fr_df = fr_map.get(symbol)
                if fr_df is not None and not fr_df.empty:
                    try:
                        rate_col = next(
                            c for c in ("fundingRate", "rate", "funding_rate", "value")
                            if c in fr_df.columns
                        )
                        funding_raw = self._align_series(
                            candles, fr_df.set_index("timestamp")[rate_col], rate_col
                        )
                    except Exception:
                        pass

        # ── Extract OI series ─────────────────────────────────────────────
        oi_raw: pd.Series | None = None
        if aux_data and "open_interest" in aux_data:
            oi_map = aux_data["open_interest"]
            if isinstance(oi_map, dict):
                oi_df = oi_map.get(symbol)
                if oi_df is not None and not oi_df.empty:
                    try:
                        oi_col = next(
                            c for c in ("sumOpenInterest", "open_interest", "oi", "value")
                            if c in oi_df.columns
                        )
                        oi_raw = self._align_series(
                            candles, oi_df.set_index("timestamp")[oi_col], oi_col
                        )
                    except Exception:
                        pass

        # ── Build arrays ─────────────────────────────────────────────────
        if funding_raw is not None:
            fund_series = pd.Series(funding_raw.values, dtype=float)
            fund_smooth  = fund_series.ewm(span=smooth, adjust=False).mean()
            fund_roc     = fund_smooth.diff(fund_roc_p)
        else:
            fund_smooth = pd.Series(np.nan, index=range(len(candles)))
            fund_roc    = pd.Series(np.nan, index=range(len(candles)))

        if oi_raw is not None:
            oi_series = pd.Series(oi_raw.values, dtype=float)
            oi_smooth  = oi_series.ewm(span=smooth, adjust=False).mean()
            oi_pct_chg = oi_smooth.pct_change(oi_roc_p)
        else:
            oi_pct_chg = pd.Series(np.nan, index=range(len(candles)))

        close_arr  = close.values
        rsi_arr    = rsi_vals.values
        atr_arr    = atr_vals.values
        fund_arr   = fund_smooth.values
        froc_arr   = fund_roc.values
        oi_chg_arr = oi_pct_chg.values
        ts_arr     = candles["timestamp"].dt.to_pydatetime()
        is_clean   = candles["is_clean"].values

        has_funding = not np.all(np.isnan(fund_arr))
        has_oi      = not np.all(np.isnan(oi_chg_arr))

        warmup   = max(fund_roc_p, oi_roc_p, rsi_p, atr_p, smooth) + 2
        signals: list[Signal] = []
        open_pos: str | None  = None

        for i in range(warmup, len(candles)):
            if not is_clean[i]:
                continue
            c  = close_arr[i]
            rs = rsi_arr[i]
            at = atr_arr[i]

            if np.isnan(at) or np.isnan(rs):
                continue

            ts = ts_arr[i]

            if has_funding:
                fund  = fund_arr[i]
                froc  = froc_arr[i]
                oi_c  = oi_chg_arr[i] if has_oi else np.nan

                if np.isnan(fund) or np.isnan(froc):
                    continue

                oi_rising = (not np.isnan(oi_c) and oi_c > oi_roc_m) if has_oi else True

                # Short squeeze: deeply negative funding + rising OI + accelerating negative
                short_squeeze = (
                    fund < -fund_thr
                    and froc < -fund_roc_m   # funding getting MORE negative
                    and oi_rising
                    and rs < 55              # not already overbought
                )
                # Long squeeze: deeply positive funding + rising OI + accelerating positive
                long_squeeze = (
                    fund > fund_thr
                    and froc > fund_roc_m    # funding getting MORE positive
                    and oi_rising
                    and rs > 45
                )
                # Exit: funding normalises
                exit_long  = open_pos == "LONG"  and fund >= -fund_thr * 0.3
                exit_short = open_pos == "SHORT" and fund <= fund_thr  * 0.3

                if short_squeeze and open_pos != "LONG":
                    open_pos = "LONG"
                    signals.append(Signal(
                        strategy=self.name, symbol=symbol, direction="LONG",
                        timestamp=ts, strength=min(abs(fund) / (fund_thr + 1e-8), 1.0),
                        close_price=c, atr=at,
                        reason=["frsp_short_squeeze", f"fund={fund:.5f}", f"froc={froc:.6f}"],
                    ))
                elif long_squeeze and open_pos != "SHORT":
                    open_pos = "SHORT"
                    signals.append(Signal(
                        strategy=self.name, symbol=symbol, direction="SHORT",
                        timestamp=ts, strength=min(abs(fund) / (fund_thr + 1e-8), 1.0),
                        close_price=c, atr=at,
                        reason=["frsp_long_squeeze", f"fund={fund:.5f}", f"froc={froc:.6f}"],
                    ))
                elif exit_long:
                    open_pos = None
                    signals.append(Signal(
                        strategy=self.name, symbol=symbol, direction="EXIT_LONG",
                        timestamp=ts, strength=0.5,
                        close_price=c, atr=at, reason=["frsp_funding_normalised"],
                    ))
                elif exit_short:
                    open_pos = None
                    signals.append(Signal(
                        strategy=self.name, symbol=symbol, direction="EXIT_SHORT",
                        timestamp=ts, strength=0.5,
                        close_price=c, atr=at, reason=["frsp_funding_normalised"],
                    ))
            else:
                # Pure RSI fallback (no funding data)
                if rs < 30 and open_pos != "LONG":
                    open_pos = "LONG"
                    signals.append(Signal(
                        strategy=self.name, symbol=symbol, direction="LONG",
                        timestamp=ts, strength=(30 - rs) / 30,
                        close_price=c, atr=at, reason=["frsp_rsi_fallback_long"],
                    ))
                elif rs > 70 and open_pos != "SHORT":
                    open_pos = "SHORT"
                    signals.append(Signal(
                        strategy=self.name, symbol=symbol, direction="SHORT",
                        timestamp=ts, strength=(rs - 70) / 30,
                        close_price=c, atr=at, reason=["frsp_rsi_fallback_short"],
                    ))
                elif open_pos == "LONG" and rs > 55:
                    open_pos = None
                    signals.append(Signal(
                        strategy=self.name, symbol=symbol, direction="EXIT_LONG",
                        timestamp=ts, strength=0.5, close_price=c, atr=at,
                        reason=["frsp_rsi_exit"],
                    ))
                elif open_pos == "SHORT" and rs < 45:
                    open_pos = None
                    signals.append(Signal(
                        strategy=self.name, symbol=symbol, direction="EXIT_SHORT",
                        timestamp=ts, strength=0.5, close_price=c, atr=at,
                        reason=["frsp_rsi_exit"],
                    ))

        return signals

    def generate_signals_mtf(
        self, candles_by_tf: dict, aux_data: Any = None
    ) -> list[Signal]:
        primary = candles_by_tf.get("4h")
        if primary is None:
            primary = next(iter(candles_by_tf.values()))
        return self.generate_signals(primary, aux_data)
