"""
CrossAssetMomentumRegime — Swing futures strategy.

Logic:
  BTC's relationship to traditional assets varies by macro cycle.
  Three regimes based on rolling correlation (BTC vs SPX over ~30 bars):

    Regime A — Risk-On Correlated  (corr > +threshold):
      BTC follows SPX momentum.  Use SPX 1m-return for signal direction.

    Regime B — Decoupled           (|corr| < threshold):
      BTC moves on its own. Use BTC's own momentum + funding rate.

    Regime C — Inverse/Flight to safety (corr < -threshold):
      BTC moves against DXY (strong dollar = crypto down).
      Use DXY momentum inversed for BTC direction.

  Requires aux_data["macro"] DataFrame with columns: spx, dxy, gold (from yfinance).
  Degrades gracefully to pure BTC momentum when macro data is unavailable.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from crypto_bot.core.signals.base import BaseStrategy
from crypto_bot.core.signals.models import Signal
from crypto_bot.core.signals.indicators import atr, ema, rsi


class CrossAssetMomentumRegimeStrategy(BaseStrategy):
    """Cross-asset correlation regime switcher for BTC momentum direction."""

    @property
    def mode(self) -> str:
        return "swing"

    @property
    def param_space(self) -> dict:
        return {
            "corr_window":       (15,  40,  "int"),   # rolling correlation window (bars)
            "corr_threshold":    (0.3,  0.7),          # |corr| above this = regime A or C
            "mom_period":        (5,   20,  "int"),   # momentum lookback
            "rsi_period":        (10,  20,  "int"),
            "trend_ema":         (50,  150, "int"),
            "atr_period":        (10,  20,  "int"),
        }

    def _get_macro_series(
        self, candles: pd.DataFrame, aux_data: Any
    ) -> dict[str, pd.Series | None]:
        """Extract macro return series aligned to candle timestamps."""
        result: dict[str, pd.Series | None] = {"spx": None, "dxy": None, "gold": None}
        if aux_data is None:
            return result
        macro_df = aux_data.get("macro")
        if macro_df is None or (hasattr(macro_df, "empty") and macro_df.empty):
            return result
        try:
            # macro_df has a DatetimeIndex with daily frequency
            candle_dates = pd.to_datetime(candles["timestamp"].values).normalize()
            for col in ("spx", "dxy", "gold"):
                if col not in macro_df.columns:
                    continue
                series = macro_df[col]
                aligned = series.reindex(candle_dates, method="ffill")
                aligned.index = candles["timestamp"].values
                result[col] = aligned
        except Exception:
            pass
        return result

    def generate_signals(
        self, candles: pd.DataFrame, aux_data: Any = None
    ) -> list[Signal]:
        p = self.params
        close  = candles["close"]
        high   = candles["high"]
        low    = candles["low"]
        symbol = str(candles["symbol"].iloc[0])

        corr_win  = int(p["corr_window"])
        corr_thr  = float(p["corr_threshold"])
        mom_p     = int(p["mom_period"])
        rsi_p     = int(p["rsi_period"])
        trend_p   = int(p["trend_ema"])
        atr_p     = int(p["atr_period"])

        atr_vals  = atr(high, low, close, atr_p)
        rsi_vals  = rsi(close, rsi_p)
        trend     = ema(close, trend_p)
        btc_ret   = close.pct_change()

        macro     = self._get_macro_series(candles, aux_data)
        spx_ret   = macro["spx"].pct_change() if macro["spx"] is not None else None
        dxy_ret   = macro["dxy"].pct_change() if macro["dxy"] is not None else None

        # Rolling correlations with SPX and DXY
        if spx_ret is not None:
            corr_spx = btc_ret.rolling(corr_win).corr(spx_ret)
        else:
            corr_spx = pd.Series(np.nan, index=close.index)

        if dxy_ret is not None:
            corr_dxy = btc_ret.rolling(corr_win).corr(dxy_ret)
        else:
            corr_dxy = pd.Series(np.nan, index=close.index)

        # BTC own momentum
        btc_mom = close.pct_change(mom_p)
        spx_mom = spx_ret.rolling(mom_p).sum() if spx_ret is not None else None
        dxy_mom = dxy_ret.rolling(mom_p).sum() if dxy_ret is not None else None

        close_arr   = close.values
        rsi_arr     = rsi_vals.values
        trend_arr   = trend.values
        atr_arr     = atr_vals.values
        btc_mom_arr = btc_mom.values
        corr_spx_a  = corr_spx.values
        corr_dxy_a  = corr_dxy.values
        spx_mom_a   = spx_mom.values if spx_mom is not None else np.full(len(candles), np.nan)
        dxy_mom_a   = dxy_mom.values if dxy_mom is not None else np.full(len(candles), np.nan)
        ts_arr      = candles["timestamp"].dt.to_pydatetime()
        is_clean    = candles["is_clean"].values

        warmup   = max(corr_win, mom_p, trend_p, rsi_p, atr_p) + 2
        signals: list[Signal] = []
        open_pos: str | None  = None

        for i in range(warmup, len(candles)):
            if not is_clean[i]:
                continue
            c   = close_arr[i]
            rs  = rsi_arr[i]
            tr  = trend_arr[i]
            at  = atr_arr[i]
            bmo = btc_mom_arr[i]
            cspx = corr_spx_a[i]
            cdxy = corr_dxy_a[i]
            smo  = spx_mom_a[i]
            dmo  = dxy_mom_a[i]

            if np.isnan(at) or np.isnan(rs) or np.isnan(tr) or np.isnan(bmo):
                continue

            # Determine regime and composite signal
            signal_dir: str | None = None
            reason: list[str] = []

            if not np.isnan(cspx) and abs(cspx) >= corr_thr:
                if cspx > 0 and not np.isnan(smo):
                    # Regime A: correlated with SPX — follow SPX momentum
                    if smo > 0 and c > tr and rs < 65:
                        signal_dir = "LONG"
                        reason = [f"cam_regimeA_spx_bull", f"corr={cspx:.2f}"]
                    elif smo < 0 and c < tr and rs > 35:
                        signal_dir = "SHORT"
                        reason = [f"cam_regimeA_spx_bear", f"corr={cspx:.2f}"]
                elif cspx < 0 and not np.isnan(dmo):
                    # Regime C: inverse correlated — use DXY inversely
                    if dmo < 0 and c > tr and rs < 65:  # DXY falling → BTC up
                        signal_dir = "LONG"
                        reason = [f"cam_regimeC_dxy_fall", f"corr={cspx:.2f}"]
                    elif dmo > 0 and c < tr and rs > 35:  # DXY rising → BTC down
                        signal_dir = "SHORT"
                        reason = [f"cam_regimeC_dxy_rise", f"corr={cspx:.2f}"]
            else:
                # Regime B: decoupled — pure BTC momentum
                if bmo > 0.01 and c > tr and rs < 65 and rs > 40:
                    signal_dir = "LONG"
                    reason = [f"cam_regimeB_btc_mom", f"mom={bmo:.3f}"]
                elif bmo < -0.01 and c < tr and rs > 35 and rs < 60:
                    signal_dir = "SHORT"
                    reason = [f"cam_regimeB_btc_mom", f"mom={bmo:.3f}"]

            if signal_dir == "LONG" and open_pos != "LONG":
                open_pos = "LONG"
                signals.append(Signal(
                    strategy=self.name, symbol=symbol, direction="LONG",
                    timestamp=ts_arr[i], strength=min(abs(bmo) * 30, 1.0),
                    close_price=c, atr=at, reason=reason,
                ))
            elif signal_dir == "SHORT" and open_pos != "SHORT":
                open_pos = "SHORT"
                signals.append(Signal(
                    strategy=self.name, symbol=symbol, direction="SHORT",
                    timestamp=ts_arr[i], strength=min(abs(bmo) * 30, 1.0),
                    close_price=c, atr=at, reason=reason,
                ))
            elif open_pos == "LONG" and (bmo < -0.005 or c < tr):
                open_pos = None
                signals.append(Signal(
                    strategy=self.name, symbol=symbol, direction="EXIT_LONG",
                    timestamp=ts_arr[i], strength=0.5,
                    close_price=c, atr=at, reason=["cam_regime_shift_exit"],
                ))
            elif open_pos == "SHORT" and (bmo > 0.005 or c > tr):
                open_pos = None
                signals.append(Signal(
                    strategy=self.name, symbol=symbol, direction="EXIT_SHORT",
                    timestamp=ts_arr[i], strength=0.5,
                    close_price=c, atr=at, reason=["cam_regime_shift_exit"],
                ))

        return signals

    def generate_signals_mtf(
        self, candles_by_tf: dict, aux_data: Any = None
    ) -> list[Signal]:
        primary = candles_by_tf.get("4h")
        if primary is None:
            primary = next(iter(candles_by_tf.values()))
        return self.generate_signals(primary, aux_data)
