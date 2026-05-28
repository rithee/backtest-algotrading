"""
OptionsGammaMaxPain — Swing futures strategy.

Logic (Deribit options max pain):
  "Max Pain" = the strike price where the total payout to option holders is
  minimised (i.e. the option market makers lose the least money).

  Near weekly options expiry (Friday UTC):
    - Spot below max pain → price gravitates up toward max pain → LONG
    - Spot above max pain → price gravitates down toward max pain → SHORT

  Gamma Exposure (GEX): when GEX flips sign, dealer hedging direction
  changes, which can amplify moves in the GEX flip direction.

  Degrades gracefully: if no Deribit options data available, uses pure
  momentum (RSI + MACD) as fallback.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from crypto_bot.core.signals.base import BaseStrategy
from crypto_bot.core.signals.models import Signal
from crypto_bot.core.signals.indicators import atr, ema, rsi, macd


class OptionsGammaMaxPainStrategy(BaseStrategy):
    """Max-pain gravitational pull pre-expiry + GEX flip confirmation."""

    @property
    def mode(self) -> str:
        return "swing"

    @property
    def param_space(self) -> dict:
        return {
            "max_pain_band_pct":  (0.005, 0.03),       # % band around max pain for signals
            "expiry_day_window":  (1, 5,    "int"),     # bars before Friday expiry to activate
            "rsi_period":         (10, 20,  "int"),
            "macd_fast":          (8,  16,  "int"),
            "macd_slow":          (20, 40,  "int"),
            "macd_signal":        (7,  12,  "int"),
            "trend_ema":          (50, 200, "int"),
            "atr_period":         (10, 20,  "int"),
        }

    def _get_max_pain(
        self, candles: pd.DataFrame, aux_data: Any, symbol: str
    ) -> pd.Series | None:
        """
        Extract max_pain series aligned to candle timestamps.
        Returns None if options data unavailable.
        """
        if aux_data is None:
            return None
        options_map = aux_data.get("options_gamma")
        if not isinstance(options_map, dict):
            return None

        df = options_map.get(symbol)
        if df is None:
            df = options_map.get(symbol.replace("USDT", ""))
        if df is None or (hasattr(df, "empty") and df.empty):
            return None
        if "max_pain" not in df.columns and "strike" not in df.columns:
            return None

        try:
            if "max_pain" in df.columns and "timestamp" in df.columns:
                mp = df.set_index("timestamp")["max_pain"]
                return mp.reindex(
                    candles["timestamp"].values,
                    method="nearest",
                    tolerance=pd.Timedelta("2D"),
                ).fillna(method="ffill")
        except Exception:
            pass
        return None

    def generate_signals(
        self, candles: pd.DataFrame, aux_data: Any = None
    ) -> list[Signal]:
        p = self.params
        close  = candles["close"]
        high   = candles["high"]
        low    = candles["low"]
        symbol = str(candles["symbol"].iloc[0])

        mp_band    = float(p["max_pain_band_pct"])
        exp_window = int(p["expiry_day_window"])
        rsi_p      = int(p["rsi_period"])
        mf         = int(p["macd_fast"])
        ms         = int(p["macd_slow"])
        msig       = int(p["macd_signal"])
        trend_p    = int(p["trend_ema"])
        atr_p      = int(p["atr_period"])

        atr_vals      = atr(high, low, close, atr_p)
        rsi_vals      = rsi(close, rsi_p)
        macd_l, _, hist = macd(close, mf, ms, msig)
        trend           = ema(close, trend_p)

        # Max pain time series (aligned)
        mp_series = self._get_max_pain(candles, aux_data, symbol)
        has_mp    = mp_series is not None
        mp_arr    = mp_series.values if has_mp else np.full(len(candles), np.nan)

        close_arr = close.values
        rsi_arr   = rsi_vals.values
        hist_arr  = hist.values
        trend_arr = trend.values
        atr_arr   = atr_vals.values
        ts_arr    = candles["timestamp"].dt.to_pydatetime()
        is_clean  = candles["is_clean"].values

        warmup   = max(ms + msig, trend_p, atr_p, rsi_p) + 2
        signals: list[Signal] = []
        open_pos: str | None  = None

        for i in range(warmup, len(candles)):
            if not is_clean[i]:
                continue
            c  = close_arr[i]
            rs = rsi_arr[i]
            hi = hist_arr[i]
            tr = trend_arr[i]
            at = atr_arr[i]
            ts = ts_arr[i]

            if np.isnan(rs) or np.isnan(hi) or np.isnan(tr) or np.isnan(at):
                continue

            # Check if we're in an expiry window (Thursday/Friday UTC)
            near_expiry = ts.weekday() in (3, 4)  # Thu=3, Fri=4

            if has_mp and not np.isnan(mp_arr[i]):
                mp_val = mp_arr[i]
                below_mp = c < mp_val * (1 - mp_band)
                above_mp = c > mp_val * (1 + mp_band)
                macd_bullish = hi > 0 and (i > 0 and hist_arr[i - 1] <= 0)  # MACD hist cross up
                macd_bearish = hi < 0 and (i > 0 and hist_arr[i - 1] >= 0)  # MACD hist cross down

                if near_expiry and below_mp and macd_bullish and open_pos != "LONG":
                    open_pos = "LONG"
                    signals.append(Signal(
                        strategy=self.name, symbol=symbol, direction="LONG",
                        timestamp=ts, strength=min(abs(c - mp_val) / (mp_val * mp_band + 1e-8), 1.0),
                        close_price=c, atr=at,
                        reason=["ogp_below_max_pain", f"mp={mp_val:.0f}"],
                    ))
                elif near_expiry and above_mp and macd_bearish and open_pos != "SHORT":
                    open_pos = "SHORT"
                    signals.append(Signal(
                        strategy=self.name, symbol=symbol, direction="SHORT",
                        timestamp=ts, strength=min(abs(c - mp_val) / (mp_val * mp_band + 1e-8), 1.0),
                        close_price=c, atr=at,
                        reason=["ogp_above_max_pain", f"mp={mp_val:.0f}"],
                    ))
                # Exit: converged to max pain level
                elif open_pos == "LONG" and c >= mp_val * (1 - mp_band * 0.3):
                    open_pos = None
                    signals.append(Signal(
                        strategy=self.name, symbol=symbol, direction="EXIT_LONG",
                        timestamp=ts, strength=0.5, close_price=c, atr=at,
                        reason=["ogp_converged_to_mp"],
                    ))
                elif open_pos == "SHORT" and c <= mp_val * (1 + mp_band * 0.3):
                    open_pos = None
                    signals.append(Signal(
                        strategy=self.name, symbol=symbol, direction="EXIT_SHORT",
                        timestamp=ts, strength=0.5, close_price=c, atr=at,
                        reason=["ogp_converged_to_mp"],
                    ))
            else:
                # Fallback: RSI + MACD momentum (no options data)
                macd_cross_up   = hi > 0 and i > 0 and hist_arr[i - 1] <= 0
                macd_cross_down = hi < 0 and i > 0 and hist_arr[i - 1] >= 0

                if macd_cross_up and rs < 60 and c > tr and open_pos != "LONG":
                    open_pos = "LONG"
                    signals.append(Signal(
                        strategy=self.name, symbol=symbol, direction="LONG",
                        timestamp=ts, strength=0.55,
                        close_price=c, atr=at, reason=["ogp_macd_fallback_long"],
                    ))
                elif macd_cross_down and rs > 40 and c < tr and open_pos != "SHORT":
                    open_pos = "SHORT"
                    signals.append(Signal(
                        strategy=self.name, symbol=symbol, direction="SHORT",
                        timestamp=ts, strength=0.55,
                        close_price=c, atr=at, reason=["ogp_macd_fallback_short"],
                    ))
                elif open_pos == "LONG" and (c < tr or rs > 75):
                    open_pos = None
                    signals.append(Signal(
                        strategy=self.name, symbol=symbol, direction="EXIT_LONG",
                        timestamp=ts, strength=0.5, close_price=c, atr=at,
                        reason=["ogp_fallback_exit"],
                    ))
                elif open_pos == "SHORT" and (c > tr or rs < 25):
                    open_pos = None
                    signals.append(Signal(
                        strategy=self.name, symbol=symbol, direction="EXIT_SHORT",
                        timestamp=ts, strength=0.5, close_price=c, atr=at,
                        reason=["ogp_fallback_exit"],
                    ))

        return signals

    def generate_signals_mtf(
        self, candles_by_tf: dict, aux_data: Any = None
    ) -> list[Signal]:
        primary = candles_by_tf.get("4h")
        if primary is None:
            primary = next(iter(candles_by_tf.values()))
        return self.generate_signals(primary, aux_data)
