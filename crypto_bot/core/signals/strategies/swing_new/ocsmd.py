"""
OnChainSmartMoneyDivergence — Swing futures strategy.

Logic (exchange netflow divergence):
  Smart money reveals itself through exchange flows:
    - Price makes new highs + exchange INFLOWS rising  → distribution → SHORT
      (large players are depositing coins to sell on exchanges)
    - Price at lows    + exchange OUTFLOWS rising      → accumulation → LONG
      (large players are withdrawing coins from exchanges to hold)

  Exchange balance from Glassnode (exchange_balance metric):
    Rising exchange balance   = net inflow  (selling pressure building)
    Falling exchange balance  = net outflow (accumulation / buying pressure)

  Degrades to pure price-momentum when no Glassnode data available.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from crypto_bot.core.signals.base import BaseStrategy
from crypto_bot.core.signals.models import Signal
from crypto_bot.core.signals.indicators import atr, ema, rsi


class OnChainSmartMoneyDivergenceStrategy(BaseStrategy):
    """Exchange-flow divergence from price — smart money tracker."""

    @property
    def mode(self) -> str:
        return "swing"

    @property
    def param_space(self) -> dict:
        return {
            "price_lookback":    (10,  30,  "int"),   # bars for price high/low
            "flow_smoothing":    (3,   14,  "int"),   # EMA period for flow smoothing
            "flow_change_pct":   (0.005, 0.03),       # min % flow change to signal
            "rsi_period":        (10,  20,  "int"),
            "rsi_oversold":      (25.0, 40.0),        # for long confirmation
            "rsi_overbought":    (60.0, 75.0),        # for short confirmation
            "trend_ema":         (50,  200, "int"),
            "atr_period":        (10,  20,  "int"),
        }

    def _get_flow_series(
        self, candles: pd.DataFrame, aux_data: Any, symbol: str
    ) -> pd.Series | None:
        """Extract and align exchange_netflow (or balance) to candle timestamps."""
        if aux_data is None:
            return None
        for key in ("exchange_netflow", "onchain"):
            if key not in aux_data:
                continue
            data = aux_data[key]
            if isinstance(data, dict):
                # exchange_netflow: {symbol: df}
                df = data.get(symbol)
                if df is None:
                    df = data.get(symbol.replace("USDT", ""))
                if df is None or (hasattr(df, "empty") and df.empty):
                    continue
                try:
                    val_col = next(
                        c for c in ("value", "v", "exchange_balance", "netflow")
                        if c in df.columns
                    )
                    flow = df.set_index("timestamp")[val_col]
                    return flow.reindex(
                        candles["timestamp"].values,
                        method="nearest",
                        tolerance=pd.Timedelta("12h"),
                    ).interpolate(limit=3)
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

        price_lb    = int(p["price_lookback"])
        flow_smooth = int(p["flow_smoothing"])
        flow_chg    = float(p["flow_change_pct"])
        rsi_p       = int(p["rsi_period"])
        rsi_os      = float(p["rsi_oversold"])
        rsi_ob      = float(p["rsi_overbought"])
        trend_p     = int(p["trend_ema"])
        atr_p       = int(p["atr_period"])

        atr_vals  = atr(high, low, close, atr_p)
        rsi_vals  = rsi(close, rsi_p)
        trend     = ema(close, trend_p)
        price_max = close.rolling(price_lb).max()
        price_min = close.rolling(price_lb).min()

        # Flow data
        raw_flow = self._get_flow_series(candles, aux_data, symbol)
        has_flow = raw_flow is not None and not raw_flow.isna().all()
        if has_flow:
            flow_ema = raw_flow.ewm(span=flow_smooth, adjust=False).mean()
            flow_arr = flow_ema.values
        else:
            flow_arr = np.full(len(candles), np.nan)

        close_arr  = close.values
        rsi_arr    = rsi_vals.values
        trend_arr  = trend.values
        atr_arr    = atr_vals.values
        pmax_arr   = price_max.values
        pmin_arr   = price_min.values
        ts_arr     = candles["timestamp"].dt.to_pydatetime()
        is_clean   = candles["is_clean"].values

        warmup   = max(price_lb, flow_smooth, rsi_p, trend_p, atr_p) + 2
        signals: list[Signal] = []
        open_pos: str | None  = None

        for i in range(warmup, len(candles)):
            if not is_clean[i]:
                continue
            c  = close_arr[i]
            rs = rsi_arr[i]
            tr = trend_arr[i]
            at = atr_arr[i]
            pm = pmax_arr[i]
            pn = pmin_arr[i]

            if np.isnan(rs) or np.isnan(tr) or np.isnan(at):
                continue

            near_high = c >= pm * 0.98
            near_low  = c <= pn * 1.02

            if has_flow and not np.isnan(flow_arr[i]) and i > 0 and not np.isnan(flow_arr[i - 1]):
                flow_prev = flow_arr[i - 1]
                flow_now  = flow_arr[i]
                flow_rising  = (flow_now - flow_prev) / max(abs(flow_prev), 1e-8) > flow_chg
                flow_falling = (flow_prev - flow_now) / max(abs(flow_prev), 1e-8) > flow_chg

                # Divergence: price high + inflows rising → distribution (SHORT)
                if near_high and flow_rising and rs > rsi_ob and open_pos != "SHORT":
                    open_pos = "SHORT"
                    signals.append(Signal(
                        strategy=self.name, symbol=symbol, direction="SHORT",
                        timestamp=ts_arr[i], strength=min(rs / 100, 1.0),
                        close_price=c, atr=at,
                        reason=["ocsmd_distribution", f"rsi={rs:.0f}", "inflow_rising"],
                    ))
                # Divergence: price low + outflows rising → accumulation (LONG)
                elif near_low and flow_falling and rs < rsi_os and open_pos != "LONG":
                    open_pos = "LONG"
                    signals.append(Signal(
                        strategy=self.name, symbol=symbol, direction="LONG",
                        timestamp=ts_arr[i], strength=min((100 - rs) / 100, 1.0),
                        close_price=c, atr=at,
                        reason=["ocsmd_accumulation", f"rsi={rs:.0f}", "outflow_rising"],
                    ))
            else:
                # Fallback: pure price-momentum divergence (price at extreme + RSI divergence)
                price_range = pm - pn
                if price_range <= 0:
                    continue

                if near_high and rs > rsi_ob and c < tr and open_pos != "SHORT":
                    open_pos = "SHORT"
                    signals.append(Signal(
                        strategy=self.name, symbol=symbol, direction="SHORT",
                        timestamp=ts_arr[i], strength=0.5,
                        close_price=c, atr=at,
                        reason=["ocsmd_price_overbought_fallback", f"rsi={rs:.0f}"],
                    ))
                elif near_low and rs < rsi_os and c > tr and open_pos != "LONG":
                    open_pos = "LONG"
                    signals.append(Signal(
                        strategy=self.name, symbol=symbol, direction="LONG",
                        timestamp=ts_arr[i], strength=0.5,
                        close_price=c, atr=at,
                        reason=["ocsmd_price_oversold_fallback", f"rsi={rs:.0f}"],
                    ))

            # Exit: mean reversion toward trend
            if open_pos == "LONG" and c > tr * 1.02:
                open_pos = None
                signals.append(Signal(
                    strategy=self.name, symbol=symbol, direction="EXIT_LONG",
                    timestamp=ts_arr[i], strength=0.5,
                    close_price=c, atr=at, reason=["ocsmd_target_reached"],
                ))
            elif open_pos == "SHORT" and c < tr * 0.98:
                open_pos = None
                signals.append(Signal(
                    strategy=self.name, symbol=symbol, direction="EXIT_SHORT",
                    timestamp=ts_arr[i], strength=0.5,
                    close_price=c, atr=at, reason=["ocsmd_target_reached"],
                ))

        return signals

    def generate_signals_mtf(
        self, candles_by_tf: dict, aux_data: Any = None
    ) -> list[Signal]:
        primary = candles_by_tf.get("4h")
        if primary is None:
            primary = next(iter(candles_by_tf.values()))
        return self.generate_signals(primary, aux_data)
