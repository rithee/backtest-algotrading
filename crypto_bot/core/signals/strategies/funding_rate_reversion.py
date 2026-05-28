"""
Funding Rate Mean Reversion Strategy.
Extreme positive funding (longs overpaying) → short (crowded long unwind).
Extreme negative funding (shorts overpaying) → long (crowded short unwind).
RSI used as directional confirmation.
Exit when funding normalises (falls below exit_funding_pct × threshold) or RSI reverses.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from crypto_bot.core.signals.base import BaseStrategy
from crypto_bot.core.signals.models import Signal
from crypto_bot.core.signals.indicators import rsi, atr


class FundingRateReversionStrategy(BaseStrategy):
    @property
    def param_space(self) -> dict:
        return {
            "funding_threshold":   (0.0005, 0.003),
            "rsi_period":          (8, 21, "int"),
            "rsi_confirm_long":    (20.0, 45.0),
            "rsi_confirm_short":   (55.0, 80.0),
            "exit_funding_pct":    (0.1, 0.5),
            "atr_period":          (10, 20, "int"),
        }

    def generate_signals(self, candles: pd.DataFrame, aux_data=None) -> list[Signal]:
        p = self.params
        close  = candles["close"]
        high   = candles["high"]
        low    = candles["low"]
        symbol = str(candles["symbol"].iloc[0])

        rsi_v = rsi(close, int(p["rsi_period"]))
        atr_v = atr(high, low, close, int(p.get("atr_period", 14)))

        # Get funding rates aligned to candle timestamps
        funding_series = self._get_funding_series(candles, aux_data)

        signals: list[Signal] = []
        open_pos: str | None = None
        warmup = int(p["rsi_period"]) + 1
        threshold = float(p["funding_threshold"])
        exit_pct   = float(p["exit_funding_pct"])

        # pre-convert to numpy — eliminates pandas .iloc overhead in hot loop
        is_clean_arr   = candles["is_clean"].values
        timestamps_arr = candles["timestamp"].dt.to_pydatetime()
        close_arr      = close.values
        rsi_arr        = rsi_v.values
        atr_arr        = atr_v.values
        funding_arr    = funding_series.values

        for i in range(warmup, len(candles)):
            if not is_clean_arr[i]:
                continue
            rsi_i   = rsi_arr[i]
            atr_i   = atr_arr[i]
            funding = float(funding_arr[i])
            if np.isnan(rsi_i) or np.isnan(atr_i):
                continue

            direction: str | None = None
            reason: list[str] = []
            confirming = 0

            if funding < -threshold and rsi_i < float(p["rsi_confirm_long"]) and open_pos != "LONG":
                direction = "LONG"
                confirming = sum([funding < -threshold, rsi_i < float(p["rsi_confirm_long"])])
                reason = [f"funding={funding:.5f}<-{threshold}", f"RSI={rsi_i:.1f}"]
                open_pos = "LONG"
            elif funding > threshold and rsi_i > float(p["rsi_confirm_short"]) and open_pos != "SHORT":
                direction = "SHORT"
                confirming = 2
                reason = [f"funding={funding:.5f}>{threshold}", f"RSI={rsi_i:.1f}"]
                open_pos = "SHORT"
            elif open_pos == "LONG" and abs(funding) < threshold * exit_pct:
                direction = "EXIT_LONG"
                reason = [f"funding_normalized={funding:.5f}"]
                open_pos = None
            elif open_pos == "SHORT" and abs(funding) < threshold * exit_pct:
                direction = "EXIT_SHORT"
                reason = [f"funding_normalized={funding:.5f}"]
                open_pos = None

            if direction:
                signals.append(Signal(
                    strategy=self.name,
                    symbol=symbol,
                    timestamp=timestamps_arr[i],
                    direction=direction,
                    strength=confirming / 2.0 if confirming else 0.5,
                    close_price=float(close_arr[i]),
                    atr=float(atr_i),
                    reason=reason,
                ))

        return signals

    def _get_funding_series(self, candles: pd.DataFrame, aux_data) -> pd.Series:
        """
        Return a funding rate series aligned to candle timestamps.
        If aux_data has 'funding_rates' DataFrame, merge by nearest timestamp.
        Otherwise return zeros (no funding data available).
        """
        if aux_data is None or "funding_rates" not in aux_data:
            return pd.Series(0.0, index=candles.index)

        fr_df = aux_data["funding_rates"]
        symbol = str(candles["symbol"].iloc[0])
        fr_sym = fr_df[fr_df["symbol"] == symbol].copy()
        if fr_sym.empty:
            return pd.Series(0.0, index=candles.index)

        fr_sym = fr_sym.sort_values("timestamp").reset_index(drop=True)
        # merge_asof: for each candle timestamp, find the most recent funding rate
        merged = pd.merge_asof(
            candles[["timestamp"]].copy(),
            fr_sym[["timestamp", "rate"]].rename(columns={"timestamp": "fr_ts"}),
            left_on="timestamp",
            right_on="fr_ts",
            direction="backward",
        )
        return merged["rate"].fillna(0.0).reset_index(drop=True)
