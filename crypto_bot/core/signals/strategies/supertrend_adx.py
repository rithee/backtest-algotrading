"""
Supertrend + ADX Filter Strategy.
Entry: Supertrend direction confirmed by ADX > threshold (strong trend).
Exit: Supertrend flips direction.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from crypto_bot.core.signals.base import BaseStrategy
from crypto_bot.core.signals.models import Signal
from crypto_bot.core.signals.indicators import supertrend, adx, atr


class SupertrendADXStrategy(BaseStrategy):
    @property
    def param_space(self) -> dict:
        return {
            "atr_period":    (7,  21, "int"),
            "atr_multiplier": (1.5, 4.0),
            "adx_period":    (10, 20, "int"),
            "adx_threshold": (20.0, 35.0),
        }

    def generate_signals(self, candles: pd.DataFrame, aux_data=None) -> list[Signal]:
        p = self.params
        close  = candles["close"]
        high   = candles["high"]
        low    = candles["low"]
        symbol = str(candles["symbol"].iloc[0])

        st_line, st_dir = supertrend(
            high, low, close,
            period=int(p["atr_period"]),
            multiplier=float(p["atr_multiplier"]),
        )
        adx_v = adx(high, low, close, int(p["adx_period"]))
        atr_v = atr(high, low, close, int(p["atr_period"]))

        signals: list[Signal] = []
        open_pos: str | None = None
        adx_thresh = float(p["adx_threshold"])
        warmup = 1  # start from i=1; NaN guard handles uninitialized values

        # pre-convert to numpy — eliminates pandas .iloc overhead in hot loop
        is_clean_arr   = candles["is_clean"].values
        timestamps_arr = candles["timestamp"].dt.to_pydatetime()
        close_arr      = close.values
        st_dir_arr     = st_dir.values
        adx_arr        = adx_v.values
        atr_arr        = atr_v.values

        for i in range(warmup, len(candles)):
            if not is_clean_arr[i]:
                continue
            dir_i    = int(st_dir_arr[i])
            dir_prev = int(st_dir_arr[i - 1])
            adx_i    = adx_arr[i]
            atr_i    = atr_arr[i]
            if np.isnan(adx_i) or np.isnan(atr_i) or dir_i == 0:
                continue

            flipped_up   = dir_i == 1  and dir_prev <= 0
            flipped_down = dir_i == -1 and dir_prev >= 0

            direction: str | None = None
            reason: list[str] = []

            if flipped_up and adx_i > adx_thresh and open_pos != "LONG":
                direction = "LONG"
                reason = ["ST_flip_up", f"ADX={adx_i:.1f}>{adx_thresh}"]
                open_pos = "LONG"
            elif flipped_down and adx_i > adx_thresh and open_pos != "SHORT":
                direction = "SHORT"
                reason = ["ST_flip_down", f"ADX={adx_i:.1f}>{adx_thresh}"]
                open_pos = "SHORT"
            elif open_pos == "LONG" and dir_i == -1:
                direction = "EXIT_LONG"
                reason = ["ST_flip_down"]
                open_pos = None
            elif open_pos == "SHORT" and dir_i == 1:
                direction = "EXIT_SHORT"
                reason = ["ST_flip_up"]
                open_pos = None

            if direction:
                signals.append(Signal(
                    strategy=self.name,
                    symbol=symbol,
                    timestamp=timestamps_arr[i],
                    direction=direction,
                    strength=min(adx_i / 50.0, 1.0),
                    close_price=float(close_arr[i]),
                    atr=float(atr_i),
                    reason=reason,
                ))

        return signals
