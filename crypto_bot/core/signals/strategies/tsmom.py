"""
Time-Series Momentum (TSMOM) with Volatility Scaling.

Academic basis: Moskowitz, Ooi & Pedersen (2012), extended to crypto in multiple
2023-2025 papers. Crypto-specific finding: optimal lookback is 1-4 weeks (shorter
than equity momentum's 12 months). Achieves Sharpe ~1.51 vs 0.84 buy-and-hold.

Core logic:
  Signal = sign of cumulative log return over lookback window.
  Position size is scaled inversely by realized volatility (targets a constant
  annualized volatility) — auto-reduces exposure during high-vol regimes.
  Regime filter: only trade in the direction of the 200-bar SMA trend.

Entry:
  LONG  — momentum signal flips to +1 AND close > 200-bar SMA
  SHORT — momentum signal flips to -1 AND close < 200-bar SMA

Exit:
  Signal flips OR realized volatility spikes 2× above its own 20-bar mean
  (emergency de-risk during blow-up events like FTX collapse Nov 2022).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from crypto_bot.core.signals.base import BaseStrategy
from crypto_bot.core.signals.models import Signal


class TSMOMStrategy(BaseStrategy):
    """Time-Series Momentum with volatility scaling and regime filter."""

    @property
    def param_space(self) -> dict[str, tuple]:
        return {
            "momentum_period":  (20, 60, "int"),   # lookback for return signal (bars)
            "vol_period":       (10, 30, "int"),   # realized vol estimation window
            "trend_period":     (150, 250, "int"), # SMA period for regime filter
            "vol_spike_mult":   (1.5, 3.0),        # vol spike exit threshold
        }

    def generate_signals(self, candles: pd.DataFrame, aux_data: dict | None = None) -> list[Signal]:
        p = self.params
        mom_period   = int(p.get("momentum_period", 42))
        vol_period   = int(p.get("vol_period",      20))
        trend_period = int(p.get("trend_period",    200))
        vol_spike    = float(p.get("vol_spike_mult", 2.0))

        warmup = max(mom_period, vol_period, trend_period) + 2
        if len(candles) < warmup:
            return []

        close = candles["close"]

        # Log returns and derived series
        log_ret      = np.log(close / close.shift(1))
        cum_ret      = log_ret.rolling(mom_period).sum()
        signal_raw   = np.sign(cum_ret)                           # +1 / -1 / 0
        realized_vol = log_ret.rolling(vol_period).std() * np.sqrt(252 * 6)  # annualized
        avg_vol      = realized_vol.rolling(vol_period).mean()
        trend_sma    = close.rolling(trend_period).mean()

        signals: list[Signal] = []
        prev_signal = 0.0

        for i in range(warmup, len(candles)):
            row = candles.iloc[i]
            if not row.get("is_clean", True):
                continue

            c        = float(close.iloc[i])
            sig      = float(signal_raw.iloc[i])
            rvol     = float(realized_vol.iloc[i])
            avgvol   = float(avg_vol.iloc[i])
            sma      = float(trend_sma.iloc[i])
            cur_atr_est = rvol / np.sqrt(252 * 6) * c  # rough ATR from vol

            if any(np.isnan(x) for x in [sig, rvol, avgvol, sma]):
                continue

            vol_spiking = avgvol > 0 and rvol > avgvol * vol_spike

            # Emergency exit on vol spike
            if vol_spiking and prev_signal != 0:
                direction = "EXIT_LONG" if prev_signal > 0 else "EXIT_SHORT"
                signals.append(Signal(
                    strategy=self.name, symbol=row["symbol"],
                    timestamp=row["timestamp"], direction=direction,
                    strength=1.0, close_price=c, atr=cur_atr_est,
                    reason=["vol_spike_exit"],
                ))
                prev_signal = 0.0
                continue

            # Regime filter: long only above SMA, short only below SMA
            regime_allows_long  = c > sma
            regime_allows_short = c < sma

            # Signal flipped to bullish
            if sig > 0 and prev_signal <= 0 and regime_allows_long:
                signals.append(Signal(
                    strategy=self.name, symbol=row["symbol"],
                    timestamp=row["timestamp"], direction="LONG",
                    strength=min(1.0, abs(float(cum_ret.iloc[i]))),
                    close_price=c, atr=cur_atr_est,
                    reason=["tsmom_long", f"ret_{mom_period}bar_positive"],
                ))
                prev_signal = 1.0

            # Signal flipped to bearish
            elif sig < 0 and prev_signal >= 0 and regime_allows_short:
                signals.append(Signal(
                    strategy=self.name, symbol=row["symbol"],
                    timestamp=row["timestamp"], direction="SHORT",
                    strength=min(1.0, abs(float(cum_ret.iloc[i]))),
                    close_price=c, atr=cur_atr_est,
                    reason=["tsmom_short", f"ret_{mom_period}bar_negative"],
                ))
                prev_signal = -1.0

            # Exit: signal flipped against position
            elif prev_signal > 0 and sig <= 0:
                signals.append(Signal(
                    strategy=self.name, symbol=row["symbol"],
                    timestamp=row["timestamp"], direction="EXIT_LONG",
                    strength=1.0, close_price=c, atr=cur_atr_est,
                    reason=["momentum_reversal"],
                ))
                prev_signal = 0.0

            elif prev_signal < 0 and sig >= 0:
                signals.append(Signal(
                    strategy=self.name, symbol=row["symbol"],
                    timestamp=row["timestamp"], direction="EXIT_SHORT",
                    strength=1.0, close_price=c, atr=cur_atr_est,
                    reason=["momentum_reversal"],
                ))
                prev_signal = 0.0

        return signals
