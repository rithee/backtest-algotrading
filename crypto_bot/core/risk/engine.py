"""
Risk Engine — stateful per strategy instance.
Responsibilities: position sizing, SL/TP calculation, drawdown guard, correlation guard.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional
from crypto_bot.core.config import Config
from crypto_bot.core.risk.models import RiskDecision
from crypto_bot.core.signals.models import Signal
from crypto_bot.core.execution.models import Position


class RiskEngine:
    def __init__(self, strategy_name: str, config: Config) -> None:
        self.strategy_name = strategy_name
        self.cfg = config
        self.equity: float = config.backtest.initial_capital_per_strategy
        self.peak_equity: float = config.backtest.initial_capital_per_strategy
        self.open_positions: dict[str, Position] = {}  # symbol → Position
        self._price_history: dict[str, list[float]] = {}  # symbol → recent closes for correlation

    # ── public API ────────────────────────────────────────────────────────────

    def evaluate(self, signal: Signal) -> RiskDecision:
        """Evaluate a signal and return a RiskDecision."""
        r = self.cfg.risk

        # Drawdown guard
        if self.peak_equity > 0:
            drawdown = (self.peak_equity - self.equity) / self.peak_equity
            if drawdown > r.max_drawdown_pct:
                return self._reject(signal, f"drawdown={drawdown:.1%} > max={r.max_drawdown_pct:.1%}")

        # Skip exit signals — they are handled by the execution engine directly
        if signal.direction in ("EXIT_LONG", "EXIT_SHORT"):
            return self._approve_exit(signal)

        # No existing position in same symbol
        if signal.symbol in self.open_positions:
            existing = self.open_positions[signal.symbol]
            if existing.direction == signal.direction:
                return self._reject(signal, "duplicate_position")

        # Correlation guard
        corr_reject = self._check_correlation(signal.symbol, signal.direction)
        if corr_reject:
            return self._reject(signal, corr_reject)

        # Size position
        leverage = float(self.cfg.leverage.default)
        leverage = min(leverage, float(self.cfg.leverage.max))
        notional = self.equity * r.max_position_pct
        leveraged = notional * leverage
        quantity = leveraged / signal.close_price

        # SL / TP
        sl_mult = r.stop_loss_atr_multiplier
        tp_mult = r.take_profit_atr_multiplier
        if signal.direction == "LONG":
            sl = signal.close_price - signal.atr * sl_mult
            tp = signal.close_price + signal.atr * tp_mult
        else:
            sl = signal.close_price + signal.atr * sl_mult
            tp = signal.close_price - signal.atr * tp_mult

        return RiskDecision(
            approved=True,
            strategy=self.strategy_name,
            symbol=signal.symbol,
            direction=signal.direction,
            entry_price=signal.close_price,
            quantity=quantity,
            leverage=leverage,
            stop_loss=sl,
            take_profit=tp,
        )

    def update_equity(self, new_equity: float) -> None:
        self.equity = new_equity
        if new_equity > self.peak_equity:
            self.peak_equity = new_equity

    def record_price(self, symbol: str, close: float) -> None:
        """Feed latest close for correlation tracking."""
        if symbol not in self._price_history:
            self._price_history[symbol] = []
        self._price_history[symbol].append(close)
        max_lookback = self.cfg.risk.correlation_lookback_days * 6  # 6 × 4h candles/day
        if len(self._price_history[symbol]) > max_lookback:
            self._price_history[symbol].pop(0)

    # ── private helpers ────────────────────────────────────────────────────────

    def _approve_exit(self, signal: Signal) -> RiskDecision:
        pos = self.open_positions.get(signal.symbol)
        if pos is None:
            return self._reject(signal, "no_open_position_to_exit")
        return RiskDecision(
            approved=True,
            strategy=self.strategy_name,
            symbol=signal.symbol,
            direction=pos.direction,
            entry_price=pos.entry_price,
            quantity=pos.quantity,
            leverage=pos.leverage,
            stop_loss=pos.stop_loss,
            take_profit=pos.take_profit,
        )

    def _reject(self, signal: Signal, reason: str) -> RiskDecision:
        return RiskDecision(
            approved=False,
            strategy=self.strategy_name,
            symbol=signal.symbol,
            direction=signal.direction if signal.direction in ("LONG", "SHORT") else "LONG",
            entry_price=signal.close_price,
            quantity=0.0,
            leverage=1.0,
            stop_loss=0.0,
            take_profit=0.0,
            rejection_reason=reason,
        )

    def _check_correlation(self, new_symbol: str, direction: str) -> Optional[str]:
        r = self.cfg.risk
        if not self.open_positions:
            return None
        new_hist = self._price_history.get(new_symbol, [])
        correlated_count = 0
        for sym, pos in self.open_positions.items():
            hist = self._price_history.get(sym, [])
            n = min(len(new_hist), len(hist), r.correlation_lookback_days * 6)
            if n < 10:
                continue
            corr = float(np.corrcoef(new_hist[-n:], hist[-n:])[0, 1])
            if abs(corr) > r.correlation_threshold:
                correlated_count += 1
        if correlated_count >= r.max_correlated_positions:
            return f"correlation_guard: {correlated_count} correlated positions open"
        return None
