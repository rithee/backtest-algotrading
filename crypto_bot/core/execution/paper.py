"""
Paper Execution Engine — simulates fills with fees, slippage, and liquidation checks.

Fill timing:
  - Signal generated at candle[t] close.
  - Entry filled at candle[t+1] open (scheduled as pending order).
  - SL/TP checked at every subsequent candle using high/low.
  - Liquidation checked when candle moves against position beyond 1/leverage of entry.
"""
from __future__ import annotations
from datetime import datetime
from typing import Optional
import numpy as np
from crypto_bot.core.config import Config
from crypto_bot.core.execution.models import Fill, Position
from crypto_bot.core.risk.models import RiskDecision


class PaperExecutionEngine:
    def __init__(self, strategy_name: str, config: Config) -> None:
        self.strategy_name = strategy_name
        self.cfg = config
        self.open_positions: dict[str, Position] = {}  # symbol → Position
        self.fills: list[Fill] = []
        self._pending: dict[str, RiskDecision] = {}  # symbol → pending entry order

    # ── public API ────────────────────────────────────────────────────────────

    def schedule_entry(self, decision: RiskDecision) -> None:
        """Schedule an entry to fill at the next candle's open."""
        self._pending[decision.symbol] = decision

    def process_candle(
        self,
        symbol: str,
        timestamp: datetime,
        open_: float,
        high: float,
        low: float,
        close: float,
    ) -> list[Fill]:
        """
        Called once per candle for each symbol. Returns any fills generated this candle.
        Order of operations:
          1. Execute pending entry (fill at this candle's open).
          2. Check open position for SL / TP / liquidation.
        """
        new_fills: list[Fill] = []

        # 1. Fill pending entry at this candle's open
        if symbol in self._pending:
            fill = self._fill_entry(symbol, timestamp, open_)
            if fill:
                new_fills.append(fill)
            del self._pending[symbol]

        # 2. Check existing position for exits
        if symbol in self.open_positions:
            exit_fill = self._check_exits(symbol, timestamp, high, low, close)
            if exit_fill:
                new_fills.append(exit_fill)
                del self.open_positions[symbol]

        return new_fills

    def close_at_end(self, symbol: str, timestamp: datetime, close: float) -> Optional[Fill]:
        """Force-close open position at end of backtest data."""
        if symbol not in self.open_positions:
            return None
        pos = self.open_positions.pop(symbol)
        fill = self._create_exit_fill(pos, timestamp, close, "end_of_data")
        self.fills.append(fill)
        return fill

    def schedule_exit(self, symbol: str) -> None:
        """Signal-based exit — close position at next candle open."""
        if symbol in self.open_positions:
            pos = self.open_positions[symbol]
            # Mark for exit by creating a special pending flag
            self._pending[f"__exit_{symbol}"] = pos  # type: ignore

    # ── private helpers ────────────────────────────────────────────────────────

    def _fill_entry(self, symbol: str, timestamp: datetime, open_: float) -> Optional[Fill]:
        # Handle signal-based exit scheduled via schedule_exit
        exit_key = f"__exit_{symbol}"
        if exit_key in self._pending:
            pos = self._pending.pop(exit_key)  # type: ignore
            if isinstance(pos, Position) and symbol in self.open_positions:
                del self.open_positions[symbol]
                fill = self._create_exit_fill(pos, timestamp, open_, "signal")
                self.fills.append(fill)
                return fill
            return None

        decision = self._pending.get(symbol)
        if decision is None or not decision.approved:
            return None
        if symbol in self.open_positions:
            return None  # Already have a position

        cfg = self.cfg.backtest
        slippage = cfg.slippage_pct
        # Apply slippage to entry
        if decision.direction == "LONG":
            fill_price = open_ * (1.0 + slippage)
        else:
            fill_price = open_ * (1.0 - slippage)

        entry_fees = fill_price * decision.quantity * cfg.fees_pct

        pos = Position(
            strategy=self.strategy_name,
            symbol=symbol,
            direction=decision.direction,
            entry_price=fill_price,
            quantity=decision.quantity,
            leverage=decision.leverage,
            stop_loss=decision.stop_loss,
            take_profit=decision.take_profit,
            entry_timestamp=timestamp,
            fees_paid=entry_fees,
        )
        self.open_positions[symbol] = pos
        return None  # Entry fill is recorded when position is closed

    def _check_exits(
        self,
        symbol: str,
        timestamp: datetime,
        high: float,
        low: float,
        close: float,
    ) -> Optional[Fill]:
        pos = self.open_positions[symbol]

        # Liquidation check (must be before SL/TP)
        liq_price = self._liquidation_price(pos)
        if pos.direction == "LONG" and low <= liq_price:
            fill = self._create_exit_fill(pos, timestamp, liq_price, "liquidation")
            self.fills.append(fill)
            return fill
        elif pos.direction == "SHORT" and high >= liq_price:
            fill = self._create_exit_fill(pos, timestamp, liq_price, "liquidation")
            self.fills.append(fill)
            return fill

        # SL / TP check — assume worst case: if both hit in same candle, SL first
        if pos.direction == "LONG":
            if low <= pos.stop_loss:
                fill = self._create_exit_fill(pos, timestamp, pos.stop_loss, "sl")
                self.fills.append(fill)
                return fill
            if high >= pos.take_profit:
                fill = self._create_exit_fill(pos, timestamp, pos.take_profit, "tp")
                self.fills.append(fill)
                return fill
        else:  # SHORT
            if high >= pos.stop_loss:
                fill = self._create_exit_fill(pos, timestamp, pos.stop_loss, "sl")
                self.fills.append(fill)
                return fill
            if low <= pos.take_profit:
                fill = self._create_exit_fill(pos, timestamp, pos.take_profit, "tp")
                self.fills.append(fill)
                return fill

        return None

    def _create_exit_fill(
        self,
        pos: Position,
        timestamp: datetime,
        exit_price: float,
        reason: str,
    ) -> Fill:
        cfg = self.cfg.backtest
        slippage = cfg.slippage_pct
        # Apply slippage to exit (opposite direction to entry)
        if pos.direction == "LONG":
            exit_with_slip = exit_price * (1.0 - slippage)
        else:
            exit_with_slip = exit_price * (1.0 + slippage)

        exit_fees = exit_with_slip * pos.quantity * cfg.fees_pct
        total_fees = pos.fees_paid + exit_fees
        total_slippage = (abs(pos.entry_price - (pos.entry_price / (1.0 + slippage if pos.direction == "LONG" else 1.0 - slippage)))
                          + abs(exit_price - exit_with_slip)) * pos.quantity

        if pos.direction == "LONG":
            gross_pnl = (exit_with_slip - pos.entry_price) * pos.quantity * pos.leverage
        else:
            gross_pnl = (pos.entry_price - exit_with_slip) * pos.quantity * pos.leverage

        net_pnl = gross_pnl - total_fees

        return Fill(
            strategy=self.strategy_name,
            symbol=pos.symbol,
            timestamp=timestamp,
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=exit_with_slip,
            quantity=pos.quantity,
            leverage=pos.leverage,
            fees_paid=total_fees,
            slippage_paid=total_slippage,
            pnl=net_pnl,
            exit_reason=reason,  # type: ignore
        )

    @staticmethod
    def _liquidation_price(pos: Position) -> float:
        """Liquidation when position moves 1/leverage against entry."""
        if pos.leverage <= 1.0:
            return 0.0 if pos.direction == "LONG" else float("inf")
        margin_pct = 1.0 / pos.leverage
        if pos.direction == "LONG":
            return pos.entry_price * (1.0 - margin_pct)
        else:
            return pos.entry_price * (1.0 + margin_pct)
