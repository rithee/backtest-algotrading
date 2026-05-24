"""Tests for RiskEngine — position sizing, SL/TP, drawdown guard, correlation guard."""
import pytest
from datetime import datetime
from crypto_bot.core.config import Config
from crypto_bot.core.risk.engine import RiskEngine
from crypto_bot.core.signals.models import Signal
from crypto_bot.core.execution.models import Position


def make_signal(direction="LONG", symbol="BTCUSDT", close=30000.0, atr_val=500.0):
    return Signal(
        strategy="test",
        symbol=symbol,
        timestamp=datetime(2023, 1, 1),
        direction=direction,
        strength=0.8,
        close_price=close,
        atr=atr_val,
        reason=["test"],
    )


def make_position(symbol="BTCUSDT", direction="LONG", entry=30000.0, qty=0.01):
    return Position(
        strategy="test", symbol=symbol, direction=direction,
        entry_price=entry, quantity=qty, leverage=1.0,
        stop_loss=entry - 1000, take_profit=entry + 1500,
        entry_timestamp=datetime(2023, 1, 1), fees_paid=1.0,
    )


class TestPositionSizing:
    def test_position_quantity_correct(self):
        cfg = Config()
        engine = RiskEngine("test", cfg)
        sig = make_signal()
        decision = engine.evaluate(sig)
        assert decision.approved
        expected_qty = (cfg.backtest.initial_capital_per_strategy * cfg.risk.max_position_pct
                        * cfg.leverage.default) / sig.close_price
        assert abs(decision.quantity - expected_qty) < 1e-8

    def test_leverage_capped_at_max(self):
        cfg = Config()
        cfg.leverage.default = 10  # Request more than max
        engine = RiskEngine("test", cfg)
        decision = engine.evaluate(make_signal())
        assert decision.leverage <= cfg.leverage.max


class TestSLTP:
    def test_long_sl_below_entry(self):
        cfg = Config()
        engine = RiskEngine("test", cfg)
        decision = engine.evaluate(make_signal("LONG", close=30000.0, atr_val=500.0))
        assert decision.stop_loss < decision.entry_price

    def test_long_tp_above_entry(self):
        cfg = Config()
        engine = RiskEngine("test", cfg)
        decision = engine.evaluate(make_signal("LONG", close=30000.0, atr_val=500.0))
        assert decision.take_profit > decision.entry_price

    def test_short_sl_above_entry(self):
        cfg = Config()
        engine = RiskEngine("test", cfg)
        decision = engine.evaluate(make_signal("SHORT", close=30000.0, atr_val=500.0))
        assert decision.stop_loss > decision.entry_price

    def test_short_tp_below_entry(self):
        cfg = Config()
        engine = RiskEngine("test", cfg)
        decision = engine.evaluate(make_signal("SHORT", close=30000.0, atr_val=500.0))
        assert decision.take_profit < decision.entry_price

    def test_sl_atr_multiplier_applied(self):
        cfg = Config()
        engine = RiskEngine("test", cfg)
        atr_val = 500.0
        close = 30000.0
        decision = engine.evaluate(make_signal("LONG", close=close, atr_val=atr_val))
        expected_sl = close - atr_val * cfg.risk.stop_loss_atr_multiplier
        assert abs(decision.stop_loss - expected_sl) < 1e-6


class TestDrawdownGuard:
    def test_drawdown_halts_new_entries(self):
        cfg = Config()
        engine = RiskEngine("test", cfg)
        # Simulate 25% drawdown (max is 20%)
        engine.equity = cfg.backtest.initial_capital_per_strategy * 0.75
        decision = engine.evaluate(make_signal())
        assert not decision.approved
        assert "drawdown" in decision.rejection_reason

    def test_no_halt_below_threshold(self):
        cfg = Config()
        engine = RiskEngine("test", cfg)
        engine.equity = cfg.backtest.initial_capital_per_strategy * 0.85  # 15% drawdown
        decision = engine.evaluate(make_signal())
        assert decision.approved


class TestDuplicatePosition:
    def test_rejects_duplicate_symbol_same_direction(self):
        cfg = Config()
        engine = RiskEngine("test", cfg)
        engine.open_positions["BTCUSDT"] = make_position("BTCUSDT", "LONG")
        decision = engine.evaluate(make_signal("LONG", symbol="BTCUSDT"))
        assert not decision.approved
        assert "duplicate" in decision.rejection_reason


class TestExitSignals:
    def test_exit_long_approved(self):
        cfg = Config()
        engine = RiskEngine("test", cfg)
        engine.open_positions["BTCUSDT"] = make_position("BTCUSDT", "LONG")
        exit_sig = make_signal("EXIT_LONG")
        decision = engine.evaluate(exit_sig)
        assert decision.approved

    def test_exit_with_no_open_position_rejected(self):
        cfg = Config()
        engine = RiskEngine("test", cfg)
        exit_sig = make_signal("EXIT_LONG")
        decision = engine.evaluate(exit_sig)
        assert not decision.approved
