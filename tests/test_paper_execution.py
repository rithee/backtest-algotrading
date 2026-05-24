"""Tests for PaperExecutionEngine — fills, fees, slippage, SL/TP, liquidation."""
import pytest
from datetime import datetime
from crypto_bot.core.config import Config
from crypto_bot.core.execution.paper import PaperExecutionEngine
from crypto_bot.core.risk.models import RiskDecision


def make_decision(symbol="BTCUSDT", direction="LONG", entry=30000.0, qty=0.01,
                  sl=29000.0, tp=31500.0, leverage=1.0):
    return RiskDecision(
        approved=True, strategy="test", symbol=symbol,
        direction=direction, entry_price=entry, quantity=qty,
        leverage=leverage, stop_loss=sl, take_profit=tp,
    )


def ts(n=0):
    return datetime(2023, 1, 1 + n)


class TestEntryAndExit:
    def test_fill_at_next_candle_open(self):
        cfg = Config()
        eng = PaperExecutionEngine("test", cfg)
        eng.schedule_entry(make_decision())
        # No fill yet on signal candle
        fills = eng.process_candle("BTCUSDT", ts(0), 30000, 30200, 29800, 30100)
        assert "BTCUSDT" in eng.open_positions
        assert len(fills) == 0  # entry doesn't produce a Fill until exit

    def test_tp_hit_produces_fill(self):
        cfg = Config()
        eng = PaperExecutionEngine("test", cfg)
        eng.schedule_entry(make_decision(entry=30000, tp=31500, sl=29000))
        eng.process_candle("BTCUSDT", ts(0), 30000, 30100, 29900, 30050)  # entry
        fills = eng.process_candle("BTCUSDT", ts(1), 30100, 32000, 30000, 31000)  # TP hit
        assert len(fills) == 1
        assert fills[0].exit_reason == "tp"

    def test_sl_hit_produces_fill(self):
        cfg = Config()
        eng = PaperExecutionEngine("test", cfg)
        eng.schedule_entry(make_decision(entry=30000, tp=31500, sl=29000))
        eng.process_candle("BTCUSDT", ts(0), 30000, 30100, 29900, 30050)
        fills = eng.process_candle("BTCUSDT", ts(1), 30000, 30100, 28500, 29000)  # SL hit
        assert len(fills) == 1
        assert fills[0].exit_reason == "sl"

    def test_tp_pnl_positive(self):
        cfg = Config()
        eng = PaperExecutionEngine("test", cfg)
        eng.schedule_entry(make_decision(entry=30000, tp=31500, sl=29000, qty=0.1))
        eng.process_candle("BTCUSDT", ts(0), 30000, 30100, 29900, 30050)
        fills = eng.process_candle("BTCUSDT", ts(1), 30100, 32000, 30000, 31000)
        assert fills[0].pnl > 0

    def test_sl_pnl_negative(self):
        cfg = Config()
        eng = PaperExecutionEngine("test", cfg)
        eng.schedule_entry(make_decision(entry=30000, tp=31500, sl=29000, qty=0.1))
        eng.process_candle("BTCUSDT", ts(0), 30000, 30100, 29900, 30050)
        fills = eng.process_candle("BTCUSDT", ts(1), 30000, 30100, 28500, 29000)
        assert fills[0].pnl < 0


class TestLiquidation:
    def test_liquidation_at_margin_call_price(self):
        cfg = Config()
        eng = PaperExecutionEngine("test", cfg)
        # 3x leverage → liquidation at entry * (1 - 1/3) = entry * 0.667
        eng.schedule_entry(make_decision(entry=30000, sl=20000, tp=40000, leverage=3.0))
        eng.process_candle("BTCUSDT", ts(0), 30000, 30100, 29900, 30050)
        liq_price = 30000 * (1 - 1/3)
        fills = eng.process_candle("BTCUSDT", ts(1), 30000, 30100, liq_price - 100, 29500)
        assert len(fills) == 1
        assert fills[0].exit_reason == "liquidation"


class TestFees:
    def test_fees_deducted_from_pnl(self):
        cfg = Config()
        cfg.backtest.fees_pct = 0.001
        cfg.backtest.slippage_pct = 0.0
        eng = PaperExecutionEngine("test", cfg)
        eng.schedule_entry(make_decision(entry=30000, tp=31500, sl=29000, qty=0.1))
        eng.process_candle("BTCUSDT", ts(0), 30000, 30100, 29900, 30050)
        fills = eng.process_candle("BTCUSDT", ts(1), 30100, 32000, 30000, 31000)
        assert fills[0].fees_paid > 0
        assert fills[0].pnl < (31500 - 30000) * 0.1  # net pnl < gross pnl


class TestCloseAtEnd:
    def test_close_at_end_produces_fill(self):
        cfg = Config()
        eng = PaperExecutionEngine("test", cfg)
        eng.schedule_entry(make_decision())
        eng.process_candle("BTCUSDT", ts(0), 30000, 30100, 29900, 30050)
        fill = eng.close_at_end("BTCUSDT", ts(5), 30500.0)
        assert fill is not None
        assert fill.exit_reason == "end_of_data"
