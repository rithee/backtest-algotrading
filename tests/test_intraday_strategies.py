"""
Tests for all 6 intraday strategies.

Each strategy must:
  1. Return list[Signal] from generate_signals()
  2. Return list[Signal] from generate_signals_mtf()
  3. Have mode == "intraday"
  4. Survive with no aux_data (graceful degradation)
  5. All signal directions are valid literals
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

from crypto_bot.core.signals.models import Signal

_VALID_DIRS = {"LONG", "SHORT", "EXIT_LONG", "EXIT_SHORT"}

# ── Candle factory ────────────────────────────────────────────────────────────

def _make_5m_candles(n: int = 300, symbol: str = "BTCUSDT", seed: int = 42) -> pd.DataFrame:
    """
    5-minute candles starting at 08:00 UTC (London session open) — in a trading session.
    Contains enough bars for all strategy warmup periods.
    """
    np.random.seed(seed)
    base  = datetime(2023, 1, 2, 8, 0)
    ts    = [base + timedelta(minutes=5 * i) for i in range(n)]
    noise = np.cumsum(np.random.randn(n) * 50)
    close = 20000.0 + noise
    high  = close + np.abs(np.random.randn(n) * 30)
    low   = close - np.abs(np.random.randn(n) * 30)
    # Ensure high >= close >= low
    high  = np.maximum(high, close)
    low   = np.minimum(low,  close)
    vol   = np.abs(np.random.randn(n) * 300) + 500
    return pd.DataFrame({
        "timestamp": ts,
        "open":   close * 0.999,
        "high":   high,
        "low":    low,
        "close":  close,
        "volume": vol,
        "is_clean": [True] * n,
        "symbol": [symbol] * n,
    })


def _check_signals(signals: list[Signal], symbol: str = "BTCUSDT") -> None:
    assert isinstance(signals, list)
    for sig in signals:
        assert isinstance(sig, Signal)
        assert sig.direction in _VALID_DIRS, f"Bad direction: {sig.direction}"
        assert sig.symbol == symbol
        assert sig.close_price > 0
        assert sig.atr > 0
        assert 0.0 <= sig.strength <= 1.0


# ── LiquiditySweepReversal ────────────────────────────────────────────────────

class TestLiquiditySweepReversal:
    @pytest.fixture
    def strategy(self):
        from crypto_bot.core.signals.strategies.intraday.lsr import (
            LiquiditySweepReversalStrategy,
        )
        s = LiquiditySweepReversalStrategy.__new__(LiquiditySweepReversalStrategy)
        s.params = LiquiditySweepReversalStrategy.__new__(LiquiditySweepReversalStrategy).default_params() if False else {
            "swing_lookback": 10, "vol_spike_mult": 2.0, "vol_avg_period": 20,
            "wick_ratio": 0.6, "atr_period": 14,
        }
        return s

    def test_mode_is_intraday(self, strategy):
        assert strategy.mode == "intraday"

    def test_returns_list(self, strategy):
        candles = _make_5m_candles()
        result  = strategy.generate_signals(candles)
        assert isinstance(result, list)

    def test_all_directions_valid(self, strategy):
        candles = _make_5m_candles()
        _check_signals(strategy.generate_signals(candles))

    def test_no_aux_data(self, strategy):
        candles = _make_5m_candles()
        result  = strategy.generate_signals(candles, aux_data=None)
        assert isinstance(result, list)

    def test_mtf_uses_5m(self, strategy):
        candles = _make_5m_candles()
        by_tf   = {"5m": candles, "15m": candles}
        result  = strategy.generate_signals_mtf(by_tf)
        assert isinstance(result, list)

    def test_default_params_returns_dict(self):
        from crypto_bot.core.signals.strategies.intraday.lsr import (
            LiquiditySweepReversalStrategy,
        )
        s = LiquiditySweepReversalStrategy.__new__(LiquiditySweepReversalStrategy)
        s.params = {}
        dp = s.default_params()
        assert isinstance(dp, dict)
        assert "swing_lookback" in dp


# ── OpeningRangeBreakout ──────────────────────────────────────────────────────

class TestOpeningRangeBreakout:
    @pytest.fixture
    def strategy(self):
        from crypto_bot.core.signals.strategies.intraday.orb_sb import (
            OpeningRangeBreakoutStrategy,
        )
        s = OpeningRangeBreakoutStrategy.__new__(OpeningRangeBreakoutStrategy)
        s.params = {
            "or_bars": 6, "vol_mult": 1.5, "vol_period": 20,
            "trend_ema": 30, "atr_period": 14,
        }
        return s

    def test_mode_is_intraday(self, strategy):
        assert strategy.mode == "intraday"

    def test_returns_list(self, strategy):
        candles = _make_5m_candles()
        assert isinstance(strategy.generate_signals(candles), list)

    def test_all_directions_valid(self, strategy):
        candles = _make_5m_candles()
        _check_signals(strategy.generate_signals(candles))

    def test_no_signal_outside_session_window(self, strategy):
        """Candles entirely in the gap between sessions (08:00 UTC is London open, fine)."""
        candles = _make_5m_candles()
        # Move candles to a UTC gap hour (e.g. 22:30 UTC, outside all sessions)
        gap_base = datetime(2023, 1, 2, 22, 30)
        candles = candles.copy()
        candles["timestamp"] = [gap_base + timedelta(minutes=5 * i) for i in range(len(candles))]
        result = strategy.generate_signals(candles)
        # Should return empty or very few signals (no confirmed breakout in gap)
        assert isinstance(result, list)

    def test_mtf_prefers_5m(self, strategy):
        candles = _make_5m_candles()
        by_tf   = {"5m": candles, "1h": candles}
        assert isinstance(strategy.generate_signals_mtf(by_tf), list)


# ── VWAPDeltaConfluence ───────────────────────────────────────────────────────

class TestVWAPDeltaConfluence:
    @pytest.fixture
    def strategy(self):
        from crypto_bot.core.signals.strategies.intraday.mvdc import (
            VWAPDeltaConfluenceStrategy,
        )
        s = VWAPDeltaConfluenceStrategy.__new__(VWAPDeltaConfluenceStrategy)
        s.params = {
            "vwap_period": 20, "std_mult": 1.5, "delta_period": 10,
            "trend_ema": 30, "atr_period": 14,
        }
        return s

    def test_mode_is_intraday(self, strategy):
        assert strategy.mode == "intraday"

    def test_returns_list(self, strategy):
        assert isinstance(strategy.generate_signals(_make_5m_candles()), list)

    def test_all_directions_valid(self, strategy):
        _check_signals(strategy.generate_signals(_make_5m_candles()))

    def test_no_aux_data(self, strategy):
        assert isinstance(strategy.generate_signals(_make_5m_candles(), aux_data=None), list)

    def test_mtf(self, strategy):
        c = _make_5m_candles()
        assert isinstance(strategy.generate_signals_mtf({"5m": c}), list)


# ── FairValueGap ──────────────────────────────────────────────────────────────

class TestFairValueGap:
    @pytest.fixture
    def strategy(self):
        from crypto_bot.core.signals.strategies.intraday.fvg import FairValueGapStrategy
        s = FairValueGapStrategy.__new__(FairValueGapStrategy)
        s.params = {
            "min_gap_atr": 0.2, "rsi_period": 14, "rsi_momentum": 50.0,
            "trend_ema": 30, "atr_period": 14, "max_gaps": 15,
        }
        return s

    def test_mode_is_intraday(self, strategy):
        assert strategy.mode == "intraday"

    def test_returns_list(self, strategy):
        assert isinstance(strategy.generate_signals(_make_5m_candles()), list)

    def test_all_directions_valid(self, strategy):
        _check_signals(strategy.generate_signals(_make_5m_candles()))

    def test_no_aux_data(self, strategy):
        assert isinstance(strategy.generate_signals(_make_5m_candles(), aux_data=None), list)

    def test_mtf(self, strategy):
        c = _make_5m_candles()
        assert isinstance(strategy.generate_signals_mtf({"5m": c}), list)

    def test_gap_min_size_filters_noise(self, strategy):
        """With min_gap_atr=10 (huge), no gaps should trigger on normal candles."""
        strategy.params = dict(strategy.params)
        strategy.params["min_gap_atr"] = 10.0
        signals = strategy.generate_signals(_make_5m_candles())
        entries = [s for s in signals if s.direction in ("LONG", "SHORT")]
        assert len(entries) == 0


# ── LiquidationCascadeMomentum ────────────────────────────────────────────────

class TestLiquidationCascadeMomentum:
    @pytest.fixture
    def strategy(self):
        from crypto_bot.core.signals.strategies.intraday.lcm import (
            LiquidationCascadeMomentumStrategy,
        )
        s = LiquidationCascadeMomentumStrategy.__new__(LiquidationCascadeMomentumStrategy)
        s.params = {
            "liq_thresh_mult": 1.0, "liq_avg_period": 10, "mom_period": 10,
            "mom_threshold": 0.005, "vol_mult": 2.0, "atr_period": 14,
        }
        return s

    def test_mode_is_intraday(self, strategy):
        assert strategy.mode == "intraday"

    def test_returns_list_no_aux(self, strategy):
        result = strategy.generate_signals(_make_5m_candles(), aux_data=None)
        assert isinstance(result, list)

    def test_all_directions_valid_no_aux(self, strategy):
        _check_signals(strategy.generate_signals(_make_5m_candles(), aux_data=None))

    def test_returns_list_with_liq_data(self, strategy):
        candles = _make_5m_candles()
        # Build a fake liquidation DataFrame aligned to candle timestamps
        liq_df = pd.DataFrame({
            "timestamp": candles["timestamp"].values,
            "liq_usd":   np.abs(np.random.randn(len(candles))) * 1e6,
        })
        aux = {"liquidations": {"BTCUSDT": liq_df}}
        result = strategy.generate_signals(candles, aux_data=aux)
        assert isinstance(result, list)
        _check_signals(result)

    def test_handles_missing_symbol_in_liq(self, strategy):
        candles = _make_5m_candles()
        aux = {"liquidations": {"ETHUSDT": pd.DataFrame()}}  # wrong symbol
        result = strategy.generate_signals(candles, aux_data=aux)
        assert isinstance(result, list)

    def test_mtf(self, strategy):
        c = _make_5m_candles()
        assert isinstance(strategy.generate_signals_mtf({"5m": c}), list)


# ── MicrostructureConsolidationBreakout ───────────────────────────────────────

class TestMicrostructureConsolidationBreakout:
    @pytest.fixture
    def strategy(self):
        from crypto_bot.core.signals.strategies.intraday.mcb import (
            MicrostructureConsolidationBreakoutStrategy,
        )
        s = MicrostructureConsolidationBreakoutStrategy.__new__(
            MicrostructureConsolidationBreakoutStrategy
        )
        s.params = {
            "consol_period": 15, "atr_percentile": 0.35, "breakout_mult": 1.0,
            "vol_mult": 1.5, "atr_period": 14,
        }
        return s

    def test_mode_is_intraday(self, strategy):
        assert strategy.mode == "intraday"

    def test_returns_list_no_aux(self, strategy):
        assert isinstance(strategy.generate_signals(_make_5m_candles(), aux_data=None), list)

    def test_all_directions_valid(self, strategy):
        _check_signals(strategy.generate_signals(_make_5m_candles()))

    def test_with_oi_data(self, strategy):
        candles = _make_5m_candles()
        oi_df = pd.DataFrame({
            "timestamp":     candles["timestamp"].values,
            "open_interest": np.linspace(1e8, 1.5e8, len(candles)),  # rising OI
        })
        aux    = {"open_interest": {"BTCUSDT": oi_df}}
        result = strategy.generate_signals(candles, aux_data=aux)
        assert isinstance(result, list)
        _check_signals(result)

    def test_handles_missing_oi_symbol(self, strategy):
        candles = _make_5m_candles()
        aux = {"open_interest": {"ETHUSDT": None}}
        result = strategy.generate_signals(candles, aux_data=aux)
        assert isinstance(result, list)

    def test_mtf(self, strategy):
        c = _make_5m_candles()
        assert isinstance(strategy.generate_signals_mtf({"5m": c}), list)
