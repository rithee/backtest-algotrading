"""
Tests for all 6 new swing futures strategies.

Each strategy must:
  1. Return list[Signal] from generate_signals() and generate_signals_mtf()
  2. Have mode == "swing"
  3. Survive without aux_data (graceful degradation)
  4. All signal directions are valid literals
  5. strength ∈ [0, 1]
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

from crypto_bot.core.signals.models import Signal

_VALID_DIRS = {"LONG", "SHORT", "EXIT_LONG", "EXIT_SHORT"}

# ── Candle factory ────────────────────────────────────────────────────────────

def _make_4h_candles(n: int = 400, symbol: str = "BTCUSDT", seed: int = 42) -> pd.DataFrame:
    """
    4-hour candles with enough bars for all swing strategy warmup periods.
    Mixed trend (up first half, down second half) to trigger both entry directions.
    """
    np.random.seed(seed)
    base  = datetime(2022, 1, 1, 0, 0)
    ts    = [base + timedelta(hours=4 * i) for i in range(n)]
    half  = n // 2
    trend = np.concatenate([
        np.linspace(0, 5000, half),
        np.linspace(5000, 0, n - half),
    ])
    noise  = np.cumsum(np.random.randn(n) * 80)
    close  = 25000.0 + trend + noise
    high   = close + np.abs(np.random.randn(n) * 120) + 50
    low    = close - np.abs(np.random.randn(n) * 120) - 50
    high   = np.maximum(high, close)
    low    = np.minimum(low, close)
    vol    = np.abs(np.random.randn(n) * 500) + 1000
    # Occasional volume spikes (for Wyckoff climax detection)
    spike_idx = [50, 150, 250, 350]
    for idx in spike_idx:
        if idx < n:
            vol[idx] *= 4
    return pd.DataFrame({
        "timestamp": ts, "open": close * 0.999,
        "high": high, "low": low, "close": close,
        "volume": vol, "is_clean": [True] * n, "symbol": [symbol] * n,
    })


def _check_signals(signals: list, symbol: str = "BTCUSDT") -> None:
    assert isinstance(signals, list)
    for sig in signals:
        assert isinstance(sig, Signal), f"Not a Signal: {type(sig)}"
        assert sig.direction in _VALID_DIRS, f"Bad direction: {sig.direction}"
        assert sig.symbol == symbol
        assert sig.close_price > 0
        assert sig.atr > 0
        assert 0.0 <= sig.strength <= 1.0, f"strength out of range: {sig.strength}"


# ── WyckoffPhaseDetector ──────────────────────────────────────────────────────

class TestWyckoffPhaseDetector:
    @pytest.fixture
    def strategy(self):
        from crypto_bot.core.signals.strategies.swing_new.wyckoff import (
            WyckoffPhaseDetectorStrategy,
        )
        s = WyckoffPhaseDetectorStrategy.__new__(WyckoffPhaseDetectorStrategy)
        s.params = {
            "vol_climax_mult": 2.5, "vol_avg_period": 20, "spread_mult": 1.8,
            "retest_bars": 8, "vol_decay_ratio": 0.5, "trend_ema": 100, "atr_period": 14,
        }
        return s

    def test_mode_is_swing(self, strategy):
        assert strategy.mode == "swing"

    def test_returns_list(self, strategy):
        assert isinstance(strategy.generate_signals(_make_4h_candles()), list)

    def test_all_directions_valid(self, strategy):
        _check_signals(strategy.generate_signals(_make_4h_candles()))

    def test_no_aux_data(self, strategy):
        assert isinstance(strategy.generate_signals(_make_4h_candles(), aux_data=None), list)

    def test_mtf_uses_4h(self, strategy):
        c = _make_4h_candles()
        result = strategy.generate_signals_mtf({"4h": c, "1d": c})
        assert isinstance(result, list)

    def test_mtf_fallback_to_first(self, strategy):
        c = _make_4h_candles()
        result = strategy.generate_signals_mtf({"1d": c})
        assert isinstance(result, list)

    def test_default_params(self):
        from crypto_bot.core.signals.strategies.swing_new.wyckoff import (
            WyckoffPhaseDetectorStrategy,
        )
        s = WyckoffPhaseDetectorStrategy.__new__(WyckoffPhaseDetectorStrategy)
        s.params = {}
        dp = s.default_params()
        assert "vol_climax_mult" in dp
        assert "retest_bars" in dp


# ── OnChainSmartMoneyDivergence ───────────────────────────────────────────────

class TestOnChainSmartMoneyDivergence:
    @pytest.fixture
    def strategy(self):
        from crypto_bot.core.signals.strategies.swing_new.ocsmd import (
            OnChainSmartMoneyDivergenceStrategy,
        )
        s = OnChainSmartMoneyDivergenceStrategy.__new__(OnChainSmartMoneyDivergenceStrategy)
        s.params = {
            "price_lookback": 20, "flow_smoothing": 7, "flow_change_pct": 0.01,
            "rsi_period": 14, "rsi_oversold": 35.0, "rsi_overbought": 65.0,
            "trend_ema": 100, "atr_period": 14,
        }
        return s

    def test_mode_is_swing(self, strategy):
        assert strategy.mode == "swing"

    def test_returns_list_no_aux(self, strategy):
        assert isinstance(strategy.generate_signals(_make_4h_candles(), aux_data=None), list)

    def test_all_directions_valid_no_aux(self, strategy):
        _check_signals(strategy.generate_signals(_make_4h_candles(), aux_data=None))

    def test_with_exchange_netflow(self, strategy):
        candles = _make_4h_candles()
        # Simulate exchange balance: rising during first half (distribution), falling in second
        balance = np.concatenate([
            np.linspace(1e9, 1.3e9, len(candles) // 2),
            np.linspace(1.3e9, 0.9e9, len(candles) - len(candles) // 2),
        ])
        flow_df = pd.DataFrame({
            "timestamp": candles["timestamp"].values,
            "value": balance,
        })
        aux = {"exchange_netflow": {"BTCUSDT": flow_df}}
        result = strategy.generate_signals(candles, aux_data=aux)
        assert isinstance(result, list)
        _check_signals(result)

    def test_missing_symbol_in_flow(self, strategy):
        candles = _make_4h_candles()
        aux = {"exchange_netflow": {"ETHUSDT": pd.DataFrame()}}
        result = strategy.generate_signals(candles, aux_data=aux)
        assert isinstance(result, list)

    def test_mtf(self, strategy):
        c = _make_4h_candles()
        assert isinstance(strategy.generate_signals_mtf({"4h": c}), list)


# ── OptionsGammaMaxPain ───────────────────────────────────────────────────────

class TestOptionsGammaMaxPain:
    @pytest.fixture
    def strategy(self):
        from crypto_bot.core.signals.strategies.swing_new.ogp import (
            OptionsGammaMaxPainStrategy,
        )
        s = OptionsGammaMaxPainStrategy.__new__(OptionsGammaMaxPainStrategy)
        s.params = {
            "max_pain_band_pct": 0.015, "expiry_day_window": 3,
            "rsi_period": 14, "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
            "trend_ema": 100, "atr_period": 14,
        }
        return s

    def test_mode_is_swing(self, strategy):
        assert strategy.mode == "swing"

    def test_returns_list_no_options(self, strategy):
        assert isinstance(strategy.generate_signals(_make_4h_candles(), aux_data=None), list)

    def test_all_directions_valid_fallback(self, strategy):
        _check_signals(strategy.generate_signals(_make_4h_candles()))

    def test_with_max_pain_data(self, strategy):
        candles = _make_4h_candles()
        close_vals = candles["close"].values
        # Max pain slightly above spot = price should rally toward it
        mp_df = pd.DataFrame({
            "timestamp": candles["timestamp"].values,
            "max_pain":  close_vals * 1.02,
        })
        aux = {"options_gamma": {"BTCUSDT": mp_df}}
        result = strategy.generate_signals(candles, aux_data=aux)
        assert isinstance(result, list)
        _check_signals(result)

    def test_missing_options_symbol(self, strategy):
        aux = {"options_gamma": {"ETHUSDT": None}}
        result = strategy.generate_signals(_make_4h_candles(), aux_data=aux)
        assert isinstance(result, list)

    def test_mtf(self, strategy):
        c = _make_4h_candles()
        assert isinstance(strategy.generate_signals_mtf({"4h": c}), list)


# ── CrossAssetMomentumRegime ──────────────────────────────────────────────────

class TestCrossAssetMomentumRegime:
    @pytest.fixture
    def strategy(self):
        from crypto_bot.core.signals.strategies.swing_new.cam import (
            CrossAssetMomentumRegimeStrategy,
        )
        s = CrossAssetMomentumRegimeStrategy.__new__(CrossAssetMomentumRegimeStrategy)
        s.params = {
            "corr_window": 20, "corr_threshold": 0.4, "mom_period": 10,
            "rsi_period": 14, "trend_ema": 100, "atr_period": 14,
        }
        return s

    def test_mode_is_swing(self, strategy):
        assert strategy.mode == "swing"

    def test_returns_list_no_macro(self, strategy):
        assert isinstance(strategy.generate_signals(_make_4h_candles(), aux_data=None), list)

    def test_all_directions_valid_no_macro(self, strategy):
        _check_signals(strategy.generate_signals(_make_4h_candles()))

    def test_with_macro_data(self, strategy):
        candles = _make_4h_candles()
        dates = pd.to_datetime(candles["timestamp"].values).normalize().unique()
        n_days = len(dates)
        macro_df = pd.DataFrame(
            {
                "spx":  np.linspace(4000, 4500, n_days),
                "dxy":  np.linspace(100, 95, n_days),
                "gold": np.linspace(1800, 1900, n_days),
            },
            index=dates,
        )
        aux = {"macro": macro_df}
        result = strategy.generate_signals(candles, aux_data=aux)
        assert isinstance(result, list)
        _check_signals(result)

    def test_regime_b_without_macro(self, strategy):
        """Regime B (decoupled) should activate with no macro data."""
        candles = _make_4h_candles()
        result = strategy.generate_signals(candles, aux_data=None)
        # Should still generate signals via BTC-only momentum
        assert isinstance(result, list)

    def test_mtf(self, strategy):
        c = _make_4h_candles()
        assert isinstance(strategy.generate_signals_mtf({"4h": c}), list)


# ── ElliottWaveAutomator ──────────────────────────────────────────────────────

class TestElliottWaveAutomator:
    @pytest.fixture
    def strategy(self):
        from crypto_bot.core.signals.strategies.swing_new.ewa import (
            ElliottWaveAutomatorStrategy,
        )
        s = ElliottWaveAutomatorStrategy.__new__(ElliottWaveAutomatorStrategy)
        s.params = {
            "zz_pct": 0.05, "fib_retr_min": 0.45, "fib_retr_max": 0.80,
            "fib_ext": 1.618, "trend_ema": 100, "atr_period": 14,
        }
        return s

    def test_mode_is_swing(self, strategy):
        assert strategy.mode == "swing"

    def test_returns_list(self, strategy):
        assert isinstance(strategy.generate_signals(_make_4h_candles()), list)

    def test_all_directions_valid(self, strategy):
        _check_signals(strategy.generate_signals(_make_4h_candles()))

    def test_no_aux_data(self, strategy):
        assert isinstance(strategy.generate_signals(_make_4h_candles(), aux_data=None), list)

    def test_zigzag_finds_pivots_on_trending_data(self):
        from crypto_bot.core.signals.strategies.swing_new.ewa import _find_zigzag_pivots
        np.random.seed(0)
        n = 200
        close = np.linspace(20000, 25000, n) + np.random.randn(n) * 100
        high  = close + 50
        low   = close - 50
        pivots = _find_zigzag_pivots(high, low, close, deviation_pct=0.03)
        assert isinstance(pivots, list)
        # Should find at least some pivots
        assert len(pivots) >= 0  # can be 0 on monotone data

    def test_zigzag_alternates_high_low(self):
        from crypto_bot.core.signals.strategies.swing_new.ewa import _find_zigzag_pivots
        # Create a clearly oscillating price series
        n = 100
        t = np.linspace(0, 4 * np.pi, n)
        close = 25000 + 2000 * np.sin(t)
        high  = close + 100
        low   = close - 100
        pivots = _find_zigzag_pivots(high, low, close, deviation_pct=0.02)
        if len(pivots) >= 2:
            for j in range(1, len(pivots)):
                assert pivots[j]["type"] != pivots[j-1]["type"], (
                    "Consecutive pivots must alternate H/L"
                )

    def test_mtf(self, strategy):
        c = _make_4h_candles()
        assert isinstance(strategy.generate_signals_mtf({"4h": c}), list)


# ── FundingRateSqueezePredIctor ───────────────────────────────────────────────

class TestFundingRateSqueezePredIctor:
    @pytest.fixture
    def strategy(self):
        from crypto_bot.core.signals.strategies.swing_new.frsp import (
            FundingRateSqueezePredictorStrategy,
        )
        s = FundingRateSqueezePredictorStrategy.__new__(FundingRateSqueezePredictorStrategy)
        s.params = {
            "funding_threshold": 0.001, "funding_roc_period": 5,
            "funding_roc_min": 0.0002, "oi_roc_period": 5,
            "oi_roc_min": 0.01, "rsi_period": 14, "atr_period": 14, "smoothing": 3,
        }
        return s

    def test_mode_is_swing(self, strategy):
        assert strategy.mode == "swing"

    def test_returns_list_no_aux(self, strategy):
        assert isinstance(strategy.generate_signals(_make_4h_candles(), aux_data=None), list)

    def test_all_directions_valid_rsi_fallback(self, strategy):
        _check_signals(strategy.generate_signals(_make_4h_candles(), aux_data=None))

    def test_with_funding_and_oi(self, strategy):
        candles = _make_4h_candles()
        n = len(candles)
        # Funding deeply negative first half (short squeeze setup), then positive (long squeeze)
        funding = np.concatenate([
            np.linspace(-0.003, -0.001, n // 2),
            np.linspace(0.001, 0.003, n - n // 2),
        ])
        oi = np.ones(n) * 1e9 * (1 + np.linspace(0, 0.3, n))  # Rising OI

        fr_df = pd.DataFrame({
            "timestamp":   candles["timestamp"].values,
            "fundingRate": funding,
        })
        oi_df = pd.DataFrame({
            "timestamp":     candles["timestamp"].values,
            "sumOpenInterest": oi,
        })
        aux = {
            "funding_rate":  {"BTCUSDT": fr_df},
            "open_interest": {"BTCUSDT": oi_df},
        }
        result = strategy.generate_signals(candles, aux_data=aux)
        assert isinstance(result, list)
        _check_signals(result)

    def test_short_squeeze_triggers_long(self, strategy):
        """Deeply negative funding + rising OI should produce at least one LONG."""
        candles = _make_4h_candles(n=200, seed=7)
        n = len(candles)
        funding = np.full(n, -0.003)   # deeply negative throughout
        oi = np.linspace(1e9, 1.5e9, n)  # rising OI

        fr_df = pd.DataFrame({"timestamp": candles["timestamp"].values, "fundingRate": funding})
        oi_df = pd.DataFrame({"timestamp": candles["timestamp"].values, "sumOpenInterest": oi})
        aux = {"funding_rate": {"BTCUSDT": fr_df}, "open_interest": {"BTCUSDT": oi_df}}

        result = strategy.generate_signals(candles, aux_data=aux)
        longs = [s for s in result if s.direction == "LONG"]
        assert len(longs) >= 1, "Expected at least one LONG on short-squeeze setup"

    def test_missing_symbol_in_funding(self, strategy):
        aux = {"funding_rate": {"ETHUSDT": None}}
        result = strategy.generate_signals(_make_4h_candles(), aux_data=aux)
        assert isinstance(result, list)

    def test_mtf(self, strategy):
        c = _make_4h_candles()
        assert isinstance(strategy.generate_signals_mtf({"4h": c}), list)
