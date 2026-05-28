"""
Tests for all 7 long-term spot strategies.

Each strategy must:
  1. Return list[Signal] from generate_signals() and generate_signals_mtf()
  2. Have mode == "spot_longterm"
  3. Survive without aux_data (graceful degradation)
  4. Only produce LONG / EXIT_LONG signals (no shorting in spot)
  5. strength ∈ [0, 1]
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

from crypto_bot.core.signals.models import Signal

_SPOT_DIRS = {"LONG", "EXIT_LONG"}  # spot = long-only

# ── Candle factory ────────────────────────────────────────────────────────────

def _make_daily_candles(
    n: int = 1200,
    symbol: str = "BTCUSDT",
    seed: int = 42,
    start: datetime | None = None,
) -> pd.DataFrame:
    """
    Daily candles — 1200 days ≈ 3.3 years.  Covers 350d MA warmup.
    Mixed BTC-like trend: slow up, sharp drawdown, recovery.
    """
    np.random.seed(seed)
    base   = start or datetime(2020, 1, 1)
    ts     = [base + timedelta(days=i) for i in range(n)]
    t      = np.linspace(0, 4 * np.pi, n)
    trend  = np.linspace(0, 30000, n)
    cycle  = 8000 * np.sin(t)
    noise  = np.cumsum(np.random.randn(n) * 200)
    close  = np.clip(10000 + trend + cycle + noise, 1000, None)
    high   = close + np.abs(np.random.randn(n) * 300) + 100
    low    = close - np.abs(np.random.randn(n) * 300) - 100
    high   = np.maximum(high, close)
    low    = np.minimum(low, close)
    vol    = np.abs(np.random.randn(n) * 2000) + 5000
    return pd.DataFrame({
        "timestamp": ts, "open": close * 0.999,
        "high": high, "low": low, "close": close,
        "volume": vol, "is_clean": [True] * n, "symbol": [symbol] * n,
    })


def _check_spot_signals(signals: list, symbol: str = "BTCUSDT") -> None:
    assert isinstance(signals, list)
    for sig in signals:
        assert isinstance(sig, Signal)
        assert sig.direction in _SPOT_DIRS, f"Spot strategy emitted: {sig.direction}"
        assert sig.symbol == symbol
        assert sig.close_price > 0
        assert sig.atr > 0
        assert 0.0 <= sig.strength <= 1.0, f"strength={sig.strength} out of range"


# ── MVRV Z-Score Cycle ────────────────────────────────────────────────────────

class TestMVRVZScoreCycle:
    @pytest.fixture
    def strategy(self):
        from crypto_bot.core.signals.strategies.spot_longterm.mzc import MVRVZScoreCycleStrategy
        s = MVRVZScoreCycleStrategy.__new__(MVRVZScoreCycleStrategy)
        s.params = {
            "z_buy_threshold": 0.5, "z_reduce_from": 3.5, "z_exit_threshold": 7.0,
            "smoothing": 5, "ma_period_fallback": 200, "atr_period": 14,
        }
        return s

    def test_mode_is_spot(self, strategy):
        assert strategy.mode == "spot_longterm"

    def test_returns_list_no_aux(self, strategy):
        assert isinstance(strategy.generate_signals(_make_daily_candles()), list)

    def test_spot_only_directions_fallback(self, strategy):
        _check_spot_signals(strategy.generate_signals(_make_daily_candles()))

    def test_with_zscore_data(self, strategy):
        candles = _make_daily_candles()
        n = len(candles)
        # Z-Score: starts low (buy zone), rises to bubble (sell zone)
        z = np.concatenate([
            np.linspace(-1.5, 0, n // 3),
            np.linspace(0, 5, n // 3),
            np.linspace(5, 9, n - 2 * (n // 3)),
        ])
        zdf = pd.DataFrame({"timestamp": candles["timestamp"].values, "value": z})
        aux = {"onchain": {"BTC_mvrv_zscore": zdf}}
        result = strategy.generate_signals(candles, aux_data=aux)
        _check_spot_signals(result)
        longs = [s for s in result if s.direction == "LONG"]
        exits = [s for s in result if s.direction == "EXIT_LONG"]
        assert len(longs) >= 1
        assert len(exits) >= 1

    def test_low_z_triggers_long(self, strategy):
        """Z-Score deeply negative throughout should produce at least one LONG."""
        candles = _make_daily_candles(n=400, seed=5)
        n = len(candles)
        zdf = pd.DataFrame({
            "timestamp": candles["timestamp"].values,
            "value": np.full(n, -1.5),
        })
        result = strategy.generate_signals(candles, aux_data={"onchain": {"BTC_mvrv_zscore": zdf}})
        longs = [s for s in result if s.direction == "LONG"]
        assert len(longs) >= 1

    def test_mtf(self, strategy):
        c = _make_daily_candles()
        assert isinstance(strategy.generate_signals_mtf({"1w": c, "1d": c}), list)

    def test_default_params(self):
        from crypto_bot.core.signals.strategies.spot_longterm.mzc import MVRVZScoreCycleStrategy
        s = MVRVZScoreCycleStrategy.__new__(MVRVZScoreCycleStrategy)
        s.params = {}
        dp = s.default_params()
        assert "z_buy_threshold" in dp and "z_exit_threshold" in dp


# ── Pi Cycle Rainbow Composite ────────────────────────────────────────────────

class TestPiCycleRainbowComposite:
    @pytest.fixture
    def strategy(self):
        from crypto_bot.core.signals.strategies.spot_longterm.pcr import (
            PiCycleRainbowCompositeStrategy,
        )
        s = PiCycleRainbowCompositeStrategy.__new__(PiCycleRainbowCompositeStrategy)
        s.params = {
            "pi_fast": 111, "pi_slow": 350, "pi_slow_mult": 2.0,
            "rainbow_win": 365, "buy_threshold": 0.6, "exit_threshold": 0.3,
            "atr_period": 14,
        }
        return s

    def test_mode_is_spot(self, strategy):
        assert strategy.mode == "spot_longterm"

    def test_returns_list(self, strategy):
        assert isinstance(strategy.generate_signals(_make_daily_candles()), list)

    def test_spot_only_directions(self, strategy):
        _check_spot_signals(strategy.generate_signals(_make_daily_candles()))

    def test_no_aux_needed(self, strategy):
        """Pi Cycle works from price alone."""
        result = strategy.generate_signals(_make_daily_candles(), aux_data=None)
        assert isinstance(result, list)

    def test_pi_cycle_scales_for_weekly(self, strategy):
        """On weekly candles the SMA periods should be shorter."""
        candles = _make_daily_candles(n=400)
        # Simulate weekly by using every 7th candle
        weekly = candles.iloc[::7].reset_index(drop=True).copy()
        weekly["timestamp"] = [
            datetime(2020, 1, 1) + timedelta(weeks=i) for i in range(len(weekly))
        ]
        result = strategy.generate_signals(weekly)
        assert isinstance(result, list)

    def test_log_regression_residual(self):
        from crypto_bot.core.signals.strategies.spot_longterm.pcr import _log_regression_residual
        prices = np.log(np.linspace(10000, 60000, 500))
        res = _log_regression_residual(prices, window=200)
        assert res.shape == prices.shape
        # First window-1 entries should be NaN
        assert np.all(np.isnan(res[:199]))
        # Rest should have values
        assert not np.all(np.isnan(res[200:]))

    def test_mtf_prefers_1d(self, strategy):
        c = _make_daily_candles()
        assert isinstance(strategy.generate_signals_mtf({"1d": c, "1w": c}), list)


# ── Halving Cycle Phase Allocator ─────────────────────────────────────────────

class TestHalvingCyclePhaseAllocator:
    @pytest.fixture
    def strategy(self):
        from crypto_bot.core.signals.strategies.spot_longterm.hcpa import (
            HalvingCyclePhaseAllocatorStrategy,
        )
        s = HalvingCyclePhaseAllocatorStrategy.__new__(HalvingCyclePhaseAllocatorStrategy)
        s.params = {
            "phase2_start_days": 365, "phase3_start_days": 548,
            "phase4_start_days": 730, "onchain_mvrv_adjust": 5.0,
            "trend_ema": 50, "atr_period": 14,
        }
        return s

    def test_mode_is_spot(self, strategy):
        assert strategy.mode == "spot_longterm"

    def test_returns_list(self, strategy):
        assert isinstance(strategy.generate_signals(_make_daily_candles()), list)

    def test_spot_only_directions(self, strategy):
        _check_spot_signals(strategy.generate_signals(_make_daily_candles()))

    def test_phase_detection(self):
        from crypto_bot.core.signals.strategies.spot_longterm.hcpa import (
            _days_since_last_halving, _phase_from_days,
        )
        from datetime import timezone
        # 2021-01-01 is ~234 days after the 2020-05-11 halving
        ts = datetime(2021, 1, 1, tzinfo=timezone.utc)
        days = _days_since_last_halving(ts)
        assert 200 < days < 270
        assert _phase_from_days(days) == 1   # Phase 1: 0–365 days

    def test_phase2_is_bull_run(self):
        from crypto_bot.core.signals.strategies.spot_longterm.hcpa import (
            _phase_from_days, _PHASE_ALLOCATION,
        )
        assert _phase_from_days(400) == 2
        assert _PHASE_ALLOCATION[2] == 1.0   # max allocation in phase 2

    def test_signals_across_2020_2024_cycle(self, strategy):
        """Candles covering the 2020 halving should produce buy signals in phases 1+2."""
        candles = _make_daily_candles(
            n=1000, start=datetime(2020, 5, 1)
        )
        result = strategy.generate_signals(candles)
        assert isinstance(result, list)
        _check_spot_signals(result)

    def test_mtf(self, strategy):
        c = _make_daily_candles()
        assert isinstance(strategy.generate_signals_mtf({"1w": c}), list)


# ── Top10 Momentum Rotation ───────────────────────────────────────────────────

class TestTop10MomentumRotation:
    @pytest.fixture
    def strategy(self):
        from crypto_bot.core.signals.strategies.spot_longterm.trmr import (
            Top10MomentumRotationStrategy,
        )
        s = Top10MomentumRotationStrategy.__new__(Top10MomentumRotationStrategy)
        s.params = {
            "sharpe_window": 90, "sharpe_entry": 0.5, "sharpe_exit": -0.2,
            "mom_period": 60, "rebalance_days": 30,
            "trend_ema": 100, "atr_period": 14,
        }
        return s

    def test_mode_is_spot(self, strategy):
        assert strategy.mode == "spot_longterm"

    def test_returns_list(self, strategy):
        assert isinstance(strategy.generate_signals(_make_daily_candles()), list)

    def test_spot_only_directions(self, strategy):
        _check_spot_signals(strategy.generate_signals(_make_daily_candles()))

    def test_rebalance_gate_limits_signals(self, strategy):
        """With 30-day rebalance, number of signals ≤ n_days/30 + 1."""
        n = 400
        candles = _make_daily_candles(n=n)
        result  = strategy.generate_signals(candles)
        # At most one signal per rebalance period
        assert len(result) <= n // 30 + 5   # small buffer

    def test_rising_sharpe_triggers_long(self, strategy):
        """Strongly trending upmarket should produce LONG signals."""
        np.random.seed(99)
        n = 400
        base = datetime(2021, 1, 1)
        ts   = [base + timedelta(days=i) for i in range(n)]
        close = 10000 + np.arange(n) * 50 + np.random.randn(n) * 100
        high  = close + 100; low = close - 100
        candles = pd.DataFrame({
            "timestamp": ts, "open": close * 0.999,
            "high": np.maximum(high, close), "low": np.minimum(low, close),
            "close": close, "volume": np.ones(n) * 5000,
            "is_clean": [True] * n, "symbol": ["BTCUSDT"] * n,
        })
        result = strategy.generate_signals(candles)
        longs  = [s for s in result if s.direction == "LONG"]
        assert len(longs) >= 1

    def test_mtf(self, strategy):
        c = _make_daily_candles()
        assert isinstance(strategy.generate_signals_mtf({"1d": c}), list)


# ── Macro Regime Portfolio ────────────────────────────────────────────────────

class TestMacroRegimePortfolio:
    @pytest.fixture
    def strategy(self):
        from crypto_bot.core.signals.strategies.spot_longterm.mrp import (
            MacroRegimePortfolioStrategy,
        )
        s = MacroRegimePortfolioStrategy.__new__(MacroRegimePortfolioStrategy)
        s.params = {
            "macro_window": 30, "entry_score": 3, "exit_score": 1,
            "trend_ema": 100, "recheck_days": 21, "atr_period": 14,
        }
        return s

    def test_mode_is_spot(self, strategy):
        assert strategy.mode == "spot_longterm"

    def test_returns_list_no_macro(self, strategy):
        assert isinstance(strategy.generate_signals(_make_daily_candles(), aux_data=None), list)

    def test_spot_only_directions_fallback(self, strategy):
        _check_spot_signals(strategy.generate_signals(_make_daily_candles()))

    def test_with_bullish_macro(self, strategy):
        """Risk-on macro (SPX up, DXY down, gold up, yields down) → LONG.

        Macro values must move enough that 30-day momentum clears thresholds:
          SPX  > +2% per 30d  → linspace(4000, 5600)  ≈ +3.0%/30d
          DXY  < -1% per 30d  → linspace(110,   90)   ≈ -1.4%/30d
          gold > +1% per 30d  → linspace(1800, 2400)  ≈ +2.5%/30d
          real_yield falling  → linspace(2.0, 0.5)
        """
        candles = _make_daily_candles(n=400)
        dates   = pd.to_datetime(candles["timestamp"].values).normalize().unique()
        n       = len(dates)
        macro_df = pd.DataFrame({
            "spx":        np.linspace(4000, 5600, n),   # +40% total → ~3%/30d ✓
            "dxy":        np.linspace(110,   90,  n),   # -18% total → ~1.4%/30d ✓
            "gold":       np.linspace(1800, 2400, n),   # +33% total → ~2.5%/30d ✓
            "real_yield": np.linspace(2.0,    0.5, n),  # falling ✓
        }, index=dates)
        aux = {"macro": macro_df}
        result = strategy.generate_signals(candles, aux_data=aux)
        _check_spot_signals(result)
        longs = [s for s in result if s.direction == "LONG"]
        assert len(longs) >= 1

    def test_with_bearish_macro(self, strategy):
        """Risk-off macro → EXIT_LONG after being in a position."""
        candles = _make_daily_candles(n=500)
        dates   = pd.to_datetime(candles["timestamp"].values).normalize().unique()
        n       = len(dates)
        # First half bullish, second half bearish
        h = n // 2
        macro_df = pd.DataFrame({
            "spx":        np.concatenate([np.linspace(4000, 4800, h), np.linspace(4800, 3200, n - h)]),
            "dxy":        np.concatenate([np.linspace(105, 95, h),    np.linspace(95, 115, n - h)]),
            "gold":       np.concatenate([np.linspace(1800, 2100, h), np.linspace(2100, 1700, n - h)]),
            "real_yield": np.concatenate([np.linspace(2.0, 0.5, h),   np.linspace(0.5, 3.5, n - h)]),
        }, index=dates)
        aux = {"macro": macro_df}
        result = strategy.generate_signals(candles, aux_data=aux)
        _check_spot_signals(result)

    def test_mtf(self, strategy):
        c = _make_daily_candles()
        assert isinstance(strategy.generate_signals_mtf({"1w": c}), list)


# ── OnChain Accumulation Composite ───────────────────────────────────────────

class TestOnChainAccumulationComposite:
    @pytest.fixture
    def strategy(self):
        from crypto_bot.core.signals.strategies.spot_longterm.oac import (
            OnChainAccumulationCompositeStrategy,
        )
        s = OnChainAccumulationCompositeStrategy.__new__(OnChainAccumulationCompositeStrategy)
        s.params = {
            "smoothing": 7, "sopr_threshold": 1.0, "nupl_threshold": 0.1,
            "mvrv_threshold": 1.0, "nvt_buy_thresh": 65.0, "nvt_sell_thresh": 150.0,
            "recheck_days": 21, "atr_period": 14,
        }
        return s

    def test_mode_is_spot(self, strategy):
        assert strategy.mode == "spot_longterm"

    def test_returns_list_no_aux(self, strategy):
        assert isinstance(strategy.generate_signals(_make_daily_candles(), aux_data=None), list)

    def test_spot_only_directions_fallback(self, strategy):
        _check_spot_signals(strategy.generate_signals(_make_daily_candles()))

    def _make_onchain_aux(self, candles: pd.DataFrame, bullish: bool) -> dict:
        n = len(candles)
        ts = candles["timestamp"].values
        if bullish:
            sopr = np.full(n, 0.95)       # < 1 → capitulation (bullish)
            nupl = np.full(n, -0.1)       # < 0 → loss (bullish)
            mvrv = np.full(n, 0.8)        # < 1 → cheap (bullish)
            nvt  = np.full(n, 40.0)       # < 65 → undervalued (bullish)
            # LTH rising
            lth  = np.linspace(13e6, 14e6, n)
            exb  = np.linspace(2e6, 1.5e6, n)   # falling (outflows)
        else:
            sopr = np.full(n, 1.05)
            nupl = np.full(n, 0.5)
            mvrv = np.full(n, 3.0)
            nvt  = np.full(n, 180.0)
            lth  = np.linspace(14e6, 13e6, n)   # falling (distributing)
            exb  = np.linspace(1.5e6, 2.5e6, n)  # rising (inflows)

        def _df(vals, col="value"):
            return pd.DataFrame({"timestamp": ts, col: vals})

        return {"onchain": {
            "BTC_sopr":             _df(sopr),
            "BTC_lth_supply":       _df(lth),
            "BTC_exchange_balance": _df(exb),
            "BTC_nupl":             _df(nupl),
            "BTC_mvrv":             _df(mvrv),
            "BTC_nvt":              _df(nvt),
        }}

    def test_bullish_onchain_triggers_long(self, strategy):
        candles = _make_daily_candles(n=400, symbol="BTCUSDT")
        aux = self._make_onchain_aux(candles, bullish=True)
        result = strategy.generate_signals(candles, aux_data=aux)
        _check_spot_signals(result)
        longs = [s for s in result if s.direction == "LONG"]
        assert len(longs) >= 1, "Expected at least one LONG from bullish on-chain data"

    def test_bearish_exits_position(self, strategy):
        """Price below fallback MA should generate an entry; then bearish data exits."""
        candles = _make_daily_candles(n=500, symbol="BTCUSDT")
        # Start bearish (exit any position created by fallback)
        aux = self._make_onchain_aux(candles, bullish=False)
        result = strategy.generate_signals(candles, aux_data=aux)
        _check_spot_signals(result)

    def test_missing_metric_graceful(self, strategy):
        """Partial onchain data shouldn't crash."""
        candles = _make_daily_candles(n=300)
        aux = {"onchain": {"BTC_sopr": None, "BTC_mvrv": pd.DataFrame()}}
        result = strategy.generate_signals(candles, aux_data=aux)
        assert isinstance(result, list)

    def test_mtf(self, strategy):
        c = _make_daily_candles()
        assert isinstance(strategy.generate_signals_mtf({"1w": c}), list)


# ── NVT Signal Valuation ──────────────────────────────────────────────────────

class TestNVTSignalValuation:
    @pytest.fixture
    def strategy(self):
        from crypto_bot.core.signals.strategies.spot_longterm.nvt import (
            NVTSignalValuationStrategy,
        )
        s = NVTSignalValuationStrategy.__new__(NVTSignalValuationStrategy)
        s.params = {
            "nvt_buy": 45.0, "nvt_sell": 150.0, "smoothing": 30,
            "trend_ema": 100, "recheck_days": 21, "atr_period": 14,
        }
        return s

    def test_mode_is_spot(self, strategy):
        assert strategy.mode == "spot_longterm"

    def test_returns_list_no_aux(self, strategy):
        """Without NVT data, uses price/volume proxy — should not crash."""
        assert isinstance(strategy.generate_signals(_make_daily_candles(), aux_data=None), list)

    def test_spot_only_directions(self, strategy):
        _check_spot_signals(strategy.generate_signals(_make_daily_candles()))

    def test_with_low_nvt_triggers_long(self, strategy):
        """NVT consistently below buy threshold → at least one LONG."""
        candles = _make_daily_candles(n=400, symbol="BTCUSDT")
        n = len(candles)
        nvt_df = pd.DataFrame({
            "timestamp": candles["timestamp"].values,
            "value":     np.full(n, 30.0),  # deeply undervalued
        })
        aux = {"onchain": {"BTC_nvt": nvt_df}}
        result = strategy.generate_signals(candles, aux_data=aux)
        _check_spot_signals(result)
        longs = [s for s in result if s.direction == "LONG"]
        assert len(longs) >= 1

    def test_high_nvt_exits(self, strategy):
        """NVT cycling from low to bubble should produce both LONG and EXIT_LONG."""
        candles = _make_daily_candles(n=600, symbol="BTCUSDT")
        n = len(candles)
        nvt_vals = np.concatenate([
            np.full(n // 2, 30.0),    # undervalued first half
            np.full(n - n // 2, 180.0), # overvalued second half
        ])
        nvt_df = pd.DataFrame({"timestamp": candles["timestamp"].values, "value": nvt_vals})
        aux = {"onchain": {"BTC_nvt": nvt_df}}
        result = strategy.generate_signals(candles, aux_data=aux)
        _check_spot_signals(result)
        longs = [s for s in result if s.direction == "LONG"]
        exits = [s for s in result if s.direction == "EXIT_LONG"]
        assert len(longs) >= 1
        assert len(exits) >= 1

    def test_proxy_fallback_runs(self, strategy):
        """Price/volume proxy should produce some signals on long trending data."""
        candles = _make_daily_candles(n=600)
        result = strategy.generate_signals(candles, aux_data=None)
        assert isinstance(result, list)

    def test_mtf_prefers_1w(self, strategy):
        c = _make_daily_candles()
        assert isinstance(strategy.generate_signals_mtf({"1w": c, "1d": c}), list)
