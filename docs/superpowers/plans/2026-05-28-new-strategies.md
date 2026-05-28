# New Strategies Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 5 new trading strategies (Ichimoku Cloud, Stoch RSI, OI Divergence, MACD Hist Divergence, Market Regime) to the crypto backtesting system, each with full Optuna auto-tuning and test coverage.

**Architecture:** Each strategy is a self-contained file inheriting `BaseStrategy`, with `param_space` for Optuna and `generate_signals()` for signal production. New indicator helpers are added to `indicators.py` first. OI data is fetched from Binance Futures and cached identically to funding rates.

**Tech Stack:** Python 3.13, pandas, numpy, pydantic, pytest, python-binance

---

## File Map

| Action | File | Purpose |
|---|---|---|
| Modify | `crypto_bot/core/signals/indicators.py` | Add `ichimoku()`, `stoch_rsi()`, `ema_slope()` |
| Create | `crypto_bot/core/signals/strategies/ichimoku_cloud.py` | IchimokuCloudStrategy |
| Create | `crypto_bot/core/signals/strategies/stoch_rsi.py` | StochRSIStrategy |
| Create | `crypto_bot/core/signals/strategies/macd_hist_divergence.py` | MACDHistDivergenceStrategy |
| Create | `crypto_bot/core/signals/strategies/market_regime.py` | MarketRegimeStrategy |
| Modify | `crypto_bot/core/data/binance_history.py` | Add `fetch_open_interest()` |
| Create | `crypto_bot/core/signals/strategies/open_interest_divergence.py` | OpenInterestDivergenceStrategy |
| Modify | `tests/test_strategies.py` | 5 new test classes (26 tests) |
| Modify | `backtest/orchestrator.py` | Register 5 new strategies |
| Modify | `main.py` | Register 5 new strategies in CLI map |

---

## Task 1: Add Indicator Helpers

**Files:**
- Modify: `crypto_bot/core/signals/indicators.py` (append to end of file)

- [ ] **Step 1: Write failing indicator tests**

Create `tests/test_new_indicators.py`:

```python
"""Tests for new indicators: ichimoku, stoch_rsi, ema_slope."""
import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta
from crypto_bot.core.signals.indicators import ichimoku, stoch_rsi, ema_slope


def _make_series(n=300, trend="up", seed=42):
    np.random.seed(seed)
    if trend == "up":
        close = 30000.0 + np.arange(n) * 8.0 + np.random.randn(n) * 40.0
    else:
        close = 50000.0 - np.arange(n) * 8.0 + np.random.randn(n) * 40.0
    high = close + np.abs(np.random.randn(n) * 30.0)
    low  = close - np.abs(np.random.randn(n) * 30.0)
    idx  = [datetime(2023, 1, 1) + timedelta(hours=4 * i) for i in range(n)]
    return (
        pd.Series(high, index=idx),
        pd.Series(low,  index=idx),
        pd.Series(close, index=idx),
    )


class TestIchimoku:
    def test_returns_four_series(self):
        high, low, close = _make_series()
        result = ichimoku(high, low)
        assert len(result) == 4
        tenkan, kijun, span_a, span_b = result
        assert isinstance(tenkan, pd.Series)
        assert isinstance(span_b, pd.Series)

    def test_length_matches_input(self):
        high, low, close = _make_series()
        tenkan, kijun, span_a, span_b = ichimoku(high, low)
        assert len(tenkan) == len(high)
        assert len(span_b) == len(high)

    def test_tenkan_shorter_warmup_than_kijun(self):
        high, low, close = _make_series()
        tenkan, kijun, _, _ = ichimoku(high, low, tenkan_period=9, kijun_period=26)
        # tenkan has data sooner than kijun
        assert tenkan.first_valid_index() <= kijun.first_valid_index()

    def test_span_a_is_mean_of_tenkan_kijun(self):
        high, low, close = _make_series(200)
        tenkan, kijun, span_a, _ = ichimoku(high, low, tenkan_period=9, kijun_period=26, displacement=0)
        # With displacement=0, span_a[i] == (tenkan[i] + kijun[i]) / 2
        expected = (tenkan + kijun) / 2
        pd.testing.assert_series_equal(span_a, expected, check_names=False)


class TestStochRSI:
    def test_returns_two_series(self):
        _, _, close = _make_series()
        k, d = stoch_rsi(close)
        assert isinstance(k, pd.Series)
        assert isinstance(d, pd.Series)

    def test_values_bounded_0_100(self):
        _, _, close = _make_series()
        k, d = stoch_rsi(close)
        valid_k = k.dropna()
        valid_d = d.dropna()
        assert (valid_k >= 0).all() and (valid_k <= 100).all()
        assert (valid_d >= 0).all() and (valid_d <= 100).all()

    def test_d_is_smoother_than_k(self):
        _, _, close = _make_series()
        k, d = stoch_rsi(close, smooth_k=3, smooth_d=3)
        assert d.dropna().std() <= k.dropna().std()

    def test_length_matches_input(self):
        _, _, close = _make_series()
        k, d = stoch_rsi(close)
        assert len(k) == len(close)


class TestEMASlope:
    def test_positive_on_uptrend(self):
        _, _, close = _make_series(trend="up")
        slope = ema_slope(close, period=20, lookback=5)
        assert slope.dropna().mean() > 0

    def test_negative_on_downtrend(self):
        _, _, close = _make_series(trend="down")
        slope = ema_slope(close, period=20, lookback=5)
        assert slope.dropna().mean() < 0

    def test_length_matches_input(self):
        _, _, close = _make_series()
        slope = ema_slope(close, period=20, lookback=5)
        assert len(slope) == len(close)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ~/Desktop/backtest_test && source .venv/bin/activate
python -m pytest tests/test_new_indicators.py -v 2>&1 | tail -20
```

Expected: `ImportError` — `ichimoku`, `stoch_rsi`, `ema_slope` not yet defined.

- [ ] **Step 3: Add indicator helpers to `indicators.py`**

Append to end of `crypto_bot/core/signals/indicators.py`:

```python

def ichimoku(
    high: pd.Series,
    low: pd.Series,
    tenkan_period: int = 9,
    kijun_period: int = 26,
    senkou_b_period: int = 52,
    displacement: int = 26,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Ichimoku Cloud components.
    Returns (tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b).
    senkou_span_a and B are shifted forward by `displacement` bars.
    """
    tenkan_sen  = (high.rolling(tenkan_period).max()    + low.rolling(tenkan_period).min())    / 2
    kijun_sen   = (high.rolling(kijun_period).max()     + low.rolling(kijun_period).min())     / 2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(displacement)
    senkou_span_b = (
        (high.rolling(senkou_b_period).max() + low.rolling(senkou_b_period).min()) / 2
    ).shift(displacement)
    return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b


def stoch_rsi(
    close: pd.Series,
    rsi_period: int = 14,
    stoch_period: int = 14,
    smooth_k: int = 3,
    smooth_d: int = 3,
) -> Tuple[pd.Series, pd.Series]:
    """
    Stochastic RSI oscillator.
    Returns (%K, %D) both in range 0–100.
    """
    rsi_vals    = rsi(close, rsi_period)
    rsi_min     = rsi_vals.rolling(stoch_period).min()
    rsi_max     = rsi_vals.rolling(stoch_period).max()
    raw         = (rsi_vals - rsi_min) / (rsi_max - rsi_min).replace(0, np.nan)
    k           = raw.rolling(smooth_k).mean() * 100.0
    d           = k.rolling(smooth_d).mean()
    return k, d


def ema_slope(
    close: pd.Series,
    period: int,
    lookback: int = 5,
) -> pd.Series:
    """
    EMA slope as fractional change over `lookback` bars.
    Returns (ema[i] - ema[i-lookback]) / ema[i-lookback].
    Positive = rising, negative = falling.
    """
    ema_vals = ema(close, period)
    prev     = ema_vals.shift(lookback)
    return (ema_vals - prev) / prev.replace(0, np.nan)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_new_indicators.py -v 2>&1 | tail -20
```

Expected: `11 passed`

- [ ] **Step 5: Run full test suite to confirm no regressions**

```bash
python -m pytest tests/ -v 2>&1 | tail -10
```

Expected: `115 passed` (all existing tests still green)

- [ ] **Step 6: Commit**

```bash
git add crypto_bot/core/signals/indicators.py tests/test_new_indicators.py
git commit -m "feat: add ichimoku, stoch_rsi, ema_slope indicator helpers with tests

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 2: IchimokuCloudStrategy

**Files:**
- Create: `crypto_bot/core/signals/strategies/ichimoku_cloud.py`
- Modify: `tests/test_strategies.py` (append test class)

- [ ] **Step 1: Write the failing tests**

Append to end of `tests/test_strategies.py`:

```python
from crypto_bot.core.signals.strategies.ichimoku_cloud import IchimokuCloudStrategy


class TestIchimokuCloudStrategy:
    _p = {
        "tenkan_period": 9, "kijun_period": 26,
        "senkou_b_period": 52, "atr_period": 14,
    }

    def test_produces_long_in_uptrend(self):
        df = make_candles(300, trend="up")
        sigs = IchimokuCloudStrategy(self._p).generate_signals(df)
        assert len([s for s in sigs if s.direction == "LONG"]) > 0

    def test_produces_short_in_downtrend(self):
        df = make_candles(300, trend="down")
        sigs = IchimokuCloudStrategy(self._p).generate_signals(df)
        assert len([s for s in sigs if s.direction == "SHORT"]) > 0

    def test_no_signals_short_data(self):
        df = make_candles(50, trend="up")
        assert IchimokuCloudStrategy(self._p).generate_signals(df) == []

    def test_valid_directions(self):
        df = make_candles(300, trend="up")
        valid = {"LONG", "SHORT", "EXIT_LONG", "EXIT_SHORT"}
        for s in IchimokuCloudStrategy(self._p).generate_signals(df):
            assert s.direction in valid

    def test_default_params_in_space(self):
        s = IchimokuCloudStrategy({})
        for k, spec in s.param_space.items():
            assert spec[0] <= s.default_params()[k] <= spec[1]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_strategies.py::TestIchimokuCloudStrategy -v 2>&1 | tail -10
```

Expected: `ImportError` — module not found.

- [ ] **Step 3: Create the strategy file**

Create `crypto_bot/core/signals/strategies/ichimoku_cloud.py`:

```python
"""
Ichimoku Cloud Strategy.
Entry LONG : price above cloud + tenkan crosses above kijun + chikou above price[i-26].
Entry SHORT: price below cloud + tenkan crosses below kijun + chikou below price[i-26].
Exit       : tenkan/kijun cross reverses OR price closes inside cloud.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from crypto_bot.core.signals.base import BaseStrategy
from crypto_bot.core.signals.models import Signal
from crypto_bot.core.signals.indicators import ichimoku, atr


class IchimokuCloudStrategy(BaseStrategy):
    @property
    def param_space(self) -> dict:
        return {
            "tenkan_period":   (7,  12, "int"),
            "kijun_period":    (22, 30, "int"),
            "senkou_b_period": (44, 60, "int"),
            "atr_period":      (10, 20, "int"),
        }

    def generate_signals(self, candles: pd.DataFrame, aux_data=None) -> list[Signal]:
        p      = self.params
        close  = candles["close"]
        high   = candles["high"]
        low    = candles["low"]
        symbol = str(candles["symbol"].iloc[0])

        displacement    = 26  # standard Ichimoku constant
        tenkan_p        = int(p["tenkan_period"])
        kijun_p         = int(p["kijun_period"])
        senkou_b_p      = int(p["senkou_b_period"])
        atr_p           = int(p["atr_period"])

        tenkan, kijun, span_a, span_b = ichimoku(
            high, low,
            tenkan_period=tenkan_p,
            kijun_period=kijun_p,
            senkou_b_period=senkou_b_p,
            displacement=displacement,
        )
        atr_v  = atr(high, low, close, atr_p)
        warmup = senkou_b_p + displacement + 1

        signals: list[Signal] = []
        open_pos: str | None  = None

        for i in range(warmup, len(candles)):
            if not candles["is_clean"].iloc[i]:
                continue

            t_i    = tenkan.iloc[i];  t_p  = tenkan.iloc[i - 1]
            k_i    = kijun.iloc[i];   k_p  = kijun.iloc[i - 1]
            sa_i   = span_a.iloc[i]
            sb_i   = span_b.iloc[i]
            atr_i  = atr_v.iloc[i]
            cl_i   = close.iloc[i]

            if any(np.isnan(v) for v in (t_i, k_i, sa_i, sb_i, atr_i, t_p, k_p)):
                continue

            cloud_top    = max(sa_i, sb_i)
            cloud_bottom = min(sa_i, sb_i)
            above_cloud  = cl_i > cloud_top
            below_cloud  = cl_i < cloud_bottom
            inside_cloud = cloud_bottom <= cl_i <= cloud_top

            tk_cross_up   = (t_i > k_i) and (t_p <= k_p)
            tk_cross_down = (t_i < k_i) and (t_p >= k_p)

            # Chikou: compare current close vs close 26 bars ago
            chikou_above = cl_i > close.iloc[i - displacement]
            chikou_below = cl_i < close.iloc[i - displacement]

            direction: str | None = None
            reason: list[str]     = []

            if above_cloud and tk_cross_up and chikou_above and open_pos != "LONG":
                direction = "LONG"
                reason    = ["above_cloud", "tk_cross_up", "chikou_above"]
                open_pos  = "LONG"
            elif below_cloud and tk_cross_down and chikou_below and open_pos != "SHORT":
                direction = "SHORT"
                reason    = ["below_cloud", "tk_cross_down", "chikou_below"]
                open_pos  = "SHORT"
            elif open_pos == "LONG" and (tk_cross_down or inside_cloud):
                direction = "EXIT_LONG"
                reason    = ["tk_cross_down" if tk_cross_down else "price_in_cloud"]
                open_pos  = None
            elif open_pos == "SHORT" and (tk_cross_up or inside_cloud):
                direction = "EXIT_SHORT"
                reason    = ["tk_cross_up" if tk_cross_up else "price_in_cloud"]
                open_pos  = None

            if direction:
                dist     = abs(cl_i - cloud_top if above_cloud else cloud_bottom - cl_i)
                strength = min(dist / max(atr_i, 1e-9), 1.0)
                signals.append(Signal(
                    strategy=self.name,
                    symbol=symbol,
                    timestamp=candles["timestamp"].iloc[i],
                    direction=direction,
                    strength=strength,
                    close_price=float(cl_i),
                    atr=float(atr_i),
                    reason=reason,
                ))

        return signals
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_strategies.py::TestIchimokuCloudStrategy -v 2>&1 | tail -10
```

Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add crypto_bot/core/signals/strategies/ichimoku_cloud.py tests/test_strategies.py
git commit -m "feat: IchimokuCloudStrategy — cloud + tenkan/kijun + chikou signals

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 3: StochRSIStrategy

**Files:**
- Create: `crypto_bot/core/signals/strategies/stoch_rsi.py`
- Modify: `tests/test_strategies.py` (append test class)

- [ ] **Step 1: Write the failing tests**

Append to end of `tests/test_strategies.py`:

```python
from crypto_bot.core.signals.strategies.stoch_rsi import StochRSIStrategy


class TestStochRSIStrategy:
    _p = {
        "rsi_period": 14, "stoch_period": 14,
        "smooth_k": 3, "smooth_d": 3, "atr_period": 14,
    }

    def test_produces_long_in_uptrend(self):
        df = make_candles(300, trend="up")
        sigs = StochRSIStrategy(self._p).generate_signals(df)
        assert len([s for s in sigs if s.direction == "LONG"]) > 0

    def test_produces_short_in_downtrend(self):
        df = make_candles(300, trend="down")
        sigs = StochRSIStrategy(self._p).generate_signals(df)
        assert len([s for s in sigs if s.direction == "SHORT"]) > 0

    def test_no_signals_short_data(self):
        df = make_candles(20, trend="up")
        assert StochRSIStrategy(self._p).generate_signals(df) == []

    def test_valid_directions(self):
        df = make_candles(300, trend="up")
        valid = {"LONG", "SHORT", "EXIT_LONG", "EXIT_SHORT"}
        for s in StochRSIStrategy(self._p).generate_signals(df):
            assert s.direction in valid

    def test_default_params_in_space(self):
        s = StochRSIStrategy({})
        for k, spec in s.param_space.items():
            assert spec[0] <= s.default_params()[k] <= spec[1]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_strategies.py::TestStochRSIStrategy -v 2>&1 | tail -10
```

Expected: `ImportError` — module not found.

- [ ] **Step 3: Create the strategy file**

Create `crypto_bot/core/signals/strategies/stoch_rsi.py`:

```python
"""
Stochastic RSI Strategy.
Entry LONG : %K crosses above %D while both below 20 (oversold zone).
Entry SHORT: %K crosses below %D while both above 80 (overbought zone).
Exit       : %K/%D cross reverses OR both enter neutral zone (40-60).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from crypto_bot.core.signals.base import BaseStrategy
from crypto_bot.core.signals.models import Signal
from crypto_bot.core.signals.indicators import stoch_rsi, atr


class StochRSIStrategy(BaseStrategy):
    @property
    def param_space(self) -> dict:
        return {
            "rsi_period":   (10, 21, "int"),
            "stoch_period": (10, 21, "int"),
            "smooth_k":     (2,  5,  "int"),
            "smooth_d":     (2,  5,  "int"),
            "atr_period":   (10, 20, "int"),
        }

    def generate_signals(self, candles: pd.DataFrame, aux_data=None) -> list[Signal]:
        p      = self.params
        close  = candles["close"]
        high   = candles["high"]
        low    = candles["low"]
        symbol = str(candles["symbol"].iloc[0])

        rsi_p    = int(p["rsi_period"])
        stoch_p  = int(p["stoch_period"])
        sk       = int(p["smooth_k"])
        sd       = int(p["smooth_d"])
        atr_p    = int(p["atr_period"])

        k_series, d_series = stoch_rsi(close, rsi_p, stoch_p, sk, sd)
        atr_v = atr(high, low, close, atr_p)
        warmup = rsi_p + stoch_p + sk + sd + 1

        signals: list[Signal] = []
        open_pos: str | None  = None

        for i in range(warmup, len(candles)):
            if not candles["is_clean"].iloc[i]:
                continue

            k_i   = k_series.iloc[i];  k_p = k_series.iloc[i - 1]
            d_i   = d_series.iloc[i];  d_p = d_series.iloc[i - 1]
            atr_i = atr_v.iloc[i]

            if any(np.isnan(v) for v in (k_i, d_i, k_p, d_p, atr_i)):
                continue

            k_cross_up   = (k_i > d_i)  and (k_p <= d_p)
            k_cross_down = (k_i < d_i)  and (k_p >= d_p)
            oversold     = k_i < 20     and d_i < 20
            overbought   = k_i > 80     and d_i > 80
            neutral      = 40.0 <= k_i  <= 60.0

            direction: str | None = None
            reason: list[str]     = []

            if k_cross_up and oversold and open_pos != "LONG":
                direction = "LONG"
                reason    = ["k_cross_up", f"k={k_i:.1f}<20", f"d={d_i:.1f}<20"]
                open_pos  = "LONG"
            elif k_cross_down and overbought and open_pos != "SHORT":
                direction = "SHORT"
                reason    = ["k_cross_down", f"k={k_i:.1f}>80", f"d={d_i:.1f}>80"]
                open_pos  = "SHORT"
            elif open_pos == "LONG" and (k_cross_down or neutral):
                direction = "EXIT_LONG"
                reason    = ["k_cross_down" if k_cross_down else "neutral_zone"]
                open_pos  = None
            elif open_pos == "SHORT" and (k_cross_up or neutral):
                direction = "EXIT_SHORT"
                reason    = ["k_cross_up" if k_cross_up else "neutral_zone"]
                open_pos  = None

            if direction:
                if "LONG" in direction and not direction.startswith("EXIT"):
                    strength = min((20 - min(k_i, d_i)) / 20.0, 1.0)
                elif "SHORT" in direction and not direction.startswith("EXIT"):
                    strength = min((max(k_i, d_i) - 80) / 20.0, 1.0)
                else:
                    strength = 0.5
                signals.append(Signal(
                    strategy=self.name,
                    symbol=symbol,
                    timestamp=candles["timestamp"].iloc[i],
                    direction=direction,
                    strength=max(0.0, min(strength, 1.0)),
                    close_price=float(close.iloc[i]),
                    atr=float(atr_i),
                    reason=reason,
                ))

        return signals
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_strategies.py::TestStochRSIStrategy -v 2>&1 | tail -10
```

Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add crypto_bot/core/signals/strategies/stoch_rsi.py tests/test_strategies.py
git commit -m "feat: StochRSIStrategy — %K/%D crossover in oversold/overbought zones

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 4: MACDHistDivergenceStrategy

**Files:**
- Create: `crypto_bot/core/signals/strategies/macd_hist_divergence.py`
- Modify: `tests/test_strategies.py` (append test class)

Note: No new indicator helpers needed — `macd()` already exists in `indicators.py`.

- [ ] **Step 1: Write the failing tests**

Append to end of `tests/test_strategies.py`:

```python
from crypto_bot.core.signals.strategies.macd_hist_divergence import MACDHistDivergenceStrategy


class TestMACDHistDivergenceStrategy:
    _p = {
        "macd_fast": 12, "macd_slow": 26,
        "macd_signal": 9, "divergence_lookback": 5, "atr_period": 14,
    }

    def test_produces_long_in_uptrend(self):
        df = make_candles(300, trend="up")
        sigs = MACDHistDivergenceStrategy(self._p).generate_signals(df)
        assert len([s for s in sigs if s.direction == "LONG"]) > 0

    def test_produces_short_in_downtrend(self):
        df = make_candles(300, trend="down")
        sigs = MACDHistDivergenceStrategy(self._p).generate_signals(df)
        assert len([s for s in sigs if s.direction == "SHORT"]) > 0

    def test_no_signals_short_data(self):
        df = make_candles(20, trend="up")
        assert MACDHistDivergenceStrategy(self._p).generate_signals(df) == []

    def test_valid_directions(self):
        df = make_candles(300, trend="up")
        valid = {"LONG", "SHORT", "EXIT_LONG", "EXIT_SHORT"}
        for s in MACDHistDivergenceStrategy(self._p).generate_signals(df):
            assert s.direction in valid

    def test_default_params_in_space(self):
        s = MACDHistDivergenceStrategy({})
        for k, spec in s.param_space.items():
            assert spec[0] <= s.default_params()[k] <= spec[1]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_strategies.py::TestMACDHistDivergenceStrategy -v 2>&1 | tail -10
```

Expected: `ImportError` — module not found.

- [ ] **Step 3: Create the strategy file**

Create `crypto_bot/core/signals/strategies/macd_hist_divergence.py`:

```python
"""
MACD Histogram Divergence Strategy.
Bullish divergence: price lower low + histogram higher low + histogram turning positive.
Bearish divergence: price higher high + histogram lower high + histogram turning negative.
Exit: histogram crosses zero in opposite direction.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from crypto_bot.core.signals.base import BaseStrategy
from crypto_bot.core.signals.models import Signal
from crypto_bot.core.signals.indicators import macd, atr


class MACDHistDivergenceStrategy(BaseStrategy):
    @property
    def param_space(self) -> dict:
        return {
            "macd_fast":           (8,  16, "int"),
            "macd_slow":           (20, 30, "int"),
            "macd_signal":         (7,  12, "int"),
            "divergence_lookback": (3,  10, "int"),
            "atr_period":          (10, 20, "int"),
        }

    def generate_signals(self, candles: pd.DataFrame, aux_data=None) -> list[Signal]:
        p      = self.params
        close  = candles["close"]
        high   = candles["high"]
        low    = candles["low"]
        symbol = str(candles["symbol"].iloc[0])

        fast_p  = int(p["macd_fast"])
        slow_p  = int(p["macd_slow"])
        sig_p   = int(p["macd_signal"])
        lb      = int(p["divergence_lookback"])
        atr_p   = int(p["atr_period"])

        _, _, hist = macd(close, fast=fast_p, slow=slow_p, signal_period=sig_p)
        atr_v      = atr(high, low, close, atr_p)
        warmup     = slow_p + sig_p + lb + 1

        signals: list[Signal] = []
        open_pos: str | None  = None

        for i in range(warmup, len(candles)):
            if not candles["is_clean"].iloc[i]:
                continue

            hist_i  = hist.iloc[i]
            hist_p  = hist.iloc[i - 1]
            atr_i   = atr_v.iloc[i]
            cl_i    = close.iloc[i]

            if np.isnan(hist_i) or np.isnan(atr_i):
                continue

            # Compare current bar vs `lb` bars ago
            hist_lb = hist.iloc[i - lb]
            cl_lb   = close.iloc[i - lb]

            if np.isnan(hist_lb):
                continue

            # Bullish divergence: price lower low + histogram higher low + hist turning up
            bullish_div = (
                cl_i < cl_lb        and   # price lower low
                hist_i > hist_lb    and   # histogram higher low
                hist_i < 0          and   # still negative (not yet crossed zero)
                hist_i > hist_p           # histogram rising (turning up)
            )

            # Bearish divergence: price higher high + histogram lower high + hist turning down
            bearish_div = (
                cl_i > cl_lb        and   # price higher high
                hist_i < hist_lb    and   # histogram lower high
                hist_i > 0          and   # still positive
                hist_i < hist_p           # histogram falling (turning down)
            )

            direction: str | None = None
            reason: list[str]     = []

            if bullish_div and open_pos != "LONG":
                direction = "LONG"
                reason    = ["bullish_div", f"hist_lb={hist_lb:.4f}", f"hist={hist_i:.4f}"]
                open_pos  = "LONG"
            elif bearish_div and open_pos != "SHORT":
                direction = "SHORT"
                reason    = ["bearish_div", f"hist_lb={hist_lb:.4f}", f"hist={hist_i:.4f}"]
                open_pos  = "SHORT"
            elif open_pos == "LONG" and hist_i >= 0 and hist_p < 0:
                direction = "EXIT_LONG"
                reason    = ["hist_zero_cross_up"]
                open_pos  = None
            elif open_pos == "SHORT" and hist_i <= 0 and hist_p > 0:
                direction = "EXIT_SHORT"
                reason    = ["hist_zero_cross_down"]
                open_pos  = None

            if direction:
                strength = min(abs(hist_i) / max(atr_i * 0.01, 1e-9), 1.0)
                signals.append(Signal(
                    strategy=self.name,
                    symbol=symbol,
                    timestamp=candles["timestamp"].iloc[i],
                    direction=direction,
                    strength=min(strength, 1.0),
                    close_price=float(cl_i),
                    atr=float(atr_i),
                    reason=reason,
                ))

        return signals
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_strategies.py::TestMACDHistDivergenceStrategy -v 2>&1 | tail -10
```

Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add crypto_bot/core/signals/strategies/macd_hist_divergence.py tests/test_strategies.py
git commit -m "feat: MACDHistDivergenceStrategy — bullish/bearish histogram divergence

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 5: MarketRegimeStrategy

**Files:**
- Create: `crypto_bot/core/signals/strategies/market_regime.py`
- Modify: `tests/test_strategies.py` (append test class)

- [ ] **Step 1: Write the failing tests**

Append to end of `tests/test_strategies.py`:

```python
from crypto_bot.core.signals.strategies.market_regime import MarketRegimeStrategy


class TestMarketRegimeStrategy:
    _p = {
        "adx_period": 14, "adx_trend_threshold": 25.0,
        "adx_range_threshold": 20.0, "ema_period": 50,
        "slope_lookback": 5, "atr_vol_threshold": 2.0, "atr_period": 14,
    }

    def test_produces_long_in_uptrend(self):
        df = make_candles(300, trend="up")
        sigs = MarketRegimeStrategy(self._p).generate_signals(df)
        assert len([s for s in sigs if s.direction == "LONG"]) > 0

    def test_produces_short_in_downtrend(self):
        df = make_candles(300, trend="down")
        sigs = MarketRegimeStrategy(self._p).generate_signals(df)
        assert len([s for s in sigs if s.direction == "SHORT"]) > 0

    def test_no_signals_short_data(self):
        df = make_candles(20, trend="up")
        assert MarketRegimeStrategy(self._p).generate_signals(df) == []

    def test_valid_directions(self):
        df = make_candles(300, trend="up")
        valid = {"LONG", "SHORT", "EXIT_LONG", "EXIT_SHORT"}
        for s in MarketRegimeStrategy(self._p).generate_signals(df):
            assert s.direction in valid

    def test_default_params_in_space(self):
        s = MarketRegimeStrategy({})
        for k, spec in s.param_space.items():
            assert spec[0] <= s.default_params()[k] <= spec[1]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_strategies.py::TestMarketRegimeStrategy -v 2>&1 | tail -10
```

Expected: `ImportError` — module not found.

- [ ] **Step 3: Create the strategy file**

Create `crypto_bot/core/signals/strategies/market_regime.py`:

```python
"""
Market Regime Strategy.
Detects market regime from ADX + EMA slope + ATR volatility.
UPTREND   (ADX > trend_thresh, slope > 0) → LONG on regime entry.
DOWNTREND (ADX > trend_thresh, slope < 0) → SHORT on regime entry.
RANGING   (ADX < range_thresh)            → EXIT open position.
VOLATILE  (ATR > vol_thresh × ATR mean)   → EXIT open position.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from crypto_bot.core.signals.base import BaseStrategy
from crypto_bot.core.signals.models import Signal
from crypto_bot.core.signals.indicators import adx, ema_slope, atr


class MarketRegimeStrategy(BaseStrategy):
    @property
    def param_space(self) -> dict:
        return {
            "adx_period":           (10,  20,  "int"),
            "adx_trend_threshold":  (20.0, 30.0),
            "adx_range_threshold":  (15.0, 25.0),
            "ema_period":           (50,  200, "int"),
            "slope_lookback":       (3,   10,  "int"),
            "atr_vol_threshold":    (1.5, 3.0),
            "atr_period":           (10,  20,  "int"),
        }

    def generate_signals(self, candles: pd.DataFrame, aux_data=None) -> list[Signal]:
        p      = self.params
        close  = candles["close"]
        high   = candles["high"]
        low    = candles["low"]
        symbol = str(candles["symbol"].iloc[0])

        adx_p        = int(p["adx_period"])
        trend_thresh = float(p["adx_trend_threshold"])
        range_thresh = float(p["adx_range_threshold"])
        ema_p        = int(p["ema_period"])
        slope_lb     = int(p["slope_lookback"])
        vol_thresh   = float(p["atr_vol_threshold"])
        atr_p        = int(p["atr_period"])

        adx_v   = adx(high, low, close, adx_p)
        slope_v = ema_slope(close, ema_p, slope_lb)
        atr_v   = atr(high, low, close, atr_p)
        # Rolling ATR mean (20-bar) for volatility detection
        atr_mean = atr_v.rolling(20).mean()

        warmup = ema_p + slope_lb + 1
        signals: list[Signal] = []
        open_pos: str | None  = None
        prev_regime: str      = "UNKNOWN"

        for i in range(warmup, len(candles)):
            if not candles["is_clean"].iloc[i]:
                continue

            adx_i   = adx_v.iloc[i]
            slope_i = slope_v.iloc[i]
            atr_i   = atr_v.iloc[i]
            atr_m   = atr_mean.iloc[i]

            if any(np.isnan(v) for v in (adx_i, slope_i, atr_i, atr_m)):
                continue

            # Determine regime
            if atr_i > vol_thresh * atr_m:
                regime = "VOLATILE"
            elif adx_i > trend_thresh:
                regime = "UPTREND" if slope_i > 0 else "DOWNTREND"
            elif adx_i < range_thresh:
                regime = "RANGING"
            else:
                regime = "TRANSITION"  # between thresholds — no signal

            direction: str | None = None
            reason: list[str]     = []

            regime_changed = regime != prev_regime

            if regime_changed:
                if regime == "UPTREND" and open_pos != "LONG":
                    direction = "LONG"
                    reason    = [f"regime=UPTREND", f"ADX={adx_i:.1f}", f"slope={slope_i:.4f}"]
                    open_pos  = "LONG"
                elif regime == "DOWNTREND" and open_pos != "SHORT":
                    direction = "SHORT"
                    reason    = [f"regime=DOWNTREND", f"ADX={adx_i:.1f}", f"slope={slope_i:.4f}"]
                    open_pos  = "SHORT"
                elif regime in ("RANGING", "VOLATILE"):
                    if open_pos == "LONG":
                        direction = "EXIT_LONG"
                        reason    = [f"regime={regime}"]
                        open_pos  = None
                    elif open_pos == "SHORT":
                        direction = "EXIT_SHORT"
                        reason    = [f"regime={regime}"]
                        open_pos  = None

            prev_regime = regime

            if direction:
                signals.append(Signal(
                    strategy=self.name,
                    symbol=symbol,
                    timestamp=candles["timestamp"].iloc[i],
                    direction=direction,
                    strength=min(adx_i / 50.0, 1.0),
                    close_price=float(close.iloc[i]),
                    atr=float(atr_i),
                    reason=reason,
                ))

        return signals
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_strategies.py::TestMarketRegimeStrategy -v 2>&1 | tail -10
```

Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add crypto_bot/core/signals/strategies/market_regime.py tests/test_strategies.py
git commit -m "feat: MarketRegimeStrategy — ADX + EMA slope regime detection (uptrend/downtrend/ranging/volatile)

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 6: OI Fetcher Extension

**Files:**
- Modify: `crypto_bot/core/data/binance_history.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_oi_fetcher.py`:

```python
"""Tests for fetch_open_interest — uses a mock to avoid real network calls."""
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
from crypto_bot.core.data.binance_history import fetch_open_interest


class TestFetchOpenInterest:
    def _mock_oi_batch(self, n=5, start_ts=1_600_000_000_000):
        """Generate a fake Binance OI response list."""
        interval = 14_400_000  # 4h in ms
        return [
            {
                "symbol": "BTCUSDT",
                "sumOpenInterest": str(10_000 + i * 100),
                "sumOpenInterestValue": str(500_000_000 + i * 1_000_000),
                "timestamp": start_ts + i * interval,
            }
            for i in range(n)
        ]

    def test_returns_series(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "data" / "cache").mkdir(parents=True)

        mock_client = MagicMock()
        mock_client.futures_open_interest_hist.return_value = self._mock_oi_batch(5)

        with patch("crypto_bot.core.data.binance_history.Client", return_value=mock_client):
            result = fetch_open_interest("BTCUSDT", "4h", "2020-09-14", "2020-09-15")

        assert isinstance(result, pd.Series)
        assert len(result) == 5
        assert result.name == "open_interest"

    def test_values_are_floats(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "data" / "cache").mkdir(parents=True)

        mock_client = MagicMock()
        mock_client.futures_open_interest_hist.return_value = self._mock_oi_batch(3)

        with patch("crypto_bot.core.data.binance_history.Client", return_value=mock_client):
            result = fetch_open_interest("BTCUSDT", "4h", "2020-09-14", "2020-09-15")

        assert result.dtype == float

    def test_returns_empty_series_on_no_data(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "data" / "cache").mkdir(parents=True)

        mock_client = MagicMock()
        mock_client.futures_open_interest_hist.return_value = []

        with patch("crypto_bot.core.data.binance_history.Client", return_value=mock_client):
            result = fetch_open_interest("BTCUSDT", "4h", "2020-09-14", "2020-09-15")

        assert isinstance(result, pd.Series)
        assert len(result) == 0

    def test_uses_cache_on_second_call(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "data" / "cache").mkdir(parents=True)

        mock_client = MagicMock()
        mock_client.futures_open_interest_hist.return_value = self._mock_oi_batch(3)

        with patch("crypto_bot.core.data.binance_history.Client", return_value=mock_client):
            fetch_open_interest("BTCUSDT", "4h", "2020-09-14", "2020-09-15")
            fetch_open_interest("BTCUSDT", "4h", "2020-09-14", "2020-09-15")

        # Client should have been constructed only once (second call hits cache)
        assert mock_client.futures_open_interest_hist.call_count == 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_oi_fetcher.py -v 2>&1 | tail -10
```

Expected: `ImportError` — `fetch_open_interest` not found.

- [ ] **Step 3: Add `fetch_open_interest` to `binance_history.py`**

Append after the `fetch_funding_rates` function (before `_to_ms`):

```python

_OI_PERIOD_MAP = {
    "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "1h", "2h": "2h",  "4h": "4h",
    "6h": "6h", "12h": "12h", "1d": "1d",
}

_TIMEFRAME_MS = {
    "1m": 60_000,      "5m": 300_000,     "15m": 900_000,
    "30m": 1_800_000,  "1h": 3_600_000,   "2h": 7_200_000,
    "4h": 14_400_000,  "6h": 21_600_000,  "12h": 43_200_000,
    "1d": 86_400_000,
}


def fetch_open_interest(
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
    api_key: str = "",
    api_secret: str = "",
) -> pd.Series:
    """
    Fetch historical open interest from Binance Futures.
    Returns pd.Series with timestamp index and float values, name='open_interest'.
    Cached to data/cache/<symbol>_oi_<timeframe>_<start>_<end>.parquet.
    """
    cache = CACHE_DIR / f"{symbol}_oi_{timeframe}_{start_date}_{end_date}.parquet"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if cache.exists():
        df = pd.read_parquet(cache)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        s = df.set_index("timestamp")["open_interest"]
        s.name = "open_interest"
        return s

    client     = Client(api_key, api_secret)
    period     = _OI_PERIOD_MAP.get(timeframe, "4h")
    tf_ms      = _TIMEFRAME_MS.get(timeframe, 14_400_000)
    start_ms   = _to_ms(start_date)
    end_ms     = _to_ms(end_date)
    batch_size = 500

    records: list[dict] = []
    current_ms = start_ms

    while current_ms < end_ms:
        batch = client.futures_open_interest_hist(
            symbol=symbol,
            period=period,
            startTime=current_ms,
            endTime=min(current_ms + batch_size * tf_ms, end_ms),
            limit=batch_size,
        )
        if not batch:
            break
        for item in batch:
            ts = datetime.fromtimestamp(
                item["timestamp"] / 1000, tz=timezone.utc
            ).replace(tzinfo=None)
            records.append({
                "timestamp":      ts,
                "open_interest":  float(item["sumOpenInterest"]),
            })
        current_ms = batch[-1]["timestamp"] + 1
        time.sleep(0.1)

    if not records:
        empty = pd.Series(dtype=float, name="open_interest")
        return empty

    df = pd.DataFrame(records)
    df.to_parquet(cache, index=False)
    s = df.set_index("timestamp")["open_interest"]
    s.name = "open_interest"
    return s
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_oi_fetcher.py -v 2>&1 | tail -10
```

Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add crypto_bot/core/data/binance_history.py tests/test_oi_fetcher.py
git commit -m "feat: add fetch_open_interest to binance_history with parquet caching

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 7: OpenInterestDivergenceStrategy

**Files:**
- Create: `crypto_bot/core/signals/strategies/open_interest_divergence.py`
- Modify: `tests/test_strategies.py` (append test class)

- [ ] **Step 1: Write the failing tests**

Append to end of `tests/test_strategies.py`:

```python
from crypto_bot.core.signals.strategies.open_interest_divergence import OpenInterestDivergenceStrategy


def _make_oi_aux(candles: pd.DataFrame, direction: str = "rising") -> dict:
    """Synthetic OI series aligned to candle timestamps."""
    n = len(candles)
    np.random.seed(7)
    if direction == "rising":
        oi = 1_000_000.0 + np.arange(n) * 3000 + np.random.randn(n) * 500
    else:
        oi = 1_000_000.0 - np.arange(n) * 1000 + np.random.randn(n) * 500
    return {"open_interest": pd.Series(oi.tolist(), index=candles["timestamp"].tolist())}


class TestOpenInterestDivergenceStrategy:
    _p = {
        "oi_change_period": 5, "oi_surge_pct": 1.0,
        "rsi_period": 14, "atr_period": 14,
    }

    def test_produces_long_when_oi_rising_price_falling(self):
        df = make_candles(300, trend="down")
        aux = _make_oi_aux(df, direction="rising")
        sigs = OpenInterestDivergenceStrategy(self._p).generate_signals(df, aux)
        assert len([s for s in sigs if s.direction == "LONG"]) > 0

    def test_produces_short_when_oi_rising_price_rising(self):
        df = make_candles(300, trend="up")
        aux = _make_oi_aux(df, direction="rising")
        sigs = OpenInterestDivergenceStrategy(self._p).generate_signals(df, aux)
        assert len([s for s in sigs if s.direction == "SHORT"]) > 0

    def test_no_signals_short_data(self):
        df = make_candles(10, trend="up")
        aux = _make_oi_aux(df, direction="rising")
        assert OpenInterestDivergenceStrategy(self._p).generate_signals(df, aux) == []

    def test_valid_directions(self):
        df = make_candles(300, trend="up")
        aux = _make_oi_aux(df, direction="rising")
        valid = {"LONG", "SHORT", "EXIT_LONG", "EXIT_SHORT"}
        for s in OpenInterestDivergenceStrategy(self._p).generate_signals(df, aux):
            assert s.direction in valid

    def test_default_params_in_space(self):
        s = OpenInterestDivergenceStrategy({})
        for k, spec in s.param_space.items():
            assert spec[0] <= s.default_params()[k] <= spec[1]

    def test_graceful_without_aux_data(self):
        df = make_candles(300, trend="up")
        result = OpenInterestDivergenceStrategy(self._p).generate_signals(df, None)
        assert result == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_strategies.py::TestOpenInterestDivergenceStrategy -v 2>&1 | tail -10
```

Expected: `ImportError` — module not found.

- [ ] **Step 3: Create the strategy file**

Create `crypto_bot/core/signals/strategies/open_interest_divergence.py`:

```python
"""
Open Interest Divergence Strategy.
LONG : OI rising + price falling (shorts trapped → squeeze incoming) + RSI < 45.
SHORT: OI rising + price rising  (longs trapped → unwind incoming)  + RSI > 55.
Exit : OI drops sharply (positions unwinding).
Requires aux_data['open_interest'] — pd.Series with timestamp index.
Returns [] gracefully if aux_data is absent.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from crypto_bot.core.signals.base import BaseStrategy
from crypto_bot.core.signals.models import Signal
from crypto_bot.core.signals.indicators import rsi, atr


class OpenInterestDivergenceStrategy(BaseStrategy):
    @property
    def param_space(self) -> dict:
        return {
            "oi_change_period": (3,   10,  "int"),
            "oi_surge_pct":     (0.5, 3.0),
            "rsi_period":       (10,  21,  "int"),
            "atr_period":       (10,  20,  "int"),
        }

    def generate_signals(self, candles: pd.DataFrame, aux_data=None) -> list[Signal]:
        # Graceful degradation — no OI data, no signals
        if aux_data is None or "open_interest" not in aux_data:
            return []

        p      = self.params
        close  = candles["close"]
        high   = candles["high"]
        low    = candles["low"]
        symbol = str(candles["symbol"].iloc[0])

        oi_period  = int(p["oi_change_period"])
        surge_pct  = float(p["oi_surge_pct"]) / 100.0
        rsi_p      = int(p["rsi_period"])
        atr_p      = int(p["atr_period"])

        rsi_v  = rsi(close, rsi_p)
        atr_v  = atr(high, low, close, atr_p)

        # Align OI to candle timestamps via forward-fill
        oi_raw: pd.Series = aux_data["open_interest"]
        oi_aligned = oi_raw.reindex(candles["timestamp"].tolist(), method="ffill")

        warmup = oi_period + rsi_p + 1
        signals: list[Signal] = []
        open_pos: str | None  = None

        for i in range(warmup, len(candles)):
            if not candles["is_clean"].iloc[i]:
                continue

            rsi_i  = rsi_v.iloc[i]
            atr_i  = atr_v.iloc[i]
            cl_i   = close.iloc[i]
            cl_lb  = close.iloc[i - oi_period]

            oi_curr = oi_aligned.iloc[i]
            oi_prev = oi_aligned.iloc[i - oi_period]

            if any(np.isnan(v) for v in (rsi_i, atr_i, oi_curr, oi_prev)):
                continue
            if oi_prev == 0:
                continue

            oi_change    = (oi_curr - oi_prev) / oi_prev
            price_change = (cl_i - cl_lb) / max(cl_lb, 1e-9)

            oi_rising    = oi_change > surge_pct
            oi_unwinding = oi_change < -surge_pct
            price_falling = price_change < -0.001
            price_rising  = price_change >  0.001

            direction: str | None = None
            reason: list[str]     = []

            if oi_rising and price_falling and rsi_i < 45 and open_pos != "LONG":
                direction = "LONG"
                reason    = ["OI_rising", "price_falling", f"RSI={rsi_i:.1f}"]
                open_pos  = "LONG"
            elif oi_rising and price_rising and rsi_i > 55 and open_pos != "SHORT":
                direction = "SHORT"
                reason    = ["OI_rising", "price_rising", f"RSI={rsi_i:.1f}"]
                open_pos  = "SHORT"
            elif open_pos == "LONG" and oi_unwinding:
                direction = "EXIT_LONG"
                reason    = ["OI_unwinding"]
                open_pos  = None
            elif open_pos == "SHORT" and oi_unwinding:
                direction = "EXIT_SHORT"
                reason    = ["OI_unwinding"]
                open_pos  = None

            if direction:
                signals.append(Signal(
                    strategy=self.name,
                    symbol=symbol,
                    timestamp=candles["timestamp"].iloc[i],
                    direction=direction,
                    strength=min(abs(oi_change) / max(surge_pct * 3, 1e-9), 1.0),
                    close_price=float(cl_i),
                    atr=float(atr_i),
                    reason=reason,
                ))

        return signals
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_strategies.py::TestOpenInterestDivergenceStrategy -v 2>&1 | tail -10
```

Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add crypto_bot/core/signals/strategies/open_interest_divergence.py tests/test_strategies.py
git commit -m "feat: OpenInterestDivergenceStrategy — OI vs price divergence with graceful aux_data fallback

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 8: Register All 5 Strategies

**Files:**
- Modify: `backtest/orchestrator.py`
- Modify: `main.py`

- [ ] **Step 1: Register in `backtest/orchestrator.py`**

In `_run_single_strategy`, add 5 entries to `strategy_module_map`:

```python
        "IchimokuCloudStrategy":          "crypto_bot.core.signals.strategies.ichimoku_cloud",
        "StochRSIStrategy":               "crypto_bot.core.signals.strategies.stoch_rsi",
        "MACDHistDivergenceStrategy":     "crypto_bot.core.signals.strategies.macd_hist_divergence",
        "MarketRegimeStrategy":           "crypto_bot.core.signals.strategies.market_regime",
        "OpenInterestDivergenceStrategy": "crypto_bot.core.signals.strategies.open_interest_divergence",
```

In `run_all`, extend `strategy_names`:

```python
    strategy_names = [
        "EMARibbonStrategy", "TTMSqueezeStrategy", "RSIDivergenceStrategy",
        "SupertrendADXStrategy", "BBMeanReversionStrategy", "FundingRateReversionStrategy",
        "DonchianBreakoutStrategy", "TSMOMStrategy", "HMAChandelierStrategy",
        "AdaptiveTrendStrategy", "VWAPBreakoutStrategy",
        "IchimokuCloudStrategy", "StochRSIStrategy", "MACDHistDivergenceStrategy",
        "MarketRegimeStrategy", "OpenInterestDivergenceStrategy",
    ]
```

- [ ] **Step 2: Register in `main.py`**

In `_run_single`, add 5 entries to `strategy_module_map`:

```python
        "IchimokuCloudStrategy":          "crypto_bot.core.signals.strategies.ichimoku_cloud",
        "StochRSIStrategy":               "crypto_bot.core.signals.strategies.stoch_rsi",
        "MACDHistDivergenceStrategy":     "crypto_bot.core.signals.strategies.macd_hist_divergence",
        "MarketRegimeStrategy":           "crypto_bot.core.signals.strategies.market_regime",
        "OpenInterestDivergenceStrategy": "crypto_bot.core.signals.strategies.open_interest_divergence",
```

- [ ] **Step 3: Verify imports work**

```bash
python -c "
from crypto_bot.core.signals.strategies.ichimoku_cloud import IchimokuCloudStrategy
from crypto_bot.core.signals.strategies.stoch_rsi import StochRSIStrategy
from crypto_bot.core.signals.strategies.macd_hist_divergence import MACDHistDivergenceStrategy
from crypto_bot.core.signals.strategies.market_regime import MarketRegimeStrategy
from crypto_bot.core.signals.strategies.open_interest_divergence import OpenInterestDivergenceStrategy
print('All 5 strategies import OK')
"
```

Expected: `All 5 strategies import OK`

- [ ] **Step 4: Commit**

```bash
git add backtest/orchestrator.py main.py
git commit -m "feat: register 5 new strategies in orchestrator and CLI dispatch map

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 9: Full Test Run + Push

**Files:** None (verification only)

- [ ] **Step 1: Run full test suite**

```bash
python -m pytest tests/ -v 2>&1 | tail -20
```

Expected output:
```
tests/test_new_indicators.py ...........                          [ xx%]
tests/test_oi_fetcher.py ....                                     [ xx%]
tests/test_strategies.py ...........................................[ xx%]
...
============================== 141 passed in X.XXs ==============================
```

If any test fails, check the error message and fix before proceeding.

- [ ] **Step 2: Verify all 5 strategies instantiate with default params**

```bash
python -c "
import sys
sys.path.insert(0, '.')
strategies = [
    ('IchimokuCloudStrategy',         'crypto_bot.core.signals.strategies.ichimoku_cloud'),
    ('StochRSIStrategy',              'crypto_bot.core.signals.strategies.stoch_rsi'),
    ('MACDHistDivergenceStrategy',    'crypto_bot.core.signals.strategies.macd_hist_divergence'),
    ('MarketRegimeStrategy',          'crypto_bot.core.signals.strategies.market_regime'),
    ('OpenInterestDivergenceStrategy','crypto_bot.core.signals.strategies.open_interest_divergence'),
]
import importlib
for cls_name, mod_path in strategies:
    mod = importlib.import_module(mod_path)
    cls = getattr(mod, cls_name)
    inst = cls.__new__(cls); inst.params = {}
    defaults = inst.default_params()
    space = inst.param_space
    for k, spec in space.items():
        assert spec[0] <= defaults[k] <= spec[1], f'{cls_name}.{k} default out of range'
    print(f'  {cls_name}: OK — {len(space)} params')
print('All strategies verified.')
"
```

Expected: each strategy prints `OK` with param count.

- [ ] **Step 3: Push to GitHub**

```bash
git push origin master
```

- [ ] **Step 4: Confirm final state**

```bash
git log --oneline | head -12
```

Expected: 9 new commits on top of the pre-existing history.
