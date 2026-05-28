# New Strategies Design — Ichimoku, StochRSI, OI Divergence, MACD Hist Divergence, Market Regime

**Date:** 2026-05-28
**Status:** Approved

---

## 1. Overview

Add 5 new trading strategies to the crypto backtesting system, bringing the total from 11 to 16 active strategies. All strategies follow the existing `BaseStrategy` pattern, integrate with Optuna auto-tuning via `param_space`, and are registered in the orchestrator and CLI dispatch map.

---

## 2. Architecture

### 2.1 File Layout

Each strategy is a self-contained module:

```
crypto_bot/core/signals/strategies/
    ichimoku_cloud.py           ← new
    stoch_rsi.py                ← new
    open_interest_divergence.py ← new
    macd_hist_divergence.py     ← new
    market_regime.py            ← new

crypto_bot/core/signals/indicators.py   ← new indicator helpers added
crypto_bot/core/data/binance_history.py ← fetch_open_interest() added
tests/test_strategies.py                ← 5 new test classes appended
backtest/orchestrator.py                ← 5 new entries in strategy_map + run_all
main.py                                 ← 5 new entries in CLI dispatch map
```

### 2.2 BaseStrategy Contract

Every strategy must:
- Inherit `BaseStrategy`
- Implement `param_space` → dict of `(min, max)` float or `(min, max, "int")` tuples
- Implement `generate_signals(candles: pd.DataFrame, aux_data=None) → list[Signal]`
- Skip candles where `is_clean == False`
- Respect warmup period (no signals before indicators are stable)
- Fire signals on candle **close** only (no lookahead)

### 2.3 Optuna Auto-Tuning

No extra wiring needed. The optimizer in `backtest/optimizer.py` reads `param_space` and calls `trial.suggest_int()` or `trial.suggest_float()` based on tuple arity. All 5 new strategies get 500 Optuna trials automatically once registered.

### 2.4 OI Data Pipeline

Open Interest data is fetched from Binance Futures `/fapi/v1/openInterestHist` endpoint, one symbol at a time, cached to `data/cache/oi_<symbol>_<timeframe>.parquet`, and injected into `aux_data["open_interest"]` as a `pd.Series` aligned to candle timestamps — identical to how funding rates work.

---

## 3. Strategy Designs

### 3.1 IchimokuCloudStrategy

**Type:** Trend + Support/Resistance

**Indicators:**
- Tenkan-sen (conversion line): `(highest_high + lowest_low) / 2` over `tenkan_period`
- Kijun-sen (base line): same over `kijun_period`
- Senkou Span A: `(tenkan + kijun) / 2` displaced 26 bars forward
- Senkou Span B: `(highest_high + lowest_low) / 2` over `senkou_b_period`, displaced 26 bars forward
- Chikou Span: current close plotted 26 bars back

**Entry LONG:** Price closes above both Senkou A and B (above cloud) + Tenkan crosses above Kijun + Chikou span above price 26 bars ago

**Entry SHORT:** Price closes below both Senkou A and B (below cloud) + Tenkan crosses below Kijun + Chikou span below price 26 bars ago

**Exit:** Tenkan crosses back through Kijun in opposite direction, or price closes inside the cloud

**param_space:**
```python
{
    "tenkan_period":   (7,  12,  "int"),
    "kijun_period":    (22, 30,  "int"),
    "senkou_b_period": (44, 60,  "int"),
    "atr_period":      (10, 20,  "int"),
}
```
Note: `displacement` is fixed at 26 (standard Ichimoku convention, not tuned).

**Warmup:** `senkou_b_period + displacement + 1`

---

### 3.2 StochRSIStrategy

**Type:** Oscillator crossover

**Indicator:** Stochastic RSI — apply Stochastic formula to RSI values:
```
RSI = rsi(close, rsi_period)
StochRSI = (RSI - min(RSI, stoch_period)) / (max(RSI, stoch_period) - min(RSI, stoch_period))
%K = SMA(StochRSI, smooth_k)
%D = SMA(%K, smooth_d)
```

**Entry LONG:** %K crosses above %D while both are below 20 (oversold zone)

**Entry SHORT:** %K crosses below %D while both are above 80 (overbought zone)

**Exit:** %K/%D cross back in the opposite direction, or both enter neutral zone (40–60)

**param_space:**
```python
{
    "rsi_period":   (10, 21, "int"),
    "stoch_period": (10, 21, "int"),
    "smooth_k":     (2,  5,  "int"),
    "smooth_d":     (2,  5,  "int"),
    "atr_period":   (10, 20, "int"),
}
```

**Warmup:** `rsi_period + stoch_period + smooth_k + smooth_d + 1`

---

### 3.3 OpenInterestDivergenceStrategy

**Type:** Crypto-specific OI divergence

**Data source:** `aux_data["open_interest"]` — a `pd.Series` of OI values aligned to candle timestamps, fetched from Binance Futures `/fapi/v1/openInterestHist`.

**Signal logic:**
- Compute `oi_change` = percentage change in OI over `oi_change_period` bars
- **Entry LONG:** OI rising (oi_change > oi_surge_pct%) + price falling or flat + RSI < 45 → shorts trapped, squeeze incoming
- **Entry SHORT:** OI rising (oi_change > oi_surge_pct%) + price rising + RSI > 55 + momentum slowing → longs trapped
- **Exit:** OI drops sharply (oi_change turns negative by threshold) or price reverses by 1× ATR

**Graceful degradation:** If `aux_data` is None or `"open_interest"` key missing, return `[]` rather than raising.

**param_space:**
```python
{
    "oi_change_period": (3,   10,  "int"),
    "oi_surge_pct":     (0.5, 3.0),
    "rsi_period":       (10,  21,  "int"),
    "atr_period":       (10,  20,  "int"),
}
```

**Warmup:** `oi_change_period + rsi_period + 1`

#### Data Fetcher Extension

New method on `BinanceHistoryFetcher`:
```python
def fetch_open_interest(
    self,
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
) -> pd.Series:
    ...
```
- Calls `/fapi/v1/openInterestHist` with `period` mapped from timeframe
- Caches result to `data/cache/oi_{symbol}_{timeframe}.parquet`
- Returns `pd.Series` with `timestamp` index, `sumOpenInterest` values as floats
- Called by orchestrator worker alongside existing OHLCV + funding fetch

---

### 3.4 MACDHistDivergenceStrategy

**Type:** Histogram divergence

**Indicators:** Standard MACD — fast EMA, slow EMA, signal line, histogram

**Signal logic:**
- Scan back `divergence_lookback` bars for swing highs/lows
- **Entry LONG (bullish divergence):** Price makes a lower low AND histogram makes a higher low AND histogram is now turning positive (rising from negative)
- **Entry SHORT (bearish divergence):** Price makes a higher high AND histogram makes a lower high AND histogram is turning negative (falling from positive)
- **Exit:** Histogram crosses zero in the opposite direction

**param_space:**
```python
{
    "macd_fast":           (8,  16, "int"),
    "macd_slow":           (20, 30, "int"),
    "macd_signal":         (7,  12, "int"),
    "divergence_lookback": (3,  10, "int"),
    "atr_period":          (10, 20, "int"),
}
```

**Warmup:** `macd_slow + macd_signal + divergence_lookback + 1`

---

### 3.5 MarketRegimeStrategy

**Type:** Standalone regime-detection strategy

**Regime detection logic:**
```
ADX > adx_trend_threshold AND EMA slope > 0  → UPTREND  regime → LONG signal
ADX > adx_trend_threshold AND EMA slope < 0  → DOWNTREND regime → SHORT signal
ADX < adx_range_threshold                    → RANGING  regime → no signal / exit
ATR_stddev > atr_vol_threshold × ATR_mean    → VOLATILE regime → no signal / exit
```

EMA slope = `(ema[i] - ema[i - slope_lookback]) / ema[i - slope_lookback]`

**Entry:** On regime transition into UPTREND (→ LONG) or DOWNTREND (→ SHORT)

**Exit:** On regime transition into RANGING or VOLATILE (emit EXIT signal)

**param_space:**
```python
{
    "adx_period":            (10,  20,  "int"),
    "adx_trend_threshold":   (20.0, 30.0),
    "adx_range_threshold":   (15.0, 25.0),
    "ema_period":            (50,  200, "int"),
    "slope_lookback":        (3,   10,  "int"),
    "atr_vol_threshold":     (1.5, 3.0),
    "atr_period":            (10,  20,  "int"),
}
```

**Warmup:** `ema_period + slope_lookback + 1`

---

## 4. New Indicator Helpers (`indicators.py`)

| Function | Used by |
|---|---|
| `ichimoku(high, low, tenkan, kijun, senkou_b)` → 5 Series | IchimokuCloudStrategy |
| `stoch_rsi(close, rsi_period, stoch_period, smooth_k, smooth_d)` → `(%K, %D)` | StochRSIStrategy |
| `macd_histogram(close, fast, slow, signal)` → Series | MACDHistDivergenceStrategy (reuses existing `macd()`) |
| `ema_slope(close, period, lookback)` → Series | MarketRegimeStrategy |

---

## 5. Testing

Each strategy gets a test class in `tests/test_strategies.py` with these 5 tests:

| Test | Verifies |
|---|---|
| `test_produces_long_in_uptrend` | At least one LONG signal on 300-bar uptrend synthetic data |
| `test_produces_short_in_downtrend` | At least one SHORT signal on 300-bar downtrend synthetic data |
| `test_no_signals_short_data` | Empty list when data length < warmup |
| `test_valid_directions` | All signal directions in `{"LONG", "SHORT", "EXIT_LONG", "EXIT_SHORT"}` |
| `test_default_params_in_space` | Default params within declared `param_space` bounds |

`OpenInterestDivergenceStrategy` gets one additional test:
- `test_graceful_without_aux_data` → returns `[]` when `aux_data=None`

**Synthetic data helper:** Uses existing `make_candles(n=300, trend="up"|"down")` from conftest.

---

## 6. Registration

After all 5 strategies are built, update:

**`backtest/orchestrator.py`** — `strategy_module_map` dict + `strategy_names` list in `run_all()`

**`main.py`** — `strategy_module_map` dict in `_run_single()`

No changes needed to `RiskEngine`, `PaperExecutionEngine`, `StateStore`, `Reporter`, `WalkForwardValidator`, or `Optimizer` — they are strategy-agnostic.

---

## 7. Implementation Order

Parallel execution via 5 independent subagents:
1. Agent 1 → `ichimoku_cloud.py` + indicators + tests
2. Agent 2 → `stoch_rsi.py` + indicators + tests
3. Agent 3 → `open_interest_divergence.py` + indicators + tests + fetcher extension
4. Agent 4 → `macd_hist_divergence.py` + indicators + tests
5. Agent 5 → `market_regime.py` + indicators + tests

Sequential final pass:
6. Register all 5 in `orchestrator.py` and `main.py`
7. Run full test suite → confirm 115 + 26 = 141 tests pass
8. Commit all changes

---

## 8. Success Criteria

- [ ] All 5 strategy files created and importable
- [ ] `param_space` defined with valid Optuna-compatible tuples for all 5
- [ ] OI fetcher extended and caches to parquet
- [ ] 141 tests pass (115 existing + 26 new)
- [ ] All 5 registered in orchestrator and CLI
- [ ] `python main.py --mode backtest --strategy IchimokuCloudStrategy` runs end-to-end
