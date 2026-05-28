# Crypto Trading Bot — Design Spec
**Date:** 2026-05-24
**Status:** Phase 1 scoped (Backtest + Auto-Optimization Engine)

---

## Overview

An async, event-driven Python trading bot targeting Binance Futures. Runs **7 independent strategies** in parallel, each on separate capital allocation, each auto-tuned via Optuna Bayesian optimization if initial backtest fails profitability thresholds. Supports long and short positions with configurable leverage cap.

**Execution model:** In backtest mode, a synchronous pipeline per strategy. In paper/live modes, engines connect via asyncio queues. Each engine layer only knows its own input/output models — no cross-imports between layers.

---

## Phased Delivery Plan

| Phase | Scope | Gate to proceed |
|-------|-------|-----------------|
| 1 | Backtest + auto-optimization engine for all 7 strategies | All strategies evaluated; ≥1 strategy passes promotion criteria on out-of-sample data |
| 2 | Paper trading (live Binance Futures data, simulated execution) | 2 weeks paper; live metrics within 15% of backtest metrics |
| 3 | Live execution (real Binance Futures orders, small capital, leverage ≤ configured cap) | — |
| 4 | Web dashboard + Telegram bot | — |

---

## Full Architecture (all phases)

```
StrategyOrchestrator
│
├── [Strategy 1] EMARibbon          ──┐
├── [Strategy 2] TTMSqueeze           │  ProcessPoolExecutor
├── [Strategy 3] RSIHiddenDivergence  │  (each strategy = 1 process)
├── [Strategy 4] SupertrendADX        ├─────────────────────────┐
├── [Strategy 5] BBMeanReversion      │                         │
├── [Strategy 6] FundingRateReversion │                         │
└── [Strategy 7] XGBoostMeta        ──┘                         │
                                                                 ▼
Per-strategy pipeline (identical for all):             PortfolioAggregator
  DataEngine → SignalEngine → RiskEngine → PaperExecution    (final report)
       ↕                                        ↕
  StateStore (SQLite per strategy)    OptunaOptimizer (if needed)

Mode flag (backtest | paper | live) selects DataEngine + ExecutionEngine adapters.
SignalEngine and RiskEngine are identical across all modes.
```

---

## Project Structure

```
crypto_bot/
├── config/
│   └── settings.yaml                  # Global config: symbols, dates, capital, leverage
├── core/
│   ├── data/
│   │   ├── binance_history.py         # Historical OHLCV + funding rate fetcher (Phase 1)
│   │   ├── binance_live.py            # WebSocket + REST adapter (Phase 2+)
│   │   ├── validator.py               # Data quality checks (gaps, outliers, anomalies)
│   │   └── models.py                  # Candle, FundingRate pydantic models
│   ├── signals/
│   │   ├── base.py                    # BaseStrategy ABC with param_space interface
│   │   ├── indicators.py              # All TA indicators as pure functions
│   │   ├── strategies/
│   │   │   ├── ema_ribbon.py
│   │   │   ├── ttm_squeeze.py
│   │   │   ├── rsi_divergence.py
│   │   │   ├── supertrend_adx.py
│   │   │   ├── bb_mean_reversion.py
│   │   │   ├── funding_rate_reversion.py
│   │   │   └── xgboost_meta.py
│   │   └── models.py                  # Signal(symbol, direction, strength, timestamp)
│   ├── risk/
│   │   ├── engine.py                  # SL/TP, drawdown guard, correlation filter, leverage
│   │   └── models.py                  # TradeOrder, RiskDecision
│   ├── execution/
│   │   ├── paper.py                   # Simulated fills at next-candle open + fees + slippage
│   │   ├── live.py                    # Binance Futures REST order placement (Phase 3)
│   │   └── models.py                  # Fill, Position, Portfolio
│   └── state/
│       ├── store.py                   # SQLite per strategy (aiosqlite)
│       └── schema.sql
├── backtest/
│   ├── orchestrator.py                # ProcessPoolExecutor — runs all strategies in parallel
│   ├── runner.py                      # Single-strategy backtest loop
│   ├── optimizer.py                   # Optuna study per strategy (500 trials, TPE)
│   ├── walk_forward.py                # Rolling window in-sample/out-of-sample validation
│   ├── reporter.py                    # Per-strategy metrics + portfolio aggregation
│   └── sensitivity.py                 # Parameter sensitivity analysis post-tuning
├── dashboard/                         # Phase 4
├── bot/                               # Phase 4
├── data/
│   ├── cache/                         # Parquet files: candles per symbol/timeframe
│   └── studies/                       # Optuna SQLite studies per strategy
├── results/                           # Backtest run results (versioned by timestamp+git hash)
├── main.py                            # CLI: --mode backtest|paper|live --config ...
└── tests/
    ├── test_indicators.py
    ├── test_strategies.py
    ├── test_risk.py
    ├── test_optimizer.py
    └── test_backtest_runner.py
```

---

## Phase 1: Detailed Design

### 1.1 Configuration (`config/settings.yaml`)

```yaml
exchange:
  name: binance
  market: futures              # spot | futures
  testnet: false

backtest:
  symbols: ["BTCUSDT", "ETHUSDT"]
  timeframe: "4h"
  start_date: "2020-01-01"     # 5 years: covers COVID crash, 2021 bull, FTX, 2023-24 recovery
  end_date:   "2024-12-31"
  initial_capital_per_strategy: 10000   # USDT — each strategy gets its own allocation
  fees_pct: 0.001              # 0.1% per trade (Binance taker fee)
  slippage_pct: 0.0005         # 0.05% slippage per fill

leverage:
  max: 3                       # Hard cap — never exceeded regardless of strategy
  default: 1                   # Start unleveraged; strategies can request up to max

optimization:
  trials: 500
  sampler: "TPE"               # Tree-structured Parzen Estimator (Bayesian)
  objective: "composite"       # sharpe × (1 - max_drawdown) × profit_factor
  min_trades_per_year: 30      # Penalty applied if below this
  study_storage: "data/studies/{strategy_name}.db"

walk_forward:
  train_months: 12
  test_months: 3
  step_months: 3               # Roll every 3 months

promotion_criteria:
  min_sharpe_oos: 1.0          # Out-of-sample Sharpe
  max_drawdown_pct: 0.25
  min_profit_factor: 1.3
  min_trades_per_year: 30
  max_is_oos_divergence: 2.0   # In-sample Sharpe must be < 2× out-of-sample (robustness)
```

### 1.2 Data Engine — Historical Fetcher

**File:** `core/data/binance_history.py`

- Fetches OHLCV from Binance REST (`/fapi/v1/klines` for futures) in paginated batches
- Fetches funding rates (`/fapi/v1/fundingRate`) for FundingRateReversion strategy
- Caches to `data/cache/<symbol>_<timeframe>_<start>_<end>.parquet` (via pandas + pyarrow)
- Cache invalidation: keyed on `(symbol, timeframe, start_date, end_date)` — change any key → re-fetch

**Data validator** (`core/data/validator.py`):
- Detects and fills gaps (missing candles) with forward-fill or flags as untradeable window
- Detects price anomalies (candle range > 5× ATR → flag, don't trade)
- Reports data quality score per symbol (% clean candles)

**Models:**
```python
class Candle(BaseModel):
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    is_clean: bool = True      # False if flagged by validator

class FundingRate(BaseModel):
    symbol: str
    timestamp: datetime
    rate: float                # e.g. 0.0001 = 0.01%
```

### 1.3 Strategy Interface (BaseStrategy)

**File:** `core/signals/base.py`

Every strategy implements this contract:

```python
class BaseStrategy(ABC):
    def __init__(self, params: dict): ...

    @abstractmethod
    def generate_signals(self, candles: pd.DataFrame, aux_data: dict) -> list[Signal]:
        """Pure function — no side effects. aux_data holds funding rates, etc."""

    @property
    @abstractmethod
    def param_space(self) -> dict[str, tuple]:
        """Optuna search space: {"rsi_period": (8, 30), "threshold": (20, 40, "int")}"""

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def default_params(self) -> dict:
        """Returns midpoint of param_space as sensible defaults."""
```

Signals are generated on the **closed candle** only. Fill prices use **next candle's open** to prevent lookahead bias.

### 1.4 The 7 Strategies

---

#### Strategy 1: EMA Ribbon Trend Follow

**Logic:** 4 EMAs (fast/medium/slow/trend). Entry when all 4 align in order (bullish or bearish stack). Exit when fastest EMA crosses back through second EMA.

**Param space:**
```python
{"ema_fast": (5, 15), "ema_medium": (15, 30), "ema_slow": (30, 60), "ema_trend": (100, 250)}
```

**Best regime:** Trending markets. ADX > 20 filter applied.

---

#### Strategy 2: TTM Squeeze (LazyBear)

**Logic:** Bollinger Bands inside Keltner Channels = market in compression. When BB expands outside KC = momentum release. Enter in direction of momentum histogram. Exit when histogram reverses.

**Param space:**
```python
{"bb_period": (15, 25), "bb_std": (1.5, 2.5), "kc_period": (15, 25), "kc_mult": (1.0, 2.0), "mom_period": (10, 20)}
```

**Best regime:** Pre-breakout compression phases (common after crypto consolidations).

---

#### Strategy 3: RSI Hidden Divergence

**Logic:** Hidden bullish divergence = price makes higher low, RSI makes lower low → continuation long. Hidden bearish divergence = price lower high, RSI higher high → continuation short. Confirms with MACD direction.

**Param space:**
```python
{"rsi_period": (8, 21), "divergence_lookback": (3, 10), "macd_fast": (8, 16), "macd_slow": (20, 30), "macd_signal": (7, 12)}
```

**Best regime:** Trending with pullbacks. High win rate on swing timeframes.

---

#### Strategy 4: Supertrend + ADX Filter

**Logic:** Supertrend (ATR-based) gives trend direction. ADX confirms trend strength. Only enter when ADX > threshold (strong trend). Exit when Supertrend flips.

**Param space:**
```python
{"atr_period": (7, 21), "atr_multiplier": (1.5, 4.0), "adx_period": (10, 20), "adx_threshold": (20, 35)}
```

**Best regime:** Strong trending markets. Avoids whipsaw in low-ADX choppy conditions.

---

#### Strategy 5: Bollinger Band Mean Reversion

**Logic:** Price touches lower band (oversold) + RSI < threshold + BB width below squeeze threshold = mean reversion long. Reverse for short. Target: middle band. SL: outside band.

**Param space:**
```python
{"bb_period": (15, 30), "bb_std": (1.8, 2.5), "rsi_period": (8, 21), "rsi_threshold": (25, 40), "bb_width_threshold": (0.02, 0.08)}
```

**Best regime:** Ranging/low-volatility markets. Underperforms in strong trends.

---

#### Strategy 6: Funding Rate Mean Reversion

**Logic:** Extreme positive funding rate (longs paying shorts heavily) = crowded long trade → go short. Extreme negative funding rate → go long. Confirm with RSI divergence from extreme. Exit when funding normalizes or RSI reverses.

**Param space:**
```python
{"funding_threshold": (0.0005, 0.003), "rsi_period": (8, 21), "rsi_confirm_long": (20, 45), "rsi_confirm_short": (55, 80), "exit_funding_pct": (0.1, 0.5)}
```

**Best regime:** High-leverage periods, late bull/bear cycle extremes. Genuinely uncorrelated with price-signal strategies.

---

#### Strategy 7: XGBoost Meta-Classifier

**Logic:** Does NOT generate its own entry signals and does NOT allocate separate capital. Instead, it acts as a **go/no-go gate** on top of Strategies 1–6. Each of the other 6 strategies submits a proposed signal; the XGBoost model predicts whether the current market regime is favorable for that signal. If the model says no, the signal is suppressed — no trade is taken. Capital for each strategy is unchanged; Strategy 7 only reduces trade count, never increases it.

**Features:** RSI, MACD histogram, BB %B, ATR %, ADX, volume z-score, funding rate, day-of-week, hour-of-day, signals from each strategy (0/1/-1), recent win rate of each strategy.

**Target:** 1 if next N candles return > fee threshold, else 0.

**Param space:**
```python
{"n_estimators": (50, 300), "max_depth": (3, 8), "learning_rate": (0.01, 0.3), "subsample": (0.6, 1.0), "forward_window": (3, 12)}
```

**Training:** Walk-forward re-trained on each in-sample window (no leakage). Out-of-sample predictions only.

---

### 1.5 Risk Engine

**File:** `core/risk/engine.py`

Stateful per strategy instance — each strategy's risk engine tracks that strategy's own positions and equity.

**Responsibilities:**

1. **Position sizing (fixed-fraction + leverage)**
   ```
   notional = capital × max_position_pct
   leveraged_notional = notional × leverage_requested
   quantity = leveraged_notional / entry_price
   ```
   Leverage is capped at `settings.leverage.max` regardless of strategy request.

2. **SL/TP (ATR-based)**
   - Long: `SL = entry - atr × sl_mult`, `TP = entry + atr × tp_mult`
   - Short: `SL = entry + atr × sl_mult`, `TP = entry - atr × tp_mult`

3. **Drawdown guard:** `(peak_equity - current_equity) / peak_equity > max_drawdown_pct` → halt new entries for this strategy.

4. **Correlation guard (within strategy):** For multi-symbol strategies, prevent opening correlated positions simultaneously (BTC + ETH correlation > 0.75 = reject second entry).

**Output:**
```python
class RiskDecision(BaseModel):
    approved: bool
    symbol: str
    direction: Literal["LONG", "SHORT"]
    entry_price: float
    quantity: float
    leverage: float
    stop_loss: float
    take_profit: float
    rejection_reason: str | None
```

### 1.6 Execution Engine — Paper (Simulated with Costs)

**File:** `core/execution/paper.py`

- Fill at **next candle's open price** (no lookahead bias)
- Apply fees: `fill_price × fees_pct` per side (entry + exit)
- Apply slippage: `fill_price × (1 + slippage_pct)` for longs, `× (1 - slippage_pct)` for shorts
- Check SL/TP hit within each candle using high/low (conservative: assume worst case on the candle)
- Handle liquidation: if candle low/high breaches `entry × (1 - 1/leverage)` → liquidation event, record as -100% loss on position

```python
class Fill(BaseModel):
    strategy: str
    symbol: str
    timestamp: datetime
    direction: Literal["LONG", "SHORT"]
    entry_price: float
    exit_price: float | None
    quantity: float
    leverage: float
    fees_paid: float
    slippage_paid: float
    pnl: float | None
    exit_reason: Literal["sl", "tp", "signal", "liquidation", "end_of_data"] | None
```

### 1.7 Backtest Orchestrator

**File:** `backtest/orchestrator.py`

Runs all 7 strategies in parallel using `concurrent.futures.ProcessPoolExecutor`. Each worker process runs the full pipeline for one strategy independently.

```python
with ProcessPoolExecutor(max_workers=7) as pool:
    futures = {pool.submit(run_strategy_pipeline, strategy, config): strategy
               for strategy in registry.all_strategies()}
    results = {strategy: future.result() for strategy, future in futures.items()}
```

Each worker:
1. Loads cached candle data
2. Runs initial backtest with default params
3. Evaluates against promotion criteria
4. If failing → triggers Optuna optimizer (500 trials)
5. Re-runs with best params
6. Runs walk-forward validation
7. Runs parameter sensitivity analysis
8. Returns `StrategyResult` (metrics, params, equity curve, trade list, promotion status)

### 1.8 Optimizer

**File:** `backtest/optimizer.py`

```python
def optimize(strategy_cls, candles, config) -> dict:
    storage = f"sqlite:///data/studies/{strategy_cls.name}.db"
    study = optuna.create_study(
        study_name=strategy_cls.name,
        storage=storage,
        load_if_exists=True,         # Resume if interrupted
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=20)
    )
    study.optimize(objective, n_trials=500, n_jobs=1, show_progress_bar=True)
    return study.best_params
```

**Objective function:**
```python
def objective(trial) -> float:
    params = sample_from_param_space(trial, strategy.param_space)
    metrics = run_backtest(strategy_cls(params), candles, config)

    if metrics.trades_per_year < config.min_trades_per_year:
        return -999   # Penalize low-frequency strategies

    score = (metrics.sharpe_ratio
             * (1 - metrics.max_drawdown_pct)
             * metrics.profit_factor)
    return score
```

Pruner eliminates unpromising trials early (after 20 startup trials), saving ~40% of compute.

### 1.9 Walk-Forward Validator

**File:** `backtest/walk_forward.py`

```
Full data: 2020-01-01 → 2024-12-31 (60 months)

Window 1:  [TRAIN: Jan20–Dec20 (12mo)] [TEST: Jan21–Mar21 (3mo)]
Window 2:  [TRAIN: Apr20–Mar21 (12mo)] [TEST: Apr21–Jun21 (3mo)]
Window 3:  [TRAIN: Jul20–Jun21 (12mo)] [TEST: Jul21–Sep21 (3mo)]
...
Window 16: [TRAIN: Jan23–Dec23 (12mo)] [TEST: Jan24–Mar24 (3mo)]
```

For each window: tune params on TRAIN, evaluate on TEST. Collect all OOS test results into a combined equity curve and metrics. This is the honest performance estimate.

Reports:
- OOS Sharpe, OOS max drawdown, OOS profit factor
- Consistency score: % of test windows that were profitable
- IS vs OOS Sharpe ratio (divergence check)

### 1.10 Parameter Sensitivity Analyzer

**File:** `backtest/sensitivity.py`

After finding best params, perturb each param by ±10% and ±20%, re-run backtest, measure score change. Integer params (e.g., RSI period = 14) are rounded to nearest integer after perturbation.

```
RSI period: 14 (best)
  +10% (15.4 → 15): score change -3%   → ROBUST
  -10% (12.6 → 13): score change -8%   → ROBUST
  +20% (16.8 → 17): score change -15%  → ACCEPTABLE
  -20% (11.2 → 11): score change -45%  → BRITTLE ⚠️
```

Strategies where any param change > 20% causes score drop > 30% are flagged as **BRITTLE** and rejected.

### 1.11 Performance Reporter

**File:** `backtest/reporter.py`

**Per-strategy report:**

| Metric | Detail |
|--------|--------|
| Total return % | Net of fees and slippage |
| Sharpe ratio (IS + OOS) | Annualized, `√252 × mean/std daily returns` |
| Sortino ratio | Downside deviation only |
| Max drawdown % | Over full equity curve |
| Max drawdown duration | Longest underwater period (days) |
| Win rate | Winning trades / total |
| Profit factor | Gross profit / gross loss |
| Avg trade duration | Mean candles held |
| Trades per year | Statistical significance check |
| Fees paid (total $) | Real cost of trading |
| Slippage paid (total $) | Execution cost |
| Sensitivity result | ROBUST / BRITTLE |
| Promotion status | PROMOTE / REJECT + reason |

**Portfolio aggregation report:**
- Side-by-side table of all 7 strategies ranked by OOS Sharpe
- Correlation matrix of daily returns (flag pairs > 0.8 as redundant)
- Combined equity curve with equal capital allocation
- Combined Sharpe, combined drawdown
- Recommended capital allocation weights (proportional to OOS Sharpe)

---

## Data Flow (Backtest Mode)

```
settings.yaml
     │
     ▼
BinanceHistoryFetcher (+ FundingRates)
     │
     ▼
DataValidator → clean candles
     │
     ▼ (per strategy, in parallel)
StrategyOrchestrator
     │
     ├── [Strategy N] ──▶ SignalEngine(strategy, candles)
     │                         │ Signal
     │                         ▼
     │                   RiskEngine ◀─── Portfolio state (per strategy)
     │                         │ RiskDecision
     │                         ▼
     │                   PaperExecution (fees + slippage + liquidation check)
     │                         │ Fill
     │                         ▼
     │                   StateStore (SQLite :memory: per strategy)
     │                         │
     │                   [if metrics fail thresholds]
     │                         ▼
     │                   OptunaOptimizer (500 trials, resume-able)
     │                         │ best_params
     │                         ▼
     │                   Re-run BacktestRunner
     │                         │
     │                   WalkForwardValidator
     │                         │
     │                   SensitivityAnalyzer
     │                         │ StrategyResult
     │
     ▼
PortfolioAggregator
     │
     ▼
Terminal report + CSV exports + equity curve plots
```

---

## Testing Strategy

- **Indicators:** Pure functions — known-input/output tests (flat series RSI=50, etc.)
- **Strategies:** Synthetic candle series with planted patterns → assert expected signals
- **Risk engine:** Unit tests for each guard in isolation (sizing math, SL/TP, drawdown, leverage cap)
- **Optimizer:** Mock backtest runner returning fixed scores → assert Optuna selects correct params
- **Walk-forward:** Synthetic 24-month dataset → assert correct window boundaries and no data leakage
- **Integration:** Full pipeline on 90 days of real cached data → assert no lookahead bias, equity curve matches manual trace

---

## Dependencies (Phase 1)

Already installed: `pandas`, `numpy`, `pydantic`, `aiohttp`, `requests`, `tabulate`

Add:
```
python-binance    # Binance REST (spot + futures)
aiosqlite         # Async SQLite
optuna            # Bayesian hyperparameter optimization
xgboost           # Strategy 7 classifier
scikit-learn      # Feature scaling, train/test split for XGBoost
matplotlib        # Equity curve plots
pyarrow           # Parquet cache read/write
pytest            # Tests
pytest-asyncio    # Async test support
pyyaml            # Config loading
```

---

## Out of Scope (Phase 1)

- Live Binance WebSocket
- Real order placement
- Web dashboard
- Telegram bot
- NATS messaging
- Trailing stop-loss (Phase 2)
- Multi-timeframe analysis (Phase 2)
- On-chain data (SOPR, MVRV) — potential Phase 3 enhancement
