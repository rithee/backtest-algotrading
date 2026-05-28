# Backtest Engine — Complete Reference

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [How the Engine Works](#2-how-the-engine-works)
3. [Configuration Reference](#3-configuration-reference)
4. [Strategies](#4-strategies)
5. [Running the Backtest](#5-running-the-backtest)
6. [Understanding Results](#6-understanding-results)
7. [Optimization (Optuna)](#7-optimization-optuna)
8. [Walk-Forward Validation](#8-walk-forward-validation)
9. [Parameter Sensitivity Analysis](#9-parameter-sensitivity-analysis)
10. [Data Pipeline](#10-data-pipeline)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Quick Start

### Linux / macOS

```bash
# Install dependencies
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-linux.txt

# Run full backtest (fetches data automatically if cache is empty)
python main.py

# Run a single strategy only (fastest — good first test)
python main.py --strategy EMARibbonStrategy --no-optimize --no-walk-forward

# Run a single strategy with full pipeline (optimize + walk-forward)
python main.py --strategy IchimokuCloudStrategy

# Run without optimization (uses default params, much faster)
python main.py --no-optimize

# Run single-threaded (debug mode)
python main.py --no-parallel --no-optimize --no-walk-forward
```

### Windows (Command Prompt or PowerShell)

```cmd
:: Step 1 — Create virtual environment
python -m venv .venv

:: Step 2 — Activate (Command Prompt)
.venv\Scripts\activate.bat

:: Step 2 — Activate (PowerShell — if above doesn't work)
::   If you get "scripts cannot be run", first run:
::   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.venv\Scripts\Activate.ps1

:: Step 3 — Install dependencies (Windows-specific file — avoids Linux CUDA packages)
pip install -r requirements-windows.txt

:: Step 4 — Run (use --no-parallel on Windows to avoid multiprocessing issues)
python main.py --no-parallel --no-optimize --no-walk-forward

:: Run a single strategy
python main.py --strategy EMARibbonStrategy --no-optimize --no-walk-forward

:: Full pipeline for a single strategy
python main.py --strategy IchimokuCloudStrategy
```

> **Windows multiprocessing note:** The `ProcessPoolExecutor` used for parallel runs can behave differently on Windows due to how it spawns processes. If you see `BrokenProcessPool` or `EOFError`, use `--no-parallel`. Sequential execution is slightly slower but fully stable on Windows.

**No API key required for backtesting.** Binance historical OHLCV data is publicly accessible without authentication.

---

## 2. How the Engine Works

### Pipeline Overview

```
settings.yaml
    │
    ▼
Data Fetcher (binance_history.py)
    │  Fetches 4h OHLCV + funding rates via Binance Futures REST
    │  Caches to data/cache/*.parquet (keyed by symbol+timeframe+dates)
    ▼
Data Validator (validator.py)
    │  Flags anomalous candles (range > 5× ATR, zero volume)
    │  Sets is_clean=False on bad candles — strategies skip them
    ▼
Orchestrator (orchestrator.py) — ProcessPoolExecutor
    │  Spawns one worker process per strategy (up to 6 parallel)
    │
    └─► Per-strategy worker:
            │
            ▼
        SignalEngine (strategy.generate_signals)
            │  Reads entire candle history → produces list of Signals
            │  Signals fire on candle CLOSE (no lookahead)
            ▼
        BacktestRunner (runner.py)
            │  Time-ordered loop over candles
            │  Passes signal to RiskEngine → gets RiskDecision
            │  Schedules entry for NEXT candle open (no lookahead)
            ▼
        RiskEngine (risk/engine.py)
            │  ATR-based SL/TP calculation
            │  Position sizing (equity × max_position_pct × leverage)
            │  Drawdown guard, correlation guard, leverage cap
            ▼
        PaperExecutionEngine (execution/paper.py)
            │  Fills at next candle open + slippage
            │  Checks SL/TP hit within each candle (high/low)
            │  Checks liquidation (entry × (1 - 1/leverage))
            │  Applies fees on both entry and exit legs
            ▼
        StateStore (state/store.py)
            │  SQLite :memory: per strategy
            │  Records every fill and equity snapshot
            ▼
        [If composite_score ≤ 0] → Optuna Optimizer (500 trials)
            ▼
        Walk-Forward Validator (walk_forward.py)
            ▼
        Sensitivity Analyzer (sensitivity.py)
            ▼
        StrategyResult → Reporter → CSV + plots
```

### No Lookahead Bias

The engine enforces a strict two-step process:

1. **Signal generation:** Strategy reads the closed candle at time T, generates a Signal.
2. **Order fill:** PaperExecutionEngine fills the order at the **open of candle T+1**.

This is identical to what a real trader can do — you see the candle close, decide to enter, and your order fills at the next market open.

### Capital Isolation

Each strategy runs with completely separate capital. A drawdown in EMARibbon does not affect TTMSqueeze's capital. This reflects running 16 independent strategy accounts.

---

## 3. Configuration Reference

**File:** `config/settings.yaml`

```yaml
exchange:
  name: binance
  market: futures        # futures = Binance Futures (USDT-margined perpetuals)
  testnet: false         # false = mainnet historical data

backtest:
  symbols:
    - BTCUSDT
    - ETHUSDT
  timeframe: "4h"        # Candle interval: 1m, 5m, 15m, 1h, 4h, 1d
  start_date: "2020-01-01"
  end_date:   "2024-12-31"
  initial_capital_per_strategy: 1000.0   # USDT per strategy
  fees_pct: 0.001         # 0.1% per trade leg (Binance taker fee)
  slippage_pct: 0.0005    # 0.05% slippage per fill

leverage:
  max: 3                  # Hard cap — never exceeded regardless of strategy request
  default: 1              # Strategies run at 1x unless they request higher

risk:
  max_position_pct: 0.10        # Max 10% of equity per position
  stop_loss_atr_multiplier: 2.0 # SL = entry ± 2 × ATR
  take_profit_atr_multiplier: 3.0  # TP = entry ± 3 × ATR (R:R = 1.5:1)
  max_drawdown_pct: 0.20        # Strategy halts new entries if DD > 20%
  max_correlated_positions: 2   # Max simultaneous positions with correlation > threshold
  correlation_lookback_days: 30
  correlation_threshold: 0.75

optimization:
  trials: 500             # Optuna trials per strategy (increase for better params)
  sampler: "TPE"          # Tree-structured Parzen Estimator (Bayesian)
  objective: "composite"  # sharpe × (1-DD) × profit_factor
  min_trades_per_year: 30 # Penalty applied below this
  study_storage: "data/studies/{strategy_name}.db"  # Resume-able

walk_forward:
  train_months: 12        # In-sample window
  test_months: 3          # Out-of-sample test window
  step_months: 3          # Roll forward by this many months each window

promotion_criteria:
  min_sharpe_oos: 1.0     # Out-of-sample Sharpe must exceed this
  max_drawdown_pct: 0.25  # Max allowable OOS drawdown
  min_profit_factor: 1.3  # Gross profit / gross loss > 1.3
  min_trades_per_year: 30 # Statistical significance threshold
  max_is_oos_divergence: 2.0  # IS Sharpe must be < 2× OOS Sharpe (robustness check)
```

### Key Config Decisions

**`initial_capital_per_strategy`** — Set to $1,000 for realistic small-account simulation. Scale up to $10,000+ for lower noise in percentage returns.

**`fees_pct: 0.001`** — Binance taker fee. If you have BNB fee discount, use 0.00075. Maker orders: 0.0002.

**`leverage.max: 3`** — Conservative cap. All results shown at this leverage level. Set to 1 to see unlevered base performance.

**`optimization.trials: 500`** — More trials = better params but slower (each trial runs a full backtest). Reduce to 50–100 for quick tests.

---

## 4. Strategies

### Strategy 1: EMA Ribbon (`EMARibbonStrategy`)

**File:** `crypto_bot/core/signals/strategies/ema_ribbon.py`

**Logic:** Four EMAs (fast/medium/slow/trend). Enters when all four are perfectly stacked in bullish or bearish order AND ADX confirms trend strength. Exits when the fast EMA crosses back through the medium EMA.

**Best regime:** Strong trending markets (ADX > 20). Avoids choppy sideways conditions.

**Param space:**
| Parameter | Range | Default |
|-----------|-------|---------|
| `ema_fast` | 5–15 | 10 |
| `ema_medium` | 15–30 | 22 |
| `ema_slow` | 30–60 | 45 |
| `ema_trend` | 100–250 | 175 |
| `adx_period` | 10–20 | 15 |
| `adx_threshold` | 20–35 | 27 |

---

### Strategy 2: TTM Squeeze (`TTMSqueezeStrategy`)

**File:** `crypto_bot/core/signals/strategies/ttm_squeeze.py`

**Logic:** Measures market compression by checking if Bollinger Bands are inside Keltner Channels (squeeze = energy building). When BB expands outside KC (squeeze releases), enters in the direction of the momentum histogram. Exits when momentum reverses.

**Best regime:** Pre-breakout consolidation phases. Common after crypto runs into resistance then compresses.

**Param space:**
| Parameter | Range | Default |
|-----------|-------|---------|
| `bb_period` | 15–25 | 20 |
| `bb_std` | 1.5–2.5 | 2.0 |
| `kc_period` | 15–25 | 20 |
| `kc_mult` | 1.0–2.0 | 1.5 |
| `mom_period` | 10–20 | 15 |

---

### Strategy 3: RSI Hidden Divergence (`RSIDivergenceStrategy`)

**File:** `crypto_bot/core/signals/strategies/rsi_divergence.py`

**Logic:** Hidden bullish divergence = price makes a higher low but RSI makes a lower low → trend continuation long. Hidden bearish divergence = price makes a lower high but RSI makes a higher high → trend continuation short. Confirmed by MACD histogram direction.

**Best regime:** Trending markets with pullbacks. High win rate because it enters on pullbacks within trends, not reversals.

**Note:** This is the only strategy that passed promotion criteria on initial backtest (Sharpe 1.02).

**Param space:**
| Parameter | Range | Default |
|-----------|-------|---------|
| `rsi_period` | 8–21 | 14 |
| `divergence_lookback` | 3–10 | 6 |
| `macd_fast` | 8–16 | 12 |
| `macd_slow` | 20–30 | 25 |
| `macd_signal` | 7–12 | 9 |

---

### Strategy 4: Supertrend + ADX (`SupertrendADXStrategy`)

**File:** `crypto_bot/core/signals/strategies/supertrend_adx.py`

**Logic:** Supertrend (ATR-based trailing stop band) flips from bearish to bullish → LONG signal, only if ADX confirms the trend is strong. Exits when Supertrend flips again.

**Best regime:** Strong trends. Naturally adapts to volatility via ATR.

**Param space:**
| Parameter | Range | Default |
|-----------|-------|---------|
| `atr_period` | 7–21 | 14 |
| `atr_multiplier` | 1.5–4.0 | 2.75 |
| `adx_period` | 10–20 | 15 |
| `adx_threshold` | 20–35 | 27 |

---

### Strategy 5: Bollinger Band Mean Reversion (`BBMeanReversionStrategy`)

**File:** `crypto_bot/core/signals/strategies/bb_mean_reversion.py`

**Logic:** Price touches lower Bollinger Band AND RSI is oversold AND BB width is below squeeze threshold (markets are tight, extreme moves are mean-reverting) → LONG. Target: middle band. Reverse for short.

**Best regime:** Ranging, low-volatility markets. Underperforms in strong trends — when it triggers in a trend, it's fighting the direction.

**Param space:**
| Parameter | Range | Default |
|-----------|-------|---------|
| `bb_period` | 15–30 | 22 |
| `bb_std` | 1.8–2.5 | 2.15 |
| `rsi_period` | 8–21 | 14 |
| `rsi_threshold` | 25–40 | 32 |
| `bb_width_threshold` | 0.02–0.08 | 0.05 |

---

### Strategy 6: Funding Rate Reversion (`FundingRateReversionStrategy`)

**File:** `crypto_bot/core/signals/strategies/funding_rate_reversion.py`

**Logic:** Extreme positive funding rate (longs paying shorts heavily) = overcrowded long position → go short. Extreme negative funding → go long. Confirmed by RSI showing the same extreme. Exits when funding normalizes.

**Requires:** Funding rate data in `aux_data["funding_rates"]`. The data must be fetched separately (see [Data Pipeline](#10-data-pipeline)).

**Best regime:** Late-cycle extremes. Genuinely uncorrelated with price-signal strategies. 2021 bull top, 2022 capitulation bottoms are classic setups.

**Param space:**
| Parameter | Range | Default |
|-----------|-------|---------|
| `funding_threshold` | 0.0005–0.003 | 0.00175 |
| `rsi_period` | 8–21 | 14 |
| `rsi_confirm_long` | 20–45 | 32 |
| `rsi_confirm_short` | 55–80 | 67 |
| `exit_funding_pct` | 0.1–0.5 | 0.3 |

---

### Strategy 7: XGBoost Meta-Classifier (`XGBoostMetaStrategy`)

**File:** `crypto_bot/core/signals/strategies/xgboost_meta.py`

**Logic:** Does NOT generate signals or hold capital independently. Acts as a **go/no-go gate** on top of Strategies 1–6. Trained on features: RSI, MACD histogram, BB %B, ATR%, ADX, volume z-score, funding rate, hour-of-day, day-of-week. Predicts whether a proposed trade will be profitable. If the model says no, the signal is suppressed.

**Effect:** Reduces trade count (only takes high-confidence setups), increases win rate.

**Param space:**
| Parameter | Range | Default |
|-----------|-------|---------|
| `n_estimators` | 50–300 | 175 |
| `max_depth` | 3–8 | 5 |
| `learning_rate` | 0.01–0.3 | 0.155 |
| `subsample` | 0.6–1.0 | 0.8 |
| `forward_window` | 3–12 | 7 |

---

### Strategy 8: Donchian Channel Breakout (`DonchianBreakoutStrategy`) ⚡

**File:** `crypto_bot/core/signals/strategies/donchian_breakout.py`

**Logic:** Enters when price closes **above the highest high** (or below the lowest low) of the last N candles — but only if two confirmation filters are simultaneously true:
1. **Volume surge:** Current volume > rolling average × `vol_mult` (confirms institutional participation, not a thin-market fakeout)
2. **ATR expansion:** Current ATR > rolling average × `atr_expansion_mult` (confirms volatility is increasing — the move has energy behind it)

Exits when price crosses back through the Donchian midline (mean of upper + lower bands).

**Why this is the "explosive" strategy:**
- Crypto's biggest directional moves are Donchian breakouts: BTC above $20k (Nov 2020), $60k (Mar 2021), ETH DeFi summer 2020, bear market legs in 2022
- The volume + ATR double-filter eliminates low-energy false breakouts — only enters when the market is genuinely moving with conviction
- Unlike EMA Ribbon (which needs 4 indicators to slowly align), this fires immediately when a true breakout occurs
- Naturally catches fat-tail moves that make crypto different from equities

**Best regime:** Breakout/trending markets. The more compressed the market before the breakout, the more explosive the resulting signal.

**Param space:**
| Parameter | Range | Default |
|-----------|-------|---------|
| `channel_period` | 10–50 | 20 |
| `vol_period` | 10–30 | 20 |
| `vol_mult` | 1.2–2.5 | 1.5 |
| `atr_period` | 10–21 | 14 |
| `atr_expansion_mult` | 1.1–1.8 | 1.3 |

---

### Strategy 9: Time-Series Momentum (`TSMOMStrategy`)

**File:** `crypto_bot/core/signals/strategies/tsmom.py`

**Logic:** Classic quantitative momentum — measures the sign of returns over a trailing lookback window. Long if the asset is up over the window; short if down. Filters by momentum strength exceeding a minimum threshold. Exits when momentum reverses sign.

**Best regime:** Sustained trending markets (bull runs, bear markets). Under-performs in choppy, mean-reverting conditions.

**Param space:**
| Parameter | Range | Default |
|-----------|-------|---------|
| `lookback_period` | 20–120 | 60 |
| `momentum_threshold` | 0.01–0.10 | 0.03 |
| `atr_period` | 10–20 | 14 |

---

### Strategy 10: HMA + Chandelier Exit (`HMAChandelierStrategy`)

**File:** `crypto_bot/core/signals/strategies/hma_chandelier.py`

**Logic:** Hull Moving Average (reduces lag vs EMA) determines trend direction. Chandelier Exit (highest high / lowest low over N bars minus ATR multiple) acts as an adaptive trailing stop. Enters on HMA direction change; exits when Chandelier stop is breached.

**Best regime:** Trending markets. The HMA reacts faster to reversals than EMA-based strategies.

**Param space:**
| Parameter | Range | Default |
|-----------|-------|---------|
| `hma_period` | 20–60 | 40 |
| `chandelier_period` | 10–30 | 22 |
| `chandelier_mult` | 2.0–4.0 | 3.0 |
| `atr_period` | 10–20 | 14 |

---

### Strategy 11: VWAP Breakout (`VWAPBreakoutStrategy`)

**File:** `crypto_bot/core/signals/strategies/vwap_breakout.py`

**Logic:** Tracks rolling VWAP with standard-deviation bands. Enters when price breaks decisively above/below a band AND volume confirms the move (volume z-score > threshold). Exits when price returns to VWAP midline.

**Best regime:** High-volume breakout sessions. VWAP deviation is most meaningful when volume is elevated.

**Param space:**
| Parameter | Range | Default |
|-----------|-------|---------|
| `vwap_period` | 20–100 | 50 |
| `band_std` | 1.5–3.0 | 2.0 |
| `vol_zscore_threshold` | 1.0–3.0 | 1.5 |
| `atr_period` | 10–20 | 14 |

---

### Strategy 12: Ichimoku Cloud (`IchimokuCloudStrategy`)

**File:** `crypto_bot/core/signals/strategies/ichimoku_cloud.py`

**Logic:** Full five-component Ichimoku system. Enters LONG when all three conditions are true simultaneously:
1. Price is **above the cloud** (above both Senkou Span A and B)
2. **Tenkan-sen ≥ Kijun-sen** (conversion line at or above base line)
3. **Chikou Span is above price** 26 bars ago (lagging span confirms bullish momentum)

Exits when any condition flips bearish. Mirror logic for SHORT entries.

**Why all three matter:**
- Cloud = medium-term support/resistance zone — price above cloud = bullish structure
- Tenkan/Kijun cross = short-term momentum aligned
- Chikou Span = confirmation from 26 bars ago that the prior price level was lower (not in a chop zone)

**Best regime:** Sustained trending markets. The three-condition gate eliminates almost all false signals in ranging markets.

**Param space:**
| Parameter | Range | Default |
|-----------|-------|---------|
| `tenkan_period` | 7–12 | 9 |
| `kijun_period` | 22–30 | 26 |
| `senkou_b_period` | 44–60 | 52 |
| `atr_period` | 10–20 | 14 |

---

### Strategy 13: Stochastic RSI (`StochRSIStrategy`)

**File:** `crypto_bot/core/signals/strategies/stoch_rsi.py`

**Logic:** Applies the Stochastic formula to RSI values (not price), producing %K and %D lines in a 0–100 range. Enters LONG when %K **crosses above** %D while both lines are in oversold territory (< 20). Enters SHORT when %K **crosses below** %D while both are in overbought territory (> 80). Exits on reverse cross or when momentum reaches the neutral zone (40–60).

**Why StochRSI over plain Stochastic:**
- RSI is already normalised — applying Stochastic to it produces a faster, more sensitive oscillator
- The double-smoothing (%K and %D) filters out single-bar noise while still catching momentum turns early

**Best regime:** Ranging or mildly trending markets. Overbought/oversold signals are most reliable when the market is not in a strong trend (in a strong trend, RSI stays extreme for extended periods).

**Param space:**
| Parameter | Range | Default |
|-----------|-------|---------|
| `rsi_period` | 10–21 | 14 |
| `stoch_period` | 10–21 | 14 |
| `smooth_k` | 2–5 | 3 |
| `smooth_d` | 2–5 | 3 |
| `atr_period` | 10–20 | 14 |

---

### Strategy 14: MACD Histogram Divergence (`MACDHistDivergenceStrategy`)

**File:** `crypto_bot/core/signals/strategies/macd_hist_divergence.py`

**Logic:** Counter-trend divergence strategy. **Bullish divergence:** price makes a lower low but MACD histogram makes a higher low (while histogram is still negative) — momentum is weakening even as price falls → anticipate reversal LONG. **Bearish divergence:** price makes a higher high but histogram makes a lower high (while positive) → anticipate reversal SHORT.

Exits when the histogram crosses zero in the opposite direction (momentum fully reverses).

**Important:** This strategy fires **against** the prevailing price direction — it is a counter-trend, reversal strategy. It naturally underperforms in strongly trending markets and outperforms near turning points.

**Param space:**
| Parameter | Range | Default |
|-----------|-------|---------|
| `macd_fast` | 8–16 | 12 |
| `macd_slow` | 20–30 | 26 |
| `macd_signal` | 7–12 | 9 |
| `divergence_lookback` | 3–10 | 5 |
| `atr_period` | 10–20 | 14 |

---

### Strategy 15: Market Regime Classifier (`MarketRegimeStrategy`)

**File:** `crypto_bot/core/signals/strategies/market_regime.py`

**Logic:** Classifies the current market into one of four regimes using three indicators simultaneously:
- **VOLATILE:** ATR > `atr_vol_threshold` × ATR rolling mean — abnormally high volatility, regime unreliable
- **UPTREND:** ADX > trend threshold AND EMA slope positive
- **DOWNTREND:** ADX > trend threshold AND EMA slope negative
- **RANGING:** ADX < range threshold — no directional conviction

Enters LONG on UPTREND detection, SHORT on DOWNTREND. Exits immediately when regime transitions to RANGING or VOLATILE (conditions no longer valid).

**Why regime classification matters:**
- Trend-following strategies fail in ranging markets; mean-reversion strategies fail in trends
- ADX alone doesn't distinguish up from down trend; combining with EMA slope fixes this
- The VOLATILE exit is critical for crypto — extreme volatility events (flash crashes, liquidation cascades) invalidate trend signals

**Best regime:** All market conditions — this strategy adapts to what the market is doing rather than assuming a fixed regime.

**Param space:**
| Parameter | Range | Default |
|-----------|-------|---------|
| `adx_period` | 10–20 | 14 |
| `adx_trend_threshold` | 20.0–30.0 | 25.0 |
| `adx_range_threshold` | 15.0–25.0 | 20.0 |
| `ema_period` | 50–200 | 100 |
| `slope_lookback` | 3–10 | 5 |
| `atr_vol_threshold` | 1.5–3.0 | 2.0 |
| `atr_period` | 10–20 | 14 |

---

### Strategy 16: Open Interest Divergence (`OpenInterestDivergenceStrategy`)

**File:** `crypto_bot/core/signals/strategies/open_interest_divergence.py`

**Logic:** Uses Binance Futures open interest data to identify trapped positions. **LONG signal:** OI rising (positions being added) + price falling + RSI < 45 → shorts are trapped, a squeeze is likely. **SHORT signal:** OI rising + price rising + RSI > 55 → longs are over-extended, an unwind is likely. **Exit:** OI drops sharply (positions are being closed = the squeeze/unwind is over).

**Requires:** `aux_data["open_interest"]` — a `pd.Series` with timestamp index. Fetched via `fetch_open_interest()` in `binance_history.py`. Returns `[]` gracefully if data is absent.

**Why OI divergence works:**
- Rising OI + price move = new money entering the market in that direction
- When OI rises but price moves against the new positions, those traders are immediately underwater
- The resulting forced liquidations create explosive counter-moves — this strategy tries to front-run the squeeze

**Best regime:** Volatile derivatives markets with high OI activity. Most reliable around major news events and liquidation cascades.

**Fetching OI data:**
```python
from crypto_bot.core.data.binance_history import fetch_open_interest

oi = fetch_open_interest("BTCUSDT", "4h", "2020-01-01", "2024-12-31")
aux_data = {"open_interest": oi}
```

**Param space:**
| Parameter | Range | Default |
|-----------|-------|---------|
| `oi_change_period` | 3–10 | 5 |
| `oi_surge_pct` | 0.5–3.0 | 1.5 |
| `rsi_period` | 10–21 | 14 |
| `atr_period` | 10–20 | 14 |

---

## 5. Running the Backtest

### All Strategies (Default)

**Linux / macOS:**
```bash
python main.py
```

**Windows:**
```cmd
python main.py --no-parallel
```

Runs all 16 strategies (6 at a time on Linux, sequential on Windows). For each strategy:
1. Runs initial backtest with default params
2. If `composite_score ≤ 0` or `trades_per_year < 30`, triggers Optuna (500 trials)
3. Runs walk-forward validation
4. Runs sensitivity analysis
5. Prints per-strategy report + saves CSV + plots equity curve

### Single Strategy

```bash
# Available strategy names (all platforms):
# EMARibbonStrategy         TTMSqueezeStrategy        RSIDivergenceStrategy
# SupertrendADXStrategy     BBMeanReversionStrategy   FundingRateReversionStrategy
# DonchianBreakoutStrategy  TSMOMStrategy             HMAChandelierStrategy
# AdaptiveTrendStrategy     VWAPBreakoutStrategy      IchimokuCloudStrategy
# StochRSIStrategy          MACDHistDivergenceStrategy MarketRegimeStrategy
# OpenInterestDivergenceStrategy

python main.py --strategy IchimokuCloudStrategy
```

### Speed vs Depth Trade-offs

| Flag | Effect | When to use |
|------|--------|-------------|
| `--no-optimize` | Skip Optuna, use default params | Quick sanity check |
| `--no-walk-forward` | Skip 16-window WFV | When iterating on a new strategy |
| `--no-parallel` | Single process, sequential | Windows, or debugging crashes / pickling errors |
| Both `--no-optimize --no-walk-forward` | Fastest run | Initial development |

### Estimated Run Times

| Mode | Platform | Time estimate |
|------|----------|--------------|
| Single strategy, no optimize, no WFV | Any | ~5–30 seconds |
| All strategies, no optimize, no WFV | Linux | ~2 minutes |
| All strategies, no optimize, no WFV | Windows (sequential) | ~5 minutes |
| Single strategy, full pipeline | Any | ~5–15 minutes |
| All strategies, full pipeline | Linux (6 parallel) | ~60–120 minutes |
| All strategies, full pipeline | Windows (sequential) | ~3–5 hours |

---

## 6. Understanding Results

### Per-Strategy Report

```
────────────────────────────────────────────
  RSIDivergenceStrategy
────────────────────────────────────────────
  Final Equity   :  $1,281.64   (started $1,000)
  Total Return   :    +28.16%
  Sharpe Ratio   :      1.018   ← annualised, √(252×6) factor for 4h bars
  Sortino Ratio  :      1.043   ← like Sharpe but only penalises downside
  Max Drawdown   :      -4.76%  (713 days underwater — time in DD, not severity)
  Trades (total) :        200   (40.0/yr)
  Win Rate       :      51.0%
  Profit Factor  :      1.573   ← gross_profit / gross_loss
  Avg Trade P&L  :     $14.02
  Fees Paid      :    $481.98
  Slippage Paid  :    $240.99
  Long/Short     :    108/92
  Exit reasons   :  sl:92, tp:88, signal:18, end_of_data:2
```

### Metric Explanations

| Metric | Formula | Good value |
|--------|---------|------------|
| **Sharpe Ratio** | `mean(daily_ret) / std(daily_ret) × √252` | > 1.0 |
| **Sortino Ratio** | `mean(daily_ret) / std(downside_ret) × √252` | > 1.0 |
| **Max Drawdown** | `(peak_equity - trough_equity) / peak_equity` | < 25% |
| **Profit Factor** | `sum(winning_trades) / abs(sum(losing_trades))` | > 1.3 |
| **Win Rate** | `winning_trades / total_trades` | 45–55% (with good R:R) |
| **Composite Score** | `sharpe × (1 - DD) × PF × trade_count_penalty` | > 0 to proceed |

### Exit Reason Breakdown

| Exit reason | Meaning |
|-------------|---------|
| `sl` | Stop loss hit — candle low/high breached SL level |
| `tp` | Take profit hit — candle low/high breached TP level |
| `signal` | Strategy generated an explicit EXIT signal |
| `liquidation` | Position fully liquidated (rare at leverage ≤ 3x) |
| `end_of_data` | Position still open when backtest ended — closed at final price |

### Promotion Criteria

A strategy must pass **all four** to be promoted to paper trading (Phase 2):

| Criterion | Threshold | Checked on |
|-----------|-----------|------------|
| OOS Sharpe | > 1.0 | Walk-forward OOS |
| Max Drawdown | < 25% | Walk-forward OOS |
| Profit Factor | > 1.3 | Walk-forward OOS |
| Trades/year | > 30 | Full backtest |
| IS/OOS divergence | < 2.0× | WFV IS vs OOS Sharpe ratio |

---

## 7. Optimization (Optuna)

### When it Triggers

Automatically triggered if the initial backtest composite score ≤ 0, or if trades/year < 30 (strategy is too infrequent). Can also be run explicitly:

```bash
# Force optimization on a specific strategy
.venv/bin/python main.py --mode backtest --config config/settings.yaml \
  --strategy EMARibbonStrategy
# (optimization runs by default unless --no-optimize is passed)
```

### How it Works

Uses **Bayesian optimization** (Tree-structured Parzen Estimator — TPE) which builds a probabilistic model of the objective function. Much more efficient than random or grid search — converges to good params in 500 trials vs ~10,000 needed for grid search.

**Objective function:**
```
composite_score = sharpe × (1 - max_drawdown) × profit_factor × trade_count_penalty

where:
  trade_count_penalty = 1.0 if trades/yr ≥ min_trades_per_year
                      = trades/yr / min_trades_per_year otherwise
```

### Resuming a Study

Optuna stores studies in `data/studies/{strategy_name}.db`. If a run is interrupted, it resumes automatically from where it left off next time you run the same strategy. Delete the `.db` file to start fresh.

```bash
# Delete all studies and start fresh
rm data/studies/*.db

# Delete a specific study
rm data/studies/EMARibbonStrategy.db
```

### Tuning the Optimizer

```yaml
# In config/settings.yaml
optimization:
  trials: 100    # Reduce for quick dev, increase to 1000 for production
```

MedianPruner is active after 20 startup trials — it terminates unpromising trials early (when the trial's intermediate scores are worse than the median of completed trials). Saves ~40% compute.

---

## 8. Walk-Forward Validation

### What It Does

Avoids overfitting by testing each set of parameters on data it was **never trained on**. The in-sample data is used to find the best params via Optuna; the out-of-sample data is used to evaluate those params honestly.

### Window Structure (2020–2024)

```
Window  1: TRAIN [Jan20–Dec20] → TEST [Jan21–Mar21]
Window  2: TRAIN [Apr20–Mar21] → TEST [Apr21–Jun21]
Window  3: TRAIN [Jul20–Jun21] → TEST [Jul21–Sep21]
...
Window 16: TRAIN [Jan23–Dec23] → TEST [Jan24–Mar24]
```

### WFV Metrics

| Metric | Description |
|--------|-------------|
| `oos_sharpe` | Mean Sharpe across all OOS test windows |
| `oos_max_drawdown` | Worst drawdown across all OOS test windows |
| `oos_profit_factor` | Combined gross profit / gross loss on all OOS data |
| `consistency_score` | % of windows that were profitable |
| `is_oos_divergence` | IS Sharpe / OOS Sharpe — should be < 2.0 |

A high IS/OOS divergence (e.g., 4.0×) means the params are overfit to the training data and will not generalize.

### Reducing WFV Runtime

```bash
# Reduce Optuna trials per window (default: 30 in WFV mode vs 500 for full optimization)
# Edit backtest/orchestrator.py line:
#   wf_result = run_walk_forward(..., n_trials_per_window=30)
# Change 30 → 10 for faster runs
```

---

## 9. Parameter Sensitivity Analysis

### Purpose

After optimization finds "best" params, sensitivity analysis checks whether those params are **robust** or **brittle**. A brittle strategy only works at exactly the optimized values and will fail in production as market conditions shift slightly.

### How It Works

Each parameter is perturbed by ±10% and ±20% from its optimal value. The backtest is re-run and the composite score change is measured.

**Example for RSI period = 14:**
```
RSI period = 14 (optimal)
  +10% → 15: score change  -3%  → ROBUST
  -10% → 13: score change  -8%  → ROBUST
  +20% → 17: score change -15%  → ACCEPTABLE
  -20% → 11: score change -45%  → BRITTLE ⚠
```

### Ruling

**BRITTLE:** Any parameter where a ±20% perturbation causes a > 30% drop in composite score. The strategy is rejected even if it passes other criteria — it's too sensitive to exact parameter values to be trusted in live trading.

**ROBUST:** All perturbations cause < 30% score change at ±20%. Safe to proceed.

---

## 10. Data Pipeline

### OHLCV Data (Auto-fetched)

```
Source: Binance Futures mainnet (/fapi/v1/klines)
Auth:   None required — public endpoint
Cache:  data/cache/BTCUSDT_4h_2020-01-01_2024-12-31.parquet
```

Data is fetched once and cached. Subsequent runs read from cache (instant). To re-fetch:

```bash
rm data/cache/BTCUSDT_4h_2020-01-01_2024-12-31.parquet
rm data/cache/ETHUSDT_4h_2020-01-01_2024-12-31.parquet
.venv/bin/python main.py --mode backtest --config config/settings.yaml
```

### Funding Rate Data (Required for FundingRateReversionStrategy)

Funding rates must be fetched separately. They are not auto-fetched in the current CLI but can be fetched manually:

```python
from crypto_bot.core.data.binance_history import fetch_funding_rates
import pandas as pd

df_btc = fetch_funding_rates("BTCUSDT", "2020-01-01", "2024-12-31")
df_eth = fetch_funding_rates("ETHUSDT", "2020-01-01", "2024-12-31")

# Combine and pass as aux_data to BacktestRunner
aux_data = {"funding_rates": pd.concat([df_btc, df_eth])}
```

Without funding rate data, `FundingRateReversionStrategy` produces zero signals (by design — it logs a warning and returns empty).

### Open Interest Data (Required for OpenInterestDivergenceStrategy)

Open interest history is available for Binance Futures perpetuals. Fetched and cached automatically:

```python
from crypto_bot.core.data.binance_history import fetch_open_interest

# Fetch OI for a symbol (cached to data/cache/ as parquet)
oi_btc = fetch_open_interest("BTCUSDT", "4h", "2020-01-01", "2024-12-31")
oi_eth = fetch_open_interest("ETHUSDT", "4h", "2020-01-01", "2024-12-31")

# Pass to BacktestRunner
aux_data = {
    "open_interest": oi_btc,   # pd.Series with timestamp index
}
```

Cache location: `data/cache/BTCUSDT_oi_4h_2020-01-01_2024-12-31.parquet`

Without OI data, `OpenInterestDivergenceStrategy` returns `[]` (graceful degradation — no error, no signals).

### Data Quality

The validator (`core/data/validator.py`) checks each candle:
- **Zero volume:** Flagged as `is_clean=False`
- **Anomalous range:** Candle range (high-low) > 5× ATR — flagged as `is_clean=False`
- **Duplicate timestamps:** Deduplicated automatically

All strategies skip candles where `is_clean=False`. The data quality score (% clean candles) is printed at startup.

---

## 11. Troubleshooting

### "No candle data found"

```
ERROR: No candle data found. Run data fetch first or check cache directory.
```

The cache is empty and the Binance API fetch failed. Check:
1. Internet connectivity
2. `data/cache/` directory exists (created automatically, but check permissions)
3. The date range in `settings.yaml` is valid (not future dates)

### "Strategy produced 0 signals"

- **FundingRateReversionStrategy:** Needs `aux_data["funding_rates"]` — see [Funding Rate Data](#funding-rate-data)
- **Other strategies:** Warmup period may be too long for the data. Ensure at least 300 candles (~50 days of 4h data)

### Walk-Forward takes too long

Reduce `n_trials_per_window` in `backtest/orchestrator.py` from 30 to 5–10. WFV runs Optuna per window, so 16 windows × 30 trials = 480 mini-backtests.

### Optuna study conflict

If you change a strategy's `param_space`, delete the old study file:

```bash
rm data/studies/EMARibbonStrategy.db
```

Otherwise Optuna will try to resume a study whose parameter space no longer matches.

### Memory error on parallel run

Reduce `max_workers` or use `--no-parallel`:

```bash
python main.py --no-parallel
```

Or edit `main.py` and change `max_workers=6` to `max_workers=2` in `_run_backtest()`.

### Windows: BrokenProcessPool or EOFError

Windows spawns new Python interpreter processes for `ProcessPoolExecutor` which can fail in some environments. Use `--no-parallel`:

```cmd
python main.py --no-parallel
```

### Windows: "python" not found

If `python` is not recognised, try `py` instead:

```cmd
py main.py --strategy EMARibbonStrategy --no-optimize --no-walk-forward
```

Or ensure Python was added to PATH during installation (re-run the Python installer and check "Add Python to PATH").

### Windows: PowerShell execution policy error

```
.venv\Scripts\Activate.ps1 cannot be loaded because running scripts is disabled
```

Fix by setting execution policy for the current user:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then re-run `.venv\Scripts\Activate.ps1`.

### Results directory

All outputs are saved to `results/`:
- `{strategy}_fills_{timestamp}.csv` — every trade with entry, exit, P&L
- `{strategy}_equity_{timestamp}.csv` — equity at every candle
- `{strategy}_equity_curve_{timestamp}.png` — equity curve chart (if matplotlib installed)

Timestamp format: `YYYYMMDD_HHMMSS` matching when the run completed.
