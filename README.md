# Crypto Trading Bot — Backtesting Framework

A production-quality backtesting engine for crypto perpetual futures strategies, built on Binance Futures historical data. Features 16 trading strategies, Bayesian parameter optimization (Optuna), walk-forward validation, and sensitivity analysis.

---

## Table of Contents

- [Features](#features)
- [Setup — Linux / macOS](#setup--linux--macos)
- [Setup — Windows](#setup--windows)
- [Quick Start](#quick-start)
- [Strategies](#strategies)
- [CLI Reference](#cli-reference)
- [Configuration](#configuration)
- [Documentation](#documentation)

---

## Features

- **16 trading strategies** — trend-following, mean-reversion, breakout, oscillator, regime-based, and derivatives-based
- **No API key required** for backtesting — Binance OHLCV data is fetched from the public REST endpoint
- **Bayesian optimization** via Optuna (500 trials, TPE sampler) — auto-tunes every strategy's parameters
- **Walk-forward validation** — 12-month train / 3-month test rolling windows to prevent overfitting
- **Sensitivity analysis** — rejects strategies whose performance is brittle to ±20% parameter changes
- **Parallel execution** — up to 6 strategies run simultaneously via `ProcessPoolExecutor`
- **Zero lookahead bias** — signals fire on candle close; orders fill at the next candle open
- **Auto-caching** — fetched data stored as `.parquet`; subsequent runs use cache (instant load)

---

## Setup — Linux / macOS

### Prerequisites

- Python 3.13 or higher
- Git

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/rithee/backtest-algotrading.git
cd backtest-algotrading

# 2. Create a virtual environment
python3 -m venv .venv

# 3. Activate the virtual environment
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Verify setup
python main.py --help
```

---

## Setup — Windows

### Prerequisites

- **Python 3.13+** — Download from [python.org/downloads](https://www.python.org/downloads/)
  - ✅ During install, check **"Add Python to PATH"**
- **Git** — Download from [git-scm.com](https://git-scm.com/download/win)

### Installation (Command Prompt or PowerShell)

```cmd
:: 1. Clone the repository
git clone https://github.com/rithee/backtest-algotrading.git
cd backtest-algotrading

:: 2. Create a virtual environment
python -m venv .venv

:: 3. Activate the virtual environment (Command Prompt)
.venv\Scripts\activate.bat

:: 3. Activate the virtual environment (PowerShell — if above doesn't work)
.venv\Scripts\Activate.ps1

:: 4. Install dependencies
pip install -r requirements.txt

:: 5. Verify setup
python main.py --help
```

> **PowerShell execution policy:** If you get a "scripts cannot be run" error in PowerShell, run this once:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```
> Then re-run `.venv\Scripts\Activate.ps1`.

### Windows-Specific Notes

| Issue | Fix |
|-------|-----|
| `python` not found | Ensure "Add Python to PATH" was checked during install, or use `py` instead of `python` |
| `pip` not found | Run `python -m pip install -r requirements.txt` instead |
| Long path errors | Enable long paths: run `regedit`, navigate to `HKLM\SYSTEM\CurrentControlSet\Control\FileSystem`, set `LongPathsEnabled = 1` |
| `pyarrow` install fails | Run `pip install --upgrade pip` first, then retry |
| Parallel mode crashes | Use `--no-parallel` flag — Windows has stricter multiprocessing restrictions than Linux |

---

## Quick Start

### First run (recommended — single strategy, no optimization)

**Linux / macOS:**
```bash
source .venv/bin/activate
python main.py --strategy EMARibbonStrategy --no-optimize --no-walk-forward
```

**Windows:**
```cmd
.venv\Scripts\activate.bat
python main.py --strategy EMARibbonStrategy --no-optimize --no-walk-forward
```

On first run, Binance OHLCV data is automatically downloaded (~30 seconds). Subsequent runs load from cache instantly.

### Run a single strategy with full pipeline

```bash
# Linux / macOS
python main.py --strategy IchimokuCloudStrategy

# Windows
python main.py --strategy IchimokuCloudStrategy
```

This runs: default backtest → Optuna optimization (500 trials) → walk-forward validation → sensitivity analysis → print report → save CSV.
**Takes 5–15 minutes per strategy.**

### Run all 16 strategies in parallel

```bash
# Linux / macOS (6 parallel workers)
python main.py

# Windows (use --no-parallel to avoid multiprocessing issues)
python main.py --no-parallel
```

Full pipeline for all strategies takes **1–3 hours**.

---

## Strategies

| # | Strategy | Type | `--strategy` flag |
|---|----------|------|-------------------|
| 1 | EMA Ribbon | Trend-following | `EMARibbonStrategy` |
| 2 | TTM Squeeze | Momentum | `TTMSqueezeStrategy` |
| 3 | RSI Divergence | Counter-trend | `RSIDivergenceStrategy` |
| 4 | Supertrend + ADX | Trend-following | `SupertrendADXStrategy` |
| 5 | Bollinger Band Mean Reversion | Mean-reversion | `BBMeanReversionStrategy` |
| 6 | Funding Rate Reversion | Derivatives | `FundingRateReversionStrategy` |
| 7 | Donchian Channel Breakout | Breakout | `DonchianBreakoutStrategy` |
| 8 | Time-Series Momentum | Momentum | `TSMOMStrategy` |
| 9 | HMA + Chandelier Exit | Trend-following | `HMAChandelierStrategy` |
| 10 | Adaptive Trend (KAMA) | Adaptive | `AdaptiveTrendStrategy` |
| 11 | VWAP Breakout | Breakout | `VWAPBreakoutStrategy` |
| 12 | Ichimoku Cloud | Trend-following | `IchimokuCloudStrategy` |
| 13 | Stochastic RSI | Oscillator | `StochRSIStrategy` |
| 14 | MACD Histogram Divergence | Counter-trend | `MACDHistDivergenceStrategy` |
| 15 | Market Regime Classifier | Regime-based | `MarketRegimeStrategy` |
| 16 | Open Interest Divergence | On-chain/Derivatives | `OpenInterestDivergenceStrategy` |

> **Strategy 16 note:** `OpenInterestDivergenceStrategy` requires open interest data passed via `aux_data`. Without it, it returns no signals (by design — graceful degradation).

---

## CLI Reference

```
python main.py [options]

Options:
  --mode {backtest}        Run mode (default: backtest)
  --config CONFIG          Path to settings YAML (default: config/settings.yaml)
  --strategy STRATEGY      Run a single strategy by class name
  --no-parallel            Disable parallel execution (required on some Windows setups)
  --no-optimize            Skip Optuna optimisation (uses default params)
  --no-walk-forward        Skip walk-forward validation
  -h, --help               Show this message
```

### Common command combinations

| Goal | Command |
|------|---------|
| Fastest single-strategy test | `python main.py --strategy EMARibbonStrategy --no-optimize --no-walk-forward` |
| Full single-strategy pipeline | `python main.py --strategy IchimokuCloudStrategy` |
| All strategies, fast scan | `python main.py --no-optimize --no-walk-forward` |
| All strategies, full pipeline (Linux) | `python main.py` |
| All strategies, full pipeline (Windows) | `python main.py --no-parallel` |
| Debug mode | `python main.py --strategy EMARibbonStrategy --no-parallel --no-optimize --no-walk-forward` |

---

## Configuration

Edit `config/settings.yaml` to change:

```yaml
backtest:
  symbols: ["BTCUSDT", "ETHUSDT"]   # Symbols to backtest
  timeframe: "4h"                    # Candle interval (1m, 5m, 15m, 1h, 4h, 1d)
  start_date: "2020-01-01"
  end_date:   "2024-12-31"
  initial_capital_per_strategy: 1000.0  # USDT per strategy

optimization:
  trials: 500    # Reduce to 50–100 for faster development runs
```

Full configuration reference: [`docs/BACKTEST.md`](docs/BACKTEST.md#3-configuration-reference)

---

## Documentation

| Document | Contents |
|----------|----------|
| [`docs/BACKTEST.md`](docs/BACKTEST.md) | Full engine reference — pipeline, all strategies, optimization, walk-forward, results interpretation, troubleshooting |

---

## Output Files

All results are saved to `results/`:

| File | Contents |
|------|----------|
| `{strategy}_fills_{timestamp}.csv` | Every trade: entry, exit, P&L, exit reason |
| `{strategy}_equity_{timestamp}.csv` | Equity at every candle |
| `{strategy}_equity_curve_{timestamp}.png` | Equity curve chart |

---

## Running Tests

```bash
# Linux / macOS
pytest tests/ -v

# Windows
python -m pytest tests/ -v
```
