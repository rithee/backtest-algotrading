# Crypto Trading Bot — Backtesting Framework

A production-quality backtesting engine for crypto perpetual futures and spot strategies, built on Binance historical data. Features 29 trading strategies across three modes, Bayesian parameter optimization (Optuna), walk-forward validation, sensitivity analysis, and rich terminal reporting.

---

## Table of Contents

- [Features](#features)
- [Setup — Linux / macOS](#setup--linux--macos)
- [Setup — Windows](#setup--windows)
- [Quick Start](#quick-start)
- [Three Modes Explained](#three-modes-explained)
- [Strategies](#strategies)
- [CLI Reference](#cli-reference)
- [Configuration](#configuration)
- [Output Files](#output-files)
- [Running Tests](#running-tests)
- [Documentation](#documentation)

---

## Features

- **29 trading strategies across three modes:**
  - **Swing** (16 strategies) — 4h futures, trend/mean-reversion/breakout, BTC + ETH
  - **Intraday** (6 strategies) — 1m–1h futures, session-filtered, BTC + ETH + SOL
  - **Spot long-term** (7 strategies) — 1d–1w spot, macro/on-chain driven, top-10 by market cap
- **No API key required** for backtesting — Binance OHLCV data is fetched from the public REST endpoint
- **Bayesian optimization** via Optuna (TPE sampler, per-mode objective) — auto-tunes every strategy
- **Walk-forward validation** — rolling train/test windows to prevent overfitting
- **Sensitivity analysis** — rejects strategies whose performance is brittle to ±20% parameter changes
- **Per-mode optimizer objectives:** swing = composite score, intraday = fee-adjusted Sharpe, spot = Calmar ratio
- **OOS promotion gate** — strategies must pass walk-forward OOS criteria to be promoted to paper trading
- **Rich terminal output** — colour-coded summary tables and per-strategy detail panels
- **Export** — results to `results/<mode>_summary.csv` or `results/<mode>_report.html`
- **Parallel execution** — up to 6 strategies run simultaneously via `ProcessPoolExecutor`
- **Zero lookahead bias** — signals fire on candle close; orders fill at the next candle open
- **Auto-caching** — fetched data stored as `.parquet`; subsequent runs use cache (instant load)

---

## Requirements Files

| File | Use on |
|------|--------|
| `requirements-windows.txt` | Windows |
| `requirements-linux.txt` | Linux / macOS |
| `requirements.txt` | Cross-platform fallback (same as Windows file) |

The only difference is `requirements-linux.txt` includes `nvidia-nccl-cu12` (a CUDA GPU library present in the Linux dev environment). It does not exist as a Windows package and will error if you try to install it on Windows.

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

# 4. Install dependencies (Linux-specific file)
pip install -r requirements-linux.txt

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

:: 4. Install dependencies (Windows-specific file)
pip install -r requirements-windows.txt

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

### First run — single swing strategy, no optimization (fastest)

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

### Run all 16 swing strategies

```bash
# Linux / macOS (6 parallel workers)
python main.py

# Windows (use --no-parallel to avoid multiprocessing issues)
python main.py --no-parallel
```

Full pipeline for all strategies takes **1–3 hours**. Use `--skip-wf --skip-sensitivity` for a fast scan:

```bash
python main.py --skip-wf --skip-sensitivity    # ~5 min on Linux
```

### Run the analyze mode (rich tables + optional export)

```bash
# Analyze swing strategies with summary table + detail panels
python main.py --mode analyze --analyze-mode swing

# Analyze intraday strategies and export to CSV
python main.py --mode analyze --analyze-mode intraday --export csv

# Analyze all three modes and export an HTML report
python main.py --mode analyze --analyze-mode all --export html
```

### Speed control flags

```bash
# Skip walk-forward (no OOS gate — uses IS metrics only)
python main.py --skip-wf

# Skip sensitivity analysis
python main.py --skip-sensitivity

# Override Optuna trial count (default: from config, usually 500)
python main.py --trials 50

# Combine for maximum speed
python main.py --skip-wf --skip-sensitivity --trials 20
```

---

## Three Modes Explained

| Mode | Flag | Strategies | Timeframes | Instruments | Optimizer objective |
|------|------|-----------|------------|-------------|---------------------|
| **Swing** | `--mode swing` | 16 | 4h | BTC, ETH futures | Composite score |
| **Intraday** | `--mode intraday` | 6 | 1m / 5m / 15m / 1h | BTC, ETH, SOL futures | Fee-adjusted Sharpe |
| **Spot long-term** | `--mode spot` | 7 | 1d / 1w | Top-10 market cap spot | Calmar ratio |

Each mode loads its own YAML config (`config/settings.yaml`, `config/intraday.yaml`, `config/spot_longterm.yaml`).

**`--mode analyze`** runs one or more modes and displays rich tables — it doesn't change which strategies run, only the output format.

---

## Strategies

### Swing Futures (16 strategies)

| # | Strategy | Type | `--strategy` flag |
|---|----------|------|-------------------|
| 1 | EMA Ribbon | Trend-following | `EMARibbonStrategy` |
| 2 | TTM Squeeze | Momentum | `TTMSqueezeStrategy` |
| 3 | RSI Divergence | Counter-trend | `RSIDivergenceStrategy` |
| 4 | Supertrend + ADX | Trend-following | `SupertrendADXStrategy` |
| 5 | Bollinger Band Mean Reversion | Mean-reversion | `BBMeanReversionStrategy` |
| 6 | Funding Rate Reversion | Derivatives | `FundingRateReversionStrategy` |
| 7 | XGBoost Meta-Classifier | ML gate | `XGBoostMetaStrategy` |
| 8 | Donchian Channel Breakout | Breakout | `DonchianBreakoutStrategy` |
| 9 | Time-Series Momentum | Momentum | `TSMOMStrategy` |
| 10 | HMA + Chandelier Exit | Trend-following | `HMAChandelierStrategy` |
| 11 | Adaptive Trend (KAMA) | Adaptive | `AdaptiveTrendStrategy` |
| 12 | VWAP Breakout | Breakout | `VWAPBreakoutStrategy` |
| 13 | Ichimoku Cloud | Trend-following | `IchimokuCloudStrategy` |
| 14 | Stochastic RSI | Oscillator | `StochRSIStrategy` |
| 15 | MACD Histogram Divergence | Counter-trend | `MACDHistDivergenceStrategy` |
| 16 | Market Regime Classifier | Regime-based | `MarketRegimeStrategy` |
| 17 | Open Interest Divergence | On-chain/Derivatives | `OpenInterestDivergenceStrategy` |

> The `--strategy NAME` flag only applies to swing mode (`--mode swing`). Intraday and spot run all registered strategies.

### Intraday Futures (6 strategies — `--mode intraday`)

| Strategy | Type |
|----------|------|
| `LiquiditySweepReversalStrategy` | Order-flow |
| `OpeningRangeBreakoutStrategy` | Breakout |
| `VWAPDeltaConfluenceStrategy` | VWAP / delta |
| `FairValueGapStrategy` | SMC / price action |
| `LiquidationCascadeMomentumStrategy` | Derivatives |
| `MicrostructureConsolidationBreakoutStrategy` | Microstructure |

### Spot Long-Term (7 strategies — `--mode spot`)

| Strategy | Type |
|----------|------|
| `MVRVZScoreCycleStrategy` | On-chain cycle |
| `PiCycleRainbowCompositeStrategy` | Composite cycle |
| `HalvingCyclePhaseAllocatorStrategy` | Macro cycle |
| `Top10MomentumRotationStrategy` | Momentum rotation |
| `MacroRegimePortfolioStrategy` | Macro |
| `OnChainAccumulationCompositeStrategy` | On-chain |
| `NVTSignalValuationStrategy` | On-chain valuation |

---

## CLI Reference

```
python main.py [options]

Mode selection:
  --mode {swing,backtest,intraday,spot,analyze}
                        swing / backtest  : run swing futures strategies (default)
                        intraday          : run intraday futures strategies
                        spot              : run long-term spot strategies
                        analyze           : run a mode and display rich analysis

  --analyze-mode {intraday,swing,spot,all}
                        Which mode(s) to analyze (default: swing)
                        Only used when --mode analyze is set.

Strategy selection:
  --strategy STRATEGY   Run a single swing strategy by class name
                        (e.g. IchimokuCloudStrategy)

Execution control:
  --no-parallel         Disable parallel execution
                        Required on some Windows setups
  --no-optimize         Skip Optuna optimisation — uses default params
                        (only applies to --strategy single-strategy runs)
  --no-walk-forward     Skip walk-forward validation
                        (only applies to --strategy single-strategy runs)
  --trials N            Override Optuna trial count for this run
                        (applies to all three mode runners)
  --skip-wf             Skip walk-forward validation — IS-only promotion gate
                        (applies to all three mode runners)
  --skip-sensitivity    Skip parameter sensitivity analysis
                        (applies to all three mode runners)

Export:
  --export {csv,html}   Export results:
                          csv  → results/<mode>_summary.csv
                          html → results/<mode>_report.html

Other:
  --config CONFIG       Path to settings YAML (default: config/settings.yaml)
  -h, --help            Show this message
```

### Common command combinations

| Goal | Command |
|------|---------|
| Fastest single-strategy test | `python main.py --strategy EMARibbonStrategy --no-optimize --no-walk-forward` |
| Full single-strategy pipeline | `python main.py --strategy IchimokuCloudStrategy` |
| All swing, fast scan | `python main.py --skip-wf --skip-sensitivity` |
| All swing, full pipeline (Linux) | `python main.py` |
| All swing, full pipeline (Windows) | `python main.py --no-parallel` |
| Intraday mode, fast | `python main.py --mode intraday --skip-wf --skip-sensitivity` |
| Spot mode, fast | `python main.py --mode spot --skip-wf --skip-sensitivity` |
| Rich analysis, all modes, export HTML | `python main.py --mode analyze --analyze-mode all --export html` |
| Quick 50-trial optimization | `python main.py --trials 50` |
| Debug / single-threaded | `python main.py --strategy EMARibbonStrategy --no-parallel --no-optimize --no-walk-forward` |

### Flag compatibility matrix

| Flag | `--mode swing --strategy` | `--mode swing` (all) | `--mode intraday` | `--mode spot` | `--mode analyze` |
|------|:---:|:---:|:---:|:---:|:---:|
| `--no-optimize` | ✅ | — | — | — | — |
| `--no-walk-forward` | ✅ | — | — | — | — |
| `--trials N` | — | ✅ | ✅ | ✅ | ✅ |
| `--skip-wf` | — | ✅ | ✅ | ✅ | ✅ |
| `--skip-sensitivity` | — | ✅ | ✅ | ✅ | ✅ |
| `--no-parallel` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `--export` | ✅ | ✅ | ✅ | ✅ | ✅ |

---

## Configuration

Edit `config/settings.yaml` for swing mode:

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

Each mode has its own YAML config:
- **Swing / legacy:** `config/settings.yaml`
- **Intraday:** `config/intraday.yaml`
- **Spot long-term:** `config/spot_longterm.yaml`

Full configuration reference: [`docs/BACKTEST.md`](docs/BACKTEST.md#3-configuration-reference)

---

## Output Files

All results are saved to `results/`:

| File | Contents | Created by |
|------|----------|------------|
| `swing_summary.csv` | One row per swing strategy, all metrics | `--mode swing` |
| `intraday_summary.csv` | One row per intraday strategy | `--mode intraday` |
| `spot_summary.csv` | One row per spot strategy | `--mode spot` |
| `{mode}_summary.csv` | Analyze mode export | `--mode analyze --export csv` |
| `{mode}_report.html` | Full HTML report with tables and panels | `--mode analyze --export html` |
| `{strategy}_fills_{ts}.csv` | Every trade: entry, exit, P&L, exit reason | Single-strategy run |
| `{strategy}_equity_{ts}.csv` | Equity at every candle | Single-strategy run |
| `{strategy}_equity_{ts}.png` | Equity curve chart | Single-strategy run |

---

## Running Tests

```bash
# Linux / macOS
pytest tests/ -v

# Windows
python -m pytest tests/ -v

# Quick smoke check
pytest --tb=short -q
```

The test suite covers the full engine including signal generation, risk engine, walk-forward, sensitivity, analysis display, and all three mode entrypoints.

---

## Documentation

| Document | Contents |
|----------|----------|
| [`docs/BACKTEST.md`](docs/BACKTEST.md) | Full engine reference — pipeline, all strategies, optimization, walk-forward, results interpretation, troubleshooting |
