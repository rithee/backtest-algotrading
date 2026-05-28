# Multi-Mode Backtest System — Design Spec
**Date:** 2026-05-28  
**Status:** Approved  
**Scope:** Intraday Futures + Swing Futures (new strategies) + Long-Term Spot  

---

## 1. Overview

Extend the existing single-mode swing futures backtest system into three independent, fully-featured backtest pipelines:

| Mode | Market | Symbols | Timeframes | Capital | Leverage |
|---|---|---|---|---|---|
| **Intraday** | Futures | BTC, ETH, SOL | 1m / 5m / 15m / 1h (MTF) | $100/strategy | 3× default, 5× max |
| **Swing (new)** | Futures | BTC, ETH | 4h + 1d | $100/strategy | 2× default, 5× max |
| **Long-Term Spot** | Spot | Top 10 by market cap | 1d + 1w | $1,000/strategy | 1× (no leverage) |

The existing 16 swing strategies and their orchestrator are **untouched**. All new work lives in new files.

---

## 2. Architecture

### 2.1 Directory Structure

```
backtest_test/
├── backtest/
│   ├── modes/
│   │   ├── __init__.py
│   │   ├── intraday.py              # entrypoint: runs strategies 1–6
│   │   ├── swing.py                 # entrypoint: runs strategies 7–12 + existing 16
│   │   └── spot_longterm.py         # entrypoint: runs strategies 13–19
│   ├── runner.py                    # shared — extended for spot + intraday modes
│   ├── optimizer.py                 # shared — unchanged
│   ├── walk_forward.py              # shared — unchanged
│   ├── orchestrator.py              # existing swing default — unchanged
│   ├── reporter.py                  # shared — extended for spot PnL reporting
│   └── display/                     # NEW — CLI display renderers
│       ├── __init__.py
│       ├── summary.py
│       ├── inspect.py
│       ├── trades.py
│       ├── calendar.py
│       ├── equity.py
│       ├── compare.py
│       ├── walkforward.py
│       └── export.py
│
├── crypto_bot/core/
│   ├── data/
│   │   ├── binance_history.py       # existing
│   │   ├── coinglass.py             # NEW — liquidations, OI heatmap
│   │   ├── glassnode.py             # NEW — on-chain metrics
│   │   ├── deribit.py               # NEW — options gamma / max pain
│   │   ├── macro.py                 # NEW — SPX, DXY, Gold, FRED yields
│   │   └── coinmarketcap.py         # NEW — top-10 market cap ranking
│   │
│   └── signals/strategies/
│       ├── intraday/                # NEW — strategies 1–6
│       │   ├── lsr.py               # LiquiditySweepReversal
│       │   ├── orb_sb.py            # OpeningRangeBreakout + SessionBias
│       │   ├── mvdc.py              # VWAPDeltaConfluence
│       │   ├── fvg.py               # FairValueGap
│       │   ├── lcm.py               # LiquidationCascadeMomentum
│       │   └── mcb.py               # MicrostructureConsolidationBreakout
│       ├── swing_new/               # NEW — strategies 7–12
│       │   ├── wyckoff.py           # WyckoffPhaseDetector
│       │   ├── ocsmd.py             # OnChainSmartMoneyDivergence
│       │   ├── ogp.py               # OptionsGammaMaxPain
│       │   ├── cam.py               # CrossAssetMomentumRegime
│       │   ├── ewa.py               # ElliottWaveAutomator
│       │   └── frsp.py              # FundingRateSqueezePredIctor
│       └── spot_longterm/           # NEW — strategies 13–19
│           ├── mzc.py               # MVRV_ZScore_Cycle
│           ├── pcr.py               # PiCycleRainbowComposite
│           ├── hcpa.py              # HalvingCyclePhaseAllocator
│           ├── trmr.py              # Top10RiskAdjustedMomentumRotation
│           ├── mrp.py               # MacroRegimePortfolio
│           ├── oac.py               # OnChainAccumulationComposite
│           └── nvt.py               # NVT_SignalValuation
│
├── config/
│   ├── settings.yaml                # existing swing default — unchanged
│   ├── intraday.yaml                # NEW
│   ├── swing_new.yaml               # NEW
│   └── spot_longterm.yaml           # NEW
│
├── backtest/
│   └── cli.py                       # NEW — CLI analysis interface
│
└── tests/
    ├── (existing tests — unchanged)
    ├── test_data_coinglass.py        # NEW
    ├── test_data_glassnode.py        # NEW
    ├── test_data_deribit.py          # NEW
    ├── test_data_macro.py            # NEW
    ├── test_data_coinmarketcap.py    # NEW
    ├── test_intraday_*.py            # NEW (6 files)
    ├── test_swing_new_*.py           # NEW (6 files)
    ├── test_spot_*.py                # NEW (7 files)
    ├── test_runner_spot.py           # NEW
    ├── test_runner_intraday.py       # NEW
    ├── test_cli.py                   # NEW
    ├── test_mode_intraday.py         # NEW (smoke)
    ├── test_mode_swing_new.py        # NEW (smoke)
    └── test_mode_spot_longterm.py    # NEW (smoke)
```

### 2.2 Design Principles

- **Zero regression**: existing swing system and 16 strategies are never modified
- **Shared infrastructure**: `runner.py`, `optimizer.py`, `walk_forward.py`, `sensitivity.py`, `reporter.py` are extended, not duplicated
- **Independent entrypoints**: each mode runs completely standalone
- **Graceful degradation**: missing API keys or failed fetches log a warning and skip — never crash the backtest

---

## 3. New Strategies

### 3.1 Intraday Futures (BTC, ETH, SOL)

| # | Name | File | Signal Logic | Key Data |
|---|---|---|---|---|
| 1 | LiquiditySweepReversal | `lsr.py` | Detects stop hunts: price sweeps above swing high / below swing low with volume spike + wick confirmation, then reverses | Binance OHLCV, CoinGlass liquidation heatmap |
| 2 | OpeningRangeBreakout + SessionBias | `orb_sb.py` | 4 crypto sessions (Asia 00-08, London 08-12, NY 13-17, NY Close 17-22 UTC). Trade breakout of first 30-min range per session, filtered by volume and session volatility | Binance OHLCV (1m/5m) |
| 3 | VWAPDeltaConfluence | `mvdc.py` | Intraday VWAP ± 1.5σ as S/R. Volume delta (cumulative buy vol − sell vol) for order flow confirmation. Enter when price deviates from VWAP + delta diverges + 1h trend confirms | Binance OHLCV + trade volume |
| 4 | FairValueGap | `fvg.py` | Detects 3-candle imbalances (gap between C1 high and C3 low). Gaps act as price magnets. Trade the fill with momentum confirmation on 5m, direction filter on 15m | Binance OHLCV |
| 5 | LiquidationCascadeMomentum | `lcm.py` | When a large liquidation cluster is swept, momentum accelerates. Enter in direction of cascade. Stop at swept level | CoinGlass liquidation clusters + Binance OHLCV |
| 6 | MicrostructureConsolidationBreakout | `mcb.py` | Identifies low-ATR consolidation periods (market coiling), trades breakouts confirmed by OI increase + positive volume delta (new money, not short covering) | Binance OHLCV + Binance futures OI |

### 3.2 Swing Futures — New Strategies (BTC, ETH)

| # | Name | File | Signal Logic | Key Data |
|---|---|---|---|---|
| 7 | WyckoffPhaseDetector | `wyckoff.py` | Automated Wyckoff methodology: detects accumulation (PS→SC→AR→ST→LPS) and distribution phases using price spread + volume relationships across 4h + 1d | Binance OHLCV + volume |
| 8 | OnChainSmartMoneyDivergence | `ocsmd.py` | Price makes new highs but exchange inflows increase → smart money distributing → short. Price at lows + exchange outflows → accumulating → long | CryptoQuant/Glassnode exchange netflow |
| 9 | OptionsGammaMaxPain | `ogp.py` | Deribit weekly options: Max Pain level as price magnet pre-expiry. Gamma Exposure (GEX) flips signal dealer hedging direction changes | Deribit options API |
| 10 | CrossAssetMomentumRegime | `cam.py` | Rolling 30d correlation of BTC vs SPX/Gold/DXY. Regime A (risk-on correlated): use SPX momentum. Regime B (decoupled): pure BTC momentum. Regime C (inverse): DXY inverse | yfinance (SPX, Gold, DXY) |
| 11 | ElliottWaveAutomator | `ewa.py` | Algorithmic wave counting via ZigZag pivots. Enters on confirmed Wave 3. Fibonacci extensions (1.618×, 2.618×) for targets. Wave 2 retracement for stops | Binance OHLCV (4h + 1d) |
| 12 | FundingRateSqueezePredIctor | `frsp.py` | Enhanced funding strategy: OI rising + funding deeply negative → short squeeze → long. OI falling + funding extreme positive → long squeeze → short. Uses rate-of-change of both, not just levels | Binance funding + OI |

### 3.3 Long-Term Spot (Top 10 by Market Cap)

| # | Name | File | Signal Logic | Key Data |
|---|---|---|---|---|
| 13 | MVRV_ZScore_Cycle | `mzc.py` | BTC's Market Value ÷ Realized Value Z-Score. Z < 0: max accumulate. 0–3: hold. 3–7: reduce. > 7: exit. Applied to BTC-weighted portfolio | Glassnode (MVRV, RV) |
| 14 | PiCycleRainbowComposite | `pcr.py` | Pi Cycle Top: 111-day MA crossing 2× 350-day MA = top signal. Rainbow Chart logarithmic regression bands give undervalued/overvalued zones. Composite score drives position sizing | Glassnode + computed from Binance 1w |
| 15 | HalvingCyclePhaseAllocator | `hcpa.py` | BTC ~1460-day cycles. Phase 1 (0–12m post-halving): accumulate top 10. Phase 2 (12–18m): max exposure. Phase 3 (18–24m): rotate to BTC-only + take profits. Phase 4 (24–36m): defensive. On-chain metrics confirm transitions | Binance 1w + Glassnode |
| 16 | Top10RiskAdjustedMomentumRotation | `trmr.py` | Monthly rotation: rank top 10 by 90-day Sharpe ratio. Hold top 4. Rebalance monthly. BTC dominance rising → reduce alts, increase BTC weight | Binance 1d + CoinMarketCap |
| 17 | MacroRegimePortfolio | `mrp.py` | Score macro: DXY trend, SPX momentum, real yields (FRED), Gold trend. Score ≥ 4/5: full top-10. 2–3: BTC + ETH only. < 2: cash/stablecoins | yfinance + FRED API |
| 18 | OnChainAccumulationComposite | `oac.py` | Score 6 on-chain signals: SOPR < 1, LTH supply increasing, exchange balance falling, miner outflow low, NUPL < 0, STH MVRV < 1. Score 5–6: max buy. 3–4: buy. 1–2: hold. 0: sell | Glassnode (SOPR, LTH, NUPL, exchange balance) |
| 19 | NVT_SignalValuation | `nvt.py` | Network Value ÷ On-Chain Transaction Volume (crypto P/E). NVT > 150 (smoothed 90d): overvalued → reduce. NVT < 45: undervalued → accumulate. Applied per coin in top 10 | Glassnode / CryptoQuant NVT per coin |

---

## 4. Data Layer

### 4.1 New Data Sources

| Module | Source | Fetches | Cache |
|---|---|---|---|
| `coinglass.py` | CoinGlass API | Liquidation heatmap levels, large clusters per symbol | Parquet, per symbol/day |
| `glassnode.py` | Glassnode API | MVRV, MVRV Z-Score, SOPR, NVT, NUPL, LTH supply, exchange balance, miner outflow | Parquet, per metric/coin |
| `deribit.py` | Deribit REST API | Options OI by strike, Max Pain, Gamma Exposure per expiry | Parquet, per expiry |
| `macro.py` | yfinance + FRED | SPX (`^GSPC`), DXY (`DX-Y.NYB`), Gold (`GC=F`), 10Y real yield (FRED: `DFII10`) | Parquet, per ticker |
| `coinmarketcap.py` | CoinMarketCap API | Top 10 coins by market cap, monthly snapshots | JSON, monthly |

### 4.2 Caching Strategy

All fetchers write to `data/external/{source}/{metric_or_symbol}.parquet`.

1. Check if cached file covers requested date range
2. If yes → load from cache (no API call)
3. If no → fetch, append to cache, return

### 4.3 API Keys (`.env`)

```
COINGLASS_API_KEY=...
GLASSNODE_API_KEY=...
DERIBIT_CLIENT_ID=...
DERIBIT_CLIENT_SECRET=...
CMC_API_KEY=...
# yfinance and FRED are free — no key needed
```

### 4.4 Extended `aux_data` Dict

```python
aux_data = {
    # existing
    "open_interest":    {symbol: pd.DataFrame},
    "funding_rate":     {symbol: pd.DataFrame},

    # new — intraday
    "liquidations":     {symbol: pd.DataFrame},   # CoinGlass

    # new — swing
    "exchange_netflow": {symbol: pd.DataFrame},   # Glassnode
    "options_gamma":    {symbol: pd.DataFrame},   # Deribit
    "macro":            pd.DataFrame,             # SPX, DXY, Gold, yields

    # new — long-term spot
    "onchain":          {symbol: pd.DataFrame},   # Glassnode
    "top10_mcap":       pd.DataFrame,             # CoinMarketCap monthly
}
```

Missing keys return `None`. Strategies handle `None` gracefully with a logged warning.

### 4.5 Graceful Degradation

If an API key is missing or a fetch fails:
- Log a `WARNING` with the missing source name
- Return `None` for that data key
- Strategy continues using available OHLCV data only
- Result is flagged as `partial_data: true` in the backtest output

---

## 5. Mode Configurations

### 5.1 `config/intraday.yaml`

```yaml
exchange:
  name: binance
  market: futures
  testnet: false

backtest:
  symbols: ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
  timeframes: ["1m", "5m", "15m", "1h"]
  primary_tf: "5m"
  confirm_tf: "15m"
  trend_tf: "1h"
  start_date: "2022-01-01"
  end_date: "2024-12-31"
  initial_capital_per_strategy: 100.0
  fees_pct: 0.0002            # limit/maker orders
  slippage_pct: 0.0003

leverage:
  max: 5
  default: 3

risk:
  max_position_pct: 0.40
  stop_loss_atr_multiplier: 1.0
  take_profit_atr_multiplier: 2.5
  max_drawdown_pct: 0.10
  max_daily_loss_pct: 0.03    # kill switch
  max_trades_per_day: 5
  session_filter: true

optimization:
  trials: 300
  sampler: "TPE"
  objective: "fee_adjusted_sharpe"
  min_trades_per_year: 200

walk_forward:
  train_months: 3
  test_months: 1
  step_months: 1

promotion_criteria:
  min_sharpe_oos: 1.0
  max_drawdown_pct: 0.15
  min_profit_factor: 1.3
  min_trades_per_year: 200
  max_is_oos_divergence: 2.0
```

### 5.2 `config/swing_new.yaml`

```yaml
exchange:
  name: binance
  market: futures
  testnet: false

backtest:
  symbols: ["BTCUSDT", "ETHUSDT"]
  timeframes: ["4h", "1d"]
  primary_tf: "4h"
  confirm_tf: "1d"
  start_date: "2020-01-01"
  end_date: "2024-12-31"
  initial_capital_per_strategy: 100.0
  fees_pct: 0.001
  slippage_pct: 0.0005

leverage:
  max: 5
  default: 2

risk:
  max_position_pct: 0.30
  stop_loss_atr_multiplier: 2.0
  take_profit_atr_multiplier: 3.5
  max_drawdown_pct: 0.20
  options_expiry_aware: true
  weekly_max_pain_target: true

optimization:
  trials: 500
  sampler: "TPE"
  objective: "composite"
  min_trades_per_year: 30

walk_forward:
  train_months: 12
  test_months: 3
  step_months: 3

promotion_criteria:
  min_sharpe_oos: 1.0
  max_drawdown_pct: 0.25
  min_profit_factor: 1.3
  min_trades_per_year: 30
  max_is_oos_divergence: 2.0
```

### 5.3 `config/spot_longterm.yaml`

```yaml
exchange:
  name: binance
  market: spot
  testnet: false

backtest:
  symbols:
    - "BTCUSDT"
    - "ETHUSDT"
    - "BNBUSDT"
    - "SOLUSDT"
    - "XRPUSDT"
    - "ADAUSDT"
    - "AVAXUSDT"
    - "DOGEUSDT"
    - "DOTUSDT"
    - "MATICUSDT"
  timeframes: ["1d", "1w"]
  primary_tf: "1w"
  confirm_tf: "1d"
  start_date: "2019-01-01"
  end_date: "2024-12-31"
  initial_capital_per_strategy: 1000.0
  fees_pct: 0.001
  slippage_pct: 0.001

leverage:
  max: 1
  default: 1

risk:
  max_position_pct: 0.25
  stop_loss_atr_multiplier: 4.0
  take_profit_atr_multiplier: 0     # signal-driven exit, no fixed TP
  max_drawdown_pct: 0.45
  rebalance_frequency: "monthly"
  min_holding_days: 30
  btc_dominance_filter: true

optimization:
  trials: 200
  sampler: "TPE"
  objective: "calmar_ratio"         # protects against deep cycle drawdowns
  min_trades_per_year: 6

walk_forward:
  train_months: 24
  test_months: 6
  step_months: 6

promotion_criteria:
  min_sharpe_oos: 0.7
  max_drawdown_pct: 0.45
  min_profit_factor: 1.5
  min_trades_per_year: 6
  max_is_oos_divergence: 2.0
```

---

## 6. Runner Extensions

### 6.1 Strategy Base Class Extension

```python
class BaseStrategy:
    # existing interface unchanged
    def default_params(self) -> dict: ...
    def generate_signal(self, candles: pd.DataFrame, aux_data: dict) -> Signal: ...

    # NEW: mode declaration (default = "swing" — existing strategies unaffected)
    @property
    def mode(self) -> str:
        return "swing"

    # NEW: multi-timeframe signal generation
    def generate_signal_mtf(
        self,
        candles_by_tf: dict[str, pd.DataFrame],
        aux_data: dict,
    ) -> Signal:
        # Default falls back to generate_signal() — existing strategies unaffected
        primary = list(candles_by_tf.values())[0]
        return self.generate_signal(primary, aux_data)
```

### 6.2 BacktestRunner Extension

```python
class BacktestRunner:
    def __init__(self, config: Config, mode: str = "swing"):
        self.mode = mode

    def run(
        self,
        strategy: BaseStrategy,
        candles_by_symbol: dict[str, pd.DataFrame],
        aux_data: dict | None = None,
        candles_by_tf: dict[str, dict[str, pd.DataFrame]] | None = None,
    ) -> BacktestResult:
        if self.mode == "spot_longterm":
            return self._run_spot(...)
        elif self.mode == "intraday":
            return self._run_intraday(...)
        else:
            return self._run_futures(...)    # existing path — unchanged
```

### 6.3 Spot Runner Logic

- No leverage applied
- No funding rate deducted
- No liquidation engine
- Position sizing in base asset units
- Monthly rebalancing hook for rotation strategies
- BTC dominance filter: skip alt signals when dominance rising above threshold
- `min_holding_days` enforced — exit signals ignored before minimum hold

### 6.4 Intraday Runner Logic

- Multi-timeframe candle loading (1m/5m/15m/1h per symbol)
- Session filter: signals only generated during active sessions
- Daily loss kill switch: if daily PnL < -3% → no more trades that day
- Max trades per day enforced per symbol
- Maker fee applied by default (limit orders)

---

## 7. Auto-Optimisation

Auto-optimisation applies to all three modes via the shared `optimizer.py`. Mode-specific differences:

| | Intraday | Swing (new) | Long-Term Spot |
|---|---|---|---|
| **Auto-trigger** | score ≤ 0 OR trades/yr < 200 | score ≤ 0 OR trades/yr < 30 | score ≤ 0 OR trades/yr < 6 |
| **Trials** | 300 | 500 | 200 |
| **Objective** | Fee-adjusted Sharpe | Composite | Calmar Ratio |
| **Walk-forward** | 3m/1m | 12m/3m | 24m/6m |
| **Workers** | 6 | 6 | 4 |

### Parameter Stability Guard

After optimisation, sensitivity analysis perturbs each parameter ±10%. If performance degrades > 30% on perturbation, the strategy is flagged as `brittle` and rejected regardless of Sharpe ratio. Applies to all modes.

---

## 8. CLI Results Interface

### 8.1 Commands

```bash
python -m backtest.cli summary      --mode intraday|swing|spot
python -m backtest.cli inspect      --strategy <name> [--mode <mode>]
python -m backtest.cli trades       --strategy <name> [--symbol <sym>] [--side long|short] [--sort pnl|date|duration]
python -m backtest.cli calendar     --strategy <name> [--year <yyyy>]
python -m backtest.cli drawdown     --strategy <name>
python -m backtest.cli equity       --strategy <name> | --mode <mode> [--combined]
python -m backtest.cli compare      --mode intraday|swing|spot|all
python -m backtest.cli walkforward  --strategy <name>
python -m backtest.cli export       --mode <mode> --format html|csv
```

### 8.2 Metrics Reported

**Performance:** Total Return %, Annualised Return %, Alpha vs BTC, Sharpe, Sortino, Calmar  
**Risk:** Max Drawdown %, Max DD Duration, Avg Drawdown, VaR 95%, CVaR 95%, Ann. Volatility, Skewness, Kurtosis  
**Trades:** Total trades, Trades/year, Win Rate %, Profit Factor, Expectancy, Avg Win %, Avg Loss %, Best/Worst trade, Avg Duration, Max Consecutive Wins/Losses  
**Breakdown:** Long vs Short split, Per-symbol performance, Monthly/yearly P&L calendar, Walk-forward window results  

### 8.3 Dependencies

```
rich>=13.0     # tables, panels, colour
plotext>=5.0   # ASCII equity curves in terminal
click>=8.0     # CLI argument parsing
```

### 8.4 Results Storage

Results saved to `results/{mode}/{strategy_name}/` after each run as parquet + JSON. CLI reads from disk — analysis does not require re-running the backtest.

---

## 9. Testing Strategy

### 9.1 Coverage Targets

| Component | Target |
|---|---|
| Strategy signal logic | 90%+ |
| Data fetchers | 85%+ |
| Runner extensions | 90%+ |
| CLI display | 70%+ |
| End-to-end smoke (1 per mode) | 3 tests |

### 9.2 Test Patterns

**Strategy tests:** test signal fires on canonical setup, no signal without confirmation, graceful degradation on missing aux data, default params are valid schema.

**Data fetcher tests:** mock API responses, verify caching (API called only once), graceful `None` return on missing key.

**Runner tests:** spot mode has no leverage, `min_holding_days` enforced, intraday kill switch fires after 3% daily loss, session filter blocks off-hours signals.

**CLI tests:** all commands exit with code 0 on pre-populated mock results.

**Smoke tests:** each mode entrypoint completes end-to-end on 30 days of synthetic candles.

### 9.3 New Fixtures (`conftest.py`)

```python
mock_glassnode_api()         # synthetic MVRV, SOPR, NVT
mock_coinglass_api()         # synthetic liquidation heatmap
mock_deribit_api()           # synthetic options OI + max pain
mock_macro_data()            # synthetic SPX, DXY, Gold, yields
minimal_intraday_candles()   # 30 days synthetic 1m/5m/15m/1h OHLCV
minimal_spot_candles()       # 2 years synthetic 1d/1w OHLCV
mock_results_dir(tmp_path)   # pre-populated results/ for CLI tests
```

### 9.4 Running Tests

```bash
pytest                                                          # all tests
pytest tests/test_intraday_*.py                                 # intraday only
pytest tests/test_spot_*.py                                     # spot only
pytest --cov=crypto_bot --cov=backtest --cov-report=term-missing
pytest -m "not slow"                                            # skip end-to-end
```

---

## 10. CLI Entry Points

```bash
# Run individual modes
python -m backtest.modes.intraday
python -m backtest.modes.swing
python -m backtest.modes.spot_longterm

# Or via main.py flags
python main.py --mode intraday
python main.py --mode swing
python main.py --mode spot
```

---

## 11. Capital & Leverage Summary

| Mode | Capital/strategy | Default leverage | Max leverage | Position size | Min notional |
|---|---|---|---|---|---|
| Intraday futures | $100 | 3× | 5× | 40% ($40 → $120 notional) | Clears BTC/ETH/SOL minimums |
| Swing futures | $100 | 2× | 5× | 30% ($30 → $60 notional) | Clears BTC/ETH minimums |
| Long-term spot | $1,000 | 1× | 1× | 25% ($250 notional) | Comfortable for all top-10 |

> **Note:** $100 is learning/testing capital. Scale to $500+ for intraday live trading to reduce fee drag (intraday fees ≈ 40% of capital/year at $100 with 200 trades/year).

---

## 12. Future Extensions

- Add more symbols to any mode without code changes (config only)
- Add new strategies by creating a file in the relevant `strategies/` subfolder and registering in the mode entrypoint
- Live trading adapter: mode entrypoints are designed to be reused by a live execution layer
