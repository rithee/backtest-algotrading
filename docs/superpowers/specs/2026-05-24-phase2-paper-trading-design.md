# Phase 2: Paper Trading Design Spec

**Date:** 2026-05-24
**Status:** Not started — gate: Phase 1 ≥1 strategy passes promotion criteria
**Prerequisite:** Phase 1 complete (backtest engine + at least 1 promoted strategy)

---

## Overview

Connect the backtest engine to live Binance Futures data and run promoted strategies in real time with simulated execution. No real money changes hands. The goal is to validate that live market behavior matches backtest assumptions before committing capital.

**Gate to Phase 3:** 2 consecutive weeks of paper trading where live metrics (Sharpe, drawdown, profit factor) are within 15% of the corresponding backtest OOS metrics.

---

## Architecture

```
binance_live.py (WebSocket + REST)
        │
        ▼ Candle (4h close event)
  asyncio Queue (per symbol)
        │
        ▼
  CandleAccumulator (builds 4h bars from 1m ticks)
        │
        ▼ complete 4h Candle
  SignalEngine (same BaseStrategy instances from Phase 1)
        │
        ▼ Signal
  RiskEngine (same engine, disk-persisted state)
        │
        ▼ RiskDecision
  PaperExecutionEngine (same engine, fills at next open)
        │
        ▼ Fill
  StateStore (disk SQLite, not in-memory)
        │
        ▼
  LiveReporter (terminal + Telegram alerts)
```

All engines are identical to Phase 1. Only the data source and state persistence change.

---

## New Files

```
crypto_bot/
├── core/
│   └── data/
│       └── binance_live.py         # WebSocket candle feed + REST bootstrap
├── live/
│   ├── __init__.py
│   ├── runner.py                   # Async live loop per strategy
│   ├── accumulator.py              # 1m tick → 4h OHLCV bar builder
│   ├── orchestrator.py             # Manages multiple live strategy runners
│   └── reporter.py                 # Real-time metrics display + Telegram alerts
├── notifications/
│   ├── __init__.py
│   └── telegram.py                 # Telegram Bot API wrapper
config/
└── paper.yaml                      # Paper trading config (overrides backtest dates/capital)
tests/
├── test_accumulator.py
├── test_live_runner.py
└── test_telegram.py
```

---

## 2.1 Configuration (`config/paper.yaml`)

```yaml
mode: paper

exchange:
  name: binance
  market: futures
  testnet: true                    # Use Binance testnet for paper (no real API needed)
  api_key: ""                      # Binance testnet API key
  api_secret: ""                   # Binance testnet API secret

paper:
  symbols: ["BTCUSDT", "ETHUSDT"]
  timeframe: "4h"
  initial_capital_per_strategy: 10000   # USDT per strategy (same as backtest)
  promoted_strategies:                  # Filled by Phase 1 output
    - name: "EMARibbon"
      params: {}                        # Best params from Phase 1
    - name: "TTMSqueeze"
      params: {}
  state_db_path: "data/paper/{strategy_name}.db"   # Disk persistence

notifications:
  telegram:
    enabled: false
    bot_token: ""                  # Set via env var TELEGRAM_BOT_TOKEN
    chat_id: ""                    # Set via env var TELEGRAM_CHAT_ID
    on_fill: true
    on_daily_summary: true
    on_drawdown_alert: true
    drawdown_alert_pct: 0.10       # Alert if drawdown > 10%

validation:
  min_paper_days: 14               # Minimum paper trading days before Phase 3 gate check
  max_metric_divergence: 0.15      # 15% max divergence from backtest OOS metrics
```

---

## 2.2 Live Data Engine (`core/data/binance_live.py`)

**Responsibility:** Provide a continuous stream of completed OHLCV candles for all configured symbols.

### Bootstrap (REST)

On startup, fetch the last 200 candles via REST to warm up indicator calculations (same `fetch_candles` from Phase 1). This prevents the "cold start" period where indicators don't have enough history.

```python
async def bootstrap(symbol: str, timeframe: str, n_candles: int = 200) -> list[Candle]:
    """Fetch recent history to warm up indicators before live data starts."""
```

### Live Feed (WebSocket)

Subscribe to Binance Futures `kline` stream for each symbol. Binance pushes candle updates every second; we only care about the `x: true` (candle closed) event.

```python
async def stream_candles(symbol: str, timeframe: str, queue: asyncio.Queue) -> None:
    """
    Connect to wss://fstream.binance.com/stream?streams=btcusdt@kline_4h
    Push completed Candle to queue when x==True.
    Auto-reconnect with exponential backoff on disconnect.
    """
```

**Reconnect logic:**
- On disconnect: wait 1s, 2s, 4s, 8s (cap at 30s)
- On reconnect: re-bootstrap last 5 candles to catch any missed closes
- Log all reconnects with timestamp

### Funding Rate Poller

Funding rates settle every 8 hours. Poll `/fapi/v1/fundingRate` every 30 minutes and push to a shared dict.

```python
async def poll_funding_rates(symbols: list[str], interval_seconds: int = 1800) -> None:
    """Background task: updates shared funding_rates dict every 30 minutes."""
```

---

## 2.3 Candle Accumulator (`live/accumulator.py`)

The 4h candle stream from Binance is already aggregated — Binance sends complete 4h bars directly. The accumulator handles two edge cases:

1. **Missed candle on reconnect:** If a 4h close was missed during a disconnect, insert it from REST bootstrap data.
2. **Out-of-order delivery:** Buffer candles by timestamp and emit in order.

```python
class CandleAccumulator:
    def __init__(self, symbol: str, timeframe: str, warmup_candles: list[Candle]):
        self.history: deque[Candle]    # Rolling buffer (last 500 candles)
        self.last_timestamp: datetime | None

    def ingest(self, candle: Candle) -> Candle | None:
        """
        Accept candle. Return it if it's a new, in-order close.
        Return None if duplicate or gap detected (gap logged).
        """

    @property
    def candles_df(self) -> pd.DataFrame:
        """Current history as DataFrame (for indicator calculation)."""
```

---

## 2.4 Live Runner (`live/runner.py`)

One `LiveStrategyRunner` instance per promoted strategy. Runs as a long-lived async task.

```python
class LiveStrategyRunner:
    def __init__(
        self,
        strategy: BaseStrategy,
        config: Config,
        state: StateStore,          # Disk-persisted
        risk: RiskEngine,
        execution: PaperExecutionEngine,
        reporter: LiveReporter,
    ): ...

    async def run(self, candle_queue: asyncio.Queue) -> None:
        """
        Loop:
          1. Wait for Candle from queue
          2. accumulator.ingest(candle)
          3. strategy.generate_signals(accumulator.candles_df, aux_data)
          4. For each signal: risk.evaluate(signal)
          5. For approved entries: execution.schedule_entry(decision)
          6. execution.process_candle(...)  → fills
          7. state.record_fill(fill) for each fill
          8. state.record_equity(timestamp, equity)
          9. reporter.on_candle(strategy, fills, equity)
        """

    async def shutdown(self) -> None:
        """Close all open positions at current market price via execution.close_at_end()."""
```

**Idempotency on restart:** On startup, `LiveStrategyRunner` reads open positions from the disk `StateStore`. If positions exist (from a previous run that crashed), it reconstructs `execution.open_positions` and `risk.open_positions` before entering the loop.

---

## 2.5 Live Orchestrator (`live/orchestrator.py`)

Coordinates all strategy runners and shared resources.

```python
class LiveOrchestrator:
    def __init__(self, config_path: str): ...

    async def start(self) -> None:
        """
        1. Load paper.yaml + promoted strategy params
        2. Bootstrap candle history for each symbol
        3. Start WebSocket feed tasks (one per symbol)
        4. Broadcast candles to per-strategy queues (fan-out)
        5. Start LiveStrategyRunner tasks (one per strategy)
        6. Start funding rate poller
        7. Start daily summary reporter (midnight UTC)
        8. Await asyncio.gather(*all_tasks)
        """

    async def stop(self) -> None:
        """Graceful shutdown: cancel tasks, close all positions."""
```

**Signal: SIGTERM/SIGINT → graceful shutdown.** Each runner calls `shutdown()` before exit.

**Fan-out pattern:** One WebSocket task per symbol pushes candles into a shared queue. The orchestrator reads from that queue and distributes to each strategy runner's queue. This avoids N WebSocket connections for N strategies on the same symbol.

```
BTC WebSocket → btc_queue → [fan-out] → runner_1.btc_queue
                                      → runner_2.btc_queue
                                      → ...
```

---

## 2.6 State Persistence (`core/state/store.py` — unchanged interface)

Phase 2 uses the same `StateStore` interface but with `persist=True` and a file path:

```python
store = StateStore(
    strategy_name="EMARibbon",
    persist=True,
    db_path="data/paper/EMARibbon.db"
)
```

On startup, `StateStore` loads existing fills and equity curve from disk, allowing the runner to resume after a restart without losing history.

---

## 2.7 Live Reporter (`live/reporter.py`)

Handles real-time output to terminal and Telegram.

```python
class LiveReporter:
    async def on_candle(self, strategy: str, fills: list[Fill], equity: float) -> None:
        """Print current equity + any new fills to terminal."""

    async def on_daily_summary(self) -> None:
        """
        Every midnight UTC:
        - Print per-strategy daily P&L, total equity, drawdown
        - Send Telegram message (if enabled)
        - Write daily snapshot to results/paper/YYYY-MM-DD.csv
        """

    async def on_drawdown_alert(self, strategy: str, drawdown_pct: float) -> None:
        """Called by runner when drawdown exceeds alert threshold."""
```

**Terminal output (live):**

```
[2026-05-24 12:00:04 UTC] EMARibbon | BTCUSDT | equity: $10,247.32 | DD: 2.3%
[2026-05-24 12:00:04 UTC] EMARibbon | FILL: LONG BTCUSDT @ 68,450.00 | SL: 67,100 | TP: 71,200
[2026-05-24 12:00:04 UTC] TTMSqueeze | BTCUSDT | equity: $9,983.15 | DD: 0.2%
```

---

## 2.8 Telegram Integration (`notifications/telegram.py`)

Simple HTTP wrapper over the Telegram Bot API. No third-party library needed.

```python
class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str): ...

    async def send(self, message: str) -> None:
        """POST to https://api.telegram.org/bot{token}/sendMessage"""

    async def send_fill_alert(self, fill: Fill) -> None:
        """Format fill as readable message and send."""

    async def send_daily_summary(self, results: dict[str, BacktestResult]) -> None:
        """Format daily P&L table and send."""
```

**Credentials:** Read from environment variables `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`. Never hardcoded in config files.

**Message format (fill alert):**
```
🟢 LONG ENTRY — EMARibbon
Symbol: BTCUSDT
Entry: $68,450.00
Stop Loss: $67,100.00 (-2.0%)
Take Profit: $71,200.00 (+4.0%)
Equity: $10,247.32
```

**Message format (daily summary):**
```
📊 Daily Summary — 2026-05-24

Strategy       Equity    Day P&L   DD
EMARibbon      $10,247   +$247     2.3%
TTMSqueeze     $9,983    -$17      0.2%
──────────────────────────────────────
Total          $20,230   +$230     1.3%
```

---

## 2.9 Phase 3 Gate Check

After `min_paper_days` (14 days), evaluate whether live paper metrics are within tolerance of backtest OOS metrics.

```python
def check_phase3_gate(
    live_metrics: BacktestResult,
    backtest_oos_metrics: BacktestResult,
    max_divergence: float = 0.15,
) -> tuple[bool, list[str]]:
    """
    Returns (passes, list_of_failures).
    Checks: sharpe_ratio, max_drawdown_pct, profit_factor, trades_per_year.
    Divergence = abs(live - oos) / abs(oos).
    """
```

**Pass criteria (all must hold):**
- Sharpe divergence < 15%
- Drawdown divergence < 15%
- Profit factor divergence < 15%
- Trade count divergence < 30% (noisier, wider tolerance)

If any strategy fails the gate, it is removed from the promoted set and does not proceed to Phase 3.

---

## 2.10 CLI Integration

```bash
# Start paper trading with promoted strategies
python main.py --mode paper --config config/paper.yaml

# Check current paper metrics vs backtest OOS
python main.py --mode paper --check-gate --config config/paper.yaml

# Graceful stop
kill -SIGTERM <pid>
```

---

## Testing Strategy

| Test | What it checks |
|------|----------------|
| `test_accumulator.py` | Dedup, gap detection, in-order emission |
| `test_live_runner.py` | Resume from persisted state, graceful shutdown |
| `test_telegram.py` | Message formatting (mock HTTP call) |
| Phase 3 gate check | Divergence math (known inputs/outputs) |

Tests use mock WebSocket events and synthetic candle queues — no real Binance connection needed.

---

## Dependencies (Phase 2 additions)

```
websockets         # Binance WebSocket feed
aiofiles           # Async file I/O for state persistence
```

Telegram uses `aiohttp` (already installed in Phase 1).

---

## Out of Scope (Phase 2)

- Real order placement (Phase 3)
- Web dashboard (Phase 4)
- Multi-timeframe data (future enhancement)
- Position modification / trailing stops (future enhancement)
