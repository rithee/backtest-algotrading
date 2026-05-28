# Phase 3: Live Execution Design Spec

**Date:** 2026-05-24
**Status:** Not started — gate: Phase 2 paper trading passes 15% divergence check
**Prerequisite:** Phase 2 complete; ≥1 strategy passes Phase 3 gate check

---

## Overview

Replace simulated fills with real Binance Futures order placement. The execution engine swaps from `PaperExecutionEngine` to `LiveExecutionEngine` — all other layers (strategy, risk, state) are unchanged. Start with minimum viable position sizes, strict daily loss limits, and a dead-man's switch.

**Capital:** Start with $500–$1,000 USDT per strategy (small). Scale up only after 4 weeks of live performance within 15% of paper metrics.

---

## Architecture

```
(same as Phase 2, only execution layer changes)

RiskDecision
     │
     ▼
LiveExecutionEngine          ←── replaces PaperExecutionEngine
     │
     ├── POST /fapi/v1/order  (entry: MARKET order)
     ├── POST /fapi/v1/order  (SL: STOP_MARKET order)
     ├── POST /fapi/v1/order  (TP: TAKE_PROFIT_MARKET order)
     │
     ▼
BinanceAccountReconciler     (verifies order state matches local state)
     │
     ▼
StateStore (disk SQLite)
```

---

## New Files

```
crypto_bot/
├── core/
│   └── execution/
│       └── live.py             # Real Binance Futures order placement
├── live/
│   ├── reconciler.py           # Reconcile local positions vs Binance account
│   └── emergency_stop.py       # Dead-man's switch
config/
└── live.yaml                   # Live config (extends paper.yaml)
tests/
├── test_live_execution.py      # Mocked Binance REST calls
└── test_reconciler.py
```

---

## 3.1 Configuration (`config/live.yaml`)

```yaml
mode: live

exchange:
  name: binance
  market: futures
  testnet: false
  api_key: ""                    # Set via env var BINANCE_API_KEY
  api_secret: ""                 # Set via env var BINANCE_API_SECRET

live:
  symbols: ["BTCUSDT", "ETHUSDT"]
  timeframe: "4h"
  initial_capital_per_strategy: 500    # USDT — start small
  max_capital_per_strategy: 5000       # Hard cap on any single strategy
  promoted_strategies:
    - name: "EMARibbon"
      params: {}                        # Best params carried from Phase 2

risk:
  max_daily_loss_pct: 0.05       # 5% daily loss → halt all strategies for 24h
  max_open_positions: 2          # Per strategy, across all symbols
  emergency_stop_drawdown: 0.15  # 15% total drawdown → kill switch

notifications:
  telegram:
    enabled: true
    bot_token: ""                # TELEGRAM_BOT_TOKEN env var
    chat_id: ""                  # TELEGRAM_CHAT_ID env var
    on_fill: true
    on_daily_summary: true
    on_error: true
    on_emergency_stop: true

reconciliation:
  interval_seconds: 60           # Check account state every 60s
  max_drift_pct: 0.02            # Alert if local vs Binance position differs > 2%
```

---

## 3.2 Live Execution Engine (`core/execution/live.py`)

Implements the same interface as `PaperExecutionEngine` so the live runner needs no changes.

```python
class LiveExecutionEngine:
    def __init__(self, strategy_name: str, config: Config, client: BinanceClient): ...

    def schedule_entry(self, decision: RiskDecision) -> None:
        """Queue entry for next candle open (same timing as paper)."""

    def process_candle(
        self,
        symbol: str,
        timestamp: datetime,
        open_: float,
        high: float,
        low: float,
        close: float,
    ) -> list[Fill]:
        """
        On candle open:
          1. Execute any queued entries via MARKET order
          2. Place OCO bracket: STOP_MARKET (SL) + TAKE_PROFIT_MARKET (TP)
          3. Check if any existing SL/TP orders were filled (via order status poll)
          4. Return Fill objects for completed trades
        """

    def close_at_end(self, symbol: str, timestamp: datetime, close: float) -> Fill | None:
        """Emergency close: MARKET order to flatten position."""

    def schedule_exit(self, symbol: str) -> None:
        """Signal-driven exit: cancel bracket, place MARKET close."""
```

### Order Types Used

| Action | Binance Order Type | Notes |
|--------|-------------------|-------|
| Entry | `MARKET` | Fills immediately at market price |
| Stop loss | `STOP_MARKET` | GTC, tied to position |
| Take profit | `TAKE_PROFIT_MARKET` | GTC, tied to position |
| Signal exit | `MARKET` | Cancel bracket first, then close |
| Emergency close | `MARKET` | Cancel all open orders first |

### Order Placement Flow

```python
async def _place_entry(self, decision: RiskDecision) -> Fill:
    # 1. Set leverage
    await client.set_leverage(symbol, decision.leverage)

    # 2. Place MARKET entry
    order = await client.place_order(
        symbol=decision.symbol,
        side="BUY" if decision.direction == "LONG" else "SELL",
        type="MARKET",
        quantity=decision.quantity,
        reduce_only=False,
    )
    fill_price = float(order["avgPrice"])

    # 3. Place SL (STOP_MARKET)
    await client.place_order(
        symbol=decision.symbol,
        side="SELL" if decision.direction == "LONG" else "BUY",
        type="STOP_MARKET",
        stop_price=decision.stop_loss,
        close_position=True,
    )

    # 4. Place TP (TAKE_PROFIT_MARKET)
    await client.place_order(
        symbol=decision.symbol,
        side="SELL" if decision.direction == "LONG" else "BUY",
        type="TAKE_PROFIT_MARKET",
        stop_price=decision.take_profit,
        close_position=True,
    )

    return Fill(...)
```

### Binance Client Wrapper

Thin wrapper around `python-binance` futures REST. Handles:
- Request signing (HMAC SHA256)
- Rate limit tracking (weight per minute)
- Error parsing (Binance error codes → typed exceptions)

```python
class BinanceClient:
    async def place_order(self, **kwargs) -> dict: ...
    async def cancel_order(self, symbol: str, order_id: int) -> dict: ...
    async def get_order(self, symbol: str, order_id: int) -> dict: ...
    async def get_open_orders(self, symbol: str) -> list[dict]: ...
    async def get_account(self) -> dict: ...
    async def set_leverage(self, symbol: str, leverage: int) -> dict: ...
```

---

## 3.3 Account Reconciler (`live/reconciler.py`)

Runs every 60 seconds. Compares local `StateStore` open positions against actual Binance account positions. Detects:

1. **Drift:** Local says position open, Binance says closed (SL/TP hit between poll cycles)
2. **Ghost positions:** Binance has a position not tracked locally (manual intervention, API error)
3. **Size mismatch:** Position size differs by > 2% (partial fill, rounding)

```python
class AccountReconciler:
    async def reconcile(self) -> list[ReconciliationIssue]:
        """
        1. Fetch positions from Binance account
        2. Compare with local StateStore
        3. Sync local state to match Binance (Binance is source of truth)
        4. Alert on ghost positions or unexpected size mismatches
        """
```

**On SL/TP fill detected by reconciler:**
1. Construct a synthetic `Fill` from the Binance order data
2. Record it in `StateStore`
3. Update `RiskEngine.open_positions`
4. Send Telegram alert

---

## 3.4 Emergency Stop (`live/emergency_stop.py`)

Three triggers, all lead to the same outcome: cancel all open orders, close all positions at market, halt the bot.

```python
class EmergencyStop:
    def __init__(self, client: BinanceClient, notifier: TelegramNotifier): ...

    async def trigger(self, reason: str) -> None:
        """
        1. Log reason with timestamp
        2. Cancel ALL open orders across all symbols
        3. Close ALL open positions with MARKET orders
        4. Send Telegram alert with reason + final P&L
        5. Write emergency_stop.json (prevents restart without manual review)
        6. Exit process
        """
```

**Triggers:**
1. **Daily loss limit:** `LiveOrchestrator` monitors daily P&L; calls `trigger()` if loss > `max_daily_loss_pct`
2. **Total drawdown:** `RiskEngine` detects `peak_equity - current_equity > emergency_stop_drawdown`; calls `trigger()`
3. **SIGTERM/SIGINT:** Clean shutdown that closes all positions (not an emergency, but same close-all logic)

**Restart guard:** `emergency_stop.json` must be manually deleted before the bot will start again. This forces a human review of what went wrong.

```json
{
  "timestamp": "2026-05-24T14:32:11Z",
  "reason": "Daily loss limit exceeded: -5.3% (limit: 5.0%)",
  "final_equity": {"EMARibbon": 9470.00, "TTMSqueeze": 9850.00},
  "positions_closed": ["BTCUSDT LONG", "ETHUSDT SHORT"]
}
```

---

## 3.5 Risk Controls Summary

| Control | Where | Behavior |
|---------|--------|----------|
| Max position size | RiskEngine (Phase 1) | Reject oversized entries |
| Leverage cap | RiskEngine (Phase 1) | Never exceed config max |
| ATR stop loss | RiskEngine (Phase 1) | SL placed on every entry |
| Per-strategy drawdown guard | RiskEngine (Phase 1) | Halt new entries for that strategy |
| Correlation guard | RiskEngine (Phase 1) | Block correlated simultaneous entries |
| Daily loss limit | LiveOrchestrator (Phase 3) | Halt ALL strategies for 24h |
| Total drawdown kill switch | EmergencyStop (Phase 3) | Close all, halt, require manual restart |
| Max open positions | LiveExecutionEngine (Phase 3) | Cap at 2 per strategy |

---

## 3.6 Live Runner Changes

`LiveStrategyRunner` from Phase 2 is reused with one constructor change:

```python
runner = LiveStrategyRunner(
    strategy=strategy,
    config=config,
    state=StateStore(persist=True, db_path="data/live/{strategy_name}.db"),
    risk=RiskEngine(...),
    execution=LiveExecutionEngine(client=binance_client),   # ← only change
    reporter=LiveReporter(...),
)
```

---

## 3.7 Scale-Up Gate

After 4 weeks of live trading, check if live metrics are within 15% of Phase 2 paper metrics:

```python
def check_scaleup_gate(
    live_metrics: BacktestResult,
    paper_metrics: BacktestResult,
    max_divergence: float = 0.15,
) -> tuple[bool, list[str]]:
    """Same divergence check as Phase 2 gate, but paper vs live."""
```

If a strategy passes:
- Double `initial_capital_per_strategy` (up to `max_capital_per_strategy`)
- Telegram notification sent

---

## 3.8 CLI Integration

```bash
# Start live trading
python main.py --mode live --config config/live.yaml

# Run preflight checks (verify API connectivity, balance, order placement on testnet)
python main.py --mode live --preflight --config config/live.yaml

# Check live vs paper divergence gate
python main.py --mode live --check-gate --config config/live.yaml

# Emergency stop (manual trigger)
python main.py --mode live --emergency-stop --config config/live.yaml
```

**Preflight checks** (run before any live session):
1. Ping Binance Futures REST and WebSocket
2. Verify API key has futures trading permissions
3. Check account USDT balance >= configured capital
4. Test order placement on testnet (place + cancel a tiny order)
5. Verify Telegram bot is reachable (send test message)

---

## Testing Strategy

| Test | What it checks |
|------|----------------|
| `test_live_execution.py` | Order placement flow with mocked HTTP |
| `test_reconciler.py` | Drift detection, ghost position handling |
| Emergency stop | Trigger conditions, cancel-all logic (mocked) |
| Scale-up gate | Divergence math (same as Phase 2) |

No live Binance connection needed for tests — all external calls are mocked.

---

## Dependencies (Phase 3 additions)

```
python-binance     # Already installed in Phase 1 (used for data fetching)
```

No new dependencies — `python-binance` covers order placement.

---

## Security

- API keys read exclusively from environment variables (`BINANCE_API_KEY`, `BINANCE_API_SECRET`)
- Keys are never logged, never written to files, never included in error messages
- Minimum required API permissions: `Futures Trading` only (no spot, no withdrawals)
- IP whitelist recommended on Binance API key settings

---

## Out of Scope (Phase 3)

- Spot trading (futures only)
- Trailing stop-loss (future enhancement)
- Partial position scaling (future enhancement)
- On-chain data sources (future enhancement)
- Web dashboard (Phase 4)
