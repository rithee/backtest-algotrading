# Phase 4: Web Dashboard + Telegram Bot Design Spec

**Date:** 2026-05-24
**Status:** Not started — can be built in parallel with Phase 3 (read-only dashboard) or after
**Prerequisite:** Phase 3 state format finalized (SQLite schema stable)

---

## Overview

A real-time web dashboard to monitor all strategies, view equity curves and trade history, and control the bot. A Telegram bot supplements the dashboard for mobile-first monitoring and control. Both are read/write — you can pause strategies, adjust risk limits, and trigger emergency stops from either interface.

---

## Architecture

```
Live SQLite DBs (Phase 3)
        │
        ▼
  DashboardAPI (FastAPI)
        ├── REST endpoints (historical data, config)
        ├── WebSocket endpoint (live equity stream)
        └── Command endpoints (pause, resume, stop)
        │
        ▼
  React frontend (single page, no build step needed)
        ├── EquityCurveChart (per strategy + combined)
        ├── TradeHistoryTable
        ├── StrategyStatusGrid
        ├── RiskControlPanel
        └── AlertFeed

TelegramBot (python-telegram-bot)
        ├── /status — current equity + P&L
        ├── /pnl [strategy] — detailed P&L breakdown
        ├── /trades [strategy] [n] — last N trades
        ├── /pause [strategy] — pause new entries
        ├── /resume [strategy] — resume entries
        ├── /stop — trigger emergency stop
        └── /report — send daily report PDF
```

---

## New Files

```
dashboard/
├── __init__.py
├── api.py               # FastAPI app: REST + WebSocket
├── schemas.py           # Pydantic response models for API
├── reader.py            # Read-only interface to StateStore DBs
└── static/
    ├── index.html       # Single-page app shell
    ├── app.js           # Vanilla JS (no framework, no build step)
    └── style.css
bot/
├── __init__.py
├── telegram_bot.py      # python-telegram-bot command handlers
└── commands.py          # Command implementations (call DashboardAPI internally)
config/
└── dashboard.yaml       # Dashboard + bot config
tests/
├── test_dashboard_api.py
└── test_telegram_bot.py
```

---

## 4.1 Configuration (`config/dashboard.yaml`)

```yaml
dashboard:
  host: "0.0.0.0"
  port: 8080
  data_dirs:
    paper: "data/paper"
    live: "data/live"
  refresh_interval_ms: 5000        # WebSocket push interval

telegram:
  enabled: true
  bot_token: ""                    # TELEGRAM_BOT_TOKEN env var
  chat_id: ""                      # TELEGRAM_CHAT_ID env var
  allowed_user_ids: []             # Empty = allow anyone in the chat
  command_cooldown_seconds: 5      # Prevent rapid-fire commands

security:
  api_key: ""                      # Bearer token for REST API (optional)
  cors_origins: ["http://localhost:8080"]
```

---

## 4.2 Dashboard API (`dashboard/api.py`)

FastAPI app. Reads directly from the SQLite files written by Phase 2/3. No writes to those files — all state changes go through the live runner via a command queue.

### REST Endpoints

```
GET  /api/strategies                         → list all strategies + status
GET  /api/strategies/{name}/equity           → equity curve (daily or hourly)
GET  /api/strategies/{name}/trades           → trade history (paginated)
GET  /api/strategies/{name}/metrics          → current metrics snapshot
GET  /api/portfolio/equity                   → combined portfolio equity curve
GET  /api/portfolio/metrics                  → combined metrics
GET  /api/portfolio/correlation              → return correlation matrix
POST /api/strategies/{name}/pause            → pause new entries
POST /api/strategies/{name}/resume           → resume entries
POST /api/emergency-stop                     → trigger emergency stop
GET  /api/alerts                             → recent alerts feed
```

### WebSocket Endpoint

```
WS   /ws/live                                → push equity updates every 5s
```

**Push message format:**
```json
{
  "timestamp": "2026-05-24T14:32:11Z",
  "strategies": {
    "EMARibbon": {
      "equity": 10247.32,
      "daily_pnl": 247.32,
      "open_positions": 1,
      "drawdown_pct": 0.023
    }
  },
  "portfolio": {
    "total_equity": 20230.15,
    "daily_pnl": 230.15,
    "drawdown_pct": 0.011
  }
}
```

---

## 4.3 Data Reader (`dashboard/reader.py`)

Read-only access to Phase 2/3 SQLite databases. No shared state with the live runner.

```python
class DashboardReader:
    def __init__(self, data_dir: str): ...

    def get_equity_curve(
        self, strategy: str, mode: str = "live", granularity: str = "1d"
    ) -> list[dict]:
        """
        Reads from data/{mode}/{strategy}.db equity_curve table.
        Resamples to requested granularity (1h, 4h, 1d).
        """

    def get_trades(
        self, strategy: str, mode: str = "live", offset: int = 0, limit: int = 50
    ) -> list[dict]:
        """Reads from fills table. Converts Fill rows to API-friendly dicts."""

    def get_current_metrics(self, strategy: str, mode: str = "live") -> dict:
        """
        Compute live metrics from equity curve + fills:
        - Current equity, daily P&L, total return
        - Sharpe (rolling 30d), drawdown, win rate, profit factor
        """

    def get_correlation_matrix(self, mode: str = "live") -> dict:
        """Compute pairwise return correlation across all strategies."""
```

---

## 4.4 Web Frontend (`dashboard/static/`)

No build pipeline. Vanilla JS + CSS delivered directly by FastAPI's static file mount. Loads on `http://localhost:8080`.

### Page Layout

```
┌──────────────────────────────────────────────────────────────────┐
│  🤖 Crypto Bot Dashboard          [LIVE] [PAPER] [BACKTEST]      │
├─────────────────────────┬────────────────────────────────────────┤
│  Strategy Status Grid   │  Portfolio Equity Curve                │
│  ┌──────────────────┐   │  ╭──────────────────────────────────╮  │
│  │ EMARibbon        │   │  │          /‾\/‾\     /‾\         │  │
│  │ $10,247 +$247    │   │  │         /     \   /   \         │  │
│  │ DD: 2.3% ▶ LIVE  │   │  │        /       \_/     \_       │  │
│  ├──────────────────┤   │  ╰──────────────────────────────────╯  │
│  │ TTMSqueeze       │   │  [1D] [1W] [1M] [3M] [ALL]            │
│  │ $9,983  -$17     │   ├────────────────────────────────────────┤
│  │ DD: 0.2% ▶ LIVE  │   │  Trade History                        │
│  └──────────────────┘   │  Strategy  Symbol  Dir  Entry  Exit   │
│                         │  EMARibbon BTCUSDT LONG 68450 71200   │
│  [⏸ Pause All]          │  TTMSqueeze ETHUSDT SHORT 3450  3200  │
│  [🛑 Emergency Stop]    │                                        │
└─────────────────────────┴────────────────────────────────────────┘
```

### Technology Choices

- **Charts:** [Chart.js](https://www.chartjs.org/) loaded from CDN — no local install
- **Tables:** Plain HTML tables with CSS styling
- **Live updates:** Native WebSocket API in browser
- **No React, no Webpack, no Node.js** — zero build tooling

### Key JS Components

```javascript
// app.js

// Equity curve chart (Chart.js)
function renderEquityCurve(canvasId, datasets) { ... }

// WebSocket live feed
const ws = new WebSocket("ws://localhost:8080/ws/live");
ws.onmessage = (event) => updateStrategyCards(JSON.parse(event.data));

// Strategy control
async function pauseStrategy(name) {
    await fetch(`/api/strategies/${name}/pause`, { method: "POST" });
}

async function emergencyStop() {
    if (!confirm("EMERGENCY STOP: close all positions?")) return;
    await fetch("/api/emergency-stop", { method: "POST" });
}
```

---

## 4.5 Telegram Bot (`bot/telegram_bot.py`)

Uses `python-telegram-bot` (v20+, native async).

### Commands

| Command | Description | Example |
|---------|-------------|---------|
| `/status` | Current equity + P&L for all strategies | `/status` |
| `/pnl [strategy]` | Detailed P&L breakdown | `/pnl EMARibbon` |
| `/trades [strategy] [n]` | Last N trades | `/trades TTMSqueeze 5` |
| `/metrics [strategy]` | Sharpe, drawdown, win rate | `/metrics EMARibbon` |
| `/pause [strategy\|all]` | Halt new entries | `/pause all` |
| `/resume [strategy\|all]` | Resume entries | `/resume EMARibbon` |
| `/stop` | Trigger emergency stop (requires confirm) | `/stop` |
| `/report` | Send daily summary | `/report` |
| `/help` | List commands | `/help` |

### Security

- Only respond to messages from `allowed_user_ids` (configured in `dashboard.yaml`)
- `/stop` requires a confirmation message within 30 seconds: "CONFIRM STOP"
- All commands are logged with user ID and timestamp

### Command Implementation

```python
async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    metrics = reader.get_current_metrics_all()
    text = format_status_table(metrics)
    await update.message.reply_text(text, parse_mode="Markdown")

async def stop_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "⚠️ Type CONFIRM STOP within 30 seconds to trigger emergency stop."
    )
    context.user_data["pending_stop"] = time.time()

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if context.user_data.get("pending_stop"):
        if time.time() - context.user_data["pending_stop"] < 30:
            if update.message.text == "CONFIRM STOP":
                await emergency_stop.trigger("Manual stop via Telegram")
                return
    context.user_data.pop("pending_stop", None)
```

### Response Formats

**`/status` response:**
```
📊 *Bot Status* — 2026-05-24 14:32 UTC

Strategy      Equity      Day P&L   DD
EMARibbon     $10,247     +$247     2.3%
TTMSqueeze    $9,983      -$17      0.2%
──────────────────────────────────────
Total         $20,230     +$230     1.3%

Mode: 🟢 LIVE
```

**`/trades EMARibbon 3` response:**
```
📋 *EMARibbon — Last 3 Trades*

1. LONG BTCUSDT
   Entry: $68,450 | Exit: $71,200 (TP)
   P&L: +$270.00 (+2.7%) | 2026-05-22

2. SHORT ETHUSDT
   Entry: $3,820 | Exit: $3,750 (TP)
   P&L: +$140.00 (+3.7%) | 2026-05-20

3. LONG BTCUSDT
   Entry: $66,100 | Exit: $64,890 (SL)
   P&L: -$121.00 (-1.8%) | 2026-05-18
```

---

## 4.6 Push Notifications (Proactive Alerts)

The `TelegramNotifier` from Phase 2 is extended to push these events automatically:

| Event | Trigger | Message |
|-------|---------|---------|
| Trade fill | Every fill | Entry/exit details + updated equity |
| Drawdown alert | DD > `drawdown_alert_pct` | Strategy + drawdown % |
| Daily summary | Midnight UTC | Full P&L table |
| Strategy halted | Drawdown guard triggers | Which strategy + reason |
| Phase 3 gate result | After 4 weeks live | Pass/fail + metrics |
| Error | Unhandled exception | Error message + stack trace (truncated) |
| Emergency stop | Any trigger | Reason + final P&L + positions closed |

---

## 4.7 CLI Integration

```bash
# Start dashboard server only (read-only, safe to run alongside live bot)
python main.py --mode dashboard --config config/dashboard.yaml

# Start Telegram bot only
python main.py --mode telegram --config config/dashboard.yaml

# Start both together
python main.py --mode dashboard --mode telegram --config config/dashboard.yaml
```

The dashboard process is separate from the live trading process — it reads from the same SQLite files but never writes to them.

---

## 4.8 Performance Considerations

- Dashboard reader opens SQLite in **read-only mode** (`?mode=ro` URI) to avoid locking
- Equity curve data is aggregated in Python (not SQL) to keep queries simple
- WebSocket updates are throttled to `refresh_interval_ms` (default 5s) — no need for sub-second updates
- Static files are served by FastAPI's `StaticFiles` mount — no separate web server needed

---

## Testing Strategy

| Test | What it checks |
|------|----------------|
| `test_dashboard_api.py` | All REST endpoints return correct shape; WebSocket push format |
| `test_telegram_bot.py` | Command parsing, `/stop` confirmation flow, unauthorized user rejection |
| `test_reader.py` | Metric calculations match expected values from known DB fixtures |

---

## Dependencies (Phase 4 additions)

```
fastapi              # Dashboard API server
uvicorn              # ASGI server for FastAPI
python-telegram-bot  # Telegram Bot API wrapper (v20+, async)
```

Chart.js loaded from CDN in `index.html` — not a Python dependency.

---

## Out of Scope (Phase 4)

- User authentication (single-user bot, API key is sufficient)
- Database migrations (schema is append-only; no migrations needed)
- Historical data browser (use results/ CSVs for deep analysis)
- Multi-user support
- Mobile app (Telegram bot covers mobile use case)
