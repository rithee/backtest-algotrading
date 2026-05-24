"""
In-memory state store for backtest results.
Persists to SQLite when persist=True (--persist CLI flag).
Tracks: fills, equity curve.
"""
from __future__ import annotations
import sqlite3
from datetime import datetime
from typing import Optional
from crypto_bot.core.execution.models import Fill


class StateStore:
    def __init__(self, strategy_name: str, persist: bool = False, db_path: Optional[str] = None) -> None:
        self.strategy_name = strategy_name
        self._fills: list[Fill] = []
        self._equity_curve: list[tuple[datetime, float]] = []

        if persist:
            path = db_path or f"data/studies/{strategy_name}_state.db"
            self._conn = sqlite3.connect(path)
        else:
            self._conn = sqlite3.connect(":memory:")
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS fills (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy TEXT, symbol TEXT, timestamp TEXT,
                direction TEXT, entry_price REAL, exit_price REAL,
                quantity REAL, leverage REAL, fees_paid REAL,
                slippage_paid REAL, pnl REAL, exit_reason TEXT
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS equity_curve (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT, equity REAL
            )
        """)
        self._conn.commit()

    def record_fill(self, fill: Fill) -> None:
        self._fills.append(fill)
        self._conn.execute(
            "INSERT INTO fills VALUES (NULL,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                fill.strategy, fill.symbol, str(fill.timestamp),
                fill.direction, fill.entry_price, fill.exit_price,
                fill.quantity, fill.leverage, fill.fees_paid,
                fill.slippage_paid, fill.pnl, fill.exit_reason,
            ),
        )
        self._conn.commit()

    def record_equity(self, timestamp: datetime, equity: float) -> None:
        self._equity_curve.append((timestamp, equity))
        self._conn.execute(
            "INSERT INTO equity_curve VALUES (NULL,?,?)",
            (str(timestamp), equity),
        )
        self._conn.commit()

    @property
    def fills(self) -> list[Fill]:
        return list(self._fills)

    @property
    def equity_curve(self) -> list[tuple[datetime, float]]:
        return list(self._equity_curve)

    def close(self) -> None:
        self._conn.close()
