from __future__ import annotations
from datetime import datetime
from typing import Literal, Optional
from pydantic import BaseModel


class Position(BaseModel):
    strategy: str
    symbol: str
    direction: Literal["LONG", "SHORT"]
    entry_price: float
    quantity: float
    leverage: float
    stop_loss: float
    take_profit: float
    entry_timestamp: datetime
    fees_paid: float


class Fill(BaseModel):
    strategy: str
    symbol: str
    timestamp: datetime
    direction: Literal["LONG", "SHORT"]
    entry_price: float
    exit_price: Optional[float] = None
    quantity: float
    leverage: float
    fees_paid: float
    slippage_paid: float
    pnl: Optional[float] = None
    exit_reason: Optional[Literal["sl", "tp", "signal", "liquidation", "end_of_data"]] = None
