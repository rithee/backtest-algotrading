from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel


class RiskDecision(BaseModel):
    approved: bool
    strategy: str
    symbol: str
    direction: Literal["LONG", "SHORT"]
    entry_price: float
    quantity: float
    leverage: float
    stop_loss: float
    take_profit: float
    rejection_reason: Optional[str] = None
