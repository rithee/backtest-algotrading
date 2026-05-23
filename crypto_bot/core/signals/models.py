from __future__ import annotations
from datetime import datetime
from typing import Literal
from pydantic import BaseModel


class Signal(BaseModel):
    strategy: str
    symbol: str
    timestamp: datetime
    direction: Literal["LONG", "SHORT", "EXIT_LONG", "EXIT_SHORT"]
    strength: float  # confirming_conditions / total_conditions, 0.0–1.0
    close_price: float
    atr: float
    reason: list[str]
