from __future__ import annotations
from datetime import datetime
from pydantic import BaseModel


class Candle(BaseModel):
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    is_clean: bool = True


class FundingRate(BaseModel):
    symbol: str
    timestamp: datetime
    rate: float
