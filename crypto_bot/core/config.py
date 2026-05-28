from __future__ import annotations
from pathlib import Path
from typing import Optional
from pydantic import BaseModel
import yaml


class ExchangeConfig(BaseModel):
    name: str = "binance"
    market: str = "futures"
    testnet: bool = False


class BacktestConfig(BaseModel):
    symbols: list[str] = ["BTCUSDT", "ETHUSDT"]
    timeframe: str = "4h"                        # kept for backward compat
    timeframes: list[str] = []                   # if non-empty, overrides timeframe
    primary_tf: str = ""                         # signal generation TF; defaults to timeframe
    confirm_tf: str = ""                         # direction filter TF
    trend_tf: str = ""                           # bias filter TF
    start_date: str = "2020-01-01"
    end_date: str = "2024-12-31"
    initial_capital_per_strategy: float = 10000.0
    fees_pct: float = 0.001
    slippage_pct: float = 0.0005

    @property
    def active_timeframes(self) -> list[str]:
        """Returns the effective list of timeframes to load."""
        return self.timeframes if self.timeframes else [self.timeframe]

    @property
    def active_primary_tf(self) -> str:
        """Returns the effective primary timeframe."""
        return self.primary_tf if self.primary_tf else self.timeframe


class LeverageConfig(BaseModel):
    max: int = 3
    default: int = 1


class RiskConfig(BaseModel):
    max_position_pct: float = 0.10
    stop_loss_atr_multiplier: float = 2.0
    take_profit_atr_multiplier: float = 3.0
    max_drawdown_pct: float = 0.20
    max_correlated_positions: int = 2
    correlation_lookback_days: int = 30
    correlation_threshold: float = 0.75
    # intraday-specific
    max_daily_loss_pct: float = 0.0      # 0 = disabled; 0.03 = stop after 3% daily loss
    max_trades_per_day: int = 0          # 0 = disabled
    session_filter: bool = False         # only trade during active crypto sessions
    # swing new
    options_expiry_aware: bool = False
    weekly_max_pain_target: bool = False
    # spot long-term
    rebalance_frequency: str = ""        # "monthly" | "" = disabled
    min_holding_days: int = 0            # 0 = disabled
    btc_dominance_filter: bool = False


class OptimizationConfig(BaseModel):
    trials: int = 500
    sampler: str = "TPE"
    objective: str = "composite"
    min_trades_per_year: int = 30
    study_storage: str = "data/studies/{strategy_name}.db"


class WalkForwardConfig(BaseModel):
    train_months: int = 12
    test_months: int = 3
    step_months: int = 3


class PromotionCriteria(BaseModel):
    min_sharpe_oos: float = 1.0
    max_drawdown_pct: float = 0.25
    min_profit_factor: float = 1.3
    min_trades_per_year: int = 30
    max_is_oos_divergence: float = 2.0


class Config(BaseModel):
    exchange: ExchangeConfig = ExchangeConfig()
    backtest: BacktestConfig = BacktestConfig()
    leverage: LeverageConfig = LeverageConfig()
    risk: RiskConfig = RiskConfig()
    optimization: OptimizationConfig = OptimizationConfig()
    walk_forward: WalkForwardConfig = WalkForwardConfig()
    promotion_criteria: PromotionCriteria = PromotionCriteria()
    mode: str = "swing"                  # "intraday" | "swing" | "spot_longterm"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load config from a YAML file path."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data or {})


def load_config(path: str | Path = "config/settings.yaml") -> Config:
    with open(path) as f:
        data = yaml.safe_load(f)
    return Config.model_validate(data or {})
