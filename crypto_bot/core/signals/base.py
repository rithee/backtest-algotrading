"""BaseStrategy ABC — every strategy implements this contract."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
import pandas as pd
from crypto_bot.core.signals.models import Signal


class BaseStrategy(ABC):
    def __init__(self, params: dict[str, Any]) -> None:
        self.params = params

    @abstractmethod
    def generate_signals(
        self,
        candles: pd.DataFrame,
        aux_data: dict[str, Any] | None = None,
    ) -> list[Signal]:
        """
        Generate signals from a candle DataFrame for ONE symbol.
        candles columns: symbol, timestamp, open, high, low, close, volume, is_clean.
        aux_data may contain 'funding_rates': pd.DataFrame (symbol, timestamp, rate).
        Returns list of Signal objects — one per candle that triggers a signal.
        Signals are generated on candle CLOSE. Fills happen at next candle OPEN (enforced by runner).
        No side effects. Called once per symbol per backtest run.
        """

    @property
    @abstractmethod
    def param_space(self) -> dict[str, tuple]:
        """
        Optuna search space definition.
        Format:
          (low, high)          → float suggest_float
          (low, high, "int")   → int suggest_int
          ([v1, v2], "cat")    → categorical suggest_categorical
        """

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def default_params(self) -> dict[str, Any]:
        """Returns midpoint/first-value of each param as sensible defaults."""
        defaults: dict[str, Any] = {}
        for key, space in self.param_space.items():
            if len(space) == 2 and isinstance(space[0], list):
                defaults[key] = space[0][0]
            elif len(space) == 3 and space[2] == "int":
                defaults[key] = int((space[0] + space[1]) / 2)
            elif len(space) == 2 and isinstance(space[0], (int, float)):
                defaults[key] = float((space[0] + space[1]) / 2)
            else:
                defaults[key] = space[0]
        return defaults

    @property
    def mode(self) -> str:
        """
        Declares which trading mode this strategy is designed for.
        Values: 'intraday' | 'swing' | 'spot_longterm'
        Default is 'swing' so all existing strategies are unaffected.
        Override in new strategies.
        """
        return "swing"

    def generate_signals_mtf(
        self,
        candles_by_tf: dict[str, "pd.DataFrame"],
        aux_data: dict[str, Any] | None = None,
    ) -> list["Signal"]:
        """
        Multi-timeframe signal generation.
        Default implementation uses the first (primary) timeframe's candles,
        delegating to generate_signals() — backward compatible with all existing strategies.
        Override in strategies that need cross-timeframe logic.

        Args:
            candles_by_tf: e.g. {'5m': df, '15m': df, '1h': df}
            aux_data:      same aux_data dict as generate_signals()

        Returns:
            List of Signal objects.
        """
        primary_candles = next(iter(candles_by_tf.values()))
        return self.generate_signals(primary_candles, aux_data)
