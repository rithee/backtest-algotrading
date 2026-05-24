"""
Optuna Bayesian optimizer for a single strategy.
Creates/resumes a study per strategy (SQLite storage).
Objective: composite_score = sharpe x (1 - max_drawdown) x profit_factor.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Type
import pandas as pd

from crypto_bot.core.config import Config
from crypto_bot.core.signals.base import BaseStrategy
from backtest.runner import BacktestRunner, BacktestResult


def optimize(
    strategy_cls: Type[BaseStrategy],
    candles_by_symbol: dict[str, pd.DataFrame],
    config: Config,
    aux_data: dict | None = None,
    n_trials: int | None = None,
) -> dict:
    """
    Run Optuna optimization for strategy_cls. Returns best params dict.
    Resumes from existing study if storage file exists.
    """
    try:
        import optuna
    except ImportError:
        raise ImportError("optuna is required: pip install optuna")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    n = n_trials if n_trials is not None else config.optimization.trials
    storage_template = config.optimization.study_storage
    storage_path = storage_template.format(strategy_name=strategy_cls.__name__)
    Path(storage_path).parent.mkdir(parents=True, exist_ok=True)
    storage_url = f"sqlite:///{storage_path}"

    runner = BacktestRunner(config)

    def objective(trial: "optuna.Trial") -> float:
        params = _sample_params(trial, strategy_cls)
        strategy = strategy_cls(params)
        result = runner.run(strategy, candles_by_symbol, aux_data)

        if result.trades_per_year < config.optimization.min_trades_per_year:
            return -999.0

        return result.composite_score

    study = optuna.create_study(
        study_name=strategy_cls.__name__,
        storage=storage_url,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=20),
    )
    study.optimize(objective, n_trials=n, n_jobs=1, show_progress_bar=True)
    return study.best_params


def _sample_params(trial: "optuna.Trial", strategy_cls: Type[BaseStrategy]) -> dict:
    """Sample params from param_space using Optuna trial suggestions."""
    dummy = strategy_cls.__new__(strategy_cls)
    dummy.params = {}
    space = dummy.param_space

    params: dict = {}
    for key, spec in space.items():
        if len(spec) == 3 and spec[2] == "int":
            params[key] = trial.suggest_int(key, int(spec[0]), int(spec[1]))
        elif len(spec) == 2 and isinstance(spec[0], list):
            params[key] = trial.suggest_categorical(key, spec[0])
        else:
            params[key] = trial.suggest_float(key, float(spec[0]), float(spec[1]))
    return params
