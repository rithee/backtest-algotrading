"""
Walk-forward validation.
Splits data into rolling windows (train/test), optimises on train, evaluates on test.
Returns OOS metrics as the honest performance estimate.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Type
import pandas as pd

from crypto_bot.core.config import Config
from crypto_bot.core.signals.base import BaseStrategy
from backtest.runner import BacktestRunner, BacktestResult
from backtest.optimizer import optimize


@dataclass
class WalkForwardWindow:
    window_idx: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    best_params: dict
    oos_result: BacktestResult


@dataclass
class WalkForwardResult:
    strategy_name: str
    windows: list[WalkForwardWindow]
    oos_sharpe: float = 0.0
    oos_max_drawdown: float = 0.0
    oos_profit_factor: float = 0.0
    consistency_score: float = 0.0          # fraction of windows with positive total return
    is_oos_divergence: float = 0.0          # in-sample sharpe / out-of-sample sharpe


def run_walk_forward(
    strategy_cls: Type[BaseStrategy],
    candles_by_symbol: dict[str, pd.DataFrame],
    config: Config,
    aux_data: dict | None = None,
    n_trials_per_window: int = 50,          # fewer trials per window for speed
    objective_fn=None,
) -> WalkForwardResult:
    """Run walk-forward validation. Returns WalkForwardResult."""
    wf = config.walk_forward
    runner = BacktestRunner(config, mode=config.mode)
    windows: list[WalkForwardWindow] = []

    # Get full date range from candles — use vectorised min/max, never iterrows
    if not candles_by_symbol:
        return WalkForwardResult(strategy_name=strategy_cls.__name__, windows=[])
    all_ts = pd.concat([df["timestamp"] for df in candles_by_symbol.values()])
    start  = pd.Timestamp(all_ts.min()).to_pydatetime()
    end    = pd.Timestamp(all_ts.max()).to_pydatetime()

    from dateutil.relativedelta import relativedelta
    train_start = start
    idx = 0

    while True:
        train_end = train_start + relativedelta(months=wf.train_months)
        test_end  = train_end   + relativedelta(months=wf.test_months)
        if test_end > end:
            break

        train_candles = _slice_candles(candles_by_symbol, train_start, train_end)
        test_candles  = _slice_candles(candles_by_symbol, train_end, test_end)

        if not _has_enough_data(train_candles) or not _has_enough_data(test_candles):
            train_start += relativedelta(months=wf.step_months)
            continue

        # Optimise on train window (reduced trials for speed)
        best_params = optimize(
            strategy_cls, train_candles, config, aux_data,
            n_trials=n_trials_per_window,
            objective_fn=objective_fn,
        )

        # Evaluate on test window with best params
        strategy = strategy_cls(best_params)
        oos_result = runner.run(strategy, test_candles, aux_data)

        windows.append(WalkForwardWindow(
            window_idx=idx,
            train_start=train_start,
            train_end=train_end,
            test_start=train_end,
            test_end=test_end,
            best_params=best_params,
            oos_result=oos_result,
        ))

        idx += 1
        train_start += relativedelta(months=wf.step_months)

    return _aggregate(strategy_cls.__name__, windows)


def _slice_candles(
    candles_by_symbol: dict[str, pd.DataFrame],
    start: datetime,
    end: datetime,
) -> dict[str, pd.DataFrame]:
    result = {}
    for sym, df in candles_by_symbol.items():
        mask = (df["timestamp"] >= start) & (df["timestamp"] < end)
        sliced = df[mask].reset_index(drop=True)
        if len(sliced) > 0:
            result[sym] = sliced
    return result


def _has_enough_data(candles_by_symbol: dict[str, pd.DataFrame], min_rows: int = 50) -> bool:
    return all(len(df) >= min_rows for df in candles_by_symbol.values()) and len(candles_by_symbol) > 0


def _aggregate(name: str, windows: list[WalkForwardWindow]) -> WalkForwardResult:
    if not windows:
        return WalkForwardResult(strategy_name=name, windows=windows)

    oos_results = [w.oos_result for w in windows]
    sharpes     = [r.sharpe_ratio for r in oos_results]
    drawdowns   = [r.max_drawdown_pct for r in oos_results]
    pf          = [r.profit_factor for r in oos_results if r.profit_factor > 0]
    profitable  = [r.total_return_pct > 0 for r in oos_results]

    import numpy as np
    return WalkForwardResult(
        strategy_name=name,
        windows=windows,
        oos_sharpe=float(np.mean(sharpes)) if sharpes else 0.0,
        oos_max_drawdown=float(np.mean(drawdowns)) if drawdowns else 0.0,
        oos_profit_factor=float(np.mean(pf)) if pf else 0.0,
        consistency_score=float(np.mean(profitable)) if profitable else 0.0,
        is_oos_divergence=0.0,  # set by orchestrator after IS run
    )
