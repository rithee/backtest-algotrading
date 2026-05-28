"""
Backtest orchestrator.
Runs strategies 1-6 in parallel via ProcessPoolExecutor.
XGBoost meta is applied as a filter layer after strategies 1-6 complete.
Auto-triggers Optuna optimisation if initial backtest fails promotion criteria.
"""
from __future__ import annotations
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Type
import pandas as pd

from crypto_bot.core.config import Config
from crypto_bot.core.signals.base import BaseStrategy
from backtest.runner import BacktestRunner, BacktestResult
from backtest.optimizer import optimize
from backtest.walk_forward import run_walk_forward, WalkForwardResult
from backtest.sensitivity import analyze, SensitivityResult
from backtest.reporter import print_strategy_report, print_portfolio_report, save_results_csv


@dataclass
class StrategyResult:
    strategy_name: str
    initial_result: BacktestResult
    best_params: dict
    final_result: BacktestResult
    wf_result: Optional[WalkForwardResult]
    sensitivity: Optional[SensitivityResult]
    promoted: bool
    rejection_reason: str = ""


def _check_promotion(result: BacktestResult, wf: Optional[WalkForwardResult], sensitivity: Optional[SensitivityResult], criteria) -> tuple[bool, str]:
    if wf is None:
        return False, "no_walk_forward"
    checks = [
        (wf.oos_sharpe >= criteria.min_sharpe_oos, f"oos_sharpe={wf.oos_sharpe:.2f}<{criteria.min_sharpe_oos}"),
        (wf.oos_max_drawdown <= criteria.max_drawdown_pct, f"drawdown={wf.oos_max_drawdown:.1%}>{criteria.max_drawdown_pct:.1%}"),
        (wf.oos_profit_factor >= criteria.min_profit_factor, f"profit_factor={wf.oos_profit_factor:.2f}<{criteria.min_profit_factor}"),
        (result.trades_per_year >= criteria.min_trades_per_year, f"trades/yr={result.trades_per_year:.0f}<{criteria.min_trades_per_year}"),
    ]
    if sensitivity and not sensitivity.is_robust:
        checks.append((False, "brittle_params"))

    failed = [reason for ok, reason in checks if not ok]
    if failed:
        return False, "; ".join(failed)
    return True, ""


def _run_single_strategy(
    strategy_cls_name: str,
    candles_by_symbol: dict[str, pd.DataFrame],
    config_dict: dict,
    aux_data: dict | None,
) -> StrategyResult:
    """Worker function — runs in a subprocess. Must be picklable."""
    # Re-import inside subprocess
    import importlib
    from crypto_bot.core.config import Config
    from backtest.runner import BacktestRunner
    from backtest.optimizer import optimize
    from backtest.walk_forward import run_walk_forward
    from backtest.sensitivity import analyze

    # Reconstruct config
    cfg = Config.model_validate(config_dict)

    # Import strategy class
    strategy_module_map = {
        "EMARibbonStrategy":            "crypto_bot.core.signals.strategies.ema_ribbon",
        "TTMSqueezeStrategy":           "crypto_bot.core.signals.strategies.ttm_squeeze",
        "RSIDivergenceStrategy":        "crypto_bot.core.signals.strategies.rsi_divergence",
        "SupertrendADXStrategy":        "crypto_bot.core.signals.strategies.supertrend_adx",
        "BBMeanReversionStrategy":      "crypto_bot.core.signals.strategies.bb_mean_reversion",
        "FundingRateReversionStrategy": "crypto_bot.core.signals.strategies.funding_rate_reversion",
        "DonchianBreakoutStrategy":     "crypto_bot.core.signals.strategies.donchian_breakout",
        "TSMOMStrategy":                "crypto_bot.core.signals.strategies.tsmom",
        "HMAChandelierStrategy":        "crypto_bot.core.signals.strategies.hma_chandelier",
        "AdaptiveTrendStrategy":        "crypto_bot.core.signals.strategies.adaptive_trend",
        "VWAPBreakoutStrategy":         "crypto_bot.core.signals.strategies.vwap_breakout",
    }
    module = importlib.import_module(strategy_module_map[strategy_cls_name])
    strategy_cls = getattr(module, strategy_cls_name)

    runner = BacktestRunner(cfg)
    dummy = strategy_cls.__new__(strategy_cls)
    dummy.params = {}
    default_params = dummy.default_params()

    # Initial backtest
    initial_result = runner.run(strategy_cls(default_params), candles_by_symbol, aux_data)

    # Check if optimisation needed
    best_params = default_params
    if initial_result.composite_score <= 0 or initial_result.trades_per_year < cfg.optimization.min_trades_per_year:
        print(f"[{strategy_cls_name}] Initial score={initial_result.composite_score:.4f} — triggering optimisation")
        best_params = optimize(strategy_cls, candles_by_symbol, cfg, aux_data)

    # Final backtest with best params
    final_result = runner.run(strategy_cls(best_params), candles_by_symbol, aux_data)

    # Walk-forward validation
    wf_result = run_walk_forward(strategy_cls, candles_by_symbol, cfg, aux_data, n_trials_per_window=30)

    # Sensitivity analysis
    sensitivity = analyze(strategy_cls, best_params, candles_by_symbol, cfg, aux_data)

    promoted, reason = _check_promotion(final_result, wf_result, sensitivity, cfg.promotion_criteria)

    return StrategyResult(
        strategy_name=strategy_cls_name,
        initial_result=initial_result,
        best_params=best_params,
        final_result=final_result,
        wf_result=wf_result,
        sensitivity=sensitivity,
        promoted=promoted,
        rejection_reason=reason,
    )


def run_all(
    candles_by_symbol: dict[str, pd.DataFrame],
    config: Config,
    aux_data: dict | None = None,
    max_workers: int = 6,
) -> list[StrategyResult]:
    """
    Phase A: run strategies 1-6 in parallel.
    Phase B: run XGBoost meta filter on top of 1-6 results.
    """
    from crypto_bot.core.signals.strategies.ema_ribbon import EMARibbonStrategy
    from crypto_bot.core.signals.strategies.ttm_squeeze import TTMSqueezeStrategy
    from crypto_bot.core.signals.strategies.rsi_divergence import RSIDivergenceStrategy
    from crypto_bot.core.signals.strategies.supertrend_adx import SupertrendADXStrategy
    from crypto_bot.core.signals.strategies.bb_mean_reversion import BBMeanReversionStrategy
    from crypto_bot.core.signals.strategies.funding_rate_reversion import FundingRateReversionStrategy

    strategy_names = [
        "EMARibbonStrategy", "TTMSqueezeStrategy", "RSIDivergenceStrategy",
        "SupertrendADXStrategy", "BBMeanReversionStrategy", "FundingRateReversionStrategy",
        "DonchianBreakoutStrategy", "TSMOMStrategy", "HMAChandelierStrategy",
        "AdaptiveTrendStrategy", "VWAPBreakoutStrategy",
    ]

    config_dict = config.model_dump()
    results: list[StrategyResult] = []

    print(f"\nRunning {len(strategy_names)} strategies (up to {max_workers} in parallel)...\n")

    with ProcessPoolExecutor(max_workers=min(max_workers, len(strategy_names))) as pool:
        future_to_name = {
            pool.submit(_run_single_strategy, name, candles_by_symbol, config_dict, aux_data): name
            for name in strategy_names
        }
        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                sr = future.result()
                results.append(sr)
                status = "PROMOTED" if sr.promoted else "REJECTED"
                print(f"  [{name}] {status} — score={sr.final_result.composite_score:.4f}")
            except Exception as exc:
                print(f"  [{name}] ERROR: {exc}")

    # Print full reports
    for sr in results:
        print_strategy_report(
            sr.final_result,
            sr.wf_result,
            sr.sensitivity,
            config.promotion_criteria.model_dump(),
        )
        save_results_csv(sr.final_result)

    print_portfolio_report([sr.final_result for sr in results])
    return results
