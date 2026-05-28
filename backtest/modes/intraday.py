"""
Intraday futures backtest entrypoint.

Runs all registered intraday strategies against BTC, ETH, SOL
on multi-timeframe candles (1m/5m/15m/1h), with:
- Session filter (4 UTC windows)
- Daily kill switch at -3%
- Max 5 trades/symbol/day
- 5× max leverage, 3× default
- Auto-optimization via fee-adjusted Sharpe
- Walk-forward + sensitivity analysis for OOS promotion gate

Usage:
    from backtest.modes.intraday import run_intraday
    results = run_intraday()
"""
from __future__ import annotations

import importlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

from crypto_bot.core.config import Config
from backtest.runner import BacktestRunner, BacktestResult
from backtest.optimizer import optimize
from backtest.walk_forward import run_walk_forward, WalkForwardResult
from backtest.sensitivity import analyze, SensitivityResult
from backtest.reporter import print_strategy_report


# ── Strategy registry ─────────────────────────────────────────────────────────
STRATEGY_MODULE_MAP: dict[str, str] = {
    "LiquiditySweepReversalStrategy":              "crypto_bot.core.signals.strategies.intraday.lsr",
    "OpeningRangeBreakoutStrategy":                "crypto_bot.core.signals.strategies.intraday.orb_sb",
    "VWAPDeltaConfluenceStrategy":                 "crypto_bot.core.signals.strategies.intraday.mvdc",
    "FairValueGapStrategy":                        "crypto_bot.core.signals.strategies.intraday.fvg",
    "LiquidationCascadeMomentumStrategy":          "crypto_bot.core.signals.strategies.intraday.lcm",
    "MicrostructureConsolidationBreakoutStrategy": "crypto_bot.core.signals.strategies.intraday.mcb",
}

CONFIG_PATH = "config/intraday.yaml"


@dataclass
class IntradayStrategyResult:
    strategy_name: str
    result: BacktestResult
    best_params: dict
    wf_result: Optional[WalkForwardResult]
    sensitivity: Optional[SensitivityResult]
    promoted: bool
    rejection_reason: str = ""


def _fee_adjusted_sharpe(result: BacktestResult) -> float:
    """
    Intraday optimization objective: Sharpe penalised by fee drag.
    Returns -999 when trades < 50 or equity curve is empty.

    equity_curve is list[tuple[datetime, float]] — index [1] gives the equity value.
    """
    if result.trades_per_year < 50:
        return -999.0
    if not result.equity_curve:
        return -999.0
    start_equity = result.equity_curve[0][1]
    end_equity   = result.equity_curve[-1][1]
    gain = max(end_equity - start_equity, 1e-8)
    fee_penalty = result.total_fees / gain
    return result.sharpe_ratio * (1.0 - min(fee_penalty, 0.5))


def _run_single(
    strategy_cls_name: str,
    candles_by_symbol: dict[str, pd.DataFrame],
    candles_by_tf: dict[str, dict[str, pd.DataFrame]],
    config_dict: dict,
    aux_data: dict | None,
    n_trials: int | None = None,
    skip_wf: bool = False,
    skip_sensitivity: bool = False,
) -> IntradayStrategyResult:
    """Worker — runs inside a subprocess; must be picklable."""
    from crypto_bot.core.config import Config
    from backtest.runner import BacktestRunner
    from backtest.optimizer import optimize
    import importlib

    cfg = Config.model_validate(config_dict)
    module = importlib.import_module(STRATEGY_MODULE_MAP[strategy_cls_name])
    strategy_cls = getattr(module, strategy_cls_name)

    runner = BacktestRunner(cfg, mode="intraday")
    dummy = strategy_cls.__new__(strategy_cls)
    dummy.params = {}
    default_params = dummy.default_params()

    initial = runner.run(strategy_cls(default_params), candles_by_symbol, candles_by_tf, aux_data)

    best_params = default_params
    score = _fee_adjusted_sharpe(initial)
    if score < 0.5 or initial.trades_per_year < cfg.optimization.min_trades_per_year:
        print(f"[intraday/{strategy_cls_name}] fee_sharpe={score:.3f} — optimising")
        best_params = optimize(
            strategy_cls, candles_by_symbol, cfg, aux_data,
            n_trials=n_trials,
            objective_fn=_fee_adjusted_sharpe,
        )

    final = runner.run(strategy_cls(best_params), candles_by_symbol, candles_by_tf, aux_data)

    # Walk-forward
    wf: Optional[WalkForwardResult] = None
    if not skip_wf:
        from backtest.walk_forward import run_walk_forward
        wf = run_walk_forward(strategy_cls, candles_by_symbol, cfg, aux_data, n_trials_per_window=20)

    # Sensitivity
    sens: Optional[SensitivityResult] = None
    if not skip_sensitivity:
        from backtest.sensitivity import analyze
        sens = analyze(strategy_cls, best_params, candles_by_symbol, cfg, aux_data)

    # OOS promotion gate (falls back to IS-only when skip_wf=True)
    criteria = cfg.promotion_criteria
    if wf is not None:
        promoted = (
            wf.oos_sharpe >= criteria.min_sharpe_oos
            and wf.oos_max_drawdown <= criteria.max_drawdown_pct
            and wf.oos_profit_factor >= criteria.min_profit_factor
            and final.trades_per_year >= criteria.min_trades_per_year
            and (sens is None or sens.is_robust)
        )
        reason = "" if promoted else (
            f"oos_sharpe={wf.oos_sharpe:.2f}, dd={wf.oos_max_drawdown:.1%}, "
            f"pf={wf.oos_profit_factor:.2f}"
        )
    else:
        promoted = (
            final.sharpe_ratio >= criteria.min_sharpe_oos
            and final.max_drawdown_pct <= criteria.max_drawdown_pct
            and final.trades_per_year >= criteria.min_trades_per_year
            and (sens is None or sens.is_robust)
        )
        reason = "" if promoted else (
            f"sharpe={final.sharpe_ratio:.2f}, dd={final.max_drawdown_pct:.1%}, "
            f"trades/yr={final.trades_per_year:.0f}"
        )

    return IntradayStrategyResult(
        strategy_cls_name, final, best_params, wf, sens, promoted, reason
    )


def run_intraday(
    config_path: str = CONFIG_PATH,
    max_workers: int = 4,
    save_csv: bool = True,
    n_trials: int | None = None,
    skip_wf: bool = False,
    skip_sensitivity: bool = False,
) -> list[IntradayStrategyResult]:
    """
    Run all registered intraday strategies.

    Args:
        config_path:       Path to intraday YAML config
        max_workers:       Parallel subprocess workers
        save_csv:          Export summary CSV to results/intraday_summary.csv
        n_trials:          Override Optuna trial count (None = use config value)
        skip_wf:           Skip walk-forward validation (faster, no OOS gate)
        skip_sensitivity:  Skip sensitivity analysis

    Returns:
        List of IntradayStrategyResult, one per strategy.
    """
    from crypto_bot.core.data.loader import load_mtf_candles, load_aux_data

    cfg = Config.from_yaml(config_path)
    config_dict = cfg.model_dump()

    if not STRATEGY_MODULE_MAP:
        print("[intraday] No strategies registered yet — add entries to STRATEGY_MODULE_MAP.")
        return []

    print(f"[intraday] Loading candles for {cfg.backtest.symbols} …")
    candles_by_tf = load_mtf_candles(cfg)

    primary_tf = cfg.backtest.active_primary_tf
    candles_by_symbol: dict[str, pd.DataFrame] = {
        sym: candles_by_tf[sym][primary_tf]
        for sym in candles_by_tf
        if primary_tf in candles_by_tf[sym]
    }

    print("[intraday] Loading auxiliary data …")
    aux_data = load_aux_data(cfg, mode="intraday")

    results: list[IntradayStrategyResult] = []

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _run_single,
                name, candles_by_symbol, candles_by_tf, config_dict, aux_data,
                n_trials, skip_wf, skip_sensitivity,
            ): name
            for name in STRATEGY_MODULE_MAP
        }
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                r = fut.result()
                results.append(r)
                status = "PROMOTE" if r.promoted else "REJECT"
                fee_sh = _fee_adjusted_sharpe(r.result)
                print(
                    f"[intraday/{name}] {status}  "
                    f"fee_sharpe={fee_sh:.3f}  dd={r.result.max_drawdown_pct:.1%}"
                )
                print_strategy_report(r.result, r.wf_result, r.sensitivity)
            except Exception as exc:
                print(f"[intraday/{name}] ERROR: {exc}")

    if save_csv and results:
        from backtest.analyze import AnalysisDisplay
        AnalysisDisplay.export_summary_csv(
            [r.result for r in results], "results/intraday_summary.csv"
        )

    promoted = [r for r in results if r.promoted]
    print(f"\n[intraday] {len(promoted)}/{len(results)} strategies promoted.")
    return results
