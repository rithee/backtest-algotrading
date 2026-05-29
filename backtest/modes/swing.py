"""
Swing futures backtest entrypoint (new strategies).

Runs all registered new swing strategies against BTC, ETH
on multi-timeframe candles (4h + 1d), with:
- 5× max / 2× default leverage
- Composite score optimisation: Sharpe × (1 - DD) × PF
- Walk-forward + sensitivity analysis for promotion gate

Usage:
    from backtest.modes.swing import run_swing
    results = run_swing()
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
from backtest.reporter import print_strategy_report, save_results_csv


# ── Strategy registry ─────────────────────────────────────────────────────────
# Maps class name → module path. Add new swing strategies here.
STRATEGY_MODULE_MAP: dict[str, str] = {
    "WyckoffPhaseDetectorStrategy":           "crypto_bot.core.signals.strategies.swing_new.wyckoff",
    "OnChainSmartMoneyDivergenceStrategy":     "crypto_bot.core.signals.strategies.swing_new.ocsmd",
    "OptionsGammaMaxPainStrategy":             "crypto_bot.core.signals.strategies.swing_new.ogp",
    "CrossAssetMomentumRegimeStrategy":        "crypto_bot.core.signals.strategies.swing_new.cam",
    "ElliottWaveAutomatorStrategy":            "crypto_bot.core.signals.strategies.swing_new.ewa",
    "FundingRateSqueezePredictorStrategy":     "crypto_bot.core.signals.strategies.swing_new.frsp",
}

CONFIG_PATH = "config/swing_new.yaml"


@dataclass
class SwingStrategyResult:
    strategy_name: str
    result: BacktestResult
    best_params: dict
    wf_result: Optional[WalkForwardResult]
    sensitivity: Optional[SensitivityResult]
    promoted: bool
    rejection_reason: str = ""


def _run_single(
    strategy_cls_name: str,
    candles_by_symbol: dict[str, pd.DataFrame],
    candles_by_tf: dict[str, dict[str, pd.DataFrame]],
    config_dict: dict,
    aux_data: dict | None,
    n_trials: int | None = None,
    skip_wf: bool = False,
    skip_sensitivity: bool = False,
) -> SwingStrategyResult:
    """Worker — runs inside a subprocess; must be picklable."""
    from crypto_bot.core.config import Config
    from backtest.runner import BacktestRunner
    from backtest.optimizer import optimize
    from backtest.walk_forward import run_walk_forward
    from backtest.sensitivity import analyze
    import importlib

    cfg = Config.model_validate(config_dict)
    module = importlib.import_module(STRATEGY_MODULE_MAP[strategy_cls_name])
    strategy_cls = getattr(module, strategy_cls_name)

    runner = BacktestRunner(cfg, mode="swing")
    dummy = strategy_cls.__new__(strategy_cls)
    dummy.params = {}
    default_params = dummy.default_params()

    initial = runner.run(strategy_cls(default_params), candles_by_symbol, aux_data, candles_by_tf)

    best_params = default_params
    if initial.composite_score <= 0 or initial.trades_per_year < cfg.optimization.min_trades_per_year:
        print(f"[swing/{strategy_cls_name}] composite={initial.composite_score:.4f} — optimising")
        best_params = optimize(strategy_cls, candles_by_symbol, cfg, aux_data, n_trials=n_trials)

    final = runner.run(strategy_cls(best_params), candles_by_symbol, aux_data, candles_by_tf)
    wf = None
    if not skip_wf:
        wf = run_walk_forward(strategy_cls, candles_by_symbol, cfg, aux_data, n_trials_per_window=30)
    sens = None
    if not skip_sensitivity:
        sens = analyze(strategy_cls, best_params, candles_by_symbol, cfg, aux_data)

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
    return SwingStrategyResult(strategy_cls_name, final, best_params, wf, sens, promoted, reason)


def run_swing(
    config_path: str = CONFIG_PATH,
    max_workers: int = 4,
    save_csv: bool = True,
    n_trials: int | None = None,
    skip_wf: bool = False,
    skip_sensitivity: bool = False,
) -> list[SwingStrategyResult]:
    """
    Run all registered swing (new) strategies.

    Args:
        config_path: Path to swing YAML config (default: config/swing_new.yaml)
        max_workers: Parallel subprocess workers
        save_csv:    Export results to results/swing_results.csv

    Returns:
        List of SwingStrategyResult, one per strategy.
    """
    from crypto_bot.core.data.loader import load_mtf_candles, load_aux_data

    cfg = Config.from_yaml(config_path)
    config_dict = cfg.model_dump()

    if not STRATEGY_MODULE_MAP:
        print("[swing] No strategies registered yet — add entries to STRATEGY_MODULE_MAP.")
        return []

    print(f"[swing] Loading candles for {cfg.backtest.symbols} …")
    candles_by_tf = load_mtf_candles(cfg)

    primary_tf = cfg.backtest.active_primary_tf
    candles_by_symbol: dict[str, pd.DataFrame] = {
        sym: candles_by_tf[sym][primary_tf]
        for sym in candles_by_tf
        if primary_tf in candles_by_tf[sym]
    }

    print("[swing] Loading auxiliary data …")
    aux_data = load_aux_data(cfg, mode="swing_new")

    results: list[SwingStrategyResult] = []

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
                print(f"[swing/{name}] {status}  composite={r.result.composite_score:.4f}  dd={r.result.max_drawdown_pct:.1%}")
                print_strategy_report(r.result, r.wf_result, r.sensitivity)
            except Exception as exc:
                print(f"[swing/{name}] ERROR: {exc}")

    if save_csv and results:
        from backtest.analyze import AnalysisDisplay
        AnalysisDisplay.export_summary_csv(
            [r.result for r in results], "results/swing_summary.csv"
        )

    promoted = [r for r in results if r.promoted]
    print(f"\n[swing] {len(promoted)}/{len(results)} strategies promoted.")
    return results
