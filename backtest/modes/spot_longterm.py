"""
Long-term spot backtest entrypoint.

Runs all registered spot strategies against top-10 by market cap
on multi-timeframe candles (1d + 1w), with:
- No leverage (enforced by runner)
- Monthly rebalance, min 30-day holding
- BTC dominance filter
- Calmar Ratio optimisation objective
- $1,000 capital

Usage:
    from backtest.modes.spot_longterm import run_spot
    results = run_spot()
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
# Maps class name → module path. Add new spot strategies here.
STRATEGY_MODULE_MAP: dict[str, str] = {
    # Populated in Plan 4 — spot strategy implementations
    # "BTCDominanceCycleStrategy": "crypto_bot.core.signals.strategies.spot.btc_dominance_cycle",
    # "OnChainNUPLStrategy":        "crypto_bot.core.signals.strategies.spot.onchain_nupl",
    # "MVRVMeanReversionStrategy":  "crypto_bot.core.signals.strategies.spot.mvrv_mean_reversion",
    # "RainbowAccumulationStrategy":"crypto_bot.core.signals.strategies.spot.rainbow_accumulation",
    # "NVTSignalStrategy":          "crypto_bot.core.signals.strategies.spot.nvt_signal",
    # "StockToFlowDevStrategy":     "crypto_bot.core.signals.strategies.spot.stock_to_flow_dev",
    # "MacroCycleStrategy":         "crypto_bot.core.signals.strategies.spot.macro_cycle",
}

CONFIG_PATH = "config/spot_longterm.yaml"


@dataclass
class SpotStrategyResult:
    strategy_name: str
    result: BacktestResult
    best_params: dict
    wf_result: Optional[WalkForwardResult]
    sensitivity: Optional[SensitivityResult]
    promoted: bool
    rejection_reason: str = ""


def _calmar_ratio(result: BacktestResult) -> float:
    """
    Spot optimization objective: Calmar Ratio (annualized return / max drawdown).
    Penalises strategies with deep drawdowns — most important for long-term holders.
    """
    if result.max_drawdown_pct <= 0 or result.trades_per_year < 1:
        return -999.0
    return result.total_return_pct / result.max_drawdown_pct


def _run_single(
    strategy_cls_name: str,
    candles_by_symbol: dict[str, pd.DataFrame],
    candles_by_tf: dict[str, dict[str, pd.DataFrame]],
    config_dict: dict,
    aux_data: dict | None,
) -> SpotStrategyResult:
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

    runner = BacktestRunner(cfg, mode="spot_longterm")
    dummy = strategy_cls.__new__(strategy_cls)
    dummy.params = {}
    default_params = dummy.default_params()

    initial = runner.run(strategy_cls(default_params), candles_by_symbol, candles_by_tf, aux_data)

    best_params = default_params
    calmar = _calmar_ratio(initial)
    if calmar < 0.5 or initial.trades_per_year < 2:
        print(f"[spot/{strategy_cls_name}] calmar={calmar:.2f} — optimising")
        best_params = optimize(strategy_cls, candles_by_symbol, cfg, aux_data)

    final = runner.run(strategy_cls(best_params), candles_by_symbol, candles_by_tf, aux_data)
    wf = run_walk_forward(strategy_cls, candles_by_symbol, cfg, aux_data, n_trials_per_window=20)
    sens = analyze(strategy_cls, best_params, candles_by_symbol, cfg, aux_data)

    final_calmar = _calmar_ratio(final)
    promoted = (
        final_calmar >= 0.5
        and final.max_drawdown_pct <= 0.50      # spot can have larger drawdowns
        and final.total_return_pct > 0
        and final.trades_per_year >= 1
        and (sens is None or sens.is_robust)
    )
    reason = "" if promoted else (
        f"calmar={final_calmar:.2f}, dd={final.max_drawdown_pct:.1%}, "
        f"return={final.total_return_pct:.1%}"
    )
    return SpotStrategyResult(strategy_cls_name, final, best_params, wf, sens, promoted, reason)


def run_spot(
    config_path: str = CONFIG_PATH,
    max_workers: int = 4,
    save_csv: bool = True,
) -> list[SpotStrategyResult]:
    """
    Run all registered long-term spot strategies.

    Args:
        config_path: Path to spot YAML config (default: config/spot_longterm.yaml)
        max_workers: Parallel subprocess workers
        save_csv:    Export results to results/spot_results.csv

    Returns:
        List of SpotStrategyResult, one per strategy.
    """
    from crypto_bot.core.data.loader import load_mtf_candles, load_aux_data
    from crypto_bot.core.data.coinmarketcap import fetch_top_n

    cfg = Config.from_yaml(config_path)
    config_dict = cfg.model_dump()

    if not STRATEGY_MODULE_MAP:
        print("[spot] No strategies registered yet — add entries to STRATEGY_MODULE_MAP.")
        return []

    # Refresh top-10 symbols from CMC if available, else fall back to config
    top10 = fetch_top_n(n=10)
    if top10:
        print(f"[spot] Using CMC top-10: {top10}")
    else:
        print(f"[spot] CMC unavailable — using config symbols: {cfg.backtest.symbols}")

    print(f"[spot] Loading candles …")
    candles_by_tf = load_mtf_candles(cfg)

    primary_tf = cfg.backtest.active_primary_tf
    candles_by_symbol: dict[str, pd.DataFrame] = {
        sym: candles_by_tf[sym][primary_tf]
        for sym in candles_by_tf
        if primary_tf in candles_by_tf[sym]
    }

    print("[spot] Loading auxiliary data …")
    aux_data = load_aux_data(cfg, mode="spot_longterm")

    results: list[SpotStrategyResult] = []

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _run_single,
                name, candles_by_symbol, candles_by_tf, config_dict, aux_data,
            ): name
            for name in STRATEGY_MODULE_MAP
        }
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                r = fut.result()
                results.append(r)
                calmar = _calmar_ratio(r.result)
                status = "PROMOTE" if r.promoted else "REJECT"
                print(f"[spot/{name}] {status}  calmar={calmar:.2f}  dd={r.result.max_drawdown_pct:.1%}  return={r.result.total_return_pct:.1%}")
                print_strategy_report(r.result, r.wf_result, r.sensitivity)
            except Exception as exc:
                print(f"[spot/{name}] ERROR: {exc}")

    if save_csv and results:
        save_results_csv([r.result for r in results], "results/spot_results.csv")

    promoted = [r for r in results if r.promoted]
    print(f"\n[spot] {len(promoted)}/{len(results)} strategies promoted.")
    return results
