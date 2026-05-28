"""
Intraday futures backtest entrypoint.

Runs all registered intraday strategies against BTC, ETH, SOL
on multi-timeframe candles (1m/5m/15m/1h), with:
- Session filter (4 UTC windows)
- Daily kill switch at -3%
- Max 5 trades/symbol/day
- 5× max leverage, 3× default
- Auto-optimization via fee-adjusted Sharpe

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
from backtest.walk_forward import run_walk_forward
from backtest.reporter import print_strategy_report, save_results_csv


# ── Strategy registry ─────────────────────────────────────────────────────────
# Maps class name → module path. Add new intraday strategies here.
STRATEGY_MODULE_MAP: dict[str, str] = {
    # Populated in Plan 2 — intraday strategy implementations
    # "ScalpVWAPStrategy":         "crypto_bot.core.signals.strategies.intraday.scalp_vwap",
    # "OpeningRangeBreakoutStrategy": "crypto_bot.core.signals.strategies.intraday.orb",
    # "MomentumSpikeStrategy":     "crypto_bot.core.signals.strategies.intraday.momentum_spike",
    # "LiquidationCascadeStrategy":"crypto_bot.core.signals.strategies.intraday.liquidation_cascade",
    # "MicroStructureReversionStrategy": "crypto_bot.core.signals.strategies.intraday.microstructure_reversion",
    # "SessionBreakoutStrategy":   "crypto_bot.core.signals.strategies.intraday.session_breakout",
}

CONFIG_PATH = "config/intraday.yaml"


@dataclass
class IntradayStrategyResult:
    strategy_name: str
    result: BacktestResult
    best_params: dict
    promoted: bool
    rejection_reason: str = ""


def _fee_adjusted_sharpe(result: BacktestResult) -> float:
    """
    Intraday optimization objective: Sharpe penalised by fee drag.
    Rewards strategies that remain profitable after the higher per-trade cost.
    """
    if result.trades_per_year < 50:
        return -999.0
    fee_penalty = result.total_fees / max(result.equity_curve[-1] - result.equity_curve[0], 1e-8)
    return result.sharpe_ratio * (1.0 - min(fee_penalty, 0.5))


def _run_single(
    strategy_cls_name: str,
    candles_by_symbol: dict[str, pd.DataFrame],
    candles_by_tf: dict[str, dict[str, pd.DataFrame]],
    config_dict: dict,
    aux_data: dict | None,
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
    if initial.sharpe_ratio < 0.5 or initial.trades_per_year < 50:
        print(f"[intraday/{strategy_cls_name}] sharpe={initial.sharpe_ratio:.3f} — optimising")
        best_params = optimize(strategy_cls, candles_by_symbol, cfg, aux_data)

    final = runner.run(strategy_cls(best_params), candles_by_symbol, candles_by_tf, aux_data)

    promoted = (
        final.sharpe_ratio >= 1.0
        and final.max_drawdown_pct <= 0.25
        and final.trades_per_year >= 50
    )
    reason = "" if promoted else (
        f"sharpe={final.sharpe_ratio:.2f}, dd={final.max_drawdown_pct:.1%}, "
        f"trades/yr={final.trades_per_year:.0f}"
    )
    return IntradayStrategyResult(strategy_cls_name, final, best_params, promoted, reason)


def run_intraday(
    config_path: str = CONFIG_PATH,
    max_workers: int = 4,
    save_csv: bool = True,
) -> list[IntradayStrategyResult]:
    """
    Run all registered intraday strategies.

    Args:
        config_path: Path to intraday YAML config (default: config/intraday.yaml)
        max_workers: Parallel subprocess workers
        save_csv:    Export results to results/intraday_results.csv

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

    # Primary timeframe candles as flat dict for optimizer compatibility
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
            ): name
            for name in STRATEGY_MODULE_MAP
        }
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                r = fut.result()
                results.append(r)
                status = "PROMOTE" if r.promoted else "REJECT"
                print(f"[intraday/{name}] {status}  sharpe={r.result.sharpe_ratio:.3f}  dd={r.result.max_drawdown_pct:.1%}")
                print_strategy_report(r.result)
            except Exception as exc:
                print(f"[intraday/{name}] ERROR: {exc}")

    if save_csv and results:
        save_results_csv([r.result for r in results], "results/intraday_results.csv")

    promoted = [r for r in results if r.promoted]
    print(f"\n[intraday] {len(promoted)}/{len(results)} strategies promoted.")
    return results
