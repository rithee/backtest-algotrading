"""
Performance reporter — terminal output + CSV export for single strategy and portfolio.
"""
from __future__ import annotations
from datetime import datetime
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

from backtest.runner import BacktestResult
from backtest.walk_forward import WalkForwardResult
from backtest.sensitivity import SensitivityResult


def print_strategy_report(
    result: BacktestResult,
    wf: Optional[WalkForwardResult] = None,
    sensitivity: Optional[SensitivityResult] = None,
    promotion_criteria: Optional[dict] = None,
) -> str:
    """Print and return formatted per-strategy report."""
    try:
        from tabulate import tabulate
    except ImportError:
        tabulate = None

    rows = [
        ["Total return %",       f"{result.total_return_pct * 100:.2f}%"],
        ["Sharpe ratio (IS)",    f"{result.sharpe_ratio:.3f}"],
        ["Sortino ratio",        f"{result.sortino_ratio:.3f}"],
        ["Max drawdown %",       f"{result.max_drawdown_pct * 100:.2f}%"],
        ["Max DD duration (d)",  f"{result.max_drawdown_duration_days:.1f}"],
        ["Win rate",             f"{result.win_rate * 100:.1f}%"],
        ["Profit factor",        f"{result.profit_factor:.3f}"],
        ["Trades/year",          f"{result.trades_per_year:.1f}"],
        ["Total fees",           f"${result.total_fees:.2f}"],
        ["Total slippage",       f"${result.total_slippage:.2f}"],
        ["Composite score",      f"{result.composite_score:.4f}"],
        ["Total trades",         str(len(result.fills))],
    ]

    if wf:
        rows.extend([
            ["OOS Sharpe",        f"{wf.oos_sharpe:.3f}"],
            ["OOS Max DD %",      f"{wf.oos_max_drawdown * 100:.2f}%"],
            ["OOS Profit factor", f"{wf.oos_profit_factor:.3f}"],
            ["Consistency score", f"{wf.consistency_score * 100:.1f}%"],
        ])

    if sensitivity:
        rows.append(["Sensitivity", "ROBUST" if sensitivity.is_robust else "BRITTLE"])

    header = f"\n{'='*50}\n  {result.strategy_name}\n{'='*50}"
    if tabulate:
        body = tabulate(rows, headers=["Metric", "Value"], tablefmt="simple")
    else:
        body = "\n".join(f"  {r[0]:<30} {r[1]}" for r in rows)

    # Promotion check
    promotion = ""
    if promotion_criteria and wf:
        criteria = promotion_criteria
        checks = {
            "OOS Sharpe >= min": wf.oos_sharpe >= criteria.get("min_sharpe_oos", 1.0),
            "Max drawdown <= max": wf.oos_max_drawdown <= criteria.get("max_drawdown_pct", 0.25),
            "Profit factor >= min": wf.oos_profit_factor >= criteria.get("min_profit_factor", 1.3),
            "Trades/year >= min": result.trades_per_year >= criteria.get("min_trades_per_year", 30),
        }
        passed = all(checks.values())
        if sensitivity and not sensitivity.is_robust:
            passed = False
            checks["Sensitivity robust"] = False
        promotion = f"\n-> PROMOTION: {'PROMOTE TO PAPER' if passed else 'REJECT'}"
        for check, ok in checks.items():
            promotion += f"\n   {'OK' if ok else 'FAIL'} {check}"

    full = header + "\n" + body + promotion
    print(full)
    return full


def print_portfolio_report(results: list[BacktestResult]) -> None:
    """Print side-by-side comparison of all strategies + correlation matrix."""
    if not results:
        return

    try:
        from tabulate import tabulate
        _tab = tabulate
    except ImportError:
        _tab = None

    rows = []
    for r in sorted(results, key=lambda x: x.sharpe_ratio, reverse=True):
        rows.append([
            r.strategy_name,
            f"{r.total_return_pct * 100:.1f}%",
            f"{r.sharpe_ratio:.3f}",
            f"{r.max_drawdown_pct * 100:.1f}%",
            f"{r.win_rate * 100:.0f}%",
            f"{r.profit_factor:.2f}",
            f"{r.trades_per_year:.0f}",
        ])

    headers = ["Strategy", "Return", "Sharpe", "MaxDD", "WinRate", "PF", "Trades/yr"]
    print("\n" + "=" * 80)
    print("  PORTFOLIO SUMMARY (ranked by Sharpe)")
    print("=" * 80)
    if _tab:
        print(_tab(rows, headers=headers, tablefmt="simple"))
    else:
        print(" | ".join(f"{h:>12}" for h in headers))
        for row in rows:
            print(" | ".join(f"{v:>12}" for v in row))

    # Correlation matrix from equity curves
    equity_dfs = {}
    for r in results:
        if r.equity_curve:
            eq = pd.Series(
                {ts: val for ts, val in r.equity_curve}, name=r.strategy_name
            )
            equity_dfs[r.strategy_name] = eq.pct_change()

    if len(equity_dfs) >= 2:
        corr_df = pd.DataFrame(equity_dfs).corr()
        print("\n  Return Correlation Matrix:")
        print(corr_df.round(2).to_string())
        # Warn on high correlations
        for i, s1 in enumerate(corr_df.columns):
            for j, s2 in enumerate(corr_df.columns):
                if i < j and corr_df.loc[s1, s2] > 0.8:
                    print(f"  WARNING: {s1} and {s2} are highly correlated ({corr_df.loc[s1, s2]:.2f}) — consider dropping one")


def save_results_csv(result: BacktestResult, output_dir: str = "results") -> None:
    """Save fills and equity curve to CSV."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    if result.fills:
        fills_df = pd.DataFrame([f.model_dump() for f in result.fills])
        fills_df.to_csv(f"{output_dir}/{result.strategy_name}_fills_{ts}.csv", index=False)

    if result.equity_curve:
        eq_df = pd.DataFrame(result.equity_curve, columns=["timestamp", "equity"])
        eq_df.to_csv(f"{output_dir}/{result.strategy_name}_equity_{ts}.csv", index=False)


def plot_equity_curve(result: BacktestResult, output_dir: str = "results") -> None:
    """Plot and save equity curve. Silent if matplotlib unavailable."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if not result.equity_curve:
        return

    timestamps = [ts for ts, _ in result.equity_curve]
    equity     = [eq for _, eq in result.equity_curve]

    plt.figure(figsize=(12, 5))
    plt.plot(timestamps, equity, linewidth=1.5)
    plt.title(f"{result.strategy_name} — Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value (USDT)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"{output_dir}/{result.strategy_name}_equity_{ts}.png", dpi=100)
    plt.close()
