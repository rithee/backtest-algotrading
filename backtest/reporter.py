"""
Performance reporter — terminal output (rich) + CSV export for single strategy
and portfolio.  Falls back to plain text if rich is somehow unavailable.
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
    """Print and return formatted per-strategy report using rich."""
    from backtest.analyze import AnalysisDisplay
    from rich.console import Console
    import io

    # Capture for return value
    buf = io.StringIO()
    buf_con = Console(file=buf, force_terminal=False, width=120)
    AnalysisDisplay.detail_panel(result, wf=wf, sensitivity=sensitivity, console=buf_con)

    # Print to terminal
    term_con = Console(width=120)
    AnalysisDisplay.detail_panel(result, wf=wf, sensitivity=sensitivity, console=term_con)

    # Promotion check (kept for backward compat — appended after panel)
    if promotion_criteria and wf:
        criteria = promotion_criteria
        checks = {
            "OOS Sharpe >= min":    wf.oos_sharpe >= criteria.get("min_sharpe_oos", 1.0),
            "Max drawdown <= max":  wf.oos_max_drawdown <= criteria.get("max_drawdown_pct", 0.25),
            "Profit factor >= min": wf.oos_profit_factor >= criteria.get("min_profit_factor", 1.3),
            "Trades/year >= min":   result.trades_per_year >= criteria.get("min_trades_per_year", 30),
        }
        if sensitivity and not sensitivity.is_robust:
            checks["Sensitivity robust"] = False
        passed = all(checks.values())
        verdict = "PROMOTE TO PAPER" if passed else "REJECT"
        verdict_color = "green" if passed else "red"
        term_con.print(f"[bold]→ PROMOTION: [{verdict_color}]{verdict}[/{verdict_color}][/bold]")
        for check, ok in checks.items():
            color = "green" if ok else "red"
            term_con.print(f"   [{color}]{'OK' if ok else 'FAIL'}[/{color}]  {check}")

    return buf.getvalue()


def print_portfolio_report(results: list[BacktestResult]) -> None:
    """Print side-by-side comparison of all strategies + correlation matrix."""
    if not results:
        return

    from backtest.analyze import AnalysisDisplay
    from rich.console import Console
    con = Console(width=140)

    AnalysisDisplay.summary_table(
        results,
        title="Portfolio Summary (ranked by Sharpe)",
        console=con,
    )

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
        con.print("\n[bold]Return Correlation Matrix:[/bold]")
        con.print(corr_df.round(2).to_string())
        for i, s1 in enumerate(corr_df.columns):
            for j, s2 in enumerate(corr_df.columns):
                if i < j and corr_df.loc[s1, s2] > 0.8:
                    con.print(
                        f"  [yellow]WARNING:[/yellow] {s1} and {s2} are highly "
                        f"correlated ({corr_df.loc[s1, s2]:.2f}) — consider dropping one"
                    )


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
