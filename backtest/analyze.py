"""
Analysis display layer — rich terminal tables and export for BacktestResult lists.

Responsibilities:
  - summary_table()      : side-by-side rich table, sorted by Sharpe, color-coded
  - detail_panel()       : deep-dive rich panel for one strategy
  - export_summary_csv() : one row per strategy CSV with all metrics + calmar
  - export_html()        : full HTML report via rich Console.export_html()

This module never runs backtests — it only renders pre-computed results.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from backtest.runner import BacktestResult
from backtest.walk_forward import WalkForwardResult
from backtest.sensitivity import SensitivityResult


# ── helpers ───────────────────────────────────────────────────────────────────

def _calmar(r: BacktestResult) -> float:
    """Calmar ratio = total return / max drawdown. 0 when DD=0."""
    if r.max_drawdown_pct <= 0:
        return 0.0
    return r.total_return_pct / r.max_drawdown_pct


def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def _color_return(v: float) -> str:
    color = "green" if v >= 0.20 else ("yellow" if v >= 0 else "red")
    return f"[{color}]{_pct(v)}[/{color}]"


def _color_sharpe(v: float) -> str:
    color = "green" if v >= 1.0 else ("yellow" if v >= 0.5 else "red")
    return f"[{color}]{v:.2f}[/{color}]"


def _color_dd(v: float) -> str:
    color = "green" if v <= 0.15 else ("yellow" if v <= 0.30 else "red")
    return f"[{color}]{_pct(v)}[/{color}]"


def _color_calmar(v: float) -> str:
    color = "green" if v >= 1.5 else ("yellow" if v >= 0.5 else "red")
    return f"[{color}]{v:.2f}[/{color}]"


def _color_pf(v: float) -> str:
    color = "green" if v >= 1.5 else ("yellow" if v >= 1.1 else "red")
    return f"[{color}]{v:.2f}[/{color}]"


# ── AnalysisDisplay ───────────────────────────────────────────────────────────

class AnalysisDisplay:
    """
    Static display methods for BacktestResult collections.

    All public methods accept an optional ``console`` kwarg so tests can
    capture output via a StringIO-backed Console.
    """

    @staticmethod
    def summary_table(
        results: list[BacktestResult],
        title: str = "Strategy Summary",
        console: Optional[Console] = None,
    ) -> None:
        """Print a side-by-side comparison table sorted by Sharpe ratio."""
        con = console or Console()

        if not results:
            con.print("[yellow]No results to display.[/yellow]")
            return

        table = Table(
            title=title,
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
            expand=False,
        )
        table.add_column("Strategy",   style="bold white", no_wrap=True, min_width=28)
        table.add_column("Mode",       style="dim",        no_wrap=True)
        table.add_column("Return",     justify="right")
        table.add_column("Sharpe",     justify="right")
        table.add_column("Sortino",    justify="right")
        table.add_column("MaxDD",      justify="right")
        table.add_column("Calmar",     justify="right")
        table.add_column("WinRate",    justify="right")
        table.add_column("PF",         justify="right")
        table.add_column("Trades/yr",  justify="right")
        table.add_column("Score",      justify="right")

        sorted_results = sorted(results, key=lambda r: r.sharpe_ratio, reverse=True)
        for r in sorted_results:
            calmar = _calmar(r)
            table.add_row(
                r.strategy_name,
                r.mode,
                _color_return(r.total_return_pct),
                _color_sharpe(r.sharpe_ratio),
                f"{r.sortino_ratio:.2f}",
                _color_dd(r.max_drawdown_pct),
                _color_calmar(calmar),
                f"{r.win_rate * 100:.0f}%",
                _color_pf(r.profit_factor),
                f"{r.trades_per_year:.0f}",
                f"{r.composite_score:.3f}",
            )

        con.print(table)

    @staticmethod
    def detail_panel(
        result: BacktestResult,
        wf: Optional[WalkForwardResult] = None,
        sensitivity: Optional[SensitivityResult] = None,
        console: Optional[Console] = None,
    ) -> None:
        """Print a full detail panel for one strategy."""
        con = console or Console()
        calmar = _calmar(result)

        # ── In-sample metrics ────────────────────────────────────────────────
        is_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
        is_table.add_column("Metric", style="dim")
        is_table.add_column("Value",  justify="right")

        is_rows = [
            ("Total Return",         _color_return(result.total_return_pct)),
            ("Sharpe Ratio (IS)",     _color_sharpe(result.sharpe_ratio)),
            ("Sortino Ratio",         f"{result.sortino_ratio:.3f}"),
            ("Max Drawdown",          _color_dd(result.max_drawdown_pct)),
            ("Max DD Duration (d)",   f"{result.max_drawdown_duration_days:.1f}"),
            ("Calmar Ratio",          _color_calmar(calmar)),
            ("Win Rate",              f"{result.win_rate * 100:.1f}%"),
            ("Profit Factor",         _color_pf(result.profit_factor)),
            ("Trades / Year",         f"{result.trades_per_year:.1f}"),
            ("Total Fees",            f"${result.total_fees:.2f}"),
            ("Total Slippage",        f"${result.total_slippage:.2f}"),
            ("Composite Score",       f"{result.composite_score:.4f}"),
            ("Total Trades",          str(len(result.fills))),
        ]
        for label, value in is_rows:
            is_table.add_row(label, value)

        content = is_table

        # ── Walk-forward OOS ─────────────────────────────────────────────────
        if wf is not None:
            oos_table = Table(
                box=box.SIMPLE, show_header=False, padding=(0, 2),
                title="[bold]Walk-Forward OOS[/bold]",
            )
            oos_table.add_column("Metric", style="dim")
            oos_table.add_column("Value",  justify="right")
            oos_rows = [
                ("OOS Sharpe",        _color_sharpe(wf.oos_sharpe)),
                ("OOS Max DD",        _color_dd(wf.oos_max_drawdown)),
                ("OOS Profit Factor", _color_pf(wf.oos_profit_factor)),
                ("Consistency",       f"{wf.consistency_score * 100:.0f}%"),
                ("IS/OOS Divergence", f"{wf.is_oos_divergence:.2f}"),
            ]
            for label, value in oos_rows:
                oos_table.add_row(label, value)

            from rich.columns import Columns
            content = Columns([is_table, oos_table])

        # ── Sensitivity ──────────────────────────────────────────────────────
        if sensitivity is not None:
            verdict = (
                "[bold green]ROBUST[/bold green]" if sensitivity.is_robust
                else "[bold red]BRITTLE[/bold red]"
            )
            worst = ""
            if not sensitivity.is_robust and sensitivity.sensitivities:
                bp = max(sensitivity.sensitivities, key=lambda p: abs(p.score_change_pct))
                worst = (
                    f" — worst param: [italic]{bp.param}[/italic]"
                    f" (impact {bp.score_change_pct:.1%})"
                )
            from rich.text import Text
            sens_line = Text.from_markup(f"Sensitivity: {verdict}{worst}")
        else:
            from rich.text import Text
            sens_line = Text("")

        mode_label = {
            "intraday":     "⚡ Intraday",
            "swing":        "📈 Swing",
            "spot_longterm":"🌕 Spot Long-Term",
        }.get(result.mode, result.mode)

        from rich.console import Group
        panel_content = Group(content, sens_line) if sensitivity else content

        con.print(Panel(
            panel_content,
            title=f"[bold cyan]{result.strategy_name}[/bold cyan]  [dim]{mode_label}[/dim]",
            border_style="cyan",
            padding=(1, 2),
        ))

    @staticmethod
    def export_summary_csv(
        results: list[BacktestResult],
        path: str,
    ) -> None:
        """Write one CSV row per strategy with all metrics + calmar_ratio."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "strategy_name", "mode",
            "total_return_pct", "sharpe_ratio", "sortino_ratio",
            "max_drawdown_pct", "max_drawdown_duration_days",
            "calmar_ratio",
            "win_rate", "profit_factor",
            "trades_per_year", "avg_trade_duration_bars",
            "total_fees", "total_slippage", "composite_score",
        ]

        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow({
                    "strategy_name":              r.strategy_name,
                    "mode":                       r.mode,
                    "total_return_pct":           round(r.total_return_pct, 6),
                    "sharpe_ratio":               round(r.sharpe_ratio, 6),
                    "sortino_ratio":              round(r.sortino_ratio, 6),
                    "max_drawdown_pct":           round(r.max_drawdown_pct, 6),
                    "max_drawdown_duration_days": round(r.max_drawdown_duration_days, 2),
                    "calmar_ratio":               round(_calmar(r), 6),
                    "win_rate":                   round(r.win_rate, 6),
                    "profit_factor":              round(r.profit_factor, 6),
                    "trades_per_year":            round(r.trades_per_year, 2),
                    "avg_trade_duration_bars":    round(r.avg_trade_duration_bars, 2),
                    "total_fees":                 round(r.total_fees, 4),
                    "total_slippage":             round(r.total_slippage, 4),
                    "composite_score":            round(r.composite_score, 6),
                })

    @staticmethod
    def export_html(
        results: list[BacktestResult],
        path: str,
    ) -> None:
        """Export rich-rendered summary table and detail panels to an HTML file."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)

        con = Console(record=True, width=140)
        con.print(
            f"[bold]Backtest Analysis Report[/bold]  "
            f"[dim]{len(results)} strategies[/dim]\n"
        )
        AnalysisDisplay.summary_table(results, console=con)
        con.print()
        for r in results:
            AnalysisDisplay.detail_panel(r, console=con)
            con.print()

        html = con.export_html(inline_styles=True)
        out.write_text(html, encoding="utf-8")
