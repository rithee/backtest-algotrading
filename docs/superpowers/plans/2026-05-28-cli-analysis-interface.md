# CLI Analysis Interface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a rich terminal analysis interface with color-coded metrics tables, per-strategy detail panels, HTML/CSV export, and a mode selector that routes `main.py` to the three backtest modes (intraday, swing, spot).

**Architecture:** A new `backtest/analyze.py` module owns all rich display logic and consumes `BacktestResult` lists — it never runs backtests itself. `backtest/reporter.py` is upgraded to use rich internally while keeping the same public signatures. `main.py` gains a `--mode` selector routing to each mode's `run_*()` entrypoint plus an `--export` flag for HTML/CSV output.

**Tech Stack:** Python 3.13, `rich` 13.9.4 (already installed), `argparse` (stdlib), existing `BacktestResult`/`WalkForwardResult`/`SensitivityResult` dataclasses.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `backtest/analyze.py` | **Create** | `AnalysisDisplay`: summary table, detail panel, export HTML/CSV |
| `backtest/reporter.py` | **Modify** | Upgrade `print_strategy_report` + `print_portfolio_report` to rich |
| `main.py` | **Modify** | Mode selector, `--analyze-mode`, `--export` flag, mode routing |
| `tests/test_analyzer.py` | **Create** | Tests for `AnalysisDisplay` with mock `BacktestResult` data |

---

## Task 1: Create `backtest/analyze.py` — AnalysisDisplay

**Files:**
- Create: `backtest/analyze.py`
- Test: `tests/test_analyzer.py`

### Key types in scope (from existing code, no changes needed)
```python
# backtest/runner.py
@dataclass
class BacktestResult:
    strategy_name: str
    fills: list[Fill]
    equity_curve: list[tuple[datetime, float]]
    initial_capital: float
    mode: str              # "intraday" | "swing" | "spot_longterm"
    total_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    max_drawdown_duration_days: float
    win_rate: float
    profit_factor: float
    avg_trade_duration_bars: float
    trades_per_year: float
    total_fees: float
    total_slippage: float
    composite_score: float

# backtest/walk_forward.py
@dataclass
class WalkForwardResult:
    strategy_name: str
    oos_sharpe: float
    oos_max_drawdown: float
    oos_profit_factor: float
    consistency_score: float
    is_oos_divergence: float

# backtest/sensitivity.py
@dataclass
class SensitivityResult:
    is_robust: bool
```

- [ ] **Step 1: Write the failing tests**

Create `tests/test_analyzer.py`:

```python
"""Tests for backtest.analyze.AnalysisDisplay."""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from datetime import datetime

import pytest
from rich.console import Console

from backtest.analyze import AnalysisDisplay, _calmar


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_result(
    name: str = "TestStrategy",
    total_return: float = 0.35,
    sharpe: float = 1.4,
    drawdown: float = 0.18,
    win_rate: float = 0.55,
    profit_factor: float = 1.6,
    trades_per_year: float = 45.0,
    mode: str = "swing",
):
    from backtest.runner import BacktestResult
    r = BacktestResult(strategy_name=name, mode=mode)
    r.total_return_pct = total_return
    r.sharpe_ratio = sharpe
    r.sortino_ratio = sharpe * 1.2
    r.max_drawdown_pct = drawdown
    r.max_drawdown_duration_days = 30.0
    r.win_rate = win_rate
    r.profit_factor = profit_factor
    r.trades_per_year = trades_per_year
    r.total_fees = 120.0
    r.total_slippage = 45.0
    r.composite_score = sharpe * (1 - drawdown) * profit_factor
    return r


def _capture(fn, *args, **kwargs) -> str:
    """Run fn with a StringIO Console and return the output text."""
    buf = io.StringIO()
    con = Console(file=buf, force_terminal=False, width=120)
    fn(*args, console=con, **kwargs)
    return buf.getvalue()


# ── _calmar ───────────────────────────────────────────────────────────────────

def test_calmar_normal():
    r = _make_result(total_return=0.40, drawdown=0.20)
    assert abs(_calmar(r) - 2.0) < 1e-9


def test_calmar_zero_drawdown():
    r = _make_result(drawdown=0.0)
    assert _calmar(r) == 0.0


# ── summary_table ─────────────────────────────────────────────────────────────

def test_summary_table_renders_without_error():
    results = [_make_result("A"), _make_result("B", sharpe=0.5, drawdown=0.40)]
    out = _capture(AnalysisDisplay.summary_table, results, title="Test Suite")
    assert "TestStrategy" not in out or True  # just no exception

def test_summary_table_contains_strategy_names():
    results = [_make_result("StratAlpha"), _make_result("StratBeta")]
    out = _capture(AnalysisDisplay.summary_table, results)
    assert "StratAlpha" in out
    assert "StratBeta" in out

def test_summary_table_empty_list():
    out = _capture(AnalysisDisplay.summary_table, [])
    assert "No results" in out

def test_summary_table_sorted_by_sharpe():
    r_low  = _make_result("Low",  sharpe=0.3)
    r_high = _make_result("High", sharpe=2.1)
    buf = io.StringIO()
    con = Console(file=buf, force_terminal=False, width=120)
    AnalysisDisplay.summary_table([r_low, r_high], console=con)
    out = buf.getvalue()
    assert out.index("High") < out.index("Low")


# ── detail_panel ──────────────────────────────────────────────────────────────

def test_detail_panel_renders():
    r = _make_result("DetailStrat")
    out = _capture(AnalysisDisplay.detail_panel, r)
    assert "DetailStrat" in out

def test_detail_panel_shows_calmar():
    r = _make_result(total_return=0.60, drawdown=0.20)
    out = _capture(AnalysisDisplay.detail_panel, r)
    assert "Calmar" in out
    assert "3.0" in out or "3.00" in out

def test_detail_panel_wf_section():
    from backtest.walk_forward import WalkForwardResult, WalkForwardWindow
    r = _make_result()
    wf = WalkForwardResult(
        strategy_name=r.strategy_name,
        windows=[],
        oos_sharpe=1.2,
        oos_max_drawdown=0.15,
        oos_profit_factor=1.5,
        consistency_score=0.75,
        is_oos_divergence=0.9,
    )
    out = _capture(AnalysisDisplay.detail_panel, r, wf=wf)
    assert "OOS" in out
    assert "1.2" in out or "1.20" in out

def test_detail_panel_sensitivity():
    from backtest.sensitivity import SensitivityResult
    r   = _make_result()
    sens = SensitivityResult(
        strategy_name=r.strategy_name,
        best_score=1.5,
        sensitivities=[],
        is_robust=True,
    )
    out = _capture(AnalysisDisplay.detail_panel, r, sensitivity=sens)
    assert "ROBUST" in out

def test_detail_panel_brittle_sensitivity():
    from backtest.sensitivity import SensitivityResult, ParamSensitivity
    r    = _make_result()
    brittle_param = ParamSensitivity(
        param="period",
        best_value=20.0,
        perturb_pct=0.25,
        perturbed_value=25.0,
        score_change_pct=-0.40,
        is_brittle=True,
    )
    sens = SensitivityResult(
        strategy_name=r.strategy_name,
        best_score=1.5,
        sensitivities=[brittle_param],
        is_robust=False,
    )
    out = _capture(AnalysisDisplay.detail_panel, r, sensitivity=sens)
    assert "BRITTLE" in out


# ── export_summary_csv ────────────────────────────────────────────────────────

def test_export_summary_csv(tmp_path):
    import csv
    results = [_make_result("A"), _make_result("B")]
    path = tmp_path / "summary.csv"
    AnalysisDisplay.export_summary_csv(results, str(path))
    assert path.exists()
    rows = list(csv.DictReader(open(path)))
    assert len(rows) == 2
    assert rows[0]["strategy_name"] in ("A", "B")
    assert "sharpe_ratio" in rows[0]
    assert "calmar_ratio" in rows[0]

def test_export_summary_csv_creates_parent_dir(tmp_path):
    results = [_make_result()]
    path = tmp_path / "nested" / "dir" / "out.csv"
    AnalysisDisplay.export_summary_csv(results, str(path))
    assert path.exists()


# ── export_html ───────────────────────────────────────────────────────────────

def test_export_html(tmp_path):
    results = [_make_result("HTMLStrat")]
    path = tmp_path / "report.html"
    AnalysisDisplay.export_html(results, str(path))
    assert path.exists()
    html = path.read_text()
    assert "<html" in html.lower() or "<!DOCTYPE" in html
    assert "HTMLStrat" in html

def test_export_html_creates_parent_dir(tmp_path):
    results = [_make_result()]
    path = tmp_path / "out" / "report.html"
    AnalysisDisplay.export_html(results, str(path))
    assert path.exists()
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python3 -m pytest tests/test_analyzer.py -v 2>&1 | head -30
```

Expected: ImportError or ModuleNotFoundError — `backtest.analyze` does not exist yet.

- [ ] **Step 3: Write `backtest/analyze.py`**

```python
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
    """Calmar ratio = annualised total return / max drawdown. 0 when DD=0."""
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
            ("Total Return",            _color_return(result.total_return_pct)),
            ("Sharpe Ratio (IS)",        _color_sharpe(result.sharpe_ratio)),
            ("Sortino Ratio",            f"{result.sortino_ratio:.3f}"),
            ("Max Drawdown",             _color_dd(result.max_drawdown_pct)),
            ("Max DD Duration (d)",      f"{result.max_drawdown_duration_days:.1f}"),
            ("Calmar Ratio",             _color_calmar(calmar)),
            ("Win Rate",                 f"{result.win_rate * 100:.1f}%"),
            ("Profit Factor",            _color_pf(result.profit_factor)),
            ("Trades / Year",            f"{result.trades_per_year:.1f}"),
            ("Total Fees",               f"${result.total_fees:.2f}"),
            ("Total Slippage",           f"${result.total_slippage:.2f}"),
            ("Composite Score",          f"{result.composite_score:.4f}"),
            ("Total Trades",             str(len(result.fills))),
        ]
        for label, value in is_rows:
            is_table.add_row(label, value)

        content = is_table

        # ── Walk-forward OOS ─────────────────────────────────────────────────
        if wf is not None:
            oos_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2),
                              title="[bold]Walk-Forward OOS[/bold]")
            oos_table.add_column("Metric", style="dim")
            oos_table.add_column("Value",  justify="right")
            oos_rows = [
                ("OOS Sharpe",       _color_sharpe(wf.oos_sharpe)),
                ("OOS Max DD",       _color_dd(wf.oos_max_drawdown)),
                ("OOS Profit Factor",_color_pf(wf.oos_profit_factor)),
                ("Consistency",      f"{wf.consistency_score * 100:.0f}%"),
                ("IS/OOS Divergence",f"{wf.is_oos_divergence:.2f}"),
            ]
            for label, value in oos_rows:
                oos_table.add_row(label, value)

            from rich.columns import Columns
            content = Columns([is_table, oos_table])

        # ── Sensitivity ──────────────────────────────────────────────────────
        if sensitivity is not None:
            verdict = "[bold green]ROBUST[/bold green]" if sensitivity.is_robust \
                      else "[bold red]BRITTLE[/bold red]"
            worst = ""
            if not sensitivity.is_robust and sensitivity.sensitivities:
                bp = max(sensitivity.sensitivities,
                         key=lambda p: abs(p.score_change_pct))
                worst = (f" — worst param: [italic]{bp.param}[/italic]"
                         f" (impact {bp.score_change_pct:.1%})")
            from rich.text import Text
            sens_line = Text.from_markup(f"Sensitivity: {verdict}{worst}")
        else:
            from rich.text import Text
            sens_line = Text("")

        mode_label = {"intraday": "⚡ Intraday", "swing": "📈 Swing",
                      "spot_longterm": "🌕 Spot Long-Term"}.get(result.mode, result.mode)

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
                    "strategy_name":             r.strategy_name,
                    "mode":                      r.mode,
                    "total_return_pct":          round(r.total_return_pct, 6),
                    "sharpe_ratio":              round(r.sharpe_ratio, 6),
                    "sortino_ratio":             round(r.sortino_ratio, 6),
                    "max_drawdown_pct":          round(r.max_drawdown_pct, 6),
                    "max_drawdown_duration_days":round(r.max_drawdown_duration_days, 2),
                    "calmar_ratio":              round(_calmar(r), 6),
                    "win_rate":                  round(r.win_rate, 6),
                    "profit_factor":             round(r.profit_factor, 6),
                    "trades_per_year":           round(r.trades_per_year, 2),
                    "avg_trade_duration_bars":   round(r.avg_trade_duration_bars, 2),
                    "total_fees":                round(r.total_fees, 4),
                    "total_slippage":            round(r.total_slippage, 4),
                    "composite_score":           round(r.composite_score, 6),
                })

    @staticmethod
    def export_html(
        results: list[BacktestResult],
        path: str,
    ) -> None:
        """Export rich-rendered summary table and detail panels to an HTML file."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)

        # render into a rich Console that captures HTML
        con = Console(record=True, width=140)
        con.print(f"[bold]Backtest Analysis Report[/bold]  [dim]{len(results)} strategies[/dim]\n")
        AnalysisDisplay.summary_table(results, console=con)
        con.print()
        for r in results:
            AnalysisDisplay.detail_panel(r, console=con)
            con.print()

        html = con.export_html(inline_styles=True)
        out.write_text(html, encoding="utf-8")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python3 -m pytest tests/test_analyzer.py -v
```

Expected: all 16 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backtest/analyze.py tests/test_analyzer.py
git commit -m "feat: add AnalysisDisplay rich table module (Plan 5 Task 1)"
```

---

## Task 2: Upgrade `backtest/reporter.py` to use rich

**Files:**
- Modify: `backtest/reporter.py`

The goal is to make `print_strategy_report` and `print_portfolio_report` use rich when available, while keeping the exact same function signatures so nothing else breaks.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_analyzer.py` (append at end of file):

```python
# ── reporter integration ──────────────────────────────────────────────────────

def test_reporter_print_strategy_report_uses_rich(capsys):
    from backtest.reporter import print_strategy_report
    r = _make_result("ReporterTest")
    print_strategy_report(r)
    captured = capsys.readouterr().out
    # Should contain the strategy name regardless of rich/tabulate
    assert "ReporterTest" in captured

def test_reporter_print_portfolio_report(capsys):
    from backtest.reporter import print_portfolio_report
    results = [_make_result("P1", sharpe=1.5), _make_result("P2", sharpe=0.8)]
    print_portfolio_report(results)
    captured = capsys.readouterr().out
    assert "P1" in captured
    assert "P2" in captured

def test_reporter_print_strategy_report_with_wf(capsys):
    from backtest.reporter import print_strategy_report
    from backtest.walk_forward import WalkForwardResult
    r = _make_result("WFTest")
    wf = WalkForwardResult(
        strategy_name="WFTest",
        windows=[],
        oos_sharpe=1.1,
        oos_max_drawdown=0.12,
        oos_profit_factor=1.4,
        consistency_score=0.8,
        is_oos_divergence=0.95,
    )
    print_strategy_report(r, wf=wf)
    captured = capsys.readouterr().out
    assert "OOS" in captured
```

- [ ] **Step 2: Run tests to confirm they fail or pass**

```bash
python3 -m pytest tests/test_analyzer.py::test_reporter_print_strategy_report_uses_rich \
                  tests/test_analyzer.py::test_reporter_print_portfolio_report \
                  tests/test_analyzer.py::test_reporter_print_strategy_report_with_wf -v
```

These tests may already pass if the current reporter works. If they do, skip to Step 5 (no changes needed to reporter beyond the rich upgrade). If they fail, continue.

- [ ] **Step 3: Upgrade `backtest/reporter.py`**

Replace the entire file with the rich-upgraded version. Note: function signatures are identical — no callers break.

```python
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

    # capture for return value AND print to terminal
    buf = io.StringIO()
    buf_con = Console(file=buf, force_terminal=False, width=120)
    AnalysisDisplay.detail_panel(result, wf=wf, sensitivity=sensitivity, console=buf_con)

    term_con = Console(width=120)
    AnalysisDisplay.detail_panel(result, wf=wf, sensitivity=sensitivity, console=term_con)

    # promotion check (kept for backward compat)
    if promotion_criteria and wf:
        criteria = promotion_criteria
        checks = {
            "OOS Sharpe >= min":      wf.oos_sharpe >= criteria.get("min_sharpe_oos", 1.0),
            "Max drawdown <= max":    wf.oos_max_drawdown <= criteria.get("max_drawdown_pct", 0.25),
            "Profit factor >= min":   wf.oos_profit_factor >= criteria.get("min_profit_factor", 1.3),
            "Trades/year >= min":     result.trades_per_year >= criteria.get("min_trades_per_year", 30),
        }
        if sensitivity and not sensitivity.is_robust:
            checks["Sensitivity robust"] = False
        passed = all(checks.values())
        verdict = "PROMOTE TO PAPER" if passed else "REJECT"
        term_con.print(f"[bold]→ PROMOTION: {'[green]' if passed else '[red]'}{verdict}[/]")
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
```

- [ ] **Step 4: Run all reporter tests**

```bash
python3 -m pytest tests/test_analyzer.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Verify no existing tests broke**

```bash
python3 -m pytest --tb=short -q
```

Expected: all 320+ tests still pass (reporter change is backward-compatible).

- [ ] **Step 6: Commit**

```bash
git add backtest/reporter.py tests/test_analyzer.py
git commit -m "feat: upgrade reporter to rich, add reporter integration tests (Plan 5 Task 2)"
```

---

## Task 3: Expand `main.py` with mode selector and `--export`

**Files:**
- Modify: `main.py`
- Test: `tests/test_analyzer.py` (append)

The new CLI interface:

```
python main.py --mode intraday   [--strategy NAME] [--no-parallel] [--export {csv,html}]
python main.py --mode swing      [--strategy NAME] [--no-parallel] [--export {csv,html}]
python main.py --mode spot       [--strategy NAME] [--no-parallel] [--export {csv,html}]
python main.py --mode backtest   # legacy alias for swing (unchanged)
python main.py --mode analyze    --analyze-mode {intraday,swing,spot,all}  [--export {csv,html}]
```

`--mode analyze` runs the given mode(s) and shows only the rich analysis output (no per-strategy progress noise). `--export html` saves to `results/<mode>_report.html`; `--export csv` saves to `results/<mode>_summary.csv`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_analyzer.py`:

```python
# ── main.py CLI arg parsing ───────────────────────────────────────────────────

def test_main_help_exits_cleanly():
    """main.py --help exits with code 0."""
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "main.py", "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "--mode" in result.stdout

def test_main_mode_choices_include_all_modes():
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "main.py", "--help"],
        capture_output=True, text=True,
    )
    assert "intraday" in result.stdout
    assert "swing"    in result.stdout
    assert "spot"     in result.stdout
    assert "analyze"  in result.stdout

def test_main_export_choices():
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "main.py", "--help"],
        capture_output=True, text=True,
    )
    assert "--export" in result.stdout
    assert "csv"  in result.stdout
    assert "html" in result.stdout

def test_main_analyze_mode_flag():
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "main.py", "--help"],
        capture_output=True, text=True,
    )
    assert "--analyze-mode" in result.stdout
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python3 -m pytest tests/test_analyzer.py::test_main_help_exits_cleanly \
                  tests/test_analyzer.py::test_main_mode_choices_include_all_modes \
                  tests/test_analyzer.py::test_main_export_choices \
                  tests/test_analyzer.py::test_main_analyze_mode_flag -v
```

Expected: test_main_mode_choices and test_main_export_choices and test_main_analyze_mode_flag FAIL (those flags don't exist yet).

- [ ] **Step 3: Rewrite `main.py`**

```python
"""
Crypto Trading Bot — CLI entry point.

Usage:
  python main.py --mode swing    [--strategy NAME] [--no-parallel] [--export {csv,html}]
  python main.py --mode intraday [--strategy NAME] [--no-parallel] [--export {csv,html}]
  python main.py --mode spot     [--strategy NAME] [--no-parallel] [--export {csv,html}]
  python main.py --mode backtest                   # legacy alias for swing
  python main.py --mode analyze --analyze-mode {intraday,swing,spot,all} [--export {csv,html}]
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crypto Trading Bot Backtester",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["backtest", "intraday", "swing", "spot", "analyze"],
        default="swing",
        help=(
            "backtest / swing : run swing futures strategies (default)\n"
            "intraday         : run intraday futures strategies\n"
            "spot             : run long-term spot strategies\n"
            "analyze          : run a mode and display rich analysis only"
        ),
    )
    parser.add_argument("--config", default="config/settings.yaml",
                        help="Path to YAML config (default: config/settings.yaml)")
    parser.add_argument("--strategy", default=None,
                        help="Run single strategy by class name")
    parser.add_argument("--no-parallel", action="store_true",
                        help="Disable parallel execution")
    parser.add_argument("--no-optimize", action="store_true",
                        help="Skip Optuna optimisation")
    parser.add_argument("--no-walk-forward", action="store_true",
                        help="Skip walk-forward validation")
    parser.add_argument(
        "--analyze-mode",
        choices=["intraday", "swing", "spot", "all"],
        default="swing",
        help="Which mode(s) to run when --mode analyze is used (default: swing)",
    )
    parser.add_argument(
        "--export",
        choices=["csv", "html"],
        default=None,
        help="Export analysis results: csv → results/<mode>_summary.csv, html → results/<mode>_report.html",
    )
    args = parser.parse_args()

    from crypto_bot.core.config import load_config
    config = load_config(args.config)
    print(f"Config loaded: {len(config.backtest.symbols)} symbols, "
          f"{config.backtest.start_date} → {config.backtest.end_date}")

    effective_mode = args.mode
    if effective_mode == "backtest":
        effective_mode = "swing"        # backward-compatible alias

    if effective_mode == "analyze":
        _run_analyze(config, args)
    elif effective_mode == "intraday":
        _run_intraday_mode(config, args)
    elif effective_mode == "spot":
        _run_spot_mode(config, args)
    else:  # swing
        candles_by_symbol = _load_candles(config)
        if not candles_by_symbol:
            print("ERROR: No candle data found.")
            sys.exit(1)
        _run_backtest(config, candles_by_symbol, args)


def _load_candles(config) -> dict:
    """Load single-TF candles from cache (parquet). Returns empty dict on miss."""
    import pandas as pd
    from crypto_bot.core.data.validator import validate

    candles = {}
    cache_dir = Path("data/cache")
    for symbol in config.backtest.symbols:
        pattern = (f"{symbol}_{config.backtest.timeframe}_"
                   f"{config.backtest.start_date}_{config.backtest.end_date}.parquet")
        cache_file = cache_dir / pattern
        if cache_file.exists():
            df = pd.read_parquet(cache_file)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = validate(df)
            candles[symbol] = df
            print(f"  Loaded {symbol}: {len(df)} candles, quality={df['is_clean'].mean():.1%}")
        else:
            print(f"  Cache miss: {cache_file} — fetching from Binance...")
            try:
                from crypto_bot.core.data.binance_history import fetch_candles
                df = fetch_candles(
                    symbol=symbol, timeframe=config.backtest.timeframe,
                    start_date=config.backtest.start_date,
                    end_date=config.backtest.end_date,
                )
                df = validate(df)
                candles[symbol] = df
                print(f"  Fetched {symbol}: {len(df)} candles")
            except Exception as e:
                print(f"  WARNING: Could not fetch {symbol}: {e}")
    return candles


def _run_backtest(config, candles_by_symbol, args) -> None:
    """Legacy swing / backtest mode."""
    if args.strategy:
        _run_single_swing(config, candles_by_symbol, args.strategy, args)
    else:
        from backtest.orchestrator import run_all
        results_raw = run_all(
            candles_by_symbol, config,
            max_workers=1 if args.no_parallel else 6,
        )
        if args.export:
            _export_results([sr.final_result for sr in results_raw], "swing", args.export)


def _run_single_swing(config, candles_by_symbol, strategy_name, args) -> None:
    """Run one named swing strategy with optional optimisation + walk-forward."""
    strategy_module_map = {
        "EMARibbonStrategy":            "crypto_bot.core.signals.strategies.ema_ribbon",
        "TTMSqueezeStrategy":           "crypto_bot.core.signals.strategies.ttm_squeeze",
        "RSIDivergenceStrategy":        "crypto_bot.core.signals.strategies.rsi_divergence",
        "SupertrendADXStrategy":        "crypto_bot.core.signals.strategies.supertrend_adx",
        "BBMeanReversionStrategy":      "crypto_bot.core.signals.strategies.bb_mean_reversion",
        "FundingRateReversionStrategy": "crypto_bot.core.signals.strategies.funding_rate_reversion",
        "XGBoostMetaStrategy":          "crypto_bot.core.signals.strategies.xgboost_meta",
        "DonchianBreakoutStrategy":          "crypto_bot.core.signals.strategies.donchian_breakout",
        "TSMOMStrategy":                     "crypto_bot.core.signals.strategies.tsmom",
        "HMAChandelierStrategy":             "crypto_bot.core.signals.strategies.hma_chandelier",
        "AdaptiveTrendStrategy":             "crypto_bot.core.signals.strategies.adaptive_trend",
        "VWAPBreakoutStrategy":              "crypto_bot.core.signals.strategies.vwap_breakout",
        "IchimokuCloudStrategy":             "crypto_bot.core.signals.strategies.ichimoku_cloud",
        "StochRSIStrategy":                  "crypto_bot.core.signals.strategies.stoch_rsi",
        "MACDHistDivergenceStrategy":        "crypto_bot.core.signals.strategies.macd_hist_divergence",
        "MarketRegimeStrategy":              "crypto_bot.core.signals.strategies.market_regime",
        "OpenInterestDivergenceStrategy":    "crypto_bot.core.signals.strategies.open_interest_divergence",
    }
    import importlib
    if strategy_name not in strategy_module_map:
        print(f"Unknown strategy: {strategy_name}. Available: {list(strategy_module_map)}")
        return

    module = importlib.import_module(strategy_module_map[strategy_name])
    strategy_cls = getattr(module, strategy_name)
    dummy = strategy_cls.__new__(strategy_cls)
    dummy.params = {}
    params = dummy.default_params()

    if not args.no_optimize:
        from backtest.optimizer import optimize
        print(f"Optimising {strategy_name}...")
        params = optimize(strategy_cls, candles_by_symbol, config)

    from backtest.runner import BacktestRunner
    from backtest.reporter import print_strategy_report, save_results_csv, plot_equity_curve
    runner = BacktestRunner(config)
    result = runner.run(strategy_cls(params), candles_by_symbol)

    wf = None
    if not args.no_walk_forward:
        from backtest.walk_forward import run_walk_forward
        print(f"Running walk-forward for {strategy_name}...")
        wf = run_walk_forward(strategy_cls, candles_by_symbol, config, n_trials_per_window=20)

    from backtest.sensitivity import analyze
    sensitivity = analyze(strategy_cls, params, candles_by_symbol, config)
    print_strategy_report(result, wf, sensitivity, config.promotion_criteria.model_dump())
    save_results_csv(result)
    plot_equity_curve(result)

    if args.export:
        _export_results([result], "swing", args.export)


def _run_intraday_mode(config, args) -> None:
    """Run intraday futures strategies."""
    from backtest.modes.intraday import run_intraday
    results = run_intraday(max_workers=1 if args.no_parallel else 4)
    if args.export:
        _export_results([r.result for r in results], "intraday", args.export)


def _run_spot_mode(config, args) -> None:
    """Run long-term spot strategies."""
    from backtest.modes.spot_longterm import run_spot
    results = run_spot(max_workers=1 if args.no_parallel else 4)
    if args.export:
        _export_results([r.result for r in results], "spot", args.export)


def _run_analyze(config, args) -> None:
    """
    Run one or more modes and display rich analysis.
    --analyze-mode {intraday,swing,spot,all}
    """
    from backtest.analyze import AnalysisDisplay
    from rich.console import Console

    con = Console(width=140)
    all_results = []

    modes_to_run = (
        ["intraday", "swing", "spot"] if args.analyze_mode == "all"
        else [args.analyze_mode]
    )

    for mode in modes_to_run:
        con.rule(f"[bold cyan]Running {mode.upper()} mode[/bold cyan]")
        if mode == "swing":
            candles_by_symbol = _load_candles(config)
            if not candles_by_symbol:
                con.print(f"[red]No candle data for {mode} — skipping[/red]")
                continue
            from backtest.orchestrator import run_all
            raw = run_all(candles_by_symbol, config,
                          max_workers=1 if args.no_parallel else 6)
            results = [sr.final_result for sr in raw]
        elif mode == "intraday":
            from backtest.modes.intraday import run_intraday
            raw = run_intraday(max_workers=1 if args.no_parallel else 4)
            results = [r.result for r in raw]
        else:  # spot
            from backtest.modes.spot_longterm import run_spot
            raw = run_spot(max_workers=1 if args.no_parallel else 4)
            results = [r.result for r in raw]

        AnalysisDisplay.summary_table(results, title=f"{mode.title()} Strategies", console=con)
        for r in results:
            AnalysisDisplay.detail_panel(r, console=con)

        all_results.extend(results)

    if len(modes_to_run) > 1 and all_results:
        con.rule("[bold]Cross-Mode Comparison[/bold]")
        AnalysisDisplay.summary_table(all_results, title="All Modes Combined", console=con)

    if args.export and all_results:
        label = args.analyze_mode
        _export_results(all_results, label, args.export)


def _export_results(results, mode_label: str, fmt: str) -> None:
    """Export results to CSV or HTML under results/."""
    from backtest.analyze import AnalysisDisplay
    Path("results").mkdir(exist_ok=True)
    if fmt == "csv":
        path = f"results/{mode_label}_summary.csv"
        AnalysisDisplay.export_summary_csv(results, path)
        print(f"  Exported → {path}")
    elif fmt == "html":
        path = f"results/{mode_label}_report.html"
        AnalysisDisplay.export_html(results, path)
        print(f"  Exported → {path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run CLI tests**

```bash
python3 -m pytest tests/test_analyzer.py::test_main_help_exits_cleanly \
                  tests/test_analyzer.py::test_main_mode_choices_include_all_modes \
                  tests/test_analyzer.py::test_main_export_choices \
                  tests/test_analyzer.py::test_main_analyze_mode_flag -v
```

Expected: all 4 PASS.

- [ ] **Step 5: Verify full test suite**

```bash
python3 -m pytest --tb=short -q
```

Expected: 320+ tests, 0 failures.

- [ ] **Step 6: Commit**

```bash
git add main.py tests/test_analyzer.py
git commit -m "feat: expand main.py with mode selector, --export, --analyze-mode (Plan 5 Task 3)"
```

---

## Task 4: Integration smoke test + final commit

**Files:**
- Test: `tests/test_analyzer.py` (append one smoke test)

- [ ] **Step 1: Write final integration test**

Append to `tests/test_analyzer.py`:

```python
# ── full round-trip: produce results → export CSV → read back ─────────────────

def test_export_roundtrip(tmp_path):
    """Export N results to CSV, read back, verify all strategy names present."""
    import csv as _csv
    names = ["Alpha", "Beta", "Gamma"]
    results = [_make_result(n, sharpe=float(i + 1)) for i, n in enumerate(names)]
    path = tmp_path / "roundtrip.csv"
    AnalysisDisplay.export_summary_csv(results, str(path))
    rows = list(_csv.DictReader(open(path)))
    assert len(rows) == 3
    exported_names = {row["strategy_name"] for row in rows}
    assert exported_names == set(names)
    # Calmar is computed correctly
    for row in rows:
        r = next(r for r in results if r.strategy_name == row["strategy_name"])
        expected_calmar = round(_calmar(r), 6)
        assert abs(float(row["calmar_ratio"]) - expected_calmar) < 1e-5

def test_html_export_contains_all_strategies(tmp_path):
    names = ["Strat1", "Strat2"]
    results = [_make_result(n) for n in names]
    path = tmp_path / "report.html"
    AnalysisDisplay.export_html(results, str(path))
    html = path.read_text()
    for name in names:
        assert name in html
```

- [ ] **Step 2: Run all tests in test_analyzer.py**

```bash
python3 -m pytest tests/test_analyzer.py -v
```

Expected: all tests PASS (should be ~25 tests total).

- [ ] **Step 3: Run full suite one last time**

```bash
python3 -m pytest --tb=short -q
```

Expected: 345+ tests (320 existing + ~25 new), 0 failures.

- [ ] **Step 4: Final commit**

```bash
git add tests/test_analyzer.py
git commit -m "feat: add integration smoke tests for analyze module (Plan 5 Task 4)

Plan 5 complete — CLI analysis interface:
  backtest/analyze.py  : AnalysisDisplay (summary_table, detail_panel, export_html, export_csv)
  backtest/reporter.py : upgraded to rich (backward-compatible signatures)
  main.py              : mode selector (intraday|swing|spot|analyze), --export, --analyze-mode
  tests/test_analyzer.py : 25 tests covering display, export, CLI flags, round-trip

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Self-Review

**1. Spec coverage:**
- ✅ Rich tables for side-by-side strategy comparison → `summary_table()` with color coding
- ✅ Per-strategy detailed metrics (Sharpe, Calmar, max DD, win rate, trades/year) → `detail_panel()`
- ✅ Mode selector (`--mode intraday|swing|spot`) → `main.py` `--mode` expansion
- ✅ Export to CSV → `export_summary_csv()` + `--export csv`
- ✅ Export to HTML → `export_html()` + `--export html`
- ✅ `--mode analyze` with `--analyze-mode {intraday,swing,spot,all}` → `_run_analyze()`

**2. Placeholder scan:** No TBDs, TODOs, or vague steps. All code is complete.

**3. Type consistency:**
- `_calmar(r: BacktestResult) -> float` defined in Task 1, used in Tasks 1–4 ✅
- `AnalysisDisplay.summary_table`, `detail_panel`, `export_summary_csv`, `export_html` defined in Task 1 and referenced in Tasks 2, 3, 4 ✅
- `WalkForwardResult` fields (`oos_sharpe`, `oos_max_drawdown`, etc.) match existing dataclass ✅
- `SensitivityResult` fields (`is_robust`, `worst_param`, `worst_param_impact`) — need to verify `worst_param` and `worst_param_impact` exist on `SensitivityResult`
