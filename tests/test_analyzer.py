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
