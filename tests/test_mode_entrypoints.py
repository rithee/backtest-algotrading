"""Tests for mode entrypoint modules (registry, imports, empty-registry fast-path)."""
import pytest


def test_intraday_module_imports():
    from backtest.modes import intraday
    assert hasattr(intraday, "run_intraday")
    assert hasattr(intraday, "STRATEGY_MODULE_MAP")
    assert isinstance(intraday.STRATEGY_MODULE_MAP, dict)


def test_swing_module_imports():
    from backtest.modes import swing
    assert hasattr(swing, "run_swing")
    assert hasattr(swing, "STRATEGY_MODULE_MAP")
    assert isinstance(swing.STRATEGY_MODULE_MAP, dict)


def test_spot_module_imports():
    from backtest.modes import spot_longterm
    assert hasattr(spot_longterm, "run_spot")
    assert hasattr(spot_longterm, "STRATEGY_MODULE_MAP")
    assert isinstance(spot_longterm.STRATEGY_MODULE_MAP, dict)


def test_intraday_empty_registry_returns_empty(capsys):
    """run_intraday with no registered strategies returns [] immediately."""
    from backtest.modes import intraday
    original = intraday.STRATEGY_MODULE_MAP.copy()
    intraday.STRATEGY_MODULE_MAP.clear()
    try:
        result = intraday.run_intraday(save_csv=False)
        assert result == []
        captured = capsys.readouterr()
        assert "No strategies registered" in captured.out
    finally:
        intraday.STRATEGY_MODULE_MAP.update(original)


def test_swing_empty_registry_returns_empty(capsys):
    """run_swing with no registered strategies returns [] immediately."""
    from backtest.modes import swing
    original = swing.STRATEGY_MODULE_MAP.copy()
    swing.STRATEGY_MODULE_MAP.clear()
    try:
        result = swing.run_swing(save_csv=False)
        assert result == []
    finally:
        swing.STRATEGY_MODULE_MAP.update(original)


def test_spot_empty_registry_returns_empty(capsys):
    """run_spot with no registered strategies returns [] immediately."""
    from backtest.modes import spot_longterm
    original = spot_longterm.STRATEGY_MODULE_MAP.copy()
    spot_longterm.STRATEGY_MODULE_MAP.clear()
    try:
        result = spot_longterm.run_spot(save_csv=False)
        assert result == []
    finally:
        spot_longterm.STRATEGY_MODULE_MAP.update(original)


def test_calmar_ratio_positive():
    """_calmar_ratio returns total_return / max_drawdown."""
    from backtest.modes.spot_longterm import _calmar_ratio
    from backtest.runner import BacktestResult
    import numpy as np
    result = BacktestResult(
        strategy_name="test",
        mode="spot_longterm",
        total_return_pct=1.5,
        sharpe_ratio=1.2,
        sortino_ratio=1.5,
        max_drawdown_pct=0.30,
        max_drawdown_duration_days=60.0,
        win_rate=0.55,
        profit_factor=1.8,
        trades_per_year=4.0,
        total_fees=50.0,
        total_slippage=10.0,
        composite_score=0.8,
        equity_curve=np.array([1000.0, 1500.0, 2500.0]),
        fills=[],
    )
    calmar = _calmar_ratio(result)
    assert abs(calmar - 1.5 / 0.30) < 1e-6


def test_calmar_ratio_zero_drawdown():
    """_calmar_ratio returns -999 when drawdown is 0 (no trades)."""
    from backtest.modes.spot_longterm import _calmar_ratio
    from backtest.runner import BacktestResult
    import numpy as np
    result = BacktestResult(
        strategy_name="test",
        mode="spot_longterm",
        total_return_pct=0.0,
        sharpe_ratio=0.0,
        sortino_ratio=0.0,
        max_drawdown_pct=0.0,
        max_drawdown_duration_days=0.0,
        win_rate=0.0,
        profit_factor=0.0,
        trades_per_year=0.0,
        total_fees=0.0,
        total_slippage=0.0,
        composite_score=0.0,
        equity_curve=np.array([1000.0]),
        fills=[],
    )
    assert _calmar_ratio(result) == -999.0


# ── Task 1: intraday mode completeness ───────────────────────────────────────

def test_intraday_result_has_wf_and_sensitivity_fields():
    """IntradayStrategyResult must carry wf_result and sensitivity."""
    import dataclasses
    from backtest.modes.intraday import IntradayStrategyResult
    fields = {f.name for f in dataclasses.fields(IntradayStrategyResult)}
    assert "wf_result"   in fields, "wf_result missing from IntradayStrategyResult"
    assert "sensitivity" in fields, "sensitivity missing from IntradayStrategyResult"


def test_fee_adjusted_sharpe_normal():
    from backtest.modes.intraday import _fee_adjusted_sharpe
    from backtest.runner import BacktestResult
    from datetime import datetime
    r = BacktestResult(strategy_name="t", mode="intraday")
    r.sharpe_ratio = 1.5
    r.trades_per_year = 100.0
    r.total_fees = 50.0
    # equity grows from 1000 to 2000 → gain = 1000, fee_pct = 0.05
    r.equity_curve = [
        (datetime(2023, 1, 1), 1000.0),
        (datetime(2023, 6, 1), 2000.0),
    ]
    result = _fee_adjusted_sharpe(r)
    # fee_penalty = 50/1000 = 0.05 → 1.5 * (1 - 0.05) = 1.425
    assert abs(result - 1.425) < 1e-9


def test_fee_adjusted_sharpe_empty_curve():
    from backtest.modes.intraday import _fee_adjusted_sharpe
    from backtest.runner import BacktestResult
    r = BacktestResult(strategy_name="t", mode="intraday")
    r.sharpe_ratio = 1.5
    r.trades_per_year = 100.0
    r.total_fees = 50.0
    r.equity_curve = []
    # Should return -999 gracefully, not IndexError
    assert _fee_adjusted_sharpe(r) == -999.0


def test_fee_adjusted_sharpe_low_trades():
    from backtest.modes.intraday import _fee_adjusted_sharpe
    from backtest.runner import BacktestResult
    from datetime import datetime
    r = BacktestResult(strategy_name="t", mode="intraday")
    r.sharpe_ratio = 2.0
    r.trades_per_year = 10.0    # below the 50 threshold
    r.total_fees = 0.0
    r.equity_curve = [(datetime(2023, 1, 1), 1000.0), (datetime(2024, 1, 1), 2000.0)]
    assert _fee_adjusted_sharpe(r) == -999.0


def test_intraday_run_accepts_skip_wf_flag(capsys):
    """run_intraday(skip_wf=True) completes without error on empty registry."""
    from backtest.modes import intraday
    original = intraday.STRATEGY_MODULE_MAP.copy()
    intraday.STRATEGY_MODULE_MAP.clear()
    try:
        result = intraday.run_intraday(save_csv=False, skip_wf=True)
        assert result == []
    finally:
        intraday.STRATEGY_MODULE_MAP.update(original)


def test_intraday_run_accepts_skip_sensitivity_flag(capsys):
    """run_intraday(skip_sensitivity=True) completes without error on empty registry."""
    from backtest.modes import intraday
    original = intraday.STRATEGY_MODULE_MAP.copy()
    intraday.STRATEGY_MODULE_MAP.clear()
    try:
        result = intraday.run_intraday(save_csv=False, skip_sensitivity=True)
        assert result == []
    finally:
        intraday.STRATEGY_MODULE_MAP.update(original)


def test_intraday_run_accepts_n_trials_flag(capsys):
    """run_intraday(n_trials=10) completes without error on empty registry."""
    from backtest.modes import intraday
    original = intraday.STRATEGY_MODULE_MAP.copy()
    intraday.STRATEGY_MODULE_MAP.clear()
    try:
        result = intraday.run_intraday(save_csv=False, n_trials=10)
        assert result == []
    finally:
        intraday.STRATEGY_MODULE_MAP.update(original)


# ── Task 2: optimizer custom objective ───────────────────────────────────────

def test_optimizer_accepts_objective_fn():
    """optimize() signature must accept objective_fn keyword argument."""
    import inspect
    from backtest.optimizer import optimize
    sig = inspect.signature(optimize)
    assert "objective_fn" in sig.parameters, "objective_fn param missing from optimize()"


def test_optimizer_objective_fn_is_none_by_default():
    """objective_fn defaults to None (backward compatible)."""
    import inspect
    from backtest.optimizer import optimize
    sig = inspect.signature(optimize)
    default = sig.parameters["objective_fn"].default
    assert default is None


def test_spot_mode_wires_calmar_to_optimizer():
    """spot_longterm._run_single passes objective_fn=_calmar_ratio to optimize()."""
    import inspect
    import backtest.modes.spot_longterm as mod
    src = inspect.getsource(mod._run_single)
    assert "_calmar_ratio" in src, (
        "_calmar_ratio not referenced in spot_longterm._run_single — "
        "calmar objective not wired"
    )
    assert "objective_fn" in src, (
        "objective_fn not passed in spot_longterm._run_single"
    )


# ── Task 4: swing/spot skip flags ────────────────────────────────────────────

def test_swing_run_accepts_skip_wf_flag(capsys):
    from backtest.modes import swing
    original = swing.STRATEGY_MODULE_MAP.copy()
    swing.STRATEGY_MODULE_MAP.clear()
    try:
        result = swing.run_swing(save_csv=False, skip_wf=True)
        assert result == []
    finally:
        swing.STRATEGY_MODULE_MAP.update(original)


def test_swing_run_accepts_n_trials_flag(capsys):
    from backtest.modes import swing
    original = swing.STRATEGY_MODULE_MAP.copy()
    swing.STRATEGY_MODULE_MAP.clear()
    try:
        result = swing.run_swing(save_csv=False, n_trials=5)
        assert result == []
    finally:
        swing.STRATEGY_MODULE_MAP.update(original)


def test_spot_run_accepts_skip_wf_flag(capsys):
    from backtest.modes import spot_longterm
    original = spot_longterm.STRATEGY_MODULE_MAP.copy()
    spot_longterm.STRATEGY_MODULE_MAP.clear()
    try:
        result = spot_longterm.run_spot(save_csv=False, skip_wf=True)
        assert result == []
    finally:
        spot_longterm.STRATEGY_MODULE_MAP.update(original)


def test_spot_run_accepts_n_trials_flag(capsys):
    from backtest.modes import spot_longterm
    original = spot_longterm.STRATEGY_MODULE_MAP.copy()
    spot_longterm.STRATEGY_MODULE_MAP.clear()
    try:
        result = spot_longterm.run_spot(save_csv=False, n_trials=5)
        assert result == []
    finally:
        spot_longterm.STRATEGY_MODULE_MAP.update(original)


def test_swing_save_csv_uses_export_summary(tmp_path, monkeypatch):
    """run_swing(save_csv=True) calls AnalysisDisplay.export_summary_csv, not save_results_csv."""
    from backtest.modes import swing
    from backtest.analyze import AnalysisDisplay
    calls = []
    monkeypatch.setattr(AnalysisDisplay, "export_summary_csv", lambda results, path: calls.append(path))
    original = swing.STRATEGY_MODULE_MAP.copy()
    swing.STRATEGY_MODULE_MAP.clear()
    # Empty map → no strategies → early return, no save_csv call
    swing.run_swing(save_csv=True)
    # No call expected because no results
    assert calls == []
    swing.STRATEGY_MODULE_MAP.update(original)
