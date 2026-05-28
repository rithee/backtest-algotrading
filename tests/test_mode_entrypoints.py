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
