"""Tests that BacktestRunner routes to the correct sub-runner based on mode."""
from crypto_bot.core.config import Config
from backtest.runner import BacktestRunner


def test_runner_defaults_to_swing_mode():
    cfg = Config()
    runner = BacktestRunner(cfg)
    assert runner.mode == "swing"


def test_runner_accepts_intraday_mode():
    cfg = Config.from_yaml("config/intraday.yaml")
    runner = BacktestRunner(cfg, mode="intraday")
    assert runner.mode == "intraday"


def test_runner_accepts_spot_longterm_mode():
    cfg = Config.from_yaml("config/spot_longterm.yaml")
    runner = BacktestRunner(cfg, mode="spot_longterm")
    assert runner.mode == "spot_longterm"
