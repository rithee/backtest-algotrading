from pathlib import Path
from crypto_bot.core.config import load_config, Config


def test_load_config_returns_config_object():
    cfg = load_config("config/settings.yaml")
    assert isinstance(cfg, Config)


def test_config_symbols():
    cfg = load_config("config/settings.yaml")
    assert "BTCUSDT" in cfg.backtest.symbols
    assert "ETHUSDT" in cfg.backtest.symbols


def test_config_leverage_max():
    cfg = load_config("config/settings.yaml")
    assert cfg.leverage.max == 3


def test_config_fees():
    cfg = load_config("config/settings.yaml")
    assert cfg.backtest.fees_pct == 0.001


def test_config_defaults_without_file():
    cfg = Config()
    assert cfg.backtest.initial_capital_per_strategy == 10000.0
