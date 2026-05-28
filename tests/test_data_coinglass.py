"""Tests for CoinGlass liquidation data fetcher."""
import pandas as pd
import pytest
from unittest.mock import patch


@pytest.fixture
def mock_coinglass_response():
    return {
        "data": [
            {"createTime": 1672531200000, "longLiquidationUsd": 1200000.0, "shortLiquidationUsd": 800000.0},
            {"createTime": 1672617600000, "longLiquidationUsd": 500000.0, "shortLiquidationUsd": 3000000.0},
        ]
    }


def _make_mock_resp(data):
    return type("R", (), {
        "raise_for_status": lambda self: None,
        "json": lambda self: data,
    })()


def test_fetch_liquidations_returns_dataframe(tmp_path, mock_coinglass_response, monkeypatch):
    from crypto_bot.core.data import coinglass as cg_mod
    monkeypatch.setattr(cg_mod, "CACHE_DIR", tmp_path / "coinglass")
    monkeypatch.setenv("COINGLASS_API_KEY", "test_key")

    with patch("requests.get", return_value=_make_mock_resp(mock_coinglass_response)):
        from crypto_bot.core.data.coinglass import fetch_liquidations
        result = fetch_liquidations("BTCUSDT", "2023-01-01", "2023-12-31")

    assert result is not None
    assert isinstance(result, pd.DataFrame)
    assert "longLiquidationUsd" in result.columns


def test_fetch_liquidations_returns_none_without_api_key(tmp_path, monkeypatch):
    from crypto_bot.core.data import coinglass as cg_mod
    monkeypatch.setattr(cg_mod, "CACHE_DIR", tmp_path / "coinglass")
    monkeypatch.delenv("COINGLASS_API_KEY", raising=False)

    from crypto_bot.core.data.coinglass import fetch_liquidations
    result = fetch_liquidations("BTCUSDT", "2023-01-01", "2023-12-31")
    assert result is None


def test_fetch_liquidations_returns_none_on_failure(tmp_path, monkeypatch):
    from crypto_bot.core.data import coinglass as cg_mod
    monkeypatch.setattr(cg_mod, "CACHE_DIR", tmp_path / "coinglass")
    monkeypatch.setenv("COINGLASS_API_KEY", "test_key")

    with patch("requests.get", side_effect=Exception("timeout")):
        from crypto_bot.core.data.coinglass import fetch_liquidations
        result = fetch_liquidations("BTCUSDT", "2023-01-01", "2023-12-31")
    assert result is None


def test_fetch_liquidations_strips_usdt_suffix(tmp_path, mock_coinglass_response, monkeypatch):
    """SOLUSDT should be sent to API as SOL."""
    from crypto_bot.core.data import coinglass as cg_mod
    monkeypatch.setattr(cg_mod, "CACHE_DIR", tmp_path / "coinglass")
    monkeypatch.setenv("COINGLASS_API_KEY", "test_key")

    captured = {}

    def capture_get(url, **kwargs):
        captured["params"] = kwargs.get("params", {})
        return _make_mock_resp(mock_coinglass_response)

    with patch("requests.get", side_effect=capture_get):
        from crypto_bot.core.data.coinglass import fetch_liquidations
        fetch_liquidations("SOLUSDT", "2023-01-01", "2023-12-31")

    assert captured["params"].get("symbol") == "SOL"
