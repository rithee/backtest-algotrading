"""Tests for CoinMarketCap top-N fetcher."""
import json
import pytest
from unittest.mock import patch


@pytest.fixture
def mock_cmc_response():
    return {
        "data": [
            {"symbol": "BTC"}, {"symbol": "ETH"}, {"symbol": "BNB"},
            {"symbol": "SOL"}, {"symbol": "XRP"}, {"symbol": "ADA"},
            {"symbol": "AVAX"}, {"symbol": "DOGE"}, {"symbol": "DOT"},
            {"symbol": "MATIC"},
        ]
    }


def _make_mock_resp(data):
    return type("R", (), {
        "raise_for_status": lambda self: None,
        "json": lambda self: data,
    })()


def test_fetch_top_n_returns_list(tmp_path, mock_cmc_response, monkeypatch):
    from crypto_bot.core.data import coinmarketcap as cmc_mod
    monkeypatch.setattr(cmc_mod, "CACHE_DIR", tmp_path / "cmc")
    monkeypatch.setenv("CMC_API_KEY", "test_key")

    with patch("requests.get", return_value=_make_mock_resp(mock_cmc_response)):
        from crypto_bot.core.data.coinmarketcap import fetch_top_n
        result = fetch_top_n(10, date="2023-06-15")

    assert result is not None
    assert isinstance(result, list)
    assert len(result) == 10
    assert "BTCUSDT" in result


def test_fetch_top_n_returns_default_list_without_api_key(tmp_path, monkeypatch):
    from crypto_bot.core.data import coinmarketcap as cmc_mod
    monkeypatch.setattr(cmc_mod, "CACHE_DIR", tmp_path / "cmc")
    monkeypatch.delenv("CMC_API_KEY", raising=False)

    from crypto_bot.core.data.coinmarketcap import fetch_top_n
    result = fetch_top_n(10)

    assert result is not None
    assert "BTCUSDT" in result
    assert "ETHUSDT" in result


def test_fetch_top_n_uses_monthly_cache(tmp_path, mock_cmc_response, monkeypatch):
    from crypto_bot.core.data import coinmarketcap as cmc_mod
    monkeypatch.setattr(cmc_mod, "CACHE_DIR", tmp_path / "cmc")
    monkeypatch.setenv("CMC_API_KEY", "test_key")

    call_count = {"n": 0}

    def counting_get(*args, **kwargs):
        call_count["n"] += 1
        return _make_mock_resp(mock_cmc_response)

    with patch("requests.get", side_effect=counting_get):
        from crypto_bot.core.data.coinmarketcap import fetch_top_n
        fetch_top_n(10, date="2023-06-15")
        n1 = call_count["n"]
        fetch_top_n(10, date="2023-06-28")   # same month — should use cache
        n2 = call_count["n"]

    assert n2 == n1


def test_fetch_top_n_falls_back_on_api_failure(tmp_path, monkeypatch):
    from crypto_bot.core.data import coinmarketcap as cmc_mod
    monkeypatch.setattr(cmc_mod, "CACHE_DIR", tmp_path / "cmc")
    monkeypatch.setenv("CMC_API_KEY", "test_key")

    with patch("requests.get", side_effect=Exception("timeout")):
        from crypto_bot.core.data.coinmarketcap import fetch_top_n
        result = fetch_top_n(10, date="2023-06-15")

    assert "BTCUSDT" in result   # returns hardcoded default
