"""Tests for Glassnode on-chain metrics fetcher."""
import pandas as pd
import pytest
from unittest.mock import patch


@pytest.fixture
def mock_glassnode_response():
    return [
        {"t": 1672531200, "v": 1.23},
        {"t": 1672617600, "v": 1.45},
        {"t": 1672704000, "v": 0.98},
    ]


def _make_mock_resp(data):
    return type("R", (), {
        "raise_for_status": lambda self: None,
        "json": lambda self: data,
    })()


def test_fetch_glassnode_returns_dataframe(tmp_path, mock_glassnode_response, monkeypatch):
    from crypto_bot.core.data import glassnode as gn_mod
    monkeypatch.setattr(gn_mod, "CACHE_DIR", tmp_path / "glassnode")
    monkeypatch.setenv("GLASSNODE_API_KEY", "test_key")

    with patch("requests.get", return_value=_make_mock_resp(mock_glassnode_response)):
        from crypto_bot.core.data.glassnode import fetch_glassnode
        result = fetch_glassnode("mvrv_zscore", "BTC", "2023-01-01", "2023-12-31")

    assert result is not None
    assert isinstance(result, pd.DataFrame)
    assert "mvrv_zscore" in result.columns


def test_fetch_glassnode_returns_none_without_api_key(tmp_path, monkeypatch):
    from crypto_bot.core.data import glassnode as gn_mod
    monkeypatch.setattr(gn_mod, "CACHE_DIR", tmp_path / "glassnode")
    monkeypatch.delenv("GLASSNODE_API_KEY", raising=False)

    from crypto_bot.core.data.glassnode import fetch_glassnode
    result = fetch_glassnode("mvrv_zscore", "BTC", "2023-01-01", "2023-12-31")
    assert result is None


def test_fetch_glassnode_uses_cache(tmp_path, mock_glassnode_response, monkeypatch):
    from crypto_bot.core.data import glassnode as gn_mod
    monkeypatch.setattr(gn_mod, "CACHE_DIR", tmp_path / "glassnode")
    monkeypatch.setenv("GLASSNODE_API_KEY", "test_key")

    call_count = {"n": 0}

    def counting_get(*args, **kwargs):
        call_count["n"] += 1
        return _make_mock_resp(mock_glassnode_response)

    with patch("requests.get", side_effect=counting_get):
        from crypto_bot.core.data.glassnode import fetch_glassnode
        fetch_glassnode("mvrv_zscore", "BTC", "2023-01-01", "2023-12-31")
        n1 = call_count["n"]
        fetch_glassnode("mvrv_zscore", "BTC", "2023-01-01", "2023-12-31")
        n2 = call_count["n"]

    assert n2 == n1


def test_fetch_glassnode_returns_none_on_request_error(tmp_path, monkeypatch):
    from crypto_bot.core.data import glassnode as gn_mod
    monkeypatch.setattr(gn_mod, "CACHE_DIR", tmp_path / "glassnode")
    monkeypatch.setenv("GLASSNODE_API_KEY", "test_key")

    with patch("requests.get", side_effect=Exception("timeout")):
        from crypto_bot.core.data.glassnode import fetch_glassnode
        result = fetch_glassnode("mvrv_zscore", "BTC", "2023-01-01", "2023-12-31")
    assert result is None
