"""Tests for Deribit options fetcher and max pain computation."""
import pandas as pd
import pytest
from unittest.mock import patch


@pytest.fixture
def mock_instruments():
    return {"result": [
        {"instrument_name": "BTC-27JAN23-20000-C", "strike": 20000.0,
         "expiration_timestamp": 1674864000000, "kind": "option"},
        {"instrument_name": "BTC-27JAN23-20000-P", "strike": 20000.0,
         "expiration_timestamp": 1674864000000, "kind": "option"},
        {"instrument_name": "BTC-27JAN23-22000-C", "strike": 22000.0,
         "expiration_timestamp": 1674864000000, "kind": "option"},
        {"instrument_name": "BTC-27JAN23-22000-P", "strike": 22000.0,
         "expiration_timestamp": 1674864000000, "kind": "option"},
    ]}


@pytest.fixture
def mock_order_book():
    return {"result": {"open_interest": 100.0}}


def test_fetch_options_summary_returns_dataframe(tmp_path, mock_instruments, mock_order_book, monkeypatch):
    from crypto_bot.core.data import deribit as db_mod
    monkeypatch.setattr(db_mod, "CACHE_DIR", tmp_path / "deribit")

    responses = [mock_instruments] + [mock_order_book] * 10
    call_count = {"n": 0}

    def mock_get(url, **kwargs):
        idx = call_count["n"]
        call_count["n"] += 1
        data = responses[min(idx, len(responses) - 1)]
        resp = type("R", (), {
            "ok": True,
            "raise_for_status": lambda self: None,
            "json": lambda self, d=data: d,
        })()
        return resp

    with patch("requests.get", side_effect=mock_get):
        from crypto_bot.core.data.deribit import fetch_options_summary
        result = fetch_options_summary("BTC", "2023-01-27")

    assert result is not None
    assert isinstance(result, pd.DataFrame)
    assert "strike" in result.columns
    assert "option_type" in result.columns
    assert "open_interest" in result.columns


def test_compute_max_pain_returns_float():
    from crypto_bot.core.data.deribit import compute_max_pain
    df = pd.DataFrame([
        {"strike": 20000.0, "option_type": "C", "open_interest": 100.0},
        {"strike": 20000.0, "option_type": "P", "open_interest": 150.0},
        {"strike": 22000.0, "option_type": "C", "open_interest": 80.0},
        {"strike": 22000.0, "option_type": "P", "open_interest": 200.0},
    ])
    result = compute_max_pain(df)
    assert isinstance(result, float)
    assert result in {20000.0, 22000.0}


def test_compute_max_pain_returns_none_on_empty():
    from crypto_bot.core.data.deribit import compute_max_pain
    assert compute_max_pain(None) is None
    assert compute_max_pain(pd.DataFrame()) is None


def test_fetch_options_summary_returns_none_on_failure(tmp_path, monkeypatch):
    from crypto_bot.core.data import deribit as db_mod
    monkeypatch.setattr(db_mod, "CACHE_DIR", tmp_path / "deribit")

    with patch("requests.get", side_effect=Exception("timeout")):
        from crypto_bot.core.data.deribit import fetch_options_summary
        result = fetch_options_summary("BTC", "2023-01-27")
    assert result is None
