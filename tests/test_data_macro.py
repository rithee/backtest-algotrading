"""Tests for macro data fetcher (SPX, DXY, Gold, real yield)."""
import pandas as pd
import pytest
from unittest.mock import patch


@pytest.fixture
def mock_yf_df():
    """Single-column yfinance-style Close DataFrame."""
    return pd.DataFrame(
        {"Close": [100.0, 101.0, 102.0]},
        index=pd.date_range("2023-01-02", periods=3, freq="D"),
    )


def test_fetch_macro_returns_dataframe(tmp_path, mock_yf_df, monkeypatch):
    from crypto_bot.core.data import macro as macro_mod
    monkeypatch.setattr(macro_mod, "CACHE_DIR", tmp_path / "macro")
    monkeypatch.setenv("FRED_API_KEY", "test_key")

    mock_fred_series = pd.Series(
        [1.5, 1.6, 1.7],
        index=pd.date_range("2023-01-02", periods=3, freq="D"),
        name="DFII10",
    )

    with patch("yfinance.download", return_value=mock_yf_df), \
         patch("fredapi.Fred.get_series", return_value=mock_fred_series):
        from crypto_bot.core.data.macro import fetch_macro
        result = fetch_macro("2023-01-01", "2023-03-31")

    assert result is not None
    assert isinstance(result, pd.DataFrame)
    assert "spx" in result.columns
    assert "real_yield_10y" in result.columns


def test_fetch_macro_uses_cache_on_second_call(tmp_path, mock_yf_df, monkeypatch):
    from crypto_bot.core.data import macro as macro_mod
    monkeypatch.setattr(macro_mod, "CACHE_DIR", tmp_path / "macro")

    call_count = {"n": 0}

    def counting_download(*args, **kwargs):
        call_count["n"] += 1
        return mock_yf_df

    with patch("yfinance.download", side_effect=counting_download), \
         patch("fredapi.Fred.get_series", side_effect=Exception("no fred")):
        from crypto_bot.core.data.macro import fetch_macro
        fetch_macro("2023-01-01", "2023-03-31")
        calls_after_first = call_count["n"]
        fetch_macro("2023-01-01", "2023-03-31")
        calls_after_second = call_count["n"]

    assert calls_after_second == calls_after_first


def test_fetch_macro_returns_none_when_all_fail(tmp_path, monkeypatch):
    from crypto_bot.core.data import macro as macro_mod
    monkeypatch.setattr(macro_mod, "CACHE_DIR", tmp_path / "macro")

    with patch("yfinance.download", side_effect=Exception("network error")), \
         patch("fredapi.Fred.get_series", side_effect=Exception("network error")):
        from crypto_bot.core.data.macro import fetch_macro
        result = fetch_macro("2023-01-01", "2023-03-31")

    assert result is None


def test_fetch_macro_succeeds_without_fred(tmp_path, mock_yf_df, monkeypatch):
    """FRED failure should not prevent yfinance data from being returned."""
    from crypto_bot.core.data import macro as macro_mod
    monkeypatch.setattr(macro_mod, "CACHE_DIR", tmp_path / "macro")

    with patch("yfinance.download", return_value=mock_yf_df), \
         patch("fredapi.Fred.get_series", side_effect=Exception("no fred")):
        from crypto_bot.core.data.macro import fetch_macro
        result = fetch_macro("2023-01-01", "2023-03-31")

    assert result is not None
    assert "spx" in result.columns
    assert "real_yield_10y" not in result.columns
