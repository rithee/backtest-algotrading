"""Tests for fetch_open_interest — uses a mock to avoid real network calls."""
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
from crypto_bot.core.data.binance_history import fetch_open_interest


class TestFetchOpenInterest:
    def _mock_oi_batch(self, n=5, start_ts=1_600_000_000_000):
        interval = 14_400_000  # 4h in ms
        return [
            {
                "symbol": "BTCUSDT",
                "sumOpenInterest": str(10_000 + i * 100),
                "sumOpenInterestValue": str(500_000_000 + i * 1_000_000),
                "timestamp": start_ts + i * interval,
            }
            for i in range(n)
        ]

    def test_returns_series(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "data" / "cache").mkdir(parents=True)

        mock_client = MagicMock()
        # First call returns data, second call returns [] to terminate loop
        mock_client.futures_open_interest_hist.side_effect = [
            self._mock_oi_batch(5), []
        ]

        with patch("crypto_bot.core.data.binance_history.Client", return_value=mock_client):
            result = fetch_open_interest("BTCUSDT", "4h", "2020-09-14", "2020-09-16")

        assert isinstance(result, pd.Series)
        assert len(result) == 5
        assert result.name == "open_interest"

    def test_values_are_floats(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "data" / "cache").mkdir(parents=True)

        mock_client = MagicMock()
        mock_client.futures_open_interest_hist.side_effect = [
            self._mock_oi_batch(3), []
        ]

        with patch("crypto_bot.core.data.binance_history.Client", return_value=mock_client):
            result = fetch_open_interest("BTCUSDT", "4h", "2020-09-14", "2020-09-16")

        assert result.dtype == float

    def test_returns_empty_series_on_no_data(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "data" / "cache").mkdir(parents=True)

        mock_client = MagicMock()
        mock_client.futures_open_interest_hist.return_value = []

        with patch("crypto_bot.core.data.binance_history.Client", return_value=mock_client):
            result = fetch_open_interest("BTCUSDT", "4h", "2020-09-14", "2020-09-15")

        assert isinstance(result, pd.Series)
        assert len(result) == 0

    def test_uses_cache_on_second_call(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "data" / "cache").mkdir(parents=True)

        mock_client = MagicMock()
        # First fetch: data then empty to end loop; second fetch should hit cache
        mock_client.futures_open_interest_hist.side_effect = [
            self._mock_oi_batch(3), []
        ]

        with patch("crypto_bot.core.data.binance_history.Client", return_value=mock_client):
            fetch_open_interest("BTCUSDT", "4h", "2020-09-14", "2020-09-16")
            fetch_open_interest("BTCUSDT", "4h", "2020-09-14", "2020-09-16")

        # futures_open_interest_hist called twice (data + empty), not more
        assert mock_client.futures_open_interest_hist.call_count == 2
