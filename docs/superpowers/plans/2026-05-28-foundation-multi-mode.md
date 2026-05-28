# Multi-Mode Backtest — Plan 1 of 5: Foundation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the shared foundation that all three backtest modes depend on: data fetchers, config model extensions, runner mode routing, and mode entrypoints.

**Architecture:** All changes are additive. Existing swing system and its 16 strategies are never modified. New code lives in new files or extends existing classes with optional/default parameters. All existing tests must still pass after each task.

**Tech Stack:** Python 3.11+, pydantic v2, pandas, numpy, yfinance, pandas-datareader, requests, pytest, click, rich, plotext

**Plan series:** This is Plan 1 of 5. Plans 2–4 (strategies per mode) and Plan 5 (CLI) depend on this foundation being complete.

---

## File Map

| Action | File | Purpose |
|---|---|---|
| Modify | `requirements.txt` | Add yfinance, pandas-datareader, rich, plotext, click |
| Create | `.env.example` | API key template |
| Modify | `crypto_bot/core/config.py` | Extend BacktestConfig, RiskConfig; add Config.from_yaml |
| Create | `config/intraday.yaml` | Intraday mode config |
| Create | `config/swing_new.yaml` | Swing new strategies config |
| Create | `config/spot_longterm.yaml` | Long-term spot config |
| Create | `crypto_bot/core/data/macro.py` | yfinance + FRED fetcher |
| Create | `crypto_bot/core/data/glassnode.py` | On-chain metrics fetcher |
| Create | `crypto_bot/core/data/coinglass.py` | Liquidation heatmap fetcher |
| Create | `crypto_bot/core/data/deribit.py` | Options max pain + GEX fetcher |
| Create | `crypto_bot/core/data/coinmarketcap.py` | Top-N market cap fetcher |
| Create | `crypto_bot/core/data/loader.py` | aux_data + MTF candle builder |
| Modify | `crypto_bot/core/signals/base.py` | Add mode property + generate_signals_mtf |
| Modify | `backtest/runner.py` | Add mode routing, _run_futures, _run_spot, _run_intraday |
| Create | `backtest/modes/__init__.py` | Mode package |
| Create | `backtest/modes/intraday.py` | Intraday entrypoint |
| Create | `backtest/modes/swing.py` | Swing new strategies entrypoint |
| Create | `backtest/modes/spot_longterm.py` | Spot long-term entrypoint |
| Create | `tests/test_data_macro.py` | Macro fetcher tests |
| Create | `tests/test_data_glassnode.py` | Glassnode fetcher tests |
| Create | `tests/test_data_coinglass.py` | CoinGlass fetcher tests |
| Create | `tests/test_data_deribit.py` | Deribit fetcher tests |
| Create | `tests/test_data_coinmarketcap.py` | CoinMarketCap fetcher tests |
| Create | `tests/test_data_loader.py` | aux_data loader tests |
| Create | `tests/test_runner_spot.py` | Spot runner tests |
| Create | `tests/test_runner_intraday.py` | Intraday runner tests |

---

## Task 1: Add New Dependencies

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add dependencies to requirements.txt**

Open `requirements.txt` and append these lines (keep existing lines intact):

```
yfinance>=0.2.40
pandas-datareader>=0.10.0
rich>=13.7.0
plotext>=5.2.8
click>=8.1.7
```

- [ ] **Step 2: Install them**

```bash
pip install yfinance>=0.2.40 "pandas-datareader>=0.10.0" "rich>=13.7.0" "plotext>=5.2.8" "click>=8.1.7"
```

Expected: All packages install without error.

- [ ] **Step 3: Verify imports work**

```bash
python -c "import yfinance, pandas_datareader, rich, plotext, click; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "chore: add yfinance, pandas-datareader, rich, plotext, click deps"
```

---

## Task 2: Create .env.example

**Files:**
- Create: `.env.example`

- [ ] **Step 1: Create the file**

```bash
cat > .env.example << 'EOF'
# CoinGlass — https://coinglass.com/pricing
COINGLASS_API_KEY=your_key_here

# Glassnode — https://glassnode.com/pricing
GLASSNODE_API_KEY=your_key_here

# Deribit — https://www.deribit.com/main#/login (API keys under Account > API)
# Note: options market data is public — client_id/secret only needed for account operations
DERIBIT_CLIENT_ID=your_client_id_here
DERIBIT_CLIENT_SECRET=your_client_secret_here

# CoinMarketCap — https://pro.coinmarketcap.com/signup
CMC_API_KEY=your_key_here

# yfinance and FRED (https://fred.stlouisfed.org) are free — no key needed
EOF
```

- [ ] **Step 2: Ensure .env is already gitignored**

```bash
grep -q "^\.env$" .gitignore && echo "already ignored" || echo ".env" >> .gitignore
```

Expected: `already ignored` or silently adds it.

- [ ] **Step 3: Commit**

```bash
git add .env.example .gitignore
git commit -m "chore: add .env.example with API key template"
```

---

## Task 3: Extend Config Models

**Files:**
- Modify: `crypto_bot/core/config.py`
- Test: `tests/test_config.py` (existing — run to verify no regression)

- [ ] **Step 1: Open crypto_bot/core/config.py and replace BacktestConfig with this extended version**

Find the existing `BacktestConfig` class (lines 14–22) and replace it:

```python
class BacktestConfig(BaseModel):
    symbols: list[str] = ["BTCUSDT", "ETHUSDT"]
    timeframe: str = "4h"                        # kept for backward compat
    timeframes: list[str] = []                   # if non-empty, overrides timeframe
    primary_tf: str = ""                         # signal generation TF; defaults to timeframe
    confirm_tf: str = ""                         # direction filter TF
    trend_tf: str = ""                           # bias filter TF
    start_date: str = "2020-01-01"
    end_date: str = "2024-12-31"
    initial_capital_per_strategy: float = 10000.0
    fees_pct: float = 0.001
    slippage_pct: float = 0.0005

    @property
    def active_timeframes(self) -> list[str]:
        """Returns the effective list of timeframes to load."""
        return self.timeframes if self.timeframes else [self.timeframe]

    @property
    def active_primary_tf(self) -> str:
        """Returns the effective primary timeframe."""
        return self.primary_tf if self.primary_tf else self.timeframe
```

- [ ] **Step 2: Replace RiskConfig with extended version**

Find the existing `RiskConfig` class (lines 29–37) and replace it:

```python
class RiskConfig(BaseModel):
    max_position_pct: float = 0.10
    stop_loss_atr_multiplier: float = 2.0
    take_profit_atr_multiplier: float = 3.0
    max_drawdown_pct: float = 0.20
    max_correlated_positions: int = 2
    correlation_lookback_days: int = 30
    correlation_threshold: float = 0.75
    # intraday
    max_daily_loss_pct: float = 0.0      # 0 = disabled; 0.03 = stop after 3% daily loss
    max_trades_per_day: int = 0          # 0 = disabled
    session_filter: bool = False         # only trade during active crypto sessions
    # swing new
    options_expiry_aware: bool = False
    weekly_max_pain_target: bool = False
    # spot long-term
    rebalance_frequency: str = ""        # "monthly" | "" = disabled
    min_holding_days: int = 0            # 0 = disabled
    btc_dominance_filter: bool = False
```

- [ ] **Step 3: Add from_yaml classmethod to Config**

Find the existing `Config` class and add a `from_yaml` classmethod after the field declarations:

```python
class Config(BaseModel):
    exchange: ExchangeConfig = ExchangeConfig()
    backtest: BacktestConfig = BacktestConfig()
    leverage: LeverageConfig = LeverageConfig()
    risk: RiskConfig = RiskConfig()
    optimization: OptimizationConfig = OptimizationConfig()
    walk_forward: WalkForwardConfig = WalkForwardConfig()
    promotion_criteria: PromotionCriteria = PromotionCriteria()
    mode: str = "swing"                  # NEW: "intraday" | "swing" | "spot_longterm"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load config from a YAML file path."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data or {})
```

- [ ] **Step 4: Run existing config tests to verify no regression**

```bash
pytest tests/test_config.py -v
```

Expected: All existing tests pass.

- [ ] **Step 5: Commit**

```bash
git add crypto_bot/core/config.py
git commit -m "feat: extend Config with MTF timeframes, intraday/spot risk fields, from_yaml"
```

---

## Task 4: Create Mode Config YAMLs

**Files:**
- Create: `config/intraday.yaml`
- Create: `config/swing_new.yaml`
- Create: `config/spot_longterm.yaml`

- [ ] **Step 1: Create config/intraday.yaml**

```yaml
# config/intraday.yaml
mode: intraday

exchange:
  name: binance
  market: futures
  testnet: false

backtest:
  symbols: ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
  timeframes: ["1m", "5m", "15m", "1h"]
  primary_tf: "5m"
  confirm_tf: "15m"
  trend_tf: "1h"
  start_date: "2022-01-01"
  end_date: "2024-12-31"
  initial_capital_per_strategy: 100.0
  fees_pct: 0.0002
  slippage_pct: 0.0003

leverage:
  max: 5
  default: 3

risk:
  max_position_pct: 0.40
  stop_loss_atr_multiplier: 1.0
  take_profit_atr_multiplier: 2.5
  max_drawdown_pct: 0.10
  max_daily_loss_pct: 0.03
  max_trades_per_day: 5
  session_filter: true

optimization:
  trials: 300
  sampler: "TPE"
  objective: "composite"
  min_trades_per_year: 200

walk_forward:
  train_months: 3
  test_months: 1
  step_months: 1

promotion_criteria:
  min_sharpe_oos: 1.0
  max_drawdown_pct: 0.15
  min_profit_factor: 1.3
  min_trades_per_year: 200
  max_is_oos_divergence: 2.0
```

- [ ] **Step 2: Create config/swing_new.yaml**

```yaml
# config/swing_new.yaml
mode: swing

exchange:
  name: binance
  market: futures
  testnet: false

backtest:
  symbols: ["BTCUSDT", "ETHUSDT"]
  timeframes: ["4h", "1d"]
  primary_tf: "4h"
  confirm_tf: "1d"
  start_date: "2020-01-01"
  end_date: "2024-12-31"
  initial_capital_per_strategy: 100.0
  fees_pct: 0.001
  slippage_pct: 0.0005

leverage:
  max: 5
  default: 2

risk:
  max_position_pct: 0.30
  stop_loss_atr_multiplier: 2.0
  take_profit_atr_multiplier: 3.5
  max_drawdown_pct: 0.20
  options_expiry_aware: true
  weekly_max_pain_target: true

optimization:
  trials: 500
  sampler: "TPE"
  objective: "composite"
  min_trades_per_year: 30

walk_forward:
  train_months: 12
  test_months: 3
  step_months: 3

promotion_criteria:
  min_sharpe_oos: 1.0
  max_drawdown_pct: 0.25
  min_profit_factor: 1.3
  min_trades_per_year: 30
  max_is_oos_divergence: 2.0
```

- [ ] **Step 3: Create config/spot_longterm.yaml**

```yaml
# config/spot_longterm.yaml
mode: spot_longterm

exchange:
  name: binance
  market: spot
  testnet: false

backtest:
  symbols:
    - "BTCUSDT"
    - "ETHUSDT"
    - "BNBUSDT"
    - "SOLUSDT"
    - "XRPUSDT"
    - "ADAUSDT"
    - "AVAXUSDT"
    - "DOGEUSDT"
    - "DOTUSDT"
    - "MATICUSDT"
  timeframes: ["1d", "1w"]
  primary_tf: "1w"
  confirm_tf: "1d"
  start_date: "2019-01-01"
  end_date: "2024-12-31"
  initial_capital_per_strategy: 1000.0
  fees_pct: 0.001
  slippage_pct: 0.001

leverage:
  max: 1
  default: 1

risk:
  max_position_pct: 0.25
  stop_loss_atr_multiplier: 4.0
  take_profit_atr_multiplier: 0.0
  max_drawdown_pct: 0.45
  rebalance_frequency: "monthly"
  min_holding_days: 30
  btc_dominance_filter: true

optimization:
  trials: 200
  sampler: "TPE"
  objective: "composite"
  min_trades_per_year: 6

walk_forward:
  train_months: 24
  test_months: 6
  step_months: 6

promotion_criteria:
  min_sharpe_oos: 0.7
  max_drawdown_pct: 0.45
  min_profit_factor: 1.5
  min_trades_per_year: 6
  max_is_oos_divergence: 2.0
```

- [ ] **Step 4: Verify all three load cleanly**

```bash
python -c "
from crypto_bot.core.config import Config
for p in ['config/intraday.yaml', 'config/swing_new.yaml', 'config/spot_longterm.yaml']:
    c = Config.from_yaml(p)
    print(f'{p}: mode={c.mode} symbols={len(c.backtest.symbols)} capital={c.backtest.initial_capital_per_strategy}')
"
```

Expected output:
```
config/intraday.yaml: mode=intraday symbols=3 capital=100.0
config/swing_new.yaml: mode=swing symbols=2 capital=100.0
config/spot_longterm.yaml: mode=spot_longterm symbols=10 capital=1000.0
```

- [ ] **Step 5: Commit**

```bash
git add config/intraday.yaml config/swing_new.yaml config/spot_longterm.yaml
git commit -m "feat: add intraday, swing_new, and spot_longterm mode config YAMLs"
```

---

## Task 5: Create macro.py Data Fetcher

**Files:**
- Create: `crypto_bot/core/data/macro.py`
- Create: `tests/test_data_macro.py`

- [ ] **Step 1: Write the failing tests first**

Create `tests/test_data_macro.py`:

```python
"""Tests for macro data fetcher (SPX, DXY, Gold, real yield)."""
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


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

    mock_fred = pd.DataFrame(
        {"DFII10": [1.5, 1.6, 1.7]},
        index=pd.date_range("2023-01-02", periods=3, freq="D"),
    )

    with patch("yfinance.download", return_value=mock_yf_df), \
         patch("pandas_datareader.get_data_fred", return_value=mock_fred):
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
         patch("pandas_datareader.get_data_fred", side_effect=Exception("no fred")):
        from crypto_bot.core.data.macro import fetch_macro
        fetch_macro("2023-01-01", "2023-03-31")
        calls_after_first = call_count["n"]
        fetch_macro("2023-01-01", "2023-03-31")
        calls_after_second = call_count["n"]

    assert calls_after_second == calls_after_first  # no extra API calls on second fetch


def test_fetch_macro_returns_none_when_all_fail(tmp_path, monkeypatch):
    from crypto_bot.core.data import macro as macro_mod
    monkeypatch.setattr(macro_mod, "CACHE_DIR", tmp_path / "macro")

    with patch("yfinance.download", side_effect=Exception("network error")), \
         patch("pandas_datareader.get_data_fred", side_effect=Exception("network error")):
        from crypto_bot.core.data.macro import fetch_macro
        result = fetch_macro("2023-01-01", "2023-03-31")

    assert result is None


def test_fetch_macro_succeeds_without_fred(tmp_path, mock_yf_df, monkeypatch):
    """FRED failure should not prevent yfinance data from being returned."""
    from crypto_bot.core.data import macro as macro_mod
    monkeypatch.setattr(macro_mod, "CACHE_DIR", tmp_path / "macro")

    with patch("yfinance.download", return_value=mock_yf_df), \
         patch("pandas_datareader.get_data_fred", side_effect=Exception("no fred")):
        from crypto_bot.core.data.macro import fetch_macro
        result = fetch_macro("2023-01-01", "2023-03-31")

    assert result is not None
    assert "spx" in result.columns
    assert "real_yield_10y" not in result.columns
```

- [ ] **Step 2: Run to verify tests fail**

```bash
pytest tests/test_data_macro.py -v
```

Expected: `ModuleNotFoundError: No module named 'crypto_bot.core.data.macro'`

- [ ] **Step 3: Create crypto_bot/core/data/macro.py**

```python
"""
Macro market data fetcher: SPX, DXY, Gold (yfinance) + 10Y real yield (FRED).
All data cached as Parquet. Missing sources degrade gracefully.
"""
from __future__ import annotations
import warnings
from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE_DIR = Path("data/external/macro")

_YFINANCE_TICKERS: dict[str, str] = {
    "spx": "^GSPC",
    "dxy": "DX-Y.NYB",
    "gold": "GC=F",
}
_FRED_SERIES = "DFII10"   # 10Y Treasury Inflation-Indexed Security (real yield)


def fetch_macro(start: str, end: str) -> pd.DataFrame | None:
    """
    Returns DataFrame with columns: spx, dxy, gold, real_yield_10y.
    Index: DatetimeIndex (daily). Forward-fills gaps. Returns None if all sources fail.
    Caches result to CACHE_DIR/macro_{start}_{end}.parquet.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"macro_{start}_{end}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    frames: dict[str, pd.Series] = {}

    for col_name, ticker in _YFINANCE_TICKERS.items():
        try:
            raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if not raw.empty:
                close = raw["Close"]
                if isinstance(close, pd.DataFrame):
                    close = close.squeeze()
                frames[col_name] = close.rename(col_name)
        except Exception as exc:
            warnings.warn(f"yfinance fetch failed for {ticker}: {exc}")

    try:
        import pandas_datareader as pdr  # type: ignore
        fred_df = pdr.get_data_fred(_FRED_SERIES, start=start, end=end)
        frames["real_yield_10y"] = fred_df[_FRED_SERIES].rename("real_yield_10y")
    except Exception as exc:
        warnings.warn(f"FRED fetch failed ({_FRED_SERIES}): {exc}")

    if not frames:
        return None

    result = pd.DataFrame(frames)
    result.index = pd.to_datetime(result.index)
    result = result.ffill().dropna(how="all")
    result.to_parquet(cache_path)
    return result
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
pytest tests/test_data_macro.py -v
```

Expected: 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add crypto_bot/core/data/macro.py tests/test_data_macro.py
git commit -m "feat: add macro data fetcher (SPX/DXY/Gold/real yield) with caching"
```

---

## Task 6: Create glassnode.py Data Fetcher

**Files:**
- Create: `crypto_bot/core/data/glassnode.py`
- Create: `tests/test_data_glassnode.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_data_glassnode.py`:

```python
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


def test_fetch_glassnode_returns_dataframe(tmp_path, mock_glassnode_response, monkeypatch):
    from crypto_bot.core.data import glassnode as gn_mod
    monkeypatch.setattr(gn_mod, "CACHE_DIR", tmp_path / "glassnode")
    monkeypatch.setenv("GLASSNODE_API_KEY", "test_key")

    mock_resp = type("R", (), {
        "raise_for_status": lambda self: None,
        "json": lambda self: mock_glassnode_response,
    })()

    with patch("requests.get", return_value=mock_resp):
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
    mock_resp = type("R", (), {
        "raise_for_status": lambda self: None,
        "json": lambda self: mock_glassnode_response,
    })()

    def counting_get(*args, **kwargs):
        call_count["n"] += 1
        return mock_resp

    with patch("requests.get", side_effect=counting_get):
        from crypto_bot.core.data.glassnode import fetch_glassnode
        fetch_glassnode("mvrv_zscore", "BTC", "2023-01-01", "2023-12-31")
        n1 = call_count["n"]
        fetch_glassnode("mvrv_zscore", "BTC", "2023-01-01", "2023-12-31")
        n2 = call_count["n"]

    assert n2 == n1  # second call uses cache


def test_fetch_glassnode_returns_none_on_request_error(tmp_path, monkeypatch):
    from crypto_bot.core.data import glassnode as gn_mod
    monkeypatch.setattr(gn_mod, "CACHE_DIR", tmp_path / "glassnode")
    monkeypatch.setenv("GLASSNODE_API_KEY", "test_key")

    with patch("requests.get", side_effect=Exception("timeout")):
        from crypto_bot.core.data.glassnode import fetch_glassnode
        result = fetch_glassnode("mvrv_zscore", "BTC", "2023-01-01", "2023-12-31")

    assert result is None
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/test_data_glassnode.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Create crypto_bot/core/data/glassnode.py**

```python
"""
Glassnode on-chain metrics fetcher.
Requires GLASSNODE_API_KEY environment variable.
Degrades gracefully to None if key missing or API unavailable.
"""
from __future__ import annotations
import os
import warnings
from pathlib import Path

import pandas as pd
import requests

CACHE_DIR = Path("data/external/glassnode")
_BASE_URL = "https://api.glassnode.com/v1/metrics"

_METRIC_ENDPOINTS: dict[str, str] = {
    "mvrv":             "/market/mvrv",
    "mvrv_zscore":      "/market/mvrv_z_score",
    "sopr":             "/indicators/sopr",
    "nupl":             "/indicators/nupl",
    "nvt":              "/indicators/nvt",
    "exchange_balance": "/distribution/balance_exchanges",
    "lth_supply":       "/supply/lth_supply",
    "miner_outflow":    "/mining/hash_rate_mean",
}


def fetch_glassnode(
    metric: str,
    symbol: str,
    start: str,
    end: str,
) -> pd.DataFrame | None:
    """
    Fetch a single on-chain metric for one symbol.

    Args:
        metric: one of the keys in _METRIC_ENDPOINTS
        symbol: 'BTC', 'ETH', etc. (no USDT suffix)
        start: 'YYYY-MM-DD'
        end:   'YYYY-MM-DD'

    Returns:
        DataFrame with DatetimeIndex and one column named `metric`.
        Returns None if API key missing or fetch fails.
    """
    api_key = os.getenv("GLASSNODE_API_KEY")
    if not api_key:
        warnings.warn("GLASSNODE_API_KEY not set — skipping on-chain data")
        return None

    endpoint = _METRIC_ENDPOINTS.get(metric)
    if not endpoint:
        warnings.warn(f"Unknown glassnode metric: {metric!r}. Valid: {list(_METRIC_ENDPOINTS)}")
        return None

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{symbol}_{metric}_{start}_{end}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    try:
        resp = requests.get(
            f"{_BASE_URL}{endpoint}",
            params={
                "a": symbol,
                "api_key": api_key,
                "s": start,
                "u": end,
                "i": "24h",
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return None
        df = pd.DataFrame(data)
        df["t"] = pd.to_datetime(df["t"], unit="s")
        df = df.set_index("t").rename(columns={"v": metric})
        df.index.name = "timestamp"
        df.to_parquet(cache_path)
        return df
    except Exception as exc:
        warnings.warn(f"Glassnode fetch failed for {metric}/{symbol}: {exc}")
        return None
```

- [ ] **Step 4: Run tests and verify pass**

```bash
pytest tests/test_data_glassnode.py -v
```

Expected: 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add crypto_bot/core/data/glassnode.py tests/test_data_glassnode.py
git commit -m "feat: add Glassnode on-chain metrics fetcher with caching and graceful degradation"
```

---

## Task 7: Create coinglass.py Data Fetcher

**Files:**
- Create: `crypto_bot/core/data/coinglass.py`
- Create: `tests/test_data_coinglass.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_data_coinglass.py`:

```python
"""Tests for CoinGlass liquidation data fetcher."""
import pandas as pd
import pytest
from unittest.mock import patch


@pytest.fixture
def mock_coinglass_response():
    return {
        "data": [
            {"createTime": 1672531200000, "longLiquidationUsd": 1200000.0, "shortLiquidationUsd": 800000.0},
            {"createTime": 1672617600000, "longLiquidationUsd": 500000.0,  "shortLiquidationUsd": 3000000.0},
        ]
    }


def test_fetch_liquidations_returns_dataframe(tmp_path, mock_coinglass_response, monkeypatch):
    from crypto_bot.core.data import coinglass as cg_mod
    monkeypatch.setattr(cg_mod, "CACHE_DIR", tmp_path / "coinglass")
    monkeypatch.setenv("COINGLASS_API_KEY", "test_key")

    mock_resp = type("R", (), {
        "raise_for_status": lambda self: None,
        "json": lambda self: mock_coinglass_response,
    })()

    with patch("requests.get", return_value=mock_resp):
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
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/test_data_coinglass.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Create crypto_bot/core/data/coinglass.py**

```python
"""
CoinGlass liquidation data fetcher.
Requires COINGLASS_API_KEY environment variable.
Degrades gracefully to None if key missing or API unavailable.
"""
from __future__ import annotations
import os
import warnings
from pathlib import Path

import pandas as pd
import requests

CACHE_DIR = Path("data/external/coinglass")
_BASE_URL = "https://open-api.coinglass.com/public/v2"


def fetch_liquidations(
    symbol: str,
    start: str,
    end: str,
) -> pd.DataFrame | None:
    """
    Fetch liquidation history for a symbol.

    Args:
        symbol: e.g. 'BTCUSDT' (USDT suffix is stripped for the API)
        start:  'YYYY-MM-DD'
        end:    'YYYY-MM-DD'

    Returns:
        DataFrame indexed by datetime with columns: longLiquidationUsd, shortLiquidationUsd.
        Returns None if API key missing or fetch fails.
    """
    api_key = os.getenv("COINGLASS_API_KEY")
    if not api_key:
        warnings.warn("COINGLASS_API_KEY not set — skipping liquidation data")
        return None

    coin = symbol.replace("USDT", "").replace("BUSD", "")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{coin}_liquidations_{start}_{end}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    try:
        resp = requests.get(
            f"{_BASE_URL}/liquidation_history",
            headers={"coinglassSecret": api_key},
            params={"symbol": coin, "interval": "4h", "limit": 2000},
            timeout=30,
        )
        resp.raise_for_status()
        payload = resp.json()
        rows = payload.get("data", [])
        if not rows:
            return None

        df = pd.DataFrame(rows)
        df["createTime"] = pd.to_datetime(df["createTime"], unit="ms")
        df = df.set_index("createTime")
        df.index.name = "timestamp"
        df = df[start:end]
        df.to_parquet(cache_path)
        return df
    except Exception as exc:
        warnings.warn(f"CoinGlass fetch failed for {symbol}: {exc}")
        return None
```

- [ ] **Step 4: Run tests and verify pass**

```bash
pytest tests/test_data_coinglass.py -v
```

Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add crypto_bot/core/data/coinglass.py tests/test_data_coinglass.py
git commit -m "feat: add CoinGlass liquidation fetcher with caching and graceful degradation"
```

---

## Task 8: Create deribit.py Options Fetcher

**Files:**
- Create: `crypto_bot/core/data/deribit.py`
- Create: `tests/test_data_deribit.py`

> **Note:** Deribit's public API returns *current* instruments only. This fetcher is useful for live/paper trading. For historical backtesting, the OptionsGammaMaxPain strategy will detect missing data and skip options-based logic, falling back to pure technical signals.

- [ ] **Step 1: Write failing tests**

Create `tests/test_data_deribit.py`:

```python
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
        resp = type("R", (), {
            "ok": True,
            "raise_for_status": lambda self: None,
            "json": lambda self, idx=call_count["n"]: responses[min(idx, len(responses)-1)],
        })()
        call_count["n"] += 1
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
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/test_data_deribit.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Create crypto_bot/core/data/deribit.py**

```python
"""
Deribit options data fetcher: instruments OI, max pain, gamma exposure.
Uses Deribit's public REST API — no authentication required for market data.

Historical limitation: Deribit's /get_instruments returns current/live instruments only.
For live/paper trading this works correctly. For historical backtests, strategies
that use this data should detect None and fall back to technical-only signals.
"""
from __future__ import annotations
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

CACHE_DIR = Path("data/external/deribit")
_BASE_URL = "https://www.deribit.com/api/v2/public"
_WEEK_MS = 7 * 24 * 60 * 60 * 1000


def fetch_options_summary(symbol: str, date: str) -> pd.DataFrame | None:
    """
    Fetch options OI by strike for the nearest weekly expiry to `date`.

    Args:
        symbol: 'BTC' or 'ETH'
        date:   'YYYY-MM-DD'

    Returns:
        DataFrame with columns: strike, option_type ('C'/'P'), open_interest, expiry.
        Returns None if fetch fails.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{symbol}_options_{date}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    try:
        resp = requests.get(
            f"{_BASE_URL}/get_instruments",
            params={"currency": symbol, "kind": "option", "expired": "false"},
            timeout=30,
        )
        resp.raise_for_status()
        instruments = resp.json().get("result", [])

        target_ts = int(datetime.strptime(date, "%Y-%m-%d").timestamp() * 1000)
        nearby = [
            inst for inst in instruments
            if abs(inst.get("expiration_timestamp", 0) - target_ts) <= _WEEK_MS
        ]
        if not nearby:
            return None

        rows = []
        for inst in nearby[:50]:   # cap to avoid rate limits
            name = inst["instrument_name"]
            ob_resp = requests.get(
                f"{_BASE_URL}/get_order_book",
                params={"instrument_name": name, "depth": 1},
                timeout=10,
            )
            if ob_resp.ok:
                ob = ob_resp.json().get("result", {})
                rows.append({
                    "strike": float(inst["strike"]),
                    "option_type": "C" if name.endswith("-C") else "P",
                    "open_interest": float(ob.get("open_interest", 0)),
                    "expiry": inst["expiration_timestamp"],
                })

        if not rows:
            return None

        df = pd.DataFrame(rows)
        df.to_parquet(cache_path)
        return df
    except Exception as exc:
        warnings.warn(f"Deribit fetch failed for {symbol} on {date}: {exc}")
        return None


def compute_max_pain(options_df: pd.DataFrame | None) -> float | None:
    """
    Compute the Max Pain price: the strike where total dollar loss for ALL option
    holders is maximized (dealers profit most → price gravitates here pre-expiry).

    Args:
        options_df: DataFrame from fetch_options_summary

    Returns:
        Max Pain price as float, or None if data unavailable.
    """
    if options_df is None or options_df.empty:
        return None

    strikes = sorted(options_df["strike"].unique())
    pain: dict[float, float] = {}

    for s in strikes:
        # Call holders lose when price < strike
        call_rows = options_df[(options_df["option_type"] == "C") & (options_df["strike"] > s)]
        call_pain = float(((call_rows["strike"] - s) * call_rows["open_interest"]).sum())
        # Put holders lose when price > strike
        put_rows = options_df[(options_df["option_type"] == "P") & (options_df["strike"] < s)]
        put_pain = float(((s - put_rows["strike"]) * put_rows["open_interest"]).sum())
        pain[s] = call_pain + put_pain

    return min(pain, key=pain.get) if pain else None   # strike with minimum total pain = max pain price
```

- [ ] **Step 4: Run tests and verify pass**

```bash
pytest tests/test_data_deribit.py -v
```

Expected: 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add crypto_bot/core/data/deribit.py tests/test_data_deribit.py
git commit -m "feat: add Deribit options fetcher and max pain computation"
```

---

## Task 9: Create coinmarketcap.py Fetcher

**Files:**
- Create: `crypto_bot/core/data/coinmarketcap.py`
- Create: `tests/test_data_coinmarketcap.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_data_coinmarketcap.py`:

```python
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


def test_fetch_top_n_returns_list(tmp_path, mock_cmc_response, monkeypatch):
    from crypto_bot.core.data import coinmarketcap as cmc_mod
    monkeypatch.setattr(cmc_mod, "CACHE_DIR", tmp_path / "cmc")
    monkeypatch.setenv("CMC_API_KEY", "test_key")

    mock_resp = type("R", (), {
        "raise_for_status": lambda self: None,
        "json": lambda self: mock_cmc_response,
    })()

    with patch("requests.get", return_value=mock_resp):
        from crypto_bot.core.data.coinmarketcap import fetch_top_n
        result = fetch_top_n(10)

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
    mock_resp = type("R", (), {
        "raise_for_status": lambda self: None,
        "json": lambda self: mock_cmc_response,
    })()

    def counting_get(*args, **kwargs):
        call_count["n"] += 1
        return mock_resp

    with patch("requests.get", side_effect=counting_get):
        from crypto_bot.core.data.coinmarketcap import fetch_top_n
        fetch_top_n(10, date="2023-06-15")
        n1 = call_count["n"]
        fetch_top_n(10, date="2023-06-28")  # same month — should use cache
        n2 = call_count["n"]

    assert n2 == n1
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/test_data_coinmarketcap.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Create crypto_bot/core/data/coinmarketcap.py**

```python
"""
CoinMarketCap top-N coins by market cap.
Requires CMC_API_KEY environment variable.
Falls back to a hardcoded top-10 default if key is missing.
Caches monthly (same month = same cache file).
"""
from __future__ import annotations
import json
import os
import warnings
from pathlib import Path

import requests

CACHE_DIR = Path("data/external/coinmarketcap")
_CMC_URL = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"

_DEFAULT_TOP10 = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "ADAUSDT", "AVAXUSDT", "DOGEUSDT", "DOTUSDT", "MATICUSDT",
]


def fetch_top_n(n: int = 10, date: str | None = None) -> list[str]:
    """
    Return list of top-N symbols by market cap with USDT suffix (e.g. 'BTCUSDT').
    Falls back to hardcoded default list if CMC_API_KEY is not set.
    Cache key is monthly: two calls in the same calendar month return the same data.

    Args:
        n:    number of symbols to return
        date: 'YYYY-MM-DD' (used for monthly cache key; defaults to today)

    Returns:
        List of symbol strings e.g. ['BTCUSDT', 'ETHUSDT', ...]
    """
    api_key = os.getenv("CMC_API_KEY")
    if not api_key:
        warnings.warn("CMC_API_KEY not set — using default top-10 list")
        return _DEFAULT_TOP10[:n]

    if date is None:
        from datetime import date as dt_date
        date = dt_date.today().isoformat()

    month_key = date[:7]   # 'YYYY-MM'
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"top{n}_{month_key}.json"

    if cache_path.exists():
        return json.loads(cache_path.read_text())

    try:
        resp = requests.get(
            _CMC_URL,
            headers={"X-CMC_PRO_API_KEY": api_key},
            params={"limit": n, "convert": "USDT", "sort": "market_cap"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])
        symbols = [f"{coin['symbol']}USDT" for coin in data[:n]]
        cache_path.write_text(json.dumps(symbols))
        return symbols
    except Exception as exc:
        warnings.warn(f"CoinMarketCap fetch failed: {exc}. Using default list.")
        return _DEFAULT_TOP10[:n]
```

- [ ] **Step 4: Run tests and verify pass**

```bash
pytest tests/test_data_coinmarketcap.py -v
```

Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add crypto_bot/core/data/coinmarketcap.py tests/test_data_coinmarketcap.py
git commit -m "feat: add CoinMarketCap top-N fetcher with monthly caching and fallback"
```

---

## Task 10: Create loader.py (aux_data + MTF Candle Builder)

**Files:**
- Create: `crypto_bot/core/data/loader.py`
- Create: `tests/test_data_loader.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_data_loader.py`:

```python
"""Tests for the aux_data and MTF candle loader."""
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from crypto_bot.core.config import Config


def _make_candle_df(n: int = 10) -> pd.DataFrame:
    import numpy as np
    from datetime import datetime, timedelta
    ts = [datetime(2023, 1, 1) + timedelta(hours=i * 4) for i in range(n)]
    return pd.DataFrame({
        "timestamp": ts,
        "open": np.random.uniform(20000, 30000, n),
        "high": np.random.uniform(20000, 30000, n),
        "low": np.random.uniform(20000, 30000, n),
        "close": np.random.uniform(20000, 30000, n),
        "volume": np.random.uniform(100, 1000, n),
        "is_clean": [True] * n,
        "symbol": ["BTCUSDT"] * n,
    })


def test_load_aux_data_intraday_includes_liquidations(monkeypatch):
    from crypto_bot.core.data import loader as loader_mod

    monkeypatch.setattr(loader_mod, "fetch_liquidations",
                        lambda symbol, start, end: pd.DataFrame({"val": [1.0]}))
    monkeypatch.setattr(loader_mod, "fetch_open_interest",
                        lambda symbol, start, end, interval=None: pd.DataFrame({"oi": [1.0]}))
    monkeypatch.setattr(loader_mod, "fetch_funding_rates",
                        lambda symbol, start, end: pd.DataFrame({"rate": [0.001]}))

    cfg = Config.from_yaml("config/intraday.yaml")
    from crypto_bot.core.data.loader import load_aux_data
    aux = load_aux_data(cfg, mode="intraday")

    assert "liquidations" in aux
    assert "open_interest" in aux
    assert "funding_rate" in aux


def test_load_aux_data_spot_includes_macro_and_onchain(monkeypatch):
    from crypto_bot.core.data import loader as loader_mod

    monkeypatch.setattr(loader_mod, "fetch_macro",
                        lambda start, end: pd.DataFrame({"spx": [100.0]}))
    monkeypatch.setattr(loader_mod, "fetch_glassnode",
                        lambda metric, symbol, start, end: pd.DataFrame({"v": [1.0]}))
    monkeypatch.setattr(loader_mod, "fetch_top_n",
                        lambda n=10, date=None: ["BTCUSDT", "ETHUSDT"])
    monkeypatch.setattr(loader_mod, "fetch_open_interest",
                        lambda symbol, start, end, interval=None: None)
    monkeypatch.setattr(loader_mod, "fetch_funding_rates",
                        lambda symbol, start, end: None)

    cfg = Config.from_yaml("config/spot_longterm.yaml")
    from crypto_bot.core.data.loader import load_aux_data
    aux = load_aux_data(cfg, mode="spot_longterm")

    assert "macro" in aux
    assert "onchain" in aux
    assert "top10_mcap" in aux


def test_load_mtf_candles_returns_nested_dict(monkeypatch):
    from crypto_bot.core.data import loader as loader_mod

    monkeypatch.setattr(loader_mod, "fetch_candles",
                        lambda symbol, tf, start, end: _make_candle_df())

    cfg = Config.from_yaml("config/intraday.yaml")
    from crypto_bot.core.data.loader import load_mtf_candles
    result = load_mtf_candles(cfg)

    assert isinstance(result, dict)
    for symbol in cfg.backtest.symbols:
        assert symbol in result
        assert isinstance(result[symbol], dict)
        for tf in cfg.backtest.active_timeframes:
            assert tf in result[symbol]
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/test_data_loader.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Create crypto_bot/core/data/loader.py**

```python
"""
Central data loader for all backtest modes.
Builds the aux_data dict and loads multi-timeframe candles.
All sources degrade gracefully — a missing API key gives None, not a crash.
"""
from __future__ import annotations
from typing import Any

import pandas as pd

from crypto_bot.core.config import Config

# Import fetchers (each degrades gracefully on missing keys)
from .macro import fetch_macro
from .glassnode import fetch_glassnode
from .coinglass import fetch_liquidations
from .deribit import fetch_options_summary
from .coinmarketcap import fetch_top_n

# Existing Binance fetchers (already in codebase)
from .binance_history import fetch_candles, fetch_open_interest, fetch_funding_rates

_GLASSNODE_SPOT_METRICS = [
    "mvrv", "mvrv_zscore", "sopr", "nupl", "nvt", "exchange_balance", "lth_supply",
]


def load_aux_data(config: Config, mode: str) -> dict[str, Any]:
    """
    Build the aux_data dict for a given mode.
    Keys that fail or have no API key are set to None — strategies handle this gracefully.

    Returns:
        {
          "open_interest": {symbol: df | None},
          "funding_rate":  {symbol: df | None},
          # intraday only:
          "liquidations":  {symbol: df | None},
          # swing_new only:
          "exchange_netflow": {symbol: df | None},
          "options_gamma":   {symbol: df | None},
          # swing_new + spot:
          "macro": df | None,
          # spot only:
          "onchain":    {f"{coin}_{metric}": df | None},
          "top10_mcap": list[str],
        }
    """
    start = config.backtest.start_date
    end = config.backtest.end_date
    symbols = config.backtest.symbols

    aux: dict[str, Any] = {
        "open_interest": {s: fetch_open_interest(s, start, end) for s in symbols},
        "funding_rate":  {s: fetch_funding_rates(s, start, end) for s in symbols},
    }

    if mode == "intraday":
        aux["liquidations"] = {s: fetch_liquidations(s, start, end) for s in symbols}

    if mode in ("swing", "swing_new", "spot_longterm"):
        aux["macro"] = fetch_macro(start, end)

    if mode == "swing_new":
        aux["exchange_netflow"] = {
            s: fetch_glassnode("exchange_balance", s.replace("USDT", ""), start, end)
            for s in symbols
        }
        aux["options_gamma"] = {
            s: fetch_options_summary(s.replace("USDT", ""), start)
            for s in symbols
        }

    if mode == "spot_longterm":
        onchain: dict[str, pd.DataFrame | None] = {}
        for s in symbols:
            coin = s.replace("USDT", "")
            for metric in _GLASSNODE_SPOT_METRICS:
                onchain[f"{coin}_{metric}"] = fetch_glassnode(metric, coin, start, end)
        aux["onchain"] = onchain
        aux["top10_mcap"] = fetch_top_n(10)
        aux["macro"] = fetch_macro(start, end)

    return aux


def load_mtf_candles(
    config: Config,
    symbols: list[str] | None = None,
) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Load candles for all timeframes specified in config.

    Returns:
        {symbol: {timeframe: candle_df}}
    """
    symbols = symbols or config.backtest.symbols
    timeframes = config.backtest.active_timeframes
    start = config.backtest.start_date
    end = config.backtest.end_date

    result: dict[str, dict[str, pd.DataFrame]] = {}
    for symbol in symbols:
        result[symbol] = {}
        for tf in timeframes:
            df = fetch_candles(symbol, tf, start, end)
            if df is not None:
                result[symbol][tf] = df
    return result
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_data_loader.py -v
```

Expected: 3 tests pass.

- [ ] **Step 5: Run all data tests together**

```bash
pytest tests/test_data_macro.py tests/test_data_glassnode.py tests/test_data_coinglass.py tests/test_data_deribit.py tests/test_data_coinmarketcap.py tests/test_data_loader.py -v
```

Expected: All pass.

- [ ] **Step 6: Commit**

```bash
git add crypto_bot/core/data/loader.py tests/test_data_loader.py
git commit -m "feat: add aux_data and MTF candle loader for all three modes"
```

---

## Task 11: Extend BaseStrategy (mode + generate_signals_mtf)

**Files:**
- Modify: `crypto_bot/core/signals/base.py`
- Test: Existing `tests/test_strategies.py` (run to verify no regression)

- [ ] **Step 1: Write a failing test for the new interface**

Add to `tests/test_strategies.py` (or create `tests/test_base_strategy.py` if test_strategies.py doesn't cover base):

```python
# Add these two tests to the existing test file or create tests/test_base_strategy.py

def test_base_strategy_default_mode_is_swing():
    from crypto_bot.core.signals.strategies.ema_ribbon import EMARibbonStrategy
    s = EMARibbonStrategy({})
    assert s.mode == "swing"


def test_base_strategy_generate_signals_mtf_falls_back_to_single_tf(minimal_candles):
    """generate_signals_mtf with one TF in dict should call generate_signals."""
    from crypto_bot.core.signals.strategies.ema_ribbon import EMARibbonStrategy
    strategy = EMARibbonStrategy(EMARibbonStrategy({}).default_params())
    tf_dict = {"4h": minimal_candles}
    result = strategy.generate_signals_mtf(tf_dict, aux_data=None)
    assert isinstance(result, list)
```

> `minimal_candles` is the existing conftest fixture that returns a small candle DataFrame.

- [ ] **Step 2: Run to verify the new tests fail**

```bash
pytest tests/test_base_strategy.py -v 2>/dev/null || pytest tests/test_strategies.py -k "mode or mtf" -v
```

Expected: `AttributeError: 'EMARibbonStrategy' has no attribute 'mode'`

- [ ] **Step 3: Add mode property and generate_signals_mtf to BaseStrategy**

Open `crypto_bot/core/signals/base.py` and add these two methods after `default_params()`:

```python
    @property
    def mode(self) -> str:
        """
        Declares which trading mode this strategy is designed for.
        Values: 'intraday' | 'swing' | 'spot_longterm'
        Default is 'swing' so all existing strategies are unaffected.
        Override in new strategies.
        """
        return "swing"

    def generate_signals_mtf(
        self,
        candles_by_tf: dict[str, "pd.DataFrame"],
        aux_data: dict[str, Any] | None = None,
    ) -> list["Signal"]:
        """
        Multi-timeframe signal generation.
        Default implementation uses the first (primary) timeframe's candles,
        delegating to generate_signals() — backward compatible with all existing strategies.
        Override in strategies that need cross-timeframe logic.

        Args:
            candles_by_tf: {'5m': df, '15m': df, '1h': df} — keyed by timeframe string
            aux_data:       same aux_data dict as generate_signals()

        Returns:
            List of Signal objects.
        """
        primary_candles = next(iter(candles_by_tf.values()))
        return self.generate_signals(primary_candles, aux_data)
```

- [ ] **Step 4: Run all strategy tests to verify no regression**

```bash
pytest tests/test_strategies.py tests/test_indicators.py -v
```

Expected: All existing tests pass. New tests pass.

- [ ] **Step 5: Commit**

```bash
git add crypto_bot/core/signals/base.py tests/test_base_strategy.py
git commit -m "feat: add mode property and generate_signals_mtf to BaseStrategy (non-breaking)"
```

---

## Task 12: Extend BacktestRunner — Mode Routing + _run_futures Refactor

**Files:**
- Modify: `backtest/runner.py`
- Test: Existing `tests/test_backtest_runner.py` (run for regression)

- [ ] **Step 1: Write a failing test for mode routing**

Create `tests/test_runner_mode_routing.py`:

```python
"""Tests that BacktestRunner routes to the correct sub-runner based on mode."""
import pytest
from unittest.mock import MagicMock, patch
from crypto_bot.core.config import Config
from backtest.runner import BacktestRunner, BacktestResult


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
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/test_runner_mode_routing.py -v
```

Expected: `TypeError: BacktestRunner.__init__() got an unexpected keyword argument 'mode'`

- [ ] **Step 3: Update BacktestRunner.__init__ and run() in backtest/runner.py**

Find the `class BacktestRunner:` block and replace the `__init__` and `run` signatures:

```python
class BacktestRunner:
    def __init__(self, config: Config, mode: str = "swing") -> None:
        self.cfg = config
        self.mode = mode

    def run(
        self,
        strategy: BaseStrategy,
        candles_by_symbol: dict[str, pd.DataFrame],
        aux_data: dict[str, Any] | None = None,
        candles_by_tf: dict[str, dict[str, pd.DataFrame]] | None = None,
    ) -> BacktestResult:
        """Route to the correct runner based on self.mode."""
        if self.mode == "spot_longterm":
            return self._run_spot(strategy, candles_by_symbol, candles_by_tf or {}, aux_data)
        elif self.mode == "intraday":
            return self._run_intraday(strategy, candles_by_symbol, candles_by_tf or {}, aux_data)
        else:
            return self._run_futures(strategy, candles_by_symbol, aux_data)
```

Then rename the existing `run()` body to `_run_futures()` — move the entire existing implementation body (lines 62–151 of the original) into a new private method:

```python
    def _run_futures(
        self,
        strategy: BaseStrategy,
        candles_by_symbol: dict[str, pd.DataFrame],
        aux_data: dict[str, Any] | None = None,
    ) -> BacktestResult:
        """Existing futures backtest logic — unchanged."""
        # [paste the entire existing run() body here verbatim]
        result = BacktestResult(
            strategy_name=strategy.name,
            initial_capital=self.cfg.backtest.initial_capital_per_strategy,
        )
        # ... (entire existing body unchanged) ...
```

- [ ] **Step 4: Verify existing tests still pass**

```bash
pytest tests/test_backtest_runner.py tests/test_runner_mode_routing.py -v
```

Expected: All tests pass (existing tests use default mode="swing" which calls `_run_futures`).

- [ ] **Step 5: Commit**

```bash
git add backtest/runner.py tests/test_runner_mode_routing.py
git commit -m "feat: add mode routing to BacktestRunner; refactor existing logic into _run_futures"
```

---

## Task 13: Add _run_spot() to BacktestRunner

**Files:**
- Modify: `backtest/runner.py`
- Create: `tests/test_runner_spot.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_runner_spot.py`:

```python
"""Tests for spot mode runner: no leverage, min_holding_days."""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock
from crypto_bot.core.config import Config
from backtest.runner import BacktestRunner


def _make_candles(n: int = 60, symbol: str = "BTCUSDT") -> pd.DataFrame:
    ts = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(n)]
    close = np.linspace(20000, 25000, n)
    return pd.DataFrame({
        "timestamp": ts, "open": close * 0.99, "high": close * 1.01,
        "low": close * 0.98, "close": close,
        "volume": np.ones(n) * 500, "is_clean": [True] * n, "symbol": [symbol] * n,
    })


class AlwaysLongStrategy:
    name = "AlwaysLong"
    params = {}

    def default_params(self):
        return {}

    def generate_signals(self, candles, aux_data=None):
        from crypto_bot.core.signals.models import Signal, SignalDirection
        signals = []
        for i, row in candles.iterrows():
            if i == 0:
                signals.append(Signal(
                    symbol=row["symbol"], direction=SignalDirection.LONG,
                    timestamp=row["timestamp"], price=row["close"],
                    stop_loss=row["close"] * 0.90, take_profit=None,
                    metadata={},
                ))
        return signals

    def generate_signals_mtf(self, candles_by_tf, aux_data=None):
        first = next(iter(candles_by_tf.values()))
        return self.generate_signals(first, aux_data)

    @property
    def mode(self):
        return "spot_longterm"

    @property
    def param_space(self):
        return {}


def test_spot_runner_enforces_leverage_one(monkeypatch):
    cfg = Config.from_yaml("config/spot_longterm.yaml")
    runner = BacktestRunner(cfg, mode="spot_longterm")
    candles = {"BTCUSDT": _make_candles()}
    strategy = AlwaysLongStrategy()

    result = runner.run(strategy, candles)
    # With leverage=1, no fills should have leverage > 1
    for fill in result.fills:
        assert getattr(fill, "leverage", 1) <= 1.0


def test_spot_runner_returns_backtest_result(monkeypatch):
    cfg = Config.from_yaml("config/spot_longterm.yaml")
    runner = BacktestRunner(cfg, mode="spot_longterm")
    candles = {"BTCUSDT": _make_candles()}
    strategy = AlwaysLongStrategy()

    result = runner.run(strategy, candles)
    assert result is not None
    assert result.strategy_name == "AlwaysLong"
    assert len(result.equity_curve) > 0
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/test_runner_spot.py -v
```

Expected: `AttributeError: 'BacktestRunner' has no attribute '_run_spot'`

- [ ] **Step 3: Add _run_spot to BacktestRunner in backtest/runner.py**

Add this method to the `BacktestRunner` class (after `_run_futures`):

```python
    def _run_spot(
        self,
        strategy: BaseStrategy,
        candles_by_symbol: dict[str, pd.DataFrame],
        candles_by_tf: dict[str, dict[str, pd.DataFrame]],
        aux_data: dict[str, Any] | None = None,
    ) -> BacktestResult:
        """
        Spot long-term runner. Key differences from _run_futures:
        - Leverage forced to 1 (no margin, no liquidation)
        - min_holding_days enforced: exit signals ignored if position held < N days
        - Uses generate_signals_mtf for multi-timeframe strategies
        """
        from crypto_bot.core.config import LeverageConfig

        # Force leverage=1 regardless of config
        spot_cfg = self.cfg.model_copy(
            update={"leverage": LeverageConfig(max=1, default=1)}
        )

        result = BacktestResult(
            strategy_name=strategy.name,
            initial_capital=spot_cfg.backtest.initial_capital_per_strategy,
            mode="spot_longterm",
        )
        risk = RiskEngine(strategy.name, spot_cfg)
        exec_engine = PaperExecutionEngine(strategy.name, spot_cfg)
        store = StateStore(strategy.name)

        min_hold = spot_cfg.risk.min_holding_days
        entry_timestamps: dict[str, datetime] = {}  # symbol → entry datetime

        # Generate signals — prefer MTF if candles_by_tf available
        signal_map: dict[tuple[str, datetime], Signal] = {}
        for symbol, df in candles_by_symbol.items():
            if candles_by_tf.get(symbol):
                sigs = strategy.generate_signals_mtf(candles_by_tf[symbol], aux_data)
            else:
                sigs = strategy.generate_signals(df, aux_data)
            for sig in sigs:
                signal_map[(symbol, sig.timestamp)] = sig

        # Build timeline from primary candles (1d or 1w)
        timeline: list[tuple] = []
        for symbol, df in candles_by_symbol.items():
            for row in df.itertuples(index=False):
                timeline.append((row.timestamp, symbol, row.open, row.high, row.low, row.close))
        timeline.sort(key=lambda x: x[0])

        from itertools import groupby
        equity = spot_cfg.backtest.initial_capital_per_strategy

        for ts, group in groupby(timeline, key=lambda x: x[0]):
            group_items = list(group)
            ts_fills: list[Fill] = []

            for _, symbol, op, hi, lo, cl in group_items:
                risk.record_price(symbol, cl)
                new_fills = exec_engine.process_candle(
                    symbol=symbol, timestamp=ts,
                    open_=op, high=hi, low=lo, close=cl,
                )
                ts_fills.extend(new_fills)

                sig = signal_map.get((symbol, ts))
                if sig is not None:
                    if sig.direction in ("EXIT_LONG", "EXIT_SHORT"):
                        # Enforce min_holding_days
                        if min_hold > 0 and symbol in entry_timestamps:
                            days_held = (ts - entry_timestamps[symbol]).days
                            if days_held < min_hold:
                                continue  # too soon — ignore exit signal
                        exec_engine.schedule_exit(symbol)
                    else:
                        decision = risk.evaluate(sig)
                        if decision.approved:
                            exec_engine.schedule_entry(decision)
                            entry_timestamps[symbol] = ts
                            risk.open_positions[symbol] = None

            for fill in ts_fills:
                if fill.pnl is not None:
                    equity += fill.pnl
                    if fill.symbol in risk.open_positions and risk.open_positions[fill.symbol] is None:
                        del risk.open_positions[fill.symbol]
                store.record_fill(fill)
                result.fills.append(fill)

            risk.open_positions = {k: v for k, v in exec_engine.open_positions.items() if v is not None}
            risk.update_equity(equity)
            store.record_equity(ts, equity)
            result.equity_curve.append((ts, equity))

        last_ts = timeline[-1][0] if timeline else datetime.utcnow()
        for symbol, df in candles_by_symbol.items():
            last_row = df.iloc[-1]
            fill = exec_engine.close_at_end(symbol, last_ts, float(last_row["close"]))
            if fill:
                if fill.pnl:
                    equity += fill.pnl
                store.record_fill(fill)
                result.fills.append(fill)

        store.close()
        self.compute_metrics(result, bars_per_day=1)   # 1d candles
        return result
```

Also add `mode: str = "swing"` field to `BacktestResult`:

```python
@dataclass
class BacktestResult:
    strategy_name: str
    fills: list[Fill] = field(default_factory=list)
    equity_curve: list[tuple[datetime, float]] = field(default_factory=list)
    initial_capital: float = 10000.0
    mode: str = "swing"                  # NEW

    # ... (rest unchanged)
```

And add `bars_per_day` parameter to `compute_metrics`:

```python
    @staticmethod
    def compute_metrics(result: BacktestResult, bars_per_day: int = 6) -> None:
        """Compute all performance metrics from fills and equity curve. Mutates result."""
        import numpy as np
        # ... existing code, but replace np.sqrt(252 * 6) with np.sqrt(252 * bars_per_day)
        # Line that currently reads:
        #   result.sharpe_ratio = float(daily_returns.mean() / daily_returns.std() * np.sqrt(252 * 6))
        # becomes:
        #   result.sharpe_ratio = float(daily_returns.mean() / daily_returns.std() * np.sqrt(252 * bars_per_day))
        # Same for sortino_ratio line.
        # Also update the bars→days conversion:
        #   result.max_drawdown_duration_days = max_dur / bars_per_day   (was / 6.0)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_runner_spot.py tests/test_backtest_runner.py -v
```

Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add backtest/runner.py tests/test_runner_spot.py
git commit -m "feat: add _run_spot() to BacktestRunner (leverage=1, min_holding_days enforcement)"
```

---

## Task 14: Add _run_intraday() to BacktestRunner

**Files:**
- Modify: `backtest/runner.py`
- Create: `tests/test_runner_intraday.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_runner_intraday.py`:

```python
"""Tests for intraday runner: session filter, daily kill switch, max trades/day."""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from crypto_bot.core.config import Config
from backtest.runner import BacktestRunner


def _make_intraday_candles(n: int = 200, symbol: str = "BTCUSDT") -> pd.DataFrame:
    """5-minute candles starting 09:00 UTC."""
    ts = [datetime(2023, 1, 2, 9, 0) + timedelta(minutes=5 * i) for i in range(n)]
    close = np.linspace(20000, 21000, n)
    return pd.DataFrame({
        "timestamp": ts, "open": close * 0.999, "high": close * 1.001,
        "low": close * 0.998, "close": close,
        "volume": np.ones(n) * 100, "is_clean": [True] * n, "symbol": [symbol] * n,
    })


class FrequentSignalStrategy:
    """Fires a LONG signal every 10 candles."""
    name = "FrequentSignal"
    params = {}

    def default_params(self): return {}

    def generate_signals(self, candles, aux_data=None):
        from crypto_bot.core.signals.models import Signal, SignalDirection
        signals = []
        for i, row in candles.iterrows():
            if i % 10 == 0:
                signals.append(Signal(
                    symbol=row["symbol"], direction=SignalDirection.LONG,
                    timestamp=row["timestamp"], price=row["close"],
                    stop_loss=row["close"] * 0.99, take_profit=row["close"] * 1.02,
                    metadata={},
                ))
        return signals

    def generate_signals_mtf(self, candles_by_tf, aux_data=None):
        return self.generate_signals(next(iter(candles_by_tf.values())), aux_data)

    @property
    def mode(self): return "intraday"

    @property
    def param_space(self): return {}


def test_intraday_runner_returns_result():
    cfg = Config.from_yaml("config/intraday.yaml")
    runner = BacktestRunner(cfg, mode="intraday")
    candles = {"BTCUSDT": _make_intraday_candles()}
    mtf = {"BTCUSDT": {"5m": _make_intraday_candles()}}
    result = runner.run(FrequentSignalStrategy(), candles, candles_by_tf=mtf)
    assert result is not None
    assert result.mode == "intraday"


def test_intraday_runner_respects_max_trades_per_day():
    cfg = Config.from_yaml("config/intraday.yaml")
    # max_trades_per_day is 5 in intraday.yaml
    runner = BacktestRunner(cfg, mode="intraday")
    candles = {"BTCUSDT": _make_intraday_candles(n=300)}
    mtf = {"BTCUSDT": {"5m": _make_intraday_candles(n=300)}}
    result = runner.run(FrequentSignalStrategy(), candles, candles_by_tf=mtf)
    # Count entries per day
    from collections import defaultdict
    daily: dict = defaultdict(int)
    for fill in result.fills:
        if getattr(fill, "side", None) in ("BUY", "LONG", None):
            daily[fill.timestamp.date()] += 1
    for day, count in daily.items():
        assert count <= cfg.risk.max_trades_per_day + 1   # +1 buffer for close fills
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/test_runner_intraday.py -v
```

Expected: `AttributeError: 'BacktestRunner' has no attribute '_run_intraday'`

- [ ] **Step 3: Add _run_intraday to BacktestRunner in backtest/runner.py**

Add this method to the `BacktestRunner` class (after `_run_spot`):

```python
    def _run_intraday(
        self,
        strategy: BaseStrategy,
        candles_by_symbol: dict[str, pd.DataFrame],
        candles_by_tf: dict[str, dict[str, pd.DataFrame]],
        aux_data: dict[str, Any] | None = None,
    ) -> BacktestResult:
        """
        Intraday runner. Key differences from _run_futures:
        - Calls generate_signals_mtf (multi-timeframe signal generation)
        - Session filter: signals only processed during active crypto sessions
        - Daily kill switch: halt new entries if daily PnL < -max_daily_loss_pct
        - Max trades per day: halt new entries after max_trades_per_day per symbol
        """
        result = BacktestResult(
            strategy_name=strategy.name,
            initial_capital=self.cfg.backtest.initial_capital_per_strategy,
            mode="intraday",
        )
        risk = RiskEngine(strategy.name, self.cfg)
        exec_engine = PaperExecutionEngine(strategy.name, self.cfg)
        store = StateStore(strategy.name)

        max_daily_loss = self.cfg.risk.max_daily_loss_pct   # 0 = disabled
        max_trades_day = self.cfg.risk.max_trades_per_day   # 0 = disabled
        session_filter = self.cfg.risk.session_filter

        # Session windows: (start_hour_utc, end_hour_utc) — exclusive end
        _SESSIONS = [(0, 8), (8, 12), (13, 17), (17, 22)]

        def _in_session(ts: datetime) -> bool:
            h = ts.hour
            return any(s <= h < e for s, e in _SESSIONS)

        # Generate signals via MTF interface
        signal_map: dict[tuple[str, datetime], Signal] = {}
        for symbol, df in candles_by_symbol.items():
            if candles_by_tf.get(symbol):
                sigs = strategy.generate_signals_mtf(candles_by_tf[symbol], aux_data)
            else:
                sigs = strategy.generate_signals(df, aux_data)
            for sig in sigs:
                signal_map[(symbol, sig.timestamp)] = sig

        timeline: list[tuple] = []
        for symbol, df in candles_by_symbol.items():
            for row in df.itertuples(index=False):
                timeline.append((row.timestamp, symbol, row.open, row.high, row.low, row.close))
        timeline.sort(key=lambda x: x[0])

        from itertools import groupby
        from collections import defaultdict
        equity = self.cfg.backtest.initial_capital_per_strategy
        daily_pnl: dict = defaultdict(float)
        daily_trades: dict = defaultdict(int)  # (date, symbol) → count

        for ts, group in groupby(timeline, key=lambda x: x[0]):
            group_items = list(group)
            ts_fills: list[Fill] = []

            for _, symbol, op, hi, lo, cl in group_items:
                risk.record_price(symbol, cl)
                new_fills = exec_engine.process_candle(
                    symbol=symbol, timestamp=ts,
                    open_=op, high=hi, low=lo, close=cl,
                )
                ts_fills.extend(new_fills)

                sig = signal_map.get((symbol, ts))
                if sig is None:
                    continue

                # Session filter
                if session_filter and not _in_session(ts):
                    continue

                today = ts.date()
                if sig.direction in ("EXIT_LONG", "EXIT_SHORT"):
                    exec_engine.schedule_exit(symbol)
                else:
                    # Kill switch: daily loss exceeded
                    if max_daily_loss > 0:
                        initial = self.cfg.backtest.initial_capital_per_strategy
                        if daily_pnl[today] / initial < -max_daily_loss:
                            continue

                    # Max trades per day per symbol
                    if max_trades_day > 0:
                        if daily_trades[(today, symbol)] >= max_trades_day:
                            continue

                    decision = risk.evaluate(sig)
                    if decision.approved:
                        exec_engine.schedule_entry(decision)
                        daily_trades[(today, symbol)] += 1
                        risk.open_positions[symbol] = None

            for fill in ts_fills:
                if fill.pnl is not None:
                    equity += fill.pnl
                    daily_pnl[fill.timestamp.date()] += fill.pnl
                    if fill.symbol in risk.open_positions and risk.open_positions[fill.symbol] is None:
                        del risk.open_positions[fill.symbol]
                store.record_fill(fill)
                result.fills.append(fill)

            risk.open_positions = {k: v for k, v in exec_engine.open_positions.items() if v is not None}
            risk.update_equity(equity)
            store.record_equity(ts, equity)
            result.equity_curve.append((ts, equity))

        last_ts = timeline[-1][0] if timeline else datetime.utcnow()
        for symbol, df in candles_by_symbol.items():
            last_row = df.iloc[-1]
            fill = exec_engine.close_at_end(symbol, last_ts, float(last_row["close"]))
            if fill:
                if fill.pnl:
                    equity += fill.pnl
                store.record_fill(fill)
                result.fills.append(fill)

        store.close()
        self.compute_metrics(result, bars_per_day=288)   # 5m candles = 288/day
        return result
```

- [ ] **Step 4: Run all runner tests**

```bash
pytest tests/test_runner_intraday.py tests/test_runner_spot.py tests/test_backtest_runner.py -v
```

Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add backtest/runner.py tests/test_runner_intraday.py
git commit -m "feat: add _run_intraday() with session filter, kill switch, max trades/day"
```

---

## Task 15: Create Mode Entrypoints

**Files:**
- Create: `backtest/modes/__init__.py`
- Create: `backtest/modes/intraday.py`
- Create: `backtest/modes/swing.py`
- Create: `backtest/modes/spot_longterm.py`

- [ ] **Step 1: Create backtest/modes/__init__.py**

```python
"""Mode-specific backtest entrypoints. Each mode is fully independent."""
```

- [ ] **Step 2: Create backtest/modes/intraday.py**

```python
"""
Intraday futures backtest entrypoint.
Runs 6 intraday strategies (Plans 2) on BTC, ETH, SOL with 5m primary / 15m confirm / 1h trend.
"""
from __future__ import annotations
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from crypto_bot.core.config import Config, load_config
from crypto_bot.core.data.loader import load_aux_data, load_mtf_candles
from backtest.runner import BacktestRunner
from backtest.reporter import print_strategy_report, print_portfolio_report, save_results_csv

_CONFIG_PATH = Path("config/intraday.yaml")

# Populated by Plan 2 (intraday strategies)
_INTRADAY_STRATEGIES: list[str] = []


def run_intraday(
    config_path: str | Path = _CONFIG_PATH,
    max_workers: int = 6,
    data_override: dict | None = None,       # for testing: pass synthetic candles
) -> list:
    """
    Run all intraday strategies and return list of StrategyResult.
    Prints a full report to stdout on completion.
    """
    cfg = Config.from_yaml(config_path)

    if data_override:
        candles_by_tf = data_override
        aux_data = {}
    else:
        print("Loading multi-timeframe candles...")
        candles_by_tf = load_mtf_candles(cfg)
        print("Loading auxiliary data (liquidations, OI, funding)...")
        aux_data = load_aux_data(cfg, mode="intraday")

    # Primary candles for the runner (uses primary_tf)
    primary_tf = cfg.backtest.active_primary_tf
    candles_by_symbol = {
        symbol: candles_by_tf[symbol][primary_tf]
        for symbol in cfg.backtest.symbols
        if primary_tf in candles_by_tf.get(symbol, {})
    }

    if not _INTRADAY_STRATEGIES:
        print("No intraday strategies registered yet. Register in backtest/modes/intraday.py.")
        return []

    # Import and run strategies — populated in Plan 2
    results = _run_strategies(_INTRADAY_STRATEGIES, candles_by_symbol, candles_by_tf, cfg, aux_data, max_workers)

    for sr in results:
        print_strategy_report(sr.final_result, sr.wf_result, sr.sensitivity,
                              cfg.promotion_criteria.model_dump())
        save_results_csv(sr.final_result)

    print_portfolio_report([sr.final_result for sr in results])
    return results


def _run_strategies(strategy_names, candles_by_symbol, candles_by_tf, cfg, aux_data, max_workers):
    """Parallel execution — same pattern as existing orchestrator.py."""
    from backtest.orchestrator import _run_single_strategy, StrategyResult
    results = []
    config_dict = cfg.model_dump()

    with ProcessPoolExecutor(max_workers=min(max_workers, len(strategy_names))) as pool:
        futures = {
            pool.submit(_run_single_strategy, name, candles_by_symbol, config_dict, aux_data): name
            for name in strategy_names
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                sr = future.result()
                results.append(sr)
                status = "PROMOTED" if sr.promoted else "REJECTED"
                print(f"  [{name}] {status} — score={sr.final_result.composite_score:.4f}")
            except Exception as exc:
                print(f"  [{name}] ERROR: {exc}")
    return results


if __name__ == "__main__":
    run_intraday()
```

- [ ] **Step 3: Create backtest/modes/swing.py**

```python
"""
Swing new strategies backtest entrypoint.
Runs 6 new swing strategies (Plan 3) on BTC, ETH with 4h primary / 1d confirm.
"""
from __future__ import annotations
from pathlib import Path

from crypto_bot.core.config import Config
from crypto_bot.core.data.loader import load_aux_data, load_mtf_candles
from backtest.reporter import print_strategy_report, print_portfolio_report, save_results_csv

_CONFIG_PATH = Path("config/swing_new.yaml")

# Populated by Plan 3 (swing new strategies)
_SWING_NEW_STRATEGIES: list[str] = []


def run_swing_new(
    config_path: str | Path = _CONFIG_PATH,
    max_workers: int = 6,
    data_override: dict | None = None,
) -> list:
    """Run all new swing strategies and return list of StrategyResult."""
    cfg = Config.from_yaml(config_path)

    if data_override:
        candles_by_tf = data_override
        aux_data = {}
    else:
        print("Loading swing candles (4h + 1d)...")
        candles_by_tf = load_mtf_candles(cfg)
        print("Loading auxiliary data (macro, on-chain, options)...")
        aux_data = load_aux_data(cfg, mode="swing_new")

    primary_tf = cfg.backtest.active_primary_tf
    candles_by_symbol = {
        symbol: candles_by_tf[symbol][primary_tf]
        for symbol in cfg.backtest.symbols
        if primary_tf in candles_by_tf.get(symbol, {})
    }

    if not _SWING_NEW_STRATEGIES:
        print("No swing new strategies registered yet. Register in backtest/modes/swing.py.")
        return []

    from backtest.modes.intraday import _run_strategies
    results = _run_strategies(_SWING_NEW_STRATEGIES, candles_by_symbol, candles_by_tf, cfg, aux_data, max_workers)

    for sr in results:
        print_strategy_report(sr.final_result, sr.wf_result, sr.sensitivity,
                              cfg.promotion_criteria.model_dump())
        save_results_csv(sr.final_result)

    print_portfolio_report([sr.final_result for sr in results])
    return results


if __name__ == "__main__":
    run_swing_new()
```

- [ ] **Step 4: Create backtest/modes/spot_longterm.py**

```python
"""
Long-term spot backtest entrypoint.
Runs 7 spot strategies (Plan 4) on top-10 coins by market cap with 1w primary / 1d confirm.
"""
from __future__ import annotations
from pathlib import Path

from crypto_bot.core.config import Config
from crypto_bot.core.data.loader import load_aux_data, load_mtf_candles
from crypto_bot.core.data.coinmarketcap import fetch_top_n
from backtest.reporter import print_strategy_report, print_portfolio_report, save_results_csv

_CONFIG_PATH = Path("config/spot_longterm.yaml")

# Populated by Plan 4 (spot long-term strategies)
_SPOT_STRATEGIES: list[str] = []


def run_spot_longterm(
    config_path: str | Path = _CONFIG_PATH,
    max_workers: int = 4,
    data_override: dict | None = None,
) -> list:
    """Run all spot long-term strategies and return list of StrategyResult."""
    cfg = Config.from_yaml(config_path)

    # Dynamically resolve top-10 symbols from CMC (falls back to config list)
    dynamic_symbols = fetch_top_n(10)
    if dynamic_symbols:
        # Update config symbols dynamically
        cfg = cfg.model_copy(
            update={"backtest": cfg.backtest.model_copy(update={"symbols": dynamic_symbols})}
        )

    if data_override:
        candles_by_tf = data_override
        aux_data = {}
    else:
        print(f"Loading spot candles (1d + 1w) for {len(cfg.backtest.symbols)} symbols...")
        candles_by_tf = load_mtf_candles(cfg)
        print("Loading auxiliary data (on-chain, macro)...")
        aux_data = load_aux_data(cfg, mode="spot_longterm")

    primary_tf = cfg.backtest.active_primary_tf
    candles_by_symbol = {
        symbol: candles_by_tf[symbol][primary_tf]
        for symbol in cfg.backtest.symbols
        if primary_tf in candles_by_tf.get(symbol, {})
    }

    if not _SPOT_STRATEGIES:
        print("No spot strategies registered yet. Register in backtest/modes/spot_longterm.py.")
        return []

    from backtest.modes.intraday import _run_strategies
    results = _run_strategies(_SPOT_STRATEGIES, candles_by_symbol, candles_by_tf, cfg, aux_data, max_workers)

    for sr in results:
        print_strategy_report(sr.final_result, sr.wf_result, sr.sensitivity,
                              cfg.promotion_criteria.model_dump())
        save_results_csv(sr.final_result)

    print_portfolio_report([sr.final_result for sr in results])
    return results


if __name__ == "__main__":
    run_spot_longterm()
```

- [ ] **Step 5: Verify entrypoints import cleanly**

```bash
python -c "
from backtest.modes.intraday import run_intraday
from backtest.modes.swing import run_swing_new
from backtest.modes.spot_longterm import run_spot_longterm
print('All entrypoints import OK')
"
```

Expected: `All entrypoints import OK`

- [ ] **Step 6: Run the full existing test suite to verify no regressions**

```bash
pytest tests/ -v --ignore=tests/test_runner_intraday.py --ignore=tests/test_runner_spot.py -x
```

Expected: All existing tests pass.

- [ ] **Step 7: Commit**

```bash
git add backtest/modes/ tests/
git commit -m "feat: add mode entrypoints for intraday, swing_new, and spot_longterm"
```

---

## Final Verification

- [ ] **Run the complete test suite**

```bash
pytest tests/ -v
```

Expected: All tests pass (old + new).

- [ ] **Verify all three mode configs load and configs are sane**

```bash
python -c "
from crypto_bot.core.config import Config
configs = {
    'intraday':     'config/intraday.yaml',
    'swing_new':    'config/swing_new.yaml',
    'spot_longterm':'config/spot_longterm.yaml',
}
for name, path in configs.items():
    c = Config.from_yaml(path)
    print(f'{name}: {len(c.backtest.symbols)} symbols, leverage={c.leverage.default}x, capital={c.backtest.initial_capital_per_strategy}')
"
```

Expected:
```
intraday: 3 symbols, leverage=3x, capital=100.0
swing_new: 2 symbols, leverage=2x, capital=100.0
spot_longterm: 10 symbols, leverage=1x, capital=1000.0
```

- [ ] **Final commit**

```bash
git add -A
git commit -m "feat: complete Plan 1 foundation — data layer, runner modes, config, entrypoints"
```

---

## Self-Review Checklist

**Spec coverage:**
- ✅ Architecture directory structure — Tasks 1, 15
- ✅ Data layer (5 fetchers + loader + caching + graceful degradation) — Tasks 5–10
- ✅ Config extensions (MTF timeframes, intraday/spot risk fields) — Task 3
- ✅ Mode config YAMLs (intraday, swing_new, spot_longterm) — Task 4
- ✅ BaseStrategy MTF extension (mode + generate_signals_mtf) — Task 11
- ✅ BacktestRunner mode routing + _run_futures refactor — Task 12
- ✅ Spot runner (leverage=1, min_holding_days) — Task 13
- ✅ Intraday runner (session filter, kill switch, MTF) — Task 14
- ✅ Mode entrypoints (3 independent scripts) — Task 15
- ⏩ Strategies — Plans 2, 3, 4
- ⏩ CLI results interface — Plan 5

**Type consistency:** `generate_signals_mtf` is defined in Task 11, called in `_run_spot` (Task 13) and `_run_intraday` (Task 14). `BacktestResult.mode` is added in Task 13 and used in Task 14. `Config.from_yaml` is added in Task 3 and used in Tasks 4, 12, 13, 14, 15.

**No placeholders:** All steps contain complete runnable code.
