"""
Crypto Trading Bot — CLI entry point.

Usage:
  python main.py --mode backtest --config config/settings.yaml
  python main.py --mode backtest --config config/settings.yaml --strategy EMARibbonStrategy
  python main.py --mode backtest --config config/settings.yaml --no-parallel
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Crypto Trading Bot Backtester")
    parser.add_argument("--mode", choices=["backtest"], default="backtest")
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument("--strategy", default=None, help="Run single strategy by class name")
    parser.add_argument("--no-parallel", action="store_true", help="Disable parallel execution")
    parser.add_argument("--no-optimize", action="store_true", help="Skip Optuna optimisation")
    parser.add_argument("--no-walk-forward", action="store_true", help="Skip walk-forward")
    args = parser.parse_args()

    from crypto_bot.core.config import load_config
    config = load_config(args.config)
    print(f"Config loaded: {len(config.backtest.symbols)} symbols, "
          f"{config.backtest.start_date} -> {config.backtest.end_date}")

    # Load candle data
    candles_by_symbol = _load_candles(config)
    if not candles_by_symbol:
        print("ERROR: No candle data found. Run data fetch first or check cache directory.")
        sys.exit(1)

    if args.mode == "backtest":
        _run_backtest(config, candles_by_symbol, args)


def _load_candles(config) -> dict:
    """Load candles from cache (parquet). Returns empty dict if no cache found."""
    from pathlib import Path
    import pandas as pd
    from crypto_bot.core.data.validator import validate

    candles = {}
    cache_dir = Path("data/cache")
    for symbol in config.backtest.symbols:
        pattern = f"{symbol}_{config.backtest.timeframe}_{config.backtest.start_date}_{config.backtest.end_date}.parquet"
        cache_file = cache_dir / pattern
        if cache_file.exists():
            df = pd.read_parquet(cache_file)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = validate(df)
            candles[symbol] = df
            print(f"  Loaded {symbol}: {len(df)} candles, quality={df['is_clean'].mean():.1%}")
        else:
            print(f"  Cache miss: {cache_file} — fetching from Binance...")
            try:
                from crypto_bot.core.data.binance_history import fetch_candles
                df = fetch_candles(
                    symbol=symbol,
                    timeframe=config.backtest.timeframe,
                    start_date=config.backtest.start_date,
                    end_date=config.backtest.end_date,
                )
                df = validate(df)
                candles[symbol] = df
                print(f"  Fetched {symbol}: {len(df)} candles")
            except Exception as e:
                print(f"  WARNING: Could not fetch {symbol}: {e}")
    return candles


def _run_backtest(config, candles_by_symbol, args) -> None:
    if args.strategy:
        # Single strategy mode
        _run_single(config, candles_by_symbol, args.strategy, args)
    else:
        # All strategies (parallel)
        from backtest.orchestrator import run_all
        run_all(
            candles_by_symbol,
            config,
            max_workers=1 if args.no_parallel else 6,
        )


def _run_single(config, candles_by_symbol, strategy_name, args) -> None:
    """Run a single named strategy with optional optimisation + walk-forward."""
    strategy_module_map = {
        "EMARibbonStrategy":            "crypto_bot.core.signals.strategies.ema_ribbon",
        "TTMSqueezeStrategy":           "crypto_bot.core.signals.strategies.ttm_squeeze",
        "RSIDivergenceStrategy":        "crypto_bot.core.signals.strategies.rsi_divergence",
        "SupertrendADXStrategy":        "crypto_bot.core.signals.strategies.supertrend_adx",
        "BBMeanReversionStrategy":      "crypto_bot.core.signals.strategies.bb_mean_reversion",
        "FundingRateReversionStrategy": "crypto_bot.core.signals.strategies.funding_rate_reversion",
        "XGBoostMetaStrategy":          "crypto_bot.core.signals.strategies.xgboost_meta",
        "DonchianBreakoutStrategy":          "crypto_bot.core.signals.strategies.donchian_breakout",
        "TSMOMStrategy":                     "crypto_bot.core.signals.strategies.tsmom",
        "HMAChandelierStrategy":             "crypto_bot.core.signals.strategies.hma_chandelier",
        "AdaptiveTrendStrategy":             "crypto_bot.core.signals.strategies.adaptive_trend",
        "VWAPBreakoutStrategy":              "crypto_bot.core.signals.strategies.vwap_breakout",
        "IchimokuCloudStrategy":             "crypto_bot.core.signals.strategies.ichimoku_cloud",
        "StochRSIStrategy":                  "crypto_bot.core.signals.strategies.stoch_rsi",
        "MACDHistDivergenceStrategy":        "crypto_bot.core.signals.strategies.macd_hist_divergence",
        "MarketRegimeStrategy":              "crypto_bot.core.signals.strategies.market_regime",
        "OpenInterestDivergenceStrategy":    "crypto_bot.core.signals.strategies.open_interest_divergence",
    }
    import importlib
    if strategy_name not in strategy_module_map:
        print(f"Unknown strategy: {strategy_name}. Available: {list(strategy_module_map)}")
        return

    module = importlib.import_module(strategy_module_map[strategy_name])
    strategy_cls = getattr(module, strategy_name)
    dummy = strategy_cls.__new__(strategy_cls)
    dummy.params = {}
    params = dummy.default_params()

    if not args.no_optimize:
        from backtest.optimizer import optimize
        print(f"Optimising {strategy_name}...")
        params = optimize(strategy_cls, candles_by_symbol, config)

    from backtest.runner import BacktestRunner
    from backtest.reporter import print_strategy_report, save_results_csv, plot_equity_curve
    runner = BacktestRunner(config)
    result = runner.run(strategy_cls(params), candles_by_symbol)

    wf = None
    if not args.no_walk_forward:
        from backtest.walk_forward import run_walk_forward
        print(f"Running walk-forward for {strategy_name}...")
        wf = run_walk_forward(strategy_cls, candles_by_symbol, config, n_trials_per_window=20)

    from backtest.sensitivity import analyze
    sensitivity = analyze(strategy_cls, params, candles_by_symbol, config)
    print_strategy_report(result, wf, sensitivity, config.promotion_criteria.model_dump())
    save_results_csv(result)
    plot_equity_curve(result)


if __name__ == "__main__":
    main()
