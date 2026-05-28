"""
Crypto Trading Bot — CLI entry point.

Usage:
  python main.py --mode swing    [--strategy NAME] [--no-parallel] [--export {csv,html}]
  python main.py --mode intraday [--strategy NAME] [--no-parallel] [--export {csv,html}]
  python main.py --mode spot     [--strategy NAME] [--no-parallel] [--export {csv,html}]
  python main.py --mode backtest                   # legacy alias for swing
  python main.py --mode analyze --analyze-mode {intraday,swing,spot,all} [--export {csv,html}]
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crypto Trading Bot Backtester",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["backtest", "intraday", "swing", "spot", "analyze"],
        default="swing",
        help=(
            "backtest / swing : run swing futures strategies (default)\n"
            "intraday         : run intraday futures strategies\n"
            "spot             : run long-term spot strategies\n"
            "analyze          : run a mode and display rich analysis only"
        ),
    )
    parser.add_argument(
        "--config",
        default="config/settings.yaml",
        help="Path to YAML config (default: config/settings.yaml)",
    )
    parser.add_argument(
        "--strategy",
        default=None,
        help="Run single strategy by class name",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel execution",
    )
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Skip Optuna optimisation",
    )
    parser.add_argument(
        "--no-walk-forward",
        action="store_true",
        help="Skip walk-forward validation",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        metavar="N",
        help="Override Optuna trial count for this run (default: from config)",
    )
    parser.add_argument(
        "--skip-wf",
        action="store_true",
        help="Skip walk-forward validation (faster, IS-only promotion gate)",
    )
    parser.add_argument(
        "--skip-sensitivity",
        action="store_true",
        help="Skip parameter sensitivity analysis",
    )
    parser.add_argument(
        "--analyze-mode",
        choices=["intraday", "swing", "spot", "all"],
        default="swing",
        help="Which mode(s) to run when --mode analyze is used (default: swing)",
    )
    parser.add_argument(
        "--export",
        choices=["csv", "html"],
        default=None,
        help=(
            "Export analysis results:\n"
            "  csv  → results/<mode>_summary.csv\n"
            "  html → results/<mode>_report.html"
        ),
    )
    args = parser.parse_args()

    from crypto_bot.core.config import load_config
    config = load_config(args.config)
    print(
        f"Config loaded: {len(config.backtest.symbols)} symbols, "
        f"{config.backtest.start_date} → {config.backtest.end_date}"
    )

    effective_mode = args.mode
    if effective_mode == "backtest":
        effective_mode = "swing"        # backward-compatible alias

    if effective_mode == "analyze":
        _run_analyze(config, args)
    elif effective_mode == "intraday":
        _run_intraday_mode(config, args)
    elif effective_mode == "spot":
        _run_spot_mode(config, args)
    else:  # swing
        candles_by_symbol = _load_candles(config)
        if not candles_by_symbol:
            print("ERROR: No candle data found.")
            sys.exit(1)
        _run_backtest(config, candles_by_symbol, args)


# ── data loading ──────────────────────────────────────────────────────────────

def _load_candles(config) -> dict:
    """Load single-TF candles from cache (parquet). Returns empty dict on miss."""
    import pandas as pd
    from crypto_bot.core.data.validator import validate

    candles = {}
    cache_dir = Path("data/cache")
    for symbol in config.backtest.symbols:
        pattern = (
            f"{symbol}_{config.backtest.timeframe}_"
            f"{config.backtest.start_date}_{config.backtest.end_date}.parquet"
        )
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


# ── mode runners ──────────────────────────────────────────────────────────────

def _run_backtest(config, candles_by_symbol, args) -> None:
    """Legacy swing / backtest mode."""
    if args.strategy:
        _run_single_swing(config, candles_by_symbol, args.strategy, args)
    else:
        from backtest.orchestrator import run_all
        results_raw = run_all(
            candles_by_symbol, config,
            max_workers=1 if args.no_parallel else 6,
        )
        if args.export:
            _export_results([sr.final_result for sr in results_raw], "swing", args.export)


def _run_single_swing(config, candles_by_symbol, strategy_name, args) -> None:
    """Run one named swing strategy with optional optimisation + walk-forward."""
    strategy_module_map = {
        "EMARibbonStrategy":            "crypto_bot.core.signals.strategies.ema_ribbon",
        "TTMSqueezeStrategy":           "crypto_bot.core.signals.strategies.ttm_squeeze",
        "RSIDivergenceStrategy":        "crypto_bot.core.signals.strategies.rsi_divergence",
        "SupertrendADXStrategy":        "crypto_bot.core.signals.strategies.supertrend_adx",
        "BBMeanReversionStrategy":      "crypto_bot.core.signals.strategies.bb_mean_reversion",
        "FundingRateReversionStrategy": "crypto_bot.core.signals.strategies.funding_rate_reversion",
        "XGBoostMetaStrategy":          "crypto_bot.core.signals.strategies.xgboost_meta",
        "DonchianBreakoutStrategy":     "crypto_bot.core.signals.strategies.donchian_breakout",
        "TSMOMStrategy":                "crypto_bot.core.signals.strategies.tsmom",
        "HMAChandelierStrategy":        "crypto_bot.core.signals.strategies.hma_chandelier",
        "AdaptiveTrendStrategy":        "crypto_bot.core.signals.strategies.adaptive_trend",
        "VWAPBreakoutStrategy":         "crypto_bot.core.signals.strategies.vwap_breakout",
        "IchimokuCloudStrategy":        "crypto_bot.core.signals.strategies.ichimoku_cloud",
        "StochRSIStrategy":             "crypto_bot.core.signals.strategies.stoch_rsi",
        "MACDHistDivergenceStrategy":   "crypto_bot.core.signals.strategies.macd_hist_divergence",
        "MarketRegimeStrategy":         "crypto_bot.core.signals.strategies.market_regime",
        "OpenInterestDivergenceStrategy":"crypto_bot.core.signals.strategies.open_interest_divergence",
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

    if args.export:
        _export_results([result], "swing", args.export)


def _run_intraday_mode(config, args) -> None:
    """Run intraday futures strategies."""
    from backtest.modes.intraday import run_intraday
    results = run_intraday(
        max_workers=1 if args.no_parallel else 4,
        n_trials=args.trials,
        skip_wf=args.skip_wf,
        skip_sensitivity=args.skip_sensitivity,
    )
    if args.export:
        _export_results([r.result for r in results], "intraday", args.export)


def _run_spot_mode(config, args) -> None:
    """Run long-term spot strategies."""
    from backtest.modes.spot_longterm import run_spot
    results = run_spot(
        max_workers=1 if args.no_parallel else 4,
        n_trials=args.trials,
        skip_wf=args.skip_wf,
        skip_sensitivity=args.skip_sensitivity,
    )
    if args.export:
        _export_results([r.result for r in results], "spot", args.export)


def _run_analyze(config, args) -> None:
    """
    Run one or more modes and display rich analysis.
    --analyze-mode {intraday,swing,spot,all}
    """
    from backtest.analyze import AnalysisDisplay
    from rich.console import Console

    con = Console(width=140)
    all_results = []

    modes_to_run = (
        ["intraday", "swing", "spot"] if args.analyze_mode == "all"
        else [args.analyze_mode]
    )

    for mode in modes_to_run:
        con.rule(f"[bold cyan]Running {mode.upper()} mode[/bold cyan]")
        if mode == "swing":
            candles_by_symbol = _load_candles(config)
            if not candles_by_symbol:
                con.print(f"[red]No candle data for {mode} — skipping[/red]")
                continue
            from backtest.modes.swing import run_swing
            raw = run_swing(
                max_workers=1 if args.no_parallel else 4,
                n_trials=args.trials,
                skip_wf=args.skip_wf,
                skip_sensitivity=args.skip_sensitivity,
            )
            results = [sr.result for sr in raw]
        elif mode == "intraday":
            from backtest.modes.intraday import run_intraday
            raw = run_intraday(
                max_workers=1 if args.no_parallel else 4,
                n_trials=args.trials,
                skip_wf=args.skip_wf,
                skip_sensitivity=args.skip_sensitivity,
            )
            results = [r.result for r in raw]
        else:  # spot
            from backtest.modes.spot_longterm import run_spot
            raw = run_spot(
                max_workers=1 if args.no_parallel else 4,
                n_trials=args.trials,
                skip_wf=args.skip_wf,
                skip_sensitivity=args.skip_sensitivity,
            )
            results = [r.result for r in raw]

        AnalysisDisplay.summary_table(results, title=f"{mode.title()} Strategies", console=con)
        for r in results:
            AnalysisDisplay.detail_panel(r, console=con)

        all_results.extend(results)

    if len(modes_to_run) > 1 and all_results:
        con.rule("[bold]Cross-Mode Comparison[/bold]")
        AnalysisDisplay.summary_table(all_results, title="All Modes Combined", console=con)

    if args.export and all_results:
        _export_results(all_results, args.analyze_mode, args.export)


# ── export helper ─────────────────────────────────────────────────────────────

def _export_results(results, mode_label: str, fmt: str) -> None:
    """Export results to CSV or HTML under results/."""
    from backtest.analyze import AnalysisDisplay
    Path("results").mkdir(exist_ok=True)
    if fmt == "csv":
        path = f"results/{mode_label}_summary.csv"
        AnalysisDisplay.export_summary_csv(results, path)
        print(f"  Exported → {path}")
    elif fmt == "html":
        path = f"results/{mode_label}_report.html"
        AnalysisDisplay.export_html(results, path)
        print(f"  Exported → {path}")


if __name__ == "__main__":
    main()
