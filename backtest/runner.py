"""
Single-strategy synchronous backtest runner.

Pipeline:
  1. Generate all signals for each symbol (vectorized, one call per symbol).
  2. Build {(symbol, timestamp): Signal} lookup dict.
  3. Loop through all candle timestamps across all symbols in time order:
     a. record_price for correlation guard
     b. process_candle → fills (SL/TP/liquidation checks, pending entry fills)
     c. if signal at this (symbol, ts): risk evaluate → schedule entry or exit
     d. record equity after each timestamp group
  4. Close any open positions at end of data.
  5. Return BacktestResult.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
import pandas as pd

from crypto_bot.core.config import Config
from crypto_bot.core.signals.base import BaseStrategy
from crypto_bot.core.signals.models import Signal
from crypto_bot.core.risk.engine import RiskEngine
from crypto_bot.core.execution.paper import PaperExecutionEngine
from crypto_bot.core.state.store import StateStore
from crypto_bot.core.execution.models import Fill


@dataclass
class BacktestResult:
    strategy_name: str
    fills: list[Fill] = field(default_factory=list)
    equity_curve: list[tuple[datetime, float]] = field(default_factory=list)
    initial_capital: float = 10000.0

    # Computed by BacktestRunner.compute_metrics()
    total_return_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_duration_days: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_trade_duration_bars: float = 0.0
    trades_per_year: float = 0.0
    total_fees: float = 0.0
    total_slippage: float = 0.0
    composite_score: float = 0.0


class BacktestRunner:
    def __init__(self, config: Config) -> None:
        self.cfg = config

    def run(
        self,
        strategy: BaseStrategy,
        candles_by_symbol: dict[str, pd.DataFrame],
        aux_data: dict[str, Any] | None = None,
    ) -> BacktestResult:
        result = BacktestResult(
            strategy_name=strategy.name,
            initial_capital=self.cfg.backtest.initial_capital_per_strategy,
        )
        risk = RiskEngine(strategy.name, self.cfg)
        exec_engine = PaperExecutionEngine(strategy.name, self.cfg)
        store = StateStore(strategy.name)

        # Generate all signals upfront (vectorized per symbol)
        signal_map: dict[tuple[str, datetime], Signal] = {}
        for symbol, df in candles_by_symbol.items():
            sigs = strategy.generate_signals(df, aux_data)
            for sig in sigs:
                signal_map[(symbol, sig.timestamp)] = sig

        # Build sorted timeline as flat tuples — avoids iterrows() pd.Series overhead
        # Each entry: (timestamp, symbol, open, high, low, close)
        timeline: list[tuple] = []
        for symbol, df in candles_by_symbol.items():
            ts_list = df["timestamp"].tolist()
            op_list = df["open"].tolist()
            hi_list = df["high"].tolist()
            lo_list = df["low"].tolist()
            cl_list = df["close"].tolist()
            for j in range(len(ts_list)):
                timeline.append((ts_list[j], symbol, op_list[j], hi_list[j], lo_list[j], cl_list[j]))
        timeline.sort(key=lambda x: x[0])

        # Group by timestamp to process all symbols at same time together
        from itertools import groupby
        equity = self.cfg.backtest.initial_capital_per_strategy

        for ts, group in groupby(timeline, key=lambda x: x[0]):
            group_items = list(group)
            ts_fills: list[Fill] = []

            for _, symbol, op, hi, lo, cl in group_items:
                risk.record_price(symbol, cl)

                new_fills = exec_engine.process_candle(
                    symbol=symbol,
                    timestamp=ts,
                    open_=op,
                    high=hi,
                    low=lo,
                    close=cl,
                )
                ts_fills.extend(new_fills)

                # Check for signal at this candle
                sig = signal_map.get((symbol, ts))
                if sig is not None:
                    if sig.direction in ("EXIT_LONG", "EXIT_SHORT"):
                        exec_engine.schedule_exit(symbol)
                    else:
                        decision = risk.evaluate(sig)
                        if decision.approved:
                            exec_engine.schedule_entry(decision)
                            risk.open_positions[symbol] = None  # placeholder to block duplicates

            # Update equity from fills
            for fill in ts_fills:
                if fill.pnl is not None:
                    equity += fill.pnl
                    # Remove placeholder from risk engine
                    if fill.symbol in risk.open_positions and risk.open_positions[fill.symbol] is None:
                        del risk.open_positions[fill.symbol]
                store.record_fill(fill)
                result.fills.append(fill)

            # Sync risk engine open_positions with execution engine
            risk.open_positions = {k: v for k, v in exec_engine.open_positions.items() if v is not None}
            risk.update_equity(equity)
            store.record_equity(ts, equity)
            result.equity_curve.append((ts, equity))

        # Close any open positions at end of data
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
        self.compute_metrics(result)
        return result

    @staticmethod
    def compute_metrics(result: BacktestResult) -> None:
        """Compute all performance metrics from fills and equity curve. Mutates result."""
        import numpy as np

        fills = [f for f in result.fills if f.pnl is not None]
        if not fills:
            return

        equity_values = [e for _, e in result.equity_curve]
        if not equity_values:
            return

        initial = result.initial_capital
        final = equity_values[-1]
        result.total_return_pct = (final - initial) / initial

        # Daily returns from equity curve
        eq_series = pd.Series(equity_values)
        daily_returns = eq_series.pct_change().dropna()
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            result.sharpe_ratio = float(
                daily_returns.mean() / daily_returns.std() * np.sqrt(252 * 6)  # 6 x 4h per day
            )
            downside = daily_returns[daily_returns < 0]
            if len(downside) > 0 and downside.std() > 0:
                result.sortino_ratio = float(
                    daily_returns.mean() / downside.std() * np.sqrt(252 * 6)
                )

        # Max drawdown
        eq_arr = np.array(equity_values)
        peak = np.maximum.accumulate(eq_arr)
        drawdown = (peak - eq_arr) / peak
        result.max_drawdown_pct = float(drawdown.max())

        # Max drawdown duration
        underwater = drawdown > 0
        if underwater.any():
            max_dur = 0
            cur_dur = 0
            for u in underwater:
                cur_dur = cur_dur + 1 if u else 0
                max_dur = max(max_dur, cur_dur)
            # Convert bars to days (4h candles: 6 bars/day)
            result.max_drawdown_duration_days = max_dur / 6.0

        # Trade metrics
        pnls = [f.pnl for f in fills if f.pnl is not None]
        winners = [p for p in pnls if p > 0]
        losers  = [p for p in pnls if p < 0]
        result.win_rate = len(winners) / len(pnls) if pnls else 0.0
        gross_profit = sum(winners) if winners else 0.0
        gross_loss   = abs(sum(losers)) if losers else 1e-9
        result.profit_factor = gross_profit / gross_loss if gross_loss else 0.0

        result.total_fees = sum(f.fees_paid for f in fills)
        result.total_slippage = sum(f.slippage_paid for f in fills)

        # Trades per year
        timestamps = sorted([f.timestamp for f in fills])
        if len(timestamps) >= 2:
            span_years = (timestamps[-1] - timestamps[0]).days / 365.25
            result.trades_per_year = len(fills) / span_years if span_years > 0 else 0.0

        # Composite score
        min_trades = 30.0
        penalty = 1.0 if result.trades_per_year >= min_trades else result.trades_per_year / min_trades
        result.composite_score = (
            result.sharpe_ratio
            * max(0.0, 1.0 - result.max_drawdown_pct)
            * result.profit_factor
            * penalty
        )
