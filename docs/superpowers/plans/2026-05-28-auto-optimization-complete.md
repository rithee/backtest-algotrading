# Auto-Optimization Complete Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the auto-optimization pipeline so all three modes (intraday, swing, spot) are consistent: walk-forward + sensitivity in every mode, per-mode optimizer objectives, and `--trials`/`--skip-wf`/`--skip-sensitivity` CLI flags for speed/thoroughness trade-offs.

**Architecture:** Four targeted fixes: (1) rewrite intraday `_run_single` to match swing/spot; (2) refactor `optimize()` to accept an `objective_fn` callable so each mode can supply its own scoring function; (3) thread `n_trials`/`skip_wf`/`skip_sensitivity` from `main.py` → mode entrypoints → subprocess workers; (4) fix the `save_results_csv(list, path)` bug present in all three modes. No new abstractions — edit existing files only.

**Tech Stack:** Python 3.13, Optuna, dataclasses, argparse (all already installed/present).

---

## File Map

| File | Action | What changes |
|---|---|---|
| `backtest/modes/intraday.py` | **Modify** | Add wf+sensitivity to result dataclass and worker; fix `_fee_adjusted_sharpe`; OOS promotion gate; add skip/n_trials params; fix save_csv |
| `backtest/optimizer.py` | **Modify** | Add `objective_fn` param to `optimize()`; use it when provided |
| `backtest/modes/swing.py` | **Modify** | Add n_trials/skip_wf/skip_sensitivity to `run_swing` + worker; fix save_csv |
| `backtest/modes/spot_longterm.py` | **Modify** | Add n_trials/skip_wf/skip_sensitivity to `run_spot` + worker; fix save_csv; wire calmar objective |
| `main.py` | **Modify** | Add `--trials`, `--skip-wf`, `--skip-sensitivity` flags; pass to mode runners |
| `tests/test_mode_entrypoints.py` | **Modify** | Add tests for new fields, `_fee_adjusted_sharpe`, skip flags |
| `tests/test_analyzer.py` | **Modify** | Add CLI flag tests for --trials, --skip-wf |

---

## Existing types — no changes needed

```python
# backtest/runner.py
@dataclass
class BacktestResult:
    strategy_name: str
    fills: list[Fill]
    equity_curve: list[tuple[datetime, float]]   # tuples (timestamp, equity_value)
    initial_capital: float
    mode: str
    total_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    max_drawdown_duration_days: float
    win_rate: float
    profit_factor: float
    avg_trade_duration_bars: float
    trades_per_year: float
    total_fees: float
    total_slippage: float
    composite_score: float

# backtest/walk_forward.py
@dataclass
class WalkForwardResult:
    strategy_name: str
    windows: list[WalkForwardWindow]
    oos_sharpe: float
    oos_max_drawdown: float
    oos_profit_factor: float
    consistency_score: float
    is_oos_divergence: float

# backtest/sensitivity.py
@dataclass
class SensitivityResult:
    strategy_name: str
    best_score: float
    sensitivities: list[ParamSensitivity]
    is_robust: bool
```

---

## Task 1: Fix `backtest/modes/intraday.py`

Three independent bugs + one missing feature all in one file.

**Files:**
- Modify: `backtest/modes/intraday.py`
- Test: `tests/test_mode_entrypoints.py`

### Bug inventory before writing tests

| Bug | Location | Problem | Fix |
|---|---|---|---|
| `_fee_adjusted_sharpe` equity indexing | line ~51 | `equity_curve[-1]` is `(ts, val)` not a float | Use `equity_curve[-1][1]` |
| `_fee_adjusted_sharpe` empty curve | line ~51 | `equity_curve[-1]` raises IndexError when no trades | Guard `if not result.equity_curve` |
| Missing wf+sensitivity | `IntradayStrategyResult` | No `wf_result`/`sensitivity` fields | Add them |
| IS-only promotion | `_run_single` line ~91 | Checks sharpe/dd/trades but not OOS | Use `cfg.promotion_criteria` |
| `save_results_csv` wrong call | `run_intraday` line ~172 | Passes list not BacktestResult | Use `AnalysisDisplay.export_summary_csv` |
| print_strategy_report missing wf | `run_intraday` line ~168 | Prints without wf/sensitivity | Pass them |

- [ ] **Step 1: Write failing tests**

Append to `tests/test_mode_entrypoints.py`:

```python
# ── Task 1: intraday mode completeness ───────────────────────────────────────

def test_intraday_result_has_wf_and_sensitivity_fields():
    """IntradayStrategyResult must carry wf_result and sensitivity."""
    from backtest.modes.intraday import IntradayStrategyResult
    import inspect
    fields = {f.name for f in inspect.fields(IntradayStrategyResult)}
    assert "wf_result"   in fields, "wf_result missing from IntradayStrategyResult"
    assert "sensitivity" in fields, "sensitivity missing from IntradayStrategyResult"


def test_fee_adjusted_sharpe_normal():
    from backtest.modes.intraday import _fee_adjusted_sharpe
    from backtest.runner import BacktestResult
    from datetime import datetime
    r = BacktestResult(strategy_name="t", mode="intraday")
    r.sharpe_ratio = 1.5
    r.trades_per_year = 100.0
    r.total_fees = 50.0
    # equity grows from 1000 to 2000 → gain = 1000, fee_pct = 0.05
    r.equity_curve = [
        (datetime(2023, 1, 1), 1000.0),
        (datetime(2023, 6, 1), 2000.0),
    ]
    result = _fee_adjusted_sharpe(r)
    # fee_penalty = 50/1000 = 0.05 → 1.5 * (1 - 0.05) = 1.425
    assert abs(result - 1.425) < 1e-9


def test_fee_adjusted_sharpe_empty_curve():
    from backtest.modes.intraday import _fee_adjusted_sharpe
    from backtest.runner import BacktestResult
    r = BacktestResult(strategy_name="t", mode="intraday")
    r.sharpe_ratio = 1.5
    r.trades_per_year = 100.0
    r.total_fees = 50.0
    r.equity_curve = []
    # Should return -999 gracefully, not IndexError
    assert _fee_adjusted_sharpe(r) == -999.0


def test_fee_adjusted_sharpe_low_trades():
    from backtest.modes.intraday import _fee_adjusted_sharpe
    from backtest.runner import BacktestResult
    from datetime import datetime
    r = BacktestResult(strategy_name="t", mode="intraday")
    r.sharpe_ratio = 2.0
    r.trades_per_year = 10.0    # below the 50 threshold
    r.total_fees = 0.0
    r.equity_curve = [(datetime(2023, 1, 1), 1000.0), (datetime(2024, 1, 1), 2000.0)]
    assert _fee_adjusted_sharpe(r) == -999.0


def test_intraday_run_accepts_skip_wf_flag(capsys):
    """run_intraday(skip_wf=True) completes without error on empty registry."""
    from backtest.modes import intraday
    original = intraday.STRATEGY_MODULE_MAP.copy()
    intraday.STRATEGY_MODULE_MAP.clear()
    try:
        result = intraday.run_intraday(save_csv=False, skip_wf=True)
        assert result == []
    finally:
        intraday.STRATEGY_MODULE_MAP.update(original)


def test_intraday_run_accepts_skip_sensitivity_flag(capsys):
    """run_intraday(skip_sensitivity=True) completes without error on empty registry."""
    from backtest.modes import intraday
    original = intraday.STRATEGY_MODULE_MAP.copy()
    intraday.STRATEGY_MODULE_MAP.clear()
    try:
        result = intraday.run_intraday(save_csv=False, skip_sensitivity=True)
        assert result == []
    finally:
        intraday.STRATEGY_MODULE_MAP.update(original)


def test_intraday_run_accepts_n_trials_flag(capsys):
    """run_intraday(n_trials=10) completes without error on empty registry."""
    from backtest.modes import intraday
    original = intraday.STRATEGY_MODULE_MAP.copy()
    intraday.STRATEGY_MODULE_MAP.clear()
    try:
        result = intraday.run_intraday(save_csv=False, n_trials=10)
        assert result == []
    finally:
        intraday.STRATEGY_MODULE_MAP.update(original)
```

- [ ] **Step 2: Run failing tests**

```bash
python3 -m pytest tests/test_mode_entrypoints.py::test_intraday_result_has_wf_and_sensitivity_fields \
                  tests/test_mode_entrypoints.py::test_fee_adjusted_sharpe_normal \
                  tests/test_mode_entrypoints.py::test_fee_adjusted_sharpe_empty_curve \
                  tests/test_mode_entrypoints.py::test_intraday_run_accepts_skip_wf_flag -v 2>&1 | tail -20
```

Expected: all 4 FAIL — fields missing, IndexError or wrong result, parameter not accepted.

- [ ] **Step 3: Replace `backtest/modes/intraday.py`**

```python
"""
Intraday futures backtest entrypoint.

Runs all registered intraday strategies against BTC, ETH, SOL
on multi-timeframe candles (1m/5m/15m/1h), with:
- Session filter (4 UTC windows)
- Daily kill switch at -3%
- Max 5 trades/symbol/day
- 5× max leverage, 3× default
- Auto-optimization via fee-adjusted Sharpe
- Walk-forward + sensitivity analysis for OOS promotion gate

Usage:
    from backtest.modes.intraday import run_intraday
    results = run_intraday()
"""
from __future__ import annotations

import importlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

from crypto_bot.core.config import Config
from backtest.runner import BacktestRunner, BacktestResult
from backtest.optimizer import optimize
from backtest.walk_forward import run_walk_forward, WalkForwardResult
from backtest.sensitivity import analyze, SensitivityResult
from backtest.reporter import print_strategy_report, save_results_csv


# ── Strategy registry ─────────────────────────────────────────────────────────
STRATEGY_MODULE_MAP: dict[str, str] = {
    "LiquiditySweepReversalStrategy":              "crypto_bot.core.signals.strategies.intraday.lsr",
    "OpeningRangeBreakoutStrategy":                "crypto_bot.core.signals.strategies.intraday.orb_sb",
    "VWAPDeltaConfluenceStrategy":                 "crypto_bot.core.signals.strategies.intraday.mvdc",
    "FairValueGapStrategy":                        "crypto_bot.core.signals.strategies.intraday.fvg",
    "LiquidationCascadeMomentumStrategy":          "crypto_bot.core.signals.strategies.intraday.lcm",
    "MicrostructureConsolidationBreakoutStrategy": "crypto_bot.core.signals.strategies.intraday.mcb",
}

CONFIG_PATH = "config/intraday.yaml"


@dataclass
class IntradayStrategyResult:
    strategy_name: str
    result: BacktestResult
    best_params: dict
    wf_result: Optional[WalkForwardResult]
    sensitivity: Optional[SensitivityResult]
    promoted: bool
    rejection_reason: str = ""


def _fee_adjusted_sharpe(result: BacktestResult) -> float:
    """
    Intraday optimization objective: Sharpe penalised by fee drag.
    Returns -999 when trades < 50 or equity curve is empty.

    equity_curve is list[tuple[datetime, float]] — index [1] gives the equity value.
    """
    if result.trades_per_year < 50:
        return -999.0
    if not result.equity_curve:
        return -999.0
    start_equity = result.equity_curve[0][1]
    end_equity   = result.equity_curve[-1][1]
    gain = max(end_equity - start_equity, 1e-8)
    fee_penalty = result.total_fees / gain
    return result.sharpe_ratio * (1.0 - min(fee_penalty, 0.5))


def _run_single(
    strategy_cls_name: str,
    candles_by_symbol: dict[str, pd.DataFrame],
    candles_by_tf: dict[str, dict[str, pd.DataFrame]],
    config_dict: dict,
    aux_data: dict | None,
    n_trials: int | None = None,
    skip_wf: bool = False,
    skip_sensitivity: bool = False,
) -> IntradayStrategyResult:
    """Worker — runs inside a subprocess; must be picklable."""
    from crypto_bot.core.config import Config
    from backtest.runner import BacktestRunner
    from backtest.optimizer import optimize
    import importlib

    cfg = Config.model_validate(config_dict)
    module = importlib.import_module(STRATEGY_MODULE_MAP[strategy_cls_name])
    strategy_cls = getattr(module, strategy_cls_name)

    runner = BacktestRunner(cfg, mode="intraday")
    dummy = strategy_cls.__new__(strategy_cls)
    dummy.params = {}
    default_params = dummy.default_params()

    initial = runner.run(strategy_cls(default_params), candles_by_symbol, candles_by_tf, aux_data)

    best_params = default_params
    score = _fee_adjusted_sharpe(initial)
    if score < 0.5 or initial.trades_per_year < cfg.optimization.min_trades_per_year:
        print(f"[intraday/{strategy_cls_name}] fee_sharpe={score:.3f} — optimising")
        best_params = optimize(
            strategy_cls, candles_by_symbol, cfg, aux_data,
            n_trials=n_trials,
            objective_fn=_fee_adjusted_sharpe,
        )

    final = runner.run(strategy_cls(best_params), candles_by_symbol, candles_by_tf, aux_data)

    # Walk-forward
    wf: Optional[WalkForwardResult] = None
    if not skip_wf:
        from backtest.walk_forward import run_walk_forward
        wf = run_walk_forward(strategy_cls, candles_by_symbol, cfg, aux_data, n_trials_per_window=20)

    # Sensitivity
    sens: Optional[SensitivityResult] = None
    if not skip_sensitivity:
        from backtest.sensitivity import analyze
        sens = analyze(strategy_cls, best_params, candles_by_symbol, cfg, aux_data)

    # OOS promotion gate (falls back to IS-only when skip_wf=True)
    criteria = cfg.promotion_criteria
    if wf is not None:
        promoted = (
            wf.oos_sharpe >= criteria.min_sharpe_oos
            and wf.oos_max_drawdown <= criteria.max_drawdown_pct
            and wf.oos_profit_factor >= criteria.min_profit_factor
            and final.trades_per_year >= criteria.min_trades_per_year
            and (sens is None or sens.is_robust)
        )
        reason = "" if promoted else (
            f"oos_sharpe={wf.oos_sharpe:.2f}, dd={wf.oos_max_drawdown:.1%}, "
            f"pf={wf.oos_profit_factor:.2f}"
        )
    else:
        promoted = (
            final.sharpe_ratio >= criteria.min_sharpe_oos
            and final.max_drawdown_pct <= criteria.max_drawdown_pct
            and final.trades_per_year >= criteria.min_trades_per_year
            and (sens is None or sens.is_robust)
        )
        reason = "" if promoted else (
            f"sharpe={final.sharpe_ratio:.2f}, dd={final.max_drawdown_pct:.1%}, "
            f"trades/yr={final.trades_per_year:.0f}"
        )

    return IntradayStrategyResult(
        strategy_cls_name, final, best_params, wf, sens, promoted, reason
    )


def run_intraday(
    config_path: str = CONFIG_PATH,
    max_workers: int = 4,
    save_csv: bool = True,
    n_trials: int | None = None,
    skip_wf: bool = False,
    skip_sensitivity: bool = False,
) -> list[IntradayStrategyResult]:
    """
    Run all registered intraday strategies.

    Args:
        config_path:       Path to intraday YAML config
        max_workers:       Parallel subprocess workers
        save_csv:          Export summary CSV to results/intraday_summary.csv
        n_trials:          Override Optuna trial count (None = use config value)
        skip_wf:           Skip walk-forward validation (faster, no OOS gate)
        skip_sensitivity:  Skip sensitivity analysis

    Returns:
        List of IntradayStrategyResult, one per strategy.
    """
    from crypto_bot.core.data.loader import load_mtf_candles, load_aux_data

    cfg = Config.from_yaml(config_path)
    config_dict = cfg.model_dump()

    if not STRATEGY_MODULE_MAP:
        print("[intraday] No strategies registered yet — add entries to STRATEGY_MODULE_MAP.")
        return []

    print(f"[intraday] Loading candles for {cfg.backtest.symbols} …")
    candles_by_tf = load_mtf_candles(cfg)

    primary_tf = cfg.backtest.active_primary_tf
    candles_by_symbol: dict[str, pd.DataFrame] = {
        sym: candles_by_tf[sym][primary_tf]
        for sym in candles_by_tf
        if primary_tf in candles_by_tf[sym]
    }

    print("[intraday] Loading auxiliary data …")
    aux_data = load_aux_data(cfg, mode="intraday")

    results: list[IntradayStrategyResult] = []

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _run_single,
                name, candles_by_symbol, candles_by_tf, config_dict, aux_data,
                n_trials, skip_wf, skip_sensitivity,
            ): name
            for name in STRATEGY_MODULE_MAP
        }
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                r = fut.result()
                results.append(r)
                status = "PROMOTE" if r.promoted else "REJECT"
                fee_sh = _fee_adjusted_sharpe(r.result)
                print(
                    f"[intraday/{name}] {status}  "
                    f"fee_sharpe={fee_sh:.3f}  dd={r.result.max_drawdown_pct:.1%}"
                )
                print_strategy_report(r.result, r.wf_result, r.sensitivity)
            except Exception as exc:
                print(f"[intraday/{name}] ERROR: {exc}")

    if save_csv and results:
        from backtest.analyze import AnalysisDisplay
        AnalysisDisplay.export_summary_csv(
            [r.result for r in results], "results/intraday_summary.csv"
        )

    promoted = [r for r in results if r.promoted]
    print(f"\n[intraday] {len(promoted)}/{len(results)} strategies promoted.")
    return results
```

- [ ] **Step 4: Run the tests**

```bash
python3 -m pytest tests/test_mode_entrypoints.py::test_intraday_result_has_wf_and_sensitivity_fields \
                  tests/test_mode_entrypoints.py::test_fee_adjusted_sharpe_normal \
                  tests/test_mode_entrypoints.py::test_fee_adjusted_sharpe_empty_curve \
                  tests/test_mode_entrypoints.py::test_fee_adjusted_sharpe_low_trades \
                  tests/test_mode_entrypoints.py::test_intraday_run_accepts_skip_wf_flag \
                  tests/test_mode_entrypoints.py::test_intraday_run_accepts_skip_sensitivity_flag \
                  tests/test_mode_entrypoints.py::test_intraday_run_accepts_n_trials_flag -v 2>&1 | tail -15
```

Expected: all 7 PASS.

- [ ] **Step 5: Full suite smoke check**

```bash
python3 -m pytest --tb=short -q 2>&1 | tail -5
```

Expected: 344+ passing, 0 failures.

- [ ] **Step 6: Commit**

```bash
git add backtest/modes/intraday.py tests/test_mode_entrypoints.py
git commit -m "fix: complete intraday auto-opt pipeline — wf+sensitivity, OOS gate, fee_sharpe bug (Plan 6 Task 1)"
```

---

## Task 2: Refactor `optimize()` to accept per-mode `objective_fn`

**Files:**
- Modify: `backtest/optimizer.py`
- Modify: `backtest/modes/spot_longterm.py`
- Modify: `backtest/modes/swing.py`
- Test: `tests/test_mode_entrypoints.py`

### Context

`optimize()` currently always uses `result.composite_score`. Each mode has its own scoring philosophy:
- **Intraday**: fee-adjusted Sharpe (already fixed in Task 1, now needs to flow into optimizer)
- **Swing**: composite score (existing behaviour — no change)
- **Spot**: Calmar ratio (already defined as `_calmar_ratio()` in spot_longterm.py)

The fix adds `objective_fn: Callable[[BacktestResult], float] | None = None` to `optimize()`. When `None`, falls back to `result.composite_score`.

- [ ] **Step 1: Write failing test**

Append to `tests/test_mode_entrypoints.py`:

```python
# ── Task 2: optimizer custom objective ───────────────────────────────────────

def test_optimizer_accepts_objective_fn():
    """optimize() signature must accept objective_fn keyword argument."""
    import inspect
    from backtest.optimizer import optimize
    sig = inspect.signature(optimize)
    assert "objective_fn" in sig.parameters, "objective_fn param missing from optimize()"


def test_optimizer_objective_fn_is_none_by_default():
    """objective_fn defaults to None (backward compatible)."""
    import inspect
    from backtest.optimizer import optimize
    sig = inspect.signature(optimize)
    default = sig.parameters["objective_fn"].default
    assert default is None


def test_spot_mode_wires_calmar_to_optimizer():
    """spot_longterm._run_single passes objective_fn=_calmar_ratio to optimize()."""
    import inspect, ast, textwrap
    import backtest.modes.spot_longterm as mod
    src = inspect.getsource(mod._run_single)
    assert "_calmar_ratio" in src, (
        "_calmar_ratio not referenced in spot_longterm._run_single — "
        "calmar objective not wired"
    )
    assert "objective_fn" in src, (
        "objective_fn not passed in spot_longterm._run_single"
    )
```

- [ ] **Step 2: Run failing tests**

```bash
python3 -m pytest tests/test_mode_entrypoints.py::test_optimizer_accepts_objective_fn \
                  tests/test_mode_entrypoints.py::test_optimizer_objective_fn_is_none_by_default \
                  tests/test_mode_entrypoints.py::test_spot_mode_wires_calmar_to_optimizer -v 2>&1 | tail -15
```

Expected: all 3 FAIL.

- [ ] **Step 3: Update `backtest/optimizer.py`**

Add `objective_fn` parameter. The only change is in the `objective` inner function and the `optimize` signature.

Replace the existing `optimize()` function body with:

```python
from __future__ import annotations
import os
from pathlib import Path
from typing import Callable, Type
import pandas as pd

from crypto_bot.core.config import Config
from crypto_bot.core.signals.base import BaseStrategy
from backtest.runner import BacktestRunner, BacktestResult


def optimize(
    strategy_cls: Type[BaseStrategy],
    candles_by_symbol: dict[str, pd.DataFrame],
    config: Config,
    aux_data: dict | None = None,
    n_trials: int | None = None,
    objective_fn: Callable[[BacktestResult], float] | None = None,
) -> dict:
    """
    Run Optuna optimization for strategy_cls. Returns best params dict.
    Resumes from existing study if storage file exists.

    Args:
        strategy_cls:  Strategy class to optimise.
        candles_by_symbol: Training candles keyed by symbol.
        config:        Config (provides trial count, storage path, min trades).
        aux_data:      Optional auxiliary data (on-chain, macro, etc.).
        n_trials:      Override config trial count.
        objective_fn:  Scoring function (BacktestResult) -> float.
                       When None, uses result.composite_score.
    """
    try:
        import optuna
    except ImportError:
        raise ImportError("optuna is required: pip install optuna")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    n = n_trials if n_trials is not None else config.optimization.trials
    storage_template = config.optimization.study_storage
    storage_path = storage_template.format(strategy_name=strategy_cls.__name__)
    Path(storage_path).parent.mkdir(parents=True, exist_ok=True)
    storage_url = f"sqlite:///{storage_path}"

    runner = BacktestRunner(config)

    def _score(result: BacktestResult) -> float:
        if objective_fn is not None:
            return objective_fn(result)
        return result.composite_score

    def objective(trial: "optuna.Trial") -> float:
        params = _sample_params(trial, strategy_cls)
        strategy = strategy_cls(params)
        result = runner.run(strategy, candles_by_symbol, aux_data)

        if result.trades_per_year < config.optimization.min_trades_per_year:
            return -999.0

        return _score(result)

    study = optuna.create_study(
        study_name=strategy_cls.__name__,
        storage=storage_url,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=20),
    )
    study.optimize(objective, n_trials=n, n_jobs=1, show_progress_bar=True)
    return study.best_params


def _sample_params(trial: "optuna.Trial", strategy_cls: Type[BaseStrategy]) -> dict:
    """Sample params from param_space using Optuna trial suggestions."""
    dummy = strategy_cls.__new__(strategy_cls)
    dummy.params = {}
    space = dummy.param_space

    params: dict = {}
    for key, spec in space.items():
        if len(spec) == 3 and spec[2] == "int":
            params[key] = trial.suggest_int(key, int(spec[0]), int(spec[1]))
        elif len(spec) == 2 and isinstance(spec[0], list):
            params[key] = trial.suggest_categorical(key, spec[0])
        else:
            params[key] = trial.suggest_float(key, float(spec[0]), float(spec[1]))
    return params
```

- [ ] **Step 4: Wire `_calmar_ratio` in `backtest/modes/spot_longterm.py`**

Find the `optimize(...)` call inside `_run_single` in `spot_longterm.py`. It currently reads:

```python
best_params = optimize(strategy_cls, candles_by_symbol, cfg, aux_data)
```

Replace that one line with:

```python
best_params = optimize(
    strategy_cls, candles_by_symbol, cfg, aux_data,
    objective_fn=_calmar_ratio,
)
```

`_calmar_ratio` is already defined in the same file.

- [ ] **Step 5: Fix `save_results_csv` bug in `swing.py` and `spot_longterm.py`**

In `backtest/modes/swing.py`, find:
```python
    if save_csv and results:
        save_results_csv([r.result for r in results], "results/swing_results.csv")
```
Replace with:
```python
    if save_csv and results:
        from backtest.analyze import AnalysisDisplay
        AnalysisDisplay.export_summary_csv(
            [r.result for r in results], "results/swing_summary.csv"
        )
```

In `backtest/modes/spot_longterm.py`, find:
```python
    if save_csv and results:
        save_results_csv([r.result for r in results], "results/spot_results.csv")
```
Replace with:
```python
    if save_csv and results:
        from backtest.analyze import AnalysisDisplay
        AnalysisDisplay.export_summary_csv(
            [r.result for r in results], "results/spot_summary.csv"
        )
```

- [ ] **Step 6: Run all new tests**

```bash
python3 -m pytest tests/test_mode_entrypoints.py::test_optimizer_accepts_objective_fn \
                  tests/test_mode_entrypoints.py::test_optimizer_objective_fn_is_none_by_default \
                  tests/test_mode_entrypoints.py::test_spot_mode_wires_calmar_to_optimizer -v 2>&1 | tail -12
```

Expected: all 3 PASS.

- [ ] **Step 7: Full suite smoke check**

```bash
python3 -m pytest --tb=short -q 2>&1 | tail -5
```

Expected: 344+ passing, 0 failures.

- [ ] **Step 8: Commit**

```bash
git add backtest/optimizer.py backtest/modes/spot_longterm.py backtest/modes/swing.py \
        tests/test_mode_entrypoints.py
git commit -m "feat: add objective_fn to optimize(); wire per-mode scoring; fix save_csv bug (Plan 6 Task 2)"
```

---

## Task 3: Add `--trials`, `--skip-wf`, `--skip-sensitivity` to CLI

**Files:**
- Modify: `main.py`
- Modify: `backtest/modes/swing.py`
- Modify: `backtest/modes/spot_longterm.py`
- Test: `tests/test_analyzer.py`

The flags thread from `main.py` → each `run_*()` call → subprocess worker.

- [ ] **Step 1: Write failing CLI tests**

Append to `tests/test_analyzer.py`:

```python
# ── Plan 6 Task 3: optimizer CLI flags ────────────────────────────────────────

def test_main_has_trials_flag():
    import subprocess, sys
    r = subprocess.run(
        [sys.executable, "main.py", "--help"],
        capture_output=True, text=True,
        cwd="/home/rithee/Desktop/backtest_test",
    )
    assert "--trials" in r.stdout

def test_main_has_skip_wf_flag():
    import subprocess, sys
    r = subprocess.run(
        [sys.executable, "main.py", "--help"],
        capture_output=True, text=True,
        cwd="/home/rithee/Desktop/backtest_test",
    )
    assert "--skip-wf" in r.stdout

def test_main_has_skip_sensitivity_flag():
    import subprocess, sys
    r = subprocess.run(
        [sys.executable, "main.py", "--help"],
        capture_output=True, text=True,
        cwd="/home/rithee/Desktop/backtest_test",
    )
    assert "--skip-sensitivity" in r.stdout
```

- [ ] **Step 2: Run failing tests**

```bash
python3 -m pytest tests/test_analyzer.py::test_main_has_trials_flag \
                  tests/test_analyzer.py::test_main_has_skip_wf_flag \
                  tests/test_analyzer.py::test_main_has_skip_sensitivity_flag -v 2>&1 | tail -10
```

Expected: all 3 FAIL.

- [ ] **Step 3: Add `n_trials`/`skip_wf`/`skip_sensitivity` params to `run_swing()` in `swing.py`**

Replace `run_swing` signature from:
```python
def run_swing(
    config_path: str = CONFIG_PATH,
    max_workers: int = 4,
    save_csv: bool = True,
) -> list[SwingStrategyResult]:
```
to:
```python
def run_swing(
    config_path: str = CONFIG_PATH,
    max_workers: int = 4,
    save_csv: bool = True,
    n_trials: int | None = None,
    skip_wf: bool = False,
    skip_sensitivity: bool = False,
) -> list[SwingStrategyResult]:
```

Update the `_run_single` signature in `swing.py` from:
```python
def _run_single(
    strategy_cls_name: str,
    candles_by_symbol: dict[str, pd.DataFrame],
    candles_by_tf: dict[str, dict[str, pd.DataFrame]],
    config_dict: dict,
    aux_data: dict | None,
) -> SwingStrategyResult:
```
to:
```python
def _run_single(
    strategy_cls_name: str,
    candles_by_symbol: dict[str, pd.DataFrame],
    candles_by_tf: dict[str, dict[str, pd.DataFrame]],
    config_dict: dict,
    aux_data: dict | None,
    n_trials: int | None = None,
    skip_wf: bool = False,
    skip_sensitivity: bool = False,
) -> SwingStrategyResult:
```

Inside `swing.py`'s `_run_single`, update the `optimize()` call:
```python
        best_params = optimize(strategy_cls, candles_by_symbol, cfg, aux_data)
```
to:
```python
        best_params = optimize(strategy_cls, candles_by_symbol, cfg, aux_data, n_trials=n_trials)
```

Replace the unconditional WF call:
```python
    wf = run_walk_forward(strategy_cls, candles_by_symbol, cfg, aux_data, n_trials_per_window=30)
    sens = analyze(strategy_cls, best_params, candles_by_symbol, cfg, aux_data)
```
with:
```python
    wf = None
    if not skip_wf:
        wf = run_walk_forward(strategy_cls, candles_by_symbol, cfg, aux_data, n_trials_per_window=30)
    sens = None
    if not skip_sensitivity:
        sens = analyze(strategy_cls, best_params, candles_by_symbol, cfg, aux_data)
```

Update `pool.submit` in `run_swing` to pass the new args:
```python
            pool.submit(
                _run_single,
                name, candles_by_symbol, candles_by_tf, config_dict, aux_data,
                n_trials, skip_wf, skip_sensitivity,
            ): name
```

- [ ] **Step 4: Add same params to `run_spot()` in `spot_longterm.py`**

Same pattern as swing. Replace `run_spot` signature:
```python
def run_spot(
    config_path: str = CONFIG_PATH,
    max_workers: int = 4,
    save_csv: bool = True,
) -> list[SpotStrategyResult]:
```
to:
```python
def run_spot(
    config_path: str = CONFIG_PATH,
    max_workers: int = 4,
    save_csv: bool = True,
    n_trials: int | None = None,
    skip_wf: bool = False,
    skip_sensitivity: bool = False,
) -> list[SpotStrategyResult]:
```

Update `_run_single` in `spot_longterm.py` signature identically (add the three params after `aux_data`).

Inside `spot_longterm.py`'s `_run_single`, update optimize call to:
```python
        best_params = optimize(
            strategy_cls, candles_by_symbol, cfg, aux_data,
            n_trials=n_trials,
            objective_fn=_calmar_ratio,
        )
```

Replace unconditional WF + sensitivity calls:
```python
    wf = run_walk_forward(strategy_cls, candles_by_symbol, cfg, aux_data, n_trials_per_window=20)
    sens = analyze(strategy_cls, best_params, candles_by_symbol, cfg, aux_data)
```
with:
```python
    wf = None
    if not skip_wf:
        wf = run_walk_forward(strategy_cls, candles_by_symbol, cfg, aux_data, n_trials_per_window=20)
    sens = None
    if not skip_sensitivity:
        sens = analyze(strategy_cls, best_params, candles_by_symbol, cfg, aux_data)
```

Update `pool.submit` in `run_spot`:
```python
            pool.submit(
                _run_single,
                name, candles_by_symbol, candles_by_tf, config_dict, aux_data,
                n_trials, skip_wf, skip_sensitivity,
            ): name
```

- [ ] **Step 5: Add flags to `main.py`**

Add three new arguments to the `argparse` parser in `main()`. Insert after the existing `--no-walk-forward` argument:

```python
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
```

Update `_run_intraday_mode` to pass the new flags:
```python
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
```

Update `_run_spot_mode`:
```python
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
```

Update `_run_backtest` (swing) to pass `n_trials` and skip flags through:
```python
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
```

Note: `orchestrator.py` is the legacy swing runner; it doesn't use the new params. The `--mode swing` (i.e. `backtest/modes/swing.py`) path is through `_run_analyze`. Update `_run_analyze` to pass the flags:

In `_run_analyze`, update the swing sub-block:
```python
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
```

Update the intraday sub-block in `_run_analyze`:
```python
        elif mode == "intraday":
            from backtest.modes.intraday import run_intraday
            raw = run_intraday(
                max_workers=1 if args.no_parallel else 4,
                n_trials=args.trials,
                skip_wf=args.skip_wf,
                skip_sensitivity=args.skip_sensitivity,
            )
            results = [r.result for r in raw]
```

Update the spot sub-block in `_run_analyze`:
```python
        else:  # spot
            from backtest.modes.spot_longterm import run_spot
            raw = run_spot(
                max_workers=1 if args.no_parallel else 4,
                n_trials=args.trials,
                skip_wf=args.skip_wf,
                skip_sensitivity=args.skip_sensitivity,
            )
            results = [r.result for r in raw]
```

- [ ] **Step 6: Run CLI tests**

```bash
python3 -m pytest tests/test_analyzer.py::test_main_has_trials_flag \
                  tests/test_analyzer.py::test_main_has_skip_wf_flag \
                  tests/test_analyzer.py::test_main_has_skip_sensitivity_flag -v 2>&1 | tail -10
```

Expected: all 3 PASS.

- [ ] **Step 7: Full suite smoke check**

```bash
python3 -m pytest --tb=short -q 2>&1 | tail -5
```

Expected: 344+ passing, 0 failures.

- [ ] **Step 8: Commit**

```bash
git add main.py backtest/modes/swing.py backtest/modes/spot_longterm.py tests/test_analyzer.py
git commit -m "feat: add --trials, --skip-wf, --skip-sensitivity CLI flags; thread through all mode runners (Plan 6 Task 3)"
```

---

## Task 4: Integration tests + final commit

**Files:**
- Modify: `tests/test_mode_entrypoints.py` (append)

- [ ] **Step 1: Add swing/spot skip-wf flag tests and save_csv fix verification**

Append to `tests/test_mode_entrypoints.py`:

```python
# ── Task 4: swing/spot skip flags ────────────────────────────────────────────

def test_swing_run_accepts_skip_wf_flag(capsys):
    from backtest.modes import swing
    original = swing.STRATEGY_MODULE_MAP.copy()
    swing.STRATEGY_MODULE_MAP.clear()
    try:
        result = swing.run_swing(save_csv=False, skip_wf=True)
        assert result == []
    finally:
        swing.STRATEGY_MODULE_MAP.update(original)


def test_swing_run_accepts_n_trials_flag(capsys):
    from backtest.modes import swing
    original = swing.STRATEGY_MODULE_MAP.copy()
    swing.STRATEGY_MODULE_MAP.clear()
    try:
        result = swing.run_swing(save_csv=False, n_trials=5)
        assert result == []
    finally:
        swing.STRATEGY_MODULE_MAP.update(original)


def test_spot_run_accepts_skip_wf_flag(capsys):
    from backtest.modes import spot_longterm
    original = spot_longterm.STRATEGY_MODULE_MAP.copy()
    spot_longterm.STRATEGY_MODULE_MAP.clear()
    try:
        result = spot_longterm.run_spot(save_csv=False, skip_wf=True)
        assert result == []
    finally:
        spot_longterm.STRATEGY_MODULE_MAP.update(original)


def test_spot_run_accepts_n_trials_flag(capsys):
    from backtest.modes import spot_longterm
    original = spot_longterm.STRATEGY_MODULE_MAP.copy()
    spot_longterm.STRATEGY_MODULE_MAP.clear()
    try:
        result = spot_longterm.run_spot(save_csv=False, n_trials=5)
        assert result == []
    finally:
        spot_longterm.STRATEGY_MODULE_MAP.update(original)


def test_swing_save_csv_uses_export_summary(tmp_path, monkeypatch):
    """run_swing(save_csv=True) calls AnalysisDisplay.export_summary_csv, not save_results_csv."""
    from backtest.modes import swing
    from backtest.analyze import AnalysisDisplay
    calls = []
    monkeypatch.setattr(AnalysisDisplay, "export_summary_csv", lambda results, path: calls.append(path))
    original = swing.STRATEGY_MODULE_MAP.copy()
    swing.STRATEGY_MODULE_MAP.clear()
    swing.STRATEGY_MODULE_MAP["_dummy"] = "_not_a_real_module"
    # Empty map (clear the dummy too) → no strategies → early return, no save_csv call
    swing.STRATEGY_MODULE_MAP.clear()
    swing.run_swing(save_csv=True)
    # No call expected because no results
    assert calls == []
    swing.STRATEGY_MODULE_MAP.update(original)
```

- [ ] **Step 2: Run all Task 4 tests**

```bash
python3 -m pytest tests/test_mode_entrypoints.py::test_swing_run_accepts_skip_wf_flag \
                  tests/test_mode_entrypoints.py::test_swing_run_accepts_n_trials_flag \
                  tests/test_mode_entrypoints.py::test_spot_run_accepts_skip_wf_flag \
                  tests/test_mode_entrypoints.py::test_spot_run_accepts_n_trials_flag -v 2>&1 | tail -12
```

Expected: all 4 PASS.

- [ ] **Step 3: Run full final test suite**

```bash
python3 -m pytest --tb=short -q 2>&1 | tail -5
```

Expected: 355+ tests, 0 failures.

- [ ] **Step 4: Final commit**

```bash
git add tests/test_mode_entrypoints.py
git commit -m "feat: add integration tests for swing/spot skip flags (Plan 6 Task 4)

Plan 6 complete — auto-optimization pipeline:
  backtest/modes/intraday.py    : complete wf+sensitivity, OOS gate, _fee_adjusted_sharpe fix
  backtest/optimizer.py         : objective_fn param for per-mode scoring
  backtest/modes/swing.py       : n_trials/skip_wf/skip_sensitivity; fix save_csv
  backtest/modes/spot_longterm.py: calmar objective wired; n_trials/skip_wf/skip_sensitivity; fix save_csv
  main.py                       : --trials, --skip-wf, --skip-sensitivity flags

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Self-Review

**1. Spec coverage:**
- ✅ Intraday mode gets walk-forward + sensitivity → Task 1
- ✅ Intraday OOS promotion gate (not IS-only) → Task 1
- ✅ `_fee_adjusted_sharpe` equity indexing bug fixed → Task 1
- ✅ `save_results_csv(list, path)` bug fixed in all 3 modes → Tasks 1 + 2
- ✅ `optimize()` accepts `objective_fn` → Task 2
- ✅ Spot mode wires `_calmar_ratio` to optimizer → Task 2
- ✅ `--trials`, `--skip-wf`, `--skip-sensitivity` in main.py → Task 3
- ✅ All 3 mode `run_*()` functions accept new params → Tasks 3+4
- ✅ Tests for all changes → Tasks 1–4

**2. Placeholder scan:** All code blocks are complete. No "fill in" or "TBD" items.

**3. Type consistency:**
- `_fee_adjusted_sharpe(result: BacktestResult) -> float` defined in intraday.py, passed as `objective_fn` to `optimize()` ✅
- `_calmar_ratio(result: BacktestResult) -> float` already exists in spot_longterm.py, same signature ✅
- `optimize(..., objective_fn: Callable[[BacktestResult], float] | None = None)` — `Callable` imported in optimizer.py ✅
- `IntradayStrategyResult.wf_result: Optional[WalkForwardResult]` — `WalkForwardResult` imported in intraday.py ✅
- All `run_*(skip_wf=False, ...)` default values are `bool` matching `store_true` argparse action ✅
