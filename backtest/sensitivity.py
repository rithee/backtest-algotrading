"""
Parameter sensitivity analysis.
After optimisation, perturbs each param +-10% and +-20%, measures score change.
Flags strategy as BRITTLE if any perturbation > 20% causes score drop > 30%.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Type
import numpy as np
import pandas as pd

from crypto_bot.core.config import Config
from crypto_bot.core.signals.base import BaseStrategy
from backtest.runner import BacktestRunner


@dataclass
class ParamSensitivity:
    param: str
    best_value: float
    perturb_pct: float
    perturbed_value: float
    score_change_pct: float
    is_brittle: bool  # True if score_change_pct < -30 for a perturbation > 20%


@dataclass
class SensitivityResult:
    strategy_name: str
    best_score: float
    sensitivities: list[ParamSensitivity]
    is_robust: bool  # True = no brittle params found


def analyze(
    strategy_cls: Type[BaseStrategy],
    best_params: dict,
    candles_by_symbol: dict[str, pd.DataFrame],
    config: Config,
    aux_data: dict | None = None,
) -> SensitivityResult:
    runner = BacktestRunner(config, mode=config.mode)
    base_strategy = strategy_cls(best_params)
    base_result = runner.run(base_strategy, candles_by_symbol, aux_data)
    base_score = base_result.composite_score

    dummy = strategy_cls.__new__(strategy_cls)
    dummy.params = {}
    space = dummy.param_space

    sensitivities: list[ParamSensitivity] = []

    for param, spec in space.items():
        best_val = best_params.get(param)
        if best_val is None:
            continue
        is_int = len(spec) == 3 and spec[2] == "int"
        low, high = float(spec[0]), float(spec[1])

        for perturb_pct in [-0.20, -0.10, 0.10, 0.20]:
            perturbed = best_val * (1.0 + perturb_pct)
            if is_int:
                perturbed = int(round(perturbed))
                perturbed = max(int(low), min(int(high), perturbed))
            else:
                perturbed = float(np.clip(perturbed, low, high))

            new_params = dict(best_params)
            new_params[param] = perturbed
            result = runner.run(strategy_cls(new_params), candles_by_symbol, aux_data)
            score = result.composite_score

            if base_score != 0:
                change_pct = (score - base_score) / abs(base_score) * 100.0
            else:
                change_pct = 0.0

            brittle = abs(perturb_pct) >= 0.20 and change_pct < -30.0
            sensitivities.append(ParamSensitivity(
                param=param,
                best_value=float(best_val),
                perturb_pct=perturb_pct * 100,
                perturbed_value=float(perturbed),
                score_change_pct=change_pct,
                is_brittle=brittle,
            ))

    is_robust = not any(s.is_brittle for s in sensitivities)
    return SensitivityResult(
        strategy_name=strategy_cls.__name__,
        best_score=base_score,
        sensitivities=sensitivities,
        is_robust=is_robust,
    )
