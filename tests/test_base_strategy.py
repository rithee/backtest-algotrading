"""Tests for BaseStrategy extensions: mode property and generate_signals_mtf."""
import pytest


def test_base_strategy_default_mode_is_swing():
    from crypto_bot.core.signals.strategies.ema_ribbon import EMARibbonStrategy
    s = EMARibbonStrategy({})
    assert s.mode == "swing"


def test_base_strategy_generate_signals_mtf_falls_back_to_single_tf(flat_candles):
    """generate_signals_mtf with one TF in dict should call generate_signals."""
    from crypto_bot.core.signals.strategies.ema_ribbon import EMARibbonStrategy
    strategy = EMARibbonStrategy(EMARibbonStrategy({}).default_params())
    tf_dict = {"4h": flat_candles}
    result = strategy.generate_signals_mtf(tf_dict, aux_data=None)
    assert isinstance(result, list)


def test_base_strategy_generate_signals_mtf_uses_first_tf(flat_candles):
    """When multiple TFs passed, first is used for default fallback."""
    from crypto_bot.core.signals.strategies.ema_ribbon import EMARibbonStrategy
    strategy = EMARibbonStrategy(EMARibbonStrategy({}).default_params())
    tf_dict = {"4h": flat_candles, "1d": flat_candles}
    result = strategy.generate_signals_mtf(tf_dict, aux_data=None)
    assert isinstance(result, list)
