import numpy as np
import pandas as pd
import pytest

from data_source import generate_sample_prices
from signal_engine import StrategyConfig, backtest, generate_signals


def test_indicators_and_signals_are_created():
    prices = generate_sample_prices(days=500, seed=1)
    config = StrategyConfig()
    result = generate_signals(prices, config)
    expected = {
        "SMA_Fast", "SMA_Slow", "MACD", "MACD_Signal", "RSI",
        "BB_Upper", "BB_Lower", "Score", "Signal", "Position",
    }
    assert expected.issubset(result.columns)
    assert result["Score"].between(-4, 4).all()
    assert set(result["Position"].unique()).issubset({0, 1})


def test_no_lookahead_when_last_price_changes():
    prices = generate_sample_prices(days=400, seed=5)
    config = StrategyConfig()
    baseline = generate_signals(prices, config)
    changed = prices.copy()
    changed.iloc[-1, changed.columns.get_loc("Close")] *= 1.8
    altered = generate_signals(changed, config)
    pd.testing.assert_frame_equal(
        baseline.iloc[:-1][["Score", "Signal", "Position"]],
        altered.iloc[:-1][["Score", "Signal", "Position"]],
    )


def test_backtest_uses_previous_days_position():
    prices = generate_sample_prices(days=350, seed=8)
    config = StrategyConfig(score_threshold=1, fee_bps=0)
    signals = generate_signals(prices, config)
    tested, metrics, _ = backtest(signals, config)
    expected = tested["Position"].shift(1).fillna(0).astype(int)
    pd.testing.assert_series_equal(tested["HeldPosition"], expected, check_names=False)
    assert np.isfinite(metrics["total_return"])
    assert metrics["max_drawdown"] <= 0


def test_invalid_period_relationship_is_rejected():
    with pytest.raises(ValueError, match="短期SMA"):
        StrategyConfig(sma_fast=50, sma_slow=20)


def test_too_little_history_is_rejected():
    prices = generate_sample_prices(days=80, seed=2).tail(30)
    with pytest.raises(ValueError, match="データが不足"):
        generate_signals(prices, StrategyConfig())

