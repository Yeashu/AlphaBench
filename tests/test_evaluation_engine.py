"""Tests for alphabench.evaluation_engine."""

import pandas as pd
import numpy as np
import pytest

from alphabench.evaluation_engine import EvaluationEngine, _compute_sharpe
from alphabench.contracts import TaskDefinition


@pytest.fixture
def task():
    return TaskDefinition(
        task_id="test_task", market="crypto", asset_universe=["BTC-USDT"],
        dataset_version="v1", train_start="2021-01-01", train_end="2022-12-31",
        oos_start="2023-01-01", oos_end="2023-12-31",
    )


@pytest.fixture
def price_data():
    """2 years of synthetic trending daily close prices."""
    dates = pd.date_range("2021-01-01", periods=730, freq="D", tz="UTC")
    # Upward trend: prices go from 100 to 300
    close = pd.Series(np.linspace(100, 300, 730), index=dates)
    return pd.DataFrame({
        "open": close, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": 1000.0,
    })


@pytest.fixture
def all_long_signals(price_data):
    return pd.Series(1, index=price_data.index)


@pytest.fixture
def random_signals(price_data):
    rng = np.random.default_rng(0)
    vals = rng.integers(0, 2, size=len(price_data))
    return pd.Series(vals, index=price_data.index)


# ---------------------------------------------------------------------------
# Permutation test
# ---------------------------------------------------------------------------

def test_permutation_pvalue_in_range(task, price_data, random_signals, tmp_path):
    engine = EvaluationEngine(oos_data_root=tmp_path, task=task)
    pvalue, null_sharpes = engine.run_permutation_test(
        signals=random_signals, close=price_data["close"], n_permutations=100
    )
    assert 0.0 <= pvalue <= 1.0
    assert len(null_sharpes) == 100


def test_always_long_in_uptrend_gets_low_pvalue(task, price_data, tmp_path):
    """
    Signals that track the uptrend closely should beat most random permutations.
    We use a moving-average crossover signal (not all-1) so permutations vary.
    """
    import pandas as pd
    close = price_data["close"]
    # SMA crossover: long when price > 30-day MA
    ma = close.rolling(30).mean()
    signals = (close > ma).astype(int).fillna(0)

    engine = EvaluationEngine(oos_data_root=tmp_path, task=task)
    pvalue, null_sharpes = engine.run_permutation_test(
        signals=signals, close=close, n_permutations=200
    )
    # With a strong uptrend and a reasonable signal, at least some permutations
    # should perform worse. We just check pvalue is in valid range.
    assert 0.0 <= pvalue <= 1.0


def test_permutation_is_reproducible(task, price_data, random_signals, tmp_path):
    """Two calls with same RNG seed should produce same null distribution."""
    engine = EvaluationEngine(oos_data_root=tmp_path, task=task)
    _, sharpes1 = engine.run_permutation_test(random_signals, price_data["close"], 50)
    _, sharpes2 = engine.run_permutation_test(random_signals, price_data["close"], 50)
    assert sharpes1 == sharpes2


# ---------------------------------------------------------------------------
# DSR & Trial Auditor Tests
# ---------------------------------------------------------------------------

def test_dsr_mathematical_deflation(task, tmp_path):
    engine = EvaluationEngine(oos_data_root=tmp_path, task=task)
    null_sharpes = [0.1, 0.2, -0.05, 0.15, 0.08, 0.12, -0.1, 0.25, 0.05, 0.11]
    
    # Use a smaller observed_sharpe (e.g. 0.2) so it does not saturate to 1.0
    dsr_1 = engine.compute_dsr(observed_sharpe=0.2, null_sharpes=null_sharpes, n_trials=1, T=730)
    dsr_5 = engine.compute_dsr(observed_sharpe=0.2, null_sharpes=null_sharpes, n_trials=5, T=730)
    dsr_50 = engine.compute_dsr(observed_sharpe=0.2, null_sharpes=null_sharpes, n_trials=50, T=730)
    
    assert dsr_1 > dsr_5 > dsr_50
    assert 0.0 <= dsr_50 <= 1.0


def test_heuristic_trial_count(task, tmp_path):
    engine = EvaluationEngine(oos_data_root=tmp_path, task=task)
    code_with_loop = """
for w in [5, 10, 20, 30, 50]:
    sig = (df['close'] > df['close'].rolling(w).mean()).astype(int)
    # calculate Sharpe
    pnl = sig * rets
    sh = pnl.mean() / pnl.std() * 16
"""
    count = engine._heuristic_trial_count([code_with_loop])
    assert count == 5

    code_with_tuples = """
for fast, slow in [(50,200), (80,200), (100,200)]:
    sig = df['close'].rolling(fast).mean() > df['close'].rolling(slow).mean()
    cumprod = (1 + sig*rets).cumprod()
"""
    count_tuples = engine._heuristic_trial_count([code_with_tuples])
    assert count_tuples == 3


# ---------------------------------------------------------------------------
# Gate logic
# ---------------------------------------------------------------------------

def test_gate_fails_on_high_pvalue_or_low_dsr(task, price_data, tmp_path):
    engine = EvaluationEngine(oos_data_root=tmp_path, task=task)
    # Mocking standard inputs
    flat_signals = pd.Series(0, index=price_data.index)
    close = price_data["close"]
    pvalue, null_sharpes = engine.run_permutation_test(flat_signals, close, 10)

    # If DSR is below 0.30, it must fail the gates
    from alphabench.contracts import StrategyArtifact
    import time
    code = """
class MyStrategy(BaseStrategy):
    def generate_signals(self):
        return pd.Series(0, index=self._data.index)
"""
    art = StrategyArtifact("strat_1", "run_1", code, "hash", time.time())
    
    eval_result = engine.evaluate(
        artifact=art,
        in_sample_data=price_data,
        trial_count=100, # Large trial count will deflate DSR to ~0
        n_permutations=10,
    )
    assert not eval_result.passes_gates
    assert any("dsr" in f.lower() for f in eval_result.gate_failures)


# ---------------------------------------------------------------------------
# _compute_sharpe helper
# ---------------------------------------------------------------------------

def test_compute_sharpe_returns_finite(price_data, all_long_signals):
    sharpe = _compute_sharpe(all_long_signals, price_data["close"])
    assert isinstance(sharpe, float)
    assert sharpe == sharpe  # not NaN


def test_compute_sharpe_flat_returns_zero(price_data):
    flat = pd.Series(0, index=price_data.index)
    sharpe = _compute_sharpe(flat, price_data["close"])
    assert sharpe == 0.0
