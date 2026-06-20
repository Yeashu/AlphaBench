"""Tests for alphabench.strategy — BaseStrategy and StrategyRegistry."""

import pandas as pd
import numpy as np
import pytest

from alphabench.strategy import BaseStrategy, StrategyRegistry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_data():
    dates = pd.date_range("2021-01-01", periods=100, freq="D", tz="UTC")
    close = pd.Series(np.linspace(100, 200, 100), index=dates)
    return pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1000.0})


@pytest.fixture
def registry():
    return StrategyRegistry()


# ---------------------------------------------------------------------------
# Valid strategy source
# ---------------------------------------------------------------------------

VALID_SOURCE = """
class MyStrategy(BaseStrategy):
    def generate_signals(self):
        ma = self._data['close'].rolling(10).mean()
        signal = (self._data['close'] > ma).astype(int)
        return signal.fillna(0)
"""

ALL_LONG_SOURCE = """
class MyStrategy(BaseStrategy):
    def generate_signals(self):
        return pd.Series(1, index=self._data.index)
"""

ALL_FLAT_SOURCE = """
class MyStrategy(BaseStrategy):
    def generate_signals(self):
        return pd.Series(0, index=self._data.index)
"""

NON_DETERMINISTIC_SOURCE = """
import random as _r
class MyStrategy(BaseStrategy):
    def generate_signals(self):
        vals = [_r.choice([0, 1]) for _ in range(len(self._data))]
        return pd.Series(vals, index=self._data.index)
"""

WRONG_VALUES_SOURCE = """
class MyStrategy(BaseStrategy):
    def generate_signals(self):
        return pd.Series(2, index=self._data.index)
"""

NO_MY_STRATEGY_SOURCE = """
class WrongName(BaseStrategy):
    def generate_signals(self):
        return pd.Series(1, index=self._data.index)
"""

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_valid_strategy_loads(registry, sample_data):
    artifact = registry.submit(VALID_SOURCE, run_id="r1")
    strategy = registry.load(artifact, sample_data)
    signals = strategy.generate_signals()
    assert isinstance(signals, pd.Series)
    assert set(signals.unique()).issubset({0, 1})


def test_all_long_strategy(registry, sample_data):
    artifact = registry.submit(ALL_LONG_SOURCE, run_id="r1")
    strategy = registry.load(artifact, sample_data)
    signals = strategy.generate_signals()
    assert (signals == 1).all()


def test_all_flat_strategy(registry, sample_data):
    artifact = registry.submit(ALL_FLAT_SOURCE, run_id="r1")
    strategy = registry.load(artifact, sample_data)
    signals = strategy.generate_signals()
    assert (signals == 0).all()


def test_no_my_strategy_raises(registry, sample_data):
    artifact = registry.submit(NO_MY_STRATEGY_SOURCE, run_id="r1")
    with pytest.raises(ValueError, match="MyStrategy"):
        registry.load(artifact, sample_data)


def test_wrong_signal_values_raises(registry, sample_data):
    artifact = registry.submit(WRONG_VALUES_SOURCE, run_id="r1")
    with pytest.raises(AssertionError, match="0 or 1"):
        registry.load(artifact, sample_data)


def test_non_deterministic_raises(registry, sample_data):
    artifact = registry.submit(NON_DETERMINISTIC_SOURCE, run_id="r1")
    with pytest.raises((AssertionError, ValueError)):
        registry.load(artifact, sample_data)


def test_submit_produces_hash(registry):
    artifact = registry.submit(VALID_SOURCE, run_id="r1")
    assert len(artifact.source_hash) == 64  # SHA-256 hex
    assert artifact.strategy_id  # UUID4


def test_submit_same_source_same_hash(registry):
    a1 = registry.submit(VALID_SOURCE, run_id="r1")
    a2 = registry.submit(VALID_SOURCE, run_id="r1")
    assert a1.source_hash == a2.source_hash
    # But strategy_ids are unique
    assert a1.strategy_id != a2.strategy_id


def test_strategy_disallows_file_open(registry, sample_data):
    """Strategy exec environment should not allow open()."""
    source = """
class MyStrategy(BaseStrategy):
    def generate_signals(self):
        open('/etc/passwd', 'r')
        return pd.Series(0, index=self._data.index)
"""
    artifact = registry.submit(source, run_id="r1")
    with pytest.raises((ValueError, TypeError, NameError)):
        strategy = registry.load(artifact, sample_data)
        strategy.generate_signals()
