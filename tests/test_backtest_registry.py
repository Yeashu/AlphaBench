"""Tests for alphabench.backtest_registry."""

import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from alphabench.backtest_registry import BacktestRegistry
from alphabench.contracts import BacktestRecord


def _make_record(
    backtest_id: str = "b1",
    run_id: str = "r1",
    strategy_id: str = "s1",
    sharpe: float = 1.5,
    trial_index: int = 0,
) -> BacktestRecord:
    return BacktestRecord(
        backtest_id=backtest_id,
        run_id=run_id,
        strategy_id=strategy_id,
        strategy_hash="abc123",
        trial_index=trial_index,
        sharpe=sharpe,
        annual_return=0.25,
        max_drawdown=0.10,
        n_trades=30,
        created_at=datetime.now(tz=timezone.utc).isoformat(),
    )


@pytest.fixture
def registry(tmp_path):
    db_path = tmp_path / "test_backtests.db"
    reg = BacktestRegistry(db_path)
    yield reg
    reg.close()


def test_add_and_get(registry):
    record = _make_record()
    registry.add(record)
    fetched = registry.get("b1")
    assert fetched is not None
    assert fetched.backtest_id == "b1"
    assert fetched.sharpe == 1.5


def test_get_missing_returns_none(registry):
    assert registry.get("nonexistent") is None


def test_list_run(registry):
    registry.add(_make_record("b1", run_id="r1", trial_index=0))
    registry.add(_make_record("b2", run_id="r1", trial_index=1))
    registry.add(_make_record("b3", run_id="r2", trial_index=0))
    result = registry.list_run("r1")
    assert len(result) == 2
    assert all(r.run_id == "r1" for r in result)
    # Should be ordered by trial_index
    assert result[0].trial_index == 0
    assert result[1].trial_index == 1


def test_best_for_run(registry):
    registry.add(_make_record("b1", run_id="r1", sharpe=1.0))
    registry.add(_make_record("b2", run_id="r1", sharpe=2.5))
    registry.add(_make_record("b3", run_id="r1", sharpe=0.5))
    best = registry.best_for_run("r1")
    assert best is not None
    assert best.sharpe == 2.5


def test_best_for_run_empty(registry):
    assert registry.best_for_run("no-such-run") is None


def test_duplicate_add_ignored(registry):
    """Duplicate backtest_id should be silently ignored (INSERT OR IGNORE)."""
    registry.add(_make_record("b1", sharpe=1.0))
    registry.add(_make_record("b1", sharpe=99.0))  # duplicate
    fetched = registry.get("b1")
    assert fetched.sharpe == 1.0  # original preserved
