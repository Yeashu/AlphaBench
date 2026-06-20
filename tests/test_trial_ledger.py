"""Tests for alphabench.trial_ledger."""

import pytest
from alphabench.trial_ledger import TrialBudgetExceeded, TrialLedger


def test_report_increments_count():
    ledger = TrialLedger(max_trials=5)
    ledger.report("backtest")
    ledger.report("backtest")
    assert ledger.count() == 2


def test_remaining_decrements():
    ledger = TrialLedger(max_trials=5)
    ledger.report("backtest")
    assert ledger.remaining() == 4


def test_is_exhausted_triggers_at_limit():
    ledger = TrialLedger(max_trials=2)
    assert not ledger.is_exhausted()
    ledger.report("a")
    assert not ledger.is_exhausted()
    ledger.report("b")
    assert ledger.is_exhausted()


def test_report_after_exhaustion_raises():
    ledger = TrialLedger(max_trials=1)
    ledger.report("first")
    with pytest.raises(TrialBudgetExceeded):
        ledger.report("second")


def test_close_prevents_reporting():
    ledger = TrialLedger(max_trials=10)
    ledger.close()
    with pytest.raises(RuntimeError, match="closed"):
        ledger.report("after_close")


def test_entries_returns_copy():
    ledger = TrialLedger(max_trials=5)
    ledger.report("a", metadata={"x": 1})
    entries = ledger.entries()
    entries.clear()
    assert ledger.count() == 1


def test_invalid_max_trials():
    with pytest.raises(ValueError):
        TrialLedger(max_trials=0)


def test_trial_entry_fields():
    ledger = TrialLedger(max_trials=5)
    entry = ledger.report("backtest", metadata={"strategy_id": "abc"})
    assert entry.trial_index == 0
    assert entry.reason == "backtest"
    assert entry.metadata["strategy_id"] == "abc"
    assert entry.timestamp > 0
