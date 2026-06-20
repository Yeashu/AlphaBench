"""
trial_ledger.py
---------------
Tracks target-aware experiments and enforces the trial budget.

A "trial" is any target-aware action:
  - each backtest call
  - each explicit target-aware EDA experiment reported by the agent

Both count as +1 trial via report().
"""

from __future__ import annotations

import time

from .contracts import TrialEntry


class TrialBudgetExceeded(Exception):
    """Raised when report() is called on an exhausted ledger."""


class TrialLedger:
    """
    Budget-enforcing ledger for target-aware experiments.

    Not thread-safe (single-agent, single-threaded loop in MVP).
    """

    def __init__(self, max_trials: int) -> None:
        if max_trials <= 0:
            raise ValueError(f"max_trials must be > 0, got {max_trials}")
        self._max = max_trials
        self._entries: list[TrialEntry] = []
        self._closed = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def report(self, reason: str, metadata: dict | None = None) -> TrialEntry:
        """
        Record a trial. Raises RuntimeError if closed, TrialBudgetExceeded if exhausted.

        Parameters
        ----------
        reason:
            Human-readable description of the trial (e.g. "backtest", "target_aware_eda").
        metadata:
            Optional dict of additional information (strategy_id, sharpe, etc.).

        Returns
        -------
        TrialEntry
        """
        if self._closed:
            raise RuntimeError("TrialLedger is closed — no further trials may be reported.")
        if self.is_exhausted():
            raise TrialBudgetExceeded(
                f"Trial budget of {self._max} exhausted. "
                f"Current count: {len(self._entries)}."
            )

        entry = TrialEntry(
            trial_index=len(self._entries),
            reason=reason,
            metadata=metadata or {},
            timestamp=time.time(),
        )
        self._entries.append(entry)
        return entry

    def count(self) -> int:
        """Return the number of trials recorded so far."""
        return len(self._entries)

    def remaining(self) -> int:
        """Return the number of trials remaining in the budget."""
        return max(0, self._max - len(self._entries))

    def is_exhausted(self) -> bool:
        """Return True if the trial budget has been reached or exceeded."""
        return len(self._entries) >= self._max

    def entries(self) -> list[TrialEntry]:
        """Return a copy of all recorded trial entries."""
        return list(self._entries)

    def close(self) -> None:
        """
        Permanently close the ledger. No further trials may be recorded.
        Called by AgentRuntime at the end of the benchmark loop.
        """
        self._closed = True

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def max_trials(self) -> int:
        return self._max
