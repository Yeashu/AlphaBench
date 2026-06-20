"""
backtest_registry.py
--------------------
SQLite-backed, searchable store for every BacktestRecord ever produced.

Created alongside every BacktestResult in BacktestEngine.run().
Provides audit-friendly search across runs (unlike NDJSON logs which are
append-only and require full scans).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from .contracts import BacktestRecord

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS backtests (
    backtest_id   TEXT PRIMARY KEY,
    run_id        TEXT NOT NULL,
    strategy_id   TEXT NOT NULL,
    strategy_hash TEXT NOT NULL,
    trial_index   INTEGER NOT NULL,
    sharpe        REAL,
    annual_return REAL,
    max_drawdown  REAL,
    n_trades      INTEGER,
    created_at    TEXT NOT NULL
);
"""

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_backtests_run ON backtests(run_id);
"""


class BacktestRegistry:
    """
    Persistent store for BacktestRecord objects.

    Backed by a SQLite database at *db_path*.
    The database and table are created automatically if they do not exist.
    """

    def __init__(self, db_path: Path) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._conn:
            self._conn.execute(_CREATE_TABLE)
            self._conn.execute(_CREATE_INDEX)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add(self, record: BacktestRecord) -> None:
        """Insert a BacktestRecord. Silently ignores duplicate backtest_ids."""
        with self._conn:
            self._conn.execute(
                """
                INSERT OR IGNORE INTO backtests
                (backtest_id, run_id, strategy_id, strategy_hash,
                 trial_index, sharpe, annual_return, max_drawdown,
                 n_trades, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.backtest_id,
                    record.run_id,
                    record.strategy_id,
                    record.strategy_hash,
                    record.trial_index,
                    record.sharpe,
                    record.annual_return,
                    record.max_drawdown,
                    record.n_trades,
                    record.created_at,
                ),
            )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(self, backtest_id: str) -> BacktestRecord | None:
        """Return a single BacktestRecord by ID, or None if not found."""
        row = self._conn.execute(
            "SELECT * FROM backtests WHERE backtest_id = ?", (backtest_id,)
        ).fetchone()
        return _row_to_record(row) if row else None

    def list_run(self, run_id: str) -> list[BacktestRecord]:
        """Return all BacktestRecords for a given run, ordered by trial_index."""
        rows = self._conn.execute(
            "SELECT * FROM backtests WHERE run_id = ? ORDER BY trial_index ASC",
            (run_id,),
        ).fetchall()
        return [_row_to_record(r) for r in rows]

    def best_for_run(self, run_id: str) -> BacktestRecord | None:
        """Return the BacktestRecord with the highest Sharpe for a given run."""
        row = self._conn.execute(
            """
            SELECT * FROM backtests
            WHERE run_id = ?
            ORDER BY sharpe DESC
            LIMIT 1
            """,
            (run_id,),
        ).fetchone()
        return _row_to_record(row) if row else None

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row_to_record(row: sqlite3.Row) -> BacktestRecord:
    return BacktestRecord(
        backtest_id=row["backtest_id"],
        run_id=row["run_id"],
        strategy_id=row["strategy_id"],
        strategy_hash=row["strategy_hash"],
        trial_index=row["trial_index"],
        sharpe=row["sharpe"] or 0.0,
        annual_return=row["annual_return"] or 0.0,
        max_drawdown=row["max_drawdown"] or 0.0,
        n_trades=row["n_trades"] or 0,
        created_at=row["created_at"],
    )
