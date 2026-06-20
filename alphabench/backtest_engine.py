"""
backtest_engine.py
------------------
Runs a validated strategy on training data using VectorBT and extracts
metrics into BacktestResult + BacktestRecord.

Design rule (from LLD §10.2):
    The VectorBT Portfolio object is created, metrics are extracted,
    and the object is immediately discarded (``del pf``).
    It never escapes the method.
"""

from __future__ import annotations

import math
import time
import uuid
from datetime import datetime, timezone

import pandas as pd
import vectorbt as vbt

from .backtest_registry import BacktestRegistry
from .contracts import BacktestRecord, BacktestResult, StrategyArtifact
from .run_logger import RunLogger
from .strategy import BaseStrategy
from .trial_ledger import TrialLedger

# Maximum equity curve length stored in NDJSON / BacktestResult
_MAX_EQUITY_CURVE_POINTS = 2000


class BacktestEngine:
    """
    Runs a strategy on price data and produces BacktestResult + BacktestRecord.

    Every call to run() is recorded in both the RunLogger (full NDJSON event)
    and the BacktestRegistry (searchable SQLite row).
    """

    def __init__(
        self,
        trial_ledger: TrialLedger,
        logger: RunLogger,
        registry: BacktestRegistry,
    ) -> None:
        self._ledger = trial_ledger
        self._logger = logger
        self._registry = registry
        self._backtest_count = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        strategy: BaseStrategy,
        price_data: pd.DataFrame,
        artifact: StrategyArtifact,
    ) -> BacktestResult:
        """
        Backtest *strategy* on *price_data* and return metrics.

        The VectorBT Portfolio object is discarded immediately after
        metric extraction.

        Parameters
        ----------
        strategy:
            Validated BaseStrategy instance.
        price_data:
            OHLCV DataFrame with DatetimeIndex; must contain a "close" column.
        artifact:
            The StrategyArtifact this strategy was compiled from.

        Returns
        -------
        BacktestResult
        """
        start = time.monotonic()
        self._backtest_count += 1

        signals = strategy.generate_signals()
        close = price_data["close"]

        pf = vbt.Portfolio.from_signals(
            close=close,
            entries=(signals == 1),
            exits=(signals == 0),
            freq="1D",
        )

        # Extract metrics — handle NaN/inf defensively
        sharpe = _safe_float(pf.sharpe_ratio())
        annual_return = _safe_float(pf.annualized_return())
        max_drawdown = _safe_float(pf.max_drawdown())
        n_trades = int(pf.trades.count())
        win_rate = _safe_float(pf.trades.win_rate()) if n_trades > 0 else 0.0
        equity_curve = pf.value().tolist()

        del pf  # LLD §10.2: discard immediately

        # Cap equity curve for storage
        if len(equity_curve) > _MAX_EQUITY_CURVE_POINTS:
            equity_curve = equity_curve[:_MAX_EQUITY_CURVE_POINTS]

        elapsed = (time.monotonic() - start) * 1000

        result = BacktestResult(
            run_id=artifact.run_id,
            trial_index=self._ledger.count(),
            strategy_id=artifact.strategy_id,
            sharpe=sharpe,
            annual_return=annual_return,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            n_trades=n_trades,
            equity_curve=equity_curve,
            elapsed_ms=elapsed,
        )

        # Write searchable record to BacktestRegistry
        record = BacktestRecord(
            backtest_id=str(uuid.uuid4()),
            run_id=artifact.run_id,
            strategy_id=artifact.strategy_id,
            strategy_hash=artifact.source_hash,
            trial_index=result.trial_index,
            sharpe=sharpe,
            annual_return=annual_return,
            max_drawdown=max_drawdown,
            n_trades=n_trades,
            created_at=datetime.now(tz=timezone.utc).isoformat(),
        )
        self._registry.add(record)

        # Log full result to NDJSON
        self._logger.log_backtest(result)

        return result

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def backtest_count(self) -> int:
        return self._backtest_count


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_float(value: object) -> float:
    """Convert VectorBT metric to a finite float; return 0.0 on NaN/inf."""
    try:
        f = float(value)  # type: ignore[arg-type]
        return f if math.isfinite(f) else 0.0
    except (TypeError, ValueError):
        return 0.0
