"""
strategy.py
-----------
BaseStrategy ABC and StrategyRegistry for loading agent-submitted strategies.

Agent contract:
    1. Define a class named ``MyStrategy``
    2. Subclass ``BaseStrategy``
    3. Implement ``generate_signals()`` returning a deterministic pd.Series
    4. Values must be 0 (flat) or 1 (long) only
    5. Use only the injected ``pd`` and ``np`` — no imports allowed
    6. No external calls, no file I/O, no network
"""

from __future__ import annotations

import hashlib
import time
import uuid
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from .contracts import StrategyArtifact


# ---------------------------------------------------------------------------
# BaseStrategy
# ---------------------------------------------------------------------------


class BaseStrategy(ABC):
    """
    Abstract base class for all AlphaBench strategies.

    Strategies receive OHLCV data via __init__.
    generate_signals() must be deterministic and return 0/1 signals.
    No external calls or file I/O are permitted.
    """

    def __init__(self, data: pd.DataFrame) -> None:
        self._data = data

    @abstractmethod
    def generate_signals(self) -> pd.Series:
        """
        Returns a pd.Series indexed by the same DatetimeIndex as self._data.
        Values: 1 = long, 0 = flat.
        """
        ...

    def validate(self) -> None:
        """
        Validate that generate_signals() satisfies the benchmark contract.
        Called automatically by StrategyRegistry.load() before returning.

        Raises
        ------
        AssertionError
            If any contract condition is violated.
        """
        signals = self.generate_signals()

        assert isinstance(signals, pd.Series), (
            f"generate_signals() must return pd.Series, got {type(signals).__name__}"
        )
        assert signals.index.equals(self._data.index), (
            "Signal index must exactly match data index"
        )

        unique_vals = set(signals.dropna().unique())
        assert unique_vals.issubset({0, 1}), (
            f"Signal values must be 0 or 1, found: {unique_vals - {0, 1}}"
        )

        # Determinism check: call twice, compare
        signals2 = self.generate_signals()
        assert signals.equals(signals2), (
            "generate_signals() must be deterministic — two consecutive calls returned different results"
        )


# ---------------------------------------------------------------------------
# StrategyRegistry
# ---------------------------------------------------------------------------


class StrategyRegistry:
    """
    Compiles and loads agent-submitted strategy source code.

    Security note:
        exec() with restricted builtins provides lightweight isolation.
        The strategy's __builtins__ is a curated whitelist — no open(),
        no __import__(), no eval(). Full Docker isolation is post-MVP.
    """

    # Restricted execution environment for strategy code.
    # Strategy code is allowed to use class syntax, pd, np, and basic builtins.
    # Imports are blocked by overriding __import__ — the real builtins module is
    # used so that class definitions and ABC machinery work correctly.
    # Note: full Docker isolation is planned for post-MVP (LLD §6.2 Layer 3).
    _BLOCKED_BUILTINS: dict = __builtins__ if isinstance(__builtins__, dict) else vars(__builtins__)  # type: ignore[arg-type]

    @classmethod
    def _make_exec_globals(cls) -> dict:
        """Build a fresh restricted globals dict for each strategy execution."""
        import builtins

        def _blocked_import(*args, **kwargs):  # noqa: ANN002
            raise ImportError(
                "import statements are not allowed in strategy code. "
                "Use the pre-injected 'pd' and 'np' objects."
            )

        restricted_builtins = {
            k: v for k, v in vars(builtins).items()
            if k not in {"open", "exec", "eval", "compile", "__import__", "breakpoint"}
        }
        restricted_builtins["__import__"] = _blocked_import

        return {
            "__builtins__": restricted_builtins,
            "pd": pd,
            "np": np,
            "BaseStrategy": None,  # populated in load()
        }

    def submit(self, source_code: str, run_id: str) -> StrategyArtifact:
        """
        Record an agent strategy submission.
        Does NOT execute or validate the code — call load() for that.

        Parameters
        ----------
        source_code:
            Python source defining MyStrategy(BaseStrategy).
        run_id:
            The current benchmark run ID.

        Returns
        -------
        StrategyArtifact
            Immutable record with UUID strategy_id and SHA-256 hash.
        """
        source_hash = hashlib.sha256(source_code.encode("utf-8")).hexdigest()
        return StrategyArtifact(
            strategy_id=str(uuid.uuid4()),
            run_id=run_id,
            source_code=source_code,
            source_hash=source_hash,
            submitted_at=time.time(),
        )

    def load(self, artifact: StrategyArtifact, data: pd.DataFrame) -> BaseStrategy:
        """
        Compile and instantiate the strategy from *artifact*.

        Parameters
        ----------
        artifact:
            StrategyArtifact produced by submit().
        data:
            Training OHLCV DataFrame to pass to the strategy __init__.

        Returns
        -------
        BaseStrategy
            Validated strategy instance ready for generate_signals().

        Raises
        ------
        ValueError
            If the source does not define ``MyStrategy``.
        AssertionError
            If validate() fails (wrong return type, wrong values, non-deterministic).
        """
        globs = self._make_exec_globals()
        globs["BaseStrategy"] = BaseStrategy
        try:
            exec(  # noqa: S102
                compile(artifact.source_code, "<strategy>", "exec"),
                globs,
            )
        except Exception as e:
            raise ValueError(f"Strategy source failed to execute: {e}") from e

        strategy_cls = globs.get("MyStrategy")
        if strategy_cls is None:
            raise ValueError(
                "Strategy source must define a class named 'MyStrategy'. "
                "None was found after execution."
            )

        instance: BaseStrategy = strategy_cls(data=data)
        instance.validate()
        return instance
