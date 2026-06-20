"""
contracts.py
------------
All typed dataclasses used as data contracts across AlphaBench module boundaries.
No logic lives here — pure data definitions only.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Market / Dataset
# ---------------------------------------------------------------------------


@dataclass
class AssetMetadata:
    """Metadata for a single tradeable asset, sourced from manifest.json."""

    asset_id: str
    exchange: str
    base_currency: str
    available_from: str  # ISO date string
    available_to: str    # ISO date string
    fields: list[str]
    instrument_type: str | None = None


@dataclass
class TaskDefinition:
    """
    A benchmark task is a first-class object.
    One task = one market + one asset universe + one hidden evaluation period.
    """

    task_id: str
    market: str
    asset_universe: list[str]
    dataset_version: str
    train_start: str  # ISO date string, e.g. "2021-01-01"
    train_end: str    # ISO date string, e.g. "2025-12-31"
    oos_start: str    # ISO date string, e.g. "2026-01-01"
    oos_end: str      # ISO date string, e.g. "2026-12-31"


# ---------------------------------------------------------------------------
# Agent configuration
# ---------------------------------------------------------------------------


@dataclass
class AgentConfig:
    """
    All agent runtime settings in one place.
    Loaded from a JSON/YAML config file; no scattered CLI args.
    """

    model_name: str
    max_turns: int             # default: 40
    max_trials: int            # default: 10
    temperature: float         # default: 0.2
    max_context_tokens: int    # default: 100_000
    system_prompt_version: str  # e.g. "v1.0.0"


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


@dataclass
class ExecutionResult:
    """Result from SandboxExecutor.run()."""

    stdout: str
    stderr: str
    returncode: int
    elapsed_ms: float
    timed_out: bool = False


# ---------------------------------------------------------------------------
# Trial tracking
# ---------------------------------------------------------------------------


@dataclass
class TrialEntry:
    """A single target-aware experiment recorded in the TrialLedger."""

    trial_index: int
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class Hypothesis:
    """A research hypothesis created and tracked by the agent."""

    hypothesis_id: str
    title: str
    description: str
    status: str  # "active", "paused", "falsified"
    notes: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


@dataclass
class StrategyArtifact:
    """Immutable record of a submitted strategy."""

    strategy_id: str   # UUID4
    run_id: str
    source_code: str
    source_hash: str   # SHA-256 of source_code
    submitted_at: float


# ---------------------------------------------------------------------------
# Backtests
# ---------------------------------------------------------------------------


@dataclass
class BacktestResult:
    """Full backtest output; returned by BacktestEngine.run()."""

    run_id: str
    trial_index: int
    strategy_id: str
    sharpe: float
    annual_return: float
    max_drawdown: float
    win_rate: float
    n_trades: int
    equity_curve: list[float]  # capped at 2000 points for storage
    elapsed_ms: float


@dataclass
class BacktestRecord:
    """
    Lightweight, searchable record stored in BacktestRegistry (SQLite).
    Created alongside every BacktestResult.
    """

    backtest_id: str   # UUID4
    run_id: str
    strategy_id: str
    strategy_hash: str
    trial_index: int
    sharpe: float
    annual_return: float
    max_drawdown: float
    n_trades: int
    created_at: str    # ISO datetime string


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@dataclass
class EvalResult:
    """Result of EvaluationEngine.evaluate() on hidden OOS data."""

    run_id: str
    strategy_id: str
    oos_sharpe: float
    dsr: float               # placeholder 0.0 until full formula is implemented
    permutation_pvalue: float
    passes_gates: bool       # MVP: gated on permutation_pvalue only; DSR is informational
    gate_failures: list[str]
    reported_trials: int = 0
    audited_trials: int = 0


# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------


@dataclass
class PromptVersion:
    """
    Tracks prompt identity for reproducibility.
    Any change to the prompt text produces a different sha256.
    """

    prompt_id: str    # e.g. "system_prompt"
    version: str      # e.g. "v1.0.0"
    sha256: str       # SHA-256 of the prompt text
    created_at: str   # ISO datetime string


@dataclass
class TokenUsage:
    """
    Accumulated token usage for a run.
    Updated after each LLM call by AgentRuntime.
    """

    input_tokens: int
    output_tokens: int
    total_tokens: int


# ---------------------------------------------------------------------------
# Run summary
# ---------------------------------------------------------------------------


@dataclass
class RunManifest:
    """
    Single source of truth for an AlphaBench run.
    Written as the final 'run_complete' NDJSON event by RunLogger.finalize().
    """

    run_id: str
    task: TaskDefinition
    config: AgentConfig
    prompt_version: PromptVersion
    token_usage: TokenUsage
    started_at: float
    finished_at: float
    total_trials: int
    total_backtests: int
    total_turns: int
    final_strategy_id: str | None
    eval_result: EvalResult | None
