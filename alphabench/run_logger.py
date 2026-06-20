"""
run_logger.py
-------------
Writes all benchmark events to a per-run NDJSON file.

Event envelope:
    {
        "event_type": "<string>",
        "run_id": "<uuid4>",
        "timestamp": 1234567890.123,
        "payload": { ... }
    }

Event types:
    agent_turn       — one LLM response turn
    tool_call        — a tool dispatch and result
    trial_report     — a TrialEntry recorded in the ledger
    backtest_result  — a BacktestResult from BacktestEngine
    strategy_submit  — a StrategyArtifact submitted by the agent
    eval_result      — an EvalResult from EvaluationEngine
    budget_warning   — injected when trial budget is exhausted
    token_update     — running TokenUsage after each LLM call
    run_complete     — final RunManifest at end of run

LangSmith is NOT used. All observability is local NDJSON only.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

from .contracts import (
    BacktestResult,
    EvalResult,
    Hypothesis,
    RunManifest,
    StrategyArtifact,
    TokenUsage,
    TrialEntry,
)


class RunLogger:
    """
    Appends structured NDJSON events to ``logs/{run_id}.ndjson``.

    File is kept open for the duration of the run and flushed after each write.
    Call finalize() to write the RunManifest and close the file.
    """

    def __init__(self, run_id: str, log_dir: Path) -> None:
        self.run_id = run_id
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        self._fh = open(log_dir / f"{run_id}.ndjson", "a", encoding="utf-8")

    # ------------------------------------------------------------------
    # Typed log methods
    # ------------------------------------------------------------------

    def log_agent_turn(
        self,
        role: str,
        content: str,
        token_count: int,
        turn_index: int,
    ) -> None:
        self._write(
            "agent_turn",
            {
                "role": role,
                "content": content[:2000],  # trim very long messages
                "token_count": token_count,
                "turn_index": turn_index,
            },
        )

    def log_tool_call(
        self,
        name: str,
        args: dict,
        result_summary: str,
        elapsed_ms: float,
        turn_index: int,
    ) -> None:
        self._write(
            "tool_call",
            {
                "name": name,
                "args": args,
                "result_summary": result_summary[:500],
                "elapsed_ms": elapsed_ms,
                "turn_index": turn_index,
            },
        )

    def log_trial_report(self, entry: TrialEntry) -> None:
        self._write("trial_report", asdict(entry))

    def log_hypothesis_creation(self, hypothesis: Hypothesis) -> None:
        self._write("hypothesis_create", asdict(hypothesis))

    def log_hypothesis_update(self, hypothesis: Hypothesis) -> None:
        self._write("hypothesis_update", asdict(hypothesis))

    def log_backtest(self, result: BacktestResult) -> None:
        # Store full result but cap equity_curve for log size
        data = asdict(result)
        if len(data["equity_curve"]) > 2000:
            data["equity_curve"] = data["equity_curve"][:2000]
            data["equity_curve_truncated"] = True
        self._write("backtest_result", data)

    def log_strategy_submission(self, artifact: StrategyArtifact) -> None:
        self._write("strategy_submit", asdict(artifact))

    def log_eval(self, result: EvalResult) -> None:
        self._write("eval_result", asdict(result))

    def log_budget_warning(self, used: int, limit: int, turn_index: int) -> None:
        self._write(
            "budget_warning",
            {
                "used": used,
                "limit": limit,
                "remaining": limit - used,
                "turn_index": turn_index,
            },
        )

    def log_token_update(self, token_usage: TokenUsage, turn_index: int) -> None:
        self._write(
            "token_update",
            {
                **asdict(token_usage),
                "turn_index": turn_index,
            },
        )

    def log_event(self, event_type: str, payload: dict) -> None:
        self._write(event_type, payload)

    def finalize(self, manifest: RunManifest) -> None:
        """Write the run_complete event and close the log file."""
        self._write("run_complete", _manifest_to_dict(manifest))
        self._fh.flush()
        self._fh.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _write(self, event_type: str, payload: dict) -> None:
        line = json.dumps(
            {
                "event_type": event_type,
                "run_id": self.run_id,
                "timestamp": time.time(),
                "payload": payload,
            },
            default=str,  # fallback for non-serialisable types
        )
        self._fh.write(line + "\n")
        self._fh.flush()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _manifest_to_dict(manifest: RunManifest) -> dict:
    """
    Convert RunManifest to a plain dict for JSON serialisation.
    Handles nested dataclasses (TaskDefinition, AgentConfig, etc.).
    """
    return asdict(manifest)
