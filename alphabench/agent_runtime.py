"""
agent_runtime.py
----------------
Central orchestrator for the AlphaBench benchmark loop.

Responsibilities:
    - Initialize all services from AgentConfig + TaskDefinition
    - Define and dispatch the 7 agent tools
    - Run the LLM turn loop with budget enforcement
    - Accumulate CostSummary from response.usage
    - Handle context trimming
    - Produce the final RunManifest

Tools exposed to the agent:
    list_assets          — 0 trials
    get_asset_metadata   — 0 trials
    run_eda              — 0 trials (unless agent also calls report_trial)
    report_trial         — +1 trial
    submit_strategy      — 0 trials
    run_backtest         — +1 trial
    submit_final         — 0 trials, terminates loop
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import openai

from .backtest_engine import BacktestEngine
from .backtest_registry import BacktestRegistry
from .contracts import (
    AgentConfig,
    Hypothesis,
    RunManifest,
    StrategyArtifact,
    TaskDefinition,
    TokenUsage,
)
from .dataset_service import DatasetService
from .prompts import format_prompt
from .run_logger import RunLogger
from .sandbox_executor import SandboxExecutor
from .strategy import StrategyRegistry
from .trial_ledger import TrialBudgetExceeded, TrialLedger

# ---------------------------------------------------------------------------
# OpenAI tool schemas
# ---------------------------------------------------------------------------

_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "list_assets",
            "description": "List all available asset IDs in the dataset. Call with no arguments: list_assets()",
            "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_asset_metadata",
            "description": "Get metadata for an asset: date range, exchange, available fields.",
            "parameters": {
                "type": "object",
                "properties": {
                    "asset_id": {
                        "type": "string",
                        "description": "Asset identifier, e.g. 'BTC-USDT'",
                    }
                },
                "required": ["asset_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_eda",
            "description": (
                "Run exploratory data analysis code. "
                "Variables pre-injected: df (OHLCV DataFrame), pd, np. "
                "No import statements needed. Returns stdout. "
                "Does NOT cost a trial unless you also call report_trial()."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute. Use df, pd, np directly.",
                    }
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "report_trial",
            "description": (
                "Declare that you just performed a target-aware EDA experiment. "
                "Call this when your analysis used returns, PnL, Sharpe, drawdown, "
                "forward returns, or any performance feedback. Costs +1 trial."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Brief description of what target-aware analysis was done.",
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Optional dict of additional context.",
                    },
                },
                "required": ["reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "report_trials",
            "description": (
                "Declare that you just performed multiple target-aware EDA experiments at once (e.g. a parameter sweep or grid search). "
                "Each item in the list costs +1 trial. Use this to report all parameter configurations tested."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "trials": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "reason": {
                                    "type": "string",
                                    "description": "Description of the specific parameter setting or trial.",
                                },
                                "metadata": {
                                    "type": "object",
                                    "description": "Optional dict of additional context for this specific trial.",
                                },
                            },
                            "required": ["reason"],
                        },
                        "description": "List of trial objects to report.",
                    }
                },
                "required": ["trials"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_hypothesis",
            "description": (
                "Create a new quantitative alpha research hypothesis. "
                "Returns a unique hypothesis_id. "
                "Use this to document a new research idea before testing it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Short title of the hypothesis (e.g. 'RSI Mean Reversion').",
                    },
                    "description": {
                        "type": "string",
                        "description": "Detailed explanation of the hypothesis, signal logic, and underlying rationale.",
                    },
                },
                "required": ["title", "description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_hypothesis",
            "description": (
                "Update the status and append research notes/conclusions for an existing hypothesis."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "hypothesis_id": {
                        "type": "string",
                        "description": "The unique ID returned by create_hypothesis().",
                    },
                    "status": {
                        "type": "string",
                        "enum": ["active", "paused", "falsified"],
                        "description": "The current status of this research path.",
                    },
                    "notes": {
                        "type": "string",
                        "description": "Optional research notes, observations, backtest findings, or reasons for status change.",
                    },
                },
                "required": ["hypothesis_id", "status"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_strategy",
            "description": (
                "Submit a MyStrategy implementation. Returns a strategy_id. "
                "You can submit multiple strategies; only the one you declare "
                "final via submit_final() is evaluated. Does NOT cost a trial."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source_code": {
                        "type": "string",
                        "description": (
                            "Python source code defining MyStrategy(BaseStrategy). "
                            "Must implement generate_signals() returning 0/1 pd.Series."
                        ),
                    }
                },
                "required": ["source_code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_backtest",
            "description": (
                "Run a backtest on the specified strategy using training data. "
                "Returns sharpe, annual_return, max_drawdown, win_rate, n_trades. "
                "Costs +1 trial."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "strategy_id": {
                        "type": "string",
                        "description": "strategy_id returned by submit_strategy().",
                    }
                },
                "required": ["strategy_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_final",
            "description": (
                "Declare your final strategy. This ends the research loop. "
                "The declared strategy will be evaluated on hidden OOS data. "
                "Does NOT cost a trial."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "strategy_id": {
                        "type": "string",
                        "description": "strategy_id of the strategy you want to submit as final.",
                    }
                },
                "required": ["strategy_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "checkpoint",
            "description": (
                "Mark this strategy as your current best fallback. The research loop "
                "continues. If the session ends without you calling submit_final(), "
                "this strategy is auto-submitted for evaluation. Call again to upgrade. "
                "Does NOT end the research loop. Does NOT cost a trial."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "strategy_id": {
                        "type": "string",
                        "description": "strategy_id of the strategy to checkpoint.",
                    }
                },
                "required": ["strategy_id"],
            },
        },
    },
]

# Context trimming: keep this many recent tool results when trimming
_KEEP_RECENT_TOOL_RESULTS = 5

# Fraction of max_context_tokens at which trimming kicks in
_TRIM_THRESHOLD = 0.85


# ---------------------------------------------------------------------------
# AgentRuntime
# ---------------------------------------------------------------------------


class AgentRuntime:
    """
    Runs the AlphaBench agent loop.

    Accepts fully-constructed service instances so that the caller controls
    dependency wiring (facilitates testing and swapping components).
    """

    def __init__(
        self,
        run_id: str,
        config: AgentConfig,
        task: TaskDefinition,
        dataset_service: DatasetService,
        sandbox: SandboxExecutor,
        ledger: TrialLedger,
        logger: RunLogger,
        registry: BacktestRegistry,
        backtest_engine: BacktestEngine,
        strategy_registry: StrategyRegistry,
        openai_client: openai.OpenAI,
    ) -> None:
        self._run_id = run_id
        self._config = config
        self._task = task
        self._dataset = dataset_service
        self._sandbox = sandbox
        self._ledger = ledger
        self._logger = logger
        self._bt_registry = registry
        self._backtest_engine = backtest_engine
        self._strategy_registry = strategy_registry
        self._client = openai_client

        # Mutable run state
        self._artifacts: dict[str, StrategyArtifact] = {}  # strategy_id → artifact
        self._final_strategy_id: str | None = None
        self._checkpoint_strategy_id: str | None = None
        self._token_usage = TokenUsage(0, 0, 0)
        self._started_at: float = 0.0
        self._hypotheses: dict[str, Hypothesis] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self) -> RunManifest:
        """
        Execute the full agent loop and return the RunManifest.

        The loop ends when:
            - submit_final() is called by the agent, OR
            - max_turns is reached, OR
            - max_context_tokens is reached
        """
        self._started_at = time.time()
        turn = 0
        budget_warned = False
        turn_warned = False
        nudged = False

        # --- Build system prompt ---
        primary_asset = self._task.asset_universe[0]
        prompt_text, prompt_version = format_prompt(
            self._config.system_prompt_version,
            asset_id=primary_asset,
            max_trials=str(self._config.max_trials),
        )

        messages: list[dict] = [
            {"role": "system", "content": prompt_text},
            {
                "role": "user",
                "content": (
                    f"Begin your research. Your task is {primary_asset}, "
                    f"trial budget: {self._config.max_trials}. "
                    "Call list_assets() to start."
                ),
            },
        ]

        # --- Main loop ---
        while turn < self._config.max_turns:

            # Context trimming check
            if self._token_usage.total_tokens > self._config.max_context_tokens * _TRIM_THRESHOLD:
                messages = _trim_context(messages)

            # LLM call
            try:
                response = self._client.chat.completions.create(
                    model=self._config.model_name,
                    messages=messages,  # type: ignore[arg-type]
                    tools=_TOOLS,
                    tool_choice="auto",
                    temperature=self._config.temperature,
                )
            except Exception as e:
                # Log and break on API error
                self._logger.log_agent_turn(
                    role="error",
                    content=str(e),
                    token_count=0,
                    turn_index=turn,
                )
                break

            # Accumulate token usage
            if response.usage:
                self._token_usage = _accumulate_token_usage(
                    self._token_usage,
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                )
                self._logger.log_token_update(self._token_usage, turn_index=turn)

            if not response.choices:
                self._logger.log_agent_turn(
                    role="error",
                    content=f"API returned response with no choices. Full response: {response}",
                    token_count=0,
                    turn_index=turn,
                )
                break

            msg = response.choices[0].message


            # Log the assistant turn
            self._logger.log_agent_turn(
                role="assistant",
                content=msg.content or "",
                token_count=response.usage.completion_tokens if response.usage else 0,
                turn_index=turn,
            )

            # Append assistant message to conversation
            messages.append(msg.model_dump(exclude_unset=True))

            # --- Finish reason: stop ---
            if response.choices[0].finish_reason == "stop":
                if self._final_strategy_id is not None or self._checkpoint_strategy_id is not None:
                    break  # clean exit
                if not nudged:
                    # Nudge agent to submit_final
                    nudge_msg = (
                        "You have stopped without submitting a final strategy. "
                        "Please call submit_final(strategy_id) with your best strategy. "
                        f"Remaining trials: {self._ledger.remaining()}."
                    )
                    messages.append({"role": "user", "content": nudge_msg})
                    nudged = True
                    turn += 1
                    continue
                else:
                    break  # already nudged once; give up

            # --- Dispatch tool calls ---
            if msg.tool_calls:
                nudged = False  # reset nudge if agent is actively calling tools
                for tc in msg.tool_calls:
                    tool_start = time.monotonic()
                    result_content = self._dispatch_tool(tc.function.name, tc.function.arguments)
                    elapsed = (time.monotonic() - tool_start) * 1000

                    # Parse args safely for logging
                    try:
                        args_dict = json.loads(tc.function.arguments)
                    except Exception:
                        args_dict = {"raw": tc.function.arguments}

                    self._logger.log_tool_call(
                        name=tc.function.name,
                        args=args_dict,
                        result_summary=result_content,
                        elapsed_ms=elapsed,
                        turn_index=turn,
                    )

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result_content,
                        }
                    )

                # Check: did the agent just call submit_final?
                if self._final_strategy_id is not None:
                    break

                # Budget warning injection
                if self._ledger.is_exhausted() and not budget_warned:
                    if self._checkpoint_strategy_id:
                        warning = (
                            f"⚠️ BUDGET WARNING: You have used all {self._config.max_trials} trials. "
                            "No further backtests or report_trial() calls are allowed. "
                            f"Your current checkpoint (strategy {self._checkpoint_strategy_id[:8]}...) "
                            "will be auto-submitted if you do not call submit_final(). "
                            "If you want to choose a different strategy, call submit_final() or checkpoint() now; otherwise you can end your turn."
                        )
                    else:
                        warning = (
                            f"⚠️ BUDGET WARNING: You have used all {self._config.max_trials} trials. "
                            "No further backtests or report_trial() calls are allowed. "
                            "Please call checkpoint(strategy_id) or submit_final(strategy_id) with your best strategy now."
                        )
                    messages.append({"role": "user", "content": warning})
                    self._logger.log_budget_warning(
                        used=self._ledger.count(),
                        limit=self._config.max_trials,
                        turn_index=turn,
                    )
                    budget_warned = True

                # Turn warning injection
                if turn >= self._config.max_turns - 3 and not turn_warned:
                    if self._checkpoint_strategy_id:
                        warning = (
                            f"⚠️ TURN WARNING: You are approaching the maximum turn limit ({self._config.max_turns}). "
                            f"You have only {self._config.max_turns - turn} turns remaining. "
                            f"Your current checkpoint (strategy {self._checkpoint_strategy_id[:8]}...) "
                            "will be auto-submitted if you do not call submit_final(). "
                            "If you have new results to test, do so now; otherwise call submit_final()."
                        )
                    else:
                        warning = (
                            f"⚠️ TURN WARNING: You are approaching the maximum turn limit ({self._config.max_turns}). "
                            f"You have only {self._config.max_turns - turn} turns remaining. "
                            "You MUST call checkpoint(strategy_id) or submit_final(strategy_id) now before the loop terminates."
                        )
                    messages.append({"role": "user", "content": warning})
                    turn_warned = True

            turn += 1

        # Auto-submit checkpoint or best backtested strategy if no explicit submit_final was called
        if self._final_strategy_id is None:
            best_id = self._checkpoint_strategy_id
            try:
                records = self._bt_registry.list_run(self._run_id)
                if records:
                    passing_records = [r for r in records if r.n_trades >= 30]
                    if passing_records:
                        passing_records.sort(key=lambda r: r.sharpe, reverse=True)
                        best_id = passing_records[0].strategy_id
                    else:
                        records.sort(key=lambda r: r.sharpe, reverse=True)
                        best_id = records[0].strategy_id
            except Exception:
                pass

            if best_id is not None:
                self._final_strategy_id = best_id
                self._logger.log_event("checkpoint_auto_submitted", {
                    "strategy_id": best_id,
                    "reason": "session_ended_without_submit_final",
                })

        # --- Close ledger ---
        self._ledger.close()

        # --- Build RunManifest ---
        manifest = RunManifest(
            run_id=self._run_id,
            task=self._task,
            config=self._config,
            prompt_version=prompt_version,
            token_usage=self._token_usage,
            started_at=self._started_at,
            finished_at=time.time(),
            total_trials=self._ledger.count(),
            total_backtests=self._backtest_engine.backtest_count,
            total_turns=turn,
            final_strategy_id=self._final_strategy_id,
            eval_result=None,  # filled in by run_benchmark.py after EvaluationEngine
        )
        return manifest

    def get_final_artifact(self) -> StrategyArtifact | None:
        """Return the artifact for the final strategy, or None if not submitted."""
        if self._final_strategy_id and self._final_strategy_id in self._artifacts:
            return self._artifacts[self._final_strategy_id]
        return None

    # ------------------------------------------------------------------
    # Tool dispatch
    # ------------------------------------------------------------------

    def _dispatch_tool(self, name: str, arguments_json: str) -> str:
        """Route a tool call to the appropriate handler. Returns a string result."""
        # Some models (e.g. deepseek) emit trailing garbage after the JSON object,
        # e.g. '{}""'. Strip whitespace and attempt to parse just the leading object.
        raw = (arguments_json or "").strip()
        args: dict = {}
        if raw:
            try:
                args = json.loads(raw)
            except json.JSONDecodeError:
                # Try to salvage by taking only up to the first closing brace
                try:
                    brace_end = raw.index("}") + 1
                    args = json.loads(raw[:brace_end])
                except (ValueError, json.JSONDecodeError):
                    # For no-arg tools it's safe to proceed with empty args;
                    # for tools that require args this will surface as a missing-arg error.
                    args = {}

        handlers = {
            "list_assets": self._tool_list_assets,
            "get_asset_metadata": self._tool_get_asset_metadata,
            "run_eda": self._tool_run_eda,
            "report_trial": self._tool_report_trial,
            "report_trials": self._tool_report_trials,
            "create_hypothesis": self._tool_create_hypothesis,
            "update_hypothesis": self._tool_update_hypothesis,
            "submit_strategy": self._tool_submit_strategy,
            "run_backtest": self._tool_run_backtest,
            "submit_final": self._tool_submit_final,
            "checkpoint": self._tool_checkpoint,
        }

        handler = handlers.get(name)
        if handler is None:
            return json.dumps({"error": f"Unknown tool: {name!r}"})

        try:
            return handler(**args)
        except Exception as e:
            return json.dumps({"error": str(e)})

    def _tool_list_assets(self) -> str:
        assets = self._dataset.list_assets()
        return json.dumps({"assets": assets})

    def _tool_get_asset_metadata(self, asset_id: str) -> str:
        try:
            meta = self._dataset.get_metadata(asset_id)
            return json.dumps({
                "asset_id": meta.asset_id,
                "exchange": meta.exchange,
                "base_currency": meta.base_currency,
                "available_from": meta.available_from,
                "available_to": meta.available_to,
                "fields": meta.fields,
            })
        except KeyError as e:
            return json.dumps({"error": str(e)})

    def _tool_run_eda(self, code: str) -> str:
        # Load the primary asset's full training data
        primary = self._task.asset_universe[0]
        try:
            df = self._dataset.load_full(primary)
        except Exception as e:
            return json.dumps({"error": f"Failed to load data: {e}"})

        import pandas as pd
        import numpy as np

        result = self._sandbox.run(
            source=code,
            globals={"pd": pd, "np": np, "df": df},
        )

        if result.timed_out:
            return json.dumps({"error": "EDA code timed out"})
        if result.returncode != 0:
            return json.dumps({
                "error": "EDA code failed",
                "stderr": result.stderr[:1000],
            })
        return json.dumps({
            "stdout": result.stdout,
            "elapsed_ms": round(result.elapsed_ms, 1),
        })

    def _tool_report_trial(self, reason: str, metadata: dict | None = None) -> str:
        try:
            entry = self._ledger.report(reason=reason, metadata=metadata or {})
            self._logger.log_trial_report(entry)
            return json.dumps({
                "trial_index": entry.trial_index,
                "remaining": self._ledger.remaining(),
            })
        except TrialBudgetExceeded as e:
            return json.dumps({"error": str(e)})

    def _tool_report_trials(self, trials: list[dict]) -> str:
        results = []
        try:
            for t in trials:
                reason = t.get("reason", "")
                metadata = t.get("metadata", {})
                entry = self._ledger.report(reason=reason, metadata=metadata)
                self._logger.log_trial_report(entry)
                results.append(entry.trial_index)
            return json.dumps({
                "status": "success",
                "trial_indices": results,
                "remaining": self._ledger.remaining(),
            })
        except TrialBudgetExceeded as e:
            return json.dumps({
                "error": str(e),
                "partially_reported_indices": results,
                "remaining": self._ledger.remaining(),
            })
        except Exception as e:
            return json.dumps({"error": str(e)})

    def _tool_create_hypothesis(self, title: str, description: str) -> str:
        hypothesis_id = str(uuid.uuid4())
        now = time.time()
        hypothesis = Hypothesis(
            hypothesis_id=hypothesis_id,
            title=title,
            description=description,
            status="active",
            notes="",
            created_at=now,
            updated_at=now,
        )
        self._hypotheses[hypothesis_id] = hypothesis
        self._logger.log_hypothesis_creation(hypothesis)
        return json.dumps({
            "status": "created",
            "hypothesis_id": hypothesis_id,
        })

    def _tool_update_hypothesis(self, hypothesis_id: str, status: str, notes: str | None = None) -> str:
        if hypothesis_id not in self._hypotheses:
            return json.dumps({"error": f"Unknown hypothesis_id: {hypothesis_id!r}"})

        if status not in ["active", "paused", "falsified"]:
            return json.dumps({"error": f"Invalid status: {status!r}. Must be active, paused, or falsified."})

        hypothesis = self._hypotheses[hypothesis_id]
        hypothesis.status = status
        if notes:
            if hypothesis.notes:
                hypothesis.notes += f"\n{notes}"
            else:
                hypothesis.notes = notes
        hypothesis.updated_at = time.time()

        self._logger.log_hypothesis_update(hypothesis)
        return json.dumps({
            "status": "updated",
            "hypothesis_id": hypothesis_id,
            "current_status": status,
        })

    def _tool_submit_strategy(self, source_code: str) -> str:
        artifact = self._strategy_registry.submit(source_code, self._run_id)
        self._artifacts[artifact.strategy_id] = artifact
        self._logger.log_strategy_submission(artifact)
        return json.dumps({
            "strategy_id": artifact.strategy_id,
            "source_hash": artifact.source_hash[:12] + "...",
        })

    def _tool_run_backtest(self, strategy_id: str) -> str:
        if strategy_id not in self._artifacts:
            return json.dumps({"error": f"Unknown strategy_id: {strategy_id!r}"})

        if self._ledger.is_exhausted():
            return json.dumps({"error": "Trial budget exhausted. Call submit_final() now."})

        # Charge trial
        try:
            entry = self._ledger.report(
                reason="backtest",
                metadata={"strategy_id": strategy_id},
            )
            self._logger.log_trial_report(entry)
        except TrialBudgetExceeded as e:
            return json.dumps({"error": str(e)})

        # Load data and strategy
        primary = self._task.asset_universe[0]
        try:
            data = self._dataset.load_full(primary)
        except Exception as e:
            return json.dumps({"error": f"Failed to load data: {e}"})

        artifact = self._artifacts[strategy_id]
        try:
            strategy = self._strategy_registry.load(artifact, data)
        except Exception as e:
            return json.dumps({"error": f"Strategy failed to load/validate: {e}"})

        try:
            result = self._backtest_engine.run(strategy, data, artifact)
        except Exception as e:
            return json.dumps({"error": f"Backtest failed: {e}"})

        # Return trimmed summary (no equity curve)
        return json.dumps({
            "strategy_id": result.strategy_id,
            "sharpe": round(result.sharpe, 4),
            "annual_return": round(result.annual_return, 4),
            "max_drawdown": round(result.max_drawdown, 4),
            "win_rate": round(result.win_rate, 4),
            "n_trades": result.n_trades,
            "elapsed_ms": round(result.elapsed_ms, 1),
            "trials_remaining": self._ledger.remaining(),
        })

    def _tool_submit_final(self, strategy_id: str) -> str:
        if strategy_id not in self._artifacts:
            return json.dumps({"error": f"Unknown strategy_id: {strategy_id!r}"})
        self._final_strategy_id = strategy_id
        return json.dumps({
            "status": "accepted",
            "strategy_id": strategy_id,
            "message": "Final strategy recorded. The research loop will now end.",
        })

    def _tool_checkpoint(self, strategy_id: str) -> str:
        if strategy_id not in self._artifacts:
            return json.dumps({"error": f"Unknown strategy_id: {strategy_id!r}"})
        self._checkpoint_strategy_id = strategy_id
        self._logger.log_event("checkpoint", {"strategy_id": strategy_id})
        return json.dumps({
            "status": "checkpointed",
            "strategy_id": strategy_id,
            "message": (
                "Strategy saved as your current best fallback. "
                "The research loop continues. "
                "Use remaining budget to explore new, independent signal families."
            ),
        })


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _accumulate_token_usage(
    current: TokenUsage,
    prompt_tokens: int,
    completion_tokens: int,
) -> TokenUsage:
    """Add usage from one LLM call to the running TokenUsage."""
    return TokenUsage(
        input_tokens=current.input_tokens + prompt_tokens,
        output_tokens=current.output_tokens + completion_tokens,
        total_tokens=current.total_tokens + prompt_tokens + completion_tokens,
    )


def _trim_context(messages: list[dict]) -> list[dict]:
    """
    Trim older messages safely to reduce context size.

    Preservation policy:
        - Always keep: system message + first user message
        - Keep a safe suffix of the rolling conversation that doesn't split
          any assistant tool calls from their corresponding tool results.
    """
    if len(messages) <= 6:
        return messages

    preamble = messages[:2]  # system + first user
    conversation = messages[2:]

    # Keep a maximum target of messages from the end
    target_keep = 12
    if len(conversation) <= target_keep:
        return messages

    cut_idx = len(conversation) - target_keep

    # Walk forward to find a valid cut point that doesn't break API constraints:
    # 1. The first message in the kept slice must not be a 'tool' message.
    # 2. The message immediately preceding it must not be an assistant message with tool calls.
    while cut_idx < len(conversation):
        msg = conversation[cut_idx]
        prev_msg = conversation[cut_idx - 1] if cut_idx > 0 else None

        if msg.get("role") == "tool":
            cut_idx += 1
            continue

        if prev_msg and prev_msg.get("role") == "assistant" and prev_msg.get("tool_calls"):
            # Include the assistant message to avoid dangling tool results
            cut_idx -= 1
            continue

        break

    if cut_idx >= len(conversation):
        return messages

    return preamble + conversation[cut_idx:]

