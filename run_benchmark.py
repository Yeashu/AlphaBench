"""
run_benchmark.py
----------------
CLI entrypoint for a single AlphaBench run.

Usage:
    python run_benchmark.py \\
        --task-config config/default_task.json \\
        --agent-config config/default_agent.json \\
        --data-root data/ \\
        --oos-data-root data_hidden/v1/ \\
        --log-dir logs/ \\
        --backtest-db outputs/backtests.db

Environment variables:
    OPENAI_API_KEY      Required for OpenAI API access.
    OPENAI_BASE_URL     Optional: override base URL for compatible APIs.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from dataclasses import asdict
from pathlib import Path

import openai
from dotenv import load_dotenv

# Load .env from the project root before reading any env vars.
# override=False so real shell variables always take precedence.
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=False)

from alphabench.agent_runtime import AgentRuntime
from alphabench.backtest_engine import BacktestEngine
from alphabench.backtest_registry import BacktestRegistry
from alphabench.config import load_agent_config, load_task_definition
from alphabench.dataset_service import DatasetService
from alphabench.evaluation_engine import EvaluationEngine
from alphabench.run_logger import RunLogger
from alphabench.sandbox_executor import SandboxExecutor
from alphabench.strategy import StrategyRegistry
from alphabench.trial_ledger import TrialLedger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_benchmark.py",
        description="Run a single AlphaBench agent benchmark.",
    )
    parser.add_argument(
        "--task-config",
        type=Path,
        default=None,
        help="Path to task JSON config (default: built-in defaults)",
    )
    parser.add_argument(
        "--agent-config",
        type=Path,
        default=None,
        help="Path to agent JSON config (default: built-in defaults)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root directory for training data (default: data/)",
    )
    parser.add_argument(
        "--oos-data-root",
        type=Path,
        default=Path("data_hidden/v1"),
        help="Root directory for hidden OOS data (default: data_hidden/v1/)",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs"),
        help="Directory to write NDJSON run logs (default: logs/)",
    )
    parser.add_argument(
        "--backtest-db",
        type=Path,
        default=Path("outputs/backtests.db"),
        help="Path to BacktestRegistry SQLite DB (default: outputs/backtests.db)",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional explicit run ID (default: auto-generated UUID4)",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=1000,
        help="Number of permutations for significance test (default: 1000)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # --- Load config ---
    task = load_task_definition(args.task_config)
    config = load_agent_config(args.agent_config)
    run_id = args.run_id or str(uuid.uuid4())

    print(f"\n{'=' * 60}")
    print(f"AlphaBench Run")
    print(f"  run_id   : {run_id}")
    print(f"  task     : {task.task_id}")
    print(f"  model    : {config.model_name}")
    print(f"  trials   : {config.max_trials}")
    print(f"  turns    : {config.max_turns}")
    print(f"{'=' * 60}\n")

    # --- OpenAI client (values sourced from .env or shell environment) ---
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. "
            "Add it to your .env file or export it in your shell."
        )
    base_url = os.environ.get("OPENAI_BASE_URL") or None  # None → OpenAI default
    client = openai.OpenAI(api_key=api_key, base_url=base_url)

    # --- Services ---
    logger = RunLogger(run_id=run_id, log_dir=args.log_dir)
    dataset = DatasetService(data_root=args.data_root, task=task)
    sandbox = SandboxExecutor(timeout=30, max_output_chars=4000)
    ledger = TrialLedger(max_trials=config.max_trials)
    bt_registry = BacktestRegistry(db_path=args.backtest_db)
    strategy_registry = StrategyRegistry()
    backtest_engine = BacktestEngine(
        trial_ledger=ledger,
        logger=logger,
        registry=bt_registry,
    )

    runtime = AgentRuntime(
        run_id=run_id,
        config=config,
        task=task,
        dataset_service=dataset,
        sandbox=sandbox,
        ledger=ledger,
        logger=logger,
        registry=bt_registry,
        backtest_engine=backtest_engine,
        strategy_registry=strategy_registry,
        openai_client=client,
    )

    # --- Run agent loop ---
    print("Starting agent loop...")
    t0 = time.monotonic()
    manifest = runtime.run()
    elapsed = time.monotonic() - t0
    print(f"Agent loop complete in {elapsed:.1f}s")
    print(f"  turns    : {manifest.total_turns}")
    print(f"  trials   : {manifest.total_trials}")
    print(f"  backtests: {manifest.total_backtests}")
    print(f"  tokens   : {manifest.cost_summary.total_tokens:,}")
    print(f"  cost     : ${manifest.cost_summary.estimated_cost_usd:.4f}")
    print(f"  final    : {manifest.final_strategy_id or 'none submitted'}")

    # --- Evaluation ---
    eval_result = None
    final_artifact = runtime.get_final_artifact()

    if final_artifact is not None:
        print("\nRunning evaluation on hidden OOS data...")
        try:
            eval_engine = EvaluationEngine(
                oos_data_root=args.oos_data_root,
                task=task,
                openai_client=client,
                model_name=config.model_name,
            )
            in_sample_data = dataset.load_full(task.asset_universe[0])
            run_log_path = args.log_dir / f"{run_id}.ndjson"
            eval_result = eval_engine.evaluate(
                artifact=final_artifact,
                in_sample_data=in_sample_data,
                trial_count=manifest.total_trials,
                n_permutations=args.n_permutations,
                run_log_path=run_log_path,
            )
            logger.log_eval(eval_result)
            print(f"  passes_gates      : {eval_result.passes_gates}")
            print(f"  permutation pvalue: {eval_result.permutation_pvalue:.4f}")
            print(f"  DSR               : {eval_result.dsr:.4f}")
            print(f"  reported trials   : {eval_result.reported_trials}")
            print(f"  audited trials    : {eval_result.audited_trials}")
            print(f"  OOS Sharpe        : {eval_result.oos_sharpe:.4f}")
            if eval_result.gate_failures:
                print(f"  gate failures     : {eval_result.gate_failures}")
        except Exception as e:
            print(f"  Evaluation failed: {e}")
    else:
        print("\nNo final strategy submitted — skipping evaluation.")

    # --- Finalize ---
    manifest.eval_result = eval_result
    manifest.finished_at = time.time()
    logger.finalize(manifest)

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print("Run complete.")
    print(f"  Log   : {args.log_dir / run_id}.ndjson")
    print(f"  DB    : {args.backtest_db}")
    print(f"{'=' * 60}\n")
    print("To update the leaderboard:")
    print(f"  python -m alphabench.leaderboard build --log-dir {args.log_dir} --backtest-db {args.backtest_db}")


if __name__ == "__main__":
    main()
