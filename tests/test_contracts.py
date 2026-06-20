"""Tests for alphabench.contracts — all dataclasses construct correctly."""

from alphabench.contracts import (
    AgentConfig,
    AssetMetadata,
    BacktestRecord,
    BacktestResult,
    EvalResult,
    ExecutionResult,
    PromptVersion,
    RunManifest,
    StrategyArtifact,
    TaskDefinition,
    TokenUsage,
    TrialEntry,
)


def test_task_definition():
    t = TaskDefinition(
        task_id="t1", market="crypto", asset_universe=["BTC-USDT"],
        dataset_version="v1", train_start="2021-01-01", train_end="2025-12-31",
        oos_start="2026-01-01", oos_end="2026-12-31",
    )
    assert t.task_id == "t1"
    assert "BTC-USDT" in t.asset_universe


def test_agent_config():
    c = AgentConfig(
        model_name="gpt-4o", max_turns=40, max_trials=10,
        temperature=0.2, max_context_tokens=100_000, system_prompt_version="v1.0.0",
    )
    assert c.max_trials == 10
    assert c.temperature == 0.2


def test_backtest_record():
    r = BacktestRecord(
        backtest_id="b1", run_id="r1", strategy_id="s1", strategy_hash="abc",
        trial_index=0, sharpe=1.5, annual_return=0.3, max_drawdown=0.1,
        n_trades=50, created_at="2026-01-01T00:00:00+00:00",
    )
    assert r.sharpe == 1.5


def test_prompt_version():
    pv = PromptVersion(prompt_id="system_prompt", version="v1.0.0", sha256="abc", created_at="now")
    assert pv.version == "v1.0.0"


def test_token_usage():
    tu = TokenUsage(input_tokens=100, output_tokens=50, total_tokens=150)
    assert tu.total_tokens == 150


def test_run_manifest_structure():
    task = TaskDefinition(
        task_id="t1", market="crypto", asset_universe=["BTC-USDT"],
        dataset_version="v1", train_start="2021-01-01", train_end="2025-12-31",
        oos_start="2026-01-01", oos_end="2026-12-31",
    )
    config = AgentConfig(
        model_name="gpt-4o", max_turns=40, max_trials=10,
        temperature=0.2, max_context_tokens=100_000, system_prompt_version="v1.0.0",
    )
    pv = PromptVersion(prompt_id="system_prompt", version="v1.0.0", sha256="abc", created_at="now")
    tu = TokenUsage(0, 0, 0)

    manifest = RunManifest(
        run_id="r1", task=task, config=config, prompt_version=pv, token_usage=tu,
        started_at=0.0, finished_at=1.0, total_trials=5, total_backtests=3,
        total_turns=10, final_strategy_id="s1", eval_result=None,
    )
    assert manifest.task.task_id == "t1"
    assert manifest.config.model_name == "gpt-4o"
    assert manifest.final_strategy_id == "s1"
    assert manifest.eval_result is None
