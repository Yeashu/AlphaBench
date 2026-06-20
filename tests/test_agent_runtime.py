"""Tests for AgentRuntime hypothesis tracking tools."""

import json
from unittest.mock import MagicMock
import pytest

from alphabench.agent_runtime import AgentRuntime
from alphabench.contracts import AgentConfig, TaskDefinition, Hypothesis


@pytest.fixture
def runtime():
    config = AgentConfig(
        model_name="test-model",
        max_turns=10,
        max_trials=5,
        temperature=0.2,
        max_context_tokens=10000,
        system_prompt_version="v3.0.0",
    )
    task = TaskDefinition(
        task_id="test-task",
        market="crypto",
        asset_universe=["BTC-USDT"],
        dataset_version="v1",
        train_start="2021-01-01",
        train_end="2021-12-31",
        oos_start="2022-01-01",
        oos_end="2022-12-31",
    )
    
    # Mock all other services
    dataset_service = MagicMock()
    sandbox = MagicMock()
    ledger = MagicMock()
    logger = MagicMock()
    registry = MagicMock()
    backtest_engine = MagicMock()
    strategy_registry = MagicMock()
    openai_client = MagicMock()

    return AgentRuntime(
        run_id="test-run",
        config=config,
        task=task,
        dataset_service=dataset_service,
        sandbox=sandbox,
        ledger=ledger,
        logger=logger,
        registry=registry,
        backtest_engine=backtest_engine,
        strategy_registry=strategy_registry,
        openai_client=openai_client,
    )


def test_create_hypothesis(runtime):
    res = runtime._tool_create_hypothesis(
        title="RSI Mean Reversion",
        description="Buy oversold RSI, sell overbought RSI."
    )
    res_dict = json.loads(res)
    assert res_dict["status"] == "created"
    assert "hypothesis_id" in res_dict
    
    h_id = res_dict["hypothesis_id"]
    assert h_id in runtime._hypotheses
    
    hypothesis = runtime._hypotheses[h_id]
    assert hypothesis.title == "RSI Mean Reversion"
    assert hypothesis.description == "Buy oversold RSI, sell overbought RSI."
    assert hypothesis.status == "active"
    assert hypothesis.notes == ""
    assert hypothesis.created_at > 0
    assert hypothesis.updated_at > 0

    runtime._logger.log_hypothesis_creation.assert_called_once_with(hypothesis)


def test_update_hypothesis(runtime):
    # First create
    res = runtime._tool_create_hypothesis(
        title="Trend Following",
        description="Buy when price > 200 SMA."
    )
    h_id = json.loads(res)["hypothesis_id"]

    # Update to paused with notes
    update_res = runtime._tool_update_hypothesis(
        hypothesis_id=h_id,
        status="paused",
        notes="Backtest showed high drawdown. Pausing to investigate."
    )
    update_dict = json.loads(update_res)
    assert update_dict["status"] == "updated"
    assert update_dict["current_status"] == "paused"

    hypothesis = runtime._hypotheses[h_id]
    assert hypothesis.status == "paused"
    assert hypothesis.notes == "Backtest showed high drawdown. Pausing to investigate."

    # Update to falsified with more notes
    update_res_2 = runtime._tool_update_hypothesis(
        hypothesis_id=h_id,
        status="falsified",
        notes="Falsified. Moving average crossovers lag too much in chop."
    )
    
    assert hypothesis.status == "falsified"
    assert "lag too much" in hypothesis.notes
    assert "Pausing to investigate." in hypothesis.notes  # notes should accumulate

    runtime._logger.log_hypothesis_update.assert_called()
    assert runtime._logger.log_hypothesis_update.call_count == 2


def test_update_hypothesis_invalid_params(runtime):
    # Update unknown hypothesis
    res = runtime._tool_update_hypothesis(
        hypothesis_id="unknown-id",
        status="paused"
    )
    assert "error" in json.loads(res)

    # First create
    res_create = runtime._tool_create_hypothesis(title="T", description="D")
    h_id = json.loads(res_create)["hypothesis_id"]

    # Update with invalid status
    res_invalid = runtime._tool_update_hypothesis(
        hypothesis_id=h_id,
        status="invalid_status"
    )
    assert "error" in json.loads(res_invalid)
