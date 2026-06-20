"""
config.py
---------
Loaders for AgentConfig and TaskDefinition from JSON files.
Falls back to sensible MVP defaults when fields are missing.

Environment variables (loaded from .env if present):
    LLM_MODEL_NAME   — overrides model_name in the agent config JSON.
    OPENAI_API_KEY   — required; read in run_benchmark.py.
    OPENAI_BASE_URL  — optional; read in run_benchmark.py.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

from .contracts import AgentConfig, TaskDefinition

# Load .env from the project root (two levels up from this file).
# override=False so already-set shell env vars take precedence.
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=False)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_AGENT_CONFIG = {
    "model_name": "gpt-4o",
    "max_turns": 40,
    "max_trials": 200,
    "temperature": 0.2,
    "max_context_tokens": 100_000,
    "system_prompt_version": "v1.0.0",
}

_DEFAULT_TASK = {
    "task_id": "crypto_spot_v1",
    "market": "crypto_spot",
    "asset_universe": ["BTC-USDT"],
    "dataset_version": "v1",
    "train_start": "2021-01-01",
    "train_end": "2024-12-31",
    "oos_start": "2025-01-01",
    "oos_end": "2025-12-31",
}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_agent_config(path: Path | None = None) -> AgentConfig:
    """
    Load AgentConfig from a JSON file, then apply .env overrides.

    Priority (highest → lowest):
        1. Shell environment variables
        2. .env file values
        3. JSON config file values
        4. Built-in defaults

    Supported env overrides:
        LLM_MODEL_NAME  →  model_name
    """
    data = dict(_DEFAULT_AGENT_CONFIG)
    if path is not None:
        override = json.loads(Path(path).read_text())
        data.update(override)

    # Apply environment overrides (loaded from .env by load_dotenv above)
    if model_env := os.environ.get("LLM_MODEL_NAME"):
        data["model_name"] = model_env

    return AgentConfig(**data)


def load_task_definition(path: Path | None = None) -> TaskDefinition:
    """
    Load TaskDefinition from a JSON file.
    Missing keys fall back to _DEFAULT_TASK.
    If path is None, returns the default task.
    """
    data = dict(_DEFAULT_TASK)
    if path is not None:
        override = json.loads(Path(path).read_text())
        data.update(override)
    return TaskDefinition(**data)
