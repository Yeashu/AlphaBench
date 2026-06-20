"""
sandbox_executor.py
-------------------
Runs agent-written EDA code in an isolated subprocess.

Interface change from LLD v0.2:
    SandboxExecutor.run(source, globals)

where globals = {"pd": pd, "np": np, "df": dataframe}

The runtime serializes the DataFrame to a temp parquet file, then injects
a header into the agent's source that loads it back. The agent's code can
use `df`, `pd`, and `np` directly without any import statements.

Security layers:
    Layer 1: AST import whitelist on the agent's source only (not the injected header)
    Layer 2: subprocess timeout + trimmed environment
    Layer 3: Docker isolation (post-MVP)
"""

from __future__ import annotations

import ast
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Any

import pandas as pd

from .contracts import ExecutionResult


# ---------------------------------------------------------------------------
# Allowed imports in agent EDA code
# ---------------------------------------------------------------------------

ALLOWED_IMPORTS: frozenset[str] = frozenset(
    {
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "math",
        "statistics",
        "datetime",
        "collections",
        "itertools",
        "functools",
        "typing",
    }
)


class ImportChecker(ast.NodeVisitor):
    """AST visitor that raises ValueError on any disallowed import."""

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            top = alias.name.split(".")[0]
            if top not in ALLOWED_IMPORTS:
                raise ValueError(
                    f"Disallowed import: {alias.name!r}. "
                    f"Allowed top-level modules: {sorted(ALLOWED_IMPORTS)}"
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        top = (node.module or "").split(".")[0]
        if top not in ALLOWED_IMPORTS:
            raise ValueError(
                f"Disallowed import from: {node.module!r}. "
                f"Allowed top-level modules: {sorted(ALLOWED_IMPORTS)}"
            )
        self.generic_visit(node)


# ---------------------------------------------------------------------------
# SandboxExecutor
# ---------------------------------------------------------------------------


class SandboxExecutor:
    """
    Runs agent EDA code in a subprocess with injected globals.

    The agent's source is NOT modified. A header is prepended that:
      1. Imports pandas/numpy
      2. Loads the DataFrame from a temp parquet file
      3. Binds the names ``df``, ``pd``, ``np`` in the script's namespace

    The agent code itself must NOT contain any import statements beyond
    the ALLOWED_IMPORTS whitelist — this is checked via AST before execution.
    """

    def __init__(
        self,
        timeout: int = 30,
        max_output_chars: int = 4000,
    ) -> None:
        self.timeout = timeout
        self.max_output = max_output_chars

    def run(
        self,
        source: str,
        globals: dict[str, Any],
    ) -> ExecutionResult:
        """
        Execute *source* in a subprocess with *globals* injected.

        Parameters
        ----------
        source:
            Agent-written Python code. Must pass the import whitelist check.
        globals:
            Variables to inject into the script's namespace before execution.
            Supported keys: ``"pd"`` (pandas), ``"np"`` (numpy), ``"df"`` (DataFrame).
            Unknown keys are silently ignored.

        Returns
        -------
        ExecutionResult
        """
        # 1. AST check on agent source only
        try:
            tree = ast.parse(source)
            ImportChecker().visit(tree)
        except SyntaxError as e:
            return ExecutionResult(stdout="", stderr=str(e), returncode=1, elapsed_ms=0.0)
        except ValueError as e:
            return ExecutionResult(stdout="", stderr=str(e), returncode=1, elapsed_ms=0.0)

        # 2. Run in subprocess
        return self._run_subprocess(source, globals)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_subprocess(
        self,
        source: str,
        globals: dict[str, Any],
    ) -> ExecutionResult:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            # Serialize DataFrame to temp parquet if provided
            df: pd.DataFrame | None = globals.get("df")
            data_parquet = tmp_path / "data.parquet"
            if df is not None:
                df.reset_index().to_parquet(data_parquet, index=False)

            # Build injection header
            header = self._build_header(data_parquet if df is not None else None)

            # Write final script: header + agent source
            script = tmp_path / "eda.py"
            script.write_text(header + "\n" + source)

            # Execute in subprocess
            start = time.monotonic()
            timed_out = False
            try:
                result = subprocess.run(
                    [sys.executable, str(script)],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    env={
                        "PATH": "/usr/bin:/bin",
                        "MPLBACKEND": "Agg",  # headless matplotlib
                    },
                )
                rc = result.returncode
                out = result.stdout[-self.max_output:]
                err = result.stderr[-2000:]
            except subprocess.TimeoutExpired:
                rc, out, err, timed_out = -1, "", f"Timeout after {self.timeout}s", True

            elapsed = (time.monotonic() - start) * 1000
            return ExecutionResult(
                stdout=out,
                stderr=err,
                returncode=rc,
                elapsed_ms=elapsed,
                timed_out=timed_out,
            )

    @staticmethod
    def _build_header(data_parquet: Path | None) -> str:
        """
        Build a Python header that injects ``pd``, ``np``, and ``df``
        into the script namespace without requiring agent imports.
        """
        lines = [
            "# --- AlphaBench EDA runtime header (auto-injected) ---",
            "import pandas as pd",
            "import numpy as np",
        ]
        if data_parquet is not None:
            lines += [
                f"_df_raw = pd.read_parquet({str(data_parquet)!r})",
                "# Restore timestamp as DatetimeIndex if present",
                "if 'timestamp' in _df_raw.columns:",
                "    _df_raw['timestamp'] = pd.to_datetime(_df_raw['timestamp'], utc=True)",
                "    _df_raw = _df_raw.set_index('timestamp').sort_index()",
                "df = _df_raw",
                "del _df_raw",
            ]
        lines.append("# --- end of runtime header ---")
        return textwrap.dedent("\n".join(lines)) + "\n"
