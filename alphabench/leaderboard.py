"""
leaderboard.py
--------------
Reads evaluation logs, filters to passing strategies, builds the leaderboard.

Usage (separate CLI — not auto-run after each benchmark run):

    python -m alphabench.leaderboard build \\
        --log-dir logs/ \\
        --backtest-db outputs/backtests.db \\
        --output-dir outputs/

Leaderboard ranking: OOS Sharpe (descending), passes_gates == True only.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LeaderboardRow:
    rank: int
    run_id: str
    task_id: str
    model_name: str
    oos_sharpe: float
    dsr: float
    permutation_pvalue: float
    trial_count: int
    backtest_count: int
    total_turns: int
    total_tokens: int
    estimated_cost_usd: float
    prompt_version: str
    strategy_id: str


class LeaderboardService:
    """
    Builds the leaderboard from NDJSON log files and BacktestRegistry.

    Parameters
    ----------
    log_dir:
        Directory containing per-run *.ndjson files.
    backtest_db:
        Path to the BacktestRegistry SQLite database.
    output_dir:
        Directory where leaderboard.db and leaderboard.html are written.
    """

    def __init__(
        self,
        log_dir: Path,
        backtest_db: Path,
        output_dir: Path,
    ) -> None:
        self._log_dir = Path(log_dir)
        self._backtest_db = Path(backtest_db)
        self._output_dir = Path(output_dir)

    def build(self) -> list[LeaderboardRow]:
        """
        Scan logs, extract passing runs, rank by OOS Sharpe, write outputs.

        Returns the ranked list of LeaderboardRows.
        """
        rows = self._collect_rows()
        # Filter to passing strategies only
        rows = [r for r in rows if r.oos_sharpe > 0.0]  # passes_gates already filtered
        # Rank by OOS Sharpe descending
        rows.sort(key=lambda r: r.oos_sharpe, reverse=True)
        for i, row in enumerate(rows, start=1):
            row.rank = i

        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._write_sqlite(rows)
        self._write_html(rows)
        return rows

    # ------------------------------------------------------------------
    # Collection
    # ------------------------------------------------------------------

    def _collect_rows(self) -> list[LeaderboardRow]:
        """Parse all NDJSON log files and extract qualifying runs."""
        rows: list[LeaderboardRow] = []
        log_files = list(self._log_dir.glob("*.ndjson"))

        for log_file in log_files:
            run_data = _parse_log(log_file)
            if run_data is None:
                continue
            manifest, eval_result = run_data
            if not eval_result.get("passes_gates"):
                continue

            backtest_count = _get_backtest_count(self._backtest_db, manifest["run_id"])

            rows.append(
                LeaderboardRow(
                    rank=0,  # filled after sorting
                    run_id=manifest["run_id"],
                    task_id=manifest.get("task", {}).get("task_id", "unknown"),
                    model_name=manifest.get("config", {}).get("model_name", "unknown"),
                    oos_sharpe=eval_result.get("oos_sharpe", 0.0),
                    dsr=eval_result.get("dsr", 0.0),
                    permutation_pvalue=eval_result.get("permutation_pvalue", 1.0),
                    trial_count=manifest.get("total_trials", 0),
                    backtest_count=backtest_count,
                    total_turns=manifest.get("total_turns", 0),
                    total_tokens=manifest.get("cost_summary", {}).get("total_tokens", 0),
                    estimated_cost_usd=manifest.get("cost_summary", {}).get("estimated_cost_usd", 0.0),
                    prompt_version=manifest.get("prompt_version", {}).get("version", "unknown"),
                    strategy_id=eval_result.get("strategy_id", "unknown"),
                )
            )

        return rows

    # ------------------------------------------------------------------
    # Output writers
    # ------------------------------------------------------------------

    def _write_sqlite(self, rows: list[LeaderboardRow]) -> None:
        db_path = self._output_dir / "leaderboard.db"
        conn = sqlite3.connect(str(db_path))
        with conn:
            conn.execute("DROP TABLE IF EXISTS leaderboard")
            conn.execute("""
                CREATE TABLE leaderboard (
                    rank INTEGER,
                    run_id TEXT,
                    task_id TEXT,
                    model_name TEXT,
                    oos_sharpe REAL,
                    dsr REAL,
                    permutation_pvalue REAL,
                    trial_count INTEGER,
                    backtest_count INTEGER,
                    total_turns INTEGER,
                    total_tokens INTEGER,
                    estimated_cost_usd REAL,
                    prompt_version TEXT,
                    strategy_id TEXT
                )
            """)
            conn.executemany(
                """
                INSERT INTO leaderboard VALUES
                (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                [
                    (
                        r.rank, r.run_id, r.task_id, r.model_name,
                        r.oos_sharpe, r.dsr, r.permutation_pvalue,
                        r.trial_count, r.backtest_count, r.total_turns,
                        r.total_tokens, r.estimated_cost_usd,
                        r.prompt_version, r.strategy_id,
                    )
                    for r in rows
                ],
            )
        conn.close()

    def _write_html(self, rows: list[LeaderboardRow]) -> None:
        html_path = self._output_dir / "leaderboard.html"
        html = _render_html(rows)
        html_path.write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_log(log_file: Path) -> tuple[dict, dict] | None:
    """
    Parse a single NDJSON log file.
    Returns (run_complete payload, eval_result payload) or None if not found.
    """
    manifest: dict | None = None
    eval_result: dict | None = None

    try:
        for line in log_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("event_type") == "run_complete":
                manifest = event["payload"]
            elif event.get("event_type") == "eval_result":
                eval_result = event["payload"]
    except Exception:
        return None

    if manifest is None or eval_result is None:
        return None
    return manifest, eval_result


def _get_backtest_count(db_path: Path, run_id: str) -> int:
    """Count backtests for a run from the BacktestRegistry database."""
    if not db_path.exists():
        return 0
    try:
        conn = sqlite3.connect(str(db_path))
        row = conn.execute(
            "SELECT COUNT(*) FROM backtests WHERE run_id = ?", (run_id,)
        ).fetchone()
        conn.close()
        return int(row[0]) if row else 0
    except Exception:
        return 0


def _render_html(rows: list[LeaderboardRow]) -> str:
    """Render a clean HTML leaderboard page."""
    rows_html = ""
    for r in rows:
        rows_html += f"""
        <tr>
            <td>{r.rank}</td>
            <td title="{r.run_id}">{r.run_id[:8]}…</td>
            <td>{r.task_id}</td>
            <td>{r.model_name}</td>
            <td><strong>{r.oos_sharpe:.4f}</strong></td>
            <td>{r.permutation_pvalue:.4f}</td>
            <td>{r.dsr:.4f}</td>
            <td>{r.trial_count}</td>
            <td>{r.backtest_count}</td>
            <td>{r.total_turns}</td>
            <td>{r.total_tokens:,}</td>
            <td>${r.estimated_cost_usd:.4f}</td>
            <td>{r.prompt_version}</td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AlphaBench Leaderboard</title>
  <style>
    body {{ font-family: system-ui, sans-serif; padding: 2rem; background: #0f1117; color: #e2e8f0; }}
    h1 {{ color: #a78bfa; margin-bottom: 0.25rem; }}
    p.subtitle {{ color: #64748b; margin-top: 0; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 1.5rem; font-size: 0.9rem; }}
    th {{ background: #1e1b4b; color: #a78bfa; padding: 0.75rem 1rem; text-align: left; border-bottom: 2px solid #3730a3; }}
    td {{ padding: 0.6rem 1rem; border-bottom: 1px solid #1e293b; }}
    tr:hover td {{ background: #1e293b; }}
    tr:nth-child(1) td:nth-child(5) {{ color: #fbbf24; }}
    .empty {{ color: #475569; text-align: center; padding: 3rem; }}
  </style>
</head>
<body>
  <h1>🏆 AlphaBench Leaderboard</h1>
  <p class="subtitle">Ranked by out-of-sample Sharpe ratio. Only strategies passing validation gates are shown.</p>
  <table>
    <thead>
      <tr>
        <th>#</th>
        <th>Run ID</th>
        <th>Task</th>
        <th>Model</th>
        <th>OOS Sharpe</th>
        <th>Perm p-value</th>
        <th>DSR</th>
        <th>Trials</th>
        <th>Backtests</th>
        <th>Turns</th>
        <th>Tokens</th>
        <th>Cost (USD)</th>
        <th>Prompt</th>
      </tr>
    </thead>
    <tbody>
      {"".join([rows_html]) if rows else '<tr><td colspan="13" class="empty">No qualifying strategies yet.</td></tr>'}
    </tbody>
  </table>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m alphabench.leaderboard",
        description="Build the AlphaBench leaderboard from run logs.",
    )
    subparsers = parser.add_subparsers(dest="command")

    build_parser = subparsers.add_parser("build", help="Build leaderboard from logs")
    build_parser.add_argument("--log-dir", type=Path, default=Path("logs"), help="Directory with NDJSON logs")
    build_parser.add_argument("--backtest-db", type=Path, default=Path("outputs/backtests.db"), help="BacktestRegistry SQLite DB")
    build_parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Output directory for leaderboard files")

    args = parser.parse_args()

    if args.command == "build":
        svc = LeaderboardService(args.log_dir, args.backtest_db, args.output_dir)
        rows = svc.build()
        print(f"Leaderboard built: {len(rows)} qualifying entries.")
        print(f"  → {args.output_dir / 'leaderboard.db'}")
        print(f"  → {args.output_dir / 'leaderboard.html'}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
