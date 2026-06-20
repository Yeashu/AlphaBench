"""
evaluation_engine.py
--------------------
Hidden out-of-sample evaluation and statistical validation.

Two-stage process (per EVALUATION_SPEC.md §2.2):
    1. In-sample validation gates
    2. Hidden OOS Sharpe ranking

MVP gates:
    - permutation p-value <= 0.05  (active)
    - DSR >= 0.30                  (placeholder — NOT used for gating in MVP)

DSR status:
    compute_dsr() returns 0.0 as a placeholder.
    TODO: Implement Bailey & López de Prado (2014) Deflated Sharpe Ratio.
    The gate check is intentionally skipped until the formula is implemented.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import vectorbt as vbt

from .contracts import EvalResult, StrategyArtifact, TaskDefinition
from .strategy import BaseStrategy, StrategyRegistry


class EvaluationEngine:
    """
    Runs statistical validation and hidden OOS evaluation for a submitted strategy.

    The engine has access to both in-sample data (for permutation testing)
    and hidden OOS data (for final scoring). Neither is accessible to the agent.

    Parameters
    ----------
    oos_data_root:
        Path to the hidden OOS data directory (e.g. data_hidden/v1/).
        Must contain the same asset parquet files as the training root.
    task:
        TaskDefinition describing the OOS date range and asset universe.
    """

    def __init__(
        self,
        oos_data_root: Path,
        task: TaskDefinition,
        openai_client: openai.OpenAI | None = None,
        model_name: str | None = None,
    ) -> None:
        self._oos_root = Path(oos_data_root)
        self._task = task
        self._strategy_registry = StrategyRegistry()
        self._client = openai_client
        self._model_name = model_name

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def evaluate(
        self,
        artifact: StrategyArtifact,
        in_sample_data: pd.DataFrame,
        trial_count: int,
        n_permutations: int = 1000,
        run_log_path: Path | None = None,
    ) -> EvalResult:
        """
        Run full evaluation for a submitted strategy.

        Steps:
            1. Load the strategy from source code
            2. Run permutation significance test on in-sample data
            3. Run trial audit on run NDJSON logs (to find actual in-sample trials)
            4. Compute Deflated Sharpe Ratio (DSR) using audited trial count
            5. Check validation gates (permutation pvalue, DSR, and minimum trade count)
            6. If gates pass: compute OOS Sharpe on hidden data
            7. Return EvalResult
        """
        # Load and validate strategy
        strategy = self._strategy_registry.load(artifact, in_sample_data)
        signals = strategy.generate_signals()
        close = in_sample_data["close"]

        # --- Permutation test ---
        pvalue, null_sharpes = self.run_permutation_test(signals, close, n_permutations)

        # --- LLM Trial Audit ---
        audited_trials = trial_count
        if pvalue <= 0.05 and run_log_path is not None:
            try:
                audited_trials = self.audit_trials(run_log_path)
            except Exception:
                # Fallback to reported trial count if auditor fails
                pass

        # --- DSR ---
        observed_sharpe = _compute_sharpe(signals, close)
        dsr = self.compute_dsr(
            observed_sharpe=observed_sharpe,
            null_sharpes=null_sharpes,
            n_trials=audited_trials,
            T=len(close),
        )

        # --- Count trades for gating ---
        try:
            pf = vbt.Portfolio.from_signals(
                close=close,
                entries=(signals == 1),
                exits=(signals == 0),
                freq="1D",
            )
            n_trades = int(pf.trades.count())
            del pf
        except Exception:
            n_trades = 0

        # --- Gate checks ---
        gate_failures: list[str] = []
        if pvalue > 0.05:
            gate_failures.append(f"permutation_pvalue={pvalue:.4f} > 0.05")
        if dsr < 0.30:
            gate_failures.append(f"dsr={dsr:.4f} < 0.30")
        if n_trades < 30:
            gate_failures.append(f"n_trades={n_trades} < 30")

        passes_gates = len(gate_failures) == 0

        # --- OOS Sharpe (only computed if gates pass) ---
        oos_sharpe = 0.0
        if passes_gates:
            oos_sharpe = self._compute_oos_sharpe(strategy, artifact)

        return EvalResult(
            run_id=artifact.run_id,
            strategy_id=artifact.strategy_id,
            oos_sharpe=oos_sharpe,
            dsr=dsr,
            permutation_pvalue=pvalue,
            passes_gates=passes_gates,
            gate_failures=gate_failures,
            reported_trials=trial_count,
            audited_trials=audited_trials,
        )

    def run_permutation_test(
        self,
        signals: pd.Series,
        close: pd.Series,
        n_permutations: int = 1000,
    ) -> tuple[float, list[float]]:
        """
        Signal permutation significance test (LLD §12.4).

        Shuffles the signal vector *n_permutations* times, preserving the
        original signal distribution. The empirical p-value is the fraction
        of permutations whose Sharpe meets or exceeds the observed Sharpe.

        Parameters
        ----------
        signals:
            Original strategy signal vector (0/1 pd.Series).
        close:
            Price series aligned with signals.
        n_permutations:
            Number of shuffle iterations.

        Returns
        -------
        (pvalue, null_sharpe_distribution)
        """
        observed_sharpe = _compute_sharpe(signals, close)
        signal_array = signals.to_numpy().copy()
        null_sharpes: list[float] = []

        rng = np.random.default_rng(seed=42)  # reproducible null distribution

        for _ in range(n_permutations):
            shuffled = signal_array.copy()
            rng.shuffle(shuffled)
            shuffled_series = pd.Series(shuffled, index=signals.index)
            s = _compute_sharpe(shuffled_series, close)
            null_sharpes.append(s)

        beats_or_ties = sum(1 for s in null_sharpes if s >= observed_sharpe)
        pvalue = beats_or_ties / n_permutations if n_permutations > 0 else 1.0
        return pvalue, null_sharpes

    def compute_dsr(
        self,
        observed_sharpe: float,
        null_sharpes: list[float],
        n_trials: int,
        T: int,
    ) -> float:
        """
        Compute Deflated Sharpe Ratio (DSR) using Bailey & López de Prado (2014).
        """
        if n_trials <= 1 or not null_sharpes:
            return 1.0 if observed_sharpe > 0.0 else 0.0

        sigma_sr = np.std(null_sharpes)
        if sigma_sr == 0.0:
            sigma_sr = 0.1  # defensive fallback to prevent division by zero

        # Euler-Mascheroni constant
        gamma = 0.5772156649
        
        # Expected maximum Sharpe ratio under null hypothesis (Euler-Mascheroni approximation)
        try:
            from scipy.stats import norm
            ppf_1 = norm.ppf(1.0 - 1.0 / n_trials)
            ppf_2 = norm.ppf(1.0 - 1.0 / (n_trials * np.e))
            expected_max = (1.0 - gamma) * ppf_1 + gamma * ppf_2
            sr_0 = sigma_sr * expected_max
        except Exception:
            sr_0 = 0.0

        # Use the provided daily return sequence length T
        T_val = T if T > 1 else 1000

        # Compute standard deviation of the Sharpe Ratio assuming normal returns (skew=0, kurtosis=3)
        var_sr = (1.0 + 0.5 * (observed_sharpe ** 2)) / (T_val - 1)
        std_sr = np.sqrt(var_sr)

        from scipy.stats import norm
        dsr_value = float(norm.cdf((observed_sharpe - sr_0) / std_sr))
        return dsr_value if np.isfinite(dsr_value) else 0.0

    def audit_trials(self, run_log_path: Path) -> int:
        """
        Audit the run log to count the true number of target-aware trials.
        Reads all `run_eda` calls and analyzes them using the LLM client,
        then adds the backtest trials.
        """
        if not run_log_path.exists():
            return 0

        import json
        import re

        backtest_count = 0
        reported_eda_count = 0
        eda_scripts = []

        with open(run_log_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    event = json.loads(line)
                    if event.get("event_type") == "trial_report":
                        payload = event.get("payload", {})
                        reason = payload.get("reason")
                        if reason == "backtest":
                            backtest_count += 1
                        else:
                            reported_eda_count += 1
                    elif event.get("event_type") == "tool_call":
                        payload = event.get("payload", {})
                        if payload.get("name") == "run_eda":
                            code = payload.get("args", {}).get("code")
                            if code:
                                eda_scripts.append(code)
                except Exception:
                    continue

        if not eda_scripts:
            return backtest_count + reported_eda_count

        if self._client is None or self._model_name is None:
            audited_eda_count = self._heuristic_trial_count(eda_scripts)
        else:
            formatted_eda = ""
            for i, code in enumerate(eda_scripts):
                formatted_eda += f"\n--- Script {i+1} ---\n{code}\n"

            prompt = f"""You are a strict Quant Research Trial Auditor.
Your job is to analyze the Python code executed by a research agent and count the number of parameter configurations/variations that were evaluated against target performance feedback (Sharpe, Returns, PnL, Drawdowns, etc.).

A "trial" is defined as any evaluation of a specific parameter setting.
For example:
- A loop `for w in [10, 20, 30]:` evaluating moving averages counts as 3 trials.
- A grid search testing 5 fast MAs and 5 slow MAs counts as 25 trials.
- Running a single strategy with fixed parameters counts as 1 trial.
- Only count trial loops/configs that calculate performance metrics (PnL, returns, Sharpe, drawdown, etc.).
- Basic data plotting or printing shape/describe WITHOUT checking returns does not count as target-aware trials.

Analyze the following Python scripts executed during the research process:
{formatted_eda}

Provide your step-by-step reasoning identifying the parameter sweeps and evaluations.
Then, output the final total count of target-aware trial configurations evaluated in the format:
[TOTAL_TRIALS: X]
where X is the integer count."""

            audited_eda_count = 0
            try:
                response = self._client.chat.completions.create(
                    model=self._model_name,
                    messages=[
                        {"role": "system", "content": "You are a precise trial auditor."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                )
                content = response.choices[0].message.content or ""
                match = re.search(r"\[TOTAL_TRIALS:\s*(\d+)\]", content)
                if match:
                    audited_eda_count = int(match.group(1))
            except Exception:
                audited_eda_count = self._heuristic_trial_count(eda_scripts)

        return backtest_count + max(reported_eda_count, audited_eda_count)

    def _heuristic_trial_count(self, eda_scripts: list[str]) -> int:
        """Simple rule-based fallback if LLM client is unavailable or fails."""
        import re
        total = 0
        for code in eda_scripts:
            if any(term in code for term in ["pct_change", "Sharpe", "pnl", "cumprod", "vectorbt", "vbt"]):
                # Find list structures
                list_matches = re.findall(r"\[([\d\s,]+)\]", code)
                for m in list_matches:
                    vals = [v.strip() for v in m.split(",") if v.strip()]
                    if len(vals) > 1:
                        total += len(vals)
                # Check for explicit tuple grids
                tuple_matches = re.findall(r"\(\d+\s*,\s*\d+\)", code)
                if tuple_matches:
                    total += len(tuple_matches)
        return max(1, total)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_oos_data(self, asset_id: str) -> pd.DataFrame:
        """Load hidden OOS data for *asset_id* from oos_data_root."""
        path = self._oos_root / f"{asset_id}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"OOS data not found for {asset_id!r} at {path}. "
                "Ensure data_hidden/ is populated before running evaluation."
            )
        table = pq.read_table(path)
        df = table.to_pandas()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()
        # Slice to OOS window
        return df.loc[self._task.oos_start: self._task.oos_end]

    def _compute_oos_sharpe(
        self, strategy: BaseStrategy, artifact: StrategyArtifact
    ) -> float:
        """
        Re-run the strategy on OOS data and return the Sharpe ratio.

        The strategy is loaded fresh on OOS data to get signals for
        the hidden period.
        """
        # Determine primary asset
        asset_id = self._task.asset_universe[0]
        oos_data = self._load_oos_data(asset_id)

        if oos_data.empty:
            return 0.0

        # Load strategy on OOS data
        oos_strategy = self._strategy_registry.load(artifact, oos_data)
        oos_signals = oos_strategy.generate_signals()
        oos_close = oos_data["close"]

        return _compute_sharpe(oos_signals, oos_close)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_sharpe(signals: pd.Series, close: pd.Series) -> float:
    """
    Compute annualised Sharpe ratio via VectorBT for given signals and close.
    Returns 0.0 on error or if NaN.
    """
    try:
        pf = vbt.Portfolio.from_signals(
            close=close,
            entries=(signals == 1),
            exits=(signals == 0),
            freq="1D",
        )
        sharpe = float(pf.sharpe_ratio())
        del pf
        return sharpe if math.isfinite(sharpe) else 0.0
    except Exception:
        return 0.0
