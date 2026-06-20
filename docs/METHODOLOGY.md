# AlphaBench Methodology

## 1. Purpose

AlphaBench is a benchmark for evaluating whether an LLM agent can perform agentic quantitative research and discover statistically defensible trading alpha under a fixed research budget.

The benchmark measures the ability to:

* explore financial data
* form alpha hypotheses
* build a strategy
* validate it with backtests
* avoid overfitting
* generalize out of sample

AlphaBench does not measure whether a model can predict prices in one shot. It measures whether a model can conduct a research loop that leads to a robust strategy.

### 1.1 Core capability being measured

The target capability is end-to-end quantitative research under uncertainty.

This includes:

* data exploration
* feature observation
* hypothesis generation
* strategy construction
* target-aware experiment tracking
* iterative refinement
* statistical validation
* hidden out-of-sample generalization

---

## 2. Definitions of Terms

### 2.1 Task definition

A single AlphaBench task is defined as:

* one market
* one asset universe
* one hidden evaluation period

Example:

* crypto spot (Binance)
* training data: 2021-01-01 through 2024-12-31
* hidden evaluation: 2025-01-01 through 2025-12-31

### 2.2 Agent objective

The agent must submit a final strategy that passes validation gates and performs well on hidden OOS evaluation.

### 2.3 Trial definition

A trial is any target-aware experiment that uses performance feedback or future-return-linked information to guide strategy development.

For MVP, the following count as trials:

* each backtest call
* each explicit target-aware EDA experiment reported by the agent

### 2.4 What counts as target-aware

Target-aware operations include any analysis that directly or indirectly uses:

* returns
* PnL
* Sharpe
* drawdown
* future performance labels
* feature/target correlations
* any experiment designed to optimize a strategy based on outcome feedback

### 2.5 Trial accounting

Effective trial count is computed as:

`EffectiveTrials = BacktestCount + ReportedTargetAwareEDATrials`

Where:
* `BacktestCount` is the number of backtest tool calls
* `ReportedTargetAwareEDATrials` is the number of explicit target-aware experiments reported by the agent

### 2.6 MVP constraints

* single agent only
* rule-based strategy only
* long/flat only
* no ML in MVP
* no walk-forward in MVP
* no internet access
* no external strategy imports beyond allowed globals

### 2.7 Success condition

A run is successful if the final strategy passes validation gates and achieves strong hidden OOS performance relative to other submissions.

---

## 3. Evaluation & Validation Gates

### 3.1 Evaluation stages

AlphaBench uses a two-stage evaluation process:

1. In-sample validation gates
2. Hidden out-of-sample ranking

### 3.2 In-sample validation gates

A strategy must pass the following gates before it can be ranked on the leaderboard:

* Permutation test p-value <= 0.05
* DSR >= 0.30

### 3.3 Permutation test

For MVP, use a signal permutation test.

Procedure:

1. Run the strategy on the in-sample market data.
2. Record the original performance metric (Sharpe ratio).
3. Shuffle the signal order while preserving the original signal distribution (ratio of long vs. flat periods).
4. Apply the shuffled signals to the original price series.
5. Recompute the Sharpe ratio for many random permutations (default: 1000).
6. Compute the empirical p-value as the fraction of permutations that meet or exceed the original Sharpe ratio.

### 3.4 DSR (Deflated Sharpe Ratio)

DSR is used as a gate to penalize search breadth and multiple comparisons.

MVP threshold:

* DSR >= 0.30

### 3.5 Hidden evaluation

Only strategies that pass the gates are evaluated on the hidden OOS period.

Primary hidden metric:

* OOS Sharpe

### 3.6 Ranking rule

Leaderboard ranking is based on hidden OOS Sharpe after gate checks pass.
