#!/usr/bin/env bash
set -euo pipefail

TASK_CFG="config/default_task.json"
TOTAL=12
COUNT=0

for cfg in \
  config/grid_v600_t0p3_turns20_trials15.json \
  config/grid_v600_t0p3_turns20_trials25.json \
  config/grid_v600_t0p3_turns35_trials15.json \
  config/grid_v600_t0p3_turns35_trials25.json \
  config/grid_v600_t0p6_turns20_trials15.json \
  config/grid_v600_t0p6_turns20_trials25.json \
  config/grid_v600_t0p6_turns35_trials15.json \
  config/grid_v600_t0p6_turns35_trials25.json \
  config/grid_v600_t0p9_turns20_trials15.json \
  config/grid_v600_t0p9_turns20_trials25.json \
  config/grid_v600_t0p9_turns35_trials15.json \
  config/grid_v600_t0p9_turns35_trials25.json
do
  COUNT=$((COUNT + 1))
  echo "========================================"
  echo "[$COUNT/$TOTAL] Running: $cfg"
  echo "========================================"
  uv run python3 run_benchmark.py \
    --task-config "$TASK_CFG" \
    --agent-config "$cfg"
  echo "Done: $cfg"
  echo ""
done

echo "========================================"
echo "All $TOTAL benchmark runs complete."
echo "========================================"
