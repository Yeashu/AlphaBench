#!/usr/bin/env bash
set -euo pipefail

TASK_CFG="config/default_task.json"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# Config files and their temperature suffixes
configs=(
  "config/grid_v600_t0p6_turns35_trials15.json"
  "config/val_v600_t0p7_turns35_trials15.json"
  "config/val_v600_t0p8_turns35_trials15.json"
  "config/grid_v600_t0p9_turns35_trials15.json"
)

names=(
  "t0p6"
  "t0p7"
  "t0p8"
  "t0p9"
)

TOTAL=12
COUNT=0

echo "========================================"
echo "Starting validation sweep (12 runs)"
echo "========================================"

for i in "${!configs[@]}"; do
  cfg="${configs[$i]}"
  name="${names[$i]}"
  
  for rep in 1 2 3; do
    COUNT=$((COUNT + 1))
    run_id="val_v600_${name}_r${rep}"
    stdout_log="${LOG_DIR}/${run_id}.stdout.log"
    
    echo "[$COUNT/$TOTAL] Running $run_id (Config: $cfg)"
    echo "Logs redirected to $stdout_log"
    
    # Run the benchmark and redirect stdout and stderr to the log file
    uv run python3 run_benchmark.py \
      --task-config "$TASK_CFG" \
      --agent-config "$cfg" \
      --run-id "$run_id" > "$stdout_log" 2>&1
      
    # Print a brief summary of the run result
    if grep -q "passes_gates      : True" "$stdout_log"; then
      status="PASSED"
      sharpe=$(grep "OOS Sharpe" "$stdout_log" | awk '{print $4}')
      pval=$(grep "permutation pvalue" "$stdout_log" | awk '{print $4}')
      echo "--> Result: $status | OOS Sharpe: $sharpe | p-val: $pval"
    elif grep -q "passes_gates      : False" "$stdout_log"; then
      status="FAILED"
      pval=$(grep "permutation pvalue" "$stdout_log" | awk '{print $4}')
      echo "--> Result: $status | p-val: $pval"
    else
      echo "--> Result: Finished (or evaluation skipped/failed)"
    fi
    echo ""
  done
done

echo "========================================"
echo "Validation sweep complete!"
echo "========================================"
