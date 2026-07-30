#!/usr/bin/env bash
# Poll for benchmark completion. Exits within ~3 minutes (so it stays well
# inside copilot's Bash-tool tolerance), with one of three outcomes:
#   READY    — evaluation.json exists; prints the score summary
#   PENDING  — bench still running; takt should loop back to wait_bench
#   FAILED   — child died without producing evaluation.json
#
# Usage: scripts/poll_bench.sh <exp_name>
set -uo pipefail

exp_name=${1:?usage: poll_bench.sh <exp_name>}
cd "$(dirname "$0")/.."

pid_file="/tmp/bench_${exp_name}.pid"
log_file="/tmp/bench_${exp_name}.log"

# Up to 3 minutes total: 6 polls × 30s sleep.
for i in $(seq 1 6); do
    run_dir=$(ls -1dt artifacts/runs/${exp_name}_* 2>/dev/null | head -1)
    if [ -n "$run_dir" ] && [ -f "$run_dir/evaluation.json" ]; then
        score=$(uv run python -c "import json,sys; d=json.load(open('$run_dir/evaluation.json')); print(f\"score={d['official_score_lambda_0_5_mean']:.4f} perfect={d['perfect_recall_no_extras_count']}/{d['task_count']} miss={d['missing_prediction_count']} zero={d['zero_recall_count']} wext={d['perfect_recall_with_extras_count']} partial={d['partial_recall_count']}\")")
        echo "READY run_dir=$run_dir $score"
        exit 0
    fi
    # Process died?
    if [ -f "$pid_file" ]; then
        pid=$(cat "$pid_file")
        if ! kill -0 "$pid" 2>/dev/null; then
            # Child exited; if evaluation.json still missing, it failed.
            echo "FAILED pid=$pid (no evaluation.json)"
            tail -20 "$log_file" 2>/dev/null
            exit 1
        fi
    fi
    sleep 30
done

# Still running.
trace_count=$(find artifacts/runs/${exp_name}_* -name trace.json 2>/dev/null | wc -l)
echo "PENDING traces=${trace_count}/50 elapsed_since=$(stat -c %y /tmp/bench_${exp_name}.start 2>/dev/null || echo unknown)"
exit 0
