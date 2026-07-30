#!/usr/bin/env bash
# Launch a benchmark asynchronously and return in <1s (or after waiting for
# any other bench to finish — see global lock below).
#
# Usage: scripts/launch_bench.sh <exp_name>
#   e.g. scripts/launch_bench.sh exp_010_verify_before_execute
#
# Side-effects:
#   /tmp/bench_<exp>.log   — benchmark stdout+stderr
#   /tmp/bench_<exp>.pid   — child PID
#   /tmp/bench_<exp>.start — start timestamp (UTC ISO)
#
# Global single-bench enforcement (2026-05-03):
# Two benches running in parallel saturate the vLLM endpoint, which causes
# per-task latency spikes and depresses scores by 0.05-0.10 (replication of
# exp_040 under concurrent load showed mean=0.62 vs 0.7267 single-run).
# We now block here until any other in-flight bench finishes, so every bench
# runs solo. Workers inside a single bench (max_workers=8) are still parallel
# — only the OUTER level (multiple bench processes) is serialized.
set -uo pipefail

exp_name=${1:?usage: launch_bench.sh <exp_name>}
cd "$(dirname "$0")/.."

mkdir -p /tmp
log="/tmp/bench_${exp_name}.log"
pid_file="/tmp/bench_${exp_name}.pid"
start_file="/tmp/bench_${exp_name}.start"

# Same-exp check: if a bench for THIS exp is still running, no-op.
if [ -f "$pid_file" ] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    echo "ALREADY_RUNNING pid=$(cat "$pid_file")"
    exit 0
fi

# Global single-bench wait: block until any OTHER exp's bench finishes.
# Polls every 30s; max wait 6 hours. Prints status every 5 min.
wait_started=$(date +%s)
max_wait_sec=$((6 * 3600))
last_status=$wait_started
while :; do
    busy_pid=""
    busy_exp=""
    for other_pid_file in /tmp/bench_*.pid; do
        [ -f "$other_pid_file" ] || continue
        other_pid=$(cat "$other_pid_file" 2>/dev/null)
        [ -z "$other_pid" ] && continue
        # Skip our own pid file (already-running case handled above)
        [ "$other_pid_file" = "$pid_file" ] && continue
        if kill -0 "$other_pid" 2>/dev/null; then
            busy_pid=$other_pid
            busy_exp=$(basename "$other_pid_file" .pid | sed 's/^bench_//')
            break
        fi
    done
    if [ -z "$busy_pid" ]; then
        break
    fi
    now=$(date +%s)
    waited=$((now - wait_started))
    if [ $waited -gt $max_wait_sec ]; then
        echo "TIMEOUT waited ${waited}s (>${max_wait_sec}s) for bench ${busy_exp} pid=${busy_pid}"
        exit 1
    fi
    if [ $((now - last_status)) -ge 300 ]; then
        echo "WAIT ${exp_name}: bench ${busy_exp} pid=${busy_pid} still running (waited ${waited}s)" >&2
        last_status=$now
    fi
    sleep 30
done

# Spawn fully detached (no tty, no stdin, no parent process group)
nohup setsid bash -c "exec uv run python -m experiments.${exp_name}.run run-benchmark --evaluate" \
    > "$log" 2>&1 < /dev/null &
echo $! > "$pid_file"
date -u +%FT%TZ > "$start_file"

echo "STARTED exp=${exp_name} pid=$(cat "$pid_file") log=$log"
exit 0
