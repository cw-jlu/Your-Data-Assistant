#!/usr/bin/env bash
# Run an experiment's benchmark N times and emit per-run scores plus
# aggregate statistics (mean / median / std / min / max / 95% CI).
#
# Usage: scripts/replicate_bench.sh <exp_name> [N=5] [ABORT_BELOW=0.55]
#   BG=1 scripts/replicate_bench.sh ...   # detach into background, print PID, exit
#
# Each repetition uses scripts/launch_bench.sh which writes into a fresh
# run dir (auto-numbered exp_<name>_NNN). After all N runs finish we
# write a summary file:
#   artifacts/replications/<exp_name>/summary_<timestamp>.json
#
# Background mode (BG=1): re-execs as a detached nohup process, writes its
# PID to /tmp/replicate_${exp_name}.pid, and exits immediately with the PID
# printed to stdout (`PID=NNNN`). The caller (e.g. takt's launch_bench_next)
# can then return and let the next iter's design phase happen in parallel
# with the bench. wait_bench_current polls the PID file to gate the next
# launch so only one bench runs against vLLM at a time (single-bench lock
# preserved).
#
# Early-abort: after the first run, if its λ0.5 is below ABORT_BELOW
# (default 0.65), skip remaining runs and write summary with n=1. This
# saves ~5 hours per clearly-failed experiment.
#
# Threshold rationale (2026-05-05 user revision, was 0.55):
#   - clean baseline (exp_040) ≈ 0.66
#   - winners cluster at 0.69+ (exp_068=0.6882, exp_081=0.6906, exp_086=0.7353)
#   - run-1 noise (single sample) std ≈ 0.04 → 0.65 is one σ below baseline
#   - anything < 0.65 in run 1 is highly unlikely to be a real winner;
#     replicating it just wastes ~5 hours of bench time
#
# Set ABORT_BELOW=0 to disable.
#
# Returns 0 on success, non-zero if any run fails to evaluate.
set -uo pipefail

exp_name=${1:?usage: replicate_bench.sh <exp_name> [N] [ABORT_BELOW]}
N=${2:-5}
ABORT_BELOW=${3:-${ABORT_BELOW:-0.65}}
repo_root=$(cd "$(dirname "$0")/.." && pwd)
cd "$repo_root"

# Load .env so the auto-retry curl probe (and any sub-script) has access to
# AGENT_API_BASE / AGENT_API_KEY / CF_ACCESS_CLIENT_{ID,SECRET}. Without this
# the BG-nohup'd subprocess has empty env, the probe goes to Cloudflare
# without auth, gets HTTP 404 instead of 200, and the script falsely
# concludes vLLM is down — wasting time in the polling loop.
if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

# Validate up-front — both BG and FG paths need this.
if [ ! -d "src/experiments/${exp_name}" ]; then
    echo "FAIL: exp dir not found: src/experiments/${exp_name}" >&2
    exit 2
fi
if ! [[ "$N" =~ ^[0-9]+$ ]] || [ "$N" -lt 1 ] || [ "$N" -gt 20 ]; then
    echo "FAIL: N must be 1..20, got '$N'" >&2
    exit 2
fi

# --- Background mode: detach and exit ---------------------------------------
# Caller invokes with BG=1; we re-spawn as nohup, print PID, return.
# The detached process unsets BG so it runs the foreground codepath below.
if [ "${BG:-0}" = "1" ]; then
    pid_file="/tmp/replicate_${exp_name}.pid"
    log_file="/tmp/replicate_${exp_name}.log"
    # If a previous replicate is still alive, refuse (single-bench lock).
    if [ -f "$pid_file" ]; then
        old_pid=$(cat "$pid_file" 2>/dev/null || true)
        if [ -n "$old_pid" ] && kill -0 "$old_pid" 2>/dev/null; then
            echo "ALREADY_RUNNING pid=$old_pid log=$log_file"
            exit 0
        fi
    fi
    rm -f "$log_file"
    nohup env BG=0 bash "$0" "$exp_name" "$N" "$ABORT_BELOW" \
        >"$log_file" 2>&1 < /dev/null &
    bg_pid=$!
    echo "$bg_pid" > "$pid_file"
    # tiny grace so process starts and writes its own status
    sleep 1
    echo "STARTED pid=$bg_pid log=$log_file"
    exit 0
fi

mkdir -p artifacts/replications/${exp_name}
ts=$(date +%Y%m%d_%H%M%S)
summary_file="artifacts/replications/${exp_name}/summary_${ts}.json"

echo "===== replicate ${exp_name} N=${N} (abort_below=${ABORT_BELOW}) ====="
echo "summary will be written to: ${summary_file}"

# Track run dirs we created
collected_dirs=()
collected_scores=()

# Wait for any in-flight bench of this exp to finish before we start
pid_file="/tmp/bench_${exp_name}.pid"
if [ -f "$pid_file" ]; then
    pid=$(cat "$pid_file" 2>/dev/null)
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        echo "  waiting for in-flight bench (pid=$pid) to finish..."
        while kill -0 "$pid" 2>/dev/null; do sleep 30; done
        sleep 5  # give evaluation.json a moment to flush
    fi
fi

# Snapshot the existing run dirs so we can identify newly-created ones
mapfile -t pre_dirs < <(ls -1d artifacts/runs/${exp_name}_*/ 2>/dev/null || true)
echo "  pre-existing run dirs: ${#pre_dirs[@]}"

# Auto-retry on vLLM 502 contamination — by default retry indefinitely until
# vLLM recovers. The bench has its own per-task task_timeout_seconds so
# nothing hangs silently within a single run, and the outer takt loop can be
# killed by user if needed. Override with MAX_INFRA_RETRIES env var if you
# want a hard cap (e.g. for unattended runs you don't trust).
MAX_INFRA_RETRIES=${MAX_INFRA_RETRIES:-99}
# Per-probe interval and total polling deadline (default ~12 hours).
VLLM_PROBE_INTERVAL_SEC=${VLLM_PROBE_INTERVAL_SEC:-60}
VLLM_PROBE_DEADLINE_SEC=${VLLM_PROBE_DEADLINE_SEC:-43200}
i=1
while [ "$i" -le "$N" ]; do
    echo
    if [ -n "${retry_count:-}" ] && [ "${retry_count:-0}" -gt 0 ]; then
        echo "----- replication ${i}/${N} (retry ${retry_count}/${MAX_INFRA_RETRIES} after vLLM 502 contamination) -----"
    else
        echo "----- replication ${i}/${N} -----"
    fi

    # Launch (creates new run dir under artifacts/runs/${exp_name}_NNN)
    out=$(bash scripts/launch_bench.sh "$exp_name")
    echo "  $out"
    if echo "$out" | grep -qE '^FAILED|^ALREADY_RUNNING'; then
        echo "  unexpected launch state, aborting replication"
        break
    fi

    # Poll until the new bench process exits AND evaluation.json appears.
    # Maximum wait per run: 90 minutes (matches loop monitor in pipelined yaml).
    new_pid=$(cat "$pid_file")
    deadline=$(( $(date +%s) + 90 * 60 ))
    while true; do
        if [ "$(date +%s)" -gt "$deadline" ]; then
            echo "  TIMEOUT after 90 min — killing pid=$new_pid"
            kill "$new_pid" 2>/dev/null || true
            break
        fi
        if ! kill -0 "$new_pid" 2>/dev/null; then
            # process gone — check if evaluation.json was written
            sleep 3
            latest_dir=$(ls -1dt artifacts/runs/${exp_name}_*/ 2>/dev/null | head -1)
            if [ -f "${latest_dir}evaluation.json" ]; then
                break
            fi
            # process gone but no eval — that's a failure
            echo "  bench process exited but no evaluation.json found"
            break
        fi
        traces=$(ls "$(ls -1dt artifacts/runs/${exp_name}_*/ 2>/dev/null | head -1)"task_*/trace.json 2>/dev/null | wc -l)
        printf "  waiting... pid=%s traces=%s/50 elapsed=%dmin\n" \
            "$new_pid" "$traces" "$(( ( $(date +%s) - (deadline - 90*60) ) / 60 ))"
        sleep 60
    done

    latest_dir=$(ls -1dt artifacts/runs/${exp_name}_*/ 2>/dev/null | head -1)
    if [ ! -f "${latest_dir}evaluation.json" ]; then
        echo "  ${i}/${N} FAILED (no evaluation.json in $latest_dir)"
        i=$((i + 1))
        retry_count=0
        continue
    fi
    score=$(python3 -c "import json; d=json.load(open('${latest_dir}evaluation.json')); print(f\"{d['official_score_lambda_0_5_mean']:.4f}\")" 2>/dev/null)
    perf=$(python3 -c "import json; d=json.load(open('${latest_dir}evaluation.json')); print(d['perfect_recall_no_extras_count'])" 2>/dev/null)
    miss=$(python3 -c "import json; d=json.load(open('${latest_dir}evaluation.json')); print(d['missing_prediction_count'])" 2>/dev/null)

    # ============================================================================
    # vLLM 502 contamination detection + auto-retry
    # ============================================================================
    # If missing >= 5 AND any trace contains "Error code: 502", treat the run
    # as infra-contaminated (vLLM outage), move it to artifacts/runs_discarded/,
    # and retry this run slot up to MAX_INFRA_RETRIES times. Avoids n=N stats
    # being silently corrupted by transient outages on gpu-host.internal.
    # Detection criteria match the PRIORITIES URGENT 「502 outage 廃棄ルール」.
    contaminated=0
    if [ -n "$miss" ] && [ "$miss" -ge 5 ]; then
        # Sample any trace.json for "Error code: 502" — fast grep, no need to scan all
        if grep -lE "Error code: 502" "${latest_dir}"task_*/trace.json 2>/dev/null | head -1 | grep -q .; then
            contaminated=1
        fi
    fi

    if [ "$contaminated" = "1" ]; then
        retry_count=${retry_count:-0}
        retry_count=$((retry_count + 1))
        echo "  ${i}/${N} CONTAMINATED (miss=${miss}, 502 detected) — vLLM outage suspected"

        # Move to runs_discarded/ for forensics
        mkdir -p artifacts/runs_discarded
        suffix="_502retry$(date +%H%M%S)"
        discarded_path="artifacts/runs_discarded/$(basename ${latest_dir%/})${suffix}"
        mv "$latest_dir" "$discarded_path"
        cat > "${discarded_path}/REASON.md" <<EOF
# Auto-discarded: vLLM 502 outage detected

Detected during replicate_bench.sh run ${i}/${N}, retry attempt ${retry_count}.
- λ0.5 = ${score}
- missing = ${miss} (>= 5 = contamination threshold)
- "Error code: 502" found in at least one task trace
- moved automatically by replicate_bench.sh's contamination guard
EOF
        echo "  → moved to ${discarded_path}, REASON.md written"

        if [ "$retry_count" -ge "$MAX_INFRA_RETRIES" ]; then
            echo "  retry budget exhausted (${retry_count}/${MAX_INFRA_RETRIES}) — accepting last failed run, moving on"
            # Restore latest run from discarded so summary still gets it (last attempt wins)
            mv "$discarded_path" "$latest_dir"
            collected_dirs+=("$latest_dir")
            collected_scores+=("$score")
            retry_count=0
            i=$((i + 1))
            continue
        fi

        # Probe vLLM health and BLOCK until it recovers (no fixed cap).
        # Long probe deadline (default 12h) is just a safety net to avoid a
        # permanent hang if gpu-host.internal is decommissioned. Within that,
        # we keep polling at VLLM_PROBE_INTERVAL_SEC (default 60s) until
        # /v1/models returns 200, then retry the same slot.
        echo "  waiting for vLLM to recover..."
        wait_started=$(date +%s)
        wait_deadline=$(( wait_started + VLLM_PROBE_DEADLINE_SEC ))
        probe=0
        while [ "$(date +%s)" -lt "$wait_deadline" ]; do
            probe=$((probe + 1))
            health=$(curl -sw "%{http_code}" -m 5 -o /dev/null \
                -H "CF-Access-Client-Id: ${CF_ACCESS_CLIENT_ID:-}" \
                -H "CF-Access-Client-Secret: ${CF_ACCESS_CLIENT_SECRET:-}" \
                -H "Authorization: Bearer ${AGENT_API_KEY:-}" \
                "${AGENT_API_BASE:-http://localhost:8000}/models" 2>/dev/null)
            elapsed_min=$(( ( $(date +%s) - wait_started ) / 60 ))
            if [ "$health" = "200" ]; then
                echo "  vLLM healthy (HTTP 200) after ${elapsed_min}min wait (probe #${probe}) — retrying run ${i}"
                break
            fi
            echo "  vLLM still down (HTTP ${health}) elapsed=${elapsed_min}min probe=${probe} — sleeping ${VLLM_PROBE_INTERVAL_SEC}s"
            sleep "$VLLM_PROBE_INTERVAL_SEC"
        done
        if [ "$(date +%s)" -ge "$wait_deadline" ]; then
            echo "  ABORT: vLLM did not recover within ${VLLM_PROBE_DEADLINE_SEC}s deadline (~$((VLLM_PROBE_DEADLINE_SEC/3600))h)"
            echo "  giving up the entire replication. Successfully collected ${#collected_dirs[@]}/${N} runs before this point."
            break  # exit the outer while-loop entirely; aggregator below will write a partial summary
        fi

        # Loop back: same i, don't increment, retry this slot
        continue
    fi

    # Clean run path
    echo "  ${i}/${N} done: λ0.5=${score} perf=${perf} miss=${miss} dir=${latest_dir}"
    collected_dirs+=("$latest_dir")
    collected_scores+=("$score")
    retry_count=0  # reset for next slot

    # Early-abort: after run 1, if score is below ABORT_BELOW, skip remaining runs.
    # Saves ~5 hours per clearly-failed experiment.
    if [ "$i" = "1" ] && [ "$N" -gt 1 ] && [ -n "$score" ]; then
        below=$(awk -v s="$score" -v t="$ABORT_BELOW" 'BEGIN{print (s+0 < t+0) ? 1 : 0}')
        if [ "$below" = "1" ]; then
            echo
            echo "===== EARLY ABORT after run 1 ====="
            echo "  λ0.5=${score} < threshold=${ABORT_BELOW} — skipping runs 2..${N}"
            echo "  (saves ~$(( (N-1) * 75 )) min of bench time)"
            break
        fi
    fi

    i=$((i + 1))
done

# Aggregate
python3 - "$exp_name" "$summary_file" "${collected_dirs[@]}" <<'PYEOF'
import json
import statistics
import sys
from pathlib import Path

exp_name = sys.argv[1]
out_path = Path(sys.argv[2])
run_dirs = [Path(d) for d in sys.argv[3:]]

runs = []
for d in run_dirs:
    eval_path = d / "evaluation.json"
    if not eval_path.exists():
        continue
    ev = json.load(open(eval_path))
    runs.append(
        {
            "run_dir": str(d),
            "lambda_0_0": ev["official_score_lambda_0_0_mean"],
            "lambda_0_5": ev["official_score_lambda_0_5_mean"],
            "lambda_1_0": ev["official_score_lambda_1_0_mean"],
            "perfect": ev["perfect_recall_no_extras_count"],
            "with_extras": ev["perfect_recall_with_extras_count"],
            "zero": ev["zero_recall_count"],
            "missing": ev["missing_prediction_count"],
        }
    )

if not runs:
    print(json.dumps({"exp_name": exp_name, "error": "no completed runs"}))
    sys.exit(1)

xs = [r["lambda_0_5"] for r in runs]
mean = statistics.mean(xs)
median = statistics.median(xs)
stdev = statistics.stdev(xs) if len(xs) > 1 else 0.0
# 95% CI for mean (t-distribution approx with z=1.96 for n>=5)
ci_half = 1.96 * stdev / (len(xs) ** 0.5) if len(xs) > 1 else 0.0

summary = {
    "exp_name": exp_name,
    "n": len(runs),
    "lambda_0_5": {
        "mean": round(mean, 4),
        "median": round(median, 4),
        "std": round(stdev, 4),
        "min": round(min(xs), 4),
        "max": round(max(xs), 4),
        "ci_95_half_width": round(ci_half, 4),
        "ci_95": [round(mean - ci_half, 4), round(mean + ci_half, 4)],
    },
    "perfect": {
        "mean": round(statistics.mean([r["perfect"] for r in runs]), 1),
        "min": min(r["perfect"] for r in runs),
        "max": max(r["perfect"] for r in runs),
    },
    "missing": {
        "mean": round(statistics.mean([r["missing"] for r in runs]), 1),
        "min": min(r["missing"] for r in runs),
        "max": max(r["missing"] for r in runs),
    },
    "runs": runs,
}

out_path.write_text(json.dumps(summary, indent=2) + "\n")
print()
print(f"=== Replication summary: {exp_name} (n={len(runs)}) ===")
print(f"  λ0.5 mean   = {summary['lambda_0_5']['mean']:.4f}")
print(f"  λ0.5 median = {summary['lambda_0_5']['median']:.4f}")
print(f"  λ0.5 std    = {summary['lambda_0_5']['std']:.4f}")
print(f"  λ0.5 95% CI = [{summary['lambda_0_5']['ci_95'][0]:.4f}, {summary['lambda_0_5']['ci_95'][1]:.4f}]")
print(f"  perfect     = mean {summary['perfect']['mean']} (range {summary['perfect']['min']}-{summary['perfect']['max']})")
print(f"  missing     = mean {summary['missing']['mean']} (range {summary['missing']['min']}-{summary['missing']['max']})")
print(f"  written: {out_path}")
PYEOF

# Exit success: at least N/2 runs OR exactly 1 run (= early-abort path)
n_collected=${#collected_dirs[@]}
if [ "$n_collected" = "0" ]; then
    echo "FAIL: no runs completed"
    exit 1
fi
if [ "$n_collected" -lt $(( (N + 1) / 2 )) ] && [ "$n_collected" != "1" ]; then
    echo "FAIL: only ${n_collected}/${N} runs completed (need at least half, or 1 if early-aborted)"
    exit 1
fi
if [ "$n_collected" = "1" ] && [ "$N" -gt 1 ]; then
    echo "PASS (early-abort): 1/${N} runs — score below ${ABORT_BELOW} threshold, replication skipped"
else
    echo "PASS: ${n_collected}/${N} runs completed successfully"
fi
exit 0
