#!/usr/bin/env bash
# Outer driver: keep takt's experiment-iter workflow alive.
#
# The workflow itself loops internally (commit → pick) up to its max_steps
# budget (500). This driver restarts it whenever it ABORTs or crashes —
# transient API outages, ABORT decisions from supervisor, takt CLI bugs.
#
# Long benchmarks no longer block takt: experiment-iter.yaml uses
# scripts/launch_bench.sh + scripts/poll_bench.sh to run the benchmark
# asynchronously and poll for completion in 3-min windows.
#
# Usage:
#   tmux new -s exp-loop
#   bash scripts/loop_experiments.sh
#
# Stop with Ctrl-C in the tmux session, or kill the parent process.
set -uo pipefail

cd "$(dirname "$0")/.."

# Workflow choice: default = pipelined-v2 (pipelined + automatic leak audit
# step between implement and smoke). Override with EXP_WORKFLOW env var:
#   EXP_WORKFLOW=experiment-iter-pipelined  ← v1 without audit
#   EXP_WORKFLOW=experiment-iter            ← original sequential
WORKFLOW_NAME="${EXP_WORKFLOW:-experiment-iter-pipelined-v2}"

# Safety belt: never let takt commit on main/master, unless explicitly allowed.
# On the personal repo (kekshibata/kddcup2026-kobushi) main IS the working
# branch — set TAKT_ALLOW_MAIN=1 to bypass this guard for that case.
branch=$(git rev-parse --abbrev-ref HEAD)
if [ "$branch" = "main" ] || [ "$branch" = "master" ]; then
    if [ "${TAKT_ALLOW_MAIN:-}" != "1" ]; then
        echo "FATAL: refusing to drive experiment loop on $branch. Switch to a feature branch first, or export TAKT_ALLOW_MAIN=1 if main is the working branch on this repo."
        exit 2
    fi
    echo "NOTE: TAKT_ALLOW_MAIN=1 — running on $branch (commits will land on the protected branch)"
fi

mkdir -p /tmp/takt_logs

iter=1
while true; do
    ts=$(date -u +%FT%TZ)
    log_file="/tmp/takt_logs/iter_$(date +%Y%m%d_%H%M%S).log"
    echo "==== [$ts] outer-iteration ${iter} start (branch=${branch}) workflow=${WORKFLOW_NAME} log=${log_file} ===="

    # --skip-git: takt does not auto-commit; the workflow's commit step
    # does explicit per-iteration commits with the right messages.
    # Provider/model: Anthropic Claude direct via claude-sdk (user directive
    # 2026-05-04, post-server migration). Uses local Claude Code authenticated
    # session — no ANTHROPIC_API_KEY needed.
    # Override with TAKT_PROVIDER / TAKT_MODEL env vars.
    TAKT_PROVIDER="${TAKT_PROVIDER:-claude-sdk}"
    TAKT_MODEL="${TAKT_MODEL:-claude-sonnet-4-6}"
    if takt --workflow "$WORKFLOW_NAME" \
            --task "Iterate experiments using PRIORITIES.md / BACKLOG.md / EXPERIMENTS.md as state. The workflow loops internally; let it run until ABORT or max_steps." \
            --pipeline \
            --skip-git \
            --provider "$TAKT_PROVIDER" \
            --model "$TAKT_MODEL" \
            2>&1 | tee "$log_file"; then
        echo "==== [$ts] outer-iteration ${iter} clean exit ===="
    else
        rc=$?
        echo "==== [$ts] outer-iteration ${iter} exited rc=$rc — sleeping 60s before restart ===="
        sleep 60
    fi

    iter=$((iter + 1))
    sleep 30  # cool-off for git/API
done
