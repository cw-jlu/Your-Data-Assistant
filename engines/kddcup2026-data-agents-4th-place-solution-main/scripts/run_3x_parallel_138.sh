#!/bin/bash
# Sequential 3 runs of math_advisor_bench_138.py (= thin preamble).
# Run AFTER current exp_137 bench finishes.

set -e
cd /home/kekshibata-admin/projects/kddcup2026-kobushi
set -a && source .env && set +a

for i in 1 2 3; do
  echo "[$(date '+%H:%M:%S')] === Starting exp_138 thin preamble bench $i/3 ===" >&2
  uv run python -u scripts/math_advisor_bench_138.py
  echo "[$(date '+%H:%M:%S')] === Completed exp_138 thin preamble bench $i/3 ===" >&2
done
echo "[$(date '+%H:%M:%S')] === All 3 thin preamble runs done ===" >&2
