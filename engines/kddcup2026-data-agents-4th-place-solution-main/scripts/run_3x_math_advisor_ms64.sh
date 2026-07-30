#!/bin/bash
# Sequential 3 runs of math_advisor_full50.py at max_steps=64.
# Each run auto-creates a new exp_137_math_advisor_NNN dir.
# Total ~8h (= ~155 min × 3).

set -e
cd /home/kekshibata-admin/projects/kddcup2026-kobushi
set -a && source .env && set +a

for i in 1 2 3; do
  echo "[$(date '+%H:%M:%S')] === Starting bench run $i/3 ===" >&2
  uv run python -u scripts/math_advisor_full50.py
  echo "[$(date '+%H:%M:%S')] === Completed bench run $i/3 ===" >&2
done
echo "[$(date '+%H:%M:%S')] === All 3 runs done ===" >&2
