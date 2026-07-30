#!/bin/bash
# Sequential 3 runs of exp_139_raw_plus_outline.
set -e
cd /home/kekshibata-admin/projects/kddcup2026-kobushi
set -a && source .env && set +a
for i in 1 2 3; do
  echo "[$(date '+%H:%M:%S')] === Starting exp_139 bench $i/3 ===" >&2
  uv run python -u scripts/math_advisor_bench_139.py
  echo "[$(date '+%H:%M:%S')] === Completed exp_139 bench $i/3 ===" >&2
done
