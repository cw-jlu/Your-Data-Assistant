#!/bin/bash
# Interleaved sequential: 137 → 138 → 137 → 138 (= 2 more of each, n=5 total).
# Removes time-of-day bias by mixing the two configs.

set -e
cd /home/kekshibata-admin/projects/kddcup2026-kobushi
set -a && source .env && set +a

for round in 1 2; do
  echo "[$(date '+%H:%M:%S')] === Round $round/2: exp_137 ===" >&2
  uv run python -u scripts/math_advisor_bench_parallel.py
  echo "[$(date '+%H:%M:%S')] === Round $round/2: exp_137 done ===" >&2

  echo "[$(date '+%H:%M:%S')] === Round $round/2: exp_138 ===" >&2
  uv run python -u scripts/math_advisor_bench_138.py
  echo "[$(date '+%H:%M:%S')] === Round $round/2: exp_138 done ===" >&2
done
echo "[$(date '+%H:%M:%S')] === All 4 additional runs done ===" >&2
