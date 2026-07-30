#!/bin/bash
# Sequential 3 runs of math_advisor_bench_parallel.py.
# Each run uses multiprocessing.spawn × 3 attempts = 12 concurrent vLLM streams.
# Expected ~75-90 min per run, ~4h total.

set -e
cd /home/kekshibata-admin/projects/kddcup2026-kobushi
set -a && source .env && set +a

for i in 1 2 3; do
  echo "[$(date '+%H:%M:%S')] === Starting parallel bench $i/3 ===" >&2
  uv run python -u scripts/math_advisor_bench_parallel.py
  echo "[$(date '+%H:%M:%S')] === Completed parallel bench $i/3 ===" >&2
done
echo "[$(date '+%H:%M:%S')] === All 3 parallel runs done ===" >&2
