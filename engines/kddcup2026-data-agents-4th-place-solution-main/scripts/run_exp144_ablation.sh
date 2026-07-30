#!/bin/bash
# exp_144 per-lever ablation: 3 sequential att=1 / 60-task runs.
# Baseline for comparison = exp_143_dynamic_doc_001 (mean 0.3950) — identical
# pipeline, verified-inert gates when all flags off.
cd /home/kekshibata-admin/projects/kddcup2026-kobushi || exit 1
LOG=artifacts/exp144_ablation.log
echo "==== exp_144 ablation start $(date -u +%FT%TZ) ====" | tee -a "$LOG"

echo "---- [1/3] answer_shape ----" | tee -a "$LOG"
EXP144_ANSWER_SHAPE=1 uv run python scripts/phase2_exp144_ablation.py --workers 12 2>&1 | tee -a "$LOG"

echo "---- [2/3] prose ----" | tee -a "$LOG"
EXP144_PROSE=1 uv run python scripts/phase2_exp144_ablation.py --workers 12 2>&1 | tee -a "$LOG"

echo "---- [3/3] video ----" | tee -a "$LOG"
EXP144_VIDEO=1 uv run python scripts/phase2_exp144_ablation.py --workers 12 2>&1 | tee -a "$LOG"

echo "==== exp_144 ablation done $(date -u +%FT%TZ) ====" | tee -a "$LOG"
