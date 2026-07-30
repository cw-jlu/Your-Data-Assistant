#!/bin/bash
# exp_144 3-attempt CONFIRMATION (leak-free prompts). Same-class comparison:
#   run A = baseline (all flags off, leak-fixed)        → exp_144_baseline_3att
#   run B = answer_shape + prose combined               → exp_144_answer_shape_prose_3att
# (video lever excluded — its summary needs the dropped 2C override-lock to convert.)
cd /home/kekshibata-admin/projects/kddcup2026-kobushi || exit 1
LOG=artifacts/exp144_3att.log
echo "==== 3att start $(date -u +%FT%TZ) ====" >> "$LOG"

echo "---- [A] baseline 3att ----" >> "$LOG"
uv run python scripts/phase2_exp144_ablation.py --workers 12 --n-attempts 3 >> "$LOG" 2>&1
echo "---- [A] baseline 3att exit=$? ----" >> "$LOG"

echo "---- [B] answer_shape+prose 3att ----" >> "$LOG"
EXP144_ANSWER_SHAPE=1 EXP144_PROSE=1 uv run python scripts/phase2_exp144_ablation.py --workers 12 --n-attempts 3 >> "$LOG" 2>&1
echo "---- [B] combined 3att exit=$? ----" >> "$LOG"

echo "==== 3att done $(date -u +%FT%TZ) ====" >> "$LOG"
