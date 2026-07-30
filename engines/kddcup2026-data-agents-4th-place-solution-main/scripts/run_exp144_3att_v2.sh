#!/bin/bash
# exp_144 3-att, take 2 (leak-corrected: v11-inherited examples kept as exp_137,
# only assistant-introduced contamination cleaned).
#   step 1: baseline on the REMAINING 40 tasks (task_1..22-minus done already in
#           exp_144_baseline_3att_002) → exp_144_baseline_3att_003
#           (merge _002[20] + _003[40] = full-60 baseline)
#   step 2: answer_shape + prose combined, FULL 60 → exp_144_answer_shape_prose_3att_001
cd /home/kekshibata-admin/projects/kddcup2026-kobushi || exit 1
LOG=artifacts/exp144_3att_v2.log
REMAIN="task_15,task_20,task_23,task_24,task_25,task_26,task_27,task_28,task_29,task_30,task_31,task_32,task_33,task_34,task_35,task_36,task_37,task_38,task_39,task_40,task_41,task_42,task_43,task_44,task_45,task_46,task_47,task_48,task_49,task_50,task_51,task_52,task_53,task_54,task_55,task_56,task_57,task_58,task_59,task_60"

echo "==== 3att v2 start $(date -u +%FT%TZ) ====" >> "$LOG"

echo "---- [1] baseline remaining-40 ----" >> "$LOG"
uv run python scripts/phase2_exp144_ablation.py --workers 12 --n-attempts 3 --tasks "$REMAIN" >> "$LOG" 2>&1
echo "---- [1] baseline remaining-40 exit=$? ----" >> "$LOG"

echo "---- [2] answer_shape+prose full-60 ----" >> "$LOG"
EXP144_ANSWER_SHAPE=1 EXP144_PROSE=1 uv run python scripts/phase2_exp144_ablation.py --workers 12 --n-attempts 3 >> "$LOG" 2>&1
echo "---- [2] combined full-60 exit=$? ----" >> "$LOG"

echo "==== 3att v2 done $(date -u +%FT%TZ) ====" >> "$LOG"
