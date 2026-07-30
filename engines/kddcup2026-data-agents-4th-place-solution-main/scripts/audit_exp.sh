#!/usr/bin/env bash
# Lightweight leak / cheating audit for a freshly-implemented experiment.
#
# Runs in <5 seconds. Catches the obvious mistakes:
#   - Tier 1 verbatim leak: public-50 question text or gold value pasted into prompt/preamble
#   - Tier 2 structural leak: known public-domain keywords (student_club, VYBER, ...) in source
#   - Tier 3 task-specific tuning: prompt mentions "task_NNN" by name
#
# Usage: scripts/audit_exp.sh <exp_name>
#   e.g. scripts/audit_exp.sh exp_061_my_new_idea
#
# Exit codes:
#   0  — PASS (clean)
#   1  — FAIL (Tier 1 or Tier 2 violation)
#   2  — Internal error (bad invocation, missing files)
set -uo pipefail

exp_name=${1:?usage: audit_exp.sh <exp_name>}
repo_root=$(cd "$(dirname "$0")/.." && pwd)
exp_dir="${repo_root}/src/experiments/${exp_name}"
public_input="${repo_root}/data/public/input"
public_output="${repo_root}/data/public/output"

if [ ! -d "$exp_dir" ]; then
    echo "FAIL: exp dir not found: $exp_dir" >&2
    exit 2
fi
if [ ! -d "$public_input" ]; then
    echo "FAIL: public input not found: $public_input" >&2
    exit 2
fi

# Files to scan: only .py and .yaml inside exp_dir, no __pycache__
mapfile -t source_files < <(find "$exp_dir" -type f \( -name '*.py' -o -name '*.yaml' \) ! -path '*/__pycache__/*')
if [ ${#source_files[@]} -eq 0 ]; then
    echo "FAIL: no .py/.yaml files in $exp_dir" >&2
    exit 2
fi

violations=0
warnings=0

# ─── Tier 1: verbatim public-50 question text ────────────────────────────
# For each public task, take the first 60 chars of the question and grep for
# it in the exp source. This catches direct copy-paste of question strings.
for task_json in "$public_input"/task_*/task.json; do
    [ -f "$task_json" ] || continue
    q=$(python3 -c "import json,sys; print(json.load(open('$task_json'))['question'])" 2>/dev/null)
    [ -z "$q" ] && continue
    # Use first 50 chars as the fingerprint — long enough to be unique, short
    # enough to survive minor rewrites.
    needle=$(printf '%s' "$q" | head -c 50)
    [ ${#needle} -lt 30 ] && continue
    if grep -lF "$needle" "${source_files[@]}" 2>/dev/null | head -1 | grep -q .; then
        tid=$(basename "$(dirname "$task_json")")
        echo "FAIL [Tier 1]: ${tid} question text leaked into source" >&2
        echo "  fragment: ${needle}..." >&2
        violations=$((violations + 1))
    fi
done

# ─── Tier 1.5: gold answer values (cell strings) ────────────────────────
# Catches pasted gold rows (e.g. "Trent,Smith" or "CZE\nSVK").
# We sample the first 3 distinctive rows from each gold.csv.
for gold_csv in "$public_output"/task_*/gold.csv; do
    [ -f "$gold_csv" ] || continue
    # Read rows 2-4 (skip header), keep only those with at least 6 chars
    while IFS= read -r row; do
        [ ${#row} -ge 8 ] || continue
        # Skip pure numeric rows (too generic to be a leak signal)
        printf '%s' "$row" | grep -qE '^[0-9., +-]+$' && continue
        if grep -lF "$row" "${source_files[@]}" 2>/dev/null | head -1 | grep -q .; then
            tid=$(basename "$(dirname "$gold_csv")")
            echo "FAIL [Tier 1.5]: ${tid} gold value leaked: '${row}'" >&2
            violations=$((violations + 1))
            break  # one violation per task is enough
        fi
    done < <(tail -n +2 "$gold_csv" | head -3)
done

# ─── Tier 2: banned domain-specific keywords ────────────────────────────
# These tokens identify the public-50 datasets. None of them should appear
# in our prompt/preamble/code (only in audit-warning comments).
# We use word-boundary matching with `grep -wF` to avoid false positives.
banned_terms=(
    student_club
    VYBER
    VYDAJ
    PRIJEM
    "Physics Teaching"
    "Cross Validated"
    "Yearly Kickoff"
    "October Meeting"
    "Belgian Jupiler"
    "Belgium Jupiler"
    Illinois
    Riverside
    "rec017x6R3hQqkLAo"
    "recVYIFAwjT91pnv7"
    Officers_meeting
    "Sherri Ramsey"
    "Trent Smith"
    "Tyler Hewitt"
    "Annabella Warren"
    "Alex Yoong"
    "David P"
    constructorId
    formula1
    transactions_1k
    yearmonth
    gas_station
    GasStationID
    superhero
    creatinine
    fibrinogen
    hasContentWarning
    frpm
    satscores
    AvgScrMath
    NumGE1500
)

for term in "${banned_terms[@]}"; do
    # Grep ALL source files for the term. Skip lines inside comments that contain
    # the audit warning header (those intentionally mention banned terms).
    matches=$(grep -nF "$term" "${source_files[@]}" 2>/dev/null \
        | grep -vE '(^|/)[^:]+:[0-9]+:[[:space:]]*#' \
        | grep -vE 'banned|forbid|do NOT|DO NOT|exclude|avoid|warning' || true)
    if [ -n "$matches" ]; then
        echo "FAIL [Tier 2]: banned term '$term' found in source:" >&2
        printf '%s\n' "$matches" | head -3 | sed 's/^/  /' >&2
        violations=$((violations + 1))
    fi
done

# ─── Tier 3: task_NNN by name ───────────────────────────────────────────
# Mentions of specific public tasks by ID. This is a soft warning — it's OK
# in comments explaining failures, but suspicious in actual prompt/code.
mentions=$(grep -nE 'task_[0-9]+' "${source_files[@]}" 2>/dev/null \
    | grep -vE '^[^:]+:[0-9]+:[[:space:]]*#' \
    | grep -vE 'docstring|"""|run-task|task_id|task_dirs' || true)
if [ -n "$mentions" ]; then
    count=$(printf '%s' "$mentions" | wc -l)
    echo "WARN [Tier 3]: ${count} 'task_NNN' references in source (verify they are not task-specific tuning):" >&2
    printf '%s\n' "$mentions" | head -3 | sed 's/^/  /' >&2
    warnings=$((warnings + 1))
fi

# ─── Summary ───────────────────────────────────────────────────────────────
if [ "$violations" -gt 0 ]; then
    echo "AUDIT FAIL: ${violations} violation(s), ${warnings} warning(s)"
    exit 1
fi

echo "AUDIT PASS: ${exp_name} clean (0 violations, ${warnings} warnings)"
exit 0
