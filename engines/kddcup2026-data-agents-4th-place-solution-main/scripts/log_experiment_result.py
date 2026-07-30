"""Append a row to EXPERIMENTS.md after a benchmark completes.

Reads the run's evaluation.json and writes an entry with score + diagnosis
breakdown. Idempotent: if a row for the same run dir already exists, skips.

Usage:
    uv run python scripts/log_experiment_result.py <run_dir>
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_MD = REPO_ROOT / "EXPERIMENTS.md"
SUMMARY_TABLE_HEADER = "## Summary"


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: log_experiment_result.py <run_dir>", file=sys.stderr)
        return 2
    run_dir = Path(sys.argv[1]).resolve()
    eval_path = run_dir / "evaluation.json"
    if not eval_path.exists():
        print(f"ERROR: {eval_path} not found", file=sys.stderr)
        return 1

    payload = json.loads(eval_path.read_text())
    score = payload["official_score_lambda_0_5_mean"]
    perfect = payload["perfect_recall_no_extras_count"]
    wext = payload["perfect_recall_with_extras_count"]
    partial = payload["partial_recall_count"]
    zero = payload["zero_recall_count"]
    missing = payload["missing_prediction_count"]
    n = payload["task_count"]

    # exp_NNN_<slug> from "exp_NNN_<slug>_NNN" run dir name
    run_name = run_dir.name
    m = re.match(r"^(exp_\d{3}_[a-z0-9_]+?)_\d{3}$", run_name)
    exp_name = m.group(1) if m else run_name

    answered = perfect + wext + partial + zero
    row = (
        f"| {exp_name} | – | – | – | – | "
        f"**{score:.4f}** | {perfect}/{n}"
        f"{' (+' + str(wext) + ' wext)' if wext else ''} | "
        f"{answered}/{n} | {run_name} |"
    )

    text = EXPERIMENTS_MD.read_text()
    if run_name in text:
        print(f"already logged: {run_name}", file=sys.stderr)
        return 0

    # Insert before the closing line of the summary table (last "|" line in §Summary block)
    lines = text.splitlines()
    out: list[str] = []
    inserted = False
    in_summary = False
    summary_table_end = -1
    for i, line in enumerate(lines):
        if line.startswith("## Summary"):
            in_summary = True
        elif in_summary and line.startswith("## ") and not line.startswith("## Summary"):
            summary_table_end = i - 1
            break
        out.append(line)

    if summary_table_end < 0:
        # fall back: append at the very end of file
        out.append("")
        out.append(row)
    else:
        # find last table row before summary_table_end
        last_row_idx = -1
        for j in range(summary_table_end, -1, -1):
            if out[j].lstrip().startswith("|"):
                last_row_idx = j
                break
        if last_row_idx >= 0:
            out.insert(last_row_idx + 1, row)
        else:
            out.append(row)

    EXPERIMENTS_MD.write_text("\n".join(out) + "\n")
    print(f"appended row for {exp_name} (score={score:.4f}, perfect={perfect}/{n})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
