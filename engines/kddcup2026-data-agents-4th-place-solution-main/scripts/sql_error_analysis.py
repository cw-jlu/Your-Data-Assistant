"""Categorize answer_from_sql + execute_sql errors from exp_122 traces.

For each ok=False result of those tools, extract the error message and bucket.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BENCH = REPO / "artifacts" / "bench_phased_vote3_full50_exp126_exp123"


def parse_errors(text: str) -> list[dict]:
    """Walk trace.log; for each (action, result) pair where ok=False, capture error."""
    errors = []
    current_action = None
    for line in text.splitlines():
        m = re.match(r"\[(\w+)\] phase=(\w+) step (\d+) action=(\w+)", line)
        if m:
            current_action = m.group(4)
            current_phase = m.group(2)
            current_tid = m.group(1)
            continue
        m = re.match(r"\[(\w+)\] phase=(\w+) step (\d+) result ok=(\w+) terminal=\w+ content=(.*)", line)
        if m and m.group(4) == "False":
            errors.append({
                "task_id": m.group(1), "phase": m.group(2),
                "action": current_action,
                "content": m.group(5)[:300],
            })
    return errors


def categorize(action: str, content: str) -> str:
    c = content.lower()
    if "0 rows" in c or "returned 0 rows" in c:
        return "zero_rows_returned"
    # Try to identify common error patterns
    if "binder error" in c or "column" in c and "not exist" in c:
        return "column_not_exist"
    if "binder error" in c and "group by" in c:
        return "groupby_violation"
    if "catalog error" in c and "table" in c and "not exist" in c:
        return "table_not_exist"
    if "catalog error" in c:
        return "catalog_other"
    if "syntax" in c or "parser error" in c:
        return "syntax_error"
    if "type mismatch" in c or "no function matches" in c:
        return "type_mismatch"
    if "phase-gate" in c or "phase gate" in c:
        return "phase_gate_reject"
    if "review" in c or "rule" in c and "violation" in c:
        return "review_rule_violation"
    if "io error" in c or "file" in c and "not found" in c:
        return "file_not_found"
    if "conversion error" in c or "cast" in c:
        return "cast_error"
    if "prose" in c and ("md" in c or "txt" in c):
        return "prose_guard"
    if "no rows" in c or "row_count': 0" in c or "empty" in c:
        return "empty_result"
    if "transition" in c:
        return "transition_error"
    if "column count" in c or "column_count" in c:
        return "column_count_mismatch"
    return "other"


def main():
    all_errors = []
    for tdir in BENCH.glob("task_*"):
        for adir in tdir.glob("attempt_*"):
            tlog = adir / "trace.log"
            if not tlog.exists(): continue
            errs = parse_errors(tlog.read_text())
            all_errors.extend(errs)

    # Filter to SQL-tool errors
    sql_errs = [e for e in all_errors if e["action"] in ("answer_from_sql", "execute_sql")]
    bucket = Counter()
    samples = defaultdict(list)
    for e in sql_errs:
        cat = categorize(e["action"], e["content"])
        bucket[cat] += 1
        if len(samples[cat]) < 3:
            samples[cat].append({"task": e["task_id"], "phase": e["phase"], "action": e["action"], "content": e["content"][:200]})

    print(f"=== SQL-tool errors: {len(sql_errs)} total ===\n")
    for cat, n in bucket.most_common():
        print(f"## {cat}: {n} ({n/len(sql_errs)*100:.1f}%)")
        for s in samples[cat][:2]:
            print(f"  - [{s['task']}/{s['phase']}/{s['action']}] {s['content']}")
        print()

    # Per-action breakdown
    print("\n## By action:")
    action_split = Counter((e["action"]) for e in sql_errs)
    for a, n in action_split.items():
        print(f"  {a}: {n}")


if __name__ == "__main__":
    main()
