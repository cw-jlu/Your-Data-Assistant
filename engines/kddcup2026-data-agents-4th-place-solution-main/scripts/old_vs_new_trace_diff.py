"""Compare old v5 vs new v5_compare bench: find which tasks regressed.

Old: artifacts/bench_phased_vote3_full50_exp126_exp123/ (= submitted v5, mean=0.8)
New: artifacts/runs/exp_122_v5_compare_002/                (= recent, mean=0.74)

For each task with old=1.0 but new<1.0:
  - inspect trace.log diff (= step count, action seq, error count)
  - show old final SQL vs new final SQL (where extractable)
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OLD = REPO / "artifacts" / "bench_phased_vote3_full50_exp126_exp123"
NEW = REPO / "artifacts" / "runs" / "exp_122_v5_compare_002"


def gt_scores():
    """Load per-task scores for both runs."""
    old_s = json.load(open(OLD / "summary.json"))["scores"]
    new_s = {}
    eval_csv = NEW / "evaluation.csv"
    if eval_csv.exists():
        with open(eval_csv) as f:
            for r in csv.DictReader(f):
                new_s[r.get("task_id") or r["task"]] = float(r.get("official_score_lambda_0_5") or r.get("score", 0))
    return old_s, new_s


def parse_trace_stats(trace_path: Path) -> dict:
    if not trace_path.exists():
        return {}
    text = trace_path.read_text()
    actions = Counter(re.findall(r"action=(\w+)", text))
    n_err = len(re.findall(r"result ok=False", text))
    n_steps = sum(actions.values())
    transients = len(re.findall(r"transient", text))
    # Extract last answer_from_sql SQL
    sql_match = re.findall(r"action=answer_from_sql.*?'sql': \"(.+?)\".*?action_input", text, re.DOTALL)
    if not sql_match:
        sql_match = re.findall(r"action=answer_from_sql.*?'sql': '(.+?)'.*?\)", text, re.DOTALL)
    return {
        "n_steps": n_steps, "n_err": n_err,
        "transients": transients,
        "actions": dict(actions),
        "last_sql": (sql_match[-1][:200] if sql_match else ""),
    }


def main():
    old_s, new_s = gt_scores()
    rows = []
    for tid, old_score in old_s.items():
        new_score = new_s.get(tid)
        if new_score is None:
            continue
        if new_score >= old_score - 0.01:
            continue  # not regressed
        # Compare attempts (= attempt_00 only for clarity)
        old_attempt = OLD / tid / "attempt_00" / "trace.log"
        new_attempt = NEW / tid / "attempt_00" / "trace.log"
        old_stats = parse_trace_stats(old_attempt)
        new_stats = parse_trace_stats(new_attempt)
        rows.append({
            "tid": tid,
            "old_score": old_score,
            "new_score": new_score,
            "delta": old_score - new_score,
            "old_steps": old_stats.get("n_steps", 0),
            "new_steps": new_stats.get("n_steps", 0),
            "old_err": old_stats.get("n_err", 0),
            "new_err": new_stats.get("n_err", 0),
            "old_transients": old_stats.get("transients", 0),
            "new_transients": new_stats.get("transients", 0),
            "old_last_sql": old_stats.get("last_sql", "")[:100],
            "new_last_sql": new_stats.get("last_sql", "")[:100],
        })

    rows.sort(key=lambda r: -r["delta"])
    print(f"=== Regressed tasks: {len(rows)} ===\n")
    for r in rows:
        print(f"## {r['tid']}: {r['old_score']:.2f} → {r['new_score']:.2f} (Δ -{r['delta']:.2f})")
        print(f"  steps:    old={r['old_steps']} new={r['new_steps']}")
        print(f"  errors:   old={r['old_err']} new={r['new_err']}")
        print(f"  trans:    old={r['old_transients']} new={r['new_transients']}")
        print(f"  old_sql:  {r['old_last_sql'][:160]}")
        print(f"  new_sql:  {r['new_last_sql'][:160]}")
        print()

    # Aggregate
    if rows:
        print("\n=== Aggregate over regressions ===")
        print(f"  total Δ:           {sum(r['delta'] for r in rows):.2f}")
        print(f"  avg old steps:     {sum(r['old_steps'] for r in rows)/len(rows):.1f}")
        print(f"  avg new steps:     {sum(r['new_steps'] for r in rows)/len(rows):.1f}")
        print(f"  avg old errors:    {sum(r['old_err'] for r in rows)/len(rows):.2f}")
        print(f"  avg new errors:    {sum(r['new_err'] for r in rows)/len(rows):.2f}")
        print(f"  avg old transient: {sum(r['old_transients'] for r in rows)/len(rows):.2f}")
        print(f"  avg new transient: {sum(r['new_transients'] for r in rows)/len(rows):.2f}")


if __name__ == "__main__":
    main()
