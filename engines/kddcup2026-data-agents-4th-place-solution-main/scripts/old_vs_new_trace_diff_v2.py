"""v2: compare old v5 vs new v5_compare. Use proper trace.json for new, attempt_00/trace.log for old."""
from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OLD = REPO / "artifacts" / "bench_phased_vote3_full50_exp126_exp123"
NEW = REPO / "artifacts" / "runs" / "exp_122_v5_compare_002"


def load_scores_from_eval(eval_csv: Path) -> dict[str, float]:
    if not eval_csv.exists():
        return {}
    out = {}
    with open(eval_csv) as f:
        for r in csv.DictReader(f):
            out[r["task_id"]] = float(r["official_score_lambda_0_5"])
    return out


def parse_old_trace(tid: str, attempt: int = 0) -> dict:
    tlog = OLD / tid / f"attempt_{attempt:02d}" / "trace.log"
    if not tlog.exists():
        return {}
    text = tlog.read_text()
    actions = Counter(re.findall(r"action=(\w+)", text))
    n_err = len(re.findall(r"result ok=False", text))
    n_steps = sum(actions.values())
    sql_match = re.findall(r"answer_from_sql.*?'sql': [\"'](.+?)[\"']", text)
    return {
        "n_steps": n_steps, "n_err": n_err,
        "phase_dist": _phase_count(text),
        "last_sql": (sql_match[-1][:200] if sql_match else ""),
    }


def parse_new_trace(tid: str) -> dict:
    tjson = NEW / tid / "trace.json"
    if not tjson.exists():
        return {}
    t = json.load(open(tjson))
    steps = t.get("steps", [])
    actions = Counter(s.get("action") for s in steps)
    n_err = sum(1 for s in steps if s.get("observation", {}).get("ok") is False)
    sqls = []
    for s in steps:
        if s.get("action") == "answer_from_sql":
            sql = (s.get("action_input") or {}).get("sql", "")
            if sql:
                sqls.append(sql)
    return {
        "n_steps": len(steps), "n_err": n_err,
        "phase_dist": Counter(s.get("observation", {}).get("phase") for s in steps),
        "last_sql": (sqls[-1][:200] if sqls else ""),
        "answer": t.get("answer"),
    }


def _phase_count(text: str) -> dict:
    return dict(Counter(re.findall(r"phase=(\w+)", text)))


def main():
    old_scores = load_scores_from_eval(OLD / "evaluation.csv")
    if not old_scores:
        # Fallback to summary.json
        old_scores = json.load(open(OLD / "summary.json"))["scores"]
    new_scores = load_scores_from_eval(NEW / "evaluation.csv")

    print(f"old n_tasks={len(old_scores)}, mean={sum(old_scores.values())/len(old_scores):.3f}")
    print(f"new n_tasks={len(new_scores)}, mean={sum(new_scores.values())/len(new_scores):.3f}")

    rows = []
    for tid, old_score in old_scores.items():
        new_score = new_scores.get(tid)
        if new_score is None: continue
        if new_score >= old_score - 0.01: continue
        old_st = parse_old_trace(tid)
        new_st = parse_new_trace(tid)
        rows.append({
            "tid": tid,
            "old_score": old_score, "new_score": new_score,
            "delta": old_score - new_score,
            "old_steps": old_st.get("n_steps", 0),
            "new_steps": new_st.get("n_steps", 0),
            "old_err": old_st.get("n_err", 0),
            "new_err": new_st.get("n_err", 0),
            "old_phase": old_st.get("phase_dist", {}),
            "new_phase": new_st.get("phase_dist", {}),
            "old_sql": old_st.get("last_sql", "")[:120],
            "new_sql": new_st.get("last_sql", "")[:120],
            "new_answer": new_st.get("answer"),
        })
    rows.sort(key=lambda r: -r["delta"])
    print(f"\n=== Regressed tasks: {len(rows)} (sorted by Δ) ===\n")
    for r in rows:
        print(f"## {r['tid']}: {r['old_score']:.2f} → {r['new_score']:.2f} (Δ -{r['delta']:.2f})")
        print(f"  steps:    old={r['old_steps']:2d} new={r['new_steps']:2d}")
        print(f"  errs:     old={r['old_err']:2d} new={r['new_err']:2d}")
        print(f"  old_phase: {r['old_phase']}")
        print(f"  new_phase: {r['new_phase']}")
        print(f"  new_answer: {str(r['new_answer'])[:160]}")
        if r['old_sql']:
            print(f"  old_sql:  {r['old_sql']}")
        if r['new_sql']:
            print(f"  new_sql:  {r['new_sql']}")
        print()


if __name__ == "__main__":
    main()
