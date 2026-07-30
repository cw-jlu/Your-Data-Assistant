"""Run BIRD gold SQL against DABench context, compare with DABench gold.csv.

For each task:
  1. Take BIRD gold SQL (= from data/external/bird/dev_20240627/dev.json)
  2. Execute against DABench context_dir via duckdb_unified
  3. Compare resulting rows with data/public/output/<task_id>/gold.csv
  4. Mark match/mismatch

Quick sanity check on whether BIRD SQL is usable as DABench gold.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from dotenv import load_dotenv
load_dotenv(REPO / ".env")

from kobushi_core.benchmark.dataset import DABenchPublicDataset
from experiments.exp_122_column_auditor.tools.duckdb_unified import execute_sql


CSV_IN = REPO / "artifacts" / "plan_verify_decisions_full50" / "gold_sql_full.csv"
GOLD_DIR = REPO / "data" / "public" / "output"
OUT = REPO / "artifacts" / "plan_verify_decisions_full50" / "bird_sql_verify.csv"


def load_gold_csv(task_id: str) -> tuple[list[str], list[tuple]]:
    p = GOLD_DIR / task_id / "gold.csv"
    if not p.exists():
        return [], []
    with open(p, newline="") as f:
        r = csv.reader(f)
        rows = list(r)
    if not rows:
        return [], []
    header = rows[0]
    body = [tuple(str(c).strip() for c in row) for row in rows[1:]]
    return header, body


def normalize_rows(rows) -> set[tuple]:
    out = set()
    for r in rows:
        norm = tuple(
            str(x).strip().lower()
            if x is not None else ""
            for x in r
        )
        # Trim float trailing zeros
        norm = tuple(
            re.sub(r"\.0+$", "", v) for v in norm
        )
        out.add(norm)
    return out


def run_one(task_id: str, gold_sql: str) -> dict:
    ds = DABenchPublicDataset(root_dir=REPO / "data" / "public" / "input")
    try:
        task = ds.get_task(task_id)
    except Exception as exc:
        return {"task_id": task_id, "status": "no_task", "error": str(exc)}
    try:
        res = execute_sql(task.context_dir, gold_sql)
    except Exception as exc:
        return {"task_id": task_id, "status": "exec_error", "error": str(exc)[:300]}
    actual_rows = res.get("rows", [])
    actual_cols = res.get("columns", [])
    gold_header, gold_rows = load_gold_csv(task_id)
    actual_set = normalize_rows(actual_rows)
    gold_set = normalize_rows(gold_rows)
    n_match = len(actual_set & gold_set)
    n_actual = len(actual_set)
    n_gold = len(gold_set)
    if n_gold == 0:
        status = "no_gold_csv"
    elif actual_set == gold_set:
        status = "match"
    elif actual_set >= gold_set:
        status = "match_superset"
    elif n_match >= n_gold * 0.5:
        status = "partial"
    else:
        status = "mismatch"
    return {
        "task_id": task_id,
        "status": status,
        "n_actual": n_actual,
        "n_gold": n_gold,
        "n_intersect": n_match,
        "actual_cols": "|".join(map(str, actual_cols)),
        "gold_cols": "|".join(gold_header),
    }


def main():
    rows = list(csv.DictReader(open(CSV_IN)))
    results = []
    for r in rows:
        tid = r["task_id"]
        sql = r["gold_sql"]
        res = run_one(tid, sql)
        results.append(res)
        s = res.get("status", "?")
        extra = ""
        if s in ("mismatch", "partial", "match_superset"):
            extra = f" actual={res.get('n_actual')} gold={res.get('n_gold')} ∩={res.get('n_intersect')}"
        elif s == "exec_error":
            extra = f" {res.get('error','')[:80]}"
        print(f"  {tid}: {s}{extra}", flush=True)

    with open(OUT, "w", newline="") as f:
        keys = ["task_id", "status", "n_actual", "n_gold", "n_intersect", "actual_cols", "gold_cols", "error"]
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)

    from collections import Counter
    c = Counter(r["status"] for r in results)
    print("\nSummary:", dict(c))
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
