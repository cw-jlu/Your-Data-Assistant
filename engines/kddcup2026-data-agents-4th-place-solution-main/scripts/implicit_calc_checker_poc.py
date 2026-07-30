"""POC: mechanical implicit-computation checker.

Patterns checked (all 5):
  A. percentage: Q has "%/percentage" → SQL needs "*100" or "CAST AS REAL/FLOAT"
  B. ratio: Q has "ratio/times more" → SQL needs division between two aggregates
  C. per_X: Q has "per <unit>" → SQL needs "/Amount" or similar division
  D. avg_X_ly: Q has "average monthly/yearly/daily" → SQL needs AVG, not SUM/N
  E. ordinal_tie: Q has "lowest/highest/best" → SQL should use filter-back, not LIMIT 1

Output per task: list of violations + overall pass/fail.

Test on:
- 7 zero-recall failures (= task_25/80/89/163/169/180/199)
- 7 passing sanity tasks
"""
from __future__ import annotations

import json, re, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

# ============ Pattern detection ============

PAT_PERCENT = re.compile(r"\b(percent|percentage|%)\b", re.IGNORECASE)
PAT_RATIO = re.compile(r"\b(ratio|times more|how many times|times compared to)\b", re.IGNORECASE)
PAT_PER_UNIT = re.compile(r"\bper\s+(unit|item|day|hour|minute|capita|person|customer)\b", re.IGNORECASE)
PAT_AVG_PERIOD = re.compile(r"\baverage\s+(monthly|yearly|daily|hourly|weekly)\b", re.IGNORECASE)
PAT_ORDINAL = re.compile(r"\b(lowest|highest|best|worst|maximum|minimum|smallest|largest|cheapest|most|least)\b", re.IGNORECASE)

# ============ SQL structure checks ============

def has_percent_scaling(sql: str) -> bool:
    """Look for * 100, *100., or CAST(...AS REAL/FLOAT) anywhere in SQL."""
    if re.search(r"\*\s*100\b", sql): return True
    if re.search(r"CAST\s*\(.+?AS\s+(REAL|FLOAT|DOUBLE)\)", sql, re.IGNORECASE): return True
    return False

def has_ratio_div(sql: str) -> bool:
    """Look for division between two aggregate-like expressions."""
    # Look for / sign with SUM/COUNT/CASE on both sides (loose)
    if re.search(r"(SUM|COUNT|CASE)\s*\([^)]*\)\s*\*?\s*/\s*", sql, re.IGNORECASE): return True
    return False

def has_division(sql: str) -> bool:
    """Look for any /column or /Amount style division (= for per_unit pattern)."""
    return bool(re.search(r"/\s*[a-zA-Z_]\w*", sql))

def has_avg(sql: str) -> bool:
    """Look for AVG(...) function."""
    return bool(re.search(r"\bAVG\s*\(", sql, re.IGNORECASE))

def has_sum_div_n(sql: str) -> bool:
    """Look for SUM(...)/<number> pattern."""
    return bool(re.search(r"SUM\s*\([^)]+\)\s*/\s*[\d.]+", sql, re.IGNORECASE))

def has_filter_back(sql: str) -> bool:
    """Look for WHERE col = (SELECT MIN/MAX/...) pattern."""
    return bool(re.search(r"WHERE.+?=\s*\(\s*SELECT\s+(MIN|MAX|AVG)", sql, re.IGNORECASE | re.DOTALL))

def has_limit_1(sql: str) -> bool:
    return bool(re.search(r"\bLIMIT\s+1\b", sql, re.IGNORECASE))


# ============ Per-task check ============

def check_task(question: str, sql: str) -> dict:
    violations = []
    notes = []
    # A. percentage
    if PAT_PERCENT.search(question):
        if not has_percent_scaling(sql):
            violations.append({"pattern": "A. percentage", "issue": "no *100 or CAST AS REAL — % formula likely missing"})
        else:
            notes.append("A. percentage scaling found")
    # B. ratio
    if PAT_RATIO.search(question):
        if not has_ratio_div(sql):
            violations.append({"pattern": "B. ratio", "issue": "no SUM/COUNT/CASE division — ratio formula missing"})
        else:
            notes.append("B. ratio div found")
    # C. per_unit
    if PAT_PER_UNIT.search(question):
        if not has_division(sql):
            violations.append({"pattern": "C. per_unit", "issue": "no /<col> division — 'per unit' interpretation missing"})
        else:
            notes.append("C. per_unit div found")
    # D. avg X-ly
    if PAT_AVG_PERIOD.search(question):
        if not has_avg(sql):
            violations.append({"pattern": "D. avg_X_ly", "issue": "no AVG() — likely SUM/N instead, gold uses AVG"})
        elif has_sum_div_n(sql):
            violations.append({"pattern": "D. avg_X_ly", "issue": "uses SUM(...)/N — should use AVG()/N"})
        else:
            notes.append("D. AVG found, no SUM/N")
    # E. ordinal tie
    if PAT_ORDINAL.search(question):
        if has_limit_1(sql) and not has_filter_back(sql):
            violations.append({"pattern": "E. ordinal_tie", "issue": "uses LIMIT 1 only — should use filter-back (WHERE col = (SELECT MIN/MAX)) for ties"})
        elif has_filter_back(sql):
            notes.append("E. filter-back found")
    return {"violations": violations, "notes": notes, "all_pass": len(violations) == 0}


# ============ Driver ============

RUN = REPO / "artifacts" / "runs" / "exp_122_column_auditor_012"


def get_agent_sql(tid: str) -> str:
    tp = RUN / tid / "trace.json"
    if not tp.exists(): return ""
    t = json.load(open(tp))
    sqls = [(s.get("action_input") or {}).get("sql", "") for s in t.get("steps", []) if s.get("action") == "answer_from_sql"]
    sqls = [s for s in sqls if s]
    return sqls[-1] if sqls else ""


def get_question(tid: str) -> str:
    return json.load(open(REPO / "data" / "public" / "input" / tid / "task.json"))["question"]


# Failing zero-recall + passing sanity
FAILS = ["task_25", "task_80", "task_89", "task_163", "task_169", "task_180", "task_199"]
SANITY = ["task_11", "task_22", "task_24", "task_27", "task_67", "task_75", "task_173", "task_269", "task_287", "task_305"]


def main():
    print("=" * 80)
    print("IMPLICIT-COMPUTATION CHECKER — testing on real agent SQLs\n")
    for label, tids in [("FAILURES (= should catch)", FAILS), ("SANITY (= should NOT flag)", SANITY)]:
        print(f"\n## {label}")
        print(f"{'task':10s} | violations")
        print("-" * 80)
        for tid in tids:
            q = get_question(tid)
            sql = get_agent_sql(tid)
            if not sql:
                print(f"  {tid:8s} | <no agent SQL>")
                continue
            r = check_task(q, sql)
            if r["violations"]:
                vs = "; ".join(f"{v['pattern']}: {v['issue'][:60]}" for v in r["violations"])
                print(f"  {tid:8s} | ❌ {vs}")
            else:
                print(f"  {tid:8s} | ✓ pass (notes: {r['notes']})")


if __name__ == "__main__":
    main()
