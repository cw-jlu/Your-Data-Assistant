"""POC: math expert + SQL checker for implicit calculations.

A. math_expert(question) → pseudo-math formula (e.g., "result = SUM(CASE WHEN X) * 100 / COUNT(*)")
B. formula_checker(formula, sql) → does the SQL compute this formula? PASS / FAIL + reason

Test cases: failing tasks with implicit calc errors + sanity passing tasks.
"""
from __future__ import annotations

import json, os, re, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from dotenv import load_dotenv
load_dotenv(REPO / ".env")

from kobushi_core.benchmark.dataset import DABenchPublicDataset
from kobushi_core.model import OpenAIModelAdapter, ModelMessage


_EXPERT_SYS = """You are a math/SQL expert. Read the question and write the EXPECTED
calculation as a pseudo-math formula. Focus on the calculation structure, not
the specific column names.

The formula should make explicit:
  - aggregation type (SUM, COUNT, AVG, MIN, MAX)
  - filters on the numerator vs denominator
  - multiplication factors (*100 for %)
  - division by N (for X-ly averages of aggregated data)
  - DISTINCT scope

Output JSON:
  {"formula": "<pseudo-math expression>", "note": "<one-line key insight>"}

Examples (= unrelated topics):
Q: "What percentage of orders are shipped late?"
{"formula": "result = COUNT(orders WHERE status='late') * 100 / COUNT(orders)", "note": "percentage = COUNT/COUNT * 100"}

Q: "What is the average daily traffic in 2020?"
{"formula": "result = SUM(visits) / 366  (if records are yearly aggregates) OR AVG(daily_visits)  (if records are per-day)", "note": "depends on table grain"}

Q: "How many times did A win compared to B?"
{"formula": "result = COUNT(A.wins) / COUNT(B.wins)", "note": "ratio comparison"}

Q: "List the books with the highest rating"
{"formula": "result = SELECT books WHERE rating = MAX(rating)  (filter-back to get ties)", "note": "no calculation; superlative filter"}

ONLY the JSON. No other text."""


_CHECKER_SYS = """You check if a SQL query computes a given pseudo-math formula.

Reply with JSON:
  {"matches": true/false, "issue": "<short description if mismatch>"}

Be LENIENT (= same calculation expressed in equivalent SQL = match).
Be STRICT (= different aggregation, missing factor, wrong filter scope = mismatch).

ONLY the JSON. No other text."""


# Test cases: (task_id, agent's actual SQL, expected_status)
TESTS = [
    # FAILURES - checker should detect mismatch
    ("task_169",  # avg monthly: agent used SUM/12 instead of AVG
     "SELECT ROUND(SUM(y.Consumption) / 12.0) AS avg_monthly_consumption FROM yearmonth y JOIN customers c ON y.CustomerID = c.CustomerID WHERE c.Segment = 'SME' AND CAST(y.Date AS VARCHAR) LIKE '2013%'",
     "fail"),
    ("task_180",  # per unit: agent missing /Amount
     "SELECT ym.Consumption FROM yearmonth ym WHERE ym.Date = 201208 AND ym.CustomerID IN (SELECT CustomerID FROM transactions_1k WHERE ProductID = 5 AND Price > 29.0)",
     "fail"),
    ("task_396",  # percentage: agent missing % formula entirely
     "SELECT COUNT(s.id) FROM superhero s JOIN publisher p ON s.publisher_id = p.id WHERE p.publisher_name = 'Marvel Comics' AND s.height_cm BETWEEN 150 AND 180",
     "fail"),
    ("task_243",  # ratio: agent used wrong DISTINCT scope
     "SELECT CAST(COUNT(p.Id) AS REAL) / COUNT(DISTINCT v.Id) FROM votes v JOIN posts p ON v.UserId = p.OwnerUserId WHERE v.UserId = 24",
     "ok"),  # this IS correct ratio formula
    ("task_352",  # ratio: agent didn't use SUM(CASE)/SUM(CASE)
     "SELECT b.amount FROM budget b WHERE b.category = 'Advertisement' AND b.link_to_event = 'X'",
     "fail"),
    # SANITY - checker should NOT flag
    ("task_24",  # simple COUNT, no implicit calc
     "SELECT COUNT(T2.link_to_member) FROM event AS T1 INNER JOIN attendance AS T2 ON T1.event_id = T2.link_to_event WHERE T1.event_name = 'Women''s Soccer'",
     "ok"),
    ("task_67",  # simple AVG, matches Q
     "SELECT AVG(s.weight_kg) FROM superhero s JOIN gender g ON s.gender_id = g.id WHERE g.gender = 'Female'",
     "ok"),
    ("task_22",  # simple lookup, no calc
     "SELECT T2.date_received FROM member AS T1 INNER JOIN income AS T2 ON T1.member_id = T2.link_to_member WHERE T1.first_name = 'Connor' AND T1.last_name = 'Hilton' AND T2.source = 'Dues'",
     "ok"),
    ("task_283",  # correct percentage formula
     "SELECT CAST(COUNT(CASE WHEN c.colour = 'Blue' THEN 1 ELSE NULL END) AS REAL) * 100 / COUNT(s.id) FROM superhero s JOIN colour c ON s.eye_colour_id = c.id",
     "ok"),
]


def make_model():
    return OpenAIModelAdapter(
        model="qwen3.5-35b-a3b",
        api_base=os.environ["AGENT_API_BASE"],
        api_key=os.environ["AGENT_API_KEY"],
        temperature=0.0,
        extra_headers={
            "CF-Access-Client-Id": os.environ.get("CF_ACCESS_CLIENT_ID", ""),
            "CF-Access-Client-Secret": os.environ.get("CF_ACCESS_CLIENT_SECRET", ""),
        },
    )


def get_formula(question, model):
    r = model.complete(
        [ModelMessage(role="system", content=_EXPERT_SYS),
         ModelMessage(role="user", content=f"Q: {question}\n\nFormula:")],
        enable_thinking=False, max_tokens=512)
    m = re.search(r"\{.*\}", r, re.DOTALL)
    if not m: return {"formula": "?", "note": "parse failed"}
    try: return json.loads(m.group(0))
    except: return {"formula": "?", "note": "json error"}


def check(formula, sql, model):
    r = model.complete(
        [ModelMessage(role="system", content=_CHECKER_SYS),
         ModelMessage(role="user", content=f"Pseudo-formula: {formula}\n\nSQL: {sql}\n\nCheck:")],
        enable_thinking=False, max_tokens=512)
    m = re.search(r"\{.*\}", r, re.DOTALL)
    if not m: return {"matches": None, "issue": "parse failed"}
    try: return json.loads(m.group(0))
    except: return {"matches": None, "issue": "json error"}


def run_one(tid, sql, expected):
    ds = DABenchPublicDataset(root_dir=REPO/"data"/"public"/"input")
    task = ds.get_task(tid)
    model = make_model()
    formula = get_formula(task.question, model)
    verdict = check(formula.get("formula", ""), sql, model)
    correct = (
        (expected == "fail" and verdict.get("matches") is False) or
        (expected == "ok"   and verdict.get("matches") is True)
    )
    return {
        "tid": tid, "q": task.question[:60],
        "expected": expected,
        "formula": formula.get("formula", ""),
        "note": formula.get("note", ""),
        "matches": verdict.get("matches"),
        "issue": verdict.get("issue", ""),
        "correct": correct,
    }


def main():
    print(f"=== math expert + checker POC, {len(TESTS)} cases ===\n")
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(run_one, *t): t[0] for t in TESTS}
        results = [fut.result() for fut in as_completed(futs)]
    print(f"{'tid':10s} | exp  | matches | judge | formula / issue")
    print("-" * 130)
    for r in sorted(results, key=lambda x: x["tid"]):
        m = "✓" if r["correct"] else "✗"
        mt = str(r["matches"])[:5]
        snippet = (r["formula"][:50] + " | " + r["issue"][:60]) if not r["correct"] else r["formula"][:70]
        print(f"  {r['tid']:8s} | {r['expected']:4s} | {mt:^7s} | {m:^5s} | {snippet}")
    n = sum(1 for r in results if r["correct"])
    print(f"\naccuracy: {n}/{len(results)}")
    Path("artifacts/math_expert_poc").mkdir(parents=True, exist_ok=True)
    with open("artifacts/math_expert_poc/results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
