"""POC: 2-sub-agent filter check pipeline.

Pipeline:
  A. filter_extractor(question, schema_preview) → list of (col, op, value, source_phrase)
  B. sql_checker(filters, sql) → per-filter verdict + overall pass/fail

Test:
  - On historical failing tasks (task_80/89/180/199/200), check if pipeline
    would have caught the failure when fed the agent's bad SQL.
  - On historical passing tasks (task_22/24), check no false positive on
    correct SQL.

Output: per-task verdict table.
"""
from __future__ import annotations

import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from dotenv import load_dotenv
load_dotenv(REPO / ".env")

from kobushi_core.benchmark.dataset import DABenchPublicDataset
from kobushi_core.model import OpenAIModelAdapter, ModelMessage
from experiments.exp_122_column_auditor.tools.duckdb_unified import execute_sql


# Target tasks: (task_id, agent's bad SQL we want to check)
TARGETS = {
    # FAILURES — checker should FLAG
    "task_80": (
        "fail",
        "SELECT T2.number FROM qualifying AS T1 INNER JOIN drivers AS T2 ON T2.driverId = T1.driverId "
        "WHERE T1.raceId = 903 AND T1.q3 = '0:01:54'",  # exact match — gold uses LIKE '1:54%'
    ),
    "task_89": (
        "fail",
        "SELECT T1.time FROM results AS T1 INNER JOIN races AS T2 ON T1.raceId = T2.raceId "
        "WHERE T1.position = 2 AND T2.name = 'Chinese Grand Prix' AND T2.year = 2008",  # position vs rank
    ),
    "task_180": (
        "fail",
        "SELECT ym.Consumption FROM yearmonth ym WHERE ym.Date = 201208 "
        "AND ym.CustomerID IN (SELECT t.CustomerID FROM transactions_1k t "
        "WHERE t.ProductID = 5 AND t.Price/t.Amount > 29.00)",  # no Amount>0 guard
    ),
    "task_199": (
        "fail",
        "SELECT \"School Name\", \"Charter Funding Type\" FROM frpm "
        "WHERE \"County Name\" = 'Riverside' AND \"Charter Funding Type\" IS NOT NULL",  # wrong filter
    ),
    "task_200": (
        "fail",
        "SELECT COUNT(*) FROM atom a JOIN bond b ON a.molecule_id = b.molecule_id "
        "WHERE b.bond_type = '#' AND a.element = 'p' AND a.element = 'br'",  # AND instead of OR
    ),
    # PASSING (= sanity) — checker should NOT flag
    "task_22": (
        "ok",
        "SELECT T2.date_received FROM member AS T1 INNER JOIN income AS T2 "
        "ON T1.member_id = T2.link_to_member WHERE T1.first_name = 'Connor' "
        "AND T1.last_name = 'Hilton' AND T2.source = 'Dues'",
    ),
    "task_24": (
        "ok",
        "SELECT COUNT(T2.link_to_member) FROM event AS T1 "
        "INNER JOIN attendance AS T2 ON T1.event_id = T2.link_to_event "
        "WHERE T1.event_name = 'Women''s Soccer'",
    ),
}


_EXTRACTOR_SYS = """You extract FILTER CONDITIONS from a SQL question.

For each filter (= WHERE clause condition implied by the question), output a JSON object:
  {"phrase": "<the question phrase>", "intent": "<what the SQL must check>"}

Be specific about the column the question implies. Note units, exact values,
case-sensitivity. If the question mentions a divisor or ratio, note the
denominator guard. If "X or Y", note OR vs AND.

Examples (= unrelated demo cases):
Q: "Find books published before 1990 in the fiction genre"
[
  {"phrase": "published before 1990", "intent": "publication_year < 1990"},
  {"phrase": "fiction genre", "intent": "genre = 'fiction' (check case in data)"}
]

Q: "Top albums where average track score exceeds 8.0"
[
  {"phrase": "average track score exceeds 8.0", "intent": "AVG(score) > 8.0 grouped per album"}
]

Q: "Employees in HR department with salary above 50000 or bonus above 5000"
[
  {"phrase": "HR department", "intent": "department = 'HR'"},
  {"phrase": "salary above 50000 OR bonus above 5000", "intent": "salary > 50000 OR bonus > 5000 (OR, not AND)"}
]

Output: JSON array. ONLY the JSON array, no other text."""


_CHECKER_SYS = """You check whether a SQL query correctly implements a list of filter intents.

For each filter, check the SQL and decide:
  - "OK": SQL implements this filter correctly
  - "WRONG_COL": SQL uses a different column (= give the hint)
  - "MISSING": SQL doesn't check this filter
  - "WRONG_OP": SQL has the column but wrong operator/value (= e.g. = instead of IN, AND instead of OR)

Output JSON:
  {"verdicts": [{"filter": "<phrase>", "status": "OK|WRONG_COL|MISSING|WRONG_OP", "hint": "<fix suggestion>"}],
   "overall": "PASS|FAIL", "reason": "<short summary>"}

ONLY the JSON, no other text."""


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


def get_schema_preview(task) -> str:
    try:
        tbls = execute_sql(task.context_dir, "SHOW TABLES")["rows"]
        parts = []
        for r in tbls:
            t = r[0]
            try:
                cols = execute_sql(task.context_dir, f'DESCRIBE "{t}"')["rows"]
                parts.append(f"{t}({','.join(c[0] for c in cols)})")
            except: pass
        return " | ".join(parts)
    except: return ""


def extract_filters(question: str, schema: str, model) -> list[dict]:
    response = model.complete(
        [
            ModelMessage(role="system", content=_EXTRACTOR_SYS),
            ModelMessage(role="user", content=f"Schema: {schema}\n\nQ: {question}\n\nFilters:"),
        ],
        enable_thinking=False, max_tokens=1024,
    )
    m = re.search(r"\[.*\]", response, re.DOTALL)
    if not m: return []
    try: return json.loads(m.group(0))
    except: return []


def check_sql(filters: list[dict], sql: str, model) -> dict:
    response = model.complete(
        [
            ModelMessage(role="system", content=_CHECKER_SYS),
            ModelMessage(role="user", content=f"Filters: {json.dumps(filters)}\n\nSQL: {sql}\n\nVerdict:"),
        ],
        enable_thinking=False, max_tokens=1024,
    )
    m = re.search(r"\{.*\}", response, re.DOTALL)
    if not m: return {"overall": "?", "reason": "parse failed", "raw": response[:200]}
    try: return json.loads(m.group(0))
    except: return {"overall": "?", "reason": "json parse failed", "raw": response[:200]}


def run_one(task_id: str, expected: str, sql: str):
    ds = DABenchPublicDataset(root_dir=REPO/"data"/"public"/"input")
    task = ds.get_task(task_id)
    schema = get_schema_preview(task)
    model = make_model()
    filters = extract_filters(task.question, schema, model)
    verdict = check_sql(filters, sql, model)
    return {
        "task_id": task_id, "expected": expected,
        "question": task.question[:80],
        "filters": filters, "verdict": verdict,
        "checker_says": verdict.get("overall", "?"),
        "match_expected": (verdict.get("overall") == "FAIL") == (expected == "fail"),
    }


def main():
    print(f"=== filter check POC on {len(TARGETS)} tasks ===\n")
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(run_one, t, e, s): t for t, (e, s) in TARGETS.items()}
        results = []
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)

    print(f"{'task':10s} | {'expected':8s} | {'checker':8s} | {'match':5s} | reason")
    print("-" * 100)
    for r in sorted(results, key=lambda x: x["task_id"]):
        m = "✓" if r["match_expected"] else "✗"
        reason = (r["verdict"].get("reason") or "")[:60]
        print(f"  {r['task_id']:8s} | {r['expected']:8s} | {r['checker_says']:8s} | {m:5s} | {reason}")
    n_match = sum(1 for r in results if r["match_expected"])
    print(f"\naccuracy: {n_match}/{len(results)}")
    Path("artifacts/filter_check_poc").mkdir(parents=True, exist_ok=True)
    with open("artifacts/filter_check_poc/results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
