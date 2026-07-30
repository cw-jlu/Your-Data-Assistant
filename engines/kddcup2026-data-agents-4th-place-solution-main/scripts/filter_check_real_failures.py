"""Test loose filter checker on ACTUAL failing agent SQLs from exp_122_column_auditor_012."""
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

RUN = REPO / "artifacts" / "runs" / "exp_122_column_auditor_012"

# zero_recall failures
FAILS = ["task_25", "task_80", "task_89", "task_163", "task_169", "task_180", "task_199"]


_LISTER_SYS = """List the filter conditions a SQL query MUST implement to correctly answer
this question. Each condition is a short natural-language phrase.

Examples (= unrelated topics):
Q: "Find books published before 1990 in the fiction genre"
["published before 1990", "fiction genre"]

Q: "Top albums where average track score exceeds 8.0"
["average track score exceeds 8.0"]

Output: JSON array of short phrases. ONLY the JSON array."""


_CHECKER_SYS = """For each filter concept, check if the SQL implements it.
Output JSON:
  {"checks": [{"concept": "<phrase>", "present": true/false, "evidence": "<SQL fragment or 'not found'>"}],
   "all_present": true/false,
   "missing": ["<list of missing concepts>"]}

Be LENIENT: if a SQL fragment plausibly implements the concept, mark present=true.

ONLY the JSON."""


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


def extract_agent_sql(trace_path: Path) -> str:
    """Get final answer_from_sql call from trace."""
    if not trace_path.exists():
        return ""
    t = json.load(open(trace_path))
    sqls = []
    for s in t.get("steps", []):
        if s.get("action") == "answer_from_sql":
            sql = (s.get("action_input") or {}).get("sql", "")
            if sql: sqls.append(sql)
    return sqls[-1] if sqls else ""


def list_and_check(question, sql, model):
    r1 = model.complete(
        [ModelMessage(role="system", content=_LISTER_SYS),
         ModelMessage(role="user", content=f"Q: {question}\n\nFilters:")],
        enable_thinking=False, max_tokens=512)
    m = re.search(r"\[.*\]", r1, re.DOTALL)
    concepts = []
    if m:
        try: concepts = json.loads(m.group(0))
        except: pass
    r2 = model.complete(
        [ModelMessage(role="system", content=_CHECKER_SYS),
         ModelMessage(role="user", content=f"Concepts: {json.dumps(concepts)}\n\nSQL: {sql}\n\nCheck:")],
        enable_thinking=False, max_tokens=1024)
    m = re.search(r"\{.*\}", r2, re.DOTALL)
    if not m: return concepts, {"all_present": "?", "missing": []}
    try: return concepts, json.loads(m.group(0))
    except: return concepts, {"all_present": "?", "missing": []}


def run_one(tid):
    ds = DABenchPublicDataset(root_dir=REPO/"data"/"public"/"input")
    task = ds.get_task(tid)
    sql = extract_agent_sql(RUN/tid/"trace.json")
    if not sql:
        return {"tid": tid, "concepts": [], "missing": [], "all_present": None, "agent_sql": "(none)"}
    model = make_model()
    concepts, verdict = list_and_check(task.question, sql, model)
    return {
        "tid": tid, "question": task.question[:80],
        "agent_sql": sql[:200],
        "concepts": concepts,
        "all_present": verdict.get("all_present"),
        "missing": verdict.get("missing", []),
    }


def main():
    print(f"=== loose filter check on {len(FAILS)} real failures ===\n")
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(run_one, t): t for t in FAILS}
        results = []
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
    for r in sorted(results, key=lambda x: x["tid"]):
        print(f"## {r['tid']}: all_present={r['all_present']}")
        print(f"  Q: {r.get('question','')}")
        print(f"  concepts: {r['concepts']}")
        print(f"  missing:  {r['missing']}")
        print(f"  agent_sql: {r['agent_sql'][:160]}")
        print()
    n_caught = sum(1 for r in results if r["all_present"] is False)
    print(f"caught (= flagged missing): {n_caught} / {len(results)}")


if __name__ == "__main__":
    main()
