"""Use exp_109's 3 historic runs as 3 attempts; compare 5 vote strategies on 10 tasks.

For each task, fetch each of exp_109_plan_first_strengthen_001/002/003 's:
  - task answer (= columns + rows from trace.json)
  - terminal SQL (= last answer_from_sql or answer_from_python step's input)

Then apply 5 strategies (= maj / uni / v3 / v4 / llm) with schema + per-attempt SQL.

Usage:
    uv run python scripts/compare_vote_e109.py
"""
from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from dotenv import load_dotenv
load_dotenv()

from kobushi_core.benchmark.schema import AnswerTable
from kobushi_core.eval.csv_compare import _evaluate_task, EvaluationOptions
from kobushi_core.benchmark.dataset import DABenchPublicDataset
from kobushi_core.model import OpenAIModelAdapter, ModelMessage

# Reuse strategies from compare_vote_strategies
from compare_vote_strategies import (
    _norm, _signature, _majority, _union,
    _token_set, _adaptive_v3, _adaptive_v4,
    _llm_router, _make_judge_model, score,
)


TARGET_TASKS = [
    "task_27", "task_38", "task_196", "task_259", "task_330",
    "task_352", "task_379", "task_418", "task_180", "task_86",
]
RUNS = ["001", "002", "003"]


def load_attempt_from_e109(task_id, run):
    """Return (AnswerTable | None, sql | None) for one (task, run)."""
    p = REPO / f"artifacts/runs/exp_109_plan_first_strengthen_{run}/{task_id}/trace.json"
    if not p.exists(): return None, None
    try:
        t = json.loads(p.read_text())
    except Exception:
        return None, None
    ans = t.get("answer")
    if not ans or not ans.get("columns"):
        return None, _extract_sql_from_steps(t.get("steps", []))
    cols = list(ans["columns"])
    rows = [list(r) for r in ans.get("rows") or []]
    sql = _extract_sql_from_steps(t.get("steps", []))
    return AnswerTable(columns=cols, rows=rows), sql


def _extract_sql_from_steps(steps):
    """Find the terminal SQL or python answer from steps."""
    for s in reversed(steps):
        action = s.get("action", "")
        ai = s.get("action_input", {})
        if action == "answer_from_sql":
            if isinstance(ai, dict) and "sql" in ai:
                return ai["sql"]
        elif action == "answer_from_python":
            if isinstance(ai, dict) and "code" in ai:
                return f"# Python (not SQL):\n{ai['code'][:1000]}"
        elif action == "answer":
            return f"# Manual answer (no SQL): cols={ai.get('columns', [])}"
    return None


def schema_for(task_id):
    try:
        from experiments.exp_122_column_auditor.tools.duckdb_unified import describe_catalog
        ctx = REPO / "data" / "public" / "input" / task_id / "context"
        return describe_catalog(ctx)
    except Exception:
        return None


def main():
    use_llm = bool(os.environ.get("AGENT_API_BASE"))
    judge = _make_judge_model() if use_llm else None
    ds = DABenchPublicDataset(root_dir=REPO / "data" / "public" / "input")

    import time
    rows_out = []
    for tid in TARGET_TASKS:
        t0 = time.time()
        print(f"  {tid}: loading...", flush=True)
        attempts = []
        sqls = []
        for run in RUNS:
            ans, sql = load_attempt_from_e109(tid, run)
            attempts.append(ans)
            sqls.append(sql)
        non_none = [a for a in attempts if a is not None]
        if not non_none:
            print(f"    skip (no attempts)", flush=True)
            continue

        gold_p = REPO / "data" / "public" / "output" / tid / "gold.csv"
        question = ds.get_task(tid).question
        schema = schema_for(tid)

        s_maj = score(_majority(non_none), gold_p)
        s_uni = score(_union(non_none), gold_p)
        s_v3 = score(_adaptive_v3(non_none), gold_p)
        s_v4 = score(_adaptive_v4(non_none), gold_p)
        s_llm = None
        if judge:
            t_llm = time.time()
            s_llm = score(
                _llm_router(attempts, question, judge, sqls=sqls, schema=schema),
                gold_p,
            )
            print(f"    llm took {time.time()-t_llm:.1f}s", flush=True)
        llm_str = f"{s_llm:.2f}" if isinstance(s_llm, float) else str(s_llm)
        print(f"    {tid} done in {time.time()-t0:.1f}s: maj={s_maj:.2f} llm={llm_str}", flush=True)
        rows_out.append({
            "task": tid, "majority": s_maj, "union": s_uni,
            "v3": s_v3, "v4": s_v4, "llm": s_llm,
            "n_attempts": len(non_none), "n_sqls": sum(1 for s in sqls if s),
        })

    # Print table
    print(f"{'task':<10} {'attempts':>4} {'sqls':>4} {'maj':>6} {'uni':>6} {'v3':>6} {'v4':>6} {'llm':>6}  best")
    print("-" * 90)
    for r in rows_out:
        scores = {k: r[k] for k in ("majority", "union", "v3", "v4", "llm") if r[k] is not None}
        best_v = max(scores.values()) if scores else 0
        labels = [k for k, v in scores.items() if v == best_v]
        flag = "/".join(labels)
        print(f"  {r['task']:<10} {r['n_attempts']:>4} {r['n_sqls']:>4} {r['majority']:>6.2f} {r['union']:>6.2f} {r['v3']:>6.2f} {r['v4']:>6.2f} {r['llm'] if r['llm'] is not None else 0:>6.2f}  {flag}")
    print()
    n = len(rows_out)
    for k in ("majority", "union", "v3", "v4", "llm"):
        vals = [r[k] for r in rows_out if r[k] is not None]
        if vals:
            print(f"  {k}: sum={sum(vals):.2f}  mean={sum(vals)/len(vals):.3f}  (n={len(vals)})")


if __name__ == "__main__":
    main()
