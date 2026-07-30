"""PoC: 5-stage LogicRAG-inspired pipeline (= exp_128).

5 tasks × 1 attempt. Each stage = a separate LLM call with its own system prompt.
Stages: 1 Interpreter → 2 Explorer → 3 Aggregator → 4 Answer → 5 Verifier.

Usage:
    uv run python scripts/test_logicrag_poc.py

Target tasks (= each stresses a different 50-task pitfall):
    task_25  : amount_vs_spent
    task_180 : per_unit_price
    task_344 : subset_data
    task_379 : atom_id_numeric_sort
    task_396 : id_anchor_prose
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from dotenv import load_dotenv
load_dotenv(REPO / ".env")

from kobushi_core.benchmark.dataset import DABenchPublicDataset
from kobushi_core.eval.csv_compare import _evaluate_task, EvaluationOptions
from kobushi_core.model import OpenAIModelAdapter, ModelMessage
from experiments.exp_122_column_auditor.tools.duckdb_unified import execute_sql

POC_DIR = REPO / "artifacts" / "logicrag_poc"
POC_DIR.mkdir(parents=True, exist_ok=True)

TASKS = ["task_25", "task_180", "task_344", "task_379", "task_396"]


# =========================================================================
# Model factory
# =========================================================================

def make_model(temperature: float = 0.6) -> OpenAIModelAdapter:
    return OpenAIModelAdapter(
        model="qwen3.5-35b-a3b",
        api_base=os.environ["AGENT_API_BASE"],
        api_key=os.environ["AGENT_API_KEY"],
        temperature=temperature,
        extra_headers={
            "CF-Access-Client-Id": os.environ.get("CF_ACCESS_CLIENT_ID", ""),
            "CF-Access-Client-Secret": os.environ.get("CF_ACCESS_CLIENT_SECRET", ""),
        },
        presence_penalty=0.0,
    )


def chat(model: OpenAIModelAdapter, system: str, user: str) -> str:
    return model.complete([
        ModelMessage(role="system", content=system),
        ModelMessage(role="user", content=user),
    ])


def extract_json(text: str) -> dict | list | None:
    """Find the first JSON block in the response (greedy, balanced braces)."""
    text = text.strip()
    # Try direct parse first
    try:
        return json.loads(text)
    except Exception:
        pass
    # Find ```json ... ``` block
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # Find first { ... } or [ ... ]
    for opener, closer in [("{", "}"), ("[", "]")]:
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(text)):
            if text[i] == opener:
                depth += 1
            elif text[i] == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except Exception:
                        break
    return None


# =========================================================================
# Context builders (= what each stage sees)
# =========================================================================

def build_schema_preview(context_dir: Path) -> str:
    """1k-token DB schema overview for Stage 1."""
    try:
        res = execute_sql(context_dir, "SHOW TABLES")
        # rows are list[list], col 0 is name
        tables = [r[0] for r in res["rows"]]
    except Exception as exc:
        return f"[schema preview unavailable: {exc}]"
    lines = []
    for tbl in tables[:20]:
        try:
            cols = execute_sql(context_dir, f"DESCRIBE {tbl}")
            # DESCRIBE returns [column_name, column_type, null, key, default, extra]
            col_info = ", ".join(f"{r[0]}:{r[1]}" for r in cols["rows"][:30])
            lines.append(f"- {tbl}: {col_info}")
        except Exception as exc:
            lines.append(f"- {tbl}: [error: {exc}]")
    return "\n".join(lines)


def build_doc_toc(context_dir: Path) -> str:
    """Prose doc TOC: list .md files + their h1/h2/h3 headings."""
    docs = list(context_dir.rglob("*.md")) + list(context_dir.rglob("*.txt"))
    if not docs:
        return "(no prose docs)"
    lines = []
    for d in docs[:5]:
        rel = d.relative_to(context_dir)
        lines.append(f"## {rel} ({d.stat().st_size // 1024} KB)")
        try:
            txt = d.read_text(errors="ignore")
            headings = re.findall(r"^#{1,4}\s+(.+)$", txt, re.MULTILINE)
            for h in headings[:10]:
                lines.append(f"  - {h.strip()}")
        except Exception:
            pass
    return "\n".join(lines)


def load_knowledge_md(context_dir: Path) -> str:
    p = context_dir / "knowledge.md"
    if not p.exists():
        return "(no knowledge.md)"
    txt = p.read_text(errors="ignore")
    return txt[:3000]  # cap to ~3000 chars


# =========================================================================
# Stage 1 — Question Interpreter
# =========================================================================

STAGE1_SYS = """You are the Question Interpreter for a data-analysis agent.
Your job: turn the user's question into a structured plan so the next stage can search efficiently.

OUTPUT — strictly valid JSON, no prose around it:
{
  "rephrased": "<one sentence, entity-explicit rephrasing>",
  "intent": "<one of: lookup | count | sum | avg | min/max | ratio | filter+list | compare | rank | other>",
  "result_shape": "<one of: scalar | single_row | multi_row | percentage>",
  "result_columns": ["<col1>", ...],
  "entities": [
    {"name": "<entity name>", "table": "<table>", "column": "<column>", "value_hint": "<exact value or pattern>"}
  ],
  "sub_questions": [
    {"id": "Q1", "text": "<atomic sub-question>", "depends_on": []}
  ],
  "verification_targets": ["<what to verify in stage 5>"],
  "pitfall_hints": ["<which pattern applies, if any>"]
}

PITFALL CATALOG (= choose 0+ that apply, name them exactly):
- amount_vs_spent: in budget tables, `amount` = budgeted/allocated value, `spent` = actual paid. If question says "cost" without further qualifier, prefer expense.cost (= per-line item) over budget.spent (= per-budget rollup)
- per_unit_price: "per unit" or "per X" means Price/Amount, NOT Price alone
- case_insensitive_title: exact title string may have case mismatch with stored value (e.g., "Computer Game" vs "Computer game")
- atom_id_numeric_sort: atom_id like TR000_4 — sort by numeric suffix (= split on "_" and CAST), NOT alphabetically
- id_anchor_prose: same entity appears in multiple prose sections; join via shared ID anchor
- directional_csv: connected.csv / hero_power.csv has both directions — apply DISTINCT bond_id / power_id when counting
- subset_data: _1k.db or partial prose may be subset; answer from available rows

CRITICAL RULES:
- USE ONLY the EXACT table/column names from the provided schema. DO NOT invent table names (e.g., "Financials", "Events"). Copy names verbatim, INCLUDING case.
- If you're unsure which table holds an entity, list ALL candidate tables in "entities" rather than guessing.
- "rephrased": Make ALL implicit entities explicit, using REAL table.column names.
- "sub_questions": 1-5 atomic steps. Each step references real schema.
- Output ONLY the JSON, no commentary.
"""


def stage1_interpret(model, task, schema_preview, doc_toc, knowledge):
    user = f"""# Question
{task.question}

# DB schema preview
{schema_preview}

# Prose doc TOC
{doc_toc}

# Knowledge guide (knowledge.md, excerpt)
{knowledge}

Output the Question Plan JSON now."""
    out = chat(model, STAGE1_SYS, user)
    plan = extract_json(out)
    if not isinstance(plan, dict):
        # Fallback: identity rephrasing
        plan = {
            "rephrased": task.question,
            "intent": "other",
            "result_shape": "single_row",
            "result_columns": [],
            "entities": [],
            "sub_questions": [{"id": "Q1", "text": task.question, "depends_on": []}],
            "verification_targets": [],
            "pitfall_hints": [],
            "_fallback": True,
            "_raw": out[:500],
        }
    return plan


# =========================================================================
# Stage 2 — Explorer (= simplified: 1 LLM call that generates SQL per sub-Q)
# =========================================================================

STAGE2_SYS = """You are the Explorer. For each sub-question in the Question Plan, write ONE SQL query to gather the fact.

Available: DuckDB unified data layer. All CSV/JSON files are loaded as views; SQLite DBs as ATTACHed schemas. Query as: SELECT ... FROM <view_or_table>.

OUTPUT — strictly valid JSON array, one entry per sub_question:
[
  {"sub_question_id": "Q1", "sql": "<SELECT ...>", "expected_format": "<scalar|list|table>"},
  ...
]

RULES:
- One SQL per sub-question. Use LIMIT 200 for list/table queries.
- For prose-doc lookups, output sql=null and "doc_lookup": "<doc file path>"
- For each sub-question that depends on another's answer, write SQL that will be parameterized by the prior result. Indicate dependencies with "<<Q1_answer>>" placeholders.
- Output ONLY the JSON array.
"""


def stage2_explore(model, task, plan, schema_preview):
    user = f"""# Question Plan
{json.dumps(plan, indent=2, ensure_ascii=False)}

# DB schema
{schema_preview}

# Available prose docs
{[str(p.relative_to(task.context_dir)) for p in task.context_dir.rglob('*.md')][:5]}

Output the SQL plan now."""
    out = chat(model, STAGE2_SYS, user)
    sql_plan = extract_json(out)
    if not isinstance(sql_plan, list):
        sql_plan = []
    # Execute each SQL
    facts = []
    sub_answers = {}
    for entry in sql_plan:
        if not isinstance(entry, dict):
            continue
        sqid = entry.get("sub_question_id", "?")
        sql = entry.get("sql")
        if sql:
            # Substitute prior answers
            for pid, pans in sub_answers.items():
                # crude: substitute string representation
                if isinstance(pans, (str, int, float)):
                    sql = sql.replace(f"<<{pid}_answer>>", str(pans))
                elif isinstance(pans, list) and pans:
                    val_list = ",".join(f"'{v}'" if isinstance(v, str) else str(v) for v in pans)
                    sql = sql.replace(f"<<{pid}_answer>>", val_list)
            try:
                res = execute_sql(task.context_dir, sql, limit=200)
                rows = res["rows"][:10]  # cap reported rows
                cols = res["columns"]
                # Determine "answer"
                if len(rows) == 1 and len(cols) == 1:
                    answer = rows[0][0]
                elif len(cols) == 1:
                    answer = [r[0] for r in rows]
                else:
                    answer = {"columns": cols, "rows": rows}
                facts.append({
                    "sub_question_id": sqid,
                    "sql": sql,
                    "answer": answer,
                    "n_rows": len(res["rows"]),
                    "confidence": "high",
                })
                sub_answers[sqid] = answer
            except Exception as exc:
                facts.append({
                    "sub_question_id": sqid,
                    "sql": sql,
                    "error": str(exc)[:200],
                    "confidence": "low",
                })
        elif entry.get("doc_lookup"):
            doc_path = task.context_dir / entry["doc_lookup"]
            if doc_path.exists():
                txt = doc_path.read_text(errors="ignore")[:5000]
                facts.append({
                    "sub_question_id": sqid,
                    "doc_lookup": entry["doc_lookup"],
                    "excerpt": txt,
                    "confidence": "medium",
                })
    return {"sql_plan": sql_plan, "facts": facts}


# =========================================================================
# Stage 3 — Aggregator
# =========================================================================

STAGE3_SYS = """You are the Aggregator. Convert the exploration log into a Structured Brief.

OUTPUT — strictly valid JSON:
{
  "rephrased_question": "<from Stage 1>",
  "expected_result_shape": "<scalar | single_row | multi_row | percentage>",
  "expected_columns": [...],
  "relevant_tables": {"<table>": ["<col1>", ...]},
  "key_values": {"<entity>": "<exact value>"},
  "sub_question_answers": [
    {"id": "Q1", "answer": <...>, "evidence_summary": "<1-2 sentence>"}
  ],
  "join_path": "<table1.col1 = table2.col1 → table2.col2 = table3.col2>",
  "filter_predicates": ["<column op value>"],
  "ordering": "<asc/desc on column, if any>",
  "limit_or_distinct": "<LIMIT N | DISTINCT | none>",
  "unresolved_assumptions": ["<list>"],
  "verification_targets": [...]
}

RULES:
- Be ruthless: drop exploration noise
- "join_path" must use explicit table.col = table.col format
- "unresolved_assumptions" if Stage 2 had errors or low confidence
- Output ONLY the JSON.
"""


def stage3_aggregate(model, plan, exploration):
    user = f"""# Question Plan (Stage 1)
{json.dumps(plan, indent=2, ensure_ascii=False)}

# Exploration Log (Stage 2)
{json.dumps(exploration, indent=2, ensure_ascii=False)[:6000]}

Output the Structured Brief now."""
    out = chat(model, STAGE3_SYS, user)
    brief = extract_json(out)
    if not isinstance(brief, dict):
        brief = {
            "rephrased_question": plan.get("rephrased", ""),
            "_fallback": True,
            "_raw": out[:500],
        }
    return brief


# =========================================================================
# Stage 4 — Answer Generator
# =========================================================================

STAGE4_SYS = """You are the Answer Generator. Generate the final SQL based on the Structured Brief.

CONSTRAINTS:
- ONLY use the Brief — no further exploration
- Output: a single SQL query
- Match expected_result_shape and expected_columns exactly
- If unresolved_assumptions is non-empty, choose the most conservative interpretation

OUTPUT format: a ```sql\n<SQL>\n``` block. Nothing else.
"""


def stage4_answer(model, brief, context_dir):
    user = f"""# Structured Brief
{json.dumps(brief, indent=2, ensure_ascii=False)[:6000]}

Generate the final SQL now."""
    out = chat(model, STAGE4_SYS, user)
    # Extract SQL
    m = re.search(r"```(?:sql)?\s*\n?(.*?)\n?```", out, re.DOTALL)
    sql = m.group(1).strip() if m else out.strip()
    # Execute
    try:
        res = execute_sql(context_dir, sql, limit=500)
        return {"sql": sql, "columns": res["columns"], "rows": res["rows"]}
    except Exception as exc:
        return {"sql": sql, "error": str(exc)[:300]}


# =========================================================================
# Stage 5 — Independent Verifier (CoVe-style, simplified)
# =========================================================================

STAGE5_SYS = """You are an Independent Verifier. Verify a candidate answer using fresh SQL queries.

PROCESS:
1. Generate 1-3 verification questions that, if all answered consistently, would confirm the candidate
2. For each, write a SQL that independently computes a check
3. Output JSON for the next step to execute

OUTPUT — strictly valid JSON:
{
  "verification_questions": [
    {"q": "<verify Q>", "check_sql": "<SELECT ...>", "expected": "<should match candidate>"}
  ]
}

Output ONLY the JSON."""


def stage5_verify(model, question, brief, candidate, context_dir):
    user = f"""# Original Question
{question}

# Structured Brief
{json.dumps(brief, indent=2, ensure_ascii=False)[:3000]}

# Candidate Answer
SQL: {candidate.get('sql', '')[:500]}
Columns: {candidate.get('columns', [])}
First 5 rows: {candidate.get('rows', [])[:5]}

Generate the verification queries now."""
    out = chat(model, STAGE5_SYS, user)
    vplan = extract_json(out)
    if not isinstance(vplan, dict):
        return {"verdict": "pass", "_no_verify": True}
    vqs = vplan.get("verification_questions", [])
    results = []
    all_pass = True
    for vq in vqs[:3]:
        sql = vq.get("check_sql")
        if not sql:
            continue
        try:
            r = execute_sql(context_dir, sql, limit=10)
            results.append({"q": vq.get("q"), "rows": r["rows"], "expected": vq.get("expected")})
        except Exception as exc:
            results.append({"q": vq.get("q"), "error": str(exc)[:200]})
            all_pass = False
    return {
        "verification_questions": vqs,
        "results": results,
        "verdict": "pass" if all_pass else "uncertain",
    }


# =========================================================================
# Per-task runner
# =========================================================================

def run_one(tid: str, model_low, model_high) -> dict:
    print(f"\n{'='*70}", flush=True)
    print(f"## {tid}", flush=True)
    print(f"{'='*70}", flush=True)
    ds = DABenchPublicDataset(root_dir=REPO / "data" / "public" / "input")
    task = ds.get_task(tid)
    print(f"Q: {task.question}", flush=True)

    artifacts = POC_DIR / tid
    artifacts.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    schema_preview = build_schema_preview(task.context_dir)
    doc_toc = build_doc_toc(task.context_dir)
    knowledge = load_knowledge_md(task.context_dir)

    # Stage 1
    print(f"[stage 1] interpreting...", flush=True)
    plan = stage1_interpret(model_low, task, schema_preview, doc_toc, knowledge)
    (artifacts / "stage1_plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False))
    print(f"  rephrased: {plan.get('rephrased', '')[:120]}", flush=True)
    print(f"  intent: {plan.get('intent')}  shape: {plan.get('result_shape')}  pitfalls: {plan.get('pitfall_hints')}", flush=True)
    print(f"  sub_Qs: {len(plan.get('sub_questions', []))}", flush=True)

    # Stage 2
    print(f"[stage 2] exploring...", flush=True)
    exploration = stage2_explore(model_low, task, plan, schema_preview)
    (artifacts / "stage2_exploration.json").write_text(json.dumps(exploration, indent=2, ensure_ascii=False, default=str))
    n_ok = sum(1 for f in exploration["facts"] if f.get("confidence") == "high")
    n_err = sum(1 for f in exploration["facts"] if "error" in f)
    print(f"  facts: {len(exploration['facts'])} (high={n_ok}, error={n_err})", flush=True)

    # Stage 3
    print(f"[stage 3] aggregating...", flush=True)
    brief = stage3_aggregate(model_low, plan, exploration)
    (artifacts / "stage3_brief.json").write_text(json.dumps(brief, indent=2, ensure_ascii=False, default=str))
    print(f"  brief tables: {list(brief.get('relevant_tables', {}).keys())}", flush=True)
    print(f"  unresolved: {brief.get('unresolved_assumptions', [])[:3]}", flush=True)

    # Stage 4
    print(f"[stage 4] answering...", flush=True)
    candidate = stage4_answer(model_low, brief, task.context_dir)
    (artifacts / "stage4_candidate.json").write_text(json.dumps(candidate, indent=2, ensure_ascii=False, default=str))
    print(f"  SQL: {candidate.get('sql', '')[:200]}", flush=True)
    if "error" in candidate:
        print(f"  ERROR: {candidate['error']}", flush=True)
    else:
        print(f"  cols: {candidate.get('columns')}  n_rows: {len(candidate.get('rows', []))}", flush=True)

    # Stage 5
    print(f"[stage 5] verifying...", flush=True)
    verify = stage5_verify(model_low, task.question, brief, candidate, task.context_dir)
    (artifacts / "stage5_verify.json").write_text(json.dumps(verify, indent=2, ensure_ascii=False, default=str))
    print(f"  verdict: {verify.get('verdict')}", flush=True)

    # Evaluate
    if "rows" in candidate and "columns" in candidate:
        pred_p = artifacts / "prediction.csv"
        with open(pred_p, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(candidate["columns"])
            for row in candidate["rows"]:
                w.writerow(row)
        try:
            e = _evaluate_task(
                task_id=tid,
                prediction_path=pred_p,
                gold_path=REPO / "data" / "public" / "output" / tid / "gold.csv",
                options=EvaluationOptions(),
            )
            score = e.official_score_lambda_0_5
            recall = e.recall
            extras = e.extras_count
        except Exception as exc:
            score = 0.0
            recall = 0.0
            extras = -1
    else:
        score = 0.0
        recall = 0.0
        extras = -1

    dt = time.time() - t0
    print(f"  score={score:.4f}  recall={recall:.2f}  extras={extras}  latency={dt:.0f}s", flush=True)
    return {
        "task": tid,
        "score": score,
        "recall": recall,
        "extras": extras,
        "latency_s": int(dt),
        "stage1_pitfalls": plan.get("pitfall_hints", []),
        "stage5_verdict": verify.get("verdict"),
    }


# =========================================================================
# Main
# =========================================================================

def main():
    model_low = make_model(temperature=0.6)
    model_high = make_model(temperature=0.6)  # placeholder for verify @ different T if desired

    # Reference scores (= exp_122 / v5 baseline)
    BASELINE = {
        "task_25": 0.50,   # heuristic — 50-task report says ✅ with right SQL
        "task_180": 0.50,
        "task_344": 0.00,
        "task_379": 0.50,
        "task_396": 0.50,
    }

    results = []
    for tid in TASKS:
        try:
            results.append(run_one(tid, model_low, model_high))
        except Exception as exc:
            traceback.print_exc()
            results.append({"task": tid, "score": 0.0, "exception": repr(exc)})

    # Summary
    print("\n" + "=" * 70, flush=True)
    print("LOGICRAG POC SUMMARY (5 stages, single attempt)", flush=True)
    print("=" * 70, flush=True)
    print(f"{'task':>10} {'score':>8} {'recall':>8} {'extras':>8} {'latency':>9} {'pitfalls':>20} {'verdict':>10}", flush=True)
    total = 0.0
    for r in results:
        tid = r.get("task")
        sc = r.get("score", 0.0)
        rc = r.get("recall", 0.0)
        ex = r.get("extras", -1)
        lat = r.get("latency_s", 0)
        pf = ",".join(r.get("stage1_pitfalls", []))[:18]
        vd = r.get("stage5_verdict", "")[:8]
        print(f"{tid:>10} {sc:>8.4f} {rc:>8.2f} {ex:>8} {lat:>8}s {pf:>20} {vd:>10}", flush=True)
        total += sc
    mean = total / len(results) if results else 0.0
    print(f"{'mean':>10} {mean:>8.4f}", flush=True)

    (POC_DIR / "summary.json").write_text(json.dumps({
        "results": results,
        "mean_score": mean,
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
