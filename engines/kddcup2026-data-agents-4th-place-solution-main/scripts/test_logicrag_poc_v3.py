"""PoC v3: 5-stage pipeline with markdown intermediate + real ReAct sub-agents.

Changes vs v2:
- Stage 1/3/4 = 1-shot LLM with MARKDOWN output (no JSON parsing)
- Stage 2 (Explorer) = run_sql_subagent (= existing ReAct + SQL tools, isolated context)
- Stage 5 (Verifier) = run_sql_subagent for independent verification

Usage:
    uv run python scripts/test_logicrag_poc_v3.py
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
from experiments.exp_113_subagent_orchestrator.subagent import run_sql_subagent

POC_DIR = REPO / "artifacts" / "logicrag_poc_v3"
POC_DIR.mkdir(parents=True, exist_ok=True)

TASKS = ["task_25", "task_180", "task_344", "task_379", "task_396"]


# =========================================================================
# Model + chat helpers
# =========================================================================

def make_model(temperature: float = 0.6) -> OpenAIModelAdapter:
    """qwen3.5-35b-a3b with thinking enabled, default 81K budget."""
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


def chat(model: OpenAIModelAdapter, system: str, user: str, enable_thinking: bool = False) -> str:
    """1-shot chat. For Stage 1/3/4 (= static interpretation/aggregation/SQL emission)
    enable_thinking defaults to False — these stages don't benefit from
    multi-step reasoning and thinking budget blows latency. Sub-agents
    (Stage 2/5) keep thinking enabled via ReActAgent's own model handle."""
    return model.complete(
        [
            ModelMessage(role="system", content=system),
            ModelMessage(role="user", content=user),
        ],
        enable_thinking=enable_thinking,
    )


def build_schema_preview(context_dir: Path) -> str:
    try:
        res = execute_sql(context_dir, "SHOW TABLES")
        tables = [r[0] for r in res["rows"]]
    except Exception as exc:
        return f"[schema preview unavailable: {exc}]"
    lines = []
    for tbl in tables[:20]:
        try:
            cols = execute_sql(context_dir, f"DESCRIBE {tbl}")
            col_info = ", ".join(f"{r[0]}:{r[1]}" for r in cols["rows"][:30])
            lines.append(f"- {tbl}: {col_info}")
        except Exception as exc:
            lines.append(f"- {tbl}: [error: {exc}]")
    return "\n".join(lines)


def build_doc_toc(context_dir: Path) -> str:
    docs = list(context_dir.rglob("*.md")) + list(context_dir.rglob("*.txt"))
    docs = [d for d in docs if d.name != "knowledge.md"]
    if not docs:
        return "(no prose docs besides knowledge.md)"
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


def load_knowledge_md(context_dir: Path, cap: int = 3000) -> str:
    p = context_dir / "knowledge.md"
    if not p.exists():
        return "(no knowledge.md)"
    return p.read_text(errors="ignore")[:cap]


# =========================================================================
# Stage 1 — Question Interpreter (1-shot, markdown out)
# =========================================================================

STAGE1_SYS = """You are the Question Interpreter for a data-analysis agent.
Turn the user's question into a structured plan in MARKDOWN. The next stage will read it directly.

OUTPUT TEMPLATE (= keep section headers exactly):

# Question Plan

**Rephrased**: <one sentence, entity-explicit rephrasing using REAL table.column names>

**Intent**: <lookup | count | sum | avg | min/max | ratio | filter+list | compare | rank | other>

**Result shape**: <scalar | single_row | multi_row | percentage>
- If superlative (min/max/top), be EXPLICIT that tied rows must ALL be returned (= multi_row by default unless query says "the one with the highest...")

**Result columns**: <comma-separated list of column names ONLY — match gold CSV format>
- Do NOT add the metric column (e.g., "cost") unless the question explicitly asks for the value too.

**Pitfalls** (= zero or more from this catalog):
- amount_vs_spent: budget.amount = budgeted, budget.spent = paid. "cost" usually = expense.cost (per-line item), NOT budget.spent
- per_unit_price: "per unit" / "per X" = Price/Amount, NOT Price alone
- case_insensitive_title: exact title may have case mismatch (e.g., "Computer Game" vs "Computer game") — use case-insensitive LIKE
- atom_id_numeric_sort: atom_id like TR000_4 sorts by NUMERIC suffix (CAST split _), NOT alphabetically
- id_anchor_prose: same entity in multiple prose sections, join via shared ID
- directional_csv: connected.csv / hero_power.csv has both directions — DISTINCT bond_id / power_id
- subset_data: _1k.db or partial prose may be subset; answer from rows available

**Sub-questions**:
1. <atomic fact-finding question>
2. ...

**Verification targets** (= what Stage 5 should independently check):
- <target 1>
- ...

CRITICAL RULES:
- Use EXACT table/column names from schema. Do NOT invent names.
- For "min/max", default result shape to multi_row to handle ties safely.
- result_columns should match what gold csv has — usually just the asked-for entity, NOT extra columns.
"""


def stage1_interpret(model, task, schema_preview, doc_toc, knowledge):
    user = f"""# Question
{task.question}

# DB schema (EXACT names — copy verbatim)
{schema_preview}

# Prose doc TOC
{doc_toc}

# Knowledge guide excerpt
{knowledge}

Generate the Question Plan in markdown now."""
    return chat(model, STAGE1_SYS, user)


# =========================================================================
# Stage 2 — Explorer (= for each sub-Q, dispatch a sub-agent)
# =========================================================================

def extract_sub_questions(plan_md: str) -> list[str]:
    """Pull sub-questions from the markdown plan."""
    # Look for "**Sub-questions**:" section followed by numbered/bulleted items
    m = re.search(r"\*\*Sub-questions\*\*:?\s*\n(.*?)(?=\n\*\*[A-Z]|\Z)", plan_md, re.DOTALL | re.IGNORECASE)
    if not m:
        return []
    body = m.group(1)
    items = re.findall(r"^\s*(?:\d+\.|-|\*)\s+(.+)$", body, re.MULTILINE)
    return [item.strip() for item in items if item.strip()]


def stage2_explore(model, task, plan_md, log_lines):
    sub_qs = extract_sub_questions(plan_md)
    log_lines.append(f"  stage 2: dispatching {len(sub_qs)} sub-agents")
    explorations = []
    for i, sq in enumerate(sub_qs, start=1):
        log_lines.append(f"    Q{i}: {sq[:120]}")
        try:
            sr = run_sql_subagent(
                task=task,
                sub_question=sq,
                model=model,
                max_steps=8,
            )
            if sr.ok:
                explorations.append({
                    "q": sq,
                    "columns": sr.columns,
                    "rows": sr.rows[:20],
                    "n_rows": len(sr.rows),
                    "n_steps": sr.n_steps,
                    "ok": True,
                })
                log_lines.append(f"      → ok n_rows={len(sr.rows)} cols={sr.columns} steps={sr.n_steps}")
            else:
                explorations.append({
                    "q": sq,
                    "error": sr.failure_reason,
                    "n_steps": sr.n_steps,
                    "ok": False,
                })
                log_lines.append(f"      → FAIL: {sr.failure_reason} (steps={sr.n_steps})")
        except Exception as exc:
            explorations.append({"q": sq, "error": str(exc)[:200], "ok": False})
            log_lines.append(f"      → exception: {exc}")
    return explorations


def format_exploration_for_brief(explorations: list[dict]) -> str:
    """Render the exploration log as markdown for Stage 3 input."""
    lines = ["# Exploration Log"]
    for i, e in enumerate(explorations, start=1):
        lines.append(f"\n## Q{i}: {e['q']}")
        if e["ok"]:
            lines.append(f"- columns: {e['columns']}")
            lines.append(f"- n_rows: {e['n_rows']}")
            if e["rows"]:
                lines.append(f"- first rows:")
                for r in e["rows"][:10]:
                    lines.append(f"  - {r}")
        else:
            lines.append(f"- ERROR: {e.get('error', 'unknown')}")
    return "\n".join(lines)


# =========================================================================
# Stage 3 — Aggregator (1-shot, markdown out)
# =========================================================================

STAGE3_SYS = """You are the Aggregator. Compress the exploration log into a Structured Brief in MARKDOWN.
The Answer Generator will read this Brief directly to write the final SQL.

OUTPUT TEMPLATE:

# Structured Brief

**Rephrased question**: <from Stage 1>

**Expected result shape**: <scalar | single_row | multi_row | percentage>
- If MIN/MAX/superlative and ties found in exploration, MARK AS multi_row.

**Expected columns**: <comma-separated, EXACTLY as gold csv would have — usually just the named entity, NOT the metric value>

**Relevant tables/columns**: <table.col1, table.col2, ...>

**Key values discovered**:
- <key>: <value>

**Join path** (= explicit foreign-key chain): <table1.col = table2.col → table2.col = table3.col>

**Filter predicates**: <col op value>

**Ordering / LIMIT / DISTINCT**: <plain text instruction>

**Unresolved assumptions**: <list, or "none">

**Verification targets**: <from Stage 1>

CRITICAL:
- Be aggressive about MULTI_ROW for min/max with ties. Do NOT use LIMIT 1.
- Expected columns must match gold csv shape — usually one column for "which X", not (X, metric).
"""


def stage3_aggregate(model, plan_md, explore_md):
    user = f"{plan_md}\n\n---\n\n{explore_md}\n\nGenerate the Structured Brief in markdown now."
    return chat(model, STAGE3_SYS, user)


# =========================================================================
# Stage 4 — Answer Generator (1-shot, SQL out)
# =========================================================================

STAGE4_SYS = """You are the Answer Generator. Read the Structured Brief and emit ONE SQL query.

CONSTRAINTS:
- Match Expected columns EXACTLY (= same number + names).
- Match Expected result shape: if multi_row + ties, use filter-back (= WHERE col = (SELECT MIN/MAX ...)) NOT LIMIT 1.
- Apply ROUND() only at outermost SELECT if requested.
- One SQL only. Output format: ```sql\\n<SQL>\\n```. Nothing else.
"""


def stage4_answer(model, brief_md, context_dir):
    user = f"{brief_md}\n\nWrite the SQL now."
    out = chat(model, STAGE4_SYS, user)
    m = re.search(r"```(?:sql)?\s*\n?(.*?)\n?```", out, re.DOTALL)
    sql = m.group(1).strip() if m else out.strip()
    try:
        res = execute_sql(context_dir, sql, limit=500)
        return {"sql": sql, "columns": res["columns"], "rows": res["rows"], "raw": out}
    except Exception as exc:
        return {"sql": sql, "error": str(exc)[:300], "raw": out}


# =========================================================================
# Stage 5 — Independent Verifier (= dispatch a sub-agent for verify Q)
# =========================================================================

VERIFY_SYS = """You are the Independent Verifier. Read the candidate answer and propose ONE verification SQL whose result, if matching the candidate, would confirm it.

OUTPUT in MARKDOWN:
**Verification question**: <one short Q>
**Independent SQL**: ```sql\\n<SELECT ...>\\n```
**Expected if candidate is correct**: <what the SQL should return>
"""


def stage5_verify(model, question, brief_md, candidate, task, log_lines):
    user = f"""# Original question
{question}

# Brief
{brief_md}

# Candidate
SQL: {candidate.get('sql', '')[:300]}
Columns: {candidate.get('columns', [])}
First rows: {candidate.get('rows', [])[:5]}
N rows: {len(candidate.get('rows', [])) if 'rows' in candidate else 'ERROR'}

Propose verification."""
    out = chat(model, VERIFY_SYS, user)
    # Extract SQL
    m = re.search(r"```(?:sql)?\s*\n?(.*?)\n?```", out, re.DOTALL)
    if not m:
        return {"verdict": "skip", "raw": out}
    verify_sql = m.group(1).strip()
    try:
        res = execute_sql(task.context_dir, verify_sql, limit=20)
        log_lines.append(f"    verify_sql executed: cols={res['columns']} n_rows={len(res['rows'])}")
        # Heuristic verdict: if candidate has rows and verify also has rows → tentatively pass
        cand_rows = candidate.get("rows", [])
        verify_rows = res["rows"]
        if not cand_rows and not verify_rows:
            verdict = "pass (both empty)"
        elif not cand_rows or not verify_rows:
            verdict = "mismatch (one empty)"
        else:
            # Compare row counts as a coarse check
            verdict = f"row-count {len(cand_rows)} vs verify {len(verify_rows)}"
        return {"verdict": verdict, "verify_sql": verify_sql, "verify_rows": verify_rows[:5], "raw": out}
    except Exception as exc:
        return {"verdict": f"verify failed: {str(exc)[:100]}", "verify_sql": verify_sql, "raw": out}


# =========================================================================
# Per-task runner
# =========================================================================

def run_one(tid: str, model_fast, model_sub=None) -> dict:
    """model_fast = lower-budget 1-shot LLM for Stage 1/3/4.
    model_sub = higher-budget for sub-agent ReAct loops (Stage 2/5)."""
    if model_sub is None:
        model_sub = model_fast
    print(f"\n{'='*70}", flush=True)
    print(f"## {tid}", flush=True)
    print(f"{'='*70}", flush=True)
    ds = DABenchPublicDataset(root_dir=REPO / "data" / "public" / "input")
    task = ds.get_task(tid)
    print(f"Q: {task.question}", flush=True)

    artifacts = POC_DIR / tid
    artifacts.mkdir(parents=True, exist_ok=True)

    log_lines = []
    t0 = time.time()
    schema_preview = build_schema_preview(task.context_dir)
    doc_toc = build_doc_toc(task.context_dir)
    knowledge = load_knowledge_md(task.context_dir)

    # Stage 1
    print(f"[stage 1] interpret...", flush=True)
    plan_md = stage1_interpret(model_fast, task, schema_preview, doc_toc, knowledge)
    (artifacts / "stage1_plan.md").write_text(plan_md)
    print(f"  plan length: {len(plan_md)} chars", flush=True)
    # Show 1st line of rephrase
    m = re.search(r"\*\*Rephrased\*\*[:\s]+(.+?)$", plan_md, re.MULTILINE)
    if m:
        print(f"  rephrased: {m.group(1)[:150]}", flush=True)
    m = re.search(r"\*\*Result shape\*\*[:\s]+(.+?)$", plan_md, re.MULTILINE)
    if m:
        print(f"  shape: {m.group(1)[:80]}", flush=True)
    m = re.search(r"\*\*Pitfalls\*\*[^\n]*\n((?:- [^\n]+\n?)+)", plan_md)
    if m:
        print(f"  pitfalls listed: {len(re.findall(r'^- ', m.group(1), re.MULTILINE))}", flush=True)

    # Stage 2
    print(f"[stage 2] explore (sub-agents)...", flush=True)
    explorations = stage2_explore(model_sub, task, plan_md, log_lines)
    explore_md = format_exploration_for_brief(explorations)
    (artifacts / "stage2_exploration.md").write_text(explore_md)
    (artifacts / "stage2_explorations.json").write_text(json.dumps(explorations, indent=2, default=str))
    for line in log_lines:
        print(line, flush=True)
    log_lines.clear()

    # Stage 3
    print(f"[stage 3] aggregate...", flush=True)
    brief_md = stage3_aggregate(model_fast, plan_md, explore_md)
    (artifacts / "stage3_brief.md").write_text(brief_md)
    m = re.search(r"\*\*Expected result shape\*\*[:\s]+(.+?)$", brief_md, re.MULTILINE)
    if m: print(f"  shape: {m.group(1)[:80]}", flush=True)
    m = re.search(r"\*\*Expected columns\*\*[:\s]+(.+?)$", brief_md, re.MULTILINE)
    if m: print(f"  cols: {m.group(1)[:80]}", flush=True)

    # Stage 4
    print(f"[stage 4] answer...", flush=True)
    candidate = stage4_answer(model_fast, brief_md, task.context_dir)
    (artifacts / "stage4_candidate.json").write_text(json.dumps(candidate, indent=2, default=str))
    if "error" in candidate:
        print(f"  ERROR: {candidate['error']}", flush=True)
    else:
        print(f"  SQL: {candidate['sql'][:200]}", flush=True)
        print(f"  cols={candidate['columns']} n_rows={len(candidate['rows'])}", flush=True)

    # Stage 5
    print(f"[stage 5] verify...", flush=True)
    verify = stage5_verify(model_sub, task.question, brief_md, candidate, task, log_lines)
    (artifacts / "stage5_verify.json").write_text(json.dumps(verify, indent=2, default=str))
    for line in log_lines:
        print(line, flush=True)
    print(f"  verdict: {verify.get('verdict')}", flush=True)

    # Eval
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
            extras = e.extra_cols
        except Exception:
            score, recall, extras = 0.0, 0.0, -1
    else:
        score, recall, extras = 0.0, 0.0, -1

    dt = time.time() - t0
    print(f"  score={score:.4f}  recall={recall:.2f}  extras={extras}  latency={dt:.0f}s", flush=True)
    return {
        "task": tid,
        "score": score,
        "recall": recall,
        "extras": extras,
        "latency_s": int(dt),
        "stage5_verdict": verify.get("verdict"),
    }


# =========================================================================
# Main
# =========================================================================

def main():
    model = make_model(temperature=0.6)

    results = []
    for tid in TASKS:
        try:
            results.append(run_one(tid, model, model))
        except Exception as exc:
            traceback.print_exc()
            results.append({"task": tid, "score": 0.0, "exception": repr(exc)})

    print("\n" + "=" * 70, flush=True)
    print("LOGICRAG POC v3 SUMMARY (= markdown intermediate + real sub-agents)", flush=True)
    print("=" * 70, flush=True)
    print(f"{'task':>10} {'score':>8} {'recall':>8} {'extras':>8} {'lat':>6} {'verdict':>30}", flush=True)
    total = 0.0
    for r in results:
        tid = r.get("task")
        sc = r.get("score", 0.0)
        rc = r.get("recall", 0.0)
        ex = r.get("extras", -1)
        lat = r.get("latency_s", 0)
        vd = str(r.get("stage5_verdict", ""))[:28]
        print(f"{tid:>10} {sc:>8.4f} {rc:>8.2f} {ex:>8} {lat:>6}s {vd:>30}", flush=True)
        total += sc
    mean = total / len(results) if results else 0.0
    print(f"{'mean':>10} {mean:>8.4f}", flush=True)

    (POC_DIR / "summary.json").write_text(json.dumps({
        "results": results,
        "mean_score": mean,
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
