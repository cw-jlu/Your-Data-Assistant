"""Offline comparison of vote strategies on saved per-attempt data.

Reads attempt_<i>.json files (= each attempt's audited answer), and applies:
  - majority (= most common signature)
  - union (= column-wise merge)
  - adaptive_v3 (= plurality > n/2 → vote, else union)
  - adaptive_v4 (= any divergence → union)
Scores each strategy against gold.csv.

Usage:
    uv run python scripts/compare_vote_strategies.py [--dir artifacts/column_auditor]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import os
from kobushi_core.benchmark.schema import AnswerTable
from kobushi_core.eval.csv_compare import _evaluate_task, EvaluationOptions
from kobushi_core.benchmark.dataset import DABenchPublicDataset
from kobushi_core.model import OpenAIModelAdapter, ModelMessage
from dotenv import load_dotenv
load_dotenv()


def _norm(v):
    if v is None: return (0, "")
    s = str(v).strip()
    try: return (1, round(float(s), 3))
    except: return (2, s.lower())


def _signature(a):
    if not a or not a.columns: return ("__empty__",)
    n = len(a.columns)
    rows = [tuple(_norm(v) for v in r) for r in (a.rows or []) if len(r) == n]
    rows.sort()
    return (n, len(rows), tuple(rows))


def _majority(answers):
    s2i = {}
    for i, a in enumerate(answers):
        s2i.setdefault(_signature(a), []).append(i)
    best = max(s2i, key=lambda k: len(s2i[k]))
    return answers[s2i[best][0]]


def _union(answers):
    if not answers: return None
    sig_to_rep = {}
    for a in answers:
        sig = _signature(a)
        if sig not in sig_to_rep: sig_to_rep[sig] = a
    if len(sig_to_rep) == 1: return next(iter(sig_to_rep.values()))
    unique_cols = []
    seen = set()
    for a in answers:
        for ci, c in enumerate(a.columns):
            col_vals = [r[ci] if ci < len(r) else "" for r in a.rows]
            key = tuple(sorted(_norm(v) for v in col_vals))
            if key in seen: continue
            seen.add(key)
            unique_cols.append((c, col_vals))
    if not unique_cols: return answers[0]
    max_rows = max(len(c[1]) for c in unique_cols)
    cols = [c[0] for c in unique_cols]
    rows = [[(uc[1][ri] if ri < len(uc[1]) else "") for uc in unique_cols] for ri in range(max_rows)]
    return AnswerTable(columns=cols, rows=rows)


def _token_set(a):
    if not a or not a.rows: return frozenset()
    tokens = set()
    for row in a.rows:
        for v in row:
            if v is None: continue
            s = str(v).strip().lower()
            try: tokens.add(round(float(s), 3))
            except: tokens.update(s.split())
    return frozenset(tokens)


def _adaptive_v3(answers):
    non_empty = [a for a in answers if a and a.rows]
    if not non_empty: return answers[0] if answers else None
    groups = {}
    for a in non_empty:
        ts = _token_set(a)
        groups.setdefault(ts, []).append(a)
    sorted_g = sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True)
    largest = sorted_g[0]
    n = len(non_empty)
    if len(sorted_g) == 1:
        return _majority(non_empty)
    if len(largest[1]) > n // 2:
        return _majority(largest[1])
    return _union(non_empty)


def _adaptive_v4(answers):
    non_empty = [a for a in answers if a and a.rows]
    if not non_empty: return answers[0] if answers else None
    groups = {}
    for a in non_empty:
        ts = _token_set(a)
        groups.setdefault(ts, []).append(a)
    if len(groups) == 1:
        return _majority(non_empty)
    return _union(non_empty)


def _make_judge_model():
    return OpenAIModelAdapter(
        model="qwen3.5-35b-a3b",
        api_base=os.environ["AGENT_API_BASE"],
        api_key=os.environ["AGENT_API_KEY"],
        temperature=0.0,
        extra_headers={
            "CF-Access-Client-Id": os.environ.get("CF_ACCESS_CLIENT_ID", ""),
            "CF-Access-Client-Secret": os.environ.get("CF_ACCESS_CLIENT_SECRET", ""),
        },
        presence_penalty=0.0,
    )


LLM_ROUTER_SYSTEM = """\
You are a vote-strategy router for a SQL-based data agent. You see the
question, the database schema, and 3 candidate (SQL, result) pairs. Decide:
  - PICK <i>: attempt i's SQL+result is most likely correct
  - UNION:    attempts diverge on interpretation; combine column-wise so the
              minority candidate (= possibly the gold interpretation) is preserved

Decision principles:
- Read each SQL critically: which one matches the question's verbs/aggregations?
  E.g. "average X" → AVG(X), not SUM(X)/COUNT(*); "list of X" → SELECT X (1 col).
- Compare the SQLs' aggregation logic, JOIN structure, and filter expressions.
  If one SQL clearly diverges from the others on a meaningful axis, that's
  evidence of interpretation divergence.
- For superlatives, prefer the SQL that uses filter-back (= WHERE val = (SELECT
  MIN/MAX(val) FROM ...)) over LIMIT 1, since it preserves tied rows.
- For "list X" questions, prefer the SQL with the fewest output columns (= just
  the entity's identifier).
- For aggregate questions (= COUNT/SUM/AVG/percentage), prefer the SQL whose
  aggregation precisely matches the question's verb.
- If 2 of 3 attempts converge on one interpretation (= same SQL pattern, same
  result) and 1 differs, the convergent is usually correct unless the divergent
  SQL is more semantically aligned with the question.
- If all 3 attempts use semantically equivalent SQL but produce different result
  rows (= same logic, possibly variance in row order or rounding), PICK the
  cleanest one.
- If attempts genuinely diverge on the question's interpretation (= different
  SQL formulas reflecting different readings), UNION to preserve all.

Output format (= ONE line, no other text):
  PICK <0|1|2>
  or
  UNION
"""


def _llm_router(answers, question, model, sqls=None, schema=None):
    """LLM picks one attempt or union of all. Uses schema + per-attempt SQL when available."""
    non_empty_idx = [i for i, a in enumerate(answers) if a and a.rows]
    non_empty = [answers[i] for i in non_empty_idx]
    if not non_empty: return answers[0] if answers else None
    if len(non_empty) == 1: return non_empty[0]
    blocks = []
    for local_i, orig_i in enumerate(non_empty_idx):
        a = answers[orig_i]
        rows_preview = a.rows[:5]
        sql_text = (sqls[orig_i] if sqls and orig_i < len(sqls) and sqls[orig_i] else "(SQL not captured)")
        blocks.append(
            f"## Attempt {local_i} (= original index {orig_i})\n"
            f"### SQL\n```sql\n{sql_text[:1500]}\n```\n"
            f"### Result\n  columns: {a.columns}\n  n_rows: {len(a.rows)}\n  first 5 rows: {rows_preview}"
        )
    schema_block = f"## DuckDB schema (= views available)\n{schema}\n\n" if schema else ""
    user = (
        f"## Question\n{question}\n\n"
        + schema_block
        + "\n\n".join(blocks)
        + "\n\nDirective (= one line):"
    )
    try:
        raw = model.complete([
            ModelMessage(role="system", content=LLM_ROUTER_SYSTEM),
            ModelMessage(role="user", content=user),
        ])
    except Exception as exc:
        return _adaptive_v3(answers)
    raw = raw.strip().upper()
    if raw.startswith("UNION"):
        return _union(non_empty)
    import re
    m = re.search(r"PICK\s+(\d+)", raw)
    if m:
        idx = int(m.group(1))
        if 0 <= idx < len(non_empty):
            return non_empty[idx]
    # Fallback
    return _adaptive_v3(answers)


def score(answer, gold_path):
    if answer is None or not answer.columns:
        return 0.0
    pred_path = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", newline="")
    w = csv.writer(pred_path); w.writerow(answer.columns)
    for r in answer.rows or []: w.writerow([str(v) for v in r])
    pred_path.close()
    try:
        e = _evaluate_task(task_id="x", prediction_path=Path(pred_path.name),
                           gold_path=Path(gold_path), options=EvaluationOptions())
        return e.official_score_lambda_0_5
    except Exception:
        return 0.0
    finally:
        Path(pred_path.name).unlink(missing_ok=True)


def load_attempts(task_dir, with_sql=False):
    """Load attempt_*.json from a task dir.

    Returns list of (AnswerTable | None, sql | None) when with_sql=True,
    else list of AnswerTable | None for backward compat.
    """
    attempts = []
    sqls = []
    for i in range(3):
        p = task_dir / f"attempt_{i}.json"
        if not p.exists():
            attempts.append(None); sqls.append(None); continue
        d = json.loads(p.read_text())
        if d.get("failed") or not d.get("answer_columns"):
            attempts.append(None); sqls.append(d.get("agent_sql")); continue
        cols = d["answer_columns"]
        rows = d.get("answer_rows") or []
        attempts.append(AnswerTable(columns=cols, rows=[list(r) for r in rows]))
        sqls.append(d.get("agent_sql"))
    if with_sql:
        return attempts, sqls
    return attempts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="artifacts/column_auditor")
    args = ap.parse_args()

    base = REPO / args.dir
    if not base.exists():
        print(f"no such dir: {base}"); return

    use_llm = bool(os.environ.get("AGENT_API_BASE"))
    judge = _make_judge_model() if use_llm else None
    ds = DABenchPublicDataset(root_dir=REPO / "data" / "public" / "input")

    # Lazy import for schema (= avoids dep when not used)
    def schema_for(task_id):
        try:
            from experiments.exp_122_column_auditor.tools.duckdb_unified import describe_catalog
            ctx = REPO / "data" / "public" / "input" / task_id / "context"
            return describe_catalog(ctx)
        except Exception:
            return None

    rows = []
    for task_dir in sorted(base.iterdir()):
        if not task_dir.is_dir(): continue
        if not task_dir.name.startswith("task_"): continue
        gold_p = REPO / "data" / "public" / "output" / task_dir.name / "gold.csv"
        if not gold_p.exists(): continue
        attempts_raw, attempt_sqls = load_attempts(task_dir, with_sql=True)
        non_none = [a for a in attempts_raw if a is not None]
        if not non_none: continue

        question = ds.get_task(task_dir.name).question

        s_maj = score(_majority(non_none), gold_p)
        s_uni = score(_union(non_none), gold_p)
        s_v3 = score(_adaptive_v3(non_none), gold_p)
        s_v4 = score(_adaptive_v4(non_none), gold_p)
        s_llm = None
        if judge:
            schema = schema_for(task_dir.name)
            s_llm = score(
                _llm_router(attempts_raw, question, judge, sqls=attempt_sqls, schema=schema),
                gold_p,
            )
        rows.append({"task": task_dir.name, "majority": s_maj, "union": s_uni, "v3": s_v3, "v4": s_v4, "llm": s_llm})

    if not rows:
        print("no task data found"); return

    cols_show = ["maj", "uni", "v3", "v4"] + (["llm"] if use_llm else [])
    keys_map = {"maj": "majority", "uni": "union", "v3": "v3", "v4": "v4", "llm": "llm"}

    header = f"{'task':<10}" + " ".join(f"{c:>6}" for c in cols_show) + "  best"
    print(header)
    print("-" * len(header))
    for r in rows:
        scores = {c: r[keys_map[c]] for c in cols_show if r[keys_map[c]] is not None}
        best_v = max(scores.values())
        labels = [k for k, v in scores.items() if v == best_v]
        flag = "/".join(labels)
        line = f"  {r['task']:<10}" + " ".join(f"{r[keys_map[c]]:>6.2f}" if r[keys_map[c]] is not None else f"{'-':>6}" for c in cols_show) + f"  {flag}"
        print(line)
    print()
    n = len(rows)
    for c in cols_show:
        vals = [r[keys_map[c]] for r in rows if r[keys_map[c]] is not None]
        if vals:
            print(f"  {c}: sum={sum(vals):.2f}  mean={sum(vals)/len(vals):.3f}  (n={len(vals)})")


if __name__ == "__main__":
    main()
