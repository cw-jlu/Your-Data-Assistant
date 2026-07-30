#!/usr/bin/env python
"""PoC: prose -> structured table -> SQL answer.

A dedicated "prose path": instead of extracting values inline while answering
(which fails on adversarial narrative -> the agent falls back to a decoy table),
one LLM call READS the prose doc(s) and emits SQL that MATERIALIZES the records
into a temp table (CREATE TABLE + INSERT VALUES, faithful: exact values, NULLs
preserved, no rounding), then a final SELECT that answers the question from that
table. We run the SQL in an in-memory DuckDB and score vs gold.

Usage:
  uv run python scripts/phase2_prose_to_table_poc.py --task task_18
  uv run python scripts/phase2_prose_to_table_poc.py --task task_18,task_58,task_38
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
try:
    from dotenv import load_dotenv
    load_dotenv(REPO / ".env")
except Exception:
    pass

import duckdb  # noqa: E402
from kobushi_core.model import ModelMessage, OpenAIModelAdapter  # noqa: E402
from kobushi_core.eval.csv_compare import _evaluate_task, EvaluationOptions  # noqa: E402

IN = REPO / "data/phase2_demo/demo_samples_phase2/input"
GOLD = REPO / "data/phase2_demo/demo_samples_phase2/output"

SYSTEM = """You are a PROSE-TO-TABLE EXTRACTOR. The input is one or more adversarial PROSE documents that narrate tabular records (one record per entity/period, with the values stated inside sentences). Your ONLY goal is to faithfully reconstruct those records as a clean SQL table.

You are also given a QUESTION and COLUMN SEMANTICS (Chinese term -> table.column + unit). Use them ONLY to decide WHICH columns are relevant to extract — the field(s) the question refers to, plus any identifying / period column (entity id/code, date). Do nothing else with the question; do NOT compute, filter, aggregate, or otherwise respond to it.

Produce ONE SQL script:
1. CREATE a temp table whose columns are those relevant columns (use the column names + units from COLUMN SEMANTICS).
2. INSERT EVERY record the prose narrates, in source order — `INSERT ... VALUES` with all rows. Do NOT drop, filter, aggregate, sort, or deduplicate.
3. End with `SELECT * FROM <table>;` to return the full extracted table.

FIDELITY RULES (this is the whole job — get every value exactly right; a single wrong value ruins the column):
- Copy values EXACTLY: no rounding, no truncation. Apply a unit conversion ONLY when COLUMN SEMANTICS state one (e.g. 亿元 -> 万元; "1.55%" -> 0.0155), consistently.
- If the prose gives a preliminary/decoy value AND a final/audited value ("初步...但经最终审计确认为 X" / "误记为...修正为 X"), use the FINAL/corrected value.
- If a record's value is missing / 缺失 / 未获取 / NaN, INSERT a NULL for it and KEEP the row (never drop it, never filter IS NOT NULL).

Return ONLY the SQL (one ```sql code block). No prose, no explanation."""


def _doc_items(task_id: str) -> list[tuple[str, str]]:
    """Return [(doc_name, text), ...] — each prose doc SEPARATELY (not concatenated)."""
    docdir = IN / task_id / "context" / "doc"
    items = []
    for p in sorted(docdir.glob("*")):
        if p.suffix.lower() == ".md":
            items.append((p.name, p.read_text(encoding="utf-8", errors="ignore")))
        elif p.suffix.lower() == ".pdf":
            txt = _pdf_text(task_id, p)
            if txt:
                items.append((p.name, txt))
    return items


def _pdf_text(task_id: str, pdf: Path) -> str:
    # try the experiment's pdf text cache; fall back to pdfplumber
    try:
        from experiments.exp_167_domain_pruner.pdf_text_cache import ensure_task_pdf_cache
        from kobushi_core.benchmark.dataset import DABenchPublicDataset
        ds = {t.task_id: t for t in DABenchPublicDataset(root_dir=IN).iter_tasks()}
        for r in ensure_task_pdf_cache(ds[task_id]):
            if Path(r.source_rel).name == pdf.name:
                return Path(r.text_path).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        pass
    try:
        import pdfplumber
        with pdfplumber.open(pdf) as f:
            return "\n".join((pg.extract_text() or "") for pg in f.pages)
    except Exception:
        return ""


def _knowledge(task_id: str) -> str:
    k = IN / task_id / "context" / "knowledge.md"
    if k.is_file():
        return k.read_text(encoding="utf-8", errors="ignore")  # full, no cap
    return ""


def _model() -> OpenAIModelAdapter:
    return OpenAIModelAdapter(
        model=os.environ.get("AGENT_MODEL", "qwen3.5-35b-a3b"),
        api_base=os.environ.get("AGENT_API_BASE", ""),
        api_key=os.environ.get("AGENT_API_KEY", ""),
        temperature=0.0,
        extra_headers={
            "CF-Access-Client-Id": os.environ.get("CF_ACCESS_CLIENT_ID", ""),
            "CF-Access-Client-Secret": os.environ.get("CF_ACCESS_CLIENT_SECRET", ""),
        },
        enable_thinking=False,
        max_tokens=32000,
    )


def _extract_sql(raw: str) -> str:
    m = re.search(r"```sql\s*(.+?)```", raw, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*(.+?)```", raw, re.DOTALL)
    if m:
        return m.group(1).strip()
    return raw.strip()


def run_task(task_id: str, out_dir: Path, doc_filter: str | None = None) -> dict:
    """Extract each doc separately (one pass = one doc) into its own table; score each.

    `doc_filter`: if given, only docs whose name contains it (the upstream doc-
    selection stage / router is a separate sub-agent, built later).
    """
    q = json.loads((IN / task_id / "task.json").read_text(encoding="utf-8"))["question"]
    know = _knowledge(task_id)
    gold_csv = next((GOLD / task_id).glob("*.csv"))
    gold_rows = len(gold_csv.read_text(encoding="utf-8").splitlines()) - 1
    model = _model()

    items = [(n, t) for (n, t) in _doc_items(task_id) if (not doc_filter or doc_filter in n)]
    results = []
    for name, text in items:
        user = (
            f"QUESTION (use ONLY to pick which columns are relevant to extract):\n{q}\n\n"
            f"COLUMN SEMANTICS (from knowledge.md — use it to pick the right table/column + unit):\n{know}\n\n"
            f"PROSE DOCUMENT ({name}):\n{text[:200000]}\n\n"
            "Return the ONE SQL script (CREATE TEMP TABLE + INSERT VALUES + SELECT *)."
        )
        ddir = out_dir / task_id / name
        ddir.mkdir(parents=True, exist_ok=True)
        err = ""; score = 0.0; pred_rows = None
        try:
            raw = model.complete(
                [ModelMessage(role="system", content=SYSTEM), ModelMessage(role="user", content=user)],
                enable_thinking=False, max_tokens=32000,
            )
            sql = _extract_sql(raw)
            (ddir / "emitted.sql").write_text(sql, encoding="utf-8")
            con = duckdb.connect(":memory:")
            df = con.execute(sql).df()
            df.to_csv(ddir / "prediction.csv", index=False)
            pred_rows = len(df)
            ev = _evaluate_task(task_id=task_id, prediction_path=ddir / "prediction.csv",
                                gold_path=gold_csv, options=EvaluationOptions())
            score = float(ev.official_score_lambda_0_5)
        except Exception as exc:
            err = repr(exc)[:160]
        print(f"  {task_id} / {name:<40} score={score:.3f} rows={pred_rows}/{gold_rows} {('ERR '+err) if err else ''}", flush=True)
        results.append({"doc": name, "score": score, "pred_rows": pred_rows, "error": err})

    best = max(results, key=lambda r: r["score"]) if results else {"doc": None, "score": 0.0}
    print(f"{task_id}: BEST = {best['doc']}  score={best['score']:.3f}  (gold {gold_rows} rows)")
    return {"task_id": task_id, "best_doc": best.get("doc"), "best_score": best["score"], "docs": results}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="task id(s), comma-separated")
    ap.add_argument("--doc", default=None, help="only docs whose name contains this (one doc per pass)")
    ap.add_argument("--out-dir", type=Path, default=REPO / "artifacts/prose_to_table_poc")
    args = ap.parse_args()
    tasks = [t.strip() for t in args.task.split(",") if t.strip()]
    rows = [run_task(t, args.out_dir, doc_filter=args.doc) for t in tasks]
    if len(rows) > 1:
        print(f"\nmean BEST-doc score = {sum(r['best_score'] for r in rows)/len(rows):.4f}  n={len(rows)}")


if __name__ == "__main__":
    main()
