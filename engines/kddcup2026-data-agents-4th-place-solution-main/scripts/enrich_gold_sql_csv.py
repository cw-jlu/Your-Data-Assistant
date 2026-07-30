"""Enrich gold_sql_full.csv with two columns:
   - context_files: list of files in the task context dir (= csv/json/db/txt)
   - schema: {table: [col, col, ...]} for every DuckDB-visible view

Output: gold_sql_full_enriched.csv (= same rows + 2 new columns)
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from dotenv import load_dotenv
load_dotenv(REPO / ".env")

from kobushi_core.benchmark.dataset import DABenchPublicDataset
from experiments.exp_122_column_auditor.tools.duckdb_unified import execute_sql


IN_CSV = REPO / "artifacts" / "plan_verify_decisions_full50" / "gold_sql_full.csv"
OUT_CSV = REPO / "artifacts" / "plan_verify_decisions_full50" / "gold_sql_full_enriched.csv"


def list_context_files(context_dir: Path) -> str:
    items = []
    for p in sorted(context_dir.rglob("*")):
        if p.is_file():
            rel = p.relative_to(context_dir)
            items.append(str(rel))
    return ";".join(items)


def get_schema(task) -> str:
    try:
        res = execute_sql(task.context_dir, "SHOW TABLES")
        tables = [r[0] for r in res["rows"]]
    except Exception as exc:
        return f"<schema_error: {str(exc)[:80]}>"
    parts = []
    for tbl in tables:
        try:
            cols = execute_sql(task.context_dir, f"DESCRIBE {tbl}")
            col_names = [r[0] for r in cols["rows"]]
            parts.append(f"{tbl}({','.join(col_names)})")
        except Exception:
            parts.append(f"{tbl}(?)")
    return " | ".join(parts)


def main():
    ds = DABenchPublicDataset(root_dir=REPO / "data" / "public" / "input")
    rows_in = list(csv.DictReader(open(IN_CSV)))
    out_rows = []
    for i, r in enumerate(rows_in, 1):
        tid = r["task_id"]
        try:
            task = ds.get_task(tid)
        except Exception as exc:
            r["context_files"] = f"<no_task: {exc}>"
            r["schema"] = ""
            out_rows.append(r)
            continue
        r["context_files"] = list_context_files(task.context_dir)
        r["schema"] = get_schema(task)
        out_rows.append(r)
        print(f"  [{i}/{len(rows_in)}] {tid}: {len(r['context_files'].split(';'))} files, schema_len={len(r['schema'])}", flush=True)

    fields = ["task_id", "bird_question_id", "question", "nouns", "context_files", "schema", "gold_sql"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nSaved: {OUT_CSV}")


if __name__ == "__main__":
    main()
