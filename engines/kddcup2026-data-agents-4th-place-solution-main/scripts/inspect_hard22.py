"""Dump everything needed to manually write SQL for the 22 hard tasks."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from dotenv import load_dotenv
load_dotenv(REPO / ".env")

from kobushi_core.benchmark.dataset import DABenchPublicDataset
from experiments.exp_122_column_auditor.tools.duckdb_unified import execute_sql


HARD = ["task_25", "task_27", "task_145", "task_163", "task_169", "task_180", "task_199",
        "task_243", "task_283", "task_303", "task_330", "task_344", "task_349", "task_350",
        "task_352", "task_355", "task_379", "task_396", "task_408", "task_415", "task_418",
        "task_420"]


def main():
    csv_in = REPO / "artifacts" / "plan_verify_decisions_full50" / "gold_sql_full_enriched.csv"
    rows = {r["task_id"]: r for r in csv.DictReader(open(csv_in))}
    ds = DABenchPublicDataset(root_dir=REPO / "data" / "public" / "input")
    for tid in HARD:
        r = rows.get(tid, {})
        print(f"\n{'='*70}\n# {tid}  ({r.get('bird_db','?')})")
        print(f"Q: {r.get('question','')}")
        print(f"Files: {r.get('context_files','')}")
        print(f"Schema: {r.get('schema','')}")
        print(f"BIRD SQL: {r.get('gold_sql','')}")
        # gold.csv preview
        gp = REPO / "data" / "public" / "output" / tid / "gold.csv"
        if gp.exists():
            preview = gp.read_text().splitlines()[:8]
            print("gold.csv (top 8):")
            for line in preview:
                print(f"  {line}")
            print(f"  ... (total {len(gp.read_text().splitlines())} lines)")
        # Try running BIRD SQL for diagnostic
        try:
            task = ds.get_task(tid)
            sql = r.get("gold_sql","")
            if sql and "NEW" not in sql:
                try:
                    res = execute_sql(task.context_dir, sql)
                    print(f"BIRD exec: cols={res.get('columns')}, n={len(res.get('rows',[]))}, first={res.get('rows',[])[:3]}")
                except Exception as exc:
                    print(f"BIRD exec ERROR: {str(exc)[:200]}")
        except Exception as exc:
            print(f"task error: {exc}")


if __name__ == "__main__":
    main()
