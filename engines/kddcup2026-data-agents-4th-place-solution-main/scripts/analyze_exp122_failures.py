"""Analyze the 10 v5 (exp_122) failure cases:
  - load gold.csv
  - load each attempt's prediction.csv
  - load voted prediction (= final submission)
  - load PLAN thought from trace.log
  - compare with my manual gold SQL

For each failing task, output:
  - question
  - gold cols/rows
  - voted prediction cols/rows
  - per-attempt agent SQL (= last answer_from_sql before terminal)
  - my manual SQL
  - root-cause hypothesis
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

BENCH = REPO / "artifacts" / "bench_phased_vote3_full50_exp126_exp123"
GOLD_DIR = REPO / "data" / "public" / "output"
MANUAL = REPO / "artifacts" / "plan_verify_decisions_full50" / "manual_sql_results.csv"
OUT = REPO / "artifacts" / "plan_verify_decisions_full50" / "exp122_failure_analysis.md"

FAILS = ["task_80", "task_89", "task_163", "task_169", "task_180",
         "task_199", "task_200", "task_344", "task_379", "task_396"]


def load_csv_preview(p: Path, n=5) -> tuple[list[str], list[tuple]]:
    if not p.exists():
        return [], []
    with open(p, newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return [], []
    return rows[0], [tuple(r) for r in rows[1:1+n]]


def extract_last_sql(trace_text: str) -> str:
    """Get the last answer_from_sql action's SQL from trace.log."""
    matches = re.findall(
        r"action=answer_from_sql.*?action_input=\{'sql': \"(.+?)\".*?\}",
        trace_text, re.DOTALL,
    )
    if matches:
        return matches[-1][:400]
    # Try single-quote variant
    matches = re.findall(r"answer_from_sql.*?'sql':\s*'(.+?)'", trace_text, re.DOTALL)
    return matches[-1][:400] if matches else "(no sql found)"


def extract_plan_thought(trace_text: str) -> str:
    """Get PLAN-phase thought (= up to first plan→explore transition)."""
    m = re.search(r"phase=plan.*?thought=(.+)$", trace_text, re.MULTILINE)
    return m.group(1)[:200] if m else "(no plan)"


def main():
    manual = {r["task_id"]: r for r in csv.DictReader(open(MANUAL))}
    out_lines = ["# exp_122 (v5) failure analysis — 10 tasks\n"]

    for tid in FAILS:
        out_lines.append(f"\n---\n\n## {tid}")
        q_path = REPO / "data" / "public" / "input" / tid / "task.json"
        q = json.load(open(q_path))["question"]
        out_lines.append(f"**Q**: {q}")

        # Gold
        gold_h, gold_rows = load_csv_preview(GOLD_DIR / tid / "gold.csv", n=5)
        out_lines.append(f"\n**Gold ({GOLD_DIR/tid/'gold.csv'}):**")
        out_lines.append(f"  cols = {gold_h}")
        out_lines.append(f"  preview = {gold_rows}")

        # Voted prediction
        for vname in ("prediction_voted_v3.csv", "prediction_voted_v2.csv", "prediction_voted.csv"):
            vpath = BENCH / tid / vname
            if vpath.exists():
                vh, vrows = load_csv_preview(vpath, n=5)
                out_lines.append(f"\n**Voted ({vname}):**")
                out_lines.append(f"  cols = {vh}")
                out_lines.append(f"  preview = {vrows}")
                break

        # Per-attempt
        out_lines.append("\n**Per-attempt predictions:**")
        for i in range(3):
            apath = BENCH / tid / f"attempt_{i:02d}" / "prediction.csv"
            if not apath.exists():
                continue
            ah, arows = load_csv_preview(apath, n=3)
            out_lines.append(f"  - att_{i:02d}: cols={ah}, n_rows={len(arows)}, first={arows[:2]}")
            # Plan thought + last SQL
            tlog = BENCH / tid / f"attempt_{i:02d}" / "trace.log"
            if tlog.exists():
                ttext = tlog.read_text()
                out_lines.append(f"    plan: {extract_plan_thought(ttext)}")
                out_lines.append(f"    sql:  {extract_last_sql(ttext)[:200]}")

        # Manual SQL
        m = manual.get(tid, {})
        out_lines.append(f"\n**Manual SQL** (status={m.get('status','?')}):")
        out_lines.append(f"```sql")
        out_lines.append(m.get("sql", "?"))
        out_lines.append(f"```")

    OUT.write_text("\n".join(out_lines))
    print(f"Saved: {OUT}")
    for line in out_lines:
        print(line)


if __name__ == "__main__":
    main()
