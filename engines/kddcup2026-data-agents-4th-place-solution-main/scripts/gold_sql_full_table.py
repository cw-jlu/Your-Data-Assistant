"""Print full gold SQL body alongside extracted nouns.

Mapping: task_id → BIRD question_id from docs/BIRD_REVERSE_ENGINEERING.md
SQL source: data/external/bird/dev_20240627/dev.json (indexed by question_id)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DOC = REPO / "docs" / "BIRD_REVERSE_ENGINEERING.md"
BIRD = REPO / "data" / "external" / "bird" / "dev_20240627" / "dev.json"
NOUNS_JSON = REPO / "artifacts" / "plan_verify_decisions_full50" / "results.json"
OUT = REPO / "artifacts" / "plan_verify_decisions_full50" / "gold_sql_full.md"
OUT_CSV = REPO / "artifacts" / "plan_verify_decisions_full50" / "gold_sql_full.csv"


def parse_task_bird_map() -> dict[int, int]:
    """Parse rows like: | task_25 | easy | 1389 | student_club | ... → {25: 1389}"""
    out = {}
    for line in DOC.read_text().splitlines():
        m = re.match(r"^\|\s*task_(\d+)\s*\|\s*[a-zA-Z⚠ ]+\s*\|\s*(\d+)\s*\|", line)
        if m:
            out[int(m.group(1))] = int(m.group(2))
    return out


def main():
    import csv
    task2bird = parse_task_bird_map()
    bird = {row["question_id"]: row for row in json.load(open(BIRD))}
    nouns_data = json.load(open(NOUNS_JSON))
    nouns_map = {int(r["task_id"].split("_")[1]): r for r in nouns_data}

    lines = [
        "# Full gold SQL (= BIRD dev) vs question + nouns\n",
        "| task | question | nouns | full gold SQL |",
        "|---|---|---|---|",
    ]
    rows_for_csv = []
    for tnum in sorted(task2bird):
        bid = task2bird[tnum]
        b = bird.get(bid, {})
        n = nouns_map.get(tnum, {})
        q_full = n.get("question") or b.get("question", "")
        q_md = q_full.replace("|", "\\|")[:80]
        ns_list = n.get("nouns", [])
        ns_str = ",".join(ns_list)
        sql_full = b.get("SQL", "?")
        sql_md = sql_full.replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {tnum} | {q_md} | {ns_str} | `{sql_md}` |")
        rows_for_csv.append({
            "task_id": f"task_{tnum}",
            "bird_question_id": bid,
            "question": q_full,
            "nouns": ns_str,
            "gold_sql": sql_full,
        })
    OUT.write_text("\n".join(lines))
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task_id", "bird_question_id", "question", "nouns", "gold_sql"])
        w.writeheader()
        w.writerows(rows_for_csv)
    for line in lines:
        print(line)
    print(f"\nMD:  {OUT}\nCSV: {OUT_CSV}")


if __name__ == "__main__":
    main()
