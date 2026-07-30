"""Print a side-by-side table:
   task | question | gold SELECT cols | extracted target nouns | match?

Uses:
  - docs/GOLD_AND_FAILURES.md  (= gold SELECT expressions)
  - artifacts/plan_verify_decisions_full50/results.json (= sub-agent nouns)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DOC = REPO / "docs" / "GOLD_AND_FAILURES.md"
NOUNS_JSON = REPO / "artifacts" / "plan_verify_decisions_full50" / "results.json"
OUT = REPO / "artifacts" / "plan_verify_decisions_full50" / "gold_vs_nouns.md"


# Parse gold doc — extract one row per task_NN
_ROW_RE = re.compile(r"^\| task_(\d+) \| (.+) \|$", re.MULTILINE)


def parse_gold_doc() -> dict[int, dict]:
    """Return {task_num: {'gold_select': str, 'question': str}}."""
    text = DOC.read_text()
    out = {}
    for line in text.splitlines():
        m = re.match(r"^\|\s*task_(\d+)\s*\|", line)
        if not m:
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 4:
            continue
        tnum = int(m.group(1))
        # The "Correct" section has 5 columns: task | gold | gold_first_row | question | agent_sql
        # The "Partial" section has 8 columns: task | score | gold | preview | pred_header | pred_preview | question | agent_sql
        # The "Wrong" section has 7 columns: task | gold | preview | pred_header | pred_preview | question | agent_sql
        if len(cells) == 5:
            gold = cells[1]; question = cells[3]
        elif len(cells) == 8:
            gold = cells[2]; question = cells[6]
        elif len(cells) == 7:
            gold = cells[1]; question = cells[5]
        else:
            continue
        out[tnum] = {"gold_select": gold, "question": question}
    return out


def extract_gold_columns(gold_select: str) -> list[str]:
    """Parse e.g. `ID,SEX,Diagnosis` (plain SELECT) → ['ID','SEX','Diagnosis'].
    For SQL function expressions, grab columns referenced inside.
    """
    s = gold_select
    # Strip backticks
    s = s.strip("`")
    # Strip trailing notes like "(plain SELECT)" or "(plain)"
    s = re.sub(r"\(plain.*?\)", "", s).strip()
    # Tokens of form word.word or bare WORD (= column candidates)
    # First grab dotted refs (T1.col), then bare uppercase/snake
    cols = []
    for m in re.findall(r"\b[A-Z][A-Za-z0-9_]*\.([A-Za-z_][A-Za-z0-9_]*)", s):
        cols.append(m)
    # Also bare comma-separated words at top level (plain SELECT case)
    if not cols:
        # plain SELECT: split by comma
        for part in s.split(","):
            part = part.strip()
            # Strip function calls
            part = re.sub(r".*?\(", "", part).rstrip(")")
            m = re.match(r"^([A-Za-z_][A-Za-z0-9_ ]*)", part)
            if m:
                col = m.group(1).strip()
                if col and col.upper() not in ("CAST", "AS", "REAL", "INT", "CASE", "WHEN", "THEN", "END", "ELSE", "NULL"):
                    cols.append(col)
    # Dedupe preserving order
    seen, out = set(), []
    for c in cols:
        if c not in seen and len(c) >= 2:
            seen.add(c); out.append(c)
    return out


def overlap(nouns: list[str], gold_cols: list[str]) -> tuple[int, int]:
    """Return (matched_n, total_gold). A noun matches if it (or a word in it)
    fuzzy-equals a gold column (case-insensitive, plural-singular)."""
    def norm(s):
        return s.lower().replace("_", "").replace(" ", "").rstrip("s")
    noun_norms = set()
    for n in nouns:
        noun_norms.add(norm(n))
        for w in n.split():
            noun_norms.add(norm(w))
    matched = 0
    for gc in gold_cols:
        g = norm(gc)
        if g in noun_norms:
            matched += 1
            continue
        # Also try contains
        if any(g in nn or nn in g for nn in noun_norms if len(nn) >= 3):
            matched += 1
    return matched, len(gold_cols)


def main():
    gold = parse_gold_doc()
    nouns_data = json.load(open(NOUNS_JSON))
    nouns_map = {int(r["task_id"].split("_")[1]): r for r in nouns_data}

    rows = []
    for tnum in sorted(set(gold) | set(nouns_map)):
        g = gold.get(tnum, {})
        n = nouns_map.get(tnum, {})
        gold_cols = extract_gold_columns(g.get("gold_select", ""))
        nouns = n.get("nouns", [])
        matched, total = overlap(nouns, gold_cols)
        rows.append({
            "task_num": tnum,
            "question": (n.get("question") or g.get("question", ""))[:90],
            "gold_select": g.get("gold_select", "?")[:60],
            "gold_cols": gold_cols,
            "nouns": nouns,
            "match": f"{matched}/{total}" if total else "n/a",
            "ok": (total == 0) or (matched == total),
        })

    # Save
    lines = [
        "# Gold SQL SELECT cols vs Question target nouns\n",
        f"Total: {len(rows)} | full-match: {sum(1 for r in rows if r['ok'])}",
        "",
        "| task | match | question | gold SELECT raw | gold cols parsed | nouns extracted |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        q = r["question"].replace("|", "\\|")
        raw = (gold.get(r["task_num"], {}).get("gold_select", "?")).replace("|", "\\|")
        gc_str = ",".join(r["gold_cols"]) if r["gold_cols"] else "(agg/no col)"
        ns_str = ",".join(r["nouns"]) if r["nouns"] else "-"
        mark = "✓" if r["ok"] else "✗"
        lines.append(
            f"| {r['task_num']} | {mark} {r['match']} | {q} | {raw} | {gc_str} | {ns_str} |"
        )

    OUT.write_text("\n".join(lines))
    # Also print
    for line in lines:
        print(line)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
