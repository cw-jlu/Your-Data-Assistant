#!/usr/bin/env python3
"""Build a focused GROUP-BY-discrimination POC eval set (CN questions).

6 BULL tasks, Chinese question + Chinese-valued DB (database_cn):
  FIX targets (v8/v9 currently fail; the exp_173 teaching should recover):
    q21  -> group by InvestmentType   (twin "type" column; value-check needed)
    q45  -> group by OrganizationForm (cue 不同<公司属性>, not the counted 基金管理人)
  Regression guards (a NAME / industry column IS the correct GROUP BY key — the
  teaching must NOT push the agent off them):
    q58  -> InvestAdvisorName (fund)
    q207 -> InvestAdvisorName (fund)
    q333 -> FirstIndustryName (stock)
    q635 -> FirstIndustryName (stock)

Layout: data/external/gb_poc/{input,output}. knowledge.md = the rich per-source
demo edition (carries the ccks_(fund|stock) token so the domain router classifies).
"""
from __future__ import annotations
import csv, hashlib, json, re, shutil, sqlite3
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "data/external/gb_poc"
BULL = REPO / "data/external/eval_build/bull_extracted/BULL"
DEMO_IN = REPO / "data/phase2_demo/demo_samples_phase2/input"

# (q_id, kind) — kind is bookkeeping only
TASKS = [
    (21, "fix"),    # InvestmentType
    (45, "fix"),    # OrganizationForm
    (58, "guard"),  # InvestAdvisorName
    (207, "guard"), # InvestAdvisorName
    (333, "guard"), # FirstIndustryName (stock)
    (635, "guard"), # FirstIndustryName (stock)
]
_DB2DOM = {"ccks_fund": "fund", "ccks_stock": "stock", "ccks_macro": "macro"}


def demo_knowledge_by_source():
    """Concatenate the UNIQUE demo knowledge.md editions per finance source."""
    _CCKS = re.compile(r"ccks_(fund|stock|macro)")
    _PREFIX = {"mf": "fund", "lc": "stock", "qt": "stock", "ed": "macro", "in": "macro", "fm": "macro"}
    by_src: dict[str, list[str]] = {"fund": [], "stock": [], "macro": []}
    seen: dict[str, set] = {"fund": set(), "stock": set(), "macro": set()}
    for km in sorted(DEMO_IN.glob("task_*/context/knowledge.md")):
        txt = km.read_text(encoding="utf-8", errors="ignore")
        m = _CCKS.search(txt)
        if m:
            dom = m.group(1)
        else:
            prefs = re.findall(r"\b(mf|lc|qt|ed|in|fm)_", txt.lower())
            if not prefs:
                continue
            dom = _PREFIX.get(Counter(prefs).most_common(1)[0][0])
        if dom not in by_src:
            continue
        h = hashlib.md5(txt.encode()).hexdigest()
        if h in seen[dom]:
            continue
        seen[dom].add(h)
        by_src[dom].append(txt)
    return {d: "\n\n\n---\n\n\n".join(v) for d, v in by_src.items()}


def run_gold(db, sql):
    dbp = BULL / f"database_cn/{db}/{db}.sqlite"
    con = sqlite3.connect(str(dbp))
    cur = con.execute(sql)
    cols = [c[0] for c in cur.description]
    rows = cur.fetchall()
    return cols, rows


def write_gold_csv(path, cols, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow(["" if v is None else v for v in r])


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "input").mkdir(parents=True)
    (OUT / "output").mkdir(parents=True)
    demo_km = demo_knowledge_by_source()
    cn = {ex["q_id"]: ex for ex in json.loads((BULL / "BULL-cn/dev.json").read_text())}
    manifest = []
    for i, (qid, kind) in enumerate(TASKS, 1):
        ex = cn[qid]
        db = ex["db_name"]; sql = ex["sql_query"]
        cols, rows = run_gold(db, sql)
        tid = f"task_{i}"
        # input
        dst_db = OUT / "input" / tid / "context" / "db"
        dst_db.mkdir(parents=True)
        (dst_db / f"{db}.sqlite").symlink_to(BULL / f"database_cn/{db}/{db}.sqlite")
        km = OUT / "input" / tid / "context" / "knowledge.md"
        km_text = demo_km.get(_DB2DOM.get(db, ""), "")
        if not km_text.strip():
            km_text = f"# Step3 skipped\n\nSource database: {db}.\n"
        km.write_text(km_text, encoding="utf-8")
        (OUT / "input" / tid / "task.json").write_text(json.dumps(
            {"task_id": tid, "question": ex["question"]}, ensure_ascii=False, indent=2))
        # output
        write_gold_csv(OUT / "output" / tid / "gold.csv", cols, rows)
        manifest.append({"task_id": tid, "q_id": qid, "kind": kind, "db": db,
                         "gb_col": cols[0], "gold_rows": len(rows),
                         "question": ex["question"], "sql": sql})
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"built {len(manifest)} tasks -> {OUT}")
    for m in manifest:
        print(f"  {m['task_id']} q{m['q_id']:<4} [{m['kind']:5}] {m['db']:11} GB={m['gb_col']:20} rows={m['gold_rows']}")


if __name__ == "__main__":
    main()
