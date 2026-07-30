#!/usr/bin/env python3
"""Build a ~40-task external SQLite eval set:
  - 14 Phase-1 tasks the v8 gate-ON run failed yesterday (10 no-prose + 4 prose)
  - ~26 BULL (CN finance, ccks_fund/stock/macro) tasks with non-empty gold answers

Output DABench layout under data/external/eval_set_v1/{input,output}.
"""
from __future__ import annotations
import csv, json, re, shutil, sqlite3
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "data/external/eval_set_v1"
BULL = REPO / "data/external/eval_build/bull_extracted/BULL"
EHR = REPO / "data/external/eval_build/ehrsql2024_repo/data/mimic_iv"
P1_IN = REPO / "data/public/input"
P1_OUT = REPO / "data/public/output"
DEMO_IN = REPO / "data/phase2_demo/demo_samples_phase2/input"

PHASE1_FAILS = [27, 86, 89, 163, 169, 173, 180, 194, 199, 259,  # no-prose
                344, 379, 396, 415]                              # prose
# 14 Phase1 + 16 BULL (finance) + 10 EHR (mimic_iv) = 40
BULL_PICK = {"ccks_fund": 7, "ccks_stock": 5, "ccks_macro": 4}
EHR_PICK = 10


def run_gold(con, sql):
    cur = con.execute(sql)
    cols = [c[0] for c in cur.description]
    rows = cur.fetchall()
    return cols, rows


def demo_knowledge_by_source():
    """Concatenate the UNIQUE Phase-2 demo knowledge.md editions per finance source
    (fund/stock/macro). BULL eval tasks use the SAME ccks_fund/stock/macro DBs as the
    demo, so the demo's rich per-source schema guide (twin-table disambiguation, units,
    joins) is the A-board-equivalent input — assign it instead of a bare placeholder.
    Classified by ccks token, else by dominant table prefix (mf/lc/qt/ed/in/fm)."""
    import hashlib
    from collections import Counter
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
    manifest = []
    n = 0

    # --- 1) Phase-1 failing tasks ---
    for orig in PHASE1_FAILS:
        n += 1
        tid = f"task_{n}"
        src_in = P1_IN / f"task_{orig}"
        src_gold = P1_OUT / f"task_{orig}" / "gold.csv"
        dst_in = OUT / "input" / tid
        shutil.copytree(src_in / "context", dst_in / "context")
        q = json.loads((src_in / "task.json").read_text())["question"]
        (dst_in / "task.json").write_text(json.dumps(
            {"task_id": tid, "question": q}, ensure_ascii=False, indent=2))
        shutil.copy(src_gold, _mk(OUT / "output" / tid / "gold.csv"))
        manifest.append({"task_id": tid, "source": "phase1", "orig": f"task_{orig}",
                         "kind": "prose" if orig in (344, 379, 396, 415) else "no-prose"})

    # --- 2) BULL tasks (non-empty gold, 1..25 rows, mixed dbs) ---
    demo_km = demo_knowledge_by_source()  # A-board-equivalent per-source schema guides
    _DB2DOM = {"ccks_fund": "fund", "ccks_stock": "stock", "ccks_macro": "macro"}
    dev = json.loads((BULL / "BULL-en/dev.json").read_text())
    cons = {db: sqlite3.connect(str(BULL / f"database_en/{db}/{db}.sqlite"))
            for db in BULL_PICK}
    picked = {db: 0 for db in BULL_PICK}
    for ex in dev:
        db = ex.get("db_name"); sql = ex.get("sql_query")
        if db not in BULL_PICK or picked[db] >= BULL_PICK[db]:
            continue
        try:
            cols, rows = run_gold(cons[db], sql)
        except Exception:
            continue
        # keep non-empty, non-degenerate, modest-size gold
        if not rows or not (1 <= len(rows) <= 25):
            continue
        if not any(any(c is not None for c in r) for r in rows):
            continue
        n += 1; picked[db] += 1
        tid = f"task_{n}"
        dst_db = OUT / "input" / tid / "context" / "db"
        dst_db.mkdir(parents=True)
        # symlink the shared BULL sqlite (avoid 20MB duplication)
        (dst_db / f"{db}.sqlite").symlink_to(BULL / f"database_en/{db}/{db}.sqlite")
        # A-board-equivalent knowledge.md: assign the demo's rich per-source schema
        # guide (same ccks_fund/stock/macro DB) — twin-table disambiguation, units,
        # joins, worked examples. Carries the ccks token so the domain router still
        # classifies via `ccks_token`. Falls back to a placeholder if no demo guide.
        _km_text = demo_km.get(_DB2DOM.get(db, ""), "")
        _km_path = OUT / "input" / tid / "context" / "knowledge.md"
        _km_path.parent.mkdir(parents=True, exist_ok=True)
        if _km_text.strip():
            _km_path.write_text(_km_text, encoding="utf-8")
        else:
            _write_placeholder_knowledge(_km_path, db)
        (OUT / "input" / tid / "task.json").write_text(json.dumps(
            {"task_id": tid, "question": ex["question"]},
            ensure_ascii=False, indent=2))
        write_gold_csv(OUT / "output" / tid / "gold.csv", cols, rows)
        manifest.append({"task_id": tid, "source": "bull", "db": db,
                         "q_id": ex["q_id"], "gold_rows": len(rows), "sql": sql})

    # --- 3) EHRSQL (mimic_iv demo) tasks ---
    # cleaned copy (junk view `patient_treatment_response` dropped so the duckdb
    # layer registers all base tables without a prepare error)
    ehr_db = REPO / "data/external/eval_build/mimic_iv_clean.sqlite"
    ehr_con = sqlite3.connect(str(ehr_db))
    ehr_picked = 0
    for split in ("valid", "train"):
        if ehr_picked >= EHR_PICK:
            break
        data = {d["id"]: d["question"]
                for d in json.loads((EHR / split / "data.json").read_text())["data"]}
        labels = json.loads((EHR / split / "label.json").read_text())
        for qid, sql in labels.items():
            if ehr_picked >= EHR_PICK or sql == "null" or qid not in data:
                continue
            try:
                cols, rows = run_gold(ehr_con, sql)
            except Exception:
                continue
            if not rows or not (1 <= len(rows) <= 25):
                continue
            if not any(any(c is not None for c in r) for r in rows):
                continue
            n += 1; ehr_picked += 1
            tid = f"task_{n}"
            dst_db = OUT / "input" / tid / "context" / "db"
            dst_db.mkdir(parents=True)
            (dst_db / "mimic_iv.sqlite").symlink_to(ehr_db)
            # EHR routes via schema fingerprint (mimic tables), so a bare
            # placeholder card is enough — matches the demo's empty EHR knowledge.md.
            _write_placeholder_knowledge(OUT / "input" / tid / "context" / "knowledge.md", None)
            (OUT / "input" / tid / "task.json").write_text(json.dumps(
                {"task_id": tid, "question": data[qid]},
                ensure_ascii=False, indent=2))
            write_gold_csv(OUT / "output" / tid / "gold.csv", cols, rows)
            manifest.append({"task_id": tid, "source": "ehrsql", "db": "mimic_iv",
                             "q_id": qid, "gold_rows": len(rows), "sql": sql})

    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    n_p1 = sum(1 for m in manifest if m["source"] == "phase1")
    n_bull = sum(1 for m in manifest if m["source"] == "bull")
    n_ehr = sum(1 for m in manifest if m["source"] == "ehrsql")
    print(f"built {len(manifest)} tasks -> {OUT}")
    print(f"  phase1 fails: {n_p1}  |  bull: {n_bull} {picked}  |  ehr: {n_ehr}")


def _write_placeholder_knowledge(path: Path, ccks_db: str | None) -> None:
    """Minimal placeholder knowledge.md (no schema hints), matching the demo's
    empty '# Step3 skipped' EHR cards. For BULL it embeds the `ccks_(fund|stock|
    macro)` source-id token so the deterministic domain router classifies the
    task via its primary `ccks_token` signal (same as real Phase-2 finance tasks).
    EHR needs no token — it routes off the mimic schema fingerprint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "# Step3 skipped\n\nevidence 为空，未生成 knowledge guide。\n"
    if ccks_db:  # e.g. "ccks_fund" -> domain router reads the token
        body = f"# Step3 skipped\n\nSource database: {ccks_db}. evidence 为空，未生成 knowledge guide。\n"
    path.write_text(body, encoding="utf-8")


def _mk(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


if __name__ == "__main__":
    main()
