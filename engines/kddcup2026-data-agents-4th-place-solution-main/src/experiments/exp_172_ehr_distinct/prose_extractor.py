"""Prose-to-table extractor (startup pass) — exp_170.

A separate pass that runs at task start (alongside the other startup advisors).
For each prose doc (md/pdf) it asks an LLM to reconstruct the narrated records
as a clean SQL table (CREATE TABLE + INSERT VALUES + SELECT *), executes it, and
materialises the result as a CSV under a RUNTIME directory — NEVER the shared
context_dir (so the demo data and other experiments are not polluted).

The duckdb layer (duckdb_unified) and the preamble then scan that runtime dir so
the extracted table is queryable as a `prose__<doc>` view AND shown as a profile
(labelled "prose-extracted") like any other source.

Goal of this pass is EXTRACTION ONLY — faithful reconstruction of the prose
table. It does not answer the question (a separate step does). The question +
knowledge.md are used only to pick which columns are relevant to extract.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import duckdb

from kobushi_core.benchmark.schema import PublicTask
from kobushi_core.model import ModelAdapter, ModelMessage


SYSTEM = """You are a PROSE-TO-TABLE EXTRACTOR. The input is ONE adversarial PROSE document that narrates tabular records (one record per entity/period, with the values stated inside sentences). Your ONLY goal is to faithfully reconstruct those records as a clean SQL table.

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


def _knowledge(task: PublicTask) -> str:
    k = Path(task.context_dir) / "knowledge.md"
    return k.read_text(encoding="utf-8", errors="ignore") if k.is_file() else ""


def _pdf_text(task: PublicTask, pdf: Path) -> str:
    try:
        from experiments.exp_172_ehr_distinct.pdf_text_cache import ensure_task_pdf_cache
        for r in ensure_task_pdf_cache(task):
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


def _doc_items(task: PublicTask) -> list[tuple[str, str]]:
    docdir = Path(task.context_dir) / "doc"
    items: list[tuple[str, str]] = []
    if not docdir.is_dir():
        return items
    for p in sorted(docdir.glob("*")):
        if p.suffix.lower() == ".md":
            items.append((p.name, p.read_text(encoding="utf-8", errors="ignore")))
        elif p.suffix.lower() == ".pdf":
            txt = _pdf_text(task, p)
            if txt:
                items.append((p.name, txt))
    return items


def _extract_sql(raw: str) -> str:
    m = re.search(r"```sql\s*(.+?)```", raw, re.DOTALL | re.IGNORECASE) or re.search(r"```\s*(.+?)```", raw, re.DOTALL)
    return (m.group(1) if m else raw).strip()


_SELECT_SYSTEM = """You pick which prose document(s) hold the TABLE needed to answer a question.

You get a QUESTION, COLUMN SEMANTICS (which table/columns the answer needs), and a short PREVIEW of each prose document. Most documents are DISTRACTORS about other topics; usually exactly ONE document narrates the records the question needs (occasionally two, if the needed columns are split across documents).

Pick the FEWEST documents that together cover the needed columns. Return ONLY JSON:
{"docs": ["<exact filename>", ...]}   (most relevant first; usually a single filename)."""


def select_prose_docs(
    task: PublicTask,
    model: ModelAdapter,
    *,
    preview_chars: int = 1600,
    max_select: int = 2,
) -> list[str]:
    """Doc-routing step: choose which prose doc(s) to extract (the rest are decoys)."""
    items = _doc_items(task)
    names = [n for n, _ in items]
    if len(items) <= 1:
        return names
    previews = "\n\n".join(f"### {n}\n{t[:preview_chars]}" for n, t in items)
    user = (
        f"QUESTION:\n{task.question}\n\n"
        f"COLUMN SEMANTICS (from knowledge.md):\n{_knowledge(task)[:4000]}\n\n"
        f"DOCUMENT PREVIEWS:\n{previews}\n\n"
        "Return the JSON list of the relevant filename(s)."
    )
    try:
        raw = model.complete(
            [ModelMessage(role="system", content=_SELECT_SYSTEM), ModelMessage(role="user", content=user)],
            enable_thinking=False, max_tokens=400,
        )
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        picked = json.loads(m.group(0)).get("docs", []) if m else []
    except Exception:
        picked = []
    # validate against real names (exact, then substring), keep order, cap
    out: list[str] = []
    for p in picked:
        match = next((n for n in names if n == p), None) or next((n for n in names if str(p) in n or n in str(p)), None)
        if match and match not in out:
            out.append(match)
    return (out or names)[:max_select]


# --- exp_171: deterministic prose-needed GATE (design ③) -------------------
# Decide prose-needed (要/不要) BEFORE select+extract, so extraction only fires
# when the answer genuinely lives in a prose doc. This is the only thing that
# can prevent the always-extract decoy regression (e.g. task_14): when the
# target table is queryable, we skip extraction entirely (no prose view, no decoy).
#
# Two-stage by design: this gate only answers 要/不要 + which doc(s) by NAME match;
# if 要, the existing preview-based `select_prose_docs` still picks the actual doc.

_GATE_SYSTEM = """Given a QUESTION and COLUMN SEMANTICS (knowledge.md mapping terms to table.column), name the database TABLE(s) whose records the answer needs. Return ONLY JSON: {"tables":["<table_name>",...]} — schema table name(s) like mf_netvalueperformancehis or lc_financialexpense. NOT columns, NOT .md/.pdf filenames. Usually 1, occasionally 2 if a join is needed."""


def _doc_stems(task: PublicTask) -> dict[str, str]:
    """{lowercase doc stem -> filename} for prose docs (md/pdf)."""
    d = Path(task.context_dir) / "doc"
    if not d.is_dir():
        return {}
    return {p.stem.lower(): p.name for p in d.glob("*") if p.suffix.lower() in (".md", ".pdf")}


def _gate_views(task: PublicTask) -> set[str]:
    """Base queryable view/table names. Uses a THROWAWAY connection (NOT the cached
    get_connection) so the gate — which runs BEFORE extraction — never caches a
    connection that is missing the prose views materialised later."""
    try:
        from experiments.exp_172_ehr_distinct.tools.duckdb_unified import _build_connection
        conn, _ = _build_connection(Path(task.context_dir))
        try:
            return {r[0].lower() for r in conn.execute("SHOW TABLES").fetchall()}
        finally:
            conn.close()
    except Exception:
        return set()


def _target_tables(task: PublicTask, model: ModelAdapter) -> list[str]:
    """LLM names the target table(s) the answer needs (from question + knowledge.md)."""
    try:
        raw = model.complete(
            [
                ModelMessage(role="system", content=_GATE_SYSTEM),
                ModelMessage(
                    role="user",
                    content=f"QUESTION:\n{task.question}\n\nCOLUMN SEMANTICS:\n{_knowledge(task)[:5000]}\n\nReturn the JSON.",
                ),
            ],
            enable_thinking=False, max_tokens=300,
        )
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        return [str(t).lower() for t in json.loads(m.group(0)).get("tables", [])] if m else []
    except Exception:
        return []


def prose_gate(task: PublicTask, model: ModelAdapter) -> tuple[bool, list[str]]:
    """Deterministic prose-needed gate.

    Returns (prose_needed, matched_doc_filenames). prose_needed is True ONLY when a
    named target table maps to a prose doc that actually EXISTS (exact stem, then
    substring) — never on a mere view-name mismatch. So when the target is queryable
    (or no prose doc matches), we return (False, []) and the caller skips extraction.
    """
    tgt = _target_tables(task, model)
    views = _gate_views(task)
    stems = _doc_stems(task)
    docs: list[str] = []
    for t in tgt:
        if t in views:
            continue                                  # target is queryable -> no prose
        if t in stems:
            docs.append(stems[t])                     # exact stem match -> that doc
        else:
            hit = next((stems[s] for s in stems if t in s or s in t), None)
            if hit:
                docs.append(hit)                      # fuzzy substring match
    docs = sorted(set(docs))
    return bool(docs), docs


def extract_prose_tables(
    task: PublicTask,
    extracted_dir: Path,
    model: ModelAdapter,
    *,
    doc_filter: str | None = None,
    only_docs: list[str] | None = None,
    max_docs: int | None = None,
) -> list[dict[str, Any]]:
    """One pass per prose doc -> a materialised CSV under extracted_dir/<task_id>/.

    `doc_filter`: if set, only docs whose name contains it (the doc-routing
    sub-agent that picks the relevant doc is separate / built later).
    Returns metadata per doc (csv path, columns, nrows, or error).
    """
    out = Path(extracted_dir) / task.task_id
    out.mkdir(parents=True, exist_ok=True)
    q = task.question
    know = _knowledge(task)
    items = [
        (n, t) for (n, t) in _doc_items(task)
        if (not doc_filter or doc_filter in n) and (only_docs is None or n in only_docs)
    ]
    if max_docs is not None:
        items = items[:max_docs]

    results: list[dict[str, Any]] = []
    for name, text in items:
        user = (
            f"QUESTION (use ONLY to pick which columns are relevant to extract):\n{q}\n\n"
            f"COLUMN SEMANTICS (from knowledge.md — use it to pick the right table/column + unit):\n{know}\n\n"
            f"PROSE DOCUMENT ({name}):\n{text[:200000]}\n\n"
            "Return the ONE SQL script (CREATE TEMP TABLE + INSERT VALUES + SELECT *)."
        )
        try:
            raw = model.complete(
                [ModelMessage(role="system", content=SYSTEM), ModelMessage(role="user", content=user)],
                enable_thinking=False, max_tokens=32000,
            )
            sql = _extract_sql(raw)
            con = duckdb.connect(":memory:")
            df = con.execute(sql).df()
            csv_path = out / f"prose__{Path(name).stem}.csv"
            df.to_csv(csv_path, index=False)
            (out / f"prose__{Path(name).stem}.sql").write_text(sql, encoding="utf-8")
            results.append({
                "doc": name, "csv": str(csv_path), "view": f"prose__{Path(name).stem}",
                "columns": list(df.columns), "nrows": int(len(df)),
            })
        except Exception as exc:
            results.append({"doc": name, "error": repr(exc)[:200]})
    (out / "_manifest.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return results


def extracted_root() -> Path | None:
    r = os.environ.get("EXP172_EXTRACTED_ROOT")
    return Path(r) if r else None


def extracted_csvs(task_id: str) -> list[Path]:
    """The prose-extracted CSVs for a task (used by duckdb_unified + preamble)."""
    root = extracted_root()
    if root is None:
        return []
    d = root / task_id
    return sorted(d.glob("prose__*.csv")) if d.is_dir() else []
