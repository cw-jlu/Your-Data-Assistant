"""Deterministic preamble that dumps full context into the agent's first message.

All files are included at full fidelity up to a 150k-token budget. When the
budget is exceeded, sections are dropped in priority order:
  json(0) > csv(1) > doc(2) > sqlite(3) > knowledge_full(4)

Caps use a 4 chars/token rule of thumb (no tiktoken dependency).
"""
from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from kobushi_core.benchmark.schema import PublicTask

from experiments.exp_123_unified_data_layer.profile import (
    csv_profile_text,
    json_array_profile,
    sqlite_table_profile,
)

CHARS_PER_TOKEN = 4
DEFAULT_MAX_TOTAL_TOKENS = 150_000

# Max raw chars per file type before we sample/truncate within the section.
_CSV_MAX_CHARS = 50_000
_JSON_MAX_CHARS = 20_000
_DOC_MAX_CHARS = 30_000

_TRUNCATED_MARKER = "\n[...truncated...]"

# How many lines to sample when a CSV exceeds _CSV_MAX_CHARS.
_CSV_HEAD_ROWS = 100
_CSV_TAIL_ROWS = 50
_CSV_RANDOM_ROWS = 50

_SQLITE_MAX_ROWS = 100

# Max JSON records to include when JSON exceeds _JSON_MAX_CHARS.
_JSON_MAX_RECORDS = 10

_NOTE_FOR_AGENT = """\
## Note for agent
All relevant context files are already included below. Prefer reasoning over
the data provided here before calling file-read tools — you may not need them."""


# Kept for runner compatibility (runner.py logs detail_level from PreambleResult).
class DetailLevel(Enum):
    FULL = "full"
    MEDIUM = "medium"
    MINIMAL = "minimal"


@dataclass(frozen=True, slots=True)
class PreambleResult:
    text: str
    detail_level: DetailLevel
    char_count: int
    estimated_tokens: int
    truncated_per_file: bool
    truncated_total: bool

    def metadata(self) -> dict[str, object]:
        return {
            "detail_level": self.detail_level.value,
            "char_count": self.char_count,
            "estimated_tokens": self.estimated_tokens,
            "truncated_per_file": self.truncated_per_file,
            "truncated_total": self.truncated_total,
        }


def _estimate_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


def _truncate_chars(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    if max_chars <= len(_TRUNCATED_MARKER):
        return text[:max_chars], True
    return text[: max_chars - len(_TRUNCATED_MARKER)] + _TRUNCATED_MARKER, True


def _view_sample(conn, view_name: str, max_rows: int = 20,
                 max_chars: int = 8000) -> str:
    """Return a format-agnostic SQL view sample.

    Output: schema + first N rows via SELECT, all through DuckDB.
    Hides csv/json/sqlite distinction — agent sees the data as if it
    were always a SQL view.
    """
    import duckdb as _duckdb
    # Fetch schema (= column name + type).
    try:
        schema = conn.execute(f"DESCRIBE {view_name}").fetchall()
    except _duckdb.Error as exc:
        return f"## SQL view: {view_name}\n[schema error: {exc!r}]"
    cols_str = ", ".join(f"{r[0]} ({r[1]})" for r in schema)
    # Total row count.
    try:
        n_rows = conn.execute(f"SELECT COUNT(*) FROM {view_name}").fetchone()[0]
    except _duckdb.Error:
        n_rows = "?"
    # Sample rows.
    try:
        rows = conn.execute(f"SELECT * FROM {view_name} LIMIT {max_rows}").fetchall()
        col_names = [c[0] for c in schema]
    except _duckdb.Error as exc:
        return (f"## SQL view: {view_name}\nSchema: {cols_str}\n"
                f"Total: {n_rows} rows\n[sample error: {exc!r}]")
    # Render as a markdown table (= readable + compact).
    header = "| " + " | ".join(col_names) + " |"
    sep = "| " + " | ".join("---" for _ in col_names) + " |"
    body_lines = []
    for row in rows:
        cells = []
        for v in row:
            if v is None:
                cells.append("NULL")
            else:
                s = str(v)
                if len(s) > 60:
                    s = s[:57] + "..."
                cells.append(s.replace("|", "\\|").replace("\n", " "))
        body_lines.append("| " + " | ".join(cells) + " |")
    table = "\n".join([header, sep] + body_lines)
    label = "" if n_rows == "?" or n_rows <= max_rows else f" (showing first {max_rows})"
    section = (f"## SQL view: {view_name}\n"
               f"Schema: {cols_str}\n"
               f"Total: {n_rows} rows{label}\n"
               f"{table}")
    if len(section) > max_chars:
        section = section[: max_chars - len(_TRUNCATED_MARKER)] + _TRUNCATED_MARKER
    return section


def _csv_full(path: Path) -> str:
    """Return [CSV] section: profile (per-col stats) + raw content (small files only).

    For files <= _CSV_MAX_CHARS raw, we emit the full CSV body so the agent
    can read every row. For larger files we emit only the descriptive
    profile + a 5-row head sample (full body would just be truncated rows
    anyway, which doesn't help schema understanding).
    """
    rel = path.name
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"## [CSV] {rel}\n[read error: {exc!r}]"

    profile = csv_profile_text(path)
    if len(raw) <= _CSV_MAX_CHARS:
        if profile:
            return f"## [CSV] {rel}\n[profile]\n{profile}\n\n[raw content]\n{raw}"
        return f"## [CSV] {rel}\n{raw}"

    # File is large — emit profile only (head 5 rows are inside the profile block).
    if profile:
        return f"## [CSV] {rel}\n[profile — file too large for full inclusion]\n{profile}"
    # Profile failed (parse error) — fall back to clipped raw text.
    clipped, _ = _truncate_chars(raw, _CSV_MAX_CHARS)
    return f"## [CSV] {rel}\n[profile parse error — raw clip below]\n{clipped}"


def _sqlite_full(path: Path) -> str:
    """Return [SQLite] section: per-table DDL + descriptive profile + 5-row sample."""
    import pandas as pd

    rel = path.name
    try:
        conn = sqlite3.connect(path)
    except Exception as exc:
        return f"## [SQLite] {rel}\n[open error: {exc!r}]"
    try:
        cur = conn.cursor()
        tables = [
            row[0]
            for row in cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
        ]
        parts: list[str] = [f"tables: {tables}"]
        for table in tables:
            try:
                ddl_row = cur.execute(
                    "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
                    (table,),
                ).fetchone()
                ddl_text = ddl_row[0] if ddl_row and ddl_row[0] else "(no DDL)"
            except Exception:
                ddl_text = "(DDL fetch error)"
            # Profile (per-column stats)
            profile_lines = sqlite_table_profile(conn, table)
            # Small head sample to show row layout
            try:
                df_head = pd.read_sql_query(f'SELECT * FROM "{table}" LIMIT 5', conn)
                head_str = df_head.to_string(index=False)
            except Exception as exc:
                head_str = f"[head read error: {exc!r}]"
            parts.append(
                f"### table: {table}\nDDL:\n{ddl_text}\n[profile]\n"
                + "\n".join(profile_lines)
                + f"\n[first 5 rows]\n{head_str}"
            )
        body = "\n\n".join(parts)
    finally:
        conn.close()
    return f"## [SQLite] {rel}\n{body}"


def _json_full(path: Path) -> str:
    """Return [JSON] section: profile (if list-of-objects) + raw/sample content."""
    rel = path.name
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"## [JSON] {rel}\n[read error: {exc!r}]"

    if len(raw) <= _JSON_MAX_CHARS:
        # For small files, emit raw text (agent can read everything).
        # If it's a list-of-dicts, prepend the profile for clarity.
        try:
            obj = json.loads(raw)
            if isinstance(obj, list) and obj and isinstance(obj[0], dict):
                profile = json_array_profile(obj)
                if profile:
                    return f"## [JSON] {rel}\n[profile]\n{profile}\n\n[raw content]\n{raw}"
        except Exception:
            pass
        return f"## [JSON] {rel}\n{raw}"

    # Large JSON.
    try:
        obj = json.loads(raw)
    except Exception:
        clipped, _ = _truncate_chars(raw, _JSON_MAX_CHARS)
        return f"## [JSON] {rel}\n[parse error — truncated]\n{clipped}"

    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        profile = json_array_profile(obj)
        if profile:
            return f"## [JSON] {rel}\n[profile — file too large for full inclusion]\n{profile}"
    # Wrapper-object pattern: { "table": "...", "records": [...] } — common in
    # this dataset. Slice records by count instead of by bytes so we never emit
    # mid-record clipped JSON.
    if isinstance(obj, dict):
        records_key = next(
            (k for k in ("records", "data", "rows", "items") if isinstance(obj.get(k), list)),
            None,
        )
        if records_key:
            recs = obj[records_key]
            head = recs[:_JSON_MAX_RECORDS]
            shell = {k: v for k, v in obj.items() if k != records_key}
            body = (
                f"total_{records_key}={len(recs)} (showing first {len(head)})\n"
                + json.dumps({**shell, records_key: head},
                             ensure_ascii=False, indent=2, default=str)
            )
            return f"## [JSON] {rel}\n{body}"
    # Fallback: pretty-print head.
    if isinstance(obj, list):
        subset = obj[:_JSON_MAX_RECORDS]
        body = (
            f"total_records={len(obj)} (showing first {len(subset)})\n"
            + json.dumps(subset, ensure_ascii=False, indent=2, default=str)
        )
    else:
        pretty = json.dumps(obj, ensure_ascii=False, indent=2, default=str)
        clipped, _ = _truncate_chars(pretty, _JSON_MAX_CHARS)
        body = clipped
    return f"## [JSON] {rel}\n{body}"


def _doc_full(path: Path) -> str:
    """Return [DOC] section with full content (truncated at _DOC_MAX_CHARS)."""
    rel = path.name
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"## [DOC] {rel}\n[read error: {exc!r}]"
    clipped, _ = _truncate_chars(raw, _DOC_MAX_CHARS)
    return f"## [DOC] {rel}\n{clipped}"


_GLOSSARY_TERM_RE = re.compile(r"^-\s+\*\*([^*]+)\*\*\s*[:：]\s*(.+)$", re.MULTILINE)
_GLOSSARY_MAX_CHARS = 800


def _extract_glossary(text: str) -> list[tuple[str, str]]:
    """Extract (term, definition) pairs from `- **term**: definition` lines."""
    return [(m.group(1).strip(), m.group(2).strip()) for m in _GLOSSARY_TERM_RE.finditer(text)]


def _build_glossary_section(task: PublicTask) -> str | None:
    """Return a formatted glossary section from task's knowledge.md, or None if absent."""
    knowledge_path = task.context_dir / "knowledge.md"
    if not knowledge_path.exists():
        return None
    try:
        text = knowledge_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None
    pairs = _extract_glossary(text)
    if not pairs:
        return None
    lines = ["# Glossary"]
    for term, definition in pairs:
        lines.append(f"- **{term}**: {definition}")
    section = "\n".join(lines)
    clipped, _ = _truncate_chars(section, _GLOSSARY_MAX_CHARS)
    return clipped


def _budget_cut(
    sections: list[tuple[str, str]], budget_chars: int
) -> list[tuple[str, str]]:
    """Drop lowest-priority sections until total fits within budget_chars.

    Priority order (drop first = lowest priority):
      view_sample(0) > doc(1) > knowledge_full(2)
    view_sample is dropped first (= agent can still execute_sql on the view
    when needed); knowledge_full is preserved longest.
    Legacy keys (csv/json/sqlite) kept for backwards compatibility but
    unused in exp_123.
    """
    _DROP_ORDER = ["view_sample", "json", "csv", "doc", "sqlite", "knowledge_full"]

    def total(secs: list[tuple[str, str]]) -> int:
        return sum(len(s) for _, s in secs)

    result = list(sections)
    for key in _DROP_ORDER:
        if total(result) <= budget_chars:
            break
        result = [(k, s) for k, s in result if k != key]
    return result


def build_preamble(
    task: PublicTask,
    *,
    max_total_tokens: int = DEFAULT_MAX_TOTAL_TOKENS,
) -> PreambleResult:
    """Build a format-agnostic preamble.

    All data sources (csv/json/sqlite) are exposed uniformly as SQL views
    via the DuckDB unified layer. Text docs (md/txt) are exposed via
    read_doc / grep. The agent sees only TWO surfaces:
      - SQL views (= queryable through execute_sql)
      - Text docs (= readable through read_doc / grep)
    File format identity is intentionally hidden.

    Sections dropped first when over budget:
      view_sample > doc > view_schema > glossary > view_catalog (most stable)
    """
    max_total_chars = max_total_tokens * CHARS_PER_TOKEN

    files = sorted(p for p in task.context_dir.rglob("*") if p.is_file())

    # Get unified catalog + connection (= triggers DuckDB build).
    try:
        from experiments.exp_123_unified_data_layer.tools.duckdb_unified import (
            describe_catalog, get_connection, get_catalog,
        )
        catalog_text = describe_catalog(task.context_dir)
        conn = get_connection(task.context_dir)
        cat = get_catalog(task.context_dir)
        view_names = [e["view"] for e in cat if e.get("view")]
    except Exception as exc:
        catalog_text = f"(catalog error: {exc})"
        conn = None
        view_names = []

    catalog_section = (
        "# Data surface (= SQL views)\n"
        "All data sources are exposed as queryable SQL views in a single "
        "DuckDB connection. Query them with `SELECT ... FROM <view_name>`. "
        "JOINs across views are supported. No raw-file reading is needed.\n\n"
        + catalog_text
    )

    # Text docs section header (= md/txt only, separated from SQL views).
    md_files = [p for p in files if p.suffix.lower() in {".md", ".txt"}]
    docs_overview = ""
    if md_files:
        docs_overview = "\n# Text documents\n" + "\n".join(
            f"- `{p.name}` ({p.stat().st_size} bytes)" for p in md_files
        )

    # Detect doc-only tasks (= no SQL views buildable). For these the
    # agent uses `answer` directly (= no SQL query path).
    routing_note = ""
    if not view_names:
        routing_note = (
            "\n\n# IMPORTANT: doc-only routing\n"
            "No SQL views available for this task — only text documents.\n"
            "Use `read_doc` (with `search=` to find terms) or `grep` to read "
            "the docs, then submit your final answer via the `answer` tool "
            "directly with `columns` and `rows`.\n"
            "Do NOT call `answer_from_sql` — there is nothing to query."
        )

    # Fixed sections (always included).
    fixed_parts: list[str] = [
        _NOTE_FOR_AGENT,
        "# Task overview",
        (
            f"task_id: {task.task_id}\n"
            f"difficulty: {task.difficulty}\n"
            f"question: {task.question}"
        ),
        catalog_section + docs_overview + routing_note,
    ]
    glossary = _build_glossary_section(task)
    if glossary is not None:
        fixed_parts.append(glossary)

    # Variable sections — accumulated with a label key for budget_cut.
    knowledge_path = task.context_dir / "knowledge.md"
    knowledge_full: str | None = None
    if knowledge_path.exists():
        try:
            knowledge_full = knowledge_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            pass

    labeled: list[tuple[str, str]] = []
    # SQL views: one unified view-sample section per view (= replaces
    # csv/json/sqlite per-file dumps).
    if conn is not None:
        for vn in view_names:
            try:
                labeled.append(("view_sample", _view_sample(conn, vn)))
            except Exception as exc:
                labeled.append(("view_sample",
                                f"## SQL view: {vn}\n[sample error: {exc!r}]"))
    # Text docs (md/txt) excluding knowledge.md (= included separately).
    for path in files:
        if path.suffix.lower() in {".md", ".txt"} and path != knowledge_path:
            labeled.append(("doc", _doc_full(path)))

    if knowledge_full is not None:
        labeled.append(("knowledge_full", f"# Knowledge (full)\n{knowledge_full}"))

    fixed_text = "\n\n".join(fixed_parts)
    remaining_budget = max_total_chars - len(fixed_text) - 4  # 4-char separator margin

    kept = _budget_cut(labeled, remaining_budget)

    variable_text = "\n\n".join(s for _, s in kept)
    full_text = fixed_text + ("\n\n" + variable_text if variable_text else "")
    truncated_total = len(full_text) > max_total_chars
    text, _ = _truncate_chars(full_text, max_total_chars)

    return PreambleResult(
        text=text,
        detail_level=DetailLevel.FULL,
        char_count=len(text),
        estimated_tokens=_estimate_tokens(text),
        truncated_per_file=False,
        truncated_total=truncated_total,
    )
