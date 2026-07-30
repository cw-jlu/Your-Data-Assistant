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

from experiments.exp_124_knowledge_lean.profile import (
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

# Lean knowledge.md: keep only BIRD-grounded, non-derivable sections.
# DROP: 1 (Introduction = boilerplate), 3 (Metric Definitions = LLM-invented KPIs),
#       4 (Constraints = mostly LLM convention), 5 (Use Cases = LLM-invented SQL).
# KEEP: 2 (Core Entities = value_description-derived), 6 (Ambiguity = alias info).
_KEEP_SECTIONS = {2, 6}


def _filter_knowledge_md(text: str, keep: set[int] = _KEEP_SECTIONS) -> str:
    """Keep only the listed top-level (`## N.`) sections; drop others.

    Preserves the file header (= everything before the first `## N.`).
    """
    lines = text.split("\n")
    out: list[str] = []
    keep_block = True
    for line in lines:
        m = re.match(r"^## (\d+)\.", line)
        if m:
            keep_block = int(m.group(1)) in keep
        if keep_block:
            out.append(line)
    return "\n".join(out)


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
      json(0) > csv(1) > doc(2) > sqlite(3) > knowledge_full(4)
    i.e. json is dropped first, knowledge_full is dropped last.
    """
    _DROP_ORDER = ["json", "csv", "doc", "sqlite", "knowledge_full"]

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
    """Build a full-fidelity preamble; drop sections by priority if over budget."""
    max_total_chars = max_total_tokens * CHARS_PER_TOKEN

    files = sorted(p for p in task.context_dir.rglob("*") if p.is_file())
    overview = "\n".join(
        f"- {p.relative_to(task.context_dir)} ({p.stat().st_size} bytes)"
        for p in files
    )

    # SQL-only experiment: surface the unified DuckDB catalog so the agent
    # knows what views are queryable without an extra `describe_data` round-trip.
    try:
        from experiments.exp_124_knowledge_lean.tools.duckdb_unified import describe_catalog
        catalog_text = describe_catalog(task.context_dir)
    except Exception as exc:
        catalog_text = f"(catalog error: {exc})"
    catalog_section = (
        "# DuckDB views (= SQL surface)\n"
        "Every CSV / JSON / sqlite-DB file below is exposed as a view in a single "
        "DuckDB connection. Query them by `SELECT ... FROM <view_name>`. JOINs "
        "across files are supported.\n\n" + catalog_text
    )

    # Detect doc-only tasks (= no DB/CSV/JSON, only MD/text). For these, SQL
    # cannot compute the answer; the agent must read docs and submit a
    # hand-computed answer via the `answer` tool directly.
    has_data = any(
        p.suffix.lower() in (".db", ".sqlite", ".csv", ".json")
        for p in files
    )
    routing_note = ""
    if not has_data:
        routing_note = (
            "\n\n# IMPORTANT: doc-only routing\n"
            "This task has NO sqlite-DB / CSV / JSON files — only text docs (md/txt).\n"
            "DuckDB views are empty for this task. SQL cannot compute the answer.\n"
            "Use `read_doc` (with `search=` to grep specific terms) or `grep` to read\n"
            "the docs, count / aggregate values mentally, and submit your final answer\n"
            "via the `answer` tool directly with `columns` and `rows`.\n"
            "Do NOT call `answer_from_sql` for this task — there is nothing to query."
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
        "# Workspace overview",
        overview,
        catalog_section + routing_note,
    ]
    glossary = _build_glossary_section(task)
    if glossary is not None:
        fixed_parts.append(glossary)

    # Variable sections — accumulated with a label key for budget_cut.
    knowledge_path = task.context_dir / "knowledge.md"
    knowledge_full: str | None = None
    if knowledge_path.exists():
        try:
            raw_knowledge = knowledge_path.read_text(encoding="utf-8", errors="replace")
            knowledge_full = _filter_knowledge_md(raw_knowledge)
        except Exception:
            pass

    labeled: list[tuple[str, str]] = []
    for path in files:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            labeled.append(("csv", _csv_full(path)))
        elif suffix in {".db", ".sqlite", ".sqlite3"}:
            labeled.append(("sqlite", _sqlite_full(path)))
        elif suffix == ".json":
            labeled.append(("json", _json_full(path)))
        elif suffix in {".md", ".txt"}:
            if path == knowledge_path:
                # knowledge.md full text appended after glossary via knowledge_full key.
                continue
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
