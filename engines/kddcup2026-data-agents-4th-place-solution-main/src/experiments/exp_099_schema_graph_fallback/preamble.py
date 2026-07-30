"""Deterministic preamble that dumps full context into the agent's first message.

All files are included at full fidelity up to a 150k-token budget. When the
budget is exceeded, oversized files DOWNGRADE to a compact schema-only
summary (header + dtypes + first rows) instead of being dropped entirely.
This preserves "this column exists with this dtype" knowledge even for
gigabyte-scale files. After downgrade exhaustion, sections still drop in
priority order: json(0) > csv(1) > doc(2) > sqlite(3) > knowledge_full(4).

A schema-graph section is appended at the end listing column-pair value
overlaps (MinHash Jaccard >= 0.5) discovered across all files — useful for
DABench's CSV-heavy tasks where FK relationships are implicit.

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

from experiments.exp_099_schema_graph_fallback.schema_graph import (
    build_schema_graph_section,
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
_JSON_MAX_RECORDS = 30

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
    """Return [CSV] section with full content or a representative sample."""
    import random

    import pandas as pd

    rel = path.name
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"## [CSV] {rel}\n[read error: {exc!r}]"

    if len(raw) <= _CSV_MAX_CHARS:
        return f"## [CSV] {rel}\n{raw}"

    # File is large — sample head + tail + random rows for coverage.
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        clipped, _ = _truncate_chars(raw, _CSV_MAX_CHARS)
        return f"## [CSV] {rel}\n[parse warning: {exc!r}]\n{clipped}"

    total = len(df)
    head = df.head(_CSV_HEAD_ROWS)
    tail = df.tail(_CSV_TAIL_ROWS)
    mid_idx = list(
        set(range(_CSV_HEAD_ROWS, max(_CSV_HEAD_ROWS, total - _CSV_TAIL_ROWS)))
    )
    sample_mid = df.iloc[random.sample(mid_idx, min(_CSV_RANDOM_ROWS, len(mid_idx)))] if mid_idx else df.iloc[0:0]
    sampled = (
        pd.concat([head, sample_mid, tail])
        .drop_duplicates()
        .sort_index()
    )
    body = (
        f"shape: {df.shape} (sampled {len(sampled)} rows)\n"
        + sampled.to_string(index=True)
    )
    return f"## [CSV] {rel}\n{body}"


def _sqlite_full(path: Path) -> str:
    """Return [SQLite] section with DDL + first N rows per table."""
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
            try:
                df = pd.read_sql_query(
                    f'SELECT * FROM "{table}" LIMIT {_SQLITE_MAX_ROWS}', conn
                )
                sample = df.to_string()
            except Exception as exc:
                sample = f"[sample read error: {exc!r}]"
            parts.append(
                f"### table: {table}\nDDL:\n{ddl_text}\nsample({_SQLITE_MAX_ROWS}):\n{sample}"
            )
        body = "\n\n".join(parts)
    finally:
        conn.close()
    return f"## [SQLite] {rel}\n{body}"


def _json_full(path: Path) -> str:
    """Return [JSON] section with full content or first N records."""
    rel = path.name
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"## [JSON] {rel}\n[read error: {exc!r}]"

    if len(raw) <= _JSON_MAX_CHARS:
        return f"## [JSON] {rel}\n{raw}"

    # Try to extract first N records if it's a JSON array.
    try:
        obj = json.loads(raw)
    except Exception:
        clipped, _ = _truncate_chars(raw, _JSON_MAX_CHARS)
        return f"## [JSON] {rel}\n[parse error — truncated]\n{clipped}"

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


# --- Schema-only fallbacks (used when full content overflows the budget) ---
# These are intentionally tiny (~500-2000 chars) so even GB-scale files leave
# a footprint in the preamble. The agent still has tools to read raw content.

_SCHEMA_PREVIEW_ROWS = 3


def _csv_schema_only(path: Path) -> str:
    """Compact CSV summary: header + dtypes + first 3 rows + total row count."""
    import pandas as pd

    rel = path.name
    try:
        df_head = pd.read_csv(path, nrows=_SCHEMA_PREVIEW_ROWS)
    except Exception as exc:
        return f"## [CSV] {rel} (schema-only)\n[parse error: {exc!r}]"
    try:
        # Cheap row count: last line index via reading just one column wouldn't
        # work for ragged CSVs; instead estimate from file size / first-row bytes.
        size = path.stat().st_size
        # Rough avg-bytes-per-row from first chunk; fallback to 'unknown'.
        with path.open("rb") as fh:
            sample = fh.read(64 * 1024)
        first_row_bytes = max(1, sample.count(b"\n"))
        approx_rows = max(0, int(size / max(1, len(sample) / first_row_bytes)) - 1)
    except Exception:
        approx_rows = -1

    cols_dtypes = ", ".join(f"{c}:{t}" for c, t in zip(df_head.columns, df_head.dtypes.astype(str)))
    body = (
        f"size: {path.stat().st_size:,} bytes, approx_rows: {approx_rows if approx_rows > 0 else 'unknown'}\n"
        f"columns: {cols_dtypes}\n"
        f"first {len(df_head)} rows:\n{df_head.to_string(index=False)}"
    )
    return f"## [CSV] {rel} (schema-only — file too large for full inclusion)\n{body}"


def _json_schema_only(path: Path) -> str:
    """Compact JSON summary: top-level type + record-keys + first record sample."""
    rel = path.name
    try:
        # Read just the head to avoid loading 100MB+.
        with path.open("rb") as fh:
            head = fh.read(64 * 1024).decode("utf-8", errors="replace")
    except Exception as exc:
        return f"## [JSON] {rel} (schema-only)\n[read error: {exc!r}]"

    # Try to parse a partial array to extract the first record's structure.
    # Strategy: find first top-level object end and parse it alone.
    structure = "(unparseable head)"
    sample_record_str = ""
    try:
        if head.lstrip().startswith("["):
            depth = 0
            start = head.index("{")
            end = -1
            for i in range(start, len(head)):
                if head[i] == "{":
                    depth += 1
                elif head[i] == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            if end > start:
                first_obj = json.loads(head[start:end])
                if isinstance(first_obj, dict):
                    structure = "array of objects with keys: " + ", ".join(first_obj.keys())
                    sample_record_str = json.dumps(first_obj, ensure_ascii=False, default=str)[:1500]
        else:
            obj = json.loads(head[: head.rfind("}") + 1] if "}" in head else head)
            if isinstance(obj, dict):
                structure = "object with keys: " + ", ".join(list(obj.keys())[:30])
    except Exception:
        pass

    body = (
        f"size: {path.stat().st_size:,} bytes\n"
        f"structure: {structure}\n"
        f"sample first record:\n{sample_record_str or '(see read_json tool to inspect)'}"
    )
    return f"## [JSON] {rel} (schema-only — file too large for full inclusion)\n{body}"


def _sqlite_schema_only(path: Path) -> str:
    """SQLite DDL only — no row samples (typically smaller than _sqlite_full)."""
    rel = path.name
    try:
        conn = sqlite3.connect(path)
    except Exception as exc:
        return f"## [SQLite] {rel} (schema-only)\n[open error: {exc!r}]"
    try:
        cur = conn.cursor()
        rows = cur.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        parts: list[str] = [f"size: {path.stat().st_size:,} bytes, tables: {[r[0] for r in rows]}"]
        for name, ddl in rows:
            parts.append(f"### table: {name}\n{ddl or '(no DDL)'}")
        body = "\n\n".join(parts)
    finally:
        conn.close()
    return f"## [SQLite] {rel} (schema-only — file too large for full row samples)\n{body}"


def _doc_schema_only(path: Path) -> str:
    """Doc summary: just first 1000 chars + size."""
    rel = path.name
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            head = fh.read(1000)
    except Exception as exc:
        return f"## [DOC] {rel} (schema-only)\n[read error: {exc!r}]"
    return f"## [DOC] {rel} (schema-only — file too large)\nsize: {path.stat().st_size:,} bytes\nfirst 1000 chars:\n{head}"


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
    sections: list[tuple[str, str, str | None]], budget_chars: int
) -> list[tuple[str, str]]:
    """Fit sections into budget; downgrade oversized files to schema-only first.

    Each section is (key, full_text, fallback_text|None). When over budget:
      Pass 1: replace largest sections (across all types) with their fallback
              if available, until under budget.
      Pass 2: if still over budget, drop entire types in DROP_ORDER (legacy).

    DROP_ORDER (drop first = lowest priority):
      json(0) > csv(1) > doc(2) > sqlite(3) > knowledge_full(4)
    """
    _DROP_ORDER = ["json", "csv", "doc", "sqlite", "knowledge_full"]

    # Materialize as mutable [key, text, fallback]
    result: list[list] = [[k, full, fb] for k, full, fb in sections]

    def total(rows: list[list]) -> int:
        return sum(len(r[1]) for r in rows)

    # Pass 1: greedy downgrade — biggest section first.
    while total(result) > budget_chars:
        # Find the largest section that still has a fallback we haven't used.
        candidates = [
            (idx, len(r[1]) - len(r[2]))  # bytes saved by downgrade
            for idx, r in enumerate(result)
            if r[2] is not None and len(r[2]) < len(r[1])
        ]
        if not candidates:
            break
        candidates.sort(key=lambda x: -x[1])  # largest savings first
        idx, _ = candidates[0]
        result[idx][1] = result[idx][2]  # use fallback
        result[idx][2] = None  # mark consumed

    # Pass 2: if still over, drop entire types in priority order (rare).
    rows = [(r[0], r[1]) for r in result]
    for key in _DROP_ORDER:
        if sum(len(s) for _, s in rows) <= budget_chars:
            break
        rows = [(k, s) for k, s in rows if k != key]
    return rows


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

    # Each entry is (key, full_text, fallback_text|None).
    labeled: list[tuple[str, str, str | None]] = []
    for path in files:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            labeled.append(("csv", _csv_full(path), _csv_schema_only(path)))
        elif suffix in {".db", ".sqlite", ".sqlite3"}:
            labeled.append(("sqlite", _sqlite_full(path), _sqlite_schema_only(path)))
        elif suffix == ".json":
            labeled.append(("json", _json_full(path), _json_schema_only(path)))
        elif suffix in {".md", ".txt"}:
            if path == knowledge_path:
                # knowledge.md full text appended after glossary via knowledge_full key.
                continue
            labeled.append(("doc", _doc_full(path), _doc_schema_only(path)))

    if knowledge_full is not None:
        labeled.append(("knowledge_full", f"# Knowledge (full)\n{knowledge_full}", None))

    fixed_text = "\n\n".join(fixed_parts)

    # Reserve some budget for the schema-graph section (computed after budget_cut).
    # The graph section is small (<3KB) but we add a soft reserve.
    graph_reserve = 4_000
    remaining_budget = max_total_chars - len(fixed_text) - graph_reserve - 4

    kept = _budget_cut(labeled, remaining_budget)
    variable_text = "\n\n".join(s for _, s in kept)

    # Append schema-graph section (cross-file value overlaps).
    try:
        graph_section = build_schema_graph_section(task)
    except Exception:
        graph_section = ""

    parts = [fixed_text]
    if variable_text:
        parts.append(variable_text)
    if graph_section:
        parts.append(graph_section)
    full_text = "\n\n".join(parts)
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
