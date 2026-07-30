"""Deterministic preamble that dumps full context into the agent's first message.

All files are included at full fidelity up to a 150k-token budget. When the
budget is exceeded, sections are dropped in priority order:
  json(0) > csv(1) > doc(2) > sqlite(3) > knowledge_full(4)

Caps use a 4 chars/token rule of thumb (no tiktoken dependency).
(exp_097: + M-Schema section after glossary, ported from exp_088 sans Plan-first)
"""
from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from kobushi_core.benchmark.schema import PublicTask

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


# ============================================================================
# M-Schema preamble (exp_097, [context-single] axis)
# ============================================================================
# Source-of-truth port from exp_088_r3_mschema_plan_first (Plan-first Rule
# 12a is INTENTIONALLY OMITTED here — that addition was the main driver of
# the exp_088 three-way trap (commitment-induced extras + T-diversity ×
# union × Plan-first interaction). We retain only the structured DDL +
# sample-row block injected at the same position as exp_088 (between
# `# Workspace overview` and the variable sections).
#
# Combined with `_ATTEMPT_TEMPS = (0.6,)` (single-attempt class), two of
# the three trap components (Plan-first × T-diversity × union) are
# structurally removed: T-diversity is fixed at a single value and union
# is physically not invoked (n_attempts=1 → first answer passes through
# `_signature_majority_merge` unchanged).

_MSCHEMA_SAMPLE_VALUES = 5  # = exp_088 default; do not change without
                            # also re-baselining exp_088 attribution.
_MSCHEMA_MAX_CHARS = 4000


def _format_value_for_schema(v: Any) -> str:
    if v is None:
        return "NULL"
    s = str(v)
    return repr(s) if any(c in s for c in [" ", ",", ":"]) else s


def _build_csv_mschema(path: Path) -> str | None:
    """Return a single-table M-Schema block for a CSV file, or None on error."""
    import pandas as pd
    try:
        df = pd.read_csv(path, nrows=20)
    except Exception:
        return None
    if df.empty:
        return None
    rel = path.name
    lines = [f"### Table {rel} (CSV, ~{path.stat().st_size} bytes)"]
    for col in df.columns[:20]:
        dtype = str(df[col].dtype)
        samples = df[col].dropna().head(_MSCHEMA_SAMPLE_VALUES).tolist()
        sample_str = ", ".join(_format_value_for_schema(v) for v in samples)
        lines.append(f"  - ({col}, {dtype}, e.g. {sample_str})")
    if len(df.columns) > 20:
        lines.append(f"  - ... ({len(df.columns) - 20} more columns omitted)")
    return "\n".join(lines)


def _build_sqlite_mschema(path: Path) -> str | None:
    """Return DDL + sample-values M-Schema block for a SQLite file."""
    rel = path.name
    try:
        conn = sqlite3.connect(path)
    except Exception:
        return None
    try:
        cur = conn.cursor()
        tables = [
            row[0]
            for row in cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
        ]
        parts: list[str] = [f"### Database {rel} ({len(tables)} tables)"]
        for table in tables:
            try:
                cols_info = cur.execute(f'PRAGMA table_info("{table}")').fetchall()
                col_names = [r[1] for r in cols_info]
                col_types = {r[1]: r[2] for r in cols_info}
                pk_set = {r[1] for r in cols_info if r[5]}
                rows = cur.execute(
                    f'SELECT * FROM "{table}" LIMIT {_MSCHEMA_SAMPLE_VALUES}'
                ).fetchall()
                parts.append(f"#### {table}")
                for ci, cname in enumerate(col_names[:20]):
                    samples = [
                        _format_value_for_schema(r[ci]) for r in rows if ci < len(r)
                    ]
                    pk_marker = " PK" if cname in pk_set else ""
                    sample_str = ", ".join(samples) if samples else "(empty)"
                    parts.append(
                        f"  - ({cname}, {col_types.get(cname, '?')}{pk_marker}, e.g. {sample_str})"
                    )
                if len(col_names) > 20:
                    parts.append(f"  - ... ({len(col_names) - 20} more columns omitted)")
                fks = cur.execute(f'PRAGMA foreign_key_list("{table}")').fetchall()
                if fks:
                    fk_lines = [
                        f"    FK: {table}.{r[3]} -> {r[2]}.{r[4]}" for r in fks
                    ]
                    parts.extend(fk_lines)
            except Exception:
                continue
        body = "\n".join(parts)
    finally:
        conn.close()
    return body


def _build_mschema_section(task: PublicTask) -> str | None:
    """Build the additive `## [M-Schema]` block from CSV + SQLite files only.

    Strict rules (mirror exp_088 precisely for attribution purity):
      - Read ONLY files inside task.context_dir (gold.csv lives elsewhere).
      - JSON / DOC / knowledge.md are excluded (not tabular).
      - Cap total at _MSCHEMA_MAX_CHARS; sort tables by size desc, drop overflow.
    """
    candidates: list[tuple[int, str]] = []
    for path in sorted(task.context_dir.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        block: str | None = None
        if suffix == ".csv":
            block = _build_csv_mschema(path)
        elif suffix in {".db", ".sqlite", ".sqlite3"}:
            block = _build_sqlite_mschema(path)
        if block:
            candidates.append((path.stat().st_size, block))
    if not candidates:
        return None
    candidates.sort(key=lambda x: -x[0])
    parts = [
        "## [M-Schema]",
        "(structured DDL + sample values; refer to this block while reasoning.)",
    ]
    used = sum(len(p) for p in parts) + len(parts)
    for _, block in candidates:
        if used + len(block) + 2 > _MSCHEMA_MAX_CHARS:
            break
        parts.append(block)
        used += len(block) + 2
    if len(parts) <= 2:
        return None
    return "\n\n".join(parts)


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

    # exp_097 [context-single]: structured DDL + sample-row block (M-Schema)
    # ported from exp_088 minus Plan-first Rule 12a. Combined with
    # single-attempt T_ATTEMPTS=(0.6,) and k=1 union, two of the three
    # exp_088 trap components are structurally removed.
    mschema = _build_mschema_section(task)
    if mschema is not None:
        fixed_parts.append(mschema)

    # Variable sections — accumulated with a label key for budget_cut.
    knowledge_path = task.context_dir / "knowledge.md"
    knowledge_full: str | None = None
    if knowledge_path.exists():
        try:
            knowledge_full = knowledge_path.read_text(encoding="utf-8", errors="replace")
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
