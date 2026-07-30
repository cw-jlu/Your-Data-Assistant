"""Minimal schema-only preamble for exp_082_cursor_harness.

Shows STRUCTURE ONLY (no data values):
  - CSV: shape + column names + dtypes
  - SQLite: table names + column names + dtypes + row count
  - JSON: type + keys + size
  - DOC: size + first 50 lines + headings outline

Agent must use discovery tools (grep_file, stat_file, read_csv, read_doc)
to fetch actual data values at query time.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from kobushi_core.benchmark.schema import PublicTask

CHARS_PER_TOKEN = 4

_NOTE_FOR_AGENT = """\
## Note for agent
The preamble shows SCHEMA ONLY (no data values). Use discovery tools to fetch
actual data: stat_file (size/rows), grep_file (keyword search), read_csv (sample rows),
read_doc (document sections). Do NOT assume data values from column names alone."""


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


def _csv_schema(path: Path) -> str:
    """Return [CSV] section with shape + column names + dtypes only."""
    rel = path.name
    try:
        import pandas as pd
        df_head = pd.read_csv(path, nrows=5)
        with path.open() as f:
            total_rows = sum(1 for _ in f) - 1  # subtract header
        dtypes_str = ", ".join(
            f"{col}:{str(dtype)[:8]}" for col, dtype in df_head.dtypes.items()
        )
        return (
            f"## [CSV] {rel}\n"
            f"shape: ({total_rows} rows, {len(df_head.columns)} cols)\n"
            f"columns+dtypes: {dtypes_str}\n"
            "(Use read_csv/grep_file to fetch actual values)"
        )
    except Exception as exc:
        return f"## [CSV] {rel}\n[read error: {exc!r}]"


def _sqlite_schema(path: Path) -> str:
    """Return [SQLite] section with table names + column names + dtypes only."""
    rel = path.name
    try:
        conn = sqlite3.connect(path)
        cur = conn.cursor()
        tables = [
            row[0]
            for row in cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
        ]
        parts: list[str] = [f"tables: {tables}"]
        for table in tables:
            cols = cur.execute(f'PRAGMA table_info("{table}")').fetchall()
            col_info = ", ".join(f"{c[1]}:{c[2]}" for c in cols)
            try:
                row_count = cur.execute(
                    f'SELECT COUNT(*) FROM "{table}"'
                ).fetchone()[0]
            except Exception:
                row_count = "?"
            parts.append(f"  {table} ({row_count} rows): {col_info}")
        conn.close()
        return (
            f"## [SQLite] {rel}\n"
            + "\n".join(parts)
            + "\n(Use inspect_sqlite_schema/execute_context_sql for data)"
        )
    except Exception as exc:
        return f"## [SQLite] {rel}\n[open error: {exc!r}]"


def _json_schema(path: Path) -> str:
    """Return [JSON] section with type + keys + size only."""
    rel = path.name
    try:
        size = path.stat().st_size
        raw = path.read_text(encoding="utf-8", errors="replace")
        obj = json.loads(raw)
        if isinstance(obj, list):
            sample_keys = list(obj[0].keys())[:20] if obj and isinstance(obj[0], dict) else []
            body = f"type: list, len={len(obj)}, size={size}B, record_keys={sample_keys}"
        elif isinstance(obj, dict):
            body = f"type: dict, keys={list(obj.keys())[:20]}, size={size}B"
        else:
            body = f"type={type(obj).__name__}, size={size}B"
        return (
            f"## [JSON] {rel}\n{body}\n"
            "(Use read_json/execute_python to inspect actual values)"
        )
    except Exception as exc:
        return f"## [JSON] {rel}\n[read error: {exc!r}]"


def _doc_preview(path: Path) -> str:
    """Return [DOC] section with size + first 50 lines + heading outline only."""
    rel = path.name
    try:
        size = path.stat().st_size
        raw = path.read_text(encoding="utf-8", errors="replace")
        lines = raw.splitlines()
        headings = [ln for ln in lines if ln.startswith("#")][:20]
        preview = "\n".join(lines[:50])
        parts = [f"size={size}B, total_lines={len(lines)}"]
        if headings:
            parts.append("headings: " + " | ".join(headings[:10]))
        parts.append("--- first 50 lines ---")
        parts.append(preview)
        if len(lines) > 50:
            parts.append(f"(+{len(lines)-50} more lines — use read_doc/grep_file)")
        return f"## [DOC] {rel}\n" + "\n".join(parts)
    except Exception as exc:
        return f"## [DOC] {rel}\n[read error: {exc!r}]"


def build_preamble(task: PublicTask, *, max_total_tokens: int = 5_000) -> PreambleResult:
    """Build a schema-only preamble (~5K tokens max)."""
    files = sorted(p for p in task.context_dir.rglob("*") if p.is_file())
    overview = "\n".join(
        f"- {p.relative_to(task.context_dir)} ({p.stat().st_size} bytes)"
        for p in files
    )

    fixed_parts: list[str] = [
        _NOTE_FOR_AGENT,
        "# Workspace overview",
        overview,
    ]

    schema_parts: list[str] = []
    for path in files:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            schema_parts.append(_csv_schema(path))
        elif suffix in {".db", ".sqlite", ".sqlite3"}:
            schema_parts.append(_sqlite_schema(path))
        elif suffix == ".json":
            schema_parts.append(_json_schema(path))
        elif suffix in {".md", ".txt"}:
            schema_parts.append(_doc_preview(path))

    text = "\n\n".join(fixed_parts + schema_parts)

    return PreambleResult(
        text=text,
        detail_level=DetailLevel.MINIMAL,
        char_count=len(text),
        estimated_tokens=_estimate_tokens(text),
        truncated_per_file=False,
        truncated_total=False,
    )
