from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path
from typing import Any


def _annotate_column(name: str) -> str:
    lower = name.lower()
    if any(token in lower for token in ("date", "time", "year", "period", "day", "month")):
        return f"{name} [row-grain/time; output only if asked]"
    if lower.endswith("id") or lower in {"id", "code"} or "code" in lower:
        return f"{name} [id/code; output only if asked]"
    return name


def _annotate_columns(columns: list[str]) -> list[str]:
    return [_annotate_column(str(col)) for col in columns]


def _clip(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 24] + "\n[...truncated...]"


def _csv_profile(path: Path, *, max_rows: int = 2) -> str:
    try:
        with path.open(newline="", encoding="utf-8", errors="replace") as handle:
            reader = csv.reader(handle)
            rows = []
            for i, row in enumerate(reader):
                rows.append(row)
                if i >= max_rows:
                    break
    except Exception as exc:
        return f"- CSV {path.name}: read_error={exc!r}"
    if not rows:
        return f"- CSV {path.name}: empty"
    header = rows[0]
    samples = rows[1:]
    return (
        f"- CSV {path.name}: columns={_annotate_columns(header)}"
        + (f"; sample_rows={samples}" if samples else "")
    )


def _json_records(obj: Any) -> list[dict[str, Any]]:
    if isinstance(obj, list) and (not obj or isinstance(obj[0], dict)):
        return obj
    if isinstance(obj, dict):
        for key in ("records", "data", "rows", "items"):
            value = obj.get(key)
            if isinstance(value, list) and (not value or isinstance(value[0], dict)):
                return value
    return []


def _json_profile(path: Path, *, max_rows: int = 2) -> str:
    try:
        obj = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception as exc:
        return f"- JSON {path.name}: read_error={exc!r}"
    records = _json_records(obj)
    if records:
        keys = list(records[0].keys()) if records else []
        return (
            f"- JSON {path.name}: columns={_annotate_columns(keys)}; "
            f"sample_rows={records[:max_rows]}"
        )
    if isinstance(obj, dict):
        return f"- JSON {path.name}: top_keys={list(obj.keys())[:20]}"
    return f"- JSON {path.name}: type={type(obj).__name__}"


def _sqlite_profile(path: Path, *, max_rows: int = 2, max_tables: int = 20) -> str:
    parts = [f"- SQLITE {path.name}:"]
    try:
        conn = sqlite3.connect(path)
    except Exception as exc:
        return f"- SQLITE {path.name}: open_error={exc!r}"
    try:
        cur = conn.cursor()
        tables = [
            row[0]
            for row in cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
        ][:max_tables]
        for table in tables:
            try:
                columns = [
                    row[1]
                    for row in cur.execute(f'PRAGMA table_info("{table}")').fetchall()
                ]
                sample = cur.execute(f'SELECT * FROM "{table}" LIMIT {max_rows}').fetchall()
            except Exception as exc:
                parts.append(f"  table={table}: read_error={exc!r}")
                continue
            parts.append(
                f"  table={table}: columns={_annotate_columns(columns)}; "
                f"sample_rows={sample}"
            )
    finally:
        conn.close()
    return "\n".join(parts)


def _markdown_profile(path: Path, *, max_chars: int = 900) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"- DOC {path.name}: read_error={exc!r}"
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # Prefer headings, markdown table headers, and schema-looking lines.
        if stripped.startswith("#") or "|" in stripped or ":" in stripped or "：" in stripped:
            lines.append(stripped[:240])
        if len("\n".join(lines)) >= max_chars:
            break
    if not lines:
        for line in text.splitlines()[:8]:
            if line.strip():
                lines.append(line.strip()[:240])
    body = "\n    ".join(lines[:20])
    return f"- DOC {path.name}:\n    {_clip(body, max_chars)}"


def build_context_profile(
    context_dir: Path,
    *,
    max_chars: int = 18_000,
    max_sources: int = 40,
) -> str:
    """Build a compact, non-gold context profile for column-shape advising."""
    context_dir = Path(context_dir)
    files = sorted(p for p in context_dir.rglob("*") if p.is_file())
    parts: list[str] = []
    for path in files[:max_sources]:
        suffix = path.suffix.lower()
        rel = path.relative_to(context_dir).as_posix()
        if suffix == ".csv":
            text = _csv_profile(path)
        elif suffix == ".json":
            text = _json_profile(path)
        elif suffix in {".sqlite", ".db"}:
            text = _sqlite_profile(path)
        elif suffix in {".md", ".txt"}:
            text = _markdown_profile(path)
        elif suffix == ".pdf":
            text = f"- PDF {rel}: file available; use only if needed"
        elif suffix in {".mp4", ".mov", ".avi", ".mkv"}:
            text = f"- VIDEO {rel}: media file available"
        else:
            text = f"- FILE {rel}: suffix={suffix or '(none)'}"
        # Preserve relative path because file/table names often encode semantics.
        if not text.startswith("- "):
            text = f"- {rel}: {text}"
        else:
            text = text.replace(path.name, rel, 1)
        parts.append(text)
        if len("\n".join(parts)) >= max_chars:
            break
    return _clip("\n".join(parts), max_chars)
