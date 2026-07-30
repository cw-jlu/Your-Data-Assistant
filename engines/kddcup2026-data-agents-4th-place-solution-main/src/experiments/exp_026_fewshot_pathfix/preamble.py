"""Deterministic preamble that profiles a task's `context/` directory upfront.

The preamble is prepended to the agent's first user message so that the model
can skip several rounds of file inspection. Three detail levels are tried in
order; if the full preamble exceeds the global cap we fall back to shorter
samples, then drop describe(), then hard-truncate.

Caps are expressed in estimated tokens using a 4 chars/token rule of thumb.
"""
from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import pandas as pd

from kobushi_core.benchmark.schema import PublicTask

CHARS_PER_TOKEN = 4
DEFAULT_MAX_TOTAL_TOKENS = 8000
DEFAULT_MAX_PER_FILE_TOKENS = 1500


class DetailLevel(Enum):
    FULL = "full"
    MEDIUM = "medium"
    MINIMAL = "minimal"


_LEVEL_PARAMS: dict[DetailLevel, dict[str, int | bool]] = {
    DetailLevel.FULL: {"sample_rows": 5, "include_describe": True, "json_preview_chars": 4000},
    DetailLevel.MEDIUM: {"sample_rows": 3, "include_describe": False, "json_preview_chars": 2000},
    DetailLevel.MINIMAL: {"sample_rows": 1, "include_describe": False, "json_preview_chars": 800},
}

_TRUNCATED_MARKER = "\n[...truncated...]"


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


def _describe_csv(path: Path, level: DetailLevel) -> str:
    params = _LEVEL_PARAMS[level]
    sample_rows = int(params["sample_rows"])
    include_describe = bool(params["include_describe"])
    try:
        df = pd.read_csv(path, nrows=10000)
    except Exception as exc:
        return f"[csv read error: {exc!r}]"
    parts = [f"shape: {df.shape}", f"dtypes:\n{df.dtypes.to_string()}"]
    parts.append(f"head({sample_rows}):\n{df.head(sample_rows).to_string()}")
    if include_describe:
        try:
            parts.append(f"describe:\n{df.describe(include='all').to_string()}")
        except Exception:
            pass
    return "\n\n".join(parts)


def _describe_json(path: Path, level: DetailLevel) -> str:
    params = _LEVEL_PARAMS[level]
    preview_chars = int(params["json_preview_chars"])
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"[json read error: {exc!r}]"
    try:
        obj = json.loads(text)
    except Exception:
        body, _ = _truncate_chars(text, preview_chars)
        return f"[json parse error] preview:\n{body}"
    pretty = json.dumps(obj, ensure_ascii=False, indent=2, default=str)
    body, _ = _truncate_chars(pretty, preview_chars)
    return f"size_chars={len(text)}\npreview:\n{body}"


def _describe_sqlite(path: Path, level: DetailLevel) -> str:
    params = _LEVEL_PARAMS[level]
    sample_rows = int(params["sample_rows"])
    try:
        conn = sqlite3.connect(path)
    except Exception as exc:
        return f"[sqlite open error: {exc!r}]"
    try:
        cur = conn.cursor()
        tables = [
            row[0]
            for row in cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
        ]
        parts = [f"tables: {tables}"]
        for table in tables:
            try:
                ddl = cur.execute(
                    "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
                    (table,),
                ).fetchone()
                ddl_text = ddl[0] if ddl and ddl[0] else "(no DDL)"
            except Exception:
                ddl_text = "(DDL fetch error)"
            try:
                df = pd.read_sql_query(
                    f'SELECT * FROM "{table}" LIMIT {sample_rows}', conn
                )
                sample = df.to_string()
            except Exception as exc:
                sample = f"[sample read error: {exc!r}]"
            parts.append(f"### table: {table}\nDDL:\n{ddl_text}\nsample({sample_rows}):\n{sample}")
        return "\n\n".join(parts)
    finally:
        conn.close()


def _describe_text(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"[text read error: {exc!r}]"
    return text


def _describe_other(path: Path) -> str:
    try:
        size = path.stat().st_size
    except OSError:
        size = -1
    return f"(unknown filetype, {size} bytes)"


def _file_section(path: Path, base: Path, level: DetailLevel) -> str:
    rel = path.relative_to(base)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        body = _describe_csv(path, level)
    elif suffix == ".json":
        body = _describe_json(path, level)
    elif suffix in {".db", ".sqlite", ".sqlite3"}:
        body = _describe_sqlite(path, level)
    elif suffix in {".md", ".txt"}:
        body = _describe_text(path)
    else:
        body = _describe_other(path)
    return f"## {rel}\n{body}"


_GLOSSARY_TERM_RE = re.compile(r"^-\s+\*\*([^*]+)\*\*\s*[:：]\s*(.+)$", re.MULTILINE)
_GLOSSARY_MAX_CHARS = 800


def _extract_glossary(text: str) -> list[tuple[str, str]]:
    """Extract (term, definition) pairs from `- **term**: definition` lines."""
    return [(m.group(1).strip(), m.group(2).strip()) for m in _GLOSSARY_TERM_RE.finditer(text)]


def _build_glossary_section(task: "PublicTask") -> str | None:
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


@dataclass(slots=True)
class _BuildOutcome:
    text: str
    truncated_per_file: bool


def _assemble(
    *,
    task: PublicTask,
    files: list[Path],
    level: DetailLevel,
    max_per_file_chars: int,
) -> _BuildOutcome:
    overview = "\n".join(
        f"- {p.relative_to(task.context_dir)} ({p.stat().st_size} bytes)" for p in files
    )
    truncated_per_file = False
    sections: list[str] = []
    for path in files:
        section = _file_section(path, task.context_dir, level)
        clipped, hit = _truncate_chars(section, max_per_file_chars)
        if hit:
            truncated_per_file = True
        sections.append(clipped)
    body_parts: list[str] = [
        "# Task overview",
        f"task_id: {task.task_id}\n"
        f"difficulty: {task.difficulty}\n"
        f"question: {task.question}",
        "# Workspace overview",
        overview,
    ]
    glossary = _build_glossary_section(task)
    if glossary is not None:
        body_parts.append(glossary)
    body_parts.append("# File details")
    body_parts.extend(sections)
    body = "\n\n".join(body_parts)
    return _BuildOutcome(text=body, truncated_per_file=truncated_per_file)


def build_preamble(
    task: PublicTask,
    *,
    max_total_tokens: int = DEFAULT_MAX_TOTAL_TOKENS,
    max_per_file_tokens: int = DEFAULT_MAX_PER_FILE_TOKENS,
) -> PreambleResult:
    """Build a preamble for ``task``, gracefully degrading detail when over caps."""
    max_total_chars = max_total_tokens * CHARS_PER_TOKEN
    max_per_file_chars = max_per_file_tokens * CHARS_PER_TOKEN

    files = sorted(p for p in task.context_dir.rglob("*") if p.is_file())

    final_outcome: _BuildOutcome | None = None
    final_level = DetailLevel.MINIMAL
    for level in (DetailLevel.FULL, DetailLevel.MEDIUM, DetailLevel.MINIMAL):
        outcome = _assemble(
            task=task,
            files=files,
            level=level,
            max_per_file_chars=max_per_file_chars,
        )
        final_outcome = outcome
        final_level = level
        if len(outcome.text) <= max_total_chars:
            break

    assert final_outcome is not None
    text, truncated_total = _truncate_chars(final_outcome.text, max_total_chars)
    return PreambleResult(
        text=text,
        detail_level=final_level,
        char_count=len(text),
        estimated_tokens=_estimate_tokens(text),
        truncated_per_file=final_outcome.truncated_per_file,
        truncated_total=truncated_total,
    )
