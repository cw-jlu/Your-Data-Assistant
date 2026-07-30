"""Pre-flight task brief (v7.1 / N-3).

The agent burns 1-2 steps every task on ``list_context`` + a basic schema
read to figure out what it is looking at. That cost is silent — invisible
on success, devastating on max_steps-bound hard tasks. This module
generates a structured **brief** *before* the agent's first model call,
saving the early discovery steps:

- Available files (path · size · type · row count when cheap)
- A SQLite tables-and-row-counts inventory if any DB is present
- A keyword-derived hint at expected answer cardinality (singular vs plural)
- Likely join keys hinted by the question (column names that appear in
  both the question and a table column)

The brief is a pure read-only scan — no LLM, no tool execution, no
PII / secret extraction. It runs once per task at runner entry and gets
inlined into the task prompt as an advisory paragraph.

Time budget
-----------

The brief is hard-capped at ~1.5 s of wall-clock and 5000 chars output.
Anything that would exceed those limits gets truncated rather than
fail-blocking the task.
"""

from __future__ import annotations

import csv
import re
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.memory.task_shape import TaskShape


_BRIEF_MAX_FILES = 12               # only the most-relevant N files surfaced
_BRIEF_MAX_DB_TABLES = 16           # SQLite tables before truncation
_BRIEF_MAX_OUTPUT_CHARS = 5000      # final string cap
_CSV_PEEK_ROWS = 200                # sample size for cheap row count / dtype peek
_FILE_TYPE_BUCKETS: dict[str, str] = {
    ".csv": "csv",
    ".tsv": "csv",
    ".json": "json",
    ".jsonl": "json",
    ".db": "db",
    ".sqlite": "db",
    ".sqlite3": "db",
    ".md": "doc",
    ".txt": "doc",
    ".pdf": "pdf",
    ".xlsx": "excel",
    ".xls": "excel",
    ".parquet": "parquet",
}
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]{2,}")


@dataclass(frozen=True, slots=True)
class FileSummary:
    rel_path: str
    bucket: str                  # csv / json / db / doc / pdf / excel / parquet / other
    size_bytes: int
    columns: tuple[str, ...] = ()       # csv only
    row_estimate: int | None = None     # csv quick scan, None when unknown
    db_tables: tuple[tuple[str, int | None], ...] = ()  # (table_name, row_count)


@dataclass(frozen=True, slots=True)
class TaskBrief:
    """Compact pre-flight summary the agent gets in its very first prompt."""
    files: tuple[FileSummary, ...]
    has_knowledge_md: bool
    answer_kind_hint: str            # "singular" / "plural" / "aggregate" / "unknown"
    join_key_candidates: tuple[str, ...]
    truncated: bool

    def render(self) -> str:
        if not self.files and not self.has_knowledge_md:
            return ""
        lines: list[str] = []
        lines.append("Pre-flight task brief (read-only inventory of context/):")
        if self.has_knowledge_md:
            lines.append("- knowledge.md is present and was inlined above.")
        for fs in self.files:
            line = f"- {fs.rel_path} [{fs.bucket}, {_humanize_bytes(fs.size_bytes)}]"
            if fs.columns:
                cols = ", ".join(fs.columns[:8])
                if len(fs.columns) > 8:
                    cols += f" (+{len(fs.columns) - 8} more)"
                line += f" — columns: {cols}"
            if fs.row_estimate is not None:
                line += f" — ~{fs.row_estimate}+ rows scanned"
            if fs.db_tables:
                tab_summaries = []
                for name, rc in fs.db_tables[:_BRIEF_MAX_DB_TABLES]:
                    tab_summaries.append(f"{name}({rc if rc is not None else '?'} rows)")
                if len(fs.db_tables) > _BRIEF_MAX_DB_TABLES:
                    tab_summaries.append(f"... +{len(fs.db_tables) - _BRIEF_MAX_DB_TABLES} more")
                line += " — tables: " + ", ".join(tab_summaries)
            lines.append(line)
        if self.answer_kind_hint != "unknown":
            lines.append(f"- Question hints answer cardinality: {self.answer_kind_hint}")
        if self.join_key_candidates:
            keys = ", ".join(self.join_key_candidates[:6])
            lines.append(f"- Likely join keys (column names appearing in question): {keys}")
        if self.truncated:
            lines.append("- Inventory truncated (cap reached); use list_context for the rest.")
        rendered = "\n".join(lines)
        if len(rendered) > _BRIEF_MAX_OUTPUT_CHARS:
            rendered = rendered[: _BRIEF_MAX_OUTPUT_CHARS - 16].rstrip() + "\n... [truncated]"
        return rendered


def _humanize_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n / 1024 ** (('KB', 'MB', 'GB').index(unit) + 1):.1f} {unit}"
        n //= 1024
    return f"{n} B"


def _bucket_for(path: Path) -> str:
    return _FILE_TYPE_BUCKETS.get(path.suffix.lower(), "other")


def _peek_csv(path: Path) -> tuple[tuple[str, ...], int | None]:
    """Read the first row + count up to ``_CSV_PEEK_ROWS`` rows."""
    try:
        with path.open(newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header is None:
                return (), 0
            row_count = 0
            for _ in reader:
                row_count += 1
                if row_count >= _CSV_PEEK_ROWS:
                    return tuple(str(c) for c in header), row_count
            return tuple(str(c) for c in header), row_count
    except OSError:
        return (), None


def _peek_sqlite(path: Path) -> tuple[tuple[str, int | None], ...]:
    """List tables and row counts. Cheap COUNT(*) — capped at table count."""
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error:
        return ()
    try:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
        tables = [str(row[0]) for row in cur.fetchall()]
        out: list[tuple[str, int | None]] = []
        for name in tables[: _BRIEF_MAX_DB_TABLES * 2]:
            try:
                cur2 = conn.execute(f'SELECT COUNT(*) FROM "{name}"')
                count = int(cur2.fetchone()[0])
            except sqlite3.Error:
                count = None
            out.append((name, count))
        return tuple(out)
    finally:
        try:
            conn.close()
        except sqlite3.Error:
            pass


def _walk_context(context_dir: Path) -> Iterable[Path]:
    if not context_dir.is_dir():
        return ()
    return (p for p in context_dir.rglob("*") if p.is_file())


def _classify_answer_kind(question: str, shape: TaskShape) -> str:
    if shape.is_aggregate_question:
        return "aggregate"
    if shape.is_plural_question:
        return "plural"
    if shape.is_singular_question:
        return "singular"
    return "unknown"


def _collect_question_tokens(question: str) -> set[str]:
    return {tok.lower() for tok in _TOKEN_RE.findall(question or "")}


def _join_key_candidates(
    files: list[FileSummary], question: str
) -> tuple[str, ...]:
    tokens = _collect_question_tokens(question)
    if not tokens:
        return ()
    seen: list[str] = []
    seen_lower: set[str] = set()
    for fs in files:
        for col in fs.columns:
            col_lower = col.lower()
            if col_lower in seen_lower:
                continue
            # A column is a "candidate join key" when its name OR a
            # word-tokenized form (split on _) appears in the question.
            parts = {col_lower, *col_lower.split("_")}
            if parts & tokens:
                seen.append(col)
                seen_lower.add(col_lower)
        for tname, _rc in fs.db_tables:
            tname_lower = tname.lower()
            if tname_lower in tokens and tname_lower not in seen_lower:
                seen.append(tname)
                seen_lower.add(tname_lower)
    return tuple(seen)


def build_task_brief(task: PublicTask, shape: TaskShape | None = None) -> TaskBrief:
    """Generate a brief from the task without invoking the LLM.

    All exceptions inside per-file probes are swallowed — the brief is
    advisory; never blocks task execution.
    """
    context_dir = task.context_dir
    file_paths = list(_walk_context(context_dir))

    summaries: list[FileSummary] = []
    has_knowledge_md = False
    truncated = False

    # Sort by size descending so the most relevant (often largest) files
    # show up first within the cap.
    file_paths.sort(key=lambda p: (-(p.stat().st_size if p.exists() else 0), str(p)))

    for path in file_paths:
        if path.name == "knowledge.md":
            has_knowledge_md = True
            continue
        if len(summaries) >= _BRIEF_MAX_FILES:
            truncated = True
            break
        try:
            size_bytes = path.stat().st_size
        except OSError:
            continue
        try:
            rel = str(path.relative_to(context_dir))
        except ValueError:
            rel = path.name
        bucket = _bucket_for(path)
        cols: tuple[str, ...] = ()
        rows: int | None = None
        db_tables: tuple[tuple[str, int | None], ...] = ()
        try:
            if bucket == "csv" and size_bytes < 50 * 1024 * 1024:
                cols, rows = _peek_csv(path)
            elif bucket == "db" and size_bytes < 200 * 1024 * 1024:
                db_tables = _peek_sqlite(path)
        except Exception:  # noqa: BLE001 — brief is advisory
            cols, rows, db_tables = (), None, ()
        summaries.append(
            FileSummary(
                rel_path=rel,
                bucket=bucket,
                size_bytes=size_bytes,
                columns=cols,
                row_estimate=rows,
                db_tables=db_tables,
            )
        )

    answer_kind = _classify_answer_kind(task.question or "", shape) if shape else "unknown"
    join_keys = _join_key_candidates(summaries, task.question or "")

    return TaskBrief(
        files=tuple(summaries),
        has_knowledge_md=has_knowledge_md,
        answer_kind_hint=answer_kind,
        join_key_candidates=join_keys,
        truncated=truncated,
    )


__all__ = ["TaskBrief", "FileSummary", "build_task_brief"]
