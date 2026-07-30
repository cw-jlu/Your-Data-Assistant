"""Deterministic preamble that dumps full context into the agent's first message.

All files are included at full fidelity up to a 150k-token budget. When the
budget is exceeded, sections are dropped in priority order:
  json(0) > csv(1) > doc(2) > sqlite(3) > knowledge_full(4)

Caps use a 4 chars/token rule of thumb (no tiktoken dependency).
(exp_095: + column candidate hint section after glossary)
"""
from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

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
# Column candidate hint (exp_095, [preamble:hint] axis)
# ============================================================================
# Heuristic: extract noun-like tokens from the question, match them against
# table column names from .db/.sqlite/.csv files in task.context_dir, and
# emit a small "Column hints" block. The hint is advisory — agents are
# explicitly told they may include other columns. This is the difference
# from exp_088 M-Schema (full catalog + Plan-first commitment) which fell
# into the M-Schema × T-diversity × union three-way trap. Here we only
# show top_n <= 7 columns the question likely needs.

_COL_HINT_TOP_N = 7
_COL_HINT_MAX_CHARS = 600

# Stop-words and trivially generic terms that should not match columns.
# Stdlib only — no external dictionaries / stemmers.
_QUESTION_STOPWORDS: frozenset[str] = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "of", "in",
    "on", "at", "to", "for", "and", "or", "but", "with", "by", "from",
    "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
    "many", "much", "most", "least", "highest", "lowest", "top", "bottom",
    "show", "list", "find", "give", "return", "calculate", "compute",
    "average", "sum", "total", "count", "number", "value", "values",
    "this", "that", "these", "those", "all", "any", "each", "some",
})

# Question-noun extractor: tokenize on word boundaries; keep tokens that are
# either CamelCase / ALL_CAPS / quoted, or 4+ char lowercase nouns that are
# not in stopwords. No POS tagger dependency.
_QUOTED_TOKEN_RE = re.compile(r"['\"`]([A-Za-z][A-Za-z0-9_ ]{1,40})['\"`]")
_WORD_TOKEN_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9_]{2,}\b")


def _extract_question_nouns(question: str) -> list[str]:
    """Return a deduplicated list of candidate nouns from the question.
    Stdlib regex only; no NLP dependency. Lowercased for matching.
    """
    if not question:
        return []
    found: list[str] = []
    seen: set[str] = set()
    for m in _QUOTED_TOKEN_RE.finditer(question):
        for sub in re.findall(r"[A-Za-z][A-Za-z0-9_]{2,}", m.group(1)):
            key = sub.lower()
            if key in seen or key in _QUESTION_STOPWORDS:
                continue
            seen.add(key)
            found.append(key)
    for m in _WORD_TOKEN_RE.finditer(question):
        token = m.group(0)
        key = token.lower()
        if key in seen or key in _QUESTION_STOPWORDS or len(key) < 4:
            continue
        seen.add(key)
        found.append(key)
    return found


def _enumerate_schema_columns(task: PublicTask) -> list[tuple[str, str]]:
    """Return list of (table_label, column_name) pairs from .db/.sqlite/.csv
    files inside task.context_dir. Errors are swallowed — hint is advisory.
    """
    pairs: list[tuple[str, str]] = []
    for path in sorted(task.context_dir.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in {".db", ".sqlite", ".sqlite3"}:
            try:
                conn = sqlite3.connect(path)
                try:
                    cur = conn.cursor()
                    tables = [
                        r[0]
                        for r in cur.execute(
                            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
                        ).fetchall()
                    ]
                    for table in tables:
                        try:
                            for r in cur.execute(f'PRAGMA table_info("{table}")').fetchall():
                                pairs.append((f"{path.name}:{table}", r[1]))
                        except Exception:
                            continue
                finally:
                    conn.close()
            except Exception:
                continue
        elif suffix == ".csv":
            try:
                import pandas as pd
                df = pd.read_csv(path, nrows=1)
                for col in df.columns:
                    pairs.append((path.name, str(col)))
            except Exception:
                continue
    return pairs


def _score_column_match(question_nouns: list[str], col: str) -> float:
    """Score a column name vs question nouns. Stdlib only.
    Returns 0.0..1.0; columns scoring 0.0 are dropped before sorting.
    """
    if not col:
        return 0.0
    col_low = col.lower()
    col_tokens = set(re.findall(r"[a-z][a-z0-9]+", col_low))
    best = 0.0
    for noun in question_nouns:
        if noun == col_low:
            best = max(best, 1.0)
        elif noun in col_tokens:
            best = max(best, 0.7)
        elif noun in col_low or col_low in noun:
            best = max(best, 0.5)
        else:
            if len(noun) >= 4 and len(col_low) >= 4 and noun[:4] == col_low[:4]:
                best = max(best, 0.3)
    return best


def _build_column_candidate_hint(
    task: PublicTask, *, top_n: int = _COL_HINT_TOP_N
) -> str | None:
    """Return a small advisory hint block, or None if no candidates.
    Falls back to None on any error to preserve base behavior.
    """
    try:
        nouns = _extract_question_nouns(task.question)
        if not nouns:
            return None
        pairs = _enumerate_schema_columns(task)
        if not pairs:
            return None
        scored: list[tuple[float, str, str]] = []
        for table_label, col in pairs:
            s = _score_column_match(nouns, col)
            if s > 0.0:
                scored.append((s, table_label, col))
        if not scored:
            return None
        scored.sort(key=lambda x: (-x[0], x[1], x[2]))
        kept = scored[:top_n]
        lines = [
            "# Column hints (heuristic, may be incomplete; not authoritative)",
            (
                "Below are columns that the question phrasing most likely refers to. "
                "These are recommendations, NOT a constraint — include other columns "
                "if your analysis requires them, and ignore any hint that does not fit."
            ),
        ]
        for _, table_label, col in kept:
            lines.append(f"- {table_label}.{col}")
        section = "\n".join(lines)
        if len(section) > _COL_HINT_MAX_CHARS:
            section = section[: _COL_HINT_MAX_CHARS - len(_TRUNCATED_MARKER)] + _TRUNCATED_MARKER
        return section
    except Exception:
        return None


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

    # exp_095 [preamble:hint]: advisory column candidates derived from the
    # question text, kept small (top_n <= 7) and explicitly non-binding to
    # avoid the exp_088 M-Schema × Plan-first × union three-way trap.
    column_hint = _build_column_candidate_hint(task)
    if column_hint is not None:
        fixed_parts.append(column_hint)

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
