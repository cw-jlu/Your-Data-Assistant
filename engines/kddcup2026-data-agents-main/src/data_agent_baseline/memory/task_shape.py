"""Classify a ``PublicTask`` into a ``TaskShape`` — a compact, hashable
description of the kind of task we're looking at, used to look up
:class:`~data_agent_baseline.memory.policies.ShapePolicy`.

Why this layer exists
---------------------

We can't tune per-task at eval time (hidden tasks are different from public
tasks). What we *can* do is recognise that a hidden task is **shaped like**
a public task we've already studied. The shape captures the dimensions that
empirically drive failure / success:

- difficulty tier
- total context size + largest single file size
- which file types are present (csv / json / db / doc / pdf)
- "is this a heavy task" — the single boolean we use to gate timeout +
  max_steps multipliers
- question keywords (plural, aggregate, ranking, filter)

The classifier does NOT call the LLM. It is pure metadata + filesystem
introspection so it can run inside the eval container without budget cost.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from data_agent_baseline.benchmark.schema import PublicTask


# Size thresholds in MB. Tuned to public-set distribution: anything bigger
# than these starts to stress 32K context window or 30s execute_python budget.
_LARGE_JSON_MB: float = 50.0
_LARGE_DB_MB: float = 100.0
_LARGE_CSV_MB: float = 50.0
_HEAVY_TOTAL_MB: float = 100.0
_HEAVY_LARGEST_MB: float = 50.0

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
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".gif": "image",
    ".webp": "image",
    ".zip": "archive",
    ".tar": "archive",
    ".gz": "archive",
}


# Question-keyword detection. Ordered so we never double-count: aggregate
# keywords are checked before plural so "average list" still flags aggregate
# (which is the higher-signal pattern).
_AGGREGATE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\baverage\b", re.IGNORECASE),
    re.compile(r"\bavg\b", re.IGNORECASE),
    re.compile(r"\bmean\b", re.IGNORECASE),
    re.compile(r"\b(?:total|sum|sum of)\b", re.IGNORECASE),
    re.compile(r"\b(?:count|how many|number of)\b", re.IGNORECASE),
    re.compile(r"\bmedian\b", re.IGNORECASE),
    re.compile(r"\b(?:max|maximum|highest|largest|greatest)\b", re.IGNORECASE),
    re.compile(r"\b(?:min|minimum|lowest|smallest|fewest)\b", re.IGNORECASE),
    re.compile(r"\bratio\b", re.IGNORECASE),
    re.compile(r"\bpercentage\b", re.IGNORECASE),
)

_PLURAL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(?:list|all|every|each|which|who are|name the)\b", re.IGNORECASE),
    re.compile(r"\b(?:names|ids|titles)\b", re.IGNORECASE),
)

_RANKING_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(?:top|bottom|first|second|third|last)\b", re.IGNORECASE),
    re.compile(r"\b(?:rank|ranked|ranking)\b", re.IGNORECASE),
    re.compile(r"\b(?:fastest|slowest|cheapest|most|least)\b", re.IGNORECASE),
)

_SINGULAR_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*what is\b", re.IGNORECASE),
    re.compile(r"^\s*how (?:many|much)\b", re.IGNORECASE),
    re.compile(r"\bthe (?:largest|smallest|maximum|minimum|average|total|sum|count|number|name|id|title)\b", re.IGNORECASE),
)


@dataclass(frozen=True, slots=True)
class TaskShape:
    """Hashable shape descriptor used as a policy lookup key."""

    difficulty: str
    total_context_mb: float
    largest_file_mb: float
    file_types: frozenset[str]
    has_large_json: bool
    has_large_db: bool
    has_large_csv: bool
    is_heavy: bool
    is_aggregate_question: bool
    is_plural_question: bool
    is_ranking_question: bool
    is_singular_question: bool
    question_keywords: frozenset[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "difficulty": self.difficulty,
            "total_context_mb": round(self.total_context_mb, 2),
            "largest_file_mb": round(self.largest_file_mb, 2),
            "file_types": sorted(self.file_types),
            "has_large_json": self.has_large_json,
            "has_large_db": self.has_large_db,
            "has_large_csv": self.has_large_csv,
            "is_heavy": self.is_heavy,
            "is_aggregate_question": self.is_aggregate_question,
            "is_plural_question": self.is_plural_question,
            "is_ranking_question": self.is_ranking_question,
            "is_singular_question": self.is_singular_question,
            "question_keywords": sorted(self.question_keywords),
        }


def _file_type_bucket(suffix: str) -> str | None:
    return _FILE_TYPE_BUCKETS.get(suffix.lower())


def _walk_context_files(context_dir: Path) -> Iterable[Path]:
    if not context_dir.is_dir():
        return ()
    return (p for p in context_dir.rglob("*") if p.is_file())


def _scan_question(question: str) -> tuple[bool, bool, bool, bool, frozenset[str]]:
    """Return (aggregate, plural, ranking, singular, matched_keywords).

    ``matched_keywords`` is just the lowercased words from the question that
    triggered any pattern — kept for diagnostics, not used for policy
    matching (the four booleans carry the routing signal).
    """
    is_aggregate = any(p.search(question) for p in _AGGREGATE_PATTERNS)
    is_plural = any(p.search(question) for p in _PLURAL_PATTERNS)
    is_ranking = any(p.search(question) for p in _RANKING_PATTERNS)
    is_singular = any(p.search(question) for p in _SINGULAR_PATTERNS)

    keywords: set[str] = set()
    for pattern_list, label in (
        (_AGGREGATE_PATTERNS, "aggregate"),
        (_PLURAL_PATTERNS, "plural"),
        (_RANKING_PATTERNS, "ranking"),
        (_SINGULAR_PATTERNS, "singular"),
    ):
        if any(p.search(question) for p in pattern_list):
            keywords.add(label)
    return is_aggregate, is_plural, is_ranking, is_singular, frozenset(keywords)


def classify_task(task: PublicTask) -> TaskShape:
    """Classify a task without invoking the LLM. Pure I/O on the context tree."""
    total_bytes = 0
    largest_bytes = 0
    largest_json_bytes = 0
    largest_db_bytes = 0
    largest_csv_bytes = 0
    file_types: set[str] = set()

    for file_path in _walk_context_files(task.context_dir):
        try:
            size = file_path.stat().st_size
        except OSError:
            continue
        total_bytes += size
        if size > largest_bytes:
            largest_bytes = size
        bucket = _file_type_bucket(file_path.suffix)
        if bucket is None:
            continue
        file_types.add(bucket)
        if bucket == "json" and size > largest_json_bytes:
            largest_json_bytes = size
        elif bucket == "db" and size > largest_db_bytes:
            largest_db_bytes = size
        elif bucket == "csv" and size > largest_csv_bytes:
            largest_csv_bytes = size

    total_mb = total_bytes / (1024 * 1024)
    largest_mb = largest_bytes / (1024 * 1024)
    largest_json_mb = largest_json_bytes / (1024 * 1024)
    largest_db_mb = largest_db_bytes / (1024 * 1024)
    largest_csv_mb = largest_csv_bytes / (1024 * 1024)

    has_large_json = largest_json_mb >= _LARGE_JSON_MB
    has_large_db = largest_db_mb >= _LARGE_DB_MB
    has_large_csv = largest_csv_mb >= _LARGE_CSV_MB

    difficulty = (task.difficulty or "").lower()
    is_heavy = (
        difficulty == "extreme"
        or (difficulty == "hard" and total_mb >= _HEAVY_TOTAL_MB)
        or largest_mb >= _HEAVY_LARGEST_MB
    )

    is_aggregate, is_plural, is_ranking, is_singular, keywords = _scan_question(
        task.question or ""
    )

    return TaskShape(
        difficulty=difficulty,
        total_context_mb=total_mb,
        largest_file_mb=largest_mb,
        file_types=frozenset(file_types),
        has_large_json=has_large_json,
        has_large_db=has_large_db,
        has_large_csv=has_large_csv,
        is_heavy=is_heavy,
        is_aggregate_question=is_aggregate,
        is_plural_question=is_plural,
        is_ranking_question=is_ranking,
        is_singular_question=is_singular,
        question_keywords=keywords,
    )
