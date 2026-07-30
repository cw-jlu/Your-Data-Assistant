"""Answer-quality validator — pre-terminal sanity check on `_answer` payload.

The official scorer matches each predicted column against gold by
value-multiset signature after normalization. Common failure modes we
saw in v2 holdout:

- task_11 (easy)  : 3×6 vs gold 3×3 — over-emission of rows
- task_355 (hard) : 2×1 vs gold 3×1 — under-emission of columns
- task_408 (hard) : 1×1 shape match but value mismatch

This validator catches the shape-level patterns heuristically before
the answer terminal commits, so the conditional `_answer` can return
warnings as observation and let the agent self-correct. Severity:

- ``error``   → must fix (e.g. empty rows / empty columns)
- ``warning`` → should fix (e.g. all-null column)
- ``info``    → consider, may be noise (e.g. singular question + multi-row)

The agent can override blocking warnings by re-calling answer with
``confirm=true``. The validator never edits the answer; it only emits
warnings.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Literal

from data_agent_baseline.benchmark.schema import AnswerTable
from data_agent_baseline.scoring.normalize import is_null


Severity = Literal["info", "warning", "error"]


@dataclass(frozen=True, slots=True)
class ValidationWarning:
    severity: Severity
    code: str
    message: str
    suggestion: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "suggestion": self.suggestion,
        }


@dataclass(frozen=True, slots=True)
class ValidationReport:
    warnings: list[ValidationWarning]

    @property
    def has_blocking(self) -> bool:
        return any(w.severity in ("warning", "error") for w in self.warnings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "warnings": [w.to_dict() for w in self.warnings],
            "has_blocking": self.has_blocking,
            "warning_count": len(self.warnings),
        }


# Enumeration patterns: "list their A, B, and C" / "show the A and B"
# Captures the noun-list portion in group(1).
_ENUM_PATTERNS: tuple[re.Pattern[str], ...] = (
    # comma-separated list ending in "and N"
    re.compile(
        r"\b(?:list|show|give|return|find|name|tell|what\s+(?:are|is))\s+"
        r"(?:me\s+)?(?:their|the)?\s*"
        r"([a-zA-Z][a-zA-Z0-9_ ]*(?:,\s*[a-zA-Z][a-zA-Z0-9_ ]*)+\s+and\s+[a-zA-Z][a-zA-Z0-9_ ]*)\b",
        re.IGNORECASE,
    ),
    # simple "A and B"
    re.compile(
        r"\b(?:list|show|give|return|find|name|tell|what\s+(?:are|is))\s+"
        r"(?:me\s+)?(?:their|the)?\s*"
        r"([a-zA-Z][a-zA-Z0-9_ ]+\s+and\s+[a-zA-Z][a-zA-Z0-9_ ]+)\b",
        re.IGNORECASE,
    ),
)

# Singular answer cues — "what is the X", "how many", "the largest", etc.
_SINGULAR_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bwhat\s+is\s+the\b", re.IGNORECASE),
    re.compile(r"\bhow\s+(?:many|much)\b", re.IGNORECASE),
    re.compile(r"\b(?:total|maximum|minimum|sum|average|mean|median|count)\b", re.IGNORECASE),
    re.compile(
        r"\bthe\s+(?:largest|smallest|highest|lowest|best|worst|first|last|biggest|tallest|oldest|youngest)\b",
        re.IGNORECASE,
    ),
)

_TRAILING_NOUN = re.compile(r"^[\s,]+|[\s,]+$")


def _split_enumeration(phrase: str) -> list[str]:
    """Split 'A, B, and C' into ['A', 'B', 'C']."""
    parts = re.split(r",\s*|\s+and\s+", phrase, flags=re.IGNORECASE)
    return [_TRAILING_NOUN.sub("", p).strip() for p in parts if p.strip()]


def _count_enumerated_nouns(question: str) -> int | None:
    """Return enumerated noun count if the question matches a clear pattern, else None.

    Conservative: only matches phrases where the enumeration is obviously
    bounded by the verb cue and the final 'and N'. Free-form questions
    fall through to None to avoid false-positive column-count warnings.
    """
    for pat in _ENUM_PATTERNS:
        m = pat.search(question)
        if not m:
            continue
        nouns = _split_enumeration(m.group(1))
        if 2 <= len(nouns) <= 6:
            return len(nouns)
    return None


def _is_singular_question(question: str) -> bool:
    return any(p.search(question) for p in _SINGULAR_PATTERNS)


# Plural cues that suggest multiple rows. v4_final task_22 (Connor Hilton's
# dues — gold 2 rows, our pred 1 row) is the canonical case: LLM saw the
# read_csv preview's first matching row and committed an under-emission.
_PLURAL_CUE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(?:list\s+all|list\s+the|list\s+their|list\s+every)\b", re.IGNORECASE),
    re.compile(r"\b(?:all\s+the|every|each|which\s+(?:are|of)|who\s+are)\b", re.IGNORECASE),
    re.compile(r"\b(?:what\s+are|find\s+all|return\s+all|name\s+the)\b", re.IGNORECASE),
)

# Singular cues that override plural: when the question pins a unique
# entity, a single-row answer is correct (e.g. "the customer with id 3356").
_UNIQUE_OVERRIDE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bwith\s+(?:the\s+)?id\s+\S+", re.IGNORECASE),
    re.compile(r"\bthe\s+only\b", re.IGNORECASE),
    re.compile(r"\bwhose\s+(?:id|name|key)\s+is\b", re.IGNORECASE),
)


def _is_plural_question(question: str) -> bool:
    if not question:
        return False
    if any(p.search(question) for p in _UNIQUE_OVERRIDE_PATTERNS):
        return False
    return any(p.search(question) for p in _PLURAL_CUE_PATTERNS)


_AGGREGATE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(?:how\s+many|how\s+much|number\s+of|count|total|sum|aggregate|"
        r"average|mean|median|maximum|minimum)\b",
        re.IGNORECASE,
    ),
)


def _looks_like_aggregate_question(question: str) -> bool:
    return any(p.search(question) for p in _AGGREGATE_PATTERNS)


_ISO_DATE_RE = re.compile(r"^(\d{4})-\d{2}-\d{2}")


def _extract_iso_years(values: list[str]) -> list[int]:
    years: list[int] = []
    for raw in values:
        match = _ISO_DATE_RE.match(raw.strip())
        if match is None:
            return []
        years.append(int(match.group(1)))
    return years


def _is_blank_cell(value: Any) -> bool:
    """Treat '' / None / NaN / null-tokens as blank for all-null detection."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return is_null(value)


def validate_answer(
    answer: AnswerTable,
    *,
    normalized: AnswerTable | None = None,
    question: str = "",
) -> ValidationReport:
    """Run shape + heuristic checks. Returns a report; emits no side effects."""
    cols = list(answer.columns)
    rows = list(answer.rows)
    n_cols = len(cols)
    n_rows = len(rows)
    norm_rows = list(normalized.rows) if normalized is not None else rows

    warnings: list[ValidationWarning] = []

    # Errors first — short-circuit for hopeless cases
    if n_cols == 0:
        warnings.append(
            ValidationWarning(
                severity="error",
                code="empty_columns",
                message="Answer has no columns.",
                suggestion="Construct an answer table with at least one column.",
            )
        )
    if n_rows == 0:
        # v8: 0-row is a valid answer for true-empty filters. Demoted to info
        # so the conditional-terminal `_answer` no longer soft-rejects it.
        # Other validators below (e.g. column-count mismatch) still run.
        warnings.append(
            ValidationWarning(
                severity="info",
                code="empty_rows",
                message="Answer has no rows.",
                suggestion=(
                    "If the filter genuinely matches nothing in the source, this is the "
                    "correct answer — call `answer` again with `confirm: true` to commit. "
                    "Otherwise re-derive the query."
                ),
            )
        )
    if n_cols == 0:
        return ValidationReport(warnings=warnings)

    # All-null columns
    for col_idx, col_name in enumerate(cols):
        column_values: list[Any] = []
        for row in norm_rows:
            try:
                column_values.append(row[col_idx])
            except (IndexError, TypeError):
                continue
        if column_values and all(_is_blank_cell(v) for v in column_values):
            warnings.append(
                ValidationWarning(
                    severity="warning",
                    code="all_null_column",
                    message=f"Column {col_name!r} is entirely null/empty after normalization.",
                    suggestion=(
                        "Verify the column is needed for the answer; remove it or populate "
                        "with concrete values."
                    ),
                )
            )

    # Duplicate rows (signal of un-deduplicated query results)
    seen: set[tuple[str, ...]] = set()
    dup_count = 0
    for row in norm_rows:
        try:
            key = tuple(str(c) for c in row)
        except TypeError:
            continue
        if key in seen:
            dup_count += 1
        else:
            seen.add(key)
    if dup_count > 0:
        # v7: escalate to blocking when the question is plural AND duplicate
        # ratio is ≥50% — that combo is the v6 over-emission failure mode
        # (task_180: 153 rows where gold is 9, task_25: 28 rows where gold is 3).
        # Otherwise stay info-only so legitimate plural answers with a few
        # repeats are not blocked.
        dup_ratio = dup_count / max(n_rows, 1)
        is_plural = bool(question and _is_plural_question(question))
        severe = is_plural and dup_ratio >= 0.5
        warnings.append(
            ValidationWarning(
                severity="warning" if severe else "info",
                code="duplicate_rows",
                message=(
                    f"{dup_count} duplicate row(s) detected after normalization "
                    f"(ratio {dup_ratio:.0%})."
                ),
                suggestion=(
                    "Plural question with ≥50% duplicate rows — almost certainly missing "
                    "DISTINCT or a wrong join multiplier. Re-derive with DISTINCT applied."
                    if severe
                    else "The column-signature scorer treats duplicates literally; consider "
                    "applying DISTINCT or dedup if the question expects unique entries."
                ),
            )
        )

    # F-4: Plural-cue question with single-row answer — likely under-emission.
    # Info-only (not blocking) because false positives are easy: the LLM might
    # have legitimately filtered down to one matching entity. Logging it gives
    # the agent a chance to reconsider when reading observation feedback.
    if question and _is_plural_question(question) and n_rows == 1:
        warnings.append(
            ValidationWarning(
                severity="info",
                code="plural_question_singular_row",
                message=(
                    "Question uses plural cues (list/all/which/etc.) but the answer has only 1 row. "
                    "Verify this is intentional — under-emission was a v4_final failure mode (e.g. "
                    "task_22 emitted 1 row when gold had 2)."
                ),
                suggestion=(
                    "Re-check the source data: are there other rows that match the question's filter? "
                    "If so, include them. If a single row is genuinely correct, ignore this warning."
                ),
            )
        )

    # Question shape vs row count
    if question and _is_singular_question(question) and n_rows > 1:
        warnings.append(
            ValidationWarning(
                severity="info",
                code="singular_question_multi_row",
                message=(
                    f"Question pattern looks singular but answer has {n_rows} rows. "
                    "Confirm whether multiple rows are intended."
                ),
                suggestion="Aggregate / filter if a single value is expected.",
            )
        )

    # Question enumeration vs column count.
    # v4 50-task forensics: 4/12 zero-with-prediction cases were severe
    # over-emission (e.g. task_259 emitted 7 cols vs 1 expected). When the
    # ratio exceeds 1.5×, escalate to a blocking warning so the
    # conditional-terminal soft-rejects the answer and the agent gets a
    # chance to drop auxiliary columns. Mild mismatches stay info-only to
    # avoid false-positive blocks on legitimate near-matches.
    if question:
        expected = _count_enumerated_nouns(question)
        if expected is not None and expected != n_cols:
            ratio = (n_cols / expected) if expected > 0 else float("inf")
            severity_ratio: Severity = "warning" if ratio >= 1.5 else "info"
            warnings.append(
                ValidationWarning(
                    severity=severity_ratio,
                    code="column_count_mismatch",
                    message=(
                        f"Question enumerates {expected} attribute(s) but the answer has "
                        f"{n_cols} column(s) (ratio {ratio:.2f}×)."
                    ),
                    suggestion=(
                        "Verify each requested attribute is present as a separate column. "
                        "Severe over-emission was the dominant zero-score failure mode in v4 "
                        "50-task analysis (task_25, task_259, etc.)."
                    ),
                )
            )

    # §3.5 Content-level domain sanity — deterministic heuristics only
    # (single-model policy: no LLM-as-judge call). Each branch enforces
    # its own minimum-count guard so single-row aggregates and
    # multi-row constant columns are both caught.
    is_aggregate_q = bool(question and _looks_like_aggregate_question(question))
    for col_idx, col_name in enumerate(cols):
        column_values: list[str] = []
        for row in norm_rows:
            try:
                value = str(row[col_idx])
            except (IndexError, TypeError):
                continue
            if not _is_blank_cell(value):
                column_values.append(value)
        if not column_values:
            continue

        # Try parsing all non-null values as numeric. A single bad value
        # invalidates the numeric branch.
        numeric_values: list[float] = []
        for raw in column_values:
            try:
                numeric_values.append(float(raw))
            except (ValueError, TypeError):
                numeric_values = []
                break

        # Constant numeric column — needs at least two values to mean
        # anything (single-row constancy is trivial).
        if len(numeric_values) >= 2 and len(set(numeric_values)) == 1:
            warnings.append(
                ValidationWarning(
                    severity="info",
                    code="constant_numeric_column",
                    message=(
                        f"Column {col_name!r} has {len(numeric_values)} numeric values "
                        f"all equal to {numeric_values[0]}."
                    ),
                    suggestion=(
                        "Re-check the aggregation / filter — usually means GROUP BY "
                        "was forgotten or the join key removed every distinct case."
                    ),
                )
            )

        # Aggregate-question single-row answer that came back zero or
        # negative (count/sum/total semantics).
        if is_aggregate_q and len(numeric_values) == 1 and n_rows == 1:
            value = numeric_values[0]
            if value <= 0:
                warnings.append(
                    ValidationWarning(
                        severity="info",
                        code="aggregate_zero_or_negative",
                        message=(
                            f"Question looks like a count/sum/total but column {col_name!r} "
                            f"= {value}. A zero or negative aggregate often signals an empty "
                            "filter result."
                        ),
                        suggestion=(
                            "Verify the WHERE clause selectivity and confirm the join "
                            "produces non-empty rows before aggregation."
                        ),
                    )
                )

        # Year/date column with extreme outliers — placeholder sentinels
        # like 1900-01-01 or 9999-12-31. Need at least 2 values so a
        # single legit historical date doesn't trigger.
        date_years = _extract_iso_years(column_values)
        if len(date_years) >= 2:
            extreme = [y for y in date_years if y < 1950 or y > 2099]
            if extreme and len(extreme) == len(date_years):
                warnings.append(
                    ValidationWarning(
                        severity="info",
                        code="all_extreme_dates",
                        message=(
                            f"Column {col_name!r} has all {len(date_years)} dates outside "
                            "1950–2099 — likely sentinel/placeholder values."
                        ),
                        suggestion=(
                            "Filter rows where the date equals the source's null sentinel "
                            "(commonly 1900-01-01 or 9999-12-31)."
                        ),
                    )
                )

    return ValidationReport(warnings=warnings)
