"""Official answer normalization for column-signature scoring.

The DataAgent-Bench scorer normalizes values BEFORE comparing column
signatures (sorted-value sets). The exact rules from
https://dataagent.top/rules:

- nulls (null / none / nan / empty) → empty string
- numerics → 2 decimal places
- dates → ISO 8601 (YYYY-MM-DD)
- strings → trimmed, case-sensitive

We mirror that here so we can (a) self-validate the agent's answer before
submitting, (b) score predictions locally with the same logic, and (c)
make `_answer` produce both raw and normalized variants and let the
scorer pick the per-task winner.

Design choices that may surprise you:

- Numbers always become 2dp strings (`3` → `"3.00"`). Gold values are
  presumably normalized the same way, so equality holds.
- Date detection is regex-tight, not dateutil-permissive. We do not want
  to accidentally promote `"q4-2024"` to a date. Add formats explicitly.
- "auto" policy is per-cell: a single column with mixed types still gets
  each cell normalized correctly. That matches the scorer's column-as-
  multiset semantics.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Iterable, Literal

from data_agent_baseline.benchmark.schema import AnswerTable

Policy = Literal["auto", "string", "numeric", "date", "passthrough"]

NULL_TOKENS = frozenset({
    "",
    "none",
    "null",
    "nan",
    "n/a",
    "na",
    "nil",
    "<na>",
    "(null)",
})

# Date formats tried in order. Each must produce an unambiguous YYYY-MM-DD.
_DATE_REGEXES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^(\d{4})-(\d{2})-(\d{2})$"), "ymd"),
    (re.compile(r"^(\d{4})/(\d{2})/(\d{2})$"), "ymd"),
    (re.compile(r"^(\d{4})\.(\d{2})\.(\d{2})$"), "ymd"),
    (re.compile(r"^(\d{2})-(\d{2})-(\d{4})$"), "dmy_or_mdy"),
    (re.compile(r"^(\d{2})/(\d{2})/(\d{4})$"), "dmy_or_mdy"),
)

_NUMERIC_REGEX = re.compile(r"^-?\d+(\.\d+)?([eE][+-]?\d+)?$")
_NUMERIC_WITH_THOUSANDS = re.compile(r"^-?\d{1,3}(,\d{3})+(\.\d+)?$")
_NUMERIC_PERCENT = re.compile(r"^-?\d+(\.\d+)?%$")
_NUMERIC_CURRENCY_PREFIX = ("$", "€", "£", "¥", "₩")


@dataclass(frozen=True, slots=True)
class NormalizationReport:
    column_policies: list[Policy]
    cell_changes: int


def is_null(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str):
        return value.strip().lower() in NULL_TOKENS
    return False


def _try_parse_number(text: str) -> float | None:
    s = text.strip()
    if _NUMERIC_PERCENT.match(s):
        return float(s.rstrip("%"))
    for prefix in _NUMERIC_CURRENCY_PREFIX:
        if s.startswith(prefix):
            return _try_parse_number(s[len(prefix):])
    if _NUMERIC_WITH_THOUSANDS.match(s):
        return float(s.replace(",", ""))
    if _NUMERIC_REGEX.match(s):
        return float(s)
    return None


def _format_number(value: float) -> str:
    if math.isnan(value) or math.isinf(value):
        return ""
    # Avoid -0.00 cosmetics.
    if value == 0:
        value = 0.0
    return f"{value:.2f}"


def _try_parse_date(value: Any) -> str | None:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if not isinstance(value, str):
        return None
    s = value.strip()
    for pattern, kind in _DATE_REGEXES:
        match = pattern.match(s)
        if not match:
            continue
        groups = match.groups()
        try:
            if kind == "ymd":
                return date(int(groups[0]), int(groups[1]), int(groups[2])).isoformat()
            # dmy_or_mdy: ambiguous unless we pick a convention. We assume
            # ISO-ordering preference: if first group ≤ 12 and second ≤ 12
            # the order is ambiguous — leave as string. Otherwise infer.
            a, b, c = int(groups[0]), int(groups[1]), int(groups[2])
            if a > 12 and b <= 12:
                return date(c, b, a).isoformat()  # dmy
            if a <= 12 and b > 12:
                return date(c, a, b).isoformat()  # mdy
            return None  # ambiguous
        except ValueError:
            return None
    return None


def normalize_value(value: Any, *, policy: Policy = "auto") -> str:
    if policy == "passthrough":
        return "" if value is None else str(value)
    if is_null(value):
        return ""
    if policy == "string":
        return str(value).strip() if isinstance(value, str) else str(value)
    if policy == "numeric":
        if isinstance(value, (int, float)):
            return _format_number(float(value))
        if isinstance(value, str):
            parsed = _try_parse_number(value)
            return _format_number(parsed) if parsed is not None else value.strip()
        return str(value)
    if policy == "date":
        formatted = _try_parse_date(value)
        if formatted is not None:
            return formatted
        return str(value).strip() if isinstance(value, str) else str(value)

    # auto
    if isinstance(value, bool):
        # Booleans are not numbers for our purposes.
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return _format_number(float(value))
    if isinstance(value, (date, datetime)):
        return _try_parse_date(value) or ""
    if isinstance(value, str):
        date_form = _try_parse_date(value)
        if date_form is not None:
            return date_form
        parsed_number = _try_parse_number(value)
        if parsed_number is not None:
            return _format_number(parsed_number)
        return value.strip()
    return str(value).strip()


def infer_column_policy(values: Iterable[Any], *, parse_threshold: float = 0.95) -> Policy:
    materialized = [v for v in values if not is_null(v)]
    if not materialized:
        return "string"
    total = len(materialized)
    date_hits = sum(1 for v in materialized if _try_parse_date(v) is not None)
    if date_hits / total >= parse_threshold:
        return "date"
    numeric_hits = 0
    for v in materialized:
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            numeric_hits += 1
        elif isinstance(v, str) and _try_parse_number(v) is not None:
            numeric_hits += 1
    if numeric_hits / total >= parse_threshold:
        return "numeric"
    return "string"


def normalize_column(values: list[Any], *, policy: Policy = "auto") -> list[str]:
    effective_policy: Policy
    if policy == "auto":
        effective_policy = infer_column_policy(values)
    else:
        effective_policy = policy
    return [normalize_value(v, policy=effective_policy) for v in values]


def normalize_answer_table(
    table: AnswerTable, *, policies: list[Policy] | None = None
) -> tuple[AnswerTable, NormalizationReport]:
    if not table.columns:
        return table, NormalizationReport(column_policies=[], cell_changes=0)

    column_count = len(table.columns)
    if policies is not None and len(policies) != column_count:
        raise ValueError("policies length must equal column count.")

    columns_data: list[list[Any]] = [[] for _ in range(column_count)]
    for row in table.rows:
        if len(row) != column_count:
            raise ValueError("row width does not match column count.")
        for index, value in enumerate(row):
            columns_data[index].append(value)

    if policies is None:
        column_policies = [infer_column_policy(col) for col in columns_data]
    else:
        column_policies = list(policies)

    normalized_columns = [
        normalize_column(columns_data[index], policy=column_policies[index])
        for index in range(column_count)
    ]

    normalized_rows: list[list[Any]] = []
    cell_changes = 0
    for row_index in range(len(table.rows)):
        new_row = [normalized_columns[c][row_index] for c in range(column_count)]
        original = [
            "" if is_null(value) else str(value).strip()
            for value in table.rows[row_index]
        ]
        cell_changes += sum(1 for a, b in zip(original, new_row) if a != b)
        normalized_rows.append(new_row)

    return (
        AnswerTable(columns=list(table.columns), rows=normalized_rows),
        NormalizationReport(column_policies=column_policies, cell_changes=cell_changes),
    )


def column_signature(values: Iterable[str]) -> frozenset[tuple[str, int]]:
    """Multiset signature: each unique value paired with its count.

    The rules say "ignores column names and row order" — so we hash by
    value-multiplicity rather than ordered list. Two columns are equal
    iff their normalized multisets are equal.
    """
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return frozenset(counts.items())
