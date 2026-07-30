"""Output-direct validator for computed answer tools.

Validates the rows/columns returned by answer_from_python / answer_from_sql /
answer_from_duckdb without requiring any spec thought from the agent.
Up to ANSWER_VALIDATE_MAX_RETRIES failed attempts are allowed before
soft-accepting so the agent always terminates.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

ANSWER_VALIDATE_MAX_RETRIES = 3


@dataclass(slots=True)
class AnswerValidatorState:
    attempt_count: int = 0


def validate_answer_output(
    *,
    state: AnswerValidatorState,
    columns: list[str],
    rows: list[list[Any]],
) -> tuple[bool, bool, str]:
    """Validate computed answer output directly.

    Returns:
        (ok, is_terminal, hint)
        - ok=True, is_terminal=True  → accept and terminate
        - ok=False, is_terminal=False → reject with hint (agent retries)
    """
    state.attempt_count += 1

    # Soft accept after max retries to guarantee termination.
    if state.attempt_count > ANSWER_VALIDATE_MAX_RETRIES:
        return (True, True, "")

    # Check 1: empty result
    if not rows:
        return (
            False,
            False,
            "Empty result (0 rows). Check WHERE clause / JOIN keys / data range.",
        )

    # Check 2: all cells null/empty
    flat = [str(cell) for row in rows for cell in row]
    null_tokens = {"", "none", "null", "nan", "<na>", "nat"}
    if all(v.lower() in null_tokens for v in flat):
        return (
            False,
            False,
            "All cells are null/empty. The query likely missed the actual values.",
        )

    # Check 3: suspicious uniform output (>=3 rows, all values in a column identical)
    if len(rows) >= 3:
        for col_idx in range(len(columns)):
            vals = {str(row[col_idx]) for row in rows if col_idx < len(row)}
            if len(vals) == 1:
                return (
                    False,
                    False,
                    "Suspicious: every row has the same value in a column. "
                    "Verify GROUP BY / DISTINCT logic.",
                )

    # Check 4: error tokens in likely-numeric columns
    error_tokens = {"error", "none", "<na>", "nan", "null"}
    numeric_hints = {"count", "sum", "total", "amount", "price", "num", "id"}
    numeric_col_indices = [
        i for i, c in enumerate(columns)
        if any(kw in c.lower() for kw in numeric_hints)
    ]
    for col_idx in numeric_col_indices:
        for row in rows[:5]:
            if col_idx < len(row) and str(row[col_idx]).lower() in error_tokens:
                return (
                    False,
                    False,
                    "Result contains error/null tokens in a likely-numeric column. "
                    "Check NULL handling and type casts.",
                )

    # Check 5: way too many rows (likely missing aggregation)
    if len(rows) > 1000:
        return (
            False,
            False,
            f"Way too many rows ({len(rows)}). Likely missing aggregation "
            "(COUNT/SUM/GROUP BY). Rewrite the query.",
        )

    return (True, True, "")
