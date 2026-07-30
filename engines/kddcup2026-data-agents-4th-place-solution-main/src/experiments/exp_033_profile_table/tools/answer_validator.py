"""Diagnostic self-correction validator for the answer tool.

Inspects the submitted answer against the spec written in the agent's first
thought.  Up to ANSWER_SELF_CORRECT_MAX_RETRIES failed attempts are allowed
before the last answer is soft-accepted so the agent always terminates.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

ANSWER_SELF_CORRECT_MAX_RETRIES = 3

# Regex to count answer_columns entries: each entry has a "name:" key.
_ANSWER_COL_RE = re.compile(r"\bname\s*:", re.IGNORECASE)

# Matches: expected_row_count: <token>
_EXPECTED_ROW_RE = re.compile(
    r"expected_row_count\s*:\s*([^\n]+)", re.IGNORECASE
)


def _parse_expected_column_count(spec_thought: str) -> int | None:
    """Return the number of answer_columns entries, or None if unparseable."""
    # Look for the answer_columns block.
    block_match = re.search(
        r"answer_columns\s*:\s*\[([^\]]*)\]", spec_thought, re.IGNORECASE | re.DOTALL
    )
    if not block_match:
        return None
    count = len(_ANSWER_COL_RE.findall(block_match.group(1)))
    return count if count > 0 else None


def _parse_expected_row_count(spec_thought: str) -> tuple[str, int] | None:
    """Return (mode, value) for expected_row_count, or None.

    Modes:
      "exact" — exactly N rows expected
      "gte"   — >= N rows expected (from "1+" or ">0")
      "gt"    — > N rows expected (from ">N")
    """
    m = _EXPECTED_ROW_RE.search(spec_thought)
    if not m:
        return None
    raw = m.group(1).strip().strip('"').strip("'")
    if raw.endswith("+"):
        try:
            n = int(raw[:-1])
            return ("gte", n)
        except ValueError:
            return None
    if raw.startswith(">"):
        try:
            n = int(raw[1:].strip())
            return ("gt", n)
        except ValueError:
            return None
    try:
        return ("exact", int(raw))
    except ValueError:
        return None


def _row_count_ok(expected: tuple[str, int], actual: int) -> bool:
    mode, n = expected
    if mode == "exact":
        return actual == n
    if mode == "gte":
        return actual >= n
    # mode == "gt"
    return actual > n


@dataclass(slots=True)
class AnswerValidatorState:
    attempt_count: int = 0
    last_spec_thought: str = ""

    def record_spec(self, thought: str) -> None:
        """Called by the registry on every model thought.

        Only the first thought containing a ## Spec block is stored.
        """
        if self.last_spec_thought:
            return
        if "## Spec" in thought or "answer_columns" in thought:
            self.last_spec_thought = thought


def validate_answer(
    *,
    state: AnswerValidatorState,
    columns: list[str],
    rows: list[list[Any]],
) -> tuple[bool, bool, str]:
    """Validate a proposed answer against the recorded spec.

    Returns:
        (ok, is_terminal, hint)
        - ok=True, is_terminal=True  → accept and terminate
        - ok=False, is_terminal=False → reject with hint (agent retries)
    """
    state.attempt_count += 1

    # Soft accept after max retries to guarantee termination.
    if state.attempt_count > ANSWER_SELF_CORRECT_MAX_RETRIES:
        return (True, True, "")

    # No spec recorded — skip validation.
    if not state.last_spec_thought:
        return (True, True, "")

    actual_col_count = len(columns)
    actual_row_count = len(rows)

    expected_col_count = _parse_expected_column_count(state.last_spec_thought)
    expected_row = _parse_expected_row_count(state.last_spec_thought)

    # Detect spec_wrong_likely: empty rows when spec says >=1 row, or column mismatch.
    if expected_col_count is not None and actual_col_count != expected_col_count:
        return (
            False,
            False,
            (
                f"spec_wrong_likely: answer has {actual_col_count} column(s) "
                f"but spec says {expected_col_count}. Re-explore and rewrite your spec."
            ),
        )

    if actual_row_count == 0 and expected_row is not None:
        mode, n = expected_row
        if mode in ("gte", "gt") or (mode == "exact" and n > 0):
            return (
                False,
                False,
                "spec_wrong_likely: answer has 0 rows but spec expects rows. "
                "Re-explore and rewrite your spec.",
            )

    # Detect row_count_mismatch (exact expectation only — loose bounds are advisory).
    if expected_row is not None and not _row_count_ok(expected_row, actual_row_count):
        return (
            False,
            False,
            (
                f"row_count_mismatch: answer has {actual_row_count} row(s) "
                f"but spec says expected_row_count={state.last_spec_thought}. "
                "Verify your query and resubmit."
            ),
        )

    return (True, True, "")
