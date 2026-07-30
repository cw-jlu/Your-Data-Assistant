"""Non-LLM rule-based verifier for answer-shape pre-checks.

Estimates `expected_column_count` from the question text using regex
heuristics only — NO LLM, NO gold/question lookup, NO public-task references.
Returns (estimate, confidence) where confidence ∈ {"high", "med", "low"}.
On low confidence, the verifier callsite MUST skip to keep the base
behavior intact (regression-prevention contract).
"""
from __future__ import annotations

import re
from typing import Literal

Confidence = Literal["high", "med", "low"]

# Patterns intentionally use only generic English NLP cues (quantifiers,
# aggregators, list-vs-scalar markers). NO public-task domain words
# (student_club / formula1 / financial / molecule / cards / patient /
#  schools / matches / posts) and NO 40+ char verbatim from any task.
_SINGLE_VALUE_PATTERNS = (
    re.compile(r"\bhow many\b", re.IGNORECASE),
    re.compile(r"\bwhat is the (?:total|count|number|sum|average|mean|max|min|maximum|minimum)\b", re.IGNORECASE),
    re.compile(r"\bcalculate the\b", re.IGNORECASE),
)

_LIST_NAME_PATTERNS = (
    re.compile(r"\blist (?:all |the )?names?\b", re.IGNORECASE),
    re.compile(r"\bnames? of\b", re.IGNORECASE),
    re.compile(r"\bwho (?:are|is)\b", re.IGNORECASE),
)

_PAIR_PATTERNS = (
    re.compile(r"\b(?:name|id|title) and (?:count|total|number|amount|score|value|year|date)\b", re.IGNORECASE),
    re.compile(r"\b(?:show|list|return) (?:the )?\w+ and (?:its|their|the) \w+\b", re.IGNORECASE),
)


def estimate_expected_columns(question: str) -> tuple[int | None, Confidence]:
    if not question or len(question) < 8:
        return (None, "low")
    q = question.strip()
    for pat in _SINGLE_VALUE_PATTERNS:
        if pat.search(q):
            return (1, "high")
    for pat in _PAIR_PATTERNS:
        if pat.search(q):
            return (2, "med")
    for pat in _LIST_NAME_PATTERNS:
        if pat.search(q):
            return (1, "med")
    return (None, "low")


def verify_answer_shape(question: str, columns: list[str]) -> tuple[bool, str | None]:
    """Returns (ok, hint_or_none).
    ok == True: confidence low (skip), or column count matches.
    ok == False: high/med-confidence mismatch — caller emits `hint` once.
    """
    n_pred = len(columns or [])
    expected, conf = estimate_expected_columns(question)
    if conf == "low" or expected is None:
        return (True, None)
    if n_pred == expected:
        return (True, None)
    if conf == "high":
        return (
            False,
            f"Verifier: question phrasing strongly implies a {expected}-column answer, "
            f"but the proposed answer has {n_pred} column(s). Reconsider which columns to include.",
        )
    return (
        False,
        f"Verifier note: question phrasing suggests around {expected} column(s); "
        f"proposed answer has {n_pred}. If you intended this, ignore.",
    )
