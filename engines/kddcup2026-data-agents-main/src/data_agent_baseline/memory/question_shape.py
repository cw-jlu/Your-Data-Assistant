"""v5 S-3: Question-pattern → expected answer-shape inference.

Pure rule-based, single-model policy (no embeddings, no aux LLM). Derived
from forensic clustering of the public 50-task (question, gold.csv) pairs —
see ``scripts/derive_question_shape_policies.py`` for the source of the
heuristics.

Caller is ``runner._build_task_advisories``: the hint string is appended to
the task's policy.prompt_hints stack so the agent sees it BEFORE composing
its plan. The agent is still free to override the hint — this is advisory,
not a hard constraint (Tier-3 of the plan introduces a hard constraint via
Stage A/B; that path is deferred).

The classifier is deliberately conservative: when the question doesn't
match a known pattern, ``infer`` returns ``None`` and no hint is injected.
False positives are worse than misses here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


_AGGREGATION_VERBS = ("count", "sum", "average", "avg", "mean", "median", "total", "calculate")
_LIST_VERBS = ("list", "name", "identify", "give", "show", "find", "which")


@dataclass(frozen=True, slots=True)
class QuestionShapeHint:
    """Best-guess shape for the answer table, derived from question wording.

    ``expected_columns`` is the number of columns gold most likely has. None
    means "uncertain — let the agent decide". ``row_cardinality`` is
    "single" (1 row), "multiple" (≥1 distinct rows), or None for unknown.
    ``aggregation_kind`` is the high-level aggregation: percentage / count /
    average / list / value / None.
    """

    expected_columns: int | None
    row_cardinality: str | None  # "single" | "multiple" | None
    aggregation_kind: str | None  # "percentage" | "count" | "average" | "list" | "value"
    rationale: str = ""

    def to_prompt_line(self) -> str:
        """Render the hint as a single advisory line for prompt injection."""
        parts: list[str] = []
        if self.aggregation_kind == "percentage":
            parts.append(
                "Answer-shape hint (question pattern → percentage): expected 1×1, single numeric value. "
                "Use BIRD-style CAST(COUNT(CASE WHEN ...) AS REAL) * 100 / COUNT(*) over the same filtered population."
            )
        elif self.aggregation_kind == "count":
            parts.append(
                "Answer-shape hint (question pattern → count): expected 1×1, single integer. Question begins with 'how many' — emit only the count value, no additional columns."
            )
        elif self.aggregation_kind == "average":
            base = "Answer-shape hint (question pattern → average): expected 1×1, single numeric value."
            if self.rationale:
                base += " " + self.rationale
            parts.append(base)
        elif self.aggregation_kind == "list":
            base = "Answer-shape hint (question pattern → list): expected N×1 (or N×M if the question explicitly enumerates multiple attributes). Apply DISTINCT."
            if self.rationale:
                base += " " + self.rationale
            parts.append(base)
        elif self.aggregation_kind == "value":
            base = "Answer-shape hint (question pattern → single value): expected 1×1."
            if self.rationale:
                base += " " + self.rationale
            parts.append(base)
        return " ".join(parts) if parts else ""


def infer(question: str) -> QuestionShapeHint | None:
    """Return a shape hint when the question matches a known pattern, else None."""
    if not question:
        return None
    q = question.strip().lower()
    if not q:
        return None

    # v8 — Two-named-entity document comparison (task_352-style).
    # Pattern: question quotes two named entities and asks which is greater/less.
    # When context contains *.md/*.txt knowledge files, agent should grep_text
    # for each entity and compute the comparison in Python — NOT query a
    # structured table that likely doesn't contain the data.
    if (
        re.search(r"\b(more than|greater than|larger than|less than|smaller than|higher than|lower than)\b", q)
        and re.search(r'"[^"]+".*?"[^"]+"', question)
    ):
        return QuestionShapeHint(
            expected_columns=1,
            row_cardinality="single",
            aggregation_kind="count" if q.startswith("how many") else "value",
            rationale=(
                "Two-named-entity comparison (quoted strings): if the context contains "
                ".md / .txt / knowledge files, use grep_text/read_doc to extract each "
                "entity's value, then compute the comparison in Python. The data may not "
                "live in a structured table."
            ),
        )

    # v8 — Abnormal / normal lab range (task_344, task_418).
    # Medical questions referencing "abnormal" or "normal" levels imply a
    # reference-range lookup. Agent should search for a *_reference / *_range
    # / lab_normals table or scan knowledge.md for thresholds.
    # Match both orderings: "abnormal level" / "level is abnormal".
    if re.search(
        r"\b(?:(?:abnormal|normal)\s+(?:[a-z]+\s+)?(?:level|range|value)|"
        r"(?:level|range|value)\s+(?:is|are|of)\s+(?:abnormal|normal))\b",
        q,
    ):
        return QuestionShapeHint(
            expected_columns=1,
            row_cardinality="single",
            aggregation_kind="count" if "how many" in q else "value",
            rationale=(
                "Medical/lab 'abnormal/normal' filter: locate a *reference* table or "
                "knowledge.md describing normal_low/normal_high thresholds (columns named "
                "like normal_range, reference_range, low_value, high_value). Filter rows "
                "OUTSIDE those bounds for 'abnormal', INSIDE for 'normal'."
            ),
        )

    # v8 — Ordinal-per-group pattern (task_379).
    # "the Nth X of each Y" → ROW_NUMBER() OVER (PARTITION BY Y ORDER BY ...)
    if re.search(r"\b\d+(?:st|nd|rd|th)\s+\w+\s+of\s+each\b", q):
        return QuestionShapeHint(
            expected_columns=1,
            row_cardinality="multiple",
            aggregation_kind="list",
            rationale=(
                "Ordinal-per-group: use ROW_NUMBER() OVER (PARTITION BY <group_col> "
                "ORDER BY <seq_col>) inside a subquery, then filter WHERE row_num = N. "
                "The ordering column is typically the table's primary key or atom_id-like "
                "sequence number within the group."
            ),
        )

    # Percentage — most reliable signal. Both "percentage" word and "%".
    # v7: rationale weakened — v6 used a strong BIRD-style formula directive
    # which caused regressions on task_408 / task_420 (v5-perfect). Switch to
    # a verification-style nudge so the agent retains its own filter judgment.
    if "percentage" in q or " percent " in q or q.endswith("%") or "%" in question:
        return QuestionShapeHint(
            expected_columns=1,
            row_cardinality="single",
            aggregation_kind="percentage",
            rationale=(
                "Percentage question — answer is 1×1 numeric. After computing, "
                "double-check that the denominator population matches the question's scope "
                "(BIRD-style gold often uses one SQL expression with COUNT(CASE WHEN ...) "
                "over the same filtered set, but verify against your data — don't blindly "
                "apply a fixed template)."
            ),
        )

    # Count — "how many" prefix. Very high precision in BIRD.
    if q.startswith("how many") or q.startswith("count "):
        return QuestionShapeHint(
            expected_columns=1,
            row_cardinality="single",
            aggregation_kind="count",
        )

    # v8 fix: List patterns take precedence over the loose "average"
    # substring match below. Questions like "List the names and funding
    # types ... where the average SAT exceeds 400" must be classified as
    # list (multi-attribute), not as an average aggregate.
    if any(q.startswith(v + " ") or q.startswith(v + ", ") for v in _LIST_VERBS) or "list the" in q or "name the" in q:
        rationale = ""
        if re.search(r"\b(names? and \w+|\w+ and (?:funding types?|status|category)|\w+ and the \w+)\b", q):
            rationale = "Question explicitly enumerates multiple attributes — emit them as separate columns."
        return QuestionShapeHint(
            expected_columns=None,
            row_cardinality="multiple",
            aggregation_kind="list",
            rationale=rationale,
        )

    # Average — "average X of/per Y" → per-entity. Otherwise just single
    # numeric. Watch for the "names and X" multi-attribute exception where
    # the answer has 2+ columns (e.g. task_199 / task_249).
    if any(w in q for w in ("average ", "avg ", "mean ")):
        # Multi-attribute exception — "average X and Y" → not single value
        if re.search(r"\baverage\s+(?:of\s+the\s+)?\w+\s+(?:and|along with|together with)\s+(?:the\s+)?\w+", q):
            return None  # let agent decide
        rationale = ""
        if re.search(r"\b(of|per)\s+(?:the\s+)?(customers?|users?|members?|patients?|atoms?|drivers?|students?|employees?|products?)\b", q):
            rationale = (
                "The phrase 'of <entity>' / 'per <entity>' means group by that entity first, then average — "
                "NOT total / N_periods. Compute the per-entity aggregate, then average across entities."
            )
        return QuestionShapeHint(
            expected_columns=1,
            row_cardinality="single",
            aggregation_kind="average",
            rationale=rationale,
        )

    # (List pattern moved above the average block — see v8 fix comment.)

    # "What is the X of Y" — singular question. Almost always 1×1 unless
    # explicit multi-attribute ("What are the X and Y of...").
    if q.startswith("what is the") or q.startswith("what was the"):
        return QuestionShapeHint(
            expected_columns=1,
            row_cardinality="single",
            aggregation_kind="value",
        )

    # "Calculate the X" — also singular value.
    if q.startswith("calculate the"):
        return QuestionShapeHint(
            expected_columns=1,
            row_cardinality="single",
            aggregation_kind="value",
        )

    # No known pattern — fall through to caller's free behavior.
    return None
