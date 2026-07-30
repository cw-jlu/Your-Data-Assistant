"""DeepEye-SQL-inspired rule-based SQL checkers for λ0.5 alignment.

Implemented (= λ0.5-aligned, lowest-cost, highest-impact 4):
1. SelectChecker          — `SELECT t.*` → explicit columns (= Extra↓)
2. MaxMinChecker          — `WHERE col = (SELECT MAX/MIN col)` or
                            `SELECT MAX/MIN ... LIMIT 1` → filter-back rewrite
                            (= Recall↑ on ties)
3. OrderByNullChecker     — `ORDER BY col LIMIT N` w/o DESC/SUM/COUNT → add
                            `WHERE col IS NOT NULL` (= Extra↓)
4. TimeChecker            — `strftime(...) [><=] 1234` → `'1234'` quote
                            (regex-only, no LLM)

Pipeline (= per-attempt, before adaptive_vote):
  raw_sql → SyntaxCheck (= raw exec for sanity) → checker chain →
  if any rule fired → LLM revise via COMMON_CHECKER_PROMPT →
  re-execute revised SQL → use revised result for the attempt's answer.

Reference: DeepEye-SQL (arxiv 2510.17586, github.com/HKUSTDial/DeepEye-SQL).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional

from kobushi_core.model import ModelAdapter, ModelMessage


# =========================================================================
# COMMON revision prompt (= DeepEye COMMON_CHECKER_PROMPT pattern)
# =========================================================================

COMMON_CHECKER_PROMPT = """You are a SQL revision assistant. A rule-based checker
detected the following issue in the SQL. Apply ONLY the suggested fix and
return the revised SQL. Do not make other changes.

## Issue detected
{issue}

## Suggested fix
{suggestion}

## Current SQL
```sql
{sql}
```

Output the revised SQL inside a single ```sql ... ``` fenced block. Do not
include explanations, only the SQL."""


def _llm_revise(model: ModelAdapter, sql: str, issue: str, suggestion: str) -> str:
    """Ask the LLM to rewrite the SQL per a single checker's suggestion."""
    prompt = COMMON_CHECKER_PROMPT.format(issue=issue, suggestion=suggestion, sql=sql)
    raw = model.complete(
        [ModelMessage(role="user", content=prompt)],
        enable_thinking=False,
    )
    # Extract the SQL fence
    m = re.search(r"```(?:sql)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Fallback: return original if extract fails
    return sql.strip()


# =========================================================================
# Individual checkers
# =========================================================================

@dataclass(frozen=True, slots=True)
class CheckerResult:
    """Output of a single checker run."""
    name: str
    triggered: bool
    new_sql: str  # = same as input if not triggered
    note: str = ""


# ---- 1. SelectChecker: `table.*` / `t.*` → explicit columns -----------------

_SELECT_STAR_RE = re.compile(
    r"\bSELECT\b[^;]*?\b([A-Za-z_][A-Za-z0-9_]*)\.\*",
    re.IGNORECASE | re.DOTALL,
)


def select_checker(sql: str, model: ModelAdapter | None = None) -> CheckerResult:
    """Detect `table.*` / alias.* in SELECT and rewrite to explicit columns
    via LLM. Plain `SELECT *` (without table prefix) is also caught.

    The official λ0.5 scorer penalises Extra columns via 0.5 × Extra/Predicted,
    so collapsing `*` to the asked-for column(s) directly improves the metric.
    """
    has_alias_star = bool(_SELECT_STAR_RE.search(sql))
    # Also catch plain `SELECT * FROM ...`
    plain_star = bool(re.search(r"\bSELECT\s+\*\s+FROM\b", sql, re.IGNORECASE))
    if not (has_alias_star or plain_star):
        return CheckerResult("select", False, sql)
    if model is None:
        return CheckerResult("select", True, sql, note="detected, no model to revise")
    revised = _llm_revise(
        model,
        sql,
        issue="The SELECT clause uses `table.*` or `SELECT *`, which expands to all columns and inflates the predicted column count beyond what the question asks for.",
        suggestion="Replace `table.*` / `SELECT *` with the explicit columns required to answer the question. List only the column(s) the question asks for. Do NOT add metric/value columns unless the question asks for them.",
    )
    return CheckerResult("select", True, revised, note="alias.* → explicit cols")


# ---- 2. MaxMinChecker: WHERE col = (SELECT MAX/MIN col) → keep ties --------

# Patterns: a) WHERE col = (SELECT MAX/MIN(col) FROM t)
#           b) SELECT MAX/MIN(...) ... LIMIT 1  (= scalar shortcut)
#           c) WHERE col = (SELECT col ... ORDER BY ... LIMIT 1)

_MAXMIN_WHERE_RE = re.compile(
    r"WHERE\s+\w+(?:\.\w+)?\s*=\s*\(\s*SELECT\s+(MAX|MIN)\s*\(",
    re.IGNORECASE,
)
_MAXMIN_LIMIT1_RE = re.compile(
    r"\bORDER\s+BY\s+[^;)]*?\b(?:DESC|ASC)?\s*LIMIT\s+1\b",
    re.IGNORECASE,
)
_SELECT_MAXMIN_LIMIT1_RE = re.compile(
    r"\bSELECT\s+(MAX|MIN)\s*\([^)]+\)[^;]*?\bLIMIT\s+1\b",
    re.IGNORECASE | re.DOTALL,
)


def max_min_checker(sql: str, model: ModelAdapter | None = None) -> CheckerResult:
    """Detect `LIMIT 1` patterns on MIN/MAX queries that drop tied rows.

    Rewrite via LLM to `WHERE col = (SELECT MIN/MAX(col) FROM t)` style or
    explicit DENSE_RANK-equivalent so ALL tied rows are returned.
    """
    where_eq = bool(_MAXMIN_WHERE_RE.search(sql))  # this pattern is GOOD (= keeps ties)
    order_limit1 = bool(_MAXMIN_LIMIT1_RE.search(sql))
    select_maxmin_limit1 = bool(_SELECT_MAXMIN_LIMIT1_RE.search(sql))
    # Trigger only when LIMIT 1 is used as the "min/max picker" (= drops ties)
    triggered = order_limit1 or select_maxmin_limit1
    if not triggered:
        return CheckerResult("max_min", False, sql)
    if model is None:
        return CheckerResult("max_min", True, sql, note="detected, no model")
    revised = _llm_revise(
        model,
        sql,
        issue="The query uses `ORDER BY ... LIMIT 1` or `SELECT MAX/MIN(...) ... LIMIT 1` to pick the row with the extreme value, which silently drops rows tied at that extreme.",
        suggestion="Rewrite using a filter-back pattern: `WHERE col = (SELECT MIN(col) FROM t)` or `WHERE col = (SELECT MAX(col) FROM t)`, joining as needed. This includes ALL rows tied at the extreme value.",
    )
    return CheckerResult("max_min", True, revised, note="LIMIT 1 → filter-back")


# ---- 3. OrderByNullChecker: ORDER BY col LIMIT N → add IS NOT NULL ----------

_ORDER_LIMIT_RE = re.compile(
    r"\bORDER\s+BY\s+([\w\.]+)\s*(?:ASC)?\s*LIMIT\s+(\d+)\b",
    re.IGNORECASE,
)
_AGG_IN_ORDER_RE = re.compile(
    r"\bORDER\s+BY\s+(?:SUM|COUNT|AVG|MIN|MAX|DESC)\b",
    re.IGNORECASE,
)


def order_by_null_checker(sql: str, model: ModelAdapter | None = None) -> CheckerResult:
    """Detect `ORDER BY col LIMIT N` without DESC / aggregate, which on NULL-
    containing columns can return rows with NULL ordering values as "small".

    Add `WHERE col IS NOT NULL` to ensure we don't pick rows that aren't
    meaningfully sorted (= which would be Extra rows).
    """
    m = _ORDER_LIMIT_RE.search(sql)
    if not m:
        return CheckerResult("order_by_null", False, sql)
    # Skip if ORDER BY uses aggregate or DESC
    if _AGG_IN_ORDER_RE.search(sql):
        return CheckerResult("order_by_null", False, sql)
    if "DESC" in sql.upper():
        return CheckerResult("order_by_null", False, sql)
    col = m.group(1)
    if model is None:
        return CheckerResult("order_by_null", True, sql, note=f"detected: {col}")
    revised = _llm_revise(
        model,
        sql,
        issue=f"The query uses `ORDER BY {col} LIMIT ...` without filtering NULLs. NULL values in `{col}` will sort first under ASC, polluting the result with rows that don't have a meaningful sort value.",
        suggestion=f"Add `WHERE {col} IS NOT NULL` (or extend the existing WHERE clause) so only rows with non-NULL `{col}` are considered.",
    )
    return CheckerResult("order_by_null", True, revised, note=f"{col} IS NOT NULL added")


# ---- 4. TimeChecker: strftime(...) [><=] 1234 → quote (regex-only) ---------

_STRFTIME_INT_RE = re.compile(
    r"(strftime\s*\([^)]+\))\s*([<>=]+)\s*(\d{2,4})\b",
    re.IGNORECASE,
)


def time_checker(sql: str, model: ModelAdapter | None = None) -> CheckerResult:
    """Detect `strftime(...) [><=] 1234` (integer compare on string output)
    and quote the integer. LLM not used (= pure regex rewrite)."""
    if not _STRFTIME_INT_RE.search(sql):
        return CheckerResult("time", False, sql)

    def repl(m: re.Match) -> str:
        return f"{m.group(1)} {m.group(2)} '{m.group(3)}'"

    revised = _STRFTIME_INT_RE.sub(repl, sql)
    return CheckerResult("time", True, revised, note="strftime int → quoted")


# =========================================================================
# Pipeline: chain all checkers in order
# =========================================================================

CHECKER_CHAIN: list[Callable[[str, ModelAdapter | None], CheckerResult]] = [
    select_checker,
    max_min_checker,
    order_by_null_checker,
    time_checker,
]


@dataclass(frozen=True, slots=True)
class CheckChainResult:
    final_sql: str
    triggered: list[str]
    notes: list[str]


def run_checker_chain(sql: str, model: ModelAdapter | None = None) -> CheckChainResult:
    """Run all checkers in order, applying revisions cumulatively.

    Each checker sees the (possibly revised) SQL from the previous step.
    Returns the final SQL plus a list of which checkers fired.
    """
    current = sql
    triggered: list[str] = []
    notes: list[str] = []
    for fn in CHECKER_CHAIN:
        result = fn(current, model)
        if result.triggered:
            triggered.append(result.name)
            notes.append(result.note)
            current = result.new_sql
    return CheckChainResult(final_sql=current, triggered=triggered, notes=notes)
