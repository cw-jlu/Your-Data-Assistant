"""Prompts for the ReFoRCE-style 2-phase pipeline.

Closely modelled on the official ReFoRCE prompt templates
(references/ReFoRCE/methods/ReFoRCE/prompt.py). Adapted to:
- DuckDB dialect (vs. ReFoRCE's BigQuery / Snowflake)
- DABench's question + knowledge.md format (vs. Spider 2.0's table_info)
- Our unified DuckDB views (= every CSV/JSON/sqlite is a view)

The structural intent of each prompt is preserved verbatim.
"""
from __future__ import annotations

from kobushi_core.benchmark.schema import PublicTask


# ============================================================================
# Phase 1: EXPLORATION
# ============================================================================
# Mirrors ReFoRCE's `get_exploration_prompt()` — generate 3-10 short queries
# to understand column values BEFORE committing to an answer SQL.
EXPLORATION_SYSTEM = """\
You are an SQL exploration helper. Before generating the answer SQL, you
generate short DuckDB SELECT queries that surface the information needed:

- DISTINCT values of categorical columns
- MIN/MAX/COUNT for numeric columns
- date format / range
- row counts per filter to confirm selectivity
- column cardinality for join keys

Output 3 to 8 short queries as a ```json array. Each must start with SELECT or
WITH. Keep them brief — do NOT try to compute the final answer here.

Output format (= structural template, replace `weather`/`station_id`/`temp`
with the actual view names from this task — domains shown here do NOT
appear in the dataset):
```json
["SELECT DISTINCT category, COUNT(*) FROM weather GROUP BY category", "SELECT MIN(reading_date), MAX(reading_date), COUNT(*) FROM stations"]
```
No surrounding text.
"""


def build_exploration_user_prompt(task: PublicTask, catalog: str, knowledge: str) -> str:
    return (
        f"## Task\n{task.question}\n\n"
        f"## DuckDB views available\n{catalog}\n\n"
        f"## knowledge.md (= excerpt)\n{knowledge}\n\n"
        "Output ONE ```json array of 3-8 short exploratory SELECT statements."
    )


# ============================================================================
# Phase 2: SELF-REFINE (= main answer SQL generation, with self-consistency)
# ============================================================================
# Closely mirrors ReFoRCE's `get_self_refine_prompt()` (prompt.py:112-147).
# Key elements transplanted:
#   - "table_info" → our DuckDB catalog
#   - "Some few-shot examples after column exploration may be helpful" prefix
#   - "Task: ... Please think step by step and answer only one complete SQL"
#   - dialect tips section
SELF_REFINE_SYSTEM = """\
You are a SQL specialist for DuckDB. Generate exactly ONE complete SQL query
that answers the task. Output the SQL inside a ```sql fenced block. No
surrounding text.

# Output format (= the canonical answer header is provided to you)
You will be given a `format_csv` block — a CSV header showing the expected
output columns inferred from the question. Your SELECT MUST produce exactly
the columns listed there, in the same order. Do NOT add extras.

# Filter columns ≠ output columns
A `WHERE col = value` clause selects rows; it does NOT make `col` an output
column. The format_csv header determines the output set, not the filter clauses.

# SQL dialect (DuckDB)
- Every CSV/JSON/sqlite-DB file in the task context is exposed as a view; you
  can JOIN across them in one statement.
- For superlatives (lowest/highest/min/max), use filter-back, NOT LIMIT 1:
      SELECT col FROM t WHERE val = (SELECT MIN(val) FROM t)
  This preserves tied rows.
- Name every column explicitly. NEVER `SELECT *`.
- Apply ROUND() only at the outermost SELECT level.
- Pure SELECT/WITH only. No DDL/DML.
"""


def build_self_refine_prompt(
    *, task: PublicTask, catalog: str, knowledge: str, exploration_findings: str,
    format_csv: str = "",
) -> str:
    """First-iteration prompt for the self_refine phase.

    Mirrors ReFoRCE's structure (prompt.py:112-147):
      table_info + "Some few-shot examples after column exploration may be helpful:" + pre_info
      + "Task: ... Please think step by step and answer only one complete SQL"
    """
    parts = [
        "## DuckDB views (= the schema)",
        catalog,
        "",
        "## knowledge.md (= caveats, definitions)",
        knowledge,
    ]
    if exploration_findings:
        parts.extend([
            "",
            "Some few-shot examples after column exploration may be helpful:",
            exploration_findings,
        ])
    if format_csv:
        parts.extend([
            "",
            "## Expected answer format (= use this as your SELECT column list)",
            "```csv",
            format_csv,
            "```",
            "Your SELECT must produce exactly these columns, in this order. "
            "Re-name columns via `AS` if your source columns have different names.",
        ])
    parts.extend([
        "",
        f"Task: {task.question}",
        "",
        "Think step by step and answer ONE complete SQL in DuckDB dialect "
        "in ```sql``` format.",
        "",
        "Tips:",
        "- For decimal answers, prefer ROUND(value, N) where N matches the "
        "question's stated precision (default 2).",
        "- For superlatives, filter-back (= WHERE val = (SELECT MIN/MAX(val) "
        "FROM ...)) to keep tied rows; never LIMIT 1.",
        "- Apply ROUND() only at the outermost SELECT level.",
    ])
    return "\n".join(parts)


# ============================================================================
# SELF-CONSISTENCY VERIFY (= "is the prior answer correct?")
# ============================================================================
# Mirrors ReFoRCE's `get_self_consistency_prompt()` (prompt.py:149-153) plus
# the in-loop augmentation at agent.py:215-247 (current SQL + current CSV +
# nested-value / empty-column hints).
def build_self_consistency_prompt(
    *, task: PublicTask, prior_sql: str, prior_csv_preview: str,
    prior_columns: list[str], prior_n_rows: int,
    nested_values: list | None = None,
    empty_columns: list | None = None,
    format_csv: str = "",
) -> str:
    parts = [
        f"Please check the answer again by reviewing task:\n{task.question}\n",
        "reviewing Relevant Tables and Columns and Possible Conditions and "
        "then give the final SQL query. Don't output other queries. If you "
        "think the answer is right, just output the current SQL.",
        "",
        f"Current SQL:\n```sql\n{prior_sql}\n```",
        "",
        f"Current answer ({prior_n_rows} rows × {len(prior_columns)} cols, "
        f"columns={prior_columns}):",
        prior_csv_preview,
    ]
    if format_csv:
        parts.extend([
            "",
            f"Expected answer format header was: `{format_csv}`. Your current "
            "columns must match this header. If they don't, fix the SELECT.",
        ])
    parts.extend([
        "",
        "VERIFY before finalising:",
        "1. Do your current columns match the expected format header?",
        "2. Are filter columns leaking into the output? If yes, remove them.",
        "3. Numeric scope: if the question asks for a per-X average, did you "
        "divide by the number of X (= count of distinct customers / users / "
        "etc.)? Sum-over-all divided by 12 is NOT the same as "
        "average-per-customer / 12.",
        "4. Aggregation level: if the question references a single parent "
        "entity (= one event, one account), the answer is at that parent's "
        "level — do NOT GROUP BY a sub-attribute of it.",
    ])
    if nested_values:
        parts.append(
            f"\nValues {nested_values[:3]} look nested. Please flatten them "
            "(e.g. transfer '[\\nA,\\n B\\n]' to 'A, B')."
        )
    if empty_columns:
        parts.append(
            f"\nColumns {empty_columns} are entirely empty / zero. "
            "Please correct them."
        )
    parts.append(
        "\nDecimal precision: use ROUND(value, N) where N matches the "
        "question's stated precision (default 2)."
    )
    return "\n".join(parts)


# ============================================================================
# ERROR CORRECTION (= when SQL fails to execute)
# ============================================================================
# Mirrors ReFoRCE's `get_exploration_self_correct_prompt()` (prompt.py:108-110).
def build_error_correct_prompt(*, prior_sql: str, error: str) -> str:
    return (
        f"Input sql:\n```sql\n{prior_sql}\n```\n"
        f"The error information is:\n{error}\n"
        "Please correct it based on the previous context and output the "
        "thinking process with only one SQL query in ```sql\n--Description: \n``` "
        "format. Don't just analyze without SQL or output several SQLs."
    )


# ============================================================================
# EMPTY-RESULT SIMPLIFICATION (= "0 rows means filter too strict")
# ============================================================================
# Mirrors ReFoRCE's empty-result hint (agent.py:113):
#   "Since the output is empty, please simplify some conditions of the past sql."
def build_empty_simplify_prompt(*, prior_sql: str) -> str:
    return (
        f"Input sql:\n```sql\n{prior_sql}\n```\n"
        "Since the output is empty, please simplify some conditions of the "
        "past sql. Start by checking which single condition is the most "
        "restrictive (= run a sanity-check SELECT COUNT(*) WHERE that_condition), "
        "then either remove or relax that condition.\n"
        "Output one corrected SQL in ```sql``` format."
    )
