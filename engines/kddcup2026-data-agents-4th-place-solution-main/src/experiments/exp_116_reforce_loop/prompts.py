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

# Output column selection (= the most common failure mode)
Pick the MINIMUM SUFFICIENT set of output columns from the question. The
question's wording dictates the columns; the schema's filter columns are
NOT automatically output columns.

Rules:
- "List / Identify / Show all <entity>" → return the entity's PRIMARY
  IDENTIFIER (one column). When the schema offers both a name-like column
  and an ID-like column for the same entity, pick the human-readable name
  unless the question explicitly says "id" / "identifier" — names are the
  canonical answer for "list" questions.
  Do NOT include the entity's attributes unless the question says "with
  their X" or "and their X".
- "Which <entity> has the <superlative>" → return the entity's primary
  identifier (= same name-vs-id rule above) using filter-back, ONE column:
       SELECT <entity_id_or_name> FROM <entity_table>
       WHERE <metric> = (SELECT MIN(<metric>) FROM ...)
- "What is the <attribute> of <entity>" → SELECT <attribute> FROM ... (one column).
- "How many / count / total / sum / average" → ONE numeric column.
- "<attribute_A> and <attribute_B> of <entity>" → exactly TWO columns A, B.
- "<attribute> of X for parent <entity>" → return the attribute at the PARENT
  granularity (one row per parent), NOT GROUP BY a sub-attribute of X. If the
  question mentions a parent entity (= a specific event, a specific account),
  the answer is at that parent's level.

Filter columns ≠ output columns: a condition like `WHERE col = value` selects
rows; it does NOT make `col` an output column. The output set is determined
solely by the question's main verb and direct object.

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
    *, task: PublicTask, catalog: str, knowledge: str, exploration_findings: str
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
    parts.extend([
        "",
        f"Task: {task.question}",
        "",
        "BEFORE writing SQL, write 2 short lines of analysis:",
        "  PARSE: the question's grammatical core: <verb> <output entity/attribute> "
        "[FROM <source entity>] [WHERE <filter>].",
        "  COLUMN_PLAN: column_count=N, columns=[c1, c2, ...]. Justify each "
        "column from the question's wording — NOT from the filter clauses.",
        "",
        "Then think step by step and answer ONE complete SQL in DuckDB dialect "
        "in ```sql``` format.",
        "",
        "Tips:",
        "- For decimal answers, prefer ROUND(value, N) where N matches the "
        "question's stated precision (default 2).",
        "- For superlatives, filter-back to keep tied rows; never LIMIT 1.",
        "- The number of columns in your SELECT MUST equal the COLUMN_PLAN.",
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
        "",
        "REVERIFY the column selection (= most common failure mode):",
        "1. Re-parse the question's grammatical core. Which entity / attribute "
        "is the OUTPUT? Which clauses are FILTERS (not outputs)?",
        "2. Are your current columns the MINIMUM SUFFICIENT set the question "
        "asks for? If you have FILTER columns leaking into the output, REMOVE them.",
        "3. For 'list/identify <entity>' questions, the output is typically the "
        "entity's primary key only — not its attributes.",
        "4. For 'what is the X of Y' questions, output is one column for X.",
        "5. Re-state COLUMN_PLAN before finalising. If your prior SQL violated "
        "the plan, output a corrected SQL.",
    ]
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
