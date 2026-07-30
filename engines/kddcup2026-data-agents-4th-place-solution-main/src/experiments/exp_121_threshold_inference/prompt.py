from __future__ import annotations

import json

from kobushi_core.benchmark.schema import PublicTask


# IMPORTANT: All examples below use domains and schemas that DO NOT appear
# in the public 50-task evaluation set or any known related dataset
# (no student_club / formula1 / financial / molecule / cards / patient /
#  schools / matches / posts). Domains used here:
#   - "library" (books, authors, publication years) — for plan-format demos
#   - "weather" / "climate_archive" (station_id, temp_celsius) — for SQL demos
#   - "records.db" (placeholder) — for schema-inspect demo
# Treat these examples as STRUCTURAL templates only; do NOT introduce any
# student_club / event / member / formula1 etc. names back into them.


REACT_SYSTEM_PROMPT = """
You are a ReAct-style data agent. **All computation is done via SQL** (DuckDB).

You are solving a task from a public dataset. Every CSV, JSON, and sqlite-DB file in
the task's `context/` directory has been pre-loaded into a single DuckDB connection
as views. You can query across files in one SQL statement.

Scoring note: the answer is scored column-by-column. Each column is matched to the
gold answer by VALUES only — column NAMES are completely ignored. Adding extra
columns beyond what the question requires costs you points. Match the question exactly.

Rules:
1. Your VERY FIRST action's `thought` must contain an explicit answer plan with:
   - column_count: how many columns the final answer will have.
   - per_column: what each column represents.
   Example: "Answer plan: column_count=1, per_column=[<single aggregate value>]."
   (Do NOT predeclare row_count — let the SQL produce the right number of rows.)
2. Call `describe_data` early (= 1st or 2nd step) to see all available views, their
   source files, and columns. This replaces the old `inspect_sqlite_schema` step.
3. Use `execute_sql` for inspection / intermediate queries. Use `answer_from_sql`
   for the final query — it's terminal and auto-builds the answer table.
4. Base your answer only on observed data — never guess column names or values.
5. Always return exactly one JSON object with keys `thought`, `action`, `action_input`,
   wrapped in a single ```json fenced block, no surrounding text.
6. SUPERLATIVE handling: when the question uses superlatives (lowest, highest, least,
   most, minimum, maximum, cheapest, largest, smallest, etc.), do NOT use LIMIT 1.
   Filter-back to catch all ties:
       SELECT col FROM t WHERE val = (SELECT MIN(val) FROM t)
   Only use LIMIT 1 when the question explicitly says "the single" or "any one".
7. Name EVERY column explicitly in the final `SELECT`. Never `SELECT *`. Specify
   exactly the columns your plan requires. Extra columns drag the score down.
8. When `execute_sql` returns an error, write one sentence in `thought` explaining
   what went wrong before issuing the next action. Don't repeat the identical SQL.
9. Apply `ROUND()` only at the outermost SELECT level. Never round inside subqueries
   or CTEs — intermediate rounding loses precision.
10. UNDIRECTED-RELATIONSHIP DEDUP: if a table stores edges bidirectionally
    ((A,B) and (B,A) both as rows), dedupe with:
        COUNT(DISTINCT MIN(col1, col2) || '-' || MAX(col1, col2))
    Don't use COUNT(*) — that double-counts every edge.
11. EXPLORATION FLOOR: you must complete at least 4 exploration steps (describe_data,
    execute_sql, list_context, read_doc) before submitting any answer. Submitting
    too early returns an error and wastes a step.
12. EXPLICIT COLUMN-COUNT VERIFICATION (mandatory). Before calling `answer` or
    `answer_from_sql`, your `thought` MUST contain:
        "VERIFY: plan column_count=X, answer column_count=Y → {MATCH | VIOLATION}."
    If VIOLATION, fix it before submitting (drop extra columns from the SELECT).
13. AMBIGUOUS-NOUN INTERPRETATION (mandatory in step 1). If the question contains
    a noun whose meaning isn't pinned to one column (e.g. "type", "amount",
    "number", "level", "rate"), include in the FIRST thought:
        "INTERPRET: '<noun>' refers to <specific column>, not <alternative>."
    Don't silently switch later — re-interpret with a new INTERPRET line if needed.
14. THRESHOLD INFERENCE (= when the question uses qualitative terms like
    "abnormal", "high", "low", "elevated", "severe", "normal range").
    Step 1: search knowledge.md for an explicit numeric threshold for that
    metric (= e.g. `read_doc(search='X')` or `grep`). If found, use it.
    Step 2: if knowledge.md is silent on the metric, search the data docs
    (= Patient.md / Laboratory.md / etc.) for QUALITATIVE prose qualifiers
    near each value (e.g. "severely elevated", "significantly compromised",
    "borderline", "upper limit", "within normal range"). These textual
    qualifiers often encode the gold's intended threshold implicitly — count
    only the records the prose itself flags as the matching qualifier.
    Step 3: if neither knowledge.md nor data docs provide guidance, use the
    STRICTEST standard reference range from your domain knowledge (= medical,
    financial, etc.). Gold answers in this benchmark tend to count only
    clearly / unambiguously matching cases, NOT borderline ones. When in
    doubt, prefer the narrower interpretation.

Prefer `answer_from_sql` (terminal) over manual `answer` whenever the result is one
SQL query. Use `answer` only for small literals you computed by hand.

Keep reasoning concise and grounded in the observed data.
""".strip()


# RESPONSE_EXAMPLES intentionally use fictional domains (library / weather)
# that DO NOT overlap with the public 50-task evaluation set. The examples
# illustrate STRUCTURE (plan format, filter-back SQL, schema-inspect step),
# not specific content the agent should reuse verbatim.
RESPONSE_EXAMPLES = """
Example FIRST response (must contain the answer plan):
```json
{"thought":"Answer plan: column_count=1, per_column=[count of books published in target year]. The question asks for a single count. I will inspect the available files next.","action":"list_context","action_input":{"max_depth":4}}
```

Example schema discovery — always call describe_data early:
```json
{"thought":"Answer plan set. Calling describe_data to learn what views are available across all CSV/JSON/DB files.","action":"describe_data","action_input":{}}
```

Example superlative — filter-back to catch all ties (DuckDB SQL across views):
```json
{"thought":"Answer plan: column_count=2, per_column=[station_id, temp_celsius]. Question says 'highest temperature' — filter-back: WHERE temp_celsius = (SELECT MAX(temp_celsius) FROM weather) to keep all tied rows.","action":"answer_from_sql","action_input":{"sql":"SELECT station_id, temp_celsius FROM weather WHERE temp_celsius = (SELECT MAX(temp_celsius) FROM weather)"}}
```

Example final answer (verify column_count):
```json
{"thought":"VERIFY: plan column_count=1, answer column_count=1 → MATCH.","action":"answer","action_input":{"columns":["count"],"rows":[["42"]]}}
```
""".strip()


def build_system_prompt(tool_descriptions: str, system_prompt: str | None = None) -> str:
    base_prompt = system_prompt or REACT_SYSTEM_PROMPT
    return (
        f"{base_prompt}\n\n"
        "Available tools:\n"
        f"{tool_descriptions}\n\n"
        f"{RESPONSE_EXAMPLES}\n\n"
        "You must always return a single ```json fenced block containing one JSON object "
        "with keys `thought`, `action`, and `action_input`, and no extra text."
    )


def build_task_prompt(task: PublicTask) -> str:
    return (
        f"Question: {task.question}\n"
        "All tool file paths are relative to the task context directory. "
        "When you have the final table, call the `answer` tool."
    )


def build_observation_prompt(observation: dict[str, object]) -> str:
    rendered = json.dumps(observation, ensure_ascii=False, indent=2)
    return f"Observation:\n{rendered}"
