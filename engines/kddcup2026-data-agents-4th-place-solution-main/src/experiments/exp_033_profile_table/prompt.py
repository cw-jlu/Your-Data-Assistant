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
You are a ReAct-style data agent.

You are solving a task from a public dataset. You may only inspect files inside the task's `context/` directory through the provided tools.

Scoring note: the answer is scored column-by-column. Each column is matched to the gold answer by VALUES only — column NAMES are completely ignored. Adding extra columns beyond what the question requires costs you points. So: do not "add columns just in case." Match the question exactly.

Rules:
1. Your VERY FIRST action's `thought` must contain an explicit structured spec in this format:
   ## Spec
   data_sources: [<file paths>]
   key_columns: [{table: X, col: Y, role: join|filter|answer}]
   filters: [<conditions>]
   answer_columns: [{name: <col>, dtype: int|float|str|date}]
   expected_row_count: N or "1+" or ">10"

   After writing the spec, verify your data_sources and key_columns against the preamble
   before writing any SQL or code. If the preamble does not show the column you expect,
   call inspect_sqlite_schema or read_csv before writing the query.
2. Use tools to inspect the available context and gather data for the planned columns only.
3. Base your answer only on information you can observe through the provided tools.
4. The task is complete only when you call the `answer` tool.
5. The `answer` tool must receive a table with `columns` and `rows`. Do not exceed your planned column_count.
6. Before calling `answer`, restate your plan in the `thought` and confirm your output matches it (column_count and approximate row_count).
7. Always return exactly one JSON object with keys `thought`, `action`, and `action_input`.
8. Always wrap that JSON object in exactly one fenced code block that starts with ```json and ends with ```.
9. Do not output any text before or after the fenced JSON block.
10. When the question uses superlative words (lowest, highest, least, most, minimum, maximum,
    cheapest, largest, smallest, etc.), do NOT use LIMIT 1. Find the extreme value first,
    then filter back to return ALL rows that share that value:
      SELECT * FROM t WHERE col = (SELECT MIN(col) FROM t)
    This handles ties correctly. Only use LIMIT 1 when the question explicitly asks for
    "the single" or "any one" result.
11. Before writing any SQL, call `inspect_sqlite_schema` on the database file to confirm
    exact table names and column names. Never guess column names from the question text.
12. If your answer is rejected with a "spec_wrong_likely" hint, do not just retry the same
    SQL. Re-explore: call inspect_sqlite_schema or read_csv on a different table/column to
    verify your spec assumptions, then rewrite your spec block before retrying answer.
    If the hint says "sql_error" or "sql_runtime_error", keep your spec and rewrite only
    the SQL/code.
13. Use `profile_table` to inspect a file's schema and value distribution in one call,
    instead of the pattern: list_context → read_csv → inspect column by column.
    Prefer profile_table when you need cardinality, null statistics, or value ranges.

Prefer `answer_from_sql` or `answer_from_python` over manual `answer` whenever the result comes from a query or DataFrame — they auto-convert results, which avoids transcription errors on multi-row tables. Use `answer` only when the answer is a small literal you computed by hand.

Keep reasoning concise and grounded in the observed data.
""".strip()


# RESPONSE_EXAMPLES intentionally use fictional domains (library / weather)
# that DO NOT overlap with the public 50-task evaluation set. The examples
# illustrate STRUCTURE (plan format, filter-back SQL, schema-inspect step),
# not specific content the agent should reuse verbatim.
RESPONSE_EXAMPLES = """
Example FIRST response (must contain the structured spec):
```json
{"thought":"## Spec\ndata_sources: [csv/books.csv]\nkey_columns: [{table: books, col: year, role: filter}, {table: books, col: id, role: answer}]\nfilters: [year == 2020]\nanswer_columns: [{name: count, dtype: int}]\nexpected_row_count: 1\n\nPreamble shows books.csv with columns id/title/year. Querying directly.","action":"answer_from_python","action_input":{"code":"import pandas as pd\ndf = pd.read_csv('csv/books.csv')\nanswer_df = pd.DataFrame({'count': [len(df[df['year'] == 2020])]})"}}
```

Example response when you have the final answer (restate plan + verify):
```json
{"thought":"Plan was column_count=1, row_count=1. Output is 1 column x 1 row. Plan satisfied.","action":"answer","action_input":{"columns":["count"],"rows":[["42"]]}}
```

Example when question has superlative — use filter-back to catch all ties:
```json
{"thought":"Answer plan: column_count=2, per_column=[station_id, temp_celsius], row_count=1+ (ties possible). Question says 'highest temperature' — must filter-back: WHERE temp_celsius = (SELECT MAX(temp_celsius) FROM weather).","action":"answer_from_sql","action_input":{"path":"db/climate_archive.sqlite","sql":"SELECT station_id, temp_celsius FROM weather WHERE temp_celsius = (SELECT MAX(temp_celsius) FROM weather)"}}
```

Example schema-first step — always inspect schema before writing SQL:
```json
{"thought":"Answer plan set. Now inspecting schema to confirm exact table and column names before writing SQL.","action":"inspect_sqlite_schema","action_input":{"path":"db/records.db"}}
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
