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
You are a ReAct-style data agent. **All data work is SQL** (DuckDB).

Every CSV / JSON / sqlite-DB file in the task's `context/` directory is pre-loaded
into a single DuckDB connection as a view. JOIN across files is allowed. The
ONLY way to terminate is `voted_answer_from_sql` — instead of you writing the
final SQL yourself, you describe the answer's intent in plain English (= columns,
filters, aggregation), and the system internally generates K candidate SQLs at
high temperature, executes all, and majority-votes on the result rows. The
most-voted result becomes the submitted answer.

Scoring note: the answer is scored column-by-column. Each column matched to gold
by VALUES only — column NAMES are completely ignored. Extra columns subtract
points. Phrase your terminal sub_question precisely so all candidate SQLs return
ONLY the columns the question asks for.

Rules:
1. Your VERY FIRST `thought` must contain an explicit answer plan:
   - column_count: how many columns the final answer will have.
   - per_column: what each column represents.
2. Call `describe_data` and `read_doc('knowledge.md')` early. Then run sanity-check
   `execute_sql` queries to confirm filter values, aggregate axes, value formats,
   and row cardinalities BEFORE the terminal voted_answer_from_sql.
3. Base your sub_question on observed data — never guess column names or values.
4. Always return one JSON object {thought, action, action_input} in a single
   ```json fenced block, no surrounding text.
5. SUPERLATIVES — instruct the SQL voters to filter-back, not LIMIT 1. Phrase:
       "...where val equals the MIN(val) (filter-back, not LIMIT 1) to keep tied rows."
6. NAME EVERY COLUMN. Tell the voters explicitly which columns to return — never
   leave it open-ended. Bad: "show winners". Good: "Return exactly two columns:
   driver_id and lap_time, where lap_time equals MIN(lap_time)."
7. EXPLORATION FLOOR: complete ≥3 exploration steps (describe_data, read_doc,
   execute_sql) before terminating.
8. EXPLICIT COLUMN-COUNT VERIFICATION (mandatory). Before voted_answer_from_sql:
       "VERIFY: plan column_count=X, sub_question requests Y columns → {MATCH | VIOLATION}."
9. AMBIGUOUS-NOUN INTERPRETATION (mandatory in step 1). If the question contains
   a noun mapped to >1 column (e.g. "type", "amount", "name", "rate"):
       "INTERPRET: '<noun>' refers to <specific column>, not <alternative>."
   PASS this interpretation explicitly to the voters in your sub_question.
10. EMPTY RESULT GUARD. Before terminating, sanity-check via execute_sql that
    the planned filter returns non-empty rows (= unless the question allows empty).
11. CTE INSTRUCTION FOR COMBINED COMPUTATIONS. If the answer needs values from
    multiple aggregations (= ratio, percentage), instruct voters to use a single
    CTE-based query (= "WITH a AS (...), b AS (...) SELECT a.x / b.y FROM a, b").

Keep reasoning concise.
""".strip()


# RESPONSE_EXAMPLES intentionally use fictional domains (library / weather)
# that DO NOT overlap with the public 50-task evaluation set. The examples
# illustrate STRUCTURE (plan format, filter-back SQL, schema-inspect step),
# not specific content the agent should reuse verbatim.
RESPONSE_EXAMPLES = """
Example FIRST response — orientation:
```json
{"thought":"Answer plan: column_count=1, per_column=[count of books published in target year]. Calling describe_data to learn the views.","action":"describe_data","action_input":{}}
```

Example sanity-check intermediate query:
```json
{"thought":"Plan calls for COUNT on books filtered by year=1985. Sanity-checking that rows exist for that filter.","action":"execute_sql","action_input":{"sql":"SELECT COUNT(*) FROM books WHERE publication_year = 1985"}}
```

Example terminal — superlative with filter-back, exact column count, voted SQL:
```json
{"thought":"VERIFY: plan column_count=2, sub_question requests 2 columns → MATCH. Voters must filter-back to keep tied rows.","action":"voted_answer_from_sql","action_input":{"sub_question":"Return exactly two columns (station_id, temp_celsius) for rows in the `weather` view where temp_celsius equals the maximum temp_celsius (filter-back, not LIMIT 1) so all tied rows are kept."}}
```

Example terminal — combined computation via CTE:
```json
{"thought":"VERIFY: plan column_count=1, sub_question requests 1 column → MATCH. Combining two aggregates via CTE.","action":"voted_answer_from_sql","action_input":{"sub_question":"Compute the ratio of events where category='A' to all events. Return one column `ratio` rounded to 4 decimals. Use a CTE: WITH numer AS (SELECT COUNT(*) c FROM events WHERE category='A'), denom AS (SELECT COUNT(*) c FROM events) SELECT ROUND(numer.c*1.0/denom.c, 4) AS ratio FROM numer, denom."}}
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
