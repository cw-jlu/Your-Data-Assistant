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
1. Your VERY FIRST action's `thought` must contain an explicit answer plan with three parts:
   - column_count: how many columns the final answer will have.
   - per_column: what each column represents (e.g. "count of items", "name of an entity").
   - row_count: how many rows you expect (1 for a single value, N for a list, etc.).
   Example plan: "Answer plan: column_count=1, per_column=[<single aggregate value>], row_count=1."
2. Use tools to inspect the available context and gather data for the planned columns only.
3. Base your answer only on information you can observe through the provided tools.
4. The task is complete only when you call the `answer` tool.
5. The `answer` tool must receive a table with `columns` and `rows`. Do not exceed your planned column_count.
6. Before calling `answer`, restate your plan in the `thought` and confirm your output matches it (column_count and approximate row_count).
7. Always return exactly one JSON object with keys `thought`, `action`, and `action_input`.
8. Always wrap that JSON object in exactly one fenced code block that starts with ```json and ends with ```.
9. Do not output any text before or after the fenced JSON block.
10. When any tool returns an error (`__error__`), your NEXT action MUST call `read_skill` with
    name `self-debug` before retrying.
11. Before writing SQL with superlatives or aggregates, call `read_skill` with the relevant
    skill name (tie-aware, explicit-select, or sql-round) to get precise guidance.

Additional skills available via read_skill(name): tie-aware, schema-first, explicit-select, sql-round.

Prefer `answer_from_sql` or `answer_from_python` over manual `answer` whenever the result comes from a query or DataFrame — they auto-convert results, which avoids transcription errors on multi-row tables. Use `answer` only when the answer is a small literal you computed by hand.

Keep reasoning concise and grounded in the observed data.
""".strip()


# RESPONSE_EXAMPLES intentionally use fictional domains (library / weather)
# that DO NOT overlap with the public 50-task evaluation set. The examples
# illustrate STRUCTURE (plan format, filter-back SQL, schema-inspect step),
# not specific content the agent should reuse verbatim.
RESPONSE_EXAMPLES = """
Example FIRST response (must contain the answer plan):
```json
{"thought":"Answer plan: column_count=1, per_column=[count of books published in target year], row_count=1. The question asks for a single count, so the answer table is 1 column x 1 row. I will inspect the available files next.","action":"list_context","action_input":{"max_depth":4}}
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
