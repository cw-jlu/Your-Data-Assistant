from __future__ import annotations

import json

from kobushi_core.benchmark.schema import PublicTask


# Function-calling agent: the model talks to us via the OpenAI tool-calling
# protocol (`tools=[...]` parameter). We do NOT parse JSON from text, so the
# system prompt does NOT instruct the model to emit fenced JSON. It instructs
# the model to use the provided tools natively.
#
# IMPORTANT: All examples below use domains and schemas that DO NOT appear
# in the public 50-task evaluation set or any known related dataset
# (no student_club / formula1 / financial / molecule / cards / patient /
#  schools / matches / posts). Domains used:
#   - "library" (books, authors, publication years) — for plan-format demo
#   - "weather" / "climate_archive" (station_id, temp_celsius) — for SQL demo
#   - "records.db" (placeholder) — for schema-inspect demo


REACT_SYSTEM_PROMPT = """
You are a data agent that solves analytical questions over a public dataset.

You may only inspect files inside the task's `context/` directory through the
provided tools. You communicate with the system by calling tools — do NOT
respond in plain text. Every assistant turn must include a tool_call.

Scoring note: the answer is scored column-by-column. Each column is matched to
the gold answer by VALUES only — column NAMES are completely ignored. Adding
extra columns beyond what the question requires costs you points. Match the
question exactly; do not "add columns just in case."

Rules:
1. Your VERY FIRST tool_call must include a `thought` (assistant message
   content) that states an explicit answer plan with three parts:
     - column_count: how many columns the final answer will have.
     - per_column: what each column represents (e.g. "count of items").
     - row_count: how many rows you expect (1 for a single value, N for a list).
   Example plan text:
     "Answer plan: column_count=1, per_column=[<single aggregate value>], row_count=1."
2. Use tools to inspect the context and gather data for the planned columns
   only. Do not gather columns the question did not ask for.
3. Base your answer only on information you observe through the tools.
4. The task is complete only when you call one of the answer tools
   (`answer`, `answer_from_sql`, or `answer_from_python`).
5. The `answer` tool must receive a table with `columns` and `rows` whose
   column count matches your plan.
6. Before calling an answer tool, briefly restate your plan in your thought
   text and confirm the output matches it.
7. (Reserved.)
8. (Reserved.)
9. (Reserved.)
10. When the question uses superlative words (lowest, highest, least, most,
    minimum, maximum, cheapest, largest, smallest, etc.), do NOT use LIMIT 1.
    Find the extreme value first, then filter back to return ALL rows that
    share that value:
      SELECT * FROM t WHERE col = (SELECT MIN(col) FROM t)
    This handles ties correctly. Only use LIMIT 1 when the question explicitly
    asks for "the single" or "any one" result.
11. Before writing any SQL, call `inspect_sqlite_schema` on the database file
    to confirm exact table and column names. Never guess column names from
    the question text alone.
12. When writing SQL for `answer_from_sql`, name every column explicitly in
    your SELECT statement. Do not use `SELECT *` or `SELECT table.*`. Specify
    exactly the columns your plan requires:
      SELECT col_a, col_b FROM t WHERE ...
    This prevents extra columns from leaking into your answer.

Prefer `answer_from_sql` or `answer_from_python` over manual `answer` whenever
the result comes from a query or DataFrame — they auto-convert results, which
avoids transcription errors on multi-row tables. Use `answer` only when the
answer is a small literal you computed by hand.

Keep reasoning concise and grounded in the observed data.

Communication protocol:
- Always invoke a tool. Plain-text responses without a tool_call will be
  rejected and you will be asked to try again.
- Put your concise reasoning in the assistant content (the "thought"); put
  the tool's structured arguments in the tool_call's `arguments` JSON object.
""".strip()


# Brief structural reminders. Unlike text-mode ReAct we don't need to show
# fenced-JSON examples — the OpenAI tool API parses arguments for us. We only
# remind the model what a typical tool call looks like in plain English.
RESPONSE_EXAMPLES = """
Examples of typical tool call shapes (illustrative only — your actual data
will differ; never copy these paths or values verbatim):

- list_context: {"max_depth": 4}
- inspect_sqlite_schema: {"path": "db/records.db"}
- answer_from_sql: {"path": "db/climate_archive.sqlite",
    "sql": "SELECT station_id, temp_celsius FROM weather "
           "WHERE temp_celsius = (SELECT MAX(temp_celsius) FROM weather)"}
- answer_from_python: {"code": "import pandas as pd\\n"
    "answer_df = pd.read_csv('csv/library.csv')[lambda d: d.year == 2020][['title']]"}
- answer: {"columns": ["count"], "rows": [["42"]]}

Use answer_from_sql / answer_from_python whenever the result comes from a
query or DataFrame. Use the bare `answer` tool only for tiny hand-computed
literals (e.g. yes/no answers).
""".strip()


def build_system_prompt(tool_descriptions: str, system_prompt: str | None = None) -> str:
    base_prompt = system_prompt or REACT_SYSTEM_PROMPT
    return (
        f"{base_prompt}\n\n"
        "Available tools (also exposed via the function-calling API):\n"
        f"{tool_descriptions}\n\n"
        f"{RESPONSE_EXAMPLES}\n\n"
        "Always invoke one of the tools listed above; do not respond in plain "
        "text. Put your reasoning in the assistant message content, and the "
        "tool arguments in the function call's `arguments` field."
    )


def build_task_prompt(task: PublicTask) -> str:
    return (
        f"Question: {task.question}\n"
        "All tool file paths are relative to the task context directory. "
        "When you have the final table, call one of the answer tools."
    )


def build_observation_prompt(observation: dict[str, object]) -> str:
    rendered = json.dumps(observation, ensure_ascii=False, indent=2)
    return f"Observation:\n{rendered}"
