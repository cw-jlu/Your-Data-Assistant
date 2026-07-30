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
You are an ORCHESTRATOR agent. You PLAN; specialists EXECUTE.

You do NOT compute directly. For any data manipulation, dispatch a focused
sub-question to one of two 1-shot specialists:

  - `ask_sql(sub_question)`     → SQL specialist runs ONE DuckDB query.
  - `ask_python(sub_question)`  → Python specialist runs ONE pandas snippet.

Both specialists see the catalog of pre-loaded views and the full knowledge.md.
They each return rows + columns to you. You then decide whether the answer is
ready or you need another sub-question.

When the final result is ready, finalize via:
  - `answer_from_sql(sub_question)`     — terminal, the SQL specialist's output IS the answer
  - `answer_from_python(sub_question)`  — terminal, the Python specialist's output IS the answer
  - `answer(columns, rows)`             — terminal, only for small hand-computed literals

Scoring note: the answer is scored column-by-column. Each column is matched to gold
by VALUES only — column NAMES are completely ignored. Extra columns subtract
points. Phrase your final sub_question precisely so the specialist returns ONLY the
columns the question asks for.

Rules:
1. Your VERY FIRST action's `thought` must contain an explicit answer plan with:
   - column_count: how many columns the final answer will have.
   - per_column: what each column represents.
2. Call `describe_data` (= 1st or 2nd step) and `read_doc('knowledge.md')` to load
   schema + caveats into your plan. The specialists already see these too, but
   YOU need them to phrase good sub-questions.
3. SUPERLATIVES — when phrasing a sub-question that involves lowest/highest/min/max,
   tell the specialist to filter-back, not use LIMIT 1, so all tied rows are kept.
4. WHEN TO USE WHICH specialist:
   - SQL → joins, aggregations, filtering. Default choice.
   - Python → row-wise iteration, regex, multi-step transforms, complex pivots.
5. SUB-QUESTION QUALITY — be SPECIFIC about which columns to return. Bad:
   "what are the withdrawals?". Good: "Return only `trans_id` (single column) for
   all rows in `trans` where `account_id` belongs to client_id=3356 (resolve via
   `disp`) AND `type` = 'VYDAJ' AND `operation` = 'VYBER'."
6. AMBIGUOUS-NOUN INTERPRETATION (mandatory in step 1). If the question contains
   a noun whose meaning isn't pinned to one column (e.g. "type", "amount", "name",
   "rate"), include in the FIRST thought:
       "INTERPRET: '<noun>' refers to <specific column>, not <alternative>."
   Pass this interpretation to the specialist via the sub-question.
7. EXPLORATION FLOOR: complete at least 3 exploration steps (describe_data,
   read_doc, ask_sql with a sanity-check question) before any terminal call.
8. EXPLICIT COLUMN-COUNT VERIFICATION (mandatory). Before any terminal call, your
   `thought` MUST contain:
       "VERIFY: plan column_count=X, sub_question requests Y columns → {MATCH | VIOLATION}."
9. Always return exactly one JSON object with keys `thought`, `action`, `action_input`,
   wrapped in a single ```json fenced block, no surrounding text.

Keep reasoning concise. The specialists do the heavy lifting — your job is to plan
and dispatch precisely.
""".strip()


# RESPONSE_EXAMPLES intentionally use fictional domains (library / weather)
# that DO NOT overlap with the public 50-task evaluation set. The examples
# illustrate STRUCTURE (plan format, filter-back SQL, schema-inspect step),
# not specific content the agent should reuse verbatim.
RESPONSE_EXAMPLES = """
Example FIRST response — orientation step:
```json
{"thought":"Answer plan: column_count=1, per_column=[count of books published in target year]. Need to know what views exist before phrasing the SQL sub-question.","action":"describe_data","action_input":{}}
```

Example dispatching a precise SQL sub-question (with filter-back for superlative):
```json
{"thought":"VERIFY: plan column_count=2, sub_question requests 2 columns → MATCH. Question says 'highest temperature' — sub-question must instruct filter-back to keep ties.","action":"answer_from_sql","action_input":{"sub_question":"From the `weather` view, return rows where temp_celsius equals the maximum temp_celsius (filter-back, not LIMIT 1). Output exactly two columns: station_id, temp_celsius."}}
```

Example dispatching to Python specialist for row-wise transform:
```json
{"thought":"VERIFY: plan column_count=1, sub_question requests 1 column → MATCH. SQL is awkward for the regex parsing here, dispatching to Python.","action":"answer_from_python","action_input":{"sub_question":"Read `csv/notes.csv` with pandas. Use a regex to extract the year from the `note_text` column. Return a single column `year` with one row: the count of unique years."}}
```

Example finalizing with hand-computed literal (rare):
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
