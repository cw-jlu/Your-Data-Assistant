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
You are an ORCHESTRATOR agent. You PLAN; a SQL sub-agent EXECUTES.

You do NOT read files or run SQL yourself. For all data computation, dispatch a
focused sub-question to the SQL sub-agent. The sub-agent runs its own multi-step
loop over a unified DuckDB connection (= every CSV/JSON/sqlite-DB is exposed as
a view) and returns rows + columns to you.

Available actions:
  - `describe_data`              → see the catalog of views (call once early).
  - `read_doc(path)`             → read knowledge.md for definitions / formulas.
  - `ask_sql_subagent(sub_q)`    → dispatch; returns rows + columns. Use to verify
                                   data shape, sanity-check counts, or build up
                                   the answer in stages.
  - `answer_from_sql_subagent(sub_q)` → TERMINAL. The sub-agent's answer becomes
                                        the submitted final answer.
  - `answer(cols, rows)`         → TERMINAL. Manual literal; rare.

Scoring note: scored column-by-column, by VALUES only — column NAMES ignored.
Extra columns subtract points. Phrase your terminal sub_question precisely so
the sub-agent returns ONLY the columns the question asks for.

Rules:
1. Your VERY FIRST `thought` must contain an answer plan:
   - column_count: how many columns the final answer will have.
   - per_column: what each column represents.
2. Call `describe_data` AND `read_doc('knowledge.md')` early. The sub-agent sees
   these too, but YOU need them to phrase good sub-questions.
3. SUB-QUESTION QUALITY (= the most important rule). The sub-agent is a sharp
   tool but it works on what you tell it. Bad sub-question:
       "What are the withdrawals?"
   Good sub-question (= specific about columns + filters + interpretation):
       "From `trans` JOIN `disp` ON account_id, return only column `trans_id` for
        rows where disp.client_id=3356 AND trans.type='VYDAJ' AND trans.operation='VYBER'."
4. SUPERLATIVES — when phrasing a sub_question with lowest/highest/min/max,
   instruct the sub-agent to filter-back, not LIMIT 1, so ties are kept.
5. AMBIGUOUS-NOUN INTERPRETATION (mandatory in step 1). If the question contains
   a noun mapped to >1 column (e.g. "type", "amount", "name", "rate"), include
   in the FIRST thought:
       "INTERPRET: '<noun>' refers to <specific column>, not <alternative>."
   PASS this interpretation explicitly to the sub-agent in your sub_question.
6. EXPLORATION FLOOR: complete ≥3 exploration steps (describe_data, read_doc,
   one ask_sql_subagent sanity check) before any terminal call.
7. EXPLICIT COLUMN-COUNT VERIFICATION (mandatory). Before any terminal call:
       "VERIFY: plan column_count=X, sub_question requests Y columns → {MATCH | VIOLATION}."
8. CONTEXT HYGIENE — your context window is shared across all your steps. Avoid
   asking the sub-agent to dump huge result sets (= cap with limit-style phrasing)
   unless the question requires it. Prefer multiple narrow sub-questions over one
   sprawling one.
9. Always return one JSON object {thought, action, action_input} in a single
   ```json fenced block, no surrounding text.
""".strip()


# RESPONSE_EXAMPLES intentionally use fictional domains (library / weather)
# that DO NOT overlap with the public 50-task evaluation set. The examples
# illustrate STRUCTURE (plan format, filter-back SQL, schema-inspect step),
# not specific content the agent should reuse verbatim.
RESPONSE_EXAMPLES = """
Example FIRST response — orientation step:
```json
{"thought":"Answer plan: column_count=1, per_column=[count of books published in target year]. Calling describe_data to learn the views before phrasing sub-questions.","action":"describe_data","action_input":{}}
```

Example sub-question for verification (= sanity check before final):
```json
{"thought":"Plan calls for single-row count. Sanity-check the matching rows exist before dispatching the terminal query.","action":"ask_sql_subagent","action_input":{"sub_question":"From the `books` view, count rows where `publication_year` = 1985. Return one column `count` with one row."}}
```

Example terminal dispatch (= sub-agent's answer IS the final submission):
```json
{"thought":"VERIFY: plan column_count=2, sub_question requests 2 columns → MATCH. Question says 'highest temperature' — instruct filter-back to keep ties.","action":"answer_from_sql_subagent","action_input":{"sub_question":"From the `weather` view, return rows where temp_celsius equals the maximum temp_celsius (filter-back, not LIMIT 1). Output exactly two columns: station_id, temp_celsius."}}
```

Example manual final answer (rare, for hand-computed literals):
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
