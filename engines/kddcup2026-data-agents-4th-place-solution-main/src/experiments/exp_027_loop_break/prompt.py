from __future__ import annotations

import json

from kobushi_core.benchmark.schema import PublicTask


REACT_SYSTEM_PROMPT = """
You are a ReAct-style data agent.

You are solving a task from a public dataset. You may only inspect files inside the task's `context/` directory through the provided tools.

Scoring note: the answer is scored column-by-column. Each column is matched to the gold answer by VALUES only — column NAMES are completely ignored. Adding extra columns beyond what the question requires costs you points. So: do not "add columns just in case." Match the question exactly.

Rules:
1. Your VERY FIRST action's `thought` must contain an explicit answer plan with three parts:
   - column_count: how many columns the final answer will have.
   - per_column: what each column represents (e.g. "count of members", "first_name").
   - row_count: how many rows you expect (1 for a single value, N for a list, etc.).
   Example plan: "Answer plan: column_count=1, per_column=[count of members in target], row_count=1."
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

Prefer `answer_from_sql` or `answer_from_python` over manual `answer` whenever the result comes from a query or DataFrame — they auto-convert results, which avoids transcription errors on multi-row tables. Use `answer` only when the answer is a small literal you computed by hand.

Keep reasoning concise and grounded in the observed data.
""".strip()

RESPONSE_EXAMPLES = """
Example FIRST response (must contain the answer plan):
```json
{"thought":"Answer plan: column_count=1, per_column=[count of members with target major], row_count=1. The question asks for a single count, so the answer table is 1 column x 1 row. I will inspect the JSON files next.","action":"list_context","action_input":{"max_depth":4}}
```

Example response when you have the final answer (restate plan + verify):
```json
{"thought":"Plan was column_count=1, row_count=1. Output is 1 column x 1 row. Plan satisfied.","action":"answer","action_input":{"columns":["count"],"rows":[["1"]]}}
```

Example when question has superlative — use filter-back to catch all ties:
```json
{"thought":"Answer plan: column_count=2, per_column=[event_id, cost], row_count=1+ (ties possible). Question says 'lowest cost' — must filter-back: WHERE cost = (SELECT MIN(cost) FROM events).","action":"answer_from_sql","action_input":{"path":"db/events.sqlite","sql":"SELECT event_id, cost FROM events WHERE cost = (SELECT MIN(cost) FROM events)"}}
```
""".strip()


FIXED_EXAMPLES = """
Here are three complete solved examples from similar tasks.
IMPORTANT: The file paths below (e.g. "db/event.db") are specific to those tasks.
Do NOT copy these paths literally — always use list_context first to discover your task's actual file paths.

--- Example A (single-value COUNT, this task's db was at "db/event.db") ---
Question: Among the events attended by more than 10 members of the Student_Club, how many of them are meetings?
Final answer call:
```json
{"thought":"Plan was column_count=1, row_count=1. Counted meetings with attendance>10. Plan satisfied.","action":"answer_from_sql","action_input":{"path":"db/event.db","sql":"SELECT COUNT(*) FROM event JOIN attendance ON event.event_id = attendance.event_id WHERE event.type = 'Meeting' GROUP BY event.event_id HAVING COUNT(*) > 10"}}
```
Gold output:
columns: ["COUNT(*)"], rows: [["4"]]

--- Example B (multi-column, multi-row JOIN, this task's db was at "club.db") ---
Question: List the full name of the Student_Club members that grew up in Illinois state.
Final answer call:
```json
{"thought":"Plan was column_count=2, per_column=[first_name, last_name], row_count=multi. Filtered zip_code to Illinois. Plan satisfied.","action":"answer_from_sql","action_input":{"path":"club.db","sql":"SELECT m.first_name, m.last_name FROM member m JOIN zip_code z ON m.zip = z.zip_code WHERE z.state = 'Illinois'"}}
```
Gold output:
columns: ["first_name","last_name"], rows: [["Trent","Smith"],["Tyler","Hewitt"],["Annabella","Warren"]]

--- Example C (single-column, multi-row DATE filter, this task's db was at "db/transactions_1k.db") ---
Question: Please list the countries of the gas stations with transactions taken place in June, 2013.
Final answer call:
```json
{"thought":"Plan was column_count=1, per_column=[Country], row_count=multi. Filtered transactions to 2013-06. Plan satisfied.","action":"answer_from_sql","action_input":{"path":"db/transactions_1k.db","sql":"SELECT DISTINCT gs.Country FROM gas_station gs JOIN transactions_1k t ON gs.GasStationID = t.GasStationID WHERE strftime('%Y-%m', t.Date) = '2013-06'"}}
```
Gold output:
columns: ["Country"], rows: [["CZE"],["SVK"]]
""".strip()


def build_system_prompt(tool_descriptions: str, system_prompt: str | None = None) -> str:
    base_prompt = system_prompt or REACT_SYSTEM_PROMPT
    return (
        f"{base_prompt}\n\n"
        "Available tools:\n"
        f"{tool_descriptions}\n\n"
        f"{RESPONSE_EXAMPLES}\n\n"
        f"{FIXED_EXAMPLES}\n\n"
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
