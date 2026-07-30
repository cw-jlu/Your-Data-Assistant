"""System prompt and spec formatter for Phase 2: Answer Agent."""
from __future__ import annotations

import json
from typing import Any

ANSWER_SYSTEM_PROMPT = """
You are a DuckDB SQL execution agent (Phase 2 of 2).

You receive an exploration spec produced by Phase 1. Your ONLY job is to write and execute
one DuckDB SQL query using `answer_from_duckdb` to produce the final answer table.

Rules:
1. Read the spec in the preamble carefully. It tells you exactly which files to use,
   how to join them, what filters to apply, and what the answer columns mean.
2. Write a single DuckDB SQL query that implements the spec.
3. Use DuckDB's native functions for multi-source access:
   - CSV:    read_csv_auto('relative/path.csv')
   - JSON:   read_json_auto('relative/path.json', records=true)
   - SQLite: ATTACH 'relative/path.sqlite' AS db (TYPE sqlite); SELECT * FROM db.table
4. All paths are relative to the task context directory.
5. If your first SQL attempt returns an error, fix the SQL and retry (max 3 retries).
6. Do NOT hand-write an `answer` — use `answer_from_duckdb` exclusively.

Scoring note: extra columns cost points. Return ONLY the columns described in answer_columns.

Always return exactly one JSON object with keys `thought`, `action`, and `action_input`,
wrapped in a single ```json fenced block. No text before or after the block.
""".strip()

ANSWER_RESPONSE_EXAMPLE = """
Example response:
```json
{"thought":"Spec says: join members.json → major.csv on link_to_major=major_id, filter major_name='Physics Teaching', count members. I will use read_json_auto + read_csv_auto in a DuckDB WITH clause.","action":"answer_from_duckdb","action_input":{"sql":"WITH m AS (SELECT * FROM read_json_auto('json/members.json', records=true)), maj AS (SELECT * FROM read_csv_auto('csv/major.csv')) SELECT COUNT(*) AS count FROM m JOIN maj ON m.link_to_major = maj.major_id WHERE maj.major_name = 'Physics Teaching'"}}
```
""".strip()


def build_answer_system_prompt(tool_descriptions: str) -> str:
    return (
        f"{ANSWER_SYSTEM_PROMPT}\n\n"
        "Available tools:\n"
        f"{tool_descriptions}\n\n"
        f"{ANSWER_RESPONSE_EXAMPLE}\n\n"
        "You must always return a single ```json fenced block containing one JSON object "
        "with keys `thought`, `action`, and `action_input`, and no extra text."
    )


def build_spec_preamble(spec_data: dict[str, Any]) -> str:
    """Format the exploration spec as a preamble for the Answer Agent."""
    if not spec_data:
        return "Note: Phase 1 did not commit a spec. Infer the answer plan from the question."
    return (
        "## Exploration Spec (from Phase 1)\n\n"
        "Use this spec to write your DuckDB SQL query:\n\n"
        f"```json\n{json.dumps(spec_data, ensure_ascii=False, indent=2)}\n```"
    )
