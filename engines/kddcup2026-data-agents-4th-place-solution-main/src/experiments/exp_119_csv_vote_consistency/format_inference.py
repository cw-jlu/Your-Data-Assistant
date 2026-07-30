"""ReFoRCE-style format inference (= the CSV header pre-step).

Mirrors prompt.py:76-81 in references/ReFoRCE. Given the question + schema,
estimate the canonical answer header. The header becomes a hint passed to
the agent's preamble so its terminal SQL targets the right column shape.

Examples in the prompt are from neutral domains (= NOT in DABench's public
set) for leak hygiene.
"""
from __future__ import annotations

import re

from kobushi_core.benchmark.schema import PublicTask
from kobushi_core.model import ModelAdapter, ModelMessage

from experiments.exp_119_csv_vote_consistency.tools.duckdb_unified import describe_catalog


FORMAT_INFERENCE_SYSTEM = """\
You estimate the canonical answer format for an SQL task. Given the task
question and the available view schemas, output the SIMPLEST possible
answer format as a CSV header line.

Output as a single ```csv fenced block with just the header (no rows).

Examples (from neutral domains — do NOT reuse these names):
- Task: "Including the travel coordinates and the cumulative travel distance
  at each waypoint."
  Format:
  ```csv
  travel_coordinates,cumulative_travel_distance
  ```
- Task: "Which products had a seasonality-adjusted sales ratio above 0.8?"
  When the task says "which X" without specifying name or id, provide BOTH
  name and id columns:
  Format:
  ```csv
  product_name,product_id
  ```
- Task: "How many books were published in the target year?"
  When the task asks "how many / count / total", use ONE numeric column:
  Format:
  ```csv
  count
  ```
- Task: "List all weather stations in zone Z."
  When the task says "list all <entity>", use the entity's most identifying
  column. Prefer the name column if a name exists; otherwise use the id.
  Format:
  ```csv
  station_name
  ```
- Task: "What is the average rainfall in the dry season?"
  When the task asks "what is the X", use ONE column for X:
  Format:
  ```csv
  average_rainfall
  ```

Do NOT output any SQL queries. Do NOT include data rows. Just the header.
"""


def _extract_csv_header(raw: str) -> str:
    m = re.search(r"```csv\s*(.*?)\s*```", raw, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip().split("\n")[0].strip()
    m = re.search(r"```\s*(.*?)\s*```", raw, flags=re.DOTALL)
    if m:
        return m.group(1).strip().split("\n")[0].strip()
    return raw.strip().split("\n")[0].strip()


def run_format_inference(*, task: PublicTask, model: ModelAdapter) -> str:
    catalog = describe_catalog(task.context_dir)
    user = (
        f"## Task\n{task.question}\n\n"
        f"## DuckDB views available\n{catalog}\n\n"
        "Output ONE ```csv block with just the canonical answer header."
    )
    raw = model.complete([
        ModelMessage(role="system", content=FORMAT_INFERENCE_SYSTEM),
        ModelMessage(role="user", content=user),
    ])
    return _extract_csv_header(raw)
