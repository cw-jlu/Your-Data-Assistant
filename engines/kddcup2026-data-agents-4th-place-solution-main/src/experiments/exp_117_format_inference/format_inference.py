"""Phase 0: Format inference (= ReFoRCE's get_format_prompt analogue).

Before any SQL generation, a small LLM call estimates the canonical answer
format — i.e., a CSV header showing the expected output column(s). This
hint is then passed to the self_refine prompt as `format_csv` so the
generation model knows what columns the question is asking for.

Modeled on ReFoRCE prompt.py:76-81. Uses domain-neutral examples (= weather,
products, library) that do not appear in the DABench public-task set.
"""
from __future__ import annotations

import re
from kobushi_core.benchmark.schema import PublicTask
from kobushi_core.model import ModelAdapter, ModelMessage

from experiments.exp_117_format_inference.tools.duckdb_unified import describe_catalog


# ============================================================================
# Format-inference prompt (= ReFoRCE get_format_prompt at prompt.py:76-81)
# ============================================================================
# Verbatim transplant of ReFoRCE's design intent. Examples replaced with
# neutral domains (= NOT in our public 50-task set) for leak hygiene.
FORMAT_INFERENCE_SYSTEM = """\
You estimate the canonical answer format for an SQL task. Given the task
question and the available view schemas, output the SIMPLEST possible
answer format as a CSV header line.

Output as a single ```csv fenced block with just the header (no rows).

Examples (from neutral domains — do not reuse these names):
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
  column. If a name column exists, prefer name; otherwise use id.
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


def build_format_inference_user(task: PublicTask, catalog: str) -> str:
    return (
        f"## Task\n{task.question}\n\n"
        f"## DuckDB views available\n{catalog}\n\n"
        "Output ONE ```csv block with just the canonical answer header."
    )


def _extract_csv_header(raw: str) -> str:
    """Pull the CSV header from a fenced csv block."""
    m = re.search(r"```csv\s*(.*?)\s*```", raw, flags=re.IGNORECASE | re.DOTALL)
    if m:
        first_line = m.group(1).strip().split("\n")[0].strip()
        return first_line
    # fallback: any fenced block
    m = re.search(r"```\s*(.*?)\s*```", raw, flags=re.DOTALL)
    if m:
        return m.group(1).strip().split("\n")[0].strip()
    # last resort: use first non-empty line
    for line in raw.split("\n"):
        if line.strip() and not line.lower().startswith("here") and "," in line or line.strip().isidentifier():
            return line.strip()
    return ""


def run_format_inference(*, task: PublicTask, model: ModelAdapter) -> dict:
    """Single LLM call. Returns {format_csv: str, raw: str}."""
    catalog = describe_catalog(task.context_dir)
    user = build_format_inference_user(task, catalog)
    raw = model.complete([
        ModelMessage(role="system", content=FORMAT_INFERENCE_SYSTEM),
        ModelMessage(role="user", content=user),
    ])
    header = _extract_csv_header(raw)
    return {"format_csv": header, "raw_response": raw}
