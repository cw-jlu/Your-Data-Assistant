"""PoC for source-router v2: classify table/doc/video roles independently.

This script is offline with respect to the benchmark path: it reads existing
task traces and asks a small sub-agent to classify each source as
primary/secondary/no. It does not modify agent behavior or use gold answers.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from dotenv import load_dotenv

load_dotenv(REPO / ".env")

from kobushi_core.model import ModelMessage, OpenAIModelAdapter
from experiments.exp_152_final_sql_guard.prefix_cache import with_prefix_cache_header


DATA_ROOT = REPO / "data" / "phase2_demo" / "demo_samples_phase2"
INPUT_ROOT = DATA_ROOT / "input"
RUNS_ROOT = REPO / "artifacts" / "runs"

SYSTEM_PROMPT = """You classify evidence roles for a data-answering task.

Classify each source independently:
- table
- doc
- video

For each source, choose exactly one role:
primary, secondary, or no.

Definitions:
primary = the source provides the final answer values, rows, counts, ranking, or literals.
secondary = the source provides filters, thresholds, selected options, date scope, entity IDs, schema hints, or validation only.
no = the source is not needed for the answer.

Important:
A video/dashboard may show filters, thresholds, selected options, or partial preview rows.
That is secondary evidence, not primary final-answer evidence.
Mark video as primary only when it fully displays every requested output field after all question filters.
Video is NOT primary when it shows bins/distributions/partial rows, even if a subtotal can be calculated.
Video is NOT primary when any requested output field is missing, or when it says to reconstruct/query.
Video is NOT primary for a filtered count if its subtotal includes buckets outside the filter.
For counts from buckets, verify every included bucket satisfies the displayed filter; excluded/below-threshold buckets cannot be counted.
Phrases like "only some rows", "do not display", "hidden", or "missing" mean requested fields are absent.
If a displayed number conflicts with shown conditions or the question, do not treat it as primary.
If another source must enumerate rows, resolve names, or compute with filters, that source is primary.
For computed counts/lists over records, prefer the enumerating/computing table or document as primary.
If video supplies only part of the requested answer, mark the source supplying the missing fields as primary.
Multiple sources may be primary.

Return ONLY JSON:
{
  "table": "primary|secondary|no",
  "doc": "primary|secondary|no",
  "video": "primary|secondary|no",
  "final_answer_source": ["table"|"doc"|"video"],
  "reason": "one short sentence"
}
"""

ROLES = {"primary", "secondary", "no"}
SOURCES = {"table", "doc", "video"}


def make_model() -> OpenAIModelAdapter:
    headers = with_prefix_cache_header(
        {
            "CF-Access-Client-Id": os.environ.get("CF_ACCESS_CLIENT_ID", ""),
            "CF-Access-Client-Secret": os.environ.get("CF_ACCESS_CLIENT_SECRET", ""),
        },
        "source-router-v2-poc",
    )
    return OpenAIModelAdapter(
        model=os.environ.get("AGENT_MODEL", "qwen3.5-35b-a3b"),
        api_base=os.environ.get("AGENT_API_BASE", ""),
        api_key=os.environ.get("AGENT_API_KEY", ""),
        temperature=0.0,
        extra_headers=headers,
    )


def next_out_dir() -> Path:
    i = 1
    while (RUNS_ROOT / f"source_router_v2_poc_{i:03d}").exists():
        i += 1
    out = RUNS_ROOT / f"source_router_v2_poc_{i:03d}"
    out.mkdir(parents=True)
    return out


def task_question(task_id: str) -> str:
    return json.loads((INPUT_ROOT / task_id / "task.json").read_text(encoding="utf-8"))["question"]


def load_trace(task_dir: Path) -> dict[str, Any]:
    path = task_dir / "trace.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list) and data:
        return data[-1]
    if isinstance(data, dict):
        return data
    return {}


def clip(text: Any, max_chars: int = 1000) -> str:
    value = str(text or "").strip().replace("\n", " | ")
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3].rstrip() + "..."


def read_prediction_preview(task_dir: Path, max_rows: int = 5) -> dict[str, Any]:
    path = task_dir / "prediction.csv"
    if not path.exists():
        return {"columns": [], "row_count": None, "preview": []}
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return {"columns": [], "row_count": 0, "preview": []}
    return {
        "columns": rows[0],
        "row_count": max(0, len(rows) - 1),
        "preview": rows[1 : 1 + max_rows],
    }


def read_video_note(task_dir: Path) -> str:
    path = task_dir / "video_keyframe_note.md"
    if not path.exists():
        return ""
    return clip(path.read_text(encoding="utf-8", errors="replace"), 2200)


def final_answer_sql(trace: dict[str, Any]) -> str:
    sql = ""
    for step in trace.get("steps", []) or []:
        if step.get("action") != "answer_from_sql":
            continue
        action_input = step.get("action_input") or {}
        if isinstance(action_input, dict) and action_input.get("sql"):
            sql = str(action_input["sql"])
    return sql


def last_v1_source_route(trace: dict[str, Any]) -> dict[str, Any] | None:
    route = None
    for step in trace.get("steps", []) or []:
        action_input = step.get("action_input") or {}
        if isinstance(action_input, dict) and isinstance(action_input.get("_source_route"), dict):
            route = action_input["_source_route"]
    return route


def tool_counts(trace: dict[str, Any]) -> dict[str, int]:
    return dict(Counter(step.get("action") for step in trace.get("steps", []) or []))


def evidence_items(trace: dict[str, Any], max_items: int = 14) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for step in trace.get("steps", []) or []:
        action = step.get("action")
        if action not in {"describe_data", "execute_sql", "grep", "read_doc", "list_context"}:
            continue
        item = {
            "action": str(action),
            "observation": clip(step.get("observation_content_preview"), 1000),
        }
        action_input = step.get("action_input")
        if isinstance(action_input, dict) and action_input:
            item["input"] = clip(json.dumps(action_input, ensure_ascii=False), 700)
        items.append(item)
    if len(items) <= max_items:
        return items
    return items[:3] + items[-(max_items - 3) :]


def build_user_payload(task_id: str, task_dir: Path) -> dict[str, Any]:
    trace = load_trace(task_dir)
    payload: dict[str, Any] = {
        "question": task_question(task_id),
        "proposed_final_sql": clip(final_answer_sql(trace), 3000),
        "tool_counts": tool_counts(trace),
        "evidence": evidence_items(trace),
    }
    video_note = read_video_note(task_dir)
    if video_note:
        payload["video_keyframe_note"] = video_note
    failure = trace.get("failure_reason")
    if failure:
        payload["failure_reason"] = clip(failure, 500)
    return payload


def parse_result(raw: str) -> dict[str, Any]:
    try:
        data = json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            raise ValueError(f"no JSON object in response: {raw[:200]}")
        data = json.loads(m.group(0))
    if not isinstance(data, dict):
        raise ValueError("router response is not an object")
    for key in ("table", "doc", "video"):
        if data.get(key) not in ROLES:
            raise ValueError(f"bad role for {key}: {data.get(key)!r}")
    sources = data.get("final_answer_source")
    if not isinstance(sources, list) or not all(src in SOURCES for src in sources):
        raise ValueError(f"bad final_answer_source: {sources!r}")
    if not data.get("reason"):
        raise ValueError("missing reason")
    return {
        "table": data["table"],
        "doc": data["doc"],
        "video": data["video"],
        "final_answer_source": sources,
        "reason": clip(data["reason"], 500),
    }


def classify(model: OpenAIModelAdapter, payload: dict[str, Any]) -> tuple[dict[str, Any], str]:
    raw = model.complete(
        [
            ModelMessage(role="system", content=SYSTEM_PROMPT),
            ModelMessage(role="user", content=json.dumps(payload, ensure_ascii=False)),
        ],
        enable_thinking=False,
        max_tokens=300,
    )
    return parse_result(raw), raw


def default_tasks(run_dir: Path) -> list[str]:
    preferred = [
        "task_22", "task_26", "task_34", "task_40", "task_48", "task_52",
        "task_7", "task_8", "task_11", "task_17", "task_41", "task_45",
    ]
    return [task_id for task_id in preferred if (run_dir / task_id / "trace.json").exists()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        default="artifacts/runs/exp_152_answer_shape_prose_video_keyframe_note_pdf_preprocess_source_router_anti_agg_anti_agg_sql_guard_final_sql_guard_002",
        help="Run directory containing task_*/trace.json",
    )
    parser.add_argument("--tasks", default="", help="Comma-separated task ids. Default: targeted PoC set.")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        run_dir = REPO / run_dir
    if not run_dir.exists():
        raise SystemExit(f"run directory not found: {run_dir}")

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()] if args.tasks else default_tasks(run_dir)
    if args.limit:
        tasks = tasks[: args.limit]
    if not tasks:
        raise SystemExit("no tasks selected")

    out_dir = next_out_dir()
    model = make_model()
    rows: list[dict[str, Any]] = []
    for i, task_id in enumerate(tasks, 1):
        task_dir = run_dir / task_id
        payload = build_user_payload(task_id, task_dir)
        started = time.time()
        error = None
        raw = ""
        result: dict[str, Any] | None = None
        try:
            result, raw = classify(model, payload)
        except Exception as exc:
            error = repr(exc)
        row = {
            "task_id": task_id,
            "result": result,
            "error": error,
            "raw": raw,
            "elapsed": time.time() - started,
            "payload": payload,
        }
        rows.append(row)
        if result:
            print(
                f"[{i:02d}/{len(tasks)}] {task_id} "
                f"table={result['table']} doc={result['doc']} video={result['video']} "
                f"final={','.join(result['final_answer_source'])} :: {result['reason']}"
            )
        else:
            print(f"[{i:02d}/{len(tasks)}] {task_id} ERROR {error}")

    (out_dir / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = [
        {
            "task_id": row["task_id"],
            **(row["result"] or {}),
            "error": row["error"],
            "elapsed": row["elapsed"],
        }
        for row in rows
    ]
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {out_dir}")


if __name__ == "__main__":
    main()
