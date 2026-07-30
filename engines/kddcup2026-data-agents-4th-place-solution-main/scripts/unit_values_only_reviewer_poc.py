"""Minimal unit consistency reviewer PoC.

Input is intentionally minimal:
  - question
  - final SQL
  - final answer columns
  - final answer value preview

No source snippets, gold answers, historical scores, or task-specific examples
are provided.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from dotenv import load_dotenv

load_dotenv(REPO / ".env")

from kobushi_core.model import ModelMessage, OpenAIModelAdapter
from experiments.exp_150_projection_pruner.prefix_cache import with_prefix_cache_header


DATA_ROOT = REPO / "data" / "phase2_demo" / "demo_samples_phase2"
RUNS_ROOT = REPO / "artifacts" / "runs"

SYSTEM_PROMPT = """You are a unit consistency reviewer.

Look only at the question, final SQL, final answer column names, and final
answer values.
For each column, check whether the values appear to use a unit, scale, suffix,
or rounding format that conflicts with the question or column name.

Do not use source snippets, gold answers, prior runs, or task-specific knowledge.
Do not judge row selection, filters, metric choice, source choice, or whether
the answer is correct.
Do not rewrite values.

For text columns, say no unit check needed.
Do not flag harmless trailing zeros or high decimal precision.
Do not require unit symbols inside cells when the unit is clear from the column
name.
If both common numeric representations are plausible from the question and
column name alone, say no clear unit issue.

Return plain text only, one bullet per final answer column:
- <column name>: <not numeric / no clear unit issue / possible unit issue>
  reason: <short reason>"""


DEFAULT_CASES = [
    "exp_149_answer_shape_prose_video_keyframe_note_pdf_preprocess_source_router_anti_agg_anti_agg_sql_guard_003:task_15",
    "exp_149_answer_shape_prose_video_keyframe_note_pdf_preprocess_source_router_anti_agg_anti_agg_sql_guard_007:task_15",
    "exp_149_answer_shape_prose_video_keyframe_note_pdf_preprocess_source_router_anti_agg_anti_agg_sql_guard_007:task_22",
    "exp_149_answer_shape_prose_video_keyframe_note_pdf_preprocess_source_router_anti_agg_anti_agg_sql_guard_007:task_23",
    "exp_149_answer_shape_prose_video_keyframe_note_pdf_preprocess_source_router_anti_agg_anti_agg_sql_guard_007:task_29",
    "exp_149_answer_shape_prose_video_keyframe_note_pdf_preprocess_source_router_anti_agg_anti_agg_sql_guard_007:task_30",
    "exp_149_answer_shape_prose_video_keyframe_note_pdf_preprocess_source_router_anti_agg_anti_agg_sql_guard_003:task_39",
    "exp_149_answer_shape_prose_video_keyframe_note_pdf_preprocess_source_router_anti_agg_anti_agg_sql_guard_007:task_39",
    "exp_149_answer_shape_prose_video_keyframe_note_pdf_preprocess_source_router_anti_agg_anti_agg_sql_guard_007:task_53",
    "exp_149_answer_shape_prose_video_keyframe_note_pdf_preprocess_source_router_anti_agg_anti_agg_sql_guard_007:task_54",
    "exp_149_answer_shape_prose_video_keyframe_note_pdf_preprocess_source_router_anti_agg_anti_agg_sql_guard_007:task_49",
]


def make_model() -> OpenAIModelAdapter:
    headers = with_prefix_cache_header(
        {
            "CF-Access-Client-Id": os.environ.get("CF_ACCESS_CLIENT_ID", ""),
            "CF-Access-Client-Secret": os.environ.get("CF_ACCESS_CLIENT_SECRET", ""),
        },
        "unit-values-only-reviewer-poc",
    )
    return OpenAIModelAdapter(
        model=os.environ.get("AGENT_MODEL", "qwen3.5-35b-a3b"),
        api_base=os.environ.get("AGENT_API_BASE", ""),
        api_key=os.environ.get("AGENT_API_KEY", ""),
        temperature=0.2,
        extra_headers=headers,
    )


def read_csv(path: Path) -> list[list[str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.reader(f))


def task_question(task_id: str) -> str:
    path = DATA_ROOT / "input" / task_id / "task.json"
    return json.loads(path.read_text(encoding="utf-8"))["question"]


def load_trace(trace_path: Path) -> dict:
    data = json.loads(trace_path.read_text(encoding="utf-8"))
    if isinstance(data, list) and data:
        return data[-1]
    if isinstance(data, dict):
        return data
    return {}


def last_answer_sql(trace: dict) -> str:
    sql = ""
    for step in trace.get("steps", []):
        if step.get("action") == "answer_from_sql":
            action_input = step.get("action_input") or {}
            if isinstance(action_input, dict):
                sql = str(action_input.get("sql") or sql)
    return sql


def clip(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def build_user_prompt(
    *,
    question: str,
    sql: str,
    rows: list[list[str]],
    max_rows: int,
    max_sql_chars: int,
) -> str:
    columns = rows[0] if rows else []
    preview = rows[1 : 1 + max_rows]
    return (
        "Question:\n"
        f"{question}\n\n"
        "Final SQL:\n"
        f"{clip(sql or '(not available)', max_sql_chars)}\n\n"
        "Final answer columns:\n"
        f"{json.dumps(columns, ensure_ascii=False)}\n\n"
        "Final answer preview:\n"
        f"{json.dumps(preview, ensure_ascii=False)}\n\n"
        "Review unit consistency column by column using only this information."
    )


def next_out_dir() -> Path:
    root = RUNS_ROOT
    i = 1
    while (root / f"unit_values_only_reviewer_poc_{i:03d}").exists():
        i += 1
    out = root / f"unit_values_only_reviewer_poc_{i:03d}"
    out.mkdir(parents=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default=",".join(DEFAULT_CASES))
    parser.add_argument("--max-preview-rows", type=int, default=5)
    parser.add_argument("--max-sql-chars", type=int, default=2400)
    parser.add_argument("--max-tokens", type=int, default=500)
    args = parser.parse_args()

    cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    out = next_out_dir()
    model = make_model()
    results: list[dict[str, object]] = []

    print(f"=== unit values-only reviewer PoC -> {out} ===", flush=True)
    for case in cases:
        run_id, task_id = case.split(":", 1)
        run_dir = RUNS_ROOT / run_id / task_id
        rows = read_csv(run_dir / "prediction.csv")
        trace = load_trace(run_dir / "trace.json")
        prompt = build_user_prompt(
            question=task_question(task_id),
            sql=last_answer_sql(trace),
            rows=rows,
            max_rows=args.max_preview_rows,
            max_sql_chars=args.max_sql_chars,
        )
        response = model.complete(
            [
                ModelMessage(role="system", content=SYSTEM_PROMPT),
                ModelMessage(role="user", content=prompt),
            ],
            enable_thinking=False,
            max_tokens=args.max_tokens,
        ).strip()
        results.append(
            {
                "case": case,
                "response": response,
                "columns": rows[0] if rows else [],
                "preview": rows[1 : 1 + args.max_preview_rows],
                "prompt": prompt,
            }
        )
        print(f"{case} -> {response.replace(chr(10), ' | ')}", flush=True)

    (out / "system_prompt.txt").write_text(SYSTEM_PROMPT, encoding="utf-8")
    (out / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = ["# Unit Values-Only Reviewer PoC", ""]
    for result in results:
        lines.append(f"## {result['case']}")
        lines.append("")
        lines.append(str(result["response"]))
        lines.append("")
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report: {out / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
