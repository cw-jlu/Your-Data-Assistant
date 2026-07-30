"""Column-wise unit consistency reviewer PoC.

This script compares two prompt inputs:
  1. question + final SQL + final answer preview
  2. the same input plus source snippets from the trace

No gold answers, historical scores, or task-specific examples are provided.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
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

Review each final answer column independently.
Check only whether that column's values appear to have a numeric unit, scale,
suffix, or rounding problem.

Do not judge row selection, filters, columns, metric choice, SQL correctness,
or source choice.
Do not use gold answers, prior runs, or task-specific knowledge.
Do not rewrite the answer.

For non-numeric or text columns, say no unit check is needed.
Do not flag harmless formatting such as trailing zeros.
Do not flag high decimal precision unless the question explicitly asks for
rounding and the final answer ignores it.
CSV preview values may appear as strings; do not flag that.
Do not require unit symbols inside cells when the unit appears in the column
name.
If source evidence shows a value with a unit suffix, the final cell may either
keep that displayed number with the unit made clear by the column name, or use a
converted numeric scale. Only call it a possible issue when the scale is clearly
unsafe or ambiguous for that column.
Do not infer a unit conversion problem just because the question and column use
different wording or language for the same unit.
If there is no clear evidence of a unit or numeric-format problem, say no unit
issue found.

Return plain text only, one bullet per final answer column:
- <column name>: <not numeric / no unit issue found / possible unit issue>
  reason: <short reason>"""


DEFAULT_CASES = [
    "exp_149_answer_shape_prose_video_keyframe_note_pdf_preprocess_source_router_anti_agg_anti_agg_sql_guard_003:task_15",
    "exp_149_answer_shape_prose_video_keyframe_note_pdf_preprocess_source_router_anti_agg_anti_agg_sql_guard_007:task_15",
    "exp_149_answer_shape_prose_video_keyframe_note_pdf_preprocess_source_router_anti_agg_anti_agg_sql_guard_007:task_22",
    "exp_149_answer_shape_prose_video_keyframe_note_pdf_preprocess_source_router_anti_agg_anti_agg_sql_guard_007:task_23",
    "exp_149_answer_shape_prose_video_keyframe_note_pdf_preprocess_source_router_anti_agg_anti_agg_sql_guard_007:task_30",
    "exp_149_answer_shape_prose_video_keyframe_note_pdf_preprocess_source_router_anti_agg_anti_agg_sql_guard_003:task_39",
    "exp_149_answer_shape_prose_video_keyframe_note_pdf_preprocess_source_router_anti_agg_anti_agg_sql_guard_007:task_39",
    "exp_149_answer_shape_prose_video_keyframe_note_pdf_preprocess_source_router_anti_agg_anti_agg_sql_guard_007:task_53",
    "exp_149_answer_shape_prose_video_keyframe_note_pdf_preprocess_source_router_anti_agg_anti_agg_sql_guard_007:task_54",
    "exp_149_answer_shape_prose_video_keyframe_note_pdf_preprocess_source_router_anti_agg_anti_agg_sql_guard_007:task_49",
]

SOURCE_ACTIONS = {
    "grep",
    "read_doc",
    "read_csv",
    "read_json",
    "execute_sql",
    "watch_video",
}

NUMERIC_OR_UNIT_RE = re.compile(
    r"\\d|%|％|percent|ratio|rate|amount|unit|million|yuan|"
    r"比例|比率|占比|率|金额|规模|元|万|亿|股|只",
    re.IGNORECASE,
)


def make_model() -> OpenAIModelAdapter:
    headers = with_prefix_cache_header(
        {
            "CF-Access-Client-Id": os.environ.get("CF_ACCESS_CLIENT_ID", ""),
            "CF-Access-Client-Secret": os.environ.get("CF_ACCESS_CLIENT_SECRET", ""),
        },
        "unit-column-reviewer-poc",
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


def source_snippets(trace: dict, *, max_chars: int) -> str:
    chunks: list[str] = []
    for step in trace.get("steps", []):
        action = str(step.get("action") or "")
        if action not in SOURCE_ACTIONS:
            continue
        obs = str(step.get("observation_content_preview") or "")
        if not obs or not NUMERIC_OR_UNIT_RE.search(obs):
            continue
        thought = str(step.get("thought") or "")
        chunk = (
            f"[{action} step {step.get('i', '?')}]\n"
            f"thought: {clip(thought, 350)}\n"
            f"observation: {clip(obs, 1200)}"
        )
        chunks.append(chunk)

    text = "\n\n".join(chunks)
    return clip(text, max_chars)


def build_user_prompt(
    *,
    question: str,
    sql: str,
    rows: list[list[str]],
    evidence: str,
    max_rows: int,
    max_sql_chars: int,
) -> str:
    columns = rows[0] if rows else []
    preview = rows[1 : 1 + max_rows]
    parts = [
        "Question:",
        question,
        "",
        "Final SQL:",
        clip(sql or "(not available)", max_sql_chars),
        "",
        "Final answer columns:",
        json.dumps(columns, ensure_ascii=False),
        "",
        "Final answer preview:",
        json.dumps(preview, ensure_ascii=False),
    ]
    if evidence:
        parts.extend(["", "Source snippets from this task trace:", evidence])
    parts.extend(["", "Review unit consistency column by column only."])
    return "\n".join(parts)


def next_out_dir() -> Path:
    root = RUNS_ROOT
    i = 1
    while (root / f"unit_column_reviewer_poc_{i:03d}").exists():
        i += 1
    out = root / f"unit_column_reviewer_poc_{i:03d}"
    out.mkdir(parents=True)
    return out


def run_case(
    *,
    model: OpenAIModelAdapter,
    case: str,
    with_evidence: bool,
    max_preview_rows: int,
    max_sql_chars: int,
    max_evidence_chars: int,
    max_tokens: int,
) -> dict[str, object]:
    run_id, task_id = case.split(":", 1)
    run_dir = RUNS_ROOT / run_id / task_id
    rows = read_csv(run_dir / "prediction.csv")
    trace = load_trace(run_dir / "trace.json")
    evidence = source_snippets(trace, max_chars=max_evidence_chars) if with_evidence else ""
    prompt = build_user_prompt(
        question=task_question(task_id),
        sql=last_answer_sql(trace),
        rows=rows,
        evidence=evidence,
        max_rows=max_preview_rows,
        max_sql_chars=max_sql_chars,
    )
    response = model.complete(
        [
            ModelMessage(role="system", content=SYSTEM_PROMPT),
            ModelMessage(role="user", content=prompt),
        ],
        enable_thinking=False,
        max_tokens=max_tokens,
    ).strip()
    return {
        "case": case,
        "mode": "with_evidence" if with_evidence else "no_evidence",
        "response": response,
        "columns": rows[0] if rows else [],
        "preview": rows[1 : 1 + max_preview_rows],
        "source_snippets": evidence,
        "prompt": prompt,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default=",".join(DEFAULT_CASES))
    parser.add_argument("--max-preview-rows", type=int, default=5)
    parser.add_argument("--max-sql-chars", type=int, default=2400)
    parser.add_argument("--max-evidence-chars", type=int, default=3200)
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--no-evidence-only", action="store_true")
    parser.add_argument("--with-evidence-only", action="store_true")
    args = parser.parse_args()

    cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    modes = [False, True]
    if args.no_evidence_only:
        modes = [False]
    if args.with_evidence_only:
        modes = [True]

    out = next_out_dir()
    model = make_model()
    results: list[dict[str, object]] = []
    print(f"=== unit column reviewer PoC -> {out} ===", flush=True)
    for case in cases:
        for with_evidence in modes:
            result = run_case(
                model=model,
                case=case,
                with_evidence=with_evidence,
                max_preview_rows=args.max_preview_rows,
                max_sql_chars=args.max_sql_chars,
                max_evidence_chars=args.max_evidence_chars,
                max_tokens=args.max_tokens,
            )
            results.append(result)
            one_line = str(result["response"]).replace("\n", " | ")
            print(f"{case} [{result['mode']}] -> {one_line}", flush=True)

    (out / "system_prompt.txt").write_text(SYSTEM_PROMPT, encoding="utf-8")
    (out / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = ["# Unit Column Reviewer PoC", ""]
    for r in results:
        lines.append(f"## {r['case']} [{r['mode']}]")
        lines.append("")
        lines.append(str(r["response"]))
        lines.append("")
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report: {out / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
