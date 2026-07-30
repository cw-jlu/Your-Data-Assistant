"""PoC for a lightweight unit/numeric-format auditor.

The auditor sees only question + final SQL + final answer preview. It does not
receive gold answers, historical scores, or task-specific examples.
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

SYSTEM_PROMPT = """You are a numeric unit auditor.

Check only whether the final answer appears to have an unsafe numeric unit,
scale, suffix, or rounding change.

Do not judge whether the rows, filters, columns, metric, or source are correct.
Do not use gold answers, prior runs, or task-specific knowledge.
Do not rewrite the answer.

Be conservative. If the issue is not clearly about numeric unit or formatting,
return NOT_UNIT_ISSUE.

Do not flag harmless decimal formatting such as trailing zeros.
Do not require unit symbols inside answer cells when the unit appears in a
column name. Inspect cell values and SQL expressions.
Flag only clear value-changing transformations, such as unsafe scaling,
added/removed cell suffixes, or rounding that loses requested precision.

Return exactly:

LABEL: <OK | UNIT_SUSPECT | ROUNDING_SUSPECT | SUFFIX_SUSPECT | NOT_UNIT_ISSUE>
REASON: <one short sentence>"""


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
    "exp_149_answer_shape_prose_video_keyframe_note_pdf_preprocess_source_router_anti_agg_anti_agg_sql_guard_007:task_1",
    "exp_149_answer_shape_prose_video_keyframe_note_pdf_preprocess_source_router_anti_agg_anti_agg_sql_guard_007:task_49",
]


def make_model() -> OpenAIModelAdapter:
    headers = with_prefix_cache_header(
        {
            "CF-Access-Client-Id": os.environ.get("CF_ACCESS_CLIENT_ID", ""),
            "CF-Access-Client-Secret": os.environ.get("CF_ACCESS_CLIENT_SECRET", ""),
        },
        "unit-normalizer-auditor-poc",
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


def last_answer_sql(trace_path: Path) -> str:
    if not trace_path.exists():
        return ""
    data = json.loads(trace_path.read_text(encoding="utf-8"))
    if isinstance(data, list) and data:
        data = data[-1]
    steps = data.get("steps", []) if isinstance(data, dict) else []
    sql = ""
    for step in steps:
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


def build_user_prompt(question: str, sql: str, rows: list[list[str]], *, max_rows: int, max_sql_chars: int) -> str:
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
        "Audit only numeric unit, scale, suffix, and rounding."
    )


def parse_label(text: str) -> str:
    for line in text.splitlines():
        if line.upper().startswith("LABEL:"):
            return line.split(":", 1)[1].strip()
    return "PARSE_ERROR"


def next_out_dir() -> Path:
    root = RUNS_ROOT
    i = 1
    while (root / f"unit_normalizer_auditor_poc_{i:03d}").exists():
        i += 1
    out = root / f"unit_normalizer_auditor_poc_{i:03d}"
    out.mkdir(parents=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default=",".join(DEFAULT_CASES))
    parser.add_argument("--max-preview-rows", type=int, default=5)
    parser.add_argument("--max-sql-chars", type=int, default=2400)
    parser.add_argument("--max-tokens", type=int, default=200)
    args = parser.parse_args()

    cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    out = next_out_dir()
    model = make_model()
    results = []

    print(f"=== unit normalizer auditor PoC -> {out} ===", flush=True)
    for case in cases:
        run_id, task_id = case.split(":", 1)
        run_dir = RUNS_ROOT / run_id / task_id
        pred_path = run_dir / "prediction.csv"
        trace_path = run_dir / "trace.json"
        rows = read_csv(pred_path)
        question = task_question(task_id)
        sql = last_answer_sql(trace_path)
        user_prompt = build_user_prompt(
            question,
            sql,
            rows,
            max_rows=args.max_preview_rows,
            max_sql_chars=args.max_sql_chars,
        )
        response = model.complete(
            [
                ModelMessage(role="system", content=SYSTEM_PROMPT),
                ModelMessage(role="user", content=user_prompt),
            ],
            enable_thinking=False,
            max_tokens=args.max_tokens,
        ).strip()
        label = parse_label(response)
        result = {
            "case": case,
            "run_id": run_id,
            "task_id": task_id,
            "label": label,
            "response": response,
            "columns": rows[0] if rows else [],
            "preview": rows[1 : 1 + args.max_preview_rows],
            "sql_preview": clip(sql, args.max_sql_chars),
        }
        results.append(result)
        print(f"{case} -> {label} :: {response.replace(chr(10), ' | ')}", flush=True)

    (out / "system_prompt.txt").write_text(SYSTEM_PROMPT, encoding="utf-8")
    (out / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = ["# Unit Normalizer Auditor PoC", ""]
    for r in results:
        lines.append(f"## {r['case']}")
        lines.append("")
        lines.append(r["response"])
        lines.append("")
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report: {out / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
