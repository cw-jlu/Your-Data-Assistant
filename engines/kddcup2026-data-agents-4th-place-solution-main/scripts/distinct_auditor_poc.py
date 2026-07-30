"""PoC: classify final SQL only when DISTINCT appears.

This script reads existing benchmark traces, extracts the last answer_from_sql
SQL for each task, and calls a lightweight sub-agent only for SQL containing
DISTINCT / COUNT(DISTINCT ...). It does not modify benchmark code.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from dotenv import load_dotenv

load_dotenv(REPO / ".env")

from kobushi_core.model import ModelMessage, OpenAIModelAdapter
from experiments.exp_150_projection_pruner.prefix_cache import with_prefix_cache_header


DATA_ROOT = REPO / "data" / "phase2_demo" / "demo_samples_phase2"
RUNS_ROOT = REPO / "artifacts" / "runs"
DISTINCT_RE = re.compile(r"\bDISTINCT\b", re.IGNORECASE)


SYSTEM_PROMPT = """You classify SQL DISTINCT usage.

The SQL contains DISTINCT or COUNT(DISTINCT ...).
Do not solve the task.

Return exactly one token:
KEEP_DISTINCT
or
REMOVE_DISTINCT

Return KEEP_DISTINCT only if the question explicitly asks for distinct, unique,
different, non-duplicate, 去重, 唯一, 不同, or 不重复.

Return REMOVE_DISTINCT in all other cases.
Do not infer uniqueness from words like company, shareholder, manager, patient,
person, stock, fund, name, which, list, or how many.
"How many companies/people/patients/stocks/funds" is REMOVE_DISTINCT unless the
question explicitly says distinct/unique/different."""


ZH_SYSTEM_PROMPT = """你分类 SQL 中 DISTINCT 的使用。

SQL 包含 DISTINCT 或 COUNT(DISTINCT ...)。
不要解题。

只输出一个 token:
KEEP_DISTINCT
或
REMOVE_DISTINCT

只有问题明确要求 distinct, unique, different, non-duplicate, 去重, 唯一,
不同, 不重复 时，才输出 KEEP_DISTINCT。

其他情况都输出 REMOVE_DISTINCT。
不要从 公司、股东、基金经理、患者、人、股票、基金、名称、哪些、列出、
多少 等词推断需要去重。
“多少家公司/多少人/多少患者/多少只股票/多少基金”也是 REMOVE_DISTINCT，
除非问题明确说 distinct/unique/different/去重/唯一/不同/不重复。"""


def make_model() -> OpenAIModelAdapter:
    headers = with_prefix_cache_header(
        {
            "CF-Access-Client-Id": os.environ.get("CF_ACCESS_CLIENT_ID", ""),
            "CF-Access-Client-Secret": os.environ.get("CF_ACCESS_CLIENT_SECRET", ""),
        },
        "distinct-auditor-poc",
    )
    return OpenAIModelAdapter(
        model=os.environ.get("AGENT_MODEL", "qwen3.5-35b-a3b"),
        api_base=os.environ.get("AGENT_API_BASE", ""),
        api_key=os.environ.get("AGENT_API_KEY", ""),
        temperature=0.0,
        extra_headers=headers,
    )


def load_trace(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list) and data:
        return data[-1]
    if isinstance(data, dict):
        return data
    return {}


def task_question(task_id: str) -> str:
    path = DATA_ROOT / "input" / task_id / "task.json"
    return json.loads(path.read_text(encoding="utf-8"))["question"]


def final_answer_sql(trace: dict[str, Any]) -> str:
    sql = ""
    for step in trace.get("steps", []) or []:
        if step.get("action") != "answer_from_sql":
            continue
        action_input = step.get("action_input") or {}
        if isinstance(action_input, dict) and action_input.get("sql"):
            sql = str(action_input["sql"])
    return sql


def clip(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


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


def distinct_trace_snippets(trace: dict[str, Any], max_chars: int = 3000) -> str:
    chunks: list[str] = []
    for step in trace.get("steps", []) or []:
        action_input = step.get("action_input") or {}
        sql = ""
        if isinstance(action_input, dict):
            sql = str(action_input.get("sql") or "")
        thought = str(step.get("thought") or "")
        obs = str(step.get("observation_content_preview") or "")
        haystack = f"{sql}\n{thought}\n{obs}"
        if not re.search(r"distinct|duplicate|fan.?out|row_count|COUNT\(", haystack, re.I):
            continue
        chunks.append(
            "\n".join(
                [
                    f"[step {step.get('i', '?')} action={step.get('action')}]",
                    f"thought: {clip(thought, 450)}",
                    f"sql: {clip(sql, 650)}",
                    f"observation: {clip(obs, 900)}",
                ]
            )
        )
    return clip("\n\n".join(chunks), max_chars)


def use_chinese_prompt(question: str, sql: str) -> bool:
    haystack = f"{question}\n{sql}"
    cjk = len(re.findall(r"[\u4e00-\u9fff]", haystack))
    kana = len(re.findall(r"[\u3040-\u30ff]", haystack))
    return cjk >= 4 and kana == 0


def build_user_prompt(
    *,
    run_id: str,
    task_id: str,
    question: str,
    sql: str,
    preview: dict[str, Any],
    snippets: str,
) -> str:
    return "\n".join(
        [
            f"Run: {run_id}",
            f"Task: {task_id}",
            "",
            "Question:",
            question,
            "",
            "Final SQL containing DISTINCT:",
            clip(sql, 2200),
            "",
            "Final answer metadata:",
            json.dumps(preview, ensure_ascii=False),
            "",
            "Relevant trace snippets:",
            snippets or "(none)",
            "",
            "Classify DISTINCT only.",
        ]
    )


def label_from_raw(raw: str) -> str:
    upper = raw.strip().upper()
    if upper.startswith("KEEP_DISTINCT"):
        return "KEEP_DISTINCT"
    if upper.startswith("REMOVE_DISTINCT"):
        return "REMOVE_DISTINCT"
    return "UNKNOWN"


def score_map(run_dir: Path) -> dict[str, float]:
    path = run_dir / "summary.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {r["tid"]: float(r.get("score", 0.0)) for r in data.get("results", [])}


def discover_cases(run_dirs: list[Path], max_cases: int | None) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        scores = score_map(run_dir)
        for trace_path in sorted(run_dir.glob("task_*/trace.json")):
            task_id = trace_path.parent.name
            trace = load_trace(trace_path)
            sql = final_answer_sql(trace)
            if not DISTINCT_RE.search(sql):
                continue
            cases.append(
                {
                    "run_id": run_dir.name,
                    "task_id": task_id,
                    "trace_path": str(trace_path),
                    "task_dir": trace_path.parent,
                    "score": scores.get(task_id),
                    "sql": sql,
                    "trace": trace,
                }
            )
            if max_cases is not None and len(cases) >= max_cases:
                return cases
    return cases


def next_out_dir() -> Path:
    i = 1
    while (RUNS_ROOT / f"distinct_classifier_poc_{i:03d}").exists():
        i += 1
    out = RUNS_ROOT / f"distinct_classifier_poc_{i:03d}"
    out.mkdir(parents=True)
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--runs",
        nargs="+",
        default=[
            "exp_150_answer_shape_prose_video_keyframe_note_pdf_preprocess_source_router_anti_agg_anti_agg_sql_guard_009",
            "exp_150_answer_shape_prose_video_keyframe_note_pdf_preprocess_source_router_anti_agg_anti_agg_sql_guard_007",
        ],
        help="Run directory names under artifacts/runs.",
    )
    ap.add_argument("--max-cases", type=int, default=12)
    ap.add_argument("--max-tokens", type=int, default=192)
    ap.add_argument(
        "--with-trace",
        action="store_true",
        help="Include trace snippets. Default is question + final SQL only.",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    run_dirs = [RUNS_ROOT / r for r in args.runs]
    cases = discover_cases(run_dirs, args.max_cases)
    out = next_out_dir()
    model = make_model()
    results: list[dict[str, Any]] = []
    print(f"=== distinct classifier PoC -> {out} ===", flush=True)
    print(f"cases={len(cases)}", flush=True)
    for case in cases:
        task_id = case["task_id"]
        run_id = case["run_id"]
        question = task_question(task_id)
        sql = case["sql"]
        lang = "zh" if use_chinese_prompt(question, sql) else "en"
        system = ZH_SYSTEM_PROMPT if lang == "zh" else SYSTEM_PROMPT
        preview = read_prediction_preview(Path(case["task_dir"]))
        snippets = distinct_trace_snippets(case["trace"]) if args.with_trace else ""
        user = build_user_prompt(
            run_id=run_id,
            task_id=task_id,
            question=question,
            sql=sql,
            preview=preview,
            snippets=snippets,
        )
        t0 = time.time()
        raw = model.complete(
            [
                ModelMessage(role="system", content=system),
                ModelMessage(role="user", content=user),
            ],
            enable_thinking=False,
            max_tokens=args.max_tokens,
        ).strip()
        item = {
            "run_id": run_id,
            "task_id": task_id,
            "score": case["score"],
            "label": label_from_raw(raw),
            "raw": raw,
            "prompt_lang": lang,
            "sql": sql,
            "row_count": preview.get("row_count"),
            "elapsed": round(time.time() - t0, 3),
        }
        results.append(item)
        print(
            f"{run_id[-3:]} {task_id} score={case['score']} "
            f"rows={preview.get('row_count')} -> {item['label']} | {raw}",
            flush=True,
        )
    payload = {
        "system_prompt": SYSTEM_PROMPT,
        "zh_system_prompt": ZH_SYSTEM_PROMPT,
        "runs": args.runs,
        "results": results,
    }
    (out / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
