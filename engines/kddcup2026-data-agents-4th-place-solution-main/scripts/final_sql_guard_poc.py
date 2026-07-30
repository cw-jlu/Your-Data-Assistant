"""PoC for Final SQL Guard variants on existing traces.

This script does not change the benchmark path. It scans existing run traces,
classifies high-risk final-SQL operations, and simulates conservative rewrites:

  - DISTINCT / COUNT(DISTINCT ...)
  - NULL / blank filters
  - Chinese finance full/legal entity display

The classifier sees question + final SQL + answer metadata only. It does not
receive gold answers or prior run scores.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from dotenv import load_dotenv

load_dotenv(REPO / ".env")

from kobushi_core.eval.csv_compare import EvaluationOptions, _evaluate_task
from kobushi_core.model import ModelMessage, OpenAIModelAdapter
from experiments.exp_150_projection_pruner.prefix_cache import with_prefix_cache_header
from experiments.exp_150_projection_pruner.tools.duckdb_unified import execute_sql


DATA_ROOT = REPO / "data" / "phase2_demo" / "demo_samples_phase2"
INPUT_ROOT = DATA_ROOT / "input"
GOLD_ROOT = DATA_ROOT / "output"
RUNS_ROOT = REPO / "artifacts" / "runs"

DISTINCT_RE = re.compile(r"\bDISTINCT\b", re.IGNORECASE)
NULL_FILTER_RE = re.compile(
    r"\bIS\s+NOT\s+NULL\b|(?:<>|!=)\s*(['\"])\s*\1",
    re.IGNORECASE,
)
ENTITY_RE = re.compile(
    r"\bChiName\b|\bAShareAbbr\b|\bChiNameAbbr\b|\bSecuAbbr\b|"
    r"[\u4e00-\u9fff]{2,}(?:股份有限公司|管理有限公司|有限责任公司|有限公司)"
)


DISTINCT_SYSTEM = """You classify SQL DISTINCT usage.

The SQL contains DISTINCT or COUNT(DISTINCT ...).
Do not solve the task.

Return exactly one token:
KEEP_DISTINCT
or
REMOVE_DISTINCT

Return KEEP_DISTINCT only if the question explicitly asks for distinct, unique,
different, non-duplicate, 去重, 唯一, 不同, or 不重复.

Return REMOVE_DISTINCT in all other cases.
Do not infer uniqueness from entity nouns, plural nouns, descriptor fields, or
count questions."""


NULL_SYSTEM = """You classify final SQL NULL/blank filtering.

The SQL contains IS NOT NULL, <> '', or != ''.
Do not solve the task.

Return exactly one token:
KEEP_NULL_FILTER
or
REMOVE_NULL_FILTER

Return KEEP_NULL_FILTER only if the question explicitly asks for non-null,
non-empty, known values, available values, excluding blanks, excluding missing,
有值, 非空, 不为空, 已知, or 排除空值.

Return REMOVE_NULL_FILTER in all other cases.
Do not infer that blanks or NULLs should be removed just because the question
asks to show, list, find, retrieve, or display data."""


ENTITY_SYSTEM = """You classify Chinese finance entity display form.

The final SQL or answer appears to use a full/legal entity name.
Do not solve the task.

Return exactly one token:
KEEP_ENTITY_DISPLAY
or
USE_SHORT_DISPLAY

Return KEEP_ENTITY_DISPLAY only if the question explicitly asks for full name,
legal name, registered name, company full name, 全称, 注册名称, 法定名称, or
完整名称.

Return USE_SHORT_DISPLAY in all other cases when the question asks which company,
which stock, which fund, fund company, name, 哪些公司, 哪只股票, 哪些基金, or
基金公司.

Do not use full/legal names merely because they appear in the source."""


def make_model() -> OpenAIModelAdapter:
    headers = with_prefix_cache_header(
        {
            "CF-Access-Client-Id": os.environ.get("CF_ACCESS_CLIENT_ID", ""),
            "CF-Access-Client-Secret": os.environ.get("CF_ACCESS_CLIENT_SECRET", ""),
        },
        "final-sql-guard-poc",
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
    while (RUNS_ROOT / f"final_sql_guard_poc_{i:03d}").exists():
        i += 1
    out = RUNS_ROOT / f"final_sql_guard_poc_{i:03d}"
    out.mkdir(parents=True)
    return out


def task_question(task_id: str) -> str:
    path = INPUT_ROOT / task_id / "task.json"
    return json.loads(path.read_text(encoding="utf-8"))["question"]


def load_trace(trace_path: Path) -> dict[str, Any]:
    data = json.loads(trace_path.read_text(encoding="utf-8"))
    if isinstance(data, list) and data:
        return data[-1]
    if isinstance(data, dict):
        return data
    return {}


def final_answer_sql(trace: dict[str, Any]) -> str:
    sql = ""
    for step in trace.get("steps", []) or []:
        if step.get("action") != "answer_from_sql":
            continue
        action_input = step.get("action_input") or {}
        if isinstance(action_input, dict) and action_input.get("sql"):
            sql = str(action_input["sql"])
    return sql


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


def clip(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def build_user_prompt(
    *,
    task_id: str,
    question: str,
    sql: str,
    preview: dict[str, Any],
    variant: str,
) -> str:
    return "\n".join(
        [
            f"Task: {task_id}",
            "",
            "Question:",
            question,
            "",
            "Final SQL:",
            clip(sql, 2400),
            "",
            "Final answer metadata:",
            json.dumps(preview, ensure_ascii=False),
            "",
            f"Classify only the {variant} final SQL operation.",
        ]
    )


def classify(
    model: OpenAIModelAdapter,
    *,
    system: str,
    user: str,
    allowed: set[str],
) -> str:
    raw = model.complete(
        [
            ModelMessage(role="system", content=system),
            ModelMessage(role="user", content=user),
        ],
        enable_thinking=False,
        max_tokens=32,
    ).strip()
    token = raw.split()[0].strip().upper() if raw.split() else ""
    return token if token in allowed else "UNKNOWN"


def write_csv(path: Path, columns: list[str], rows: list[list[Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for row in rows:
            writer.writerow(["" if v is None else v for v in row])


def score_csv(pred: Path, gold: Path, task_id: str) -> float | str:
    if not pred.exists() or not gold.exists():
        return "missing"
    try:
        ev = _evaluate_task(
            task_id=task_id,
            prediction_path=pred,
            gold_path=gold,
            options=EvaluationOptions(),
        )
        return float(ev.official_score_lambda_0_5)
    except Exception as exc:
        return f"error:{exc!r}"


def rewrite_distinct(sql: str) -> str:
    out = re.sub(r"(?is)\bSELECT\s+DISTINCT\s+", "SELECT ", sql)
    out = re.sub(r"(?is)COUNT\s*\(\s*DISTINCT\s+[^)]*\)", "COUNT(*)", out)
    return out


def _strip_redundant_where(sql: str) -> str:
    sql = re.sub(r"(?is)\bWHERE\s+AND\b", "WHERE", sql)
    sql = re.sub(r"(?is)\bWHERE\s*(ORDER\s+BY|GROUP\s+BY|HAVING|LIMIT)\b", r"\1", sql)
    sql = re.sub(r"(?is)\bWHERE\s*$", "", sql).strip()
    return sql


def rewrite_null_filter(sql: str) -> str:
    out = sql
    ident = r'(?:"[^"]+"|`[^`]+`|\[[^\]]+\]|[A-Za-z_][\w.]*)(?:\s*::\s*\w+)?'
    null_cond = rf"{ident}\s+IS\s+NOT\s+NULL"
    blank_cond = rf"{ident}\s*(?:<>|!=)\s*(['\"])\s*\1"
    cond = rf"(?:{null_cond}|{blank_cond})"
    out = re.sub(rf"(?is)\bWHERE\s+{cond}\s+AND\s+", "WHERE ", out)
    out = re.sub(rf"(?is)\s+AND\s+{cond}", "", out)
    out = re.sub(rf"(?is)\bWHERE\s+{cond}", "WHERE", out)
    return _strip_redundant_where(out)


def rewrite_entity_display(sql: str) -> str:
    out = sql
    # Prefer display/short columns when the query already exposes common full-name columns.
    out = re.sub(r'(?i)"ChiName"', '"ChiNameAbbr"', out)
    out = re.sub(r"(?i)\bChiName\b", "ChiNameAbbr", out)
    out = re.sub(r'(?i)"CompanyName"', '"ChiNameAbbr"', out)
    # Literal fund/company full-name suffixes: intentionally conservative.
    literal_suffixes = [
        "基金管理有限公司",
        "股份有限公司",
        "有限责任公司",
        "管理有限公司",
        "有限公司",
    ]
    for suffix in literal_suffixes:
        out = re.sub(
            rf"'([\u4e00-\u9fff]{{2,}}){re.escape(suffix)}'",
            r"'\1'",
            out,
        )
    return out


@dataclass(frozen=True)
class Case:
    run_id: str
    task_id: str
    task_dir: Path
    trace_path: Path
    sql: str


def discover_cases(run_dirs: list[Path], variants: set[str]) -> list[Case]:
    seen: set[tuple[str, str, str]] = set()
    cases: list[Case] = []
    for run_dir in run_dirs:
        if not run_dir.exists():
            continue
        for trace_path in sorted(run_dir.glob("task_*/trace.json")):
            trace = load_trace(trace_path)
            sql = final_answer_sql(trace)
            if not sql:
                continue
            task_id = trace_path.parent.name
            triggered = (
                ("distinct" in variants and DISTINCT_RE.search(sql))
                or ("null_filter" in variants and NULL_FILTER_RE.search(sql))
                or ("entity_display" in variants and ENTITY_RE.search(sql))
            )
            if not triggered:
                continue
            key = (run_dir.name, task_id, sql)
            if key in seen:
                continue
            seen.add(key)
            cases.append(
                Case(
                    run_id=run_dir.name,
                    task_id=task_id,
                    task_dir=trace_path.parent,
                    trace_path=trace_path,
                    sql=sql,
                )
            )
    return cases


def simulate_sql(
    *,
    case: Case,
    label: str,
    out_dir: Path,
) -> tuple[str | None, int | None, float | str | None, str | None]:
    if label == "REMOVE_DISTINCT":
        new_sql = rewrite_distinct(case.sql)
    elif label == "REMOVE_NULL_FILTER":
        new_sql = rewrite_null_filter(case.sql)
    elif label == "USE_SHORT_DISPLAY":
        new_sql = rewrite_entity_display(case.sql)
    else:
        return None, None, None, None

    if new_sql.strip() == case.sql.strip():
        return new_sql, None, None, "rewrite_noop"

    sim_dir = out_dir / case.run_id / case.task_id / label.lower()
    sim_dir.mkdir(parents=True, exist_ok=True)
    (sim_dir / "sql.sql").write_text(new_sql, encoding="utf-8")
    try:
        result = execute_sql(INPUT_ROOT / case.task_id / "context", new_sql, limit=None)
        columns = [str(c) for c in result.get("columns") or []]
        rows = result.get("rows") or []
        pred = sim_dir / "prediction.csv"
        write_csv(pred, columns, rows)
        score = score_csv(pred, GOLD_ROOT / case.task_id / "gold.csv", case.task_id)
        return new_sql, len(rows), score, None
    except Exception as exc:
        return new_sql, None, None, repr(exc)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--runs",
        nargs="+",
        default=[
            "exp_150_answer_shape_prose_video_keyframe_note_pdf_preprocess_source_router_anti_agg_anti_agg_sql_guard_007",
            "exp_150_answer_shape_prose_video_keyframe_note_pdf_preprocess_source_router_anti_agg_anti_agg_sql_guard_009",
            "exp_150_answer_shape_prose_video_keyframe_note_pdf_preprocess_source_router_anti_agg_anti_agg_sql_guard_010",
        ],
    )
    ap.add_argument("--max-cases", type=int, default=40)
    ap.add_argument(
        "--variants",
        nargs="+",
        choices=["distinct", "null_filter", "entity_display"],
        default=["distinct", "null_filter", "entity_display"],
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = next_out_dir()
    model = make_model()
    variants = set(args.variants)
    cases = discover_cases([RUNS_ROOT / r for r in args.runs], variants)[: args.max_cases]
    results: list[dict[str, Any]] = []

    print(f"=== Final SQL Guard PoC -> {out_dir} ===", flush=True)
    print(f"cases={len(cases)}", flush=True)

    for case in cases:
        question = task_question(case.task_id)
        preview = read_prediction_preview(case.task_dir)
        orig_score = score_csv(
            case.task_dir / "prediction.csv",
            GOLD_ROOT / case.task_id / "gold.csv",
            case.task_id,
        )
        labels: list[dict[str, Any]] = []
        triggers: list[tuple[str, str, str, set[str]]] = []
        if "distinct" in variants and DISTINCT_RE.search(case.sql):
            triggers.append(
                (
                    "distinct",
                    DISTINCT_SYSTEM,
                    "DISTINCT",
                    {"KEEP_DISTINCT", "REMOVE_DISTINCT"},
                )
            )
        if "null_filter" in variants and NULL_FILTER_RE.search(case.sql):
            triggers.append(
                (
                    "null_filter",
                    NULL_SYSTEM,
                    "NULL/blank filtering",
                    {"KEEP_NULL_FILTER", "REMOVE_NULL_FILTER"},
                )
            )
        if "entity_display" in variants and ENTITY_RE.search(case.sql):
            triggers.append(
                (
                    "entity_display",
                    ENTITY_SYSTEM,
                    "entity display",
                    {"KEEP_ENTITY_DISPLAY", "USE_SHORT_DISPLAY"},
                )
            )

        for variant, system, variant_label, allowed in triggers:
            user = build_user_prompt(
                task_id=case.task_id,
                question=question,
                sql=case.sql,
                preview=preview,
                variant=variant_label,
            )
            t0 = time.time()
            label = classify(model, system=system, user=user, allowed=allowed)
            new_sql, sim_rows, sim_score, error = simulate_sql(
                case=case, label=label, out_dir=out_dir
            )
            rec = {
                "variant": variant,
                "label": label,
                "elapsed": round(time.time() - t0, 3),
                "sim_rows": sim_rows,
                "sim_score": sim_score,
                "error": error,
                "sim_sql": new_sql,
            }
            labels.append(rec)
            sim_text = "-" if sim_score is None else f"{sim_score:.2f}" if isinstance(sim_score, float) else str(sim_score)
            orig_text = f"{orig_score:.2f}" if isinstance(orig_score, float) else str(orig_score)
            print(
                f"{case.run_id[-3:]} {case.task_id:<8} {variant:<14} "
                f"{label:<20} orig={orig_text:<6} sim={sim_text:<8} "
                f"rows={preview.get('row_count')}->{sim_rows} {error or ''}",
                flush=True,
            )

        results.append(
            {
                "run_id": case.run_id,
                "task_id": case.task_id,
                "question": question,
                "orig_score": orig_score,
                "orig_rows": preview.get("row_count"),
                "sql": case.sql,
                "labels": labels,
            }
        )

    payload = {
        "prompts": {
            "distinct": DISTINCT_SYSTEM,
            "null_filter": NULL_SYSTEM,
            "entity_display": ENTITY_SYSTEM,
        },
        "runs": args.runs,
        "variants": args.variants,
        "results": results,
    }
    (out_dir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
