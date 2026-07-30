#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

try:
    from dotenv import load_dotenv

    load_dotenv(REPO / ".env")
except Exception:
    pass

from kobushi_core.eval.csv_compare import EvaluationOptions, _evaluate_task  # noqa: E402
from kobushi_core.model import ModelMessage, OpenAIModelAdapter  # noqa: E402

try:
    from experiments.exp_155_phase_tool_visibility.prefix_cache import (  # noqa: E402
        with_prefix_cache_header,
    )
except Exception:  # pragma: no cover

    def with_prefix_cache_header(headers: dict[str, str], task_id: str | None = None) -> dict[str, str]:
        del task_id
        return headers


DATA_ROOT = REPO / "data" / "phase2_demo" / "demo_samples_phase2"
INPUT_ROOT = DATA_ROOT / "input"
GOLD_ROOT = DATA_ROOT / "output"
RUN_ROOT = REPO / "artifacts" / "runs"
OUT_ROOT = REPO / "artifacts" / "finance_output_column_selector_poc"

FINANCE_TASKS = {
    # Exact public BULL matches.
    "task_2",
    "task_3",
    "task_8",
    "task_10",
    "task_13",
    "task_14",
    "task_15",
    "task_18",
    "task_19",
    "task_32",
    "task_33",
    "task_38",
    "task_40",
    "task_43",
    "task_45",
    "task_46",
    "task_47",
    "task_49",
    "task_50",
    "task_51",
    "task_53",
    "task_58",
    "task_59",
    # BULL-derived non-exact/demo video tasks.
    "task_1",
    "task_4",
    "task_6",
    "task_7",
    "task_11",
    "task_17",
    "task_20",
    "task_21",
    "task_22",
    "task_23",
    "task_24",
    "task_25",
    "task_26",
    "task_27",
    "task_29",
    "task_30",
    "task_31",
    "task_34",
    "task_35",
    "task_36",
    "task_37",
    "task_39",
    "task_41",
    "task_42",
    "task_52",
    "task_54",
    "task_55",
    "task_56",
    "task_57",
    "task_60",
}


SYSTEM_PROMPT = """
You select final answer columns from a candidate column list.

Input:
- Question
- Candidate final answer columns

Task:
- Choose only the candidate columns that should appear in the final CSV answer.
- Do not write SQL.
- Do not compute values.
- Do not add columns that are not in the candidate list.

Selection policy:
- Select only columns explicitly requested as final answer fields.
- Do not select columns used only for filtering, sorting, thresholds, joins,
  intermediate calculations, evidence, dates, periods, IDs, or codes unless the
  question asks to output them.
- If the question asks for metric data/records, select the requested metric
  columns only.
- If the question asks which year/date/period a value belongs to, select the
  year/date/period column and the value column.
- Time-range wording such as "these years", "over the years", "after 2005",
  "since inception", "这些年", or "以后" is usually a filter/range, not a
  request to output the date/period column.
- If the question asks for a distribution or grouped count, select the group
  label and the count.
- If the question asks for an entity, select the user-facing name/label. Select
  code or full/legal name only when explicitly requested.

Return JSON only:
{"columns": ["candidate column names to keep"]}

Examples:

Question:
王工说给他查一下510210的单位基金净值周增速数据记录
Candidate final answer columns:
SecuCode, NVWeeklyGrowthRate, EndDate
Answer:
{"columns":["NVWeeklyGrowthRate"]}

Question:
帮我看看在其他存款性公司资产负债表中，哪一年的总负债最大，达到了多少
Candidate final answer columns:
EndDate, TotalLiabilities
Answer:
{"columns":["EndDate","TotalLiabilities"]}

Question:
2021年收入从大到小排名前5的是哪几家公司
Candidate final answer columns:
ChiNameAbbr, MainOperIncome, EndDate
Answer:
{"columns":["ChiNameAbbr"]}

Question:
深科技的代码和公司简称是啥
Candidate final answer columns:
AStockCode, AShareAbbr
Answer:
{"columns":["AStockCode","AShareAbbr"]}

Question:
Which companies are ranked in the top 5 in terms of income in 2021?
Candidate final answer columns:
ChiNameAbbr, MainOperIncome, EndDate
Answer:
{"columns":["ChiNameAbbr"]}

Question:
What are the code and company abbreviation of DeepTech?
Candidate final answer columns:
AStockCode, AShareAbbr
Answer:
{"columns":["AStockCode","AShareAbbr"]}

Question:
Number of fund managers serving more than 5 funds, grouped by fund managers' highest educational level.
Candidate final answer columns:
Education, count(*), PersonalCode
Answer:
{"columns":["Education","count(*)"]}

Question:
Do you know the record of the rate of return for each fund in the past month?
Candidate final answer columns:
RRInSingleMonth, SecuAbbr, EndDate
Answer:
{"columns":["RRInSingleMonth"]}

Question:
你知道各基金一个月以来回报率记录是什么哦
Candidate final answer columns:
RRInSingleMonth, SecuAbbr, EndDate
Answer:
{"columns":["RRInSingleMonth"]}

Question:
麻烦查询一下我国历年人均国内生产总值的数据记录
Candidate final answer columns:
EndDate, GDPPerCapita
Answer:
{"columns":["GDPPerCapita"]}

Question:
What is the total amount of deposits and total amount of loans in the RMB credit balances of financial institutions in our country over the years?
Candidate final answer columns:
EndDate, TotalSavings, TotalLoans
Answer:
{"columns":["TotalSavings","TotalLoans"]}
""".strip()


@dataclass(frozen=True)
class CsvData:
    header: list[str]
    rows: list[list[str]]


def _task_sort_key(task_id: str) -> int:
    try:
        return int(task_id.split("_", 1)[1])
    except Exception:
        return 10**9


def _read_csv(path: Path) -> CsvData:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        return CsvData([], [])
    return CsvData(rows[0], rows[1:])


def _write_csv(path: Path, data: CsvData, keep_columns: list[str]) -> None:
    index_by_name = {name: i for i, name in enumerate(data.header)}
    indexes = [index_by_name[name] for name in keep_columns if name in index_by_name]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([data.header[i] for i in indexes])
        for row in data.rows:
            writer.writerow([row[i] if i < len(row) else "" for i in indexes])


def _load_question(task_id: str) -> str:
    path = INPUT_ROOT / task_id / "task.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return str(payload["question"])


def _make_model(task_id: str) -> OpenAIModelAdapter:
    api_base = os.environ.get("AGENT_API_BASE") or os.environ.get("MODEL_API_URL") or ""
    api_key = os.environ.get("AGENT_API_KEY") or os.environ.get("MODEL_API_KEY") or ""
    model_name = os.environ.get("COLUMN_SELECTOR_MODEL") or os.environ.get("AGENT_MODEL") or "qwen3.5-35b-a3b"
    headers = with_prefix_cache_header(
        {
            "CF-Access-Client-Id": os.environ.get("CF_ACCESS_CLIENT_ID", ""),
            "CF-Access-Client-Secret": os.environ.get("CF_ACCESS_CLIENT_SECRET", ""),
        },
        f"{task_id}:column-selector",
    )
    return OpenAIModelAdapter(
        model=model_name,
        api_base=api_base,
        api_key=api_key,
        temperature=0.0,
        extra_headers=headers,
        enable_thinking=False,
        max_tokens=512,
    )


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    try:
        value = json.loads(stripped)
        if isinstance(value, dict):
            return value
    except Exception:
        pass
    match = re.search(r"\{.*\}", stripped, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object in response: {text!r}")
    value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError(f"JSON response is not object: {text!r}")
    return value


def _normalize_selected(raw_columns: Any, candidates: list[str]) -> list[str]:
    if not isinstance(raw_columns, list):
        return list(candidates)
    exact = [str(item) for item in raw_columns if str(item) in candidates]
    if exact:
        return exact
    lower_map = {col.lower(): col for col in candidates}
    out: list[str] = []
    for item in raw_columns:
        col = lower_map.get(str(item).lower())
        if col and col not in out:
            out.append(col)
    return out or list(candidates)


def _select_columns(
    *,
    task_id: str,
    question: str,
    candidates: list[str],
    use_model: bool,
) -> tuple[list[str], str, str]:
    if not use_model:
        return list(candidates), "", ""
    user = (
        f"Question:\n{question}\n\n"
        "Candidate final answer columns:\n"
        f"{', '.join(candidates)}\n\n"
        "Answer:"
    )
    raw = _make_model(task_id).complete(
        [
            ModelMessage(role="system", content=SYSTEM_PROMPT),
            ModelMessage(role="user", content=user),
        ],
        enable_thinking=False,
        max_tokens=512,
    )
    payload = _extract_json_object(raw)
    selected = _normalize_selected(payload.get("columns"), candidates)
    return selected, json.dumps(payload, ensure_ascii=False), raw


def _parse_tasks(raw: str | None, run_dir: Path, *, only_extra: bool, max_tasks: int | None) -> list[str]:
    if raw:
        tasks = [part.strip() for part in raw.split(",") if part.strip()]
    else:
        tasks = sorted(
            [
                path.parent.name
                for path in run_dir.glob("task_*/prediction.csv")
                if path.parent.name in FINANCE_TASKS
            ],
            key=_task_sort_key,
        )
    if only_extra:
        filtered: list[str] = []
        for task_id in tasks:
            pred = run_dir / task_id / "prediction.csv"
            gold = GOLD_ROOT / task_id / "gold.csv"
            if not pred.is_file() or not gold.is_file():
                continue
            if len(_read_csv(pred).header) > len(_read_csv(gold).header):
                filtered.append(task_id)
        tasks = filtered
    if max_tasks is not None:
        tasks = tasks[:max_tasks]
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Review real trace/prediction candidate columns with a finance output-column selector."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--tasks", default=None, help="comma-separated task ids")
    parser.add_argument("--only-extra", action="store_true", help="only tasks where prediction has more columns than gold")
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--no-model", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=OUT_ROOT)
    args = parser.parse_args()

    run_dir = args.run_dir
    if not run_dir.is_absolute():
        run_dir = REPO / run_dir
    task_ids = _parse_tasks(args.tasks, run_dir, only_extra=args.only_extra, max_tasks=args.max_tasks)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    projected_root = args.out_dir / f"{run_dir.name}_projected"
    projected_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for task_id in task_ids:
        pred_path = run_dir / task_id / "prediction.csv"
        gold_path = GOLD_ROOT / task_id / "gold.csv"
        if not pred_path.is_file() or not gold_path.is_file():
            continue
        question = _load_question(task_id)
        pred = _read_csv(pred_path)
        gold = _read_csv(gold_path)
        original_eval = _evaluate_task(
            task_id=task_id,
            prediction_path=pred_path,
            gold_path=gold_path,
            options=EvaluationOptions(),
        )
        selected: list[str]
        payload = ""
        raw = ""
        error = ""
        try:
            selected, payload, raw = _select_columns(
                task_id=task_id,
                question=question,
                candidates=pred.header,
                use_model=not args.no_model,
            )
        except Exception as exc:
            selected = list(pred.header)
            error = repr(exc)
        task_out = projected_root / task_id
        task_out.mkdir(parents=True, exist_ok=True)
        projected_path = task_out / "prediction.csv"
        _write_csv(projected_path, pred, selected)
        projected_eval = _evaluate_task(
            task_id=task_id,
            prediction_path=projected_path,
            gold_path=gold_path,
            options=EvaluationOptions(),
        )
        rows.append(
            {
                "task_id": task_id,
                "question": question,
                "candidate_columns": " | ".join(pred.header),
                "selected_columns": " | ".join(selected),
                "gold_columns": " | ".join(gold.header),
                "original_score": float(original_eval.official_score_lambda_0_5),
                "projected_score": float(projected_eval.official_score_lambda_0_5),
                "delta": float(projected_eval.official_score_lambda_0_5)
                - float(original_eval.official_score_lambda_0_5),
                "original_diagnosis": original_eval.diagnosis,
                "projected_diagnosis": projected_eval.diagnosis,
                "original_shape": f"{original_eval.rows_pred}x{original_eval.cols_pred}",
                "projected_shape": f"{projected_eval.rows_pred}x{projected_eval.cols_pred}",
                "gold_shape": f"{original_eval.rows_gold}x{original_eval.cols_gold}",
                "payload": payload,
                "raw": raw,
                "error": error,
            }
        )
        mark = "UP" if rows[-1]["delta"] > 0 else "DOWN" if rows[-1]["delta"] < 0 else "FLAT"
        print(
            f"{mark:<4} {task_id:<8} {rows[-1]['original_score']:.3f}->{rows[-1]['projected_score']:.3f} "
            f"cand=[{rows[-1]['candidate_columns']}] selected=[{rows[-1]['selected_columns']}]",
            flush=True,
        )

    csv_path = args.out_dir / f"{run_dir.name}_selector_poc.csv"
    json_path = args.out_dir / f"{run_dir.name}_selector_poc.json"
    fieldnames = [
        "task_id",
        "question",
        "candidate_columns",
        "selected_columns",
        "gold_columns",
        "original_score",
        "projected_score",
        "delta",
        "original_diagnosis",
        "projected_diagnosis",
        "original_shape",
        "projected_shape",
        "gold_shape",
        "payload",
        "raw",
        "error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if rows:
        base = sum(float(row["original_score"]) for row in rows) / len(rows)
        projected = sum(float(row["projected_score"]) for row in rows) / len(rows)
        print(f"\nmean: {base:.4f} -> {projected:.4f} delta={projected - base:+.4f} n={len(rows)}")
        print(f"improved={sum(1 for r in rows if r['delta'] > 0)} degraded={sum(1 for r in rows if r['delta'] < 0)}")
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
