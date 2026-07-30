#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
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

from experiments.exp_156_column_advisor_poc.context_profile import (  # noqa: E402
    build_context_profile,
)
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
OUT_ROOT = REPO / "artifacts" / "finance_output_shape_from_data_poc"

FINANCE_TASKS = {
    "task_1",
    "task_2",
    "task_3",
    "task_4",
    "task_6",
    "task_7",
    "task_8",
    "task_10",
    "task_11",
    "task_13",
    "task_14",
    "task_15",
    "task_17",
    "task_18",
    "task_19",
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
    "task_32",
    "task_33",
    "task_34",
    "task_35",
    "task_36",
    "task_37",
    "task_38",
    "task_39",
    "task_40",
    "task_41",
    "task_42",
    "task_43",
    "task_45",
    "task_46",
    "task_47",
    "task_49",
    "task_50",
    "task_51",
    "task_52",
    "task_53",
    "task_54",
    "task_55",
    "task_56",
    "task_57",
    "task_58",
    "task_59",
    "task_60",
}


SYSTEM_PROMPT = """
You decide the final CSV output columns for a finance data question.

Input:
- Question
- Data profile with available tables, columns, and small samples

Task:
- Return only the columns or aggregate fields that should appear in the final CSV.
- Prefer exact column names from the data profile when the matching field is clear.
- Use simple aggregate names such as count(*), AVG(Column), MAX(Column), MIN(Column)
  when the question asks for an aggregate.
- Do not write SQL.
- Do not compute values.
- Do not include filter, sort, threshold, join, evidence, date, period, ID, or code
  fields unless the question asks to output them.
- Time-range wording such as "these years", "over the years", "after 2005",
  "since inception", "这些年", "历年", or "以后" is usually a filter/range, not
  a request to output the date/period column.
- If the question asks which year/date/period a value belongs to, output the
  year/date/period field and the value field.
- If the question asks for a distribution or grouped count, output the group label
  and count(*).
- If the question asks for an entity, output the user-facing name/label. Output
  code or full/legal name only when explicitly requested.

Return JSON only:
{"columns": ["final output column names or aggregate fields"]}

Examples:

Question:
麻烦查询一下我国历年人均国内生产总值的数据记录
Data profile:
- table ed_grossdomesticproduct: columns=EndDate, GDP, GDPPerCapita, PrimaryIndustryGDP
Answer:
{"columns":["GDPPerCapita"]}

Question:
What is the total amount of deposits and total amount of loans in the RMB credit balances of financial institutions in our country over the years?
Data profile:
- table financial_institution_credit: columns=EndDate, TotalSavings, TotalLoans, FinanceDeposits
Answer:
{"columns":["TotalSavings","TotalLoans"]}

Question:
请列出按"期末累计"记录的新增贷款金额中，金额超过100000的数据记录，展示时间、新增贷款金额记录即可
Data profile:
- table loan_stats: columns=EndDate, SubjectSum, Maturity, IndexName
Answer:
{"columns":["EndDate","SubjectSum"]}

Question:
Which companies are ranked in the top 5 in terms of income in 2021?
Data profile:
- table income: columns=ChiNameAbbr, MainOperIncome, EndDate, CompanyCode
Answer:
{"columns":["ChiNameAbbr"]}

Question:
深科技的代码和公司简称是啥
Data profile:
- table stock_archives: columns=AStockCode, AShareAbbr, ChiName, CompanyCode
Answer:
{"columns":["AStockCode","AShareAbbr"]}

Question:
Number of fund managers serving more than 5 funds, grouped by fund managers' highest educational level.
Data profile:
- table personal_info: columns=PersonalCode, Education, ChineseName
- table fund_manager: columns=PersonalCode, FundCount, SecuAbbr
Answer:
{"columns":["Education","count(*)"]}

Question:
你知道各基金一个月以来回报率记录是什么哦
Data profile:
- table net_value_performance: columns=SecuAbbr, EndDate, RRInSingleMonth, RRInThreeYear
Answer:
{"columns":["RRInSingleMonth"]}
""".strip()


JUDGE_SYSTEM = """
You judge whether predicted final CSV columns match the gold final answer columns.

Compare semantics, not exact spelling or language. PASS only if the predicted
columns have the same final answer fields as gold: no missing field and no extra
field. Ignore harmless aliases such as Year vs EndDate when the question asks for
the year/date.

Output exactly:
PASS: <short reason>
or
FAIL: <short reason>
""".strip()


@dataclass(frozen=True)
class ShapeResult:
    columns: list[str]
    raw: str
    payload: str
    error: str = ""


def _task_sort_key(task_id: str) -> int:
    try:
        return int(task_id.split("_", 1)[1])
    except Exception:
        return 10**9


def _make_model(
    task_id: str,
    *,
    max_tokens: int = 1024,
    enable_thinking: bool = False,
    temperature: float = 0.0,
) -> OpenAIModelAdapter:
    api_base = os.environ.get("AGENT_API_BASE") or os.environ.get("MODEL_API_URL") or ""
    api_key = os.environ.get("AGENT_API_KEY") or os.environ.get("MODEL_API_KEY") or ""
    model_name = (
        os.environ.get("OUTPUT_SHAPE_MODEL")
        or os.environ.get("COLUMN_SELECTOR_MODEL")
        or os.environ.get("AGENT_MODEL")
        or "qwen3.5-35b-a3b"
    )
    headers = with_prefix_cache_header(
        {
            "CF-Access-Client-Id": os.environ.get("CF_ACCESS_CLIENT_ID", ""),
            "CF-Access-Client-Secret": os.environ.get("CF_ACCESS_CLIENT_SECRET", ""),
        },
        f"{task_id}:output-shape-data",
    )
    return OpenAIModelAdapter(
        model=model_name,
        api_base=api_base,
        api_key=api_key,
        temperature=temperature,
        extra_headers=headers,
        enable_thinking=enable_thinking,
        max_tokens=max_tokens,
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


def _normalize_columns(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for item in raw:
        text = str(item).strip()
        if text and text not in out:
            out.append(text)
    return out


def _load_question(task_id: str) -> str:
    path = INPUT_ROOT / task_id / "task.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return str(payload["question"])


def _load_gold(task_id: str) -> tuple[list[str], list[list[str]]]:
    path = GOLD_ROOT / task_id / "gold.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    return (rows[0], rows[1:]) if rows else ([], [])


def _predict_columns(
    task_id: str,
    question: str,
    profile: str,
    *,
    use_model: bool,
    enable_thinking: bool,
    max_tokens: int,
    temperature: float,
) -> ShapeResult:
    if not use_model:
        return ShapeResult(columns=[], raw="", payload="")
    user = f"Question:\n{question}\n\n"
    if profile.strip():
        user += f"Data profile:\n{profile}\n\n"
    else:
        user += "Data profile:\n(not provided)\n\n"
    user += "Answer:"
    try:
        raw = _make_model(
            task_id,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
            temperature=temperature,
        ).complete(
            [
                ModelMessage(role="system", content=SYSTEM_PROMPT),
                ModelMessage(role="user", content=user),
            ],
            enable_thinking=enable_thinking,
            max_tokens=max_tokens,
        )
        payload = _extract_json_object(raw)
        return ShapeResult(
            columns=_normalize_columns(payload.get("columns")),
            raw=raw.strip(),
            payload=json.dumps(payload, ensure_ascii=False),
        )
    except Exception as exc:
        return ShapeResult(columns=[], raw="", payload="", error=repr(exc))


def _judge(
    *,
    task_id: str,
    question: str,
    predicted: list[str],
    gold: list[str],
    gold_rows: list[list[str]],
    use_model: bool,
) -> tuple[bool | None, str, str]:
    if not use_model:
        return None, "", ""
    user = (
        f"Question:\n{question}\n\n"
        f"Predicted columns:\n{', '.join(predicted)}\n\n"
        f"Gold columns:\n{', '.join(gold)}\n\n"
        f"Gold preview rows:\n{json.dumps(gold_rows[:3], ensure_ascii=False)}\n\n"
        "Verdict:"
    )
    try:
        raw = _make_model(f"{task_id}:judge", max_tokens=192).complete(
            [
                ModelMessage(role="system", content=JUDGE_SYSTEM),
                ModelMessage(role="user", content=user),
            ],
            enable_thinking=False,
            max_tokens=192,
        ).strip()
    except Exception as exc:
        return None, f"judge_error={exc!r}", ""
    upper = raw.upper()
    if upper.startswith("PASS"):
        return True, raw, raw
    if upper.startswith("FAIL"):
        return False, raw, raw
    return None, f"unparseable={raw}", raw


def _parse_tasks(raw: str | None, *, all_finance: bool) -> list[str]:
    if raw:
        return [part.strip() for part in raw.split(",") if part.strip()]
    if all_finance:
        return sorted(FINANCE_TASKS, key=_task_sort_key)
    return ["task_2", "task_3", "task_18", "task_23", "task_38", "task_40", "task_42", "task_51", "task_58"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PoC: infer final finance output columns from question + data profile."
    )
    parser.add_argument("--tasks", default=None)
    parser.add_argument("--all-finance", action="store_true")
    parser.add_argument("--profile-max-chars", type=int, default=16000)
    parser.add_argument("--no-profile", action="store_true", help="do not pass task data profile to the model")
    parser.add_argument("--no-model", action="store_true")
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--thinking", action="store_true", help="enable model thinking for the output-shape advisor")
    parser.add_argument(
        "--advisor-max-tokens",
        type=int,
        default=1024,
        help="max_tokens for the output-shape advisor call",
    )
    parser.add_argument(
        "--advisor-temperature",
        type=float,
        default=0.0,
        help="temperature for the output-shape advisor call",
    )
    parser.add_argument("--out-dir", type=Path, default=OUT_ROOT)
    args = parser.parse_args()

    task_ids = _parse_tasks(args.tasks, all_finance=args.all_finance)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for task_id in task_ids:
        question = _load_question(task_id)
        profile = "" if args.no_profile else build_context_profile(
            INPUT_ROOT / task_id / "context",
            max_chars=args.profile_max_chars,
        )
        gold_cols, gold_rows = _load_gold(task_id)
        result = _predict_columns(
            task_id,
            question,
            profile,
            use_model=not args.no_model,
            enable_thinking=args.thinking,
            max_tokens=args.advisor_max_tokens,
            temperature=args.advisor_temperature,
        )
        count_match = len(result.columns) == len(gold_cols)
        judge_match: bool | None = None
        judge_reason = ""
        judge_raw = ""
        if result.columns and not args.no_judge:
            judge_match, judge_reason, judge_raw = _judge(
                task_id=task_id,
                question=question,
                predicted=result.columns,
                gold=gold_cols,
                gold_rows=gold_rows,
                use_model=True,
            )
        row = {
            "task_id": task_id,
            "question": question,
            "predicted_columns": " | ".join(result.columns),
            "predicted_count": len(result.columns),
            "gold_columns": " | ".join(gold_cols),
            "gold_count": len(gold_cols),
            "count_match": count_match,
            "judge_match": judge_match,
            "judge_reason": judge_reason,
            "profile_chars": len(profile),
            "payload": result.payload,
            "raw": result.raw,
            "judge_raw": judge_raw,
            "error": result.error,
        }
        rows.append(row)
        count_mark = "OK" if count_match else "MISS"
        if judge_match is True:
            judge_mark = "PASS"
        elif judge_match is False:
            judge_mark = "FAIL"
        else:
            judge_mark = "SKIP"
        print(
            f"{count_mark:<4} {judge_mark:<4} {task_id:<8} "
            f"pred=[{row['predicted_columns']}] gold=[{row['gold_columns']}]",
            flush=True,
        )

    csv_path = args.out_dir / "phase2_finance_output_shape_from_data_poc.csv"
    json_path = args.out_dir / "phase2_finance_output_shape_from_data_poc.json"
    fieldnames = [
        "task_id",
        "question",
        "predicted_columns",
        "predicted_count",
        "gold_columns",
        "gold_count",
        "count_match",
        "judge_match",
        "judge_reason",
        "profile_chars",
        "payload",
        "raw",
        "judge_raw",
        "error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if rows:
        count_matches = sum(1 for row in rows if row["count_match"])
        judged = [row for row in rows if row["judge_match"] is not None]
        judge_matches = sum(1 for row in judged if row["judge_match"] is True)
        print(f"\ncount_match={count_matches}/{len(rows)} ({count_matches / len(rows):.1%})")
        if judged:
            print(f"judge_match={judge_matches}/{len(judged)} ({judge_matches / len(judged):.1%})")
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
