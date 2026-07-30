"""PoC: classify final SQL DISTINCT / NULL filters without result metadata.

This script does not modify benchmark code. It evaluates a conservative
classifier prompt on two small labeled sets:

  - trace-derived examples from prior Phase 2 runs
  - synthetic edge cases for DISTINCT and NULL/blank filters

The classifier sees only question + final SQL. It does not receive gold answers,
scores, row counts, previews, trace snippets, or task ids in the prompt.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from dotenv import load_dotenv

load_dotenv(REPO / ".env")

from kobushi_core.model import ModelMessage, OpenAIModelAdapter
from experiments.exp_150_projection_pruner.prefix_cache import with_prefix_cache_header


RUNS_ROOT = REPO / "artifacts" / "runs"

Operation = Literal["distinct", "null_filter"]
EXPLICIT_UNIQUE_RE = re.compile(
    r"\b(?:distinct|unique|non[- ]?duplicate|deduplicated|de[- ]?duplicated)\b"
    r"|去重|唯一|不重复|互不相同",
    re.IGNORECASE,
)


DISTINCT_SYSTEM = """You classify whether SQL DISTINCT should be kept.

The SQL contains DISTINCT or COUNT(DISTINCT ...).
Do not solve the task.

Default: preserve source row grain. DISTINCT is suspicious.

Return exactly one token:
KEEP_DISTINCT
or
REMOVE_DISTINCT

The user prompt lists the DISTINCT form:
- COUNT(DISTINCT ...): changes count grain.
- SELECT DISTINCT: deduplicates output rows.

Decision order:
1. If the question explicitly asks for distinct, unique, different,
   non-duplicate, deduplicated, de-duplicated, 去重, 唯一, 不同, 不重复, or
   互不相同, return KEEP_DISTINCT.
2. Otherwise return REMOVE_DISTINCT.

Return REMOVE_DISTINCT when DISTINCT merely cleans a single-table list, hides
real duplicate source rows, or changes a requested record/snapshot count into
an entity count. Do not keep DISTINCT because of joins, entity nouns, plural
nouns, descriptor columns, or count questions. If unsure, return
REMOVE_DISTINCT."""

DISTINCT_SYSTEM += """

Examples:
- Q asks to list regions recorded in one table; SQL is SELECT DISTINCT region
  FROM that table -> REMOVE_DISTINCT.
- Q asks how many records/snapshots satisfy a condition; SQL uses
  COUNT(DISTINCT entity_id) -> REMOVE_DISTINCT.
- Q asks which managers/customers qualify; SQL projects the manager/customer
  table and joins a many-side performance/order table, but the question does
  not say distinct/unique -> REMOVE_DISTINCT.
- Q explicitly says distinct patients / unique companies / 去重 -> KEEP_DISTINCT."""


NULL_SYSTEM = """You classify whether final SQL NULL/blank filtering should be kept.

The SQL contains IS NOT NULL, <> '', or != ''.
Do not solve the task.

Default: preserve source rows, including NULL and blank values.

Return exactly one token:
KEEP_NULL_FILTER
or
REMOVE_NULL_FILTER

Decision order:
1. If the question explicitly asks for non-null, non-empty, known values,
   available values, excluding blanks, excluding missing, 有值, 非空, 不为空,
   已知, or 排除空值, return KEEP_NULL_FILTER.
2. Otherwise, if the filter is on the same measure used for a requested
   ranking, highest/lowest/top/bottom comparison, or aggregate where missing
   values cannot participate, return KEEP_NULL_FILTER.
3. Otherwise return REMOVE_NULL_FILTER.

Return REMOVE_NULL_FILTER when the question asks to show, list, retrieve, or
display source rows/values and does not request excluding missing values. If
unsure, return REMOVE_NULL_FILTER."""


@dataclass(frozen=True)
class Case:
    case_id: str
    source: str
    operation: Operation
    question: str
    sql: str
    expected: str
    note: str


TRACE_CASES = [
    Case(
        case_id="trace_task_1_count_company_rows",
        source="exp145 task_1",
        operation="distinct",
        question="根据视频中展示的流通A股股本准入线和统计年份口径，统计满足条件的公司数目，按公司所在二级行业分组展示结果。",
        sql='SELECT e."SecondIndustryName", COUNT(DISTINCT f."CompanyCode") AS "公司数目" '
        'FROM "lc_freefloat" f JOIN "lc_exgindustry" e ON f."CompanyCode" = e."CompanyCode" '
        "WHERE substr(CAST(f.\"ChangeDate\" AS VARCHAR), 1, 4) = '2019' "
        'AND f."AFloats" > 10000000000 GROUP BY e."SecondIndustryName"',
        expected="REMOVE_DISTINCT",
        note="Local gold is count(*)=4; distinct company count undercounts repeated source rows.",
    ),
    Case(
        case_id="trace_task_12_explicit_distinct_patients",
        source="exp155 task_12",
        operation="distinct",
        question="According to the encounter-level diagnosis tracking rule defined in the video, how many distinct patients meet all the configured screening criteria?",
        sql="""WITH pneumonia_dxs AS (
  SELECT DISTINCT p.uniquepid, p.patienthealthsystemstayid, d.diagnosistime
  FROM "patient" p
  JOIN "diagnosis" d ON p.patientunitstayid = d.patientunitstayid
  WHERE d.diagnosisname = 'pneumonia'
    AND p.hospitaladmittime >= '2105-01-01'
),
hypomagnesemia_dxs AS (
  SELECT DISTINCT p.uniquepid, p.patienthealthsystemstayid, d.diagnosistime
  FROM "patient" p
  JOIN "diagnosis" d ON p.patientunitstayid = d.patientunitstayid
  WHERE d.diagnosisname = 'hypomagnesemia - due to etoh abuse'
    AND p.hospitaladmittime >= '2105-01-01'
)
SELECT COUNT(DISTINCT p.uniquepid) AS patient_count
FROM pneumonia_dxs p
JOIN hypomagnesemia_dxs h ON p.uniquepid = h.uniquepid
 AND p.patienthealthsystemstayid = h.patienthealthsystemstayid
WHERE p.diagnosistime < h.diagnosistime""",
        expected="KEEP_DISTINCT",
        note="Question explicitly asks distinct patients.",
    ),
    Case(
        case_id="trace_task_42_join_fanout_entity_list",
        source="exp149 task_42",
        operation="distinct",
        question="Which fund managers have less than 2 years of professional experience and achieved a positive fund growth rate during their tenure?",
        sql='SELECT DISTINCT b."基金经理姓名" FROM "公募基金经理(新)" AS a '
        'JOIN "公募基金经理基本资料" AS b ON a."所属人员代码" = b."所属人员编码" '
        'WHERE a."任职期间基金净值增长率" > 0 AND b."证券从业经历(年)" < 2',
        expected="REMOVE_DISTINCT",
        note="Explicit-only policy: no distinct/unique wording in the question.",
    ),
    Case(
        case_id="trace_task_14_max_commission_null",
        source="exp149 task_14",
        operation="null_filter",
        question="Which company has the highest commission fees?",
        sql='SELECT ChiNameAbbr, Commission FROM "lc_financialexpense" '
        'WHERE Commission = (SELECT MAX(Commission) FROM "lc_financialexpense" '
        "WHERE Commission IS NOT NULL)",
        expected="KEEP_NULL_FILTER",
        note="NULL filter is on the ranked measure for a highest-value comparison.",
    ),
    Case(
        case_id="trace_task_10_row_preserve_null",
        source="exp145 task_10",
        operation="null_filter",
        question="找一下数据。你能帮我查一下，在货币当局资产负债表中，总资产的金额大小吗",
        sql='SELECT "EndDate", "TotalAssets" FROM "ed_moneyauthoritybs" '
        'WHERE "TotalAssets" IS NOT NULL ORDER BY "EndDate"',
        expected="REMOVE_NULL_FILTER",
        note="Question asks to look up/list source values; gold preserves blank TotalAssets rows.",
    ),
]


SYNTHETIC_CASES = [
    Case(
        case_id="syn_explicit_unique_en",
        source="synthetic",
        operation="distinct",
        question="How many unique customers placed an order in 2025?",
        sql="SELECT COUNT(DISTINCT customer_id) FROM orders WHERE order_date >= '2025-01-01'",
        expected="KEEP_DISTINCT",
        note="Explicit unique.",
    ),
    Case(
        case_id="syn_explicit_unique_zh",
        source="synthetic",
        operation="distinct",
        question="去重后有多少家公司满足筛选条件？",
        sql='SELECT COUNT(DISTINCT "CompanyCode") FROM "lc_freefloat" WHERE "AFloats" > 10000000000',
        expected="KEEP_DISTINCT",
        note="Explicit 去重.",
    ),
    Case(
        case_id="syn_join_fanout_list",
        source="synthetic",
        operation="distinct",
        question="List customers who placed an order in 2025.",
        sql="SELECT DISTINCT c.name FROM customers c JOIN orders o ON c.id = o.customer_id WHERE o.order_date >= '2025-01-01'",
        expected="REMOVE_DISTINCT",
        note="Explicit-only policy: join fan-out alone is not enough.",
    ),
    Case(
        case_id="syn_single_table_list_preserve_rows",
        source="synthetic",
        operation="distinct",
        question="List the provinces recorded in the GDP table.",
        sql="SELECT DISTINCT Province FROM ed_grossdomesticproduct ORDER BY Province",
        expected="REMOVE_DISTINCT",
        note="Single-table list; duplicates are source rows unless uniqueness is requested.",
    ),
    Case(
        case_id="syn_count_companies_default_rows",
        source="synthetic",
        operation="distinct",
        question="How many records satisfy the screening condition, grouped by industry?",
        sql="SELECT industry, COUNT(DISTINCT company_id) FROM freefloat WHERE year = 2019 GROUP BY industry",
        expected="REMOVE_DISTINCT",
        note="Question asks record count; do not turn it into entity count.",
    ),
    Case(
        case_id="syn_base_history_entity_count",
        source="synthetic",
        operation="distinct",
        question="How many patients had a pneumonia diagnosis in 2025?",
        sql="SELECT COUNT(DISTINCT patient_id) FROM diagnosis_events WHERE diagnosis = 'pneumonia' AND diagnosis_date >= '2025-01-01'",
        expected="REMOVE_DISTINCT",
        note="Explicit-only policy: event grain alone is not enough.",
    ),
    Case(
        case_id="syn_graph_bond_fanout",
        source="synthetic",
        operation="distinct",
        question="What is the average number of bonds the iodine atoms have?",
        sql="SELECT AVG(bond_count) FROM (SELECT a.atom_id, COUNT(DISTINCT c.bond_id) AS bond_count FROM atom a JOIN connected c ON a.atom_id = c.atom_id WHERE a.element = 'i' GROUP BY a.atom_id)",
        expected="REMOVE_DISTINCT",
        note="Explicit-only policy: no distinct/unique wording in the question.",
    ),
    Case(
        case_id="syn_explicit_non_null_en",
        source="synthetic",
        operation="null_filter",
        question="List users with a known email address.",
        sql="SELECT user_id, email FROM users WHERE email IS NOT NULL",
        expected="KEEP_NULL_FILTER",
        note="Explicit known values.",
    ),
    Case(
        case_id="syn_explicit_non_null_zh",
        source="synthetic",
        operation="null_filter",
        question="列出邮箱非空的用户。",
        sql='SELECT "user_id", "email" FROM "users" WHERE "email" IS NOT NULL',
        expected="KEEP_NULL_FILTER",
        note="Explicit 非空.",
    ),
    Case(
        case_id="syn_row_preserve_null",
        source="synthetic",
        operation="null_filter",
        question="Show all account balances.",
        sql="SELECT account_id, balance FROM accounts WHERE balance IS NOT NULL ORDER BY account_id",
        expected="REMOVE_NULL_FILTER",
        note="Plain show/list should preserve missing balances.",
    ),
    Case(
        case_id="syn_ranking_measure_null",
        source="synthetic",
        operation="null_filter",
        question="Which product has the highest revenue?",
        sql="SELECT product_name, revenue FROM products WHERE revenue IS NOT NULL ORDER BY revenue DESC LIMIT 1",
        expected="KEEP_NULL_FILTER",
        note="NULL filter is on the ranked measure.",
    ),
]


def make_model() -> OpenAIModelAdapter:
    headers = with_prefix_cache_header(
        {
            "CF-Access-Client-Id": os.environ.get("CF_ACCESS_CLIENT_ID", ""),
            "CF-Access-Client-Secret": os.environ.get("CF_ACCESS_CLIENT_SECRET", ""),
        },
        "final-sql-classifier-poc-v2",
    )
    return OpenAIModelAdapter(
        model=os.environ.get("AGENT_MODEL", "qwen3.5-35b-a3b"),
        api_base=os.environ.get("AGENT_API_BASE", ""),
        api_key=os.environ.get("AGENT_API_KEY", ""),
        temperature=0.0,
        extra_headers=headers,
    )


def build_user_prompt(case: Case) -> str:
    parts = [
        "Question:",
        case.question,
        "",
        "Final SQL:",
        case.sql,
    ]
    if case.operation == "distinct":
        forms = []
        if re.search(r"\bCOUNT\s*\(\s*DISTINCT\b", case.sql, re.IGNORECASE):
            forms.append("COUNT(DISTINCT ...)")
        if re.search(r"\bSELECT\s+DISTINCT\b", case.sql, re.IGNORECASE):
            forms.append("SELECT DISTINCT")
        parts.extend(["", "DISTINCT form:", ", ".join(forms) if forms else "DISTINCT"])
    parts.extend(["", f"Classify only the {case.operation} operation."])
    return "\n".join(parts)


def classify(model: OpenAIModelAdapter, case: Case, max_tokens: int) -> tuple[str, str, float]:
    system = DISTINCT_SYSTEM if case.operation == "distinct" else NULL_SYSTEM
    allowed = (
        {"KEEP_DISTINCT", "REMOVE_DISTINCT"}
        if case.operation == "distinct"
        else {"KEEP_NULL_FILTER", "REMOVE_NULL_FILTER"}
    )
    t0 = time.time()
    raw = model.complete(
        [
            ModelMessage(role="system", content=system),
            ModelMessage(role="user", content=build_user_prompt(case)),
        ],
        enable_thinking=False,
        max_tokens=max_tokens,
    ).strip()
    token = raw.split()[0].strip().upper() if raw.split() else ""
    if token not in allowed:
        token = "UNKNOWN"
    return token, raw, time.time() - t0


def lexical_distinct_label(question: str) -> str:
    return "KEEP_DISTINCT" if EXPLICIT_UNIQUE_RE.search(question or "") else "REMOVE_DISTINCT"


def next_out_dir() -> Path:
    i = 1
    while (RUNS_ROOT / f"final_sql_classifier_poc_v2_{i:03d}").exists():
        i += 1
    out = RUNS_ROOT / f"final_sql_classifier_poc_v2_{i:03d}"
    out.mkdir(parents=True)
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-set", choices=["trace", "synthetic", "all"], default="all")
    ap.add_argument("--max-tokens", type=int, default=32)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if args.case_set == "trace":
        cases = TRACE_CASES
    elif args.case_set == "synthetic":
        cases = SYNTHETIC_CASES
    else:
        cases = TRACE_CASES + SYNTHETIC_CASES

    out = next_out_dir()
    model = make_model()
    results = []
    print(f"=== final SQL classifier PoC v2 -> {out} ===", flush=True)
    print(f"cases={len(cases)} metadata=none", flush=True)
    for case in cases:
        label, raw, elapsed = classify(model, case, args.max_tokens)
        lexical_label = lexical_distinct_label(case.question) if case.operation == "distinct" else None
        ok = label == case.expected
        item = {
            "case_id": case.case_id,
            "source": case.source,
            "operation": case.operation,
            "expected": case.expected,
            "label": label,
            "lexical_label": lexical_label,
            "lexical_ok": None if lexical_label is None else lexical_label == case.expected,
            "ok": ok,
            "elapsed_sec": round(elapsed, 3),
            "raw": raw,
            "question": case.question,
            "sql": case.sql,
            "note": case.note,
        }
        results.append(item)
        mark = "OK" if ok else "NG"
        print(
            f"{mark} {case.case_id} [{case.operation}] expected={case.expected} got={label} ({elapsed:.2f}s)",
            flush=True,
        )

    by_op: dict[str, dict[str, int]] = {}
    for item in results:
        stats = by_op.setdefault(item["operation"], {"ok": 0, "n": 0})
        stats["n"] += 1
        stats["ok"] += int(bool(item["ok"]))
    summary = {
        "case_set": args.case_set,
        "n": len(results),
        "ok": sum(int(bool(r["ok"])) for r in results),
        "by_operation": by_op,
        "distinct_lexical": {
            "ok": sum(
                int(bool(r.get("lexical_ok")))
                for r in results
                if r["operation"] == "distinct"
            ),
            "n": sum(1 for r in results if r["operation"] == "distinct"),
            "pattern": EXPLICIT_UNIQUE_RE.pattern,
        },
    }
    payload = {
        "summary": summary,
        "distinct_system_prompt": DISTINCT_SYSTEM,
        "null_system_prompt": NULL_SYSTEM,
        "user_prompt_template": "Question:\\n{question}\\n\\nFinal SQL:\\n{sql}\\n\\nClassify only the {operation} operation.",
        "results": results,
    }
    (out / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Final SQL Classifier PoC v2",
        "",
        f"- cases: {summary['n']}",
        f"- correct: {summary['ok']}/{summary['n']}",
    ]
    for op, stats in by_op.items():
        lines.append(f"- {op}: {stats['ok']}/{stats['n']}")
    lines.extend(["", "## Results", ""])
    for item in results:
        lines.append(
            f"- {'OK' if item['ok'] else 'NG'} `{item['case_id']}` "
            f"expected `{item['expected']}`, got `{item['label']}`"
            + (
                f", lexical `{item['lexical_label']}`"
                if item.get("lexical_label") is not None
                else ""
            )
        )
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if summary["ok"] == summary["n"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
