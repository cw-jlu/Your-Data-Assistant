"""Final-step output-column selector (runs AFTER verify + vote).

Ported from the offline finance output-column-selector PoC
(scripts/phase2_finance_output_column_selector_poc.py). Runs ONCE on the
committed final answer table as the TERMINAL pipeline step: given (question,
candidate columns) it returns the subset of columns the question actually asks
to OUTPUT, and the answer is projected to those columns before prediction.csv
is written.

This is ADDITIVE to the in-loop column_auditor (which prunes per
`answer_from_sql` call). The selector is the final authoritative projection —
belt-and-suspenders: the auditor catches extras during the loop, the selector
re-checks the committed answer once at the end.

Conservative by construction: any uncertainty / parse error / empty selection
→ keep ALL columns (never lose recall). The caller domain-gates it (finance /
EHR only, via the deterministic domain router).

No DABench test literals: the few-shot examples below are BULL public-corpus
questions (external training data), not demo tasks; the PoC's demo task-id
allowlist is intentionally NOT ported (domain routing gates instead).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from kobushi_core.model import ModelAdapter, ModelMessage


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


@dataclass
class SelectionResult:
    columns: list[str]
    rows: list[list[Any]]
    applied: bool = False
    selected: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    raw_response: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "applied": self.applied,
            "selected": self.selected,
            "dropped": self.dropped,
            "error": self.error,
            "raw_response": self.raw_response[:600],
        }


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
    """Map the model's chosen names back to the exact candidate strings.

    Falls back to ALL candidates when nothing matches (never lose recall).
    """
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


def _project(
    columns: list[str], rows: list[list[Any]], keep: list[str]
) -> tuple[list[str], list[list[Any]]]:
    """Drop to `keep`, preserving the ORIGINAL column order (no reordering)."""
    keep_set = set(keep)
    idxs = [i for i, col in enumerate(columns) if col in keep_set]
    new_cols = [columns[i] for i in idxs]
    new_rows = [[row[i] if i < len(row) else "" for i in idxs] for row in rows]
    return new_cols, new_rows


def select_output_columns(
    *,
    question: str,
    columns: list[str],
    rows: list[list[Any]],
    model: ModelAdapter,
) -> SelectionResult:
    """Project the committed answer to the question-requested output columns.

    Conservative: <=1 candidate, a parse error, or an empty/degenerate
    selection all return the input unchanged (keep every column).
    """
    cols = list(columns)
    if len(cols) <= 1:
        return SelectionResult(columns=cols, rows=rows, applied=False, selected=cols)

    user = (
        f"Question:\n{question}\n\n"
        "Candidate final answer columns:\n"
        f"{', '.join(cols)}\n\n"
        "Answer:"
    )
    try:
        raw = model.complete(
            [
                ModelMessage(role="system", content=SYSTEM_PROMPT),
                ModelMessage(role="user", content=user),
            ],
            enable_thinking=False,
        )
    except Exception as exc:  # keep all on any LLM error
        return SelectionResult(
            columns=cols, rows=rows, applied=False, selected=cols, error=repr(exc)
        )

    try:
        payload = _extract_json_object(raw)
        selected = _normalize_selected(payload.get("columns"), cols)
    except Exception as exc:
        return SelectionResult(
            columns=cols, rows=rows, applied=False, selected=cols,
            raw_response=raw, error=repr(exc),
        )

    # Preserve original order; drop only. Empty → keep all (recall guard).
    keep_set = set(selected)
    kept = [c for c in cols if c in keep_set]
    if not kept:
        kept = list(cols)
    dropped = [c for c in cols if c not in set(kept)]
    if not dropped:
        return SelectionResult(
            columns=cols, rows=rows, applied=False, selected=kept, raw_response=raw
        )

    new_cols, new_rows = _project(cols, rows, kept)
    return SelectionResult(
        columns=new_cols, rows=new_rows, applied=True,
        selected=kept, dropped=dropped, raw_response=raw,
    )
