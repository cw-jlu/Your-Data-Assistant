"""Post-execution column auditor.

After answer_from_sql produces the raw result table, a separate LLM call
inspects (question, columns, sample rows) and returns the indices of the
columns to KEEP. Columns not in the keep-set are dropped before commit.

Design philosophy: pre-execution column rules backfire on ambiguous
questions (= we observed regressions), but post-execution audit sees the
actual result and can drop obvious extras (= filter columns leaking into
output, debug attributes, redundant duplicates).

The auditor is GENERIC (= no task-specific examples) and conservative —
when uncertain whether a column is required, KEEP it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from kobushi_core.model import ModelAdapter, ModelMessage


COLUMN_AUDITOR_SYSTEM = """\
You are a column-selection auditor. The SQL has already been executed and
returned a result table. Your single job: decide which subset of columns
the QUESTION actually asks for, and instruct which to KEEP.

Rules:
- "List / Identify / Show all <entity>" → typically ONE column (the entity's
  name or id). Don't include the entity's attributes unless the question
  says "with their X" / "and their X" / "list <attribute_A> and <attribute_B>".
- "What is the <attribute> of <entity>" → ONE column for that attribute.
- "How many / count / total / sum / average" → ONE numeric column.
- "<attribute_A> and <attribute_B>" → exactly TWO columns (A, B).
- Filter columns (= the column the question says "WHERE X = ...") are NOT
  output columns — drop them. Example: "withdrawals" filters operation='VYBER'
  but doesn't ask to output `operation`.
- DUPLICATE columns (= same name appearing twice) → keep only ONE.

When uncertain (= question genuinely ambiguous), keep all columns.

Output format (= one line per directive):
KEEP: <comma-separated 0-indexed column positions to keep>
REASON: <one short sentence>

Examples:
KEEP: 0
REASON: List question; only the primary identifier column is needed.

KEEP: 0,2
REASON: Question asks for "name and cost"; columns 0 (name) and 2 (cost) match; drop column 1 (id).

KEEP: 0,1,2,3,4
REASON: Question is ambiguous about which cols to drop; keep all.
"""


# Stronger auditor used only when the domain router places the task in a known
# structured domain (fund / stock / macro / eicu / mimic — i.e. NOT 'other').
# In those domains the result table is a wide schema whose extra columns
# (dates, periods, ids, codes, entity names) are conventional metadata that the
# question rarely asks to output. Adds the policy that the offline finance
# column-selector validated (0 regressions, dropped EndDate/ChiNameAbbr extras).
# Kept domain-general (no test/source literals); only the policy is domain-aware.
DOMAIN_COLUMN_AUDITOR_SYSTEM = """\
You are a column-selection auditor for a structured-data domain (finance or
clinical EHR). The SQL has already been executed and returned a result table.
Decide which subset of columns the QUESTION actually asks to OUTPUT, and KEEP
only those. The result often carries extra schema/metadata columns the question
does not ask for.

Rules:
- "List / Identify / Show all <entity>" → typically ONE column (the entity's
  display name). Add attributes only if the question says "with their X" /
  "and their X" / "<attribute_A> and <attribute_B>".
- "What is the <attribute> of <entity>" → ONE column for that attribute.
- "How many / count / total / sum / average" → ONE numeric column.
- "<attribute_A> and <attribute_B>" → exactly TWO columns (A, B).
- Filter columns are NOT output columns — drop them. A column the question only
  uses to restrict, sort, threshold, join, or as evidence is not requested.
- DROP date / period / time / timestamp / id / code columns unless the question
  explicitly asks to output that date/period/id/code (e.g. "which year", "在哪一
  年", "其代码 / its code"). A bare metric request keeps only the metric.
- Time-range wording is a FILTER, not a request to output the date column:
  "over the years / these years / since / after <year>" and "历年 / 这些年 /
  以来 / 以后 / 年以后". Do NOT keep the date/period column for these. A request
  for "the records of <metric> over the years / 历年<metric>的数据记录" outputs
  ONLY the metric column — the time series IS the metric column, one row per
  period; the date/period column is NOT an output even though every row has one.
- When only a metric/value is asked, DROP the entity name/abbreviation unless the
  question asks for the name (名称 / 简称 / which / 哪些 / "and their name").
- DUPLICATE columns → keep only ONE.

When genuinely ambiguous, keep all columns (never drop a column the question
might want — recall must not be lost).

Output format:
KEEP: <comma-separated 0-indexed column positions to keep>
REASON: <one short sentence>

Examples (columns shown as `[index] name`):
[0] EndDate  [1] GDPPerCapita
Q asks for the records of GDP-per-capita over the years (历年...数据记录).
KEEP: 1
REASON: Time-series metric request; output only the metric, EndDate is a time filter.

[0] SecuAbbr  [1] RRInSingleMonth  [2] EndDate
Q asks for each fund's one-month return record.
KEEP: 1
REASON: Only the rate is asked; the abbreviation and the period are not requested.

[0] EndDate  [1] TotalLiabilities
Q asks which year total liabilities were largest and how much.
KEEP: 0,1
REASON: Question explicitly asks "which year and how much"; keep the period and the value.

[0] id  [1] Province  [2] EndDate  [3] ThirdIndustryGDP
Q asks for the records of Third-Industry GDP over the years.
KEEP: 3
REASON: Only the metric is requested; id/Province/EndDate are schema/filter columns.

KEEP: 0,1,2
REASON: Ambiguous which columns to drop; keep all.
"""


@dataclass(frozen=True, slots=True)
class AuditResult:
    keep_indices: list[int]
    reason: str
    raw_response: str


def _format_columns(columns: list[str], rows: list[list[Any]], max_rows: int = 3) -> str:
    lines = []
    for i, c in enumerate(columns):
        sample_vals = [str(r[i])[:30] for r in (rows or [])[:max_rows] if i < len(r)]
        lines.append(f"  [{i}] {c}: {sample_vals}")
    return "\n".join(lines)


def audit_columns(
    *,
    question: str,
    columns: list[str],
    rows: list[list[Any]],
    model: ModelAdapter,
    domain: str | None = None,
) -> AuditResult:
    """Return the keep-list. If audit fails, returns all indices (= keep everything).

    `domain` selects the prompt: a known structured domain (fund/stock/macro/
    eicu/mimic — anything other than 'other'/None) uses the stronger
    DOMAIN_COLUMN_AUDITOR_SYSTEM that drops date/period/id/name metadata; 'other'
    keeps the conservative general auditor.
    """
    if len(columns) <= 1:
        return AuditResult(keep_indices=list(range(len(columns))), reason="single column; no audit needed", raw_response="")

    use_domain = bool(domain) and domain.strip().lower() != "other"
    system = DOMAIN_COLUMN_AUDITOR_SYSTEM if use_domain else COLUMN_AUDITOR_SYSTEM

    n_rows = len(rows or [])
    cols_repr = _format_columns(columns, rows or [])
    user_prompt = (
        f"## Question\n{question}\n\n"
        f"## SQL result ({n_rows} rows × {len(columns)} columns)\n"
        f"Columns:\n{cols_repr}\n\n"
        "Pick the minimum sufficient subset that answers the question.\n"
        "Output `KEEP: <indices>` followed by `REASON: <sentence>`."
    )
    try:
        raw = model.complete(
            [
                ModelMessage(role="system", content=system),
                ModelMessage(role="user", content=user_prompt),
            ],
            enable_thinking=False,
        )
    except Exception as exc:
        return AuditResult(
            keep_indices=list(range(len(columns))),
            reason=f"audit LLM error: {exc}", raw_response="",
        )

    m = re.search(r"KEEP\s*:\s*([0-9,\s]+)", raw, flags=re.IGNORECASE)
    if not m:
        return AuditResult(
            keep_indices=list(range(len(columns))),
            reason="auditor produced no KEEP line; defaulting to all",
            raw_response=raw,
        )
    try:
        idxs = [int(x.strip()) for x in m.group(1).split(",") if x.strip().isdigit()]
    except Exception:
        idxs = list(range(len(columns)))
    # Validate
    idxs = [i for i in idxs if 0 <= i < len(columns)]
    if not idxs:
        idxs = list(range(len(columns)))
    # Dedup, preserve order
    seen = set()
    final = []
    for i in idxs:
        if i not in seen:
            seen.add(i); final.append(i)

    reason_m = re.search(r"REASON\s*:\s*(.+?)(?:\n|$)", raw, flags=re.IGNORECASE)
    reason = reason_m.group(1).strip() if reason_m else "no reason given"
    return AuditResult(keep_indices=final, reason=reason, raw_response=raw)


def apply_audit(columns: list[str], rows: list[list[Any]], audit: AuditResult) -> tuple[list[str], list[list[Any]]]:
    """Project columns/rows to the keep-list."""
    if list(audit.keep_indices) == list(range(len(columns))):
        return columns, rows
    new_cols = [columns[i] for i in audit.keep_indices]
    new_rows = [[r[i] for i in audit.keep_indices] for r in rows]
    return new_cols, new_rows
