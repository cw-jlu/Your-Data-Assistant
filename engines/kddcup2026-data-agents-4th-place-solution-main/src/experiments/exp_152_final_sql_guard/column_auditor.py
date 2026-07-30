"""Post-execution projection pruner.

After answer_from_sql produces the raw result table, a separate LLM call
inspects (question, output column names, row count) and returns the indices of
the columns to KEEP. Columns not in the keep-set are dropped before commit.

Design philosophy: pre-execution column rules backfire on ambiguous questions,
but post-execution pruning sees the actual result and can drop obvious extras:
filter/order/join/debug columns, redundant duplicate columns, or helper values
that were needed to find the answer but not requested as output.

The pruner is generic and conservative: when the projection is genuinely
ambiguous, KEEP all columns.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from kobushi_core.model import ModelAdapter, ModelMessage


COLUMN_AUDITOR_SYSTEM = """\
You are a projection pruner for an executed SQL result.

Keep only the output columns necessary to answer the QUESTION.
Do not judge filters, row count, or SQL correctness. Only decide the final
projection.

Rules:
- If the question asks for one attribute/measure, keep only that
  attribute/measure.
- If the question asks for A and B, including "A and B", "A 和 B",
  "A 与 B", or "A 及 B", keep all requested fields. Do not drop one
  requested field just because another field appears more central.
- If it asks for several named measures/statistics, keep each requested
  measure/statistic.
- For list/show/identify/which-entity questions, keep the entity column only
  unless the question asks for additional attributes or values.
- For grouped aggregate questions, keep the group column(s) and the requested
  aggregate column(s).
- If a metric is used only for filtering, thresholding, ranking, ordering, or
  choosing highest/lowest, drop it unless the question asks for its value.
- "record(s)", "data record(s)", "row(s)", "details", "记录", "数据记录",
  or "是什么样" indicate row-preserving output, but do not by themselves ask
  for date/year/period/id/index columns. Keep those columns only when
  explicitly requested.
- For data-record/list questions, keep only fields explicitly requested by the
  question. Do not add context columns such as country/region/province/area or
  entity identifiers unless the question asks to list, return, or group by them.
- Drop date/year/period/index columns unless the question explicitly asks for
  date/year/period/time or says to return them.
- Drop opaque IDs unless the question explicitly asks for ID/code/number.
- Never keep a row id/index only to identify records unless the question asks
  for ID/code/number.
- If genuinely ambiguous, keep all columns.

Output exactly:
KEEP: <comma-separated 0-indexed column indices>
REASON: <short reason naming the kept columns>
"""

CHINESE_COLUMN_AUDITOR_SYSTEM = """\
你是一个 SQL 结果的投影裁剪器。

只保留回答 QUESTION 所必需的输出列。
不要判断过滤条件、行数或 SQL 是否正确。只判断最终应该输出哪些列。

规则：
- 如果问题只问一个属性/指标，只保留那个属性/指标。
- 如果问题问 A 和 B，包括 “A and B”、“A 和 B”、“A 与 B”、“A 及 B”，
  保留所有被请求的字段。不要因为某个字段看起来更核心就丢掉另一个
  被请求字段。
- 如果问题问多个明确命名的指标/统计量，保留每一个被请求的指标/统计量。
- 对于 list/show/identify/which-entity/哪些/列出/展示 类问题，默认只
  保留实体列，除非问题还明确要求额外属性或数值。
- 对于分组聚合问题，保留分组列和被请求的聚合列。
- 如果某个指标只用于过滤、阈值、排序、排名、选择最高/最低，除非问题
  明确要求输出该指标的值，否则丢掉它。
- “record(s)”、“data record(s)”、“row(s)”、“details”、“记录”、
  “数据记录”或“是什么样”表示需要保持原始行粒度，但这些词本身不
  表示要求输出日期、年份、期间、id 或索引列。只有问题明确要求这些
  列时才保留。
- 对于数据记录/list/列出/展示问题，只保留问题明确要求的字段。不要
  添加 country/region/province/area 等地点上下文列或实体标识列，除非
  问题要求列出、返回或按这些字段分组。
- 丢掉日期/年份/期间/索引列，除非问题明确要求日期/年份/期间/时间
  或要求返回它们。
- 丢掉不透明 ID，除非问题明确要求 ID/code/编号/代码/证券代码/基金代码。
- 不要仅仅为了标识记录而保留行 id/index，除非问题要求 ID/code/编号。
- 如果确实有歧义，保留所有列。

输出格式必须完全是：
KEEP: <逗号分隔的 0-based 列索引>
REASON: <一句简短理由，说明保留的列>
"""

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


@dataclass(frozen=True, slots=True)
class AuditResult:
    keep_indices: list[int]
    reason: str
    raw_response: str
    prompt_lang: str = "en"


def _format_columns(columns: list[str]) -> str:
    return "\n".join(f"  [{i}] {c}" for i, c in enumerate(columns))


def _use_chinese_prompt(question: str, columns: list[str]) -> bool:
    haystack = f"{question or ''} {' '.join(columns or [])}"
    return bool(_CJK_RE.search(haystack))


def _build_prompt(
    *,
    question: str,
    columns: list[str],
    rows: list[list[Any]],
) -> tuple[str, str, str]:
    prompt_lang = "zh" if _use_chinese_prompt(question, columns) else "en"
    system_prompt = (
        CHINESE_COLUMN_AUDITOR_SYSTEM if prompt_lang == "zh" else COLUMN_AUDITOR_SYSTEM
    )
    n_rows = len(rows or [])
    cols_repr = _format_columns(columns)
    if prompt_lang == "zh":
        user_prompt = (
            f"## 问题\n{question}\n\n"
            f"## SQL 结果（{n_rows} 行 × {len(columns)} 列）\n"
            f"输出列：\n{cols_repr}\n\n"
            "选择能够回答问题的最小充分列集合。\n"
            "输出 `KEEP: <indices>`，然后输出 `REASON: <sentence>`."
        )
    else:
        user_prompt = (
            f"## Question\n{question}\n\n"
            f"## SQL result ({n_rows} rows × {len(columns)} columns)\n"
            f"Output columns:\n{cols_repr}\n\n"
            "Pick the minimum sufficient subset that answers the question.\n"
            "Output `KEEP: <indices>` followed by `REASON: <sentence>`."
        )
    return prompt_lang, system_prompt, user_prompt


def audit_columns(
    *,
    question: str,
    columns: list[str],
    rows: list[list[Any]],
    model: ModelAdapter,
) -> AuditResult:
    """Return the keep-list. If pruning fails, returns all indices (= keep everything)."""
    prompt_lang = "zh" if _use_chinese_prompt(question, columns) else "en"
    if len(columns) <= 1:
        return AuditResult(
            keep_indices=list(range(len(columns))),
            reason="single column; no audit needed",
            raw_response="",
            prompt_lang=prompt_lang,
        )

    prompt_lang, system_prompt, user_prompt = _build_prompt(
        question=question,
        columns=columns,
        rows=rows or [],
    )
    try:
        raw = model.complete(
            [
                ModelMessage(role="system", content=system_prompt),
                ModelMessage(role="user", content=user_prompt),
            ],
            enable_thinking=False,
        )
    except Exception as exc:
        return AuditResult(
            keep_indices=list(range(len(columns))),
            reason=f"audit LLM error: {exc}",
            raw_response="",
            prompt_lang=prompt_lang,
        )

    m = re.search(r"KEEP\s*:\s*(.+?)(?:\n|$)", raw, flags=re.IGNORECASE)
    if not m:
        return AuditResult(
            keep_indices=list(range(len(columns))),
            reason="pruner produced no KEEP line; defaulting to all",
            raw_response=raw,
            prompt_lang=prompt_lang,
        )
    try:
        idxs = [int(x) for x in re.findall(r"\d+", m.group(1))]
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
    return AuditResult(
        keep_indices=final,
        reason=reason,
        raw_response=raw,
        prompt_lang=prompt_lang,
    )


def apply_audit(columns: list[str], rows: list[list[Any]], audit: AuditResult) -> tuple[list[str], list[list[Any]]]:
    """Project columns/rows to the keep-list."""
    if list(audit.keep_indices) == list(range(len(columns))):
        return columns, rows
    new_cols = [columns[i] for i in audit.keep_indices]
    new_rows = [[r[i] for i in audit.keep_indices] for r in rows]
    return new_cols, new_rows
