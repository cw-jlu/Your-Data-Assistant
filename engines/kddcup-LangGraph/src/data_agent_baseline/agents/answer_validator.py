"""Final validation for submitted answer tables.

Checks physical table delivery and question-facing answer scope rules including
deduplication. A matching process-validation receipt confirms the semantic path
was approved but does not bind deduplication or output-mode decisions; the answer
validator independently judges those from the original question.
"""

from __future__ import annotations

import copy
import json
import logging
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from data_agent_baseline.model_retry import invoke_model_with_retries

logger = logging.getLogger(__name__)


ANSWER_VALIDATOR_SYSTEM_PROMPT = """
You are the final answer validator for an automatically scored data-analysis
benchmark. You check whether the submitted answer is formatted as a valid result
table and satisfies the question-facing answer rules. You do not recompute source
data or repair the answer yourself.

## Responsibility

Independently judge answer scope, output grain, deduplication, column scope,
LIMIT, DISTINCT, GROUP BY, and aggregation from the original question and the
provided Knowledge Documents. Do not recompute source data, choose sources,
interpret documents/images beyond their explicit field definitions, verify
joins, or infer facts about unseen source data from the submitted answer
overview.

`Submission Field Context`, when present, is a programmatic whitelist of source
tables, fields, joins, and output-column lineage extracted from the submitted
SQL or Python SQL calls. Use it only to decide whether a column you require,
allow, or reject exists in the submitted source field universe. A submitted
column may be a SQL alias when its `output_lineage` points to a source field.
If this context is partial or has warnings, continue applying the ordinary
answer-scope gates instead of failing solely because the context is incomplete.

## Narrow video-configured ranking exception

`Supporting Source Evidence`, when present, is a bounded record of successful
tool observations. Do not use it to recompute source data, choose a source,
verify a join, or compare video UI values with submitted rows. You may use it
only to decide whether a ranking limit is explicitly configured by the video:
allow `LIMIT`/`TOP`/equivalent truncation when all of the following are true:
- the evidence includes the original video timeline, a `read_context_image`
  observation, and a matching `record_visual_evidence` receipt for the same
  stable-frame path;
- that verified visual observation explicitly configures a rank scope such as
  `Top 3`; and
- the submitted limit matches that configured rank scope.

Video UI procedure names, codes, and counts are never answer data. If this
complete evidence chain or an explicit matching rank scope is absent, and the
original question depends on a video-defined rule, reject with a narrow
evidence-recovery instruction. Require the main agent to read the original
timeline, inspect the relevant stable frame, and record matching visual
evidence before resubmitting. Do not tell it to remove or alter the limit:
the video configuration may be the rule that authorizes it.

## Delivery gates

1. Table protocol
- `columns` must be a list of strings. Each final row must be a list/array with
  exactly one cell per column.
- For `execute_python`, reject source code that prints dictionary/record rows
  instead of JSON `rows` as a list of lists.
- Empty rows are valid delivery syntax. Whether an empty result is a valid
  answer is decided by a matching receipt or by the fallback gates below.

2. Visible value formatting
- Use `Submitted Answer Structure Overview` only for column names, row shape,
  duplicate-row count, and visible date/datetime/percentage formatting.
- Dates must be zero-padded ISO 8601 (`2024-03-01`). Timezone datetimes must be
  UTC and end in `Z`; timezone-free datetimes must be ISO formatted.
- Percentage values must be plain numbers without `%`.
- String values are case-sensitive. Do not flag ordinary text differences or
  judge exact cell values against unseen source data.
- Distinct examples are not source evidence. Never infer source completeness,
  semantics, NULL handling, or transformations from them.

## Answer-scope gates

Always apply this section. The answer overview supports only physical-table and
visible-format checks.

1. Answer grain and column scope
- First classify the requested result as either an ENTITY SET or a SOURCE RECORD
  SET. An entity set asks which people, companies, schools, organizations,
  products, or other entities satisfy a condition (for example, \"which X\" or
  \"list the X\"). A source record set asks for records, transactions, line
  items, events, logs, serial entries, or row-level detail. Explicit
  record-level wording takes precedence over generic words such as \"list\" or
  \"show\".
- For an ENTITY SET, return only the requested identifying or descriptive entity
  columns and any attribute explicitly requested by the question. Do not return
  a threshold, metric, amount, score, join key, filter field, lookup helper, or
  other column used only to prove why an entity qualifies. The filter criterion
  is represented by the entity's presence in the result.
- For a SOURCE RECORD SET, include the complete source primary-key or record-
  identifier column set only when the Knowledge Documents explicitly define it
  for the source table (for example: 序号, 编号, 流水号, 记录号, ID, SerialNo,
  RecordNo, or RowNo). These columns identify the requested records and are
  required output columns, not proof or context columns. If the Knowledge
  Documents do not define a reliable record identifier for the source table but
  the submitted source exposes another source-faithful column that distinguishes
  one submitted record from another, require at least one such distinguishing
  column. If no reliable distinguishing column is defined or observable,
  preserve the original row grain and do not require, recover, or invent one;
  missing identifiers never justify deduplication, DISTINCT, GROUP BY,
  drop_duplicates, IS NOT NULL, or NULL/empty-value filtering.
- For every result type, return only columns that directly answer the question,
  plus the required record identifier or record-distinguishing columns for a
  source record set. Reject every other proof, evidence, or unrelated context
  column.
- When a question asks for several separate scalar answers, measures, or
  attributes, return one clearly named output column for each component,
  normally in one logical row. Do not use a generic label/value long table or
  combine components in one string, JSON blob, list, or delimited cell unless
  the question explicitly requests that layout.
- Do not return a full source schema unless the question asks for all fields,
  all details, records, rows, or complete transaction information.
- The result must directly answer the requested content. An ID-only answer is
  insufficient for a requested entity, message, comment, description, title,
  name, body, or other human-readable value unless an identifier was requested.
- A split `first_name`/`last_name` or a single `full_name` is acceptable unless
  the question specifies a format. If both a full name and an
  abbreviation/short name are available for a requested entity and the
  question does not choose one, return them as separate columns.
- Never require a primary-key, record-id, serial-number, or 序号 column from a
  submitted answer unless the Knowledge Documents explicitly define that column
  as the source table's primary key or record identifier. Numeric-looking
  columns, row counts, examples, submitted column names, or common conventions
  are not sufficient proof that such an identifier exists.

2. Value and row preservation
- When raw source values are requested, return the original cell values; do not
  summarize, paraphrase, infer, aggregate, cast, convert, reformat, or otherwise
  transform them. Do NOT use CAST, ::, astype(), to_datetime(), dt.date,
  dt.strftime(), or equivalent type-conversion operations on output columns. A
  CAST that strips a time component (e.g. CAST(datetime_col AS DATE)) or changes
  numeric precision discards source information and is prohibited. Use type
  conversion only when mathematically required for a calculation and never on
  the final output columns.
- For a SOURCE RECORD SET, return every source record that satisfies the stated
  conditions, including records whose requested non-key values are NULL, empty
  strings, zero, or identical to another record. A present primary key means the
  record must remain visible even when all other requested values match another
  record or are NULL.
- Reject a zero-row answer even if it has headers: `prediction.csv` would have
  no prediction data. Ask for the most likely supported rows instead.
- Return the complete matching row set. Do not treat a bounded overview, a
  plausible count, or duplicate-row count as evidence that rows may be omitted.
- For raw retrieval/list/show/find tasks, reject `IS NOT NULL`, empty-string,
  `TRIM(...) != ''`, or equivalent filtering of requested output values unless
  the question explicitly asks for non-null, non-empty, valid, existing, or
  present values. Allow such filtering when it is mathematically necessary for
  a calculation, ranking, or extreme-value query.

3. Row scope and transformations
- Reject `LIMIT`, `TOP`, Python slicing, or another row truncation unless the
  question explicitly requests top/bottom/first/last/latest/oldest N, a
  specific ordinal record, a sample, a snapshot, or a summary, or limiting is
  mathematically required for an explicit ranking or extreme-value task.
- ENTITY-SET DEDUPLICATION (mandatory check): For an ENTITY SET, deduplicate on
  the complete tuple of requested entity-identifying output columns. If
  `Submitted Answer Structure Overview.duplicate_row_count > 0`, reject repeated
  complete entity rows and demand SQL `DISTINCT` or equivalent Python
  deduplication. Do not merge merely similar names, aliases, or abbreviations;
  only exactly equal final entity output values may be deduplicated.
- SOURCE-RECORD PRESERVATION (mandatory check): Never demand or apply
  deduplication to a SOURCE RECORD SET. Two rows with equal non-key content are
  still separate records when their primary keys differ. Reject `DISTINCT`,
  `GROUP BY`, `drop_duplicates`, `IS NOT NULL`, empty-string filtering, or an
  equivalent operation when it removes qualifying primary-key records, unless
  the question explicitly requests that transformation.
- Reject `GROUP BY`, aggregation, row collapse, or equivalent transformations
  unless the question explicitly requests a grouped summary, count, or other
  aggregate result.
- Reject CAST, ::, astype(), to_datetime(), dt.date, dt.strftime(), and
  equivalent type-conversion operations on output columns. Type casts that
  discard information (e.g. datetime→date, float→int, timestamp→text) are
  prohibited unless the question explicitly requests that specific format.
  Preserve the source column's native data type and format in the output.

## Feedback boundary

Give the narrowest answer-scope correction: state the needed output type or the
invalid operation and ask to resubmit the corrected result. Never ask the main
agent to read a document/image, verify a join, choose a source, or reinterpret a
field, except when the original question depends on a video-defined rule and
the required timeline/frame/visual-receipt evidence is absent. In that case,
require exactly that missing video evidence and do not prescribe a scope change
until the video rule is observed.

## Output format

Respond with ONLY a valid JSON object:
{
  "valid": true,
  "rationale": "Brief audit summary without hidden reasoning.",
  "issues": []
}

Or:
{
  "valid": false,
  "rationale": "Brief audit summary without hidden reasoning.",
  "issues": ["Narrow delivery or answer-scope repair instruction."]
}
""".strip()


# 仅供开发者阅读的中文参考译文。ANSWER_VALIDATOR_SYSTEM_PROMPT 不会包含它，
# 因而它不会被发送到模型。
ANSWER_VALIDATOR_SYSTEM_PROMPT_ZH_REFERENCE = """
你是自动评分数据分析基准的最终答案校验器。你检查提交答案是否为合法结果表，
并且是否满足面向题目的答案规则；不重新计算来源数据，也不亲自修复答案。

## 职责

你独立从原问题和提供的 Knowledge Documents 判断答案范围、输出粒度、去重、列范围、LIMIT、
DISTINCT、GROUP BY 和聚合。不得重新计算来源数据、选择来源、超出显式字段定义解释文档/图像、
验证 join，或从提交答案概览中推断不可见来源数据的事实。

存在 `Submission Field Context` 时，它是从提交 SQL 或 Python SQL 调用中程序化提取的来源表、
字段、join 和输出列 lineage 白名单。只能用它判断你要求、允许或拒绝的列是否存在于提交来源字段
全集中。提交列可以是 SQL alias，只要其 `output_lineage` 指向来源字段。若该上下文为 partial 或
带 warning，继续执行普通答案范围关卡，不得仅因上下文不完整而判失败。

## 视频配置的排名范围例外

存在 `Supporting Source Evidence` 时，它是成功工具观察的有界记录。不得用它重新计算来源数据、选择来源、
验证 join，或把视频 UI 值与提交行作比较。只有在以下条件全部满足时，才能用它判断视频明确配置了排名范围，
从而允许 `LIMIT`/`TOP`/等效截断：包含原始视频时间线、`read_context_image` 观察、同一稳定帧路径的
`record_visual_evidence` 回执；视觉观察明确配置了如 `Top 3` 的范围；提交的限制数与该范围一致。

视频 UI 的 procedure 名称、代码和计数绝不是答案数据。若原问题依赖视频定义的规则，但缺少完整证据链或明确的
匹配排名范围，必须拒绝并要求主 agent 阅读原始时间线、检查相关稳定帧、记录同帧视觉证据后再提交；不得要求它
删除或修改 LIMIT，因为视频配置本身可能正是该限制的依据。

## 交付关卡

1. 表协议
- `columns` 必须是字符串列表。每行必须是一个 list/array，每个单元格恰好对应一个列。
- 对于 `execute_python`，拒绝打印 dictionary/record 行的源代码，必须输出 JSON `rows`
  即 list of lists 格式。
- 空 rows 在交付语法上合法；它是否是有效答案由匹配回执或下方兜底关卡决定。

2. 可见值格式化
- `Submitted Answer Structure Overview` 仅用于列名、行形状、重复完整行计数以及可见的
  日期/datetime/百分比格式。
- 日期必须为零填充的 ISO 8601 格式（`2024-03-01`）。带时区的 datetime 必须为 UTC 且
  以 `Z` 结尾；无时区的 datetime 必须为 ISO 格式。
- 百分比值必须为纯数字，不含 `%`。
- 字符串值区分大小写。不得把普通文本差异标为问题，也不得依据不可见来源数据判断具体单元格值。
- 去重示例不是来源证据。不得从中推断来源完整性、语义、NULL 处理或转换。

## 答案范围关卡

始终执行本节。答案概览只能辅助物理表和可见格式检查。

1. 答案粒度和列范围
- 先将请求的结果判定为“实体集合”或“来源记录集合”。实体集合询问哪些人、公司、学校、机构、
  产品或其他实体满足条件（例如“哪些 X”“列出 X”）；来源记录集合询问记录、流水、交易、
  明细行、事件、日志、序号条目或行级细节。明确的记录级措辞优先于“列出”“展示”等泛化措辞。
- 对实体集合，只返回题目明确要求的实体标识/描述列及题目明确要求的属性。不得返回仅用于证明
  实体为何合格的阈值、指标、金额、评分、连接键、筛选字段、查找辅助字段或其他列；实体出现在
  结果中本身就表示其满足筛选条件。
- 对来源记录集合，只有当 Knowledge Documents 为该来源表显式定义完整主键或记录标识列集合时，
  才必须返回它们（例如：序号、编号、流水号、记录号、ID、SerialNo、RecordNo、RowNo）。这些列
  标识被请求的记录，是必要的输出列，不是证明或上下文列。若 Knowledge Documents 没有为该来源表
  定义可靠的记录标识列，但提交来源暴露了其他忠实于来源且可区分不同提交记录的列，则必须至少返回
  一列这类区分列。若没有定义或观测到可靠区分列，保留原始行粒度，不得要求、恢复或虚构标识列；
  缺少标识列绝不允许去重、DISTINCT、GROUP BY、drop_duplicates、IS NOT NULL 或 NULL/空值过滤。
- 对任何结果类型，只返回直接回答题目的列；来源记录集合还必须返回所需的记录标识列或记录区分列。
  拒绝所有其他证明、证据或无关上下文列。
- 当问题要求多个独立的标量答案、指标或属性时，每个组件都必须是一个有清晰名称的输出列，通常在
  一个逻辑行中。除非题目明确要求该布局，否则不得使用通用标签/值长表，也不得把多个组件塞进一个
  字符串、JSON、列表或分隔单元格。
- 除非问题要求所有字段、所有细节、记录、行或完整交易信息，不得返回完整的来源表结构。
- 结果必须直接回答所请求的内容。若问题要求实体、消息、评论、描述、标题、名称、正文或其他人类
  可读值，只有 ID 的答案不充分，除非问题明确请求 identifier。
- 除非问题规定格式，拆分的 `first_name`/`last_name` 或单个 `full_name` 都可接受。若请求的实体
  同时有全名和缩写/短名称，且问题没有指定其一，二者必须作为独立列返回。
- 除非 Knowledge Documents 显式将某列定义为该来源表的主键或记录标识列，否则绝不得要求提交答案
  携带主键、记录 ID、流水号或“序号”列。看似数字的列、行数、示例、提交列名或通用惯例都不足以证明
  这类标识列存在。

2. 值和行的保留
- 当要求原始来源值时，必须返回原单元格值；不得总结、改写、推断、聚合或以其他方式转换它们。
- 对来源记录集合，必须返回满足题目条件的每一条来源记录，包括请求的非键值为 NULL、空字符串、
  零，或与另一条记录完全相同的记录。只要主键存在，即使其他请求值全部相同或为 NULL，该记录也
  必须保留并展示。
- 拒绝零数据行的答案，即使存在表头：`prediction.csv` 将没有预测数据。应要求提交最可能且有支持的行。
- 必须返回完整的匹配行集合。不得把有界概览、看似合理的计数或重复行计数当作可以漏行的证据。
- 对原始检索、列出、展示或查找任务，拒绝对请求输出值使用 `IS NOT NULL`、空字符串、`TRIM(...) != ''`
  或等效过滤，除非题目明确要求非空、非 NULL、有效、存在或当前值。若计算、排序或极值查询在数学上
  必须排除这些值，则允许该过滤。

3. 行范围与转换
- 拒绝 `LIMIT`、`TOP`、Python 切片或其他行截断，除非问题明确要求 top/bottom/first/last/latest/oldest N、
  特定序数记录、样本、快照或摘要，或显式排序/极值任务在数学上必须限制行数。
- 实体集合去重（强制性检查）：对实体集合，按请求的实体标识输出列的完整组合去重。若
  `Submitted Answer Structure Overview.duplicate_row_count > 0`，必须拒绝重复的完整实体行，并要求
  SQL `DISTINCT` 或等价的 Python 去重。不得合并名称相似、别名或简称相近的实体；只能对最终实体
  输出值完全相同的行去重。
- 来源记录保留（强制性检查）：绝不得要求或应用对来源记录集合的去重。两行的非键内容相同、但
  主键不同，仍是两条不同记录。若 `DISTINCT`、`GROUP BY`、`drop_duplicates`、`IS NOT NULL`、
  空字符串过滤或等价操作删除了满足条件的主键记录，必须拒绝，除非题目明确要求该转换。
- 除非问题明确要求分组摘要、计数或其他聚合结果，否则拒绝 `GROUP BY`、聚合、行折叠或等效转换。

## 反馈边界

给出最窄的答案范围修复：说明所需输出类型或无效操作，并要求重新提交修正后的结果。不得要求主
agent 去读文档/图片、验证 join、选择来源或重新解释字段；但原问题依赖视频规则且缺少时间线/帧/视觉回执证据
时除外，此时必须要求补齐该视频证据，且在视频规则被观察到前不得要求改变答案范围。

## 输出格式

仅回复一个合法的 JSON 对象：
{
  "valid": true,
  "rationale": "简要审计摘要，不含隐藏推理。",
  "issues": []
}

或：
{
  "valid": false,
  "rationale": "简要审计摘要，不含隐藏推理。",
  "issues": ["针对性的交付或答案范围修复指令。"]
}
""".strip()


def _build_validation_request(
    question: str,
    answer: dict[str, Any],
    validation_history: list[dict[str, Any]] | None = None,
    answer_truncated: bool = False,
    answer_row_count: int | None = None,
    preview_row_limit: int | None = None,
    answer_structure_overview: dict[str, Any] | None = None,
    submission_risk_kinds: list[str] | None = None,
    submission_source: dict[str, Any] | None = None,
    submission_field_context: dict[str, Any] | None = None,
    supporting_source_evidence: dict[str, Any] | None = None,
    knowledge_docs: list[dict[str, Any]] | None = None,
) -> str:
    """Build bounded validator context without answer row samples."""
    del answer
    metadata: dict[str, Any] = {"answer_truncated_for_validator": answer_truncated}
    if answer_row_count is not None:
        metadata["stored_answer_row_count"] = answer_row_count
    if preview_row_limit is not None:
        metadata["legacy_answer_preview_row_limit"] = preview_row_limit
    if submission_risk_kinds:
        metadata["submission_risk_kinds"] = submission_risk_kinds

    parts = [
        f"## Original Question\n{question}\n",
        "## Answer Metadata\n"
        "Use this only to distinguish the complete stored submission from the bounded "
        "context shown to you. The validator context is intentionally bounded to prevent "
        "an overlong prompt: the runtime retains and scores the complete submitted answer "
        "separately, even when previews, examples, tool observations, or evidence excerpts "
        "are truncated. Therefore, `answer_truncated_for_validator`, "
        "`legacy_answer_preview_row_limit`, truncation notices, and truncated examples "
        "describe only context delivery limits. They are not evidence that the submitted "
        "answer omitted rows, is incomplete, or was capped. Never reject an answer solely "
        "because of these context-bound signals. Assess row completeness only from the "
        "original question and verifiable row-limiting operations in `Submission Source`, "
        "such as `LIMIT`, `TOP`, or Python slicing. Tool evidence and the answer "
        "structure overview must not be used to infer missing source "
        "records. "
        "`submission_risk_kinds` lists auto-detected operations in the submission's "
        "source query (e.g., `null_filter` = IS NOT NULL / dropna, "
        "`row_collapse` = GROUP BY / aggregation, `limit` = LIMIT / slicing, "
        "`type_cast` = CAST / astype / type conversion). "  # ← 新增 type_cast
        "Use these signals to enforce answer-scope rules.\n"
        "```json\n"
        f"{json.dumps(metadata, ensure_ascii=False, indent=2)}\n"
        "```\n",
    ]

    if knowledge_docs:
        knowledge_texts = []
        for doc in knowledge_docs:
            if isinstance(doc, dict) and isinstance(doc.get("content"), str) and doc["content"].strip():
                knowledge_texts.append(doc["content"].strip())
        if knowledge_texts:
            parts.append(
                "## Knowledge Documents (authoritative field definitions)\n"
                "These are the task-provided knowledge.md documents. Use them only "
                "to identify explicit field definitions, requested source tables, "
                "primary keys, and record-identifier columns. Require a primary-key "
                "or record-identifier column in a source-record answer only when the "
                "relevant knowledge document explicitly defines it. If no such "
                "identifier is defined, apply the system rule for observable "
                "source-faithful distinguishing columns, and never treat the missing "
                "identifier as permission to deduplicate or filter NULL values.\n\n"
                + "\n\n---\n\n".join(knowledge_texts)
                + "\n"
            )

    if submission_source is not None:
        parts.append(
            "## Submission Source\n"
            "The exact tool and arguments used to produce the submitted answer. "
            "Inspect the SQL queries, Python code, or extraction parameters to detect "
            "IS NOT NULL, GROUP BY, LIMIT, dropna, deduplication, aggregation, or "
            "other row-scope operations that may violate answer-scope rules.\n"
            "```json\n"
            f"{json.dumps(submission_source, ensure_ascii=False, indent=2)}\n"
            "```\n"
        )

    if submission_field_context is not None:
        parts.append(
            "## Submission Field Context\n"
            "Programmatic source-field whitelist extracted from the submitted SQL "
            "or Python SQL calls. Use `field_universe` as the available source "
            "columns when deciding which columns may be requested or excluded. "
            "Use `output_lineage` to understand SQL aliases. This context is not "
            "row evidence and contains no source row samples; do not recompute "
            "source data from it. If `status` is partial or warnings are present, "
            "apply ordinary answer-scope rules without rejecting solely because "
            "this context is incomplete.\n"
            "```json\n"
            f"{json.dumps(submission_field_context, ensure_ascii=False, indent=2)}\n"
            "```\n"
        )

    if supporting_source_evidence is not None:
        parts.append(
            "## Supporting Source Evidence\n"
            "This is bounded successful tool evidence. Use it only for the narrow "
            "video-configured ranking exception in the system instructions; it is "
            "not evidence for source data, row values, joins, or other semantics.\n"
            "```json\n"
            f"{json.dumps(supporting_source_evidence, ensure_ascii=False, indent=2)}\n"
            "```\n"
        )

    if answer_structure_overview is not None:
        parts.append(
            "## Submitted Answer Structure Overview\n"
            "Contains no answer row samples. Use it only for physical table and visible "
            "format checks; it is never source or semantic evidence.\n"
            "```json\n"
            f"{json.dumps(answer_structure_overview, ensure_ascii=False, indent=2)}\n"
            "```\n"
        )
    if validation_history:
        parts.append(
            "## Previous Answer Validation History\n"
            "Use it only for consistent validation criteria.\n"
            "```json\n"
            f"{json.dumps(validation_history, ensure_ascii=False, indent=2)}\n"
            "```\n"
        )
    parts.append(
        "Apply matching-receipt delivery gates, or the fallback gates when no matching "
        "receipt exists. Respond with ONLY the JSON object."
    )
    return "\n".join(parts)


def _parse_validator_response(response_text: str) -> dict[str, Any] | None:
    text = response_text.strip()
    if text.startswith("```"):
        first_newline = text.index("\n") if "\n" in text else len(text)
        text = text[first_newline + 1 :]
        if text.endswith("```"):
            text = text[:-3].strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Answer validator returned non-JSON content: %s", text[:200])
        return None
    if not isinstance(parsed, dict) or "valid" not in parsed:
        logger.warning("Answer validator JSON response is not a valid decision object.")
        return None
    return parsed


def validate_answer(
    *,
    model: BaseChatModel,
    question: str,
    answer: dict[str, Any],
    validation_history: list[dict[str, Any]] | None = None,
    answer_truncated: bool = False,
    answer_row_count: int | None = None,
    preview_row_limit: int | None = None,
    answer_structure_overview: dict[str, Any] | None = None,
    submission_risk_kinds: list[str] | None = None,
    submission_source: dict[str, Any] | None = None,
    submission_field_context: dict[str, Any] | None = None,
    supporting_source_evidence: dict[str, Any] | None = None,
    knowledge_docs: list[dict[str, Any]] | None = None,
    retry_event_callback: Any | None = None,
) -> dict[str, Any]:
    """Validate a submitted answer with one LLM call and fail open on technical errors."""
    from data_agent_baseline.tools.truncation import TRUNCATION_SUFFIX

    answer = copy.deepcopy(answer)
    if isinstance(answer.get("rows"), list) and answer["rows"]:
        if answer["rows"][-1] == TRUNCATION_SUFFIX:
            answer["rows"].pop()

    messages = [
        SystemMessage(content=ANSWER_VALIDATOR_SYSTEM_PROMPT),
        HumanMessage(
            content=_build_validation_request(
                question,
                answer,
                validation_history,
                answer_truncated=answer_truncated,
                answer_row_count=answer_row_count,
                preview_row_limit=preview_row_limit,
                answer_structure_overview=answer_structure_overview,
                submission_risk_kinds=submission_risk_kinds,
                submission_source=submission_source,
                submission_field_context=submission_field_context,
                supporting_source_evidence=supporting_source_evidence,
                knowledge_docs=knowledge_docs,
            )
        ),
    ]
    try:
        ai_message = invoke_model_with_retries(
            model,
            messages,
            on_retry_event=retry_event_callback,
            timeout_seconds=120.0,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Answer validator LLM call failed; skipping validation: %s", exc)
        return {
            "valid": True,
            "issues": [],
            "rationale": "",
            "validator_error": str(exc),
            "raw_response": None,
        }

    response_text = ""
    if isinstance(ai_message.content, str):
        response_text = ai_message.content
    elif isinstance(ai_message.content, list):
        response_text = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in ai_message.content
        )
    parsed = _parse_validator_response(response_text)
    if parsed is None:
        logger.warning("Answer validator response parsing failed; skipping validation.")
        return {
            "valid": True,
            "issues": [],
            "rationale": "",
            "validator_error": "Failed to parse validator response",
            "raw_response": response_text if response_text else None,
        }
    return {
        "valid": bool(parsed.get("valid", True)),
        "rationale": str(parsed.get("rationale", "") or ""),
        "issues": list(parsed.get("issues", [])),
        "raw_response": response_text,
    }
