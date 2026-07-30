"""Ambiguity analysis agent.

Before the main agent receives the question, this module performs a single LLM
call to identify semantic ambiguities that could lead to wrong answers.  It does
NOT field-bind, does NOT resolve ambiguities, and does NOT produce SQL or
execution plans.  It produces a risk checklist + verification questions +
candidate interpretations for the main agent to investigate with real data.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from data_agent_baseline.model_retry import invoke_model_with_retries

logger = logging.getLogger(__name__)

AMBIGUITY_TYPES = frozenset({
    # 当前分支原有（8 个）
    "field_binding",
    "metric_definition",
    "entity_resolution",
    "filter_semantics",
    "time_range",
    "grain",
    "join_path",
    "output_format",
    # LangGraph PerceptionAgent 补充（4 个）
    "geographic_scope",
    "rank_position",
    "per_unit_ratio",
    "tie_handling",
})


AMBIGUITY_ANALYZER_SYSTEM_PROMPT = """\
You are a pre-risk-identification assistant for a data analysis benchmark.

You will receive:
1. A raw user question.
2. A schemas listing extracted from a data catalog.  Each line shows \
a field path (asset_path.field_name), its type, field statistics (cardinality, \
missing count, primary key flag), sample distinct values, numeric range, and \
optionally a description and note.
3. Knowledge documents providing domain context about the dataset. \
These documents describe the problem domain, entity relationships, data definitions, \
and other background information relevant to answering the question.

Your job is two-fold:
  A. Parse the question structurally — extract entities, filters, metrics, \
     expected answer shape, and output grain.
  B. Identify the 2 highest-risk semantic ambiguities that could lead to a \
     wrong answer.  Think of yourself as a risk auditor: you surface what the \
     downstream agent should NOT take for granted.  If there are fewer than 2 \
     genuine ambiguities, list only those.  Do NOT fabricate ambiguities just \
     to fill a quota.

You MUST NOT resolve ambiguities or choose interpretations.
You MUST NOT produce field bindings or final answers.
You MUST NOT produce SQL or an execution plan.
You MUST NOT invent fields that are not present in the schemas.

## Question parsing rules (from PerceptionAgent)

1. Preserve complete filter phrases and noun phrases. Do not shrink a multi-word \
   filter, modifier, or scope phrase to only a proper name.
2. Separate output entities, filter scopes, metric concepts, operations, and \
   answer shape.
3. Identify the head entity being modified by qualifiers. For example, \
   distinguish a named geographic qualifier from the entity level it modifies.
4. Unless explicitly asked for an ID, always set column_hint to human-readable \
   display fields (e.g., name, title, text) for "what is" queries, prioritizing \
   content over system identifiers.
5. Keep the analysis compact and task-focused. Use short strings.

## Ambiguity types (fixed enum)

- field_binding: a phrase that could map to multiple different schema fields
- metric_definition: how a metric is calculated (count, sum, avg, distinct, ratio, etc.)
- entity_resolution: which real-world entity a name/code/ID refers to (splits, aliases, duplicates)
- filter_semantics: how a filter condition should be applied (boundaries, comparison targets, inclusive/exclusive)
- time_range: which time period or date boundaries apply ("latest", "current", "season", "year")
- grain: what level of detail each row represents (per person, per event, per order, per group)
- join_path: which relationship between tables/entities is correct when multiple paths exist
- output_format: whether the result should be a single value, sorted list, grouped set, etc.
- geographic_scope: a place-like phrase could map to multiple geographic or administrative levels (county vs city vs district vs school)
- rank_position: rank and position/order may not be interchangeable (e.g., "first" may mean rank=1 or position=1)
- per_unit_ratio: a per-unit phrase may require a ratio instead of a total (e.g., "per capita", "per item")
- tie_handling: an extreme/ranking question may require preserving ties (e.g., "lowest score" with multiple rows at the minimum)

## Output Format

You MUST respond with ONLY a valid JSON object (no markdown fences, no explanation):

{
  "question_intent": {
    "entities": ["entity1", "entity2"],
    "filters": ["filter description 1"],
    "metrics": ["metric phrase 1"],
    "requested_output": "what the question is asking for",
    "grain": "best guess at row/entity/event level, or 'unknown'",
    "expected_answer_shape": {
      "row_shape": "single_row|multiple_rows|unknown",
      "column_hint": "short description of requested output columns"
    }
  },
  "ambiguities": [
    {
      "id": "amb_001",
      "phrase": "the ambiguous phrase from the question",
      "type": "one of the 12 ambiguity types above",
      "clarifying_question": "a question whose answer would resolve this ambiguity",
      "required_verification": [
        "concrete step, e.g. probe distinct values of X, COUNT rows grouped by Y"
      ]
    }
  ],
  "resolved_by_knowledge": [
    {
      "phrase": "phrase from the question",
      "chosen_meaning": "the disambiguated meaning",
      "evidence": "quote or reference from the knowledge documents"
    }
  ],
  "non_ambiguous_candidates": [
    {
      "phrase": "phrase from the question with a clear field mapping",
      "candidate_fields": ["asset_path.field_name"],
      "note": "low risk — still requires data verification"
    }
  ]
}

## Rules

- question_intent: capture what the question appears to be asking, at face value.
- entities: concrete people, places, organizations, products, codes mentioned. \
  Distinguish the head entity from its qualifiers (e.g., "Riverside County" is a \
  geographic scope qualifier, not the output entity).
- filters: conditions that narrow down the data, expressed as complete natural \
  language phrases. Preserve multi-word scope phrases intact.
- metrics: quantitative concepts the question asks about (count, average, max, \
  ratio, score, rank, etc.).
- requested_output: what the answer should contain.
- grain: your best guess at what one row should represent. Use "unknown" if unclear.
- expected_answer_shape.row_shape: "single_row" for questions expecting exactly one \
  result row, "multiple_rows" otherwise, "unknown" if unclear.
- expected_answer_shape.column_hint: short description of requested output columns. \
  Unless the question explicitly asks for an ID/identifier, prefer human-readable \
  display fields (name, title, text, body, label) over system identifiers or codes.
- ambiguities: list at most 2, only the highest-risk ones. Assign unique ids \
  (amb_001, amb_002). Quality over quantity — fewer is better than filler.
- Each ambiguity MUST have exactly one type from the fixed enum above.
- clarifying_question: a natural language question that, if answered, would resolve \
  the ambiguity. Be specific and concrete.
- required_verification: concrete, actionable data-probe steps for the downstream \
  agent. Prefer actual data probes over schema-only lookups.
- resolved_by_knowledge: only include entries where a knowledge document provides \
  an explicit, unambiguous definition. Do NOT guess or infer from field names alone.
- non_ambiguous_candidates: phrases with a clear 1:1 schema field mapping. Even \
  these require data verification.
- If no knowledge documents are provided or they are empty, resolved_by_knowledge \
  must be an empty list.
- Knowledge documents are background context only. Do not reference them in \
  candidate_fields.
- The downstream agent will verify every candidate with actual data probes before \
  choosing fields.
"""

# Chinese translation of the system prompt, for reference and documentation.
# This block mirrors the English prompt above.
_AMBIGUITY_ANALYZER_SYSTEM_PROMPT_ZH = """\
---
您是一个数据分析基准测试的前置风险识别助手。

您将收到：
1. 一个原始用户问题。
2. 从数据目录中提取的模式清单。每行显示一个字段路径（asset_path.field_name）、\
其类型、字段统计信息（基数、缺失计数、主键标志）、样本唯一值、数值范围以及可选的描述和注释。
3. 提供数据集领域上下文的知识文档。这些文档描述了问题领域、实体关系、数据定义以及\
与回答问题相关的其他背景信息。

您的任务有两项：
  A. 结构化解析问题——提取实体、过滤条件、指标、预期答案形状和输出粒度。
  B. 识别问题中风险最高的 2 个可能导致错误答案的语义歧义。您的角色是风险审计员：\
     暴露下游 Agent 不能想当然的地方。如果不足 2 个真正的歧义，只列出存在的即可。\
     不要为了凑数而编造歧义。

您**不得**消解歧义或选择解释。
您**不得**进行字段绑定或给出最终答案。
您**不得**生成 SQL 或执行计划。
您**不得**编造不存在于模式中的字段。

## 问题解析规则（来自 PerceptionAgent）

1. 保留完整的过滤短语和名词短语。不要将多词组成的过滤词、修饰词或范围短语缩减为单一专有名词。
2. 分离输出实体、过滤范围、指标概念、操作类型和答案形状。
3. 识别被限定词修饰的中心实体。例如，区分一个命名地理限定词和它所修饰的实体层级。
4. 除非问题明确要求 ID，否则对 "what is" 类查询始终将 column_hint 设为人可读的展示\
   字段（如 name、title、text），优先展示内容而非系统标识符。
5. 保持分析紧凑、聚焦任务。使用简短字符串。

## 歧义类型（固定枚举）

- field_binding：某个短语可能对应多个不同的模式字段
- metric_definition：指标的计算方式不明确（计数、求和、平均、去重计数、比率等）
- entity_resolution：名称/代码/ID 指代哪个真实实体（拆分、别名、重名）
- filter_semantics：筛选条件的边界或比较对象不明确（是否包含边界值、比较目标等）
- time_range：时间范围或日期边界不明确（"最新"、"当前"、"赛季"、"年度"）
- grain：每行数据的粒度不明确（每人、每次事件、每订单、每组聚合）
- join_path：多表之间的正确关联路径不明确
- output_format：结果应为单个值、排序列表、分组集合等不明确
- geographic_scope：地名类短语可能对应多个地理或行政层级（县 vs 市 vs 区 vs 学校）
- rank_position：排名和位置/顺序可能不可互换（如 "first" 可能指 rank=1 或 position=1）
- per_unit_ratio："每单位"类短语可能需要比率而非总数（如 "人均"、"每项"）
- tie_handling：极值/排名问题可能需要保留所有并列值（如 "最低分" 有多个并列最小值的行）

## 输出格式

您**必须**仅响应一个有效的 JSON 对象（没有 markdown 围栏，没有解释）：

{
  "question_intent": {
    "entities": ["实体1", "实体2"],
    "filters": ["筛选条件描述 1"],
    "metrics": ["指标短语 1"],
    "requested_output": "问题要求输出什么",
    "grain": "对行/实体/事件级别的最佳猜测，或 'unknown'",
    "expected_answer_shape": {
      "row_shape": "single_row|multiple_rows|unknown",
      "column_hint": "对要求的输出列的简短描述"
    }
  },
  "ambiguities": [
    {
      "id": "amb_001",
      "phrase": "问题中的歧义短语",
      "type": "上述 12 种歧义类型之一",
      "clarifying_question": "回答后能消解该歧义的问题",
      "required_verification": [
        "下游 Agent 应执行的具体验证步骤，如：探测 X 的去重值、按 Y 分组 COUNT 行数"
      ]
    }
  ],
  "resolved_by_knowledge": [
    {
      "phrase": "问题中的短语",
      "chosen_meaning": "消歧后的含义",
      "evidence": "知识文档中的引用或参考"
    }
  ],
  "non_ambiguous_candidates": [
    {
      "phrase": "问题中字段映射明确的短语",
      "candidate_fields": ["asset_path.field_name"],
      "note": "低风险 — 仍需数据验证"
    }
  ]
}

## 规则

- question_intent：捕捉问题表面上在问什么。
- entities：问题中具体提到的人物、地点、组织、产品、代码等。区分中心实体和它的限定词\
 （如 "Riverside County" 是地理范围限定，而非输出实体）。
- filters：用于筛选数据的条件，以完整自然语言短语描述。保留多词范围短语的完整性。
- metrics：问题中涉及的量化概念（计数、平均值、最大值、比率、得分、排名等）。
- requested_output：答案应包含的内容描述。
- grain：对一行数据代表什么的最佳猜测。如果不清楚，使用 "unknown"。
- expected_answer_shape.row_shape：若问题预期恰好一行结果则为 "single_row"，\
  否则为 "multiple_rows"，不确定则为 "unknown"。
- expected_answer_shape.column_hint：对输出列的简短描述。除非问题明确要求 ID/标识符，\
  否则优先推荐人可读的展示字段（name、title、text、body、label），而非系统标识符或代码。
- ambiguities：最多列出 2 个风险最高的歧义。分配唯一 id（amb_001, amb_002）。\
  质量重于数量——宁缺毋滥。
- 每个歧义必须有且仅有一个来自上述固定枚举的 type。
- clarifying_question：一个自然语言问题，回答后能消解该歧义。要具体、有针对性。
- required_verification：给下游 Agent 的具体、可操作的数据探测步骤。优先实际数据探测\
  而非仅模式查询。
- resolved_by_knowledge：仅当知识文档提供了明确、无歧义的定义时才列入。不要仅凭字段名\
  猜测或推断。
- non_ambiguous_candidates：字段映射清晰的短语，低风险但仍需数据验证。
- 如果没有提供知识文档或为空，resolved_by_knowledge 必须为空列表。
- 知识文档仅作为背景上下文，不要在 candidate_fields 中引用。
- 下游 Agent 将在选择字段前通过实际数据探查验证每一个候选。
"""


def analyze_ambiguity(
    *,
    model: BaseChatModel,
    question: str,
    schemas: list[dict[str, Any]] | None = None,
    knowledge_docs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    schemas_str = _build_schemas_listing(schemas) if schemas else ""
    knowledge_str = _build_knowledge_listing(knowledge_docs) if knowledge_docs else ""

    user_parts: list[str] = [
        "<user_question>\n"
        f"{question}\n"
        "</user_question>",
    ]
    if schemas_str:
        user_parts.append(
            "<schemas>\n"
            f"{schemas_str}\n"
            "</schemas>"
        )
    if knowledge_str:
        user_parts.append(knowledge_str)
    user_parts.append(
        "<risk_label_guidance>\n"
        "aggregation_grain: The grouping or row grain could change the answer.\n"
        "geographic_scope_ambiguity: A place-like phrase could map to multiple geographic or administrative levels.\n"
        "entity_level_ambiguity: The question may refer to different entity levels, such as item versus group.\n"
        "metric_operation_ambiguity: The metric operation or comparison target is not fully explicit.\n"
        "filter_scope_ambiguity: A filter phrase could apply to different entities or scopes.\n"
        "join_key_ambiguity: The task likely requires linking assets and the key choice may matter.\n"
        "rank_position_ambiguity: Rank and position/order may not be interchangeable.\n"
        "type_category_level_ambiguity: A requested type/category label may belong to different entity levels.\n"
        "per_unit_ratio: A per-unit phrase may require a ratio instead of a total.\n"
        "tie_handling_ambiguity: An extreme/ranking question may require preserving ties.\n"
        "</risk_label_guidance>"
    )
    user_parts.append(
        "Perform two tasks:\n"
        "1. Parse the question structurally: entities, filters, metrics, expected_answer_shape, grain.\n"
        "2. Identify the 2 highest-risk semantic ambiguities. "
        "Surface risks and candidate interpretations only — do NOT resolve, "
        "do NOT field-bind, do NOT produce SQL."
    )
    user_message = "\n\n".join(user_parts)

    messages = [
        SystemMessage(content=AMBIGUITY_ANALYZER_SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ]

    try:
        ai_message = invoke_model_with_retries(model, messages)
    except Exception as exc:
        logger.warning("Ambiguity analyzer LLM call failed; skipping analysis: %s", exc)
        return _fallback_ambiguity_result()

    response_text = _extract_response_text(ai_message)
    parsed = _parse_ambiguity_response(response_text)
    if parsed is None:
        logger.warning("Ambiguity analyzer response parsing failed; skipping analysis.")
        return _fallback_ambiguity_result()

    parsed["ambiguities"] = _validate_ambiguities(parsed.get("ambiguities", []))
    if schemas:
        parsed["non_ambiguous_candidates"] = _validate_non_ambiguous_candidates(
            parsed.get("non_ambiguous_candidates", []), schemas
        )
    parsed["resolved_by_knowledge"] = _validate_resolved_by_knowledge(
        parsed.get("resolved_by_knowledge", [])
    )
    return parsed


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _parse_ambiguity_response(response_text: str) -> dict[str, Any] | None:
    text = response_text.strip()

    if text.startswith("```"):
        first_newline = text.index("\n") if "\n" in text else len(text)
        text = text[first_newline + 1:]
        if text.endswith("```"):
            text = text[:-3].strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Ambiguity analyzer returned non-JSON content: %s", text[:200])
        return None

    if not isinstance(parsed, dict):
        logger.warning("Ambiguity analyzer JSON response is not an object.")
        return None

    qi = parsed.get("question_intent", {})
    if not isinstance(qi, dict):
        qi = {}

    answer_shape = qi.get("expected_answer_shape")
    if not isinstance(answer_shape, dict):
        answer_shape = {}

    return {
        "question_intent": {
            "entities": _list_str(qi.get("entities")),
            "filters": _list_str(qi.get("filters")),
            "metrics": _list_str(qi.get("metrics")),
            "requested_output": _str_or(qi.get("requested_output"), ""),
            "grain": _str_or(qi.get("grain"), ""),
            "expected_answer_shape": {
                "row_shape": _str_or(answer_shape.get("row_shape"), "unknown"),
                "column_hint": _str_or(answer_shape.get("column_hint"), ""),
            },
        },
        "ambiguities": parsed.get("ambiguities", []),
        "resolved_by_knowledge": parsed.get("resolved_by_knowledge", []),
        "non_ambiguous_candidates": parsed.get("non_ambiguous_candidates", []),
    }


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_ambiguities(
    ambiguities: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not isinstance(ambiguities, list):
        return []
    cleaned: list[dict[str, Any]] = []
    for amb in ambiguities:
        if not isinstance(amb, dict):
            continue
        amb_id = amb.get("id")
        phrase = amb.get("phrase")
        amb_type = amb.get("type")
        if not isinstance(amb_id, str) or not amb_id.strip():
            continue
        if not isinstance(phrase, str) or not phrase.strip():
            continue
        if not isinstance(amb_type, str) or amb_type.strip() not in AMBIGUITY_TYPES:
            logger.info("Ambiguity stripped: id=%r type=%r not in valid types", amb_id, amb_type)
            continue

        cleaned.append({
            "id": amb_id.strip(),
            "phrase": phrase.strip(),
            "type": amb_type.strip(),
            "clarifying_question": _str_or(amb.get("clarifying_question"), ""),
            "required_verification": _list_str(amb.get("required_verification")),
        })
    return cleaned[:2]


def _validate_non_ambiguous_candidates(
    non_ambiguous_candidates: list[dict[str, Any]],
    schemas: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not isinstance(non_ambiguous_candidates, list):
        return []
    valid_paths = _collect_valid_paths(schemas)
    cleaned: list[dict[str, Any]] = []
    for entry in non_ambiguous_candidates:
        if not isinstance(entry, dict):
            continue
        phrase = entry.get("phrase")
        if not isinstance(phrase, str) or not phrase.strip():
            continue
        raw_fields = entry.get("candidate_fields")
        if not isinstance(raw_fields, list):
            cleaned.append({"phrase": phrase.strip(), "candidate_fields": [], "note": _str_or(entry.get("note"), "")})
            continue
        valid_fields = [f for f in raw_fields if isinstance(f, str) and f.strip() in valid_paths]
        if raw_fields and not valid_fields:
            continue
        cleaned.append({
            "phrase": phrase.strip(),
            "candidate_fields": valid_fields,
            "note": _str_or(entry.get("note"), ""),
        })
    return cleaned


def _validate_resolved_by_knowledge(
    resolved: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not isinstance(resolved, list):
        return []
    cleaned: list[dict[str, Any]] = []
    for entry in resolved:
        if not isinstance(entry, dict):
            continue
        phrase = entry.get("phrase")
        if not isinstance(phrase, str) or not phrase.strip():
            continue
        cleaned.append({
            "phrase": phrase.strip(),
            "chosen_meaning": _str_or(entry.get("chosen_meaning"), ""),
            "evidence": _str_or(entry.get("evidence"), ""),
        })
    return cleaned


# ---------------------------------------------------------------------------
# Shared helpers (reused from question_analyzer.py)
# ---------------------------------------------------------------------------


def _extract_response_text(ai_message: Any) -> str:
    if isinstance(ai_message.content, str):
        return ai_message.content
    if isinstance(ai_message.content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in ai_message.content
        )
    return ""


def _build_schemas_listing(schemas: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    field_count = 0
    max_fields = 200

    for schema in schemas:
        asset = schema.get("asset_path", "?")
        kind = schema.get("kind", "?")

        if kind == "sqlite":
            for table in schema.get("tables", []):
                tname = table.get("name", "?")
                for field in table.get("fields", []):
                    if field_count >= max_fields:
                        break
                    lines.append(_format_field_line(asset, field, table_name=tname))
                    field_count += 1
        else:
            for field in schema.get("fields", []):
                if field_count >= max_fields:
                    break
                lines.append(_format_field_line(asset, field))
                field_count += 1

        if field_count >= max_fields:
            omitted = _count_total_fields(schemas) - max_fields
            lines.append(f"... ({omitted} more fields omitted)")
            break

    return "\n".join(lines)


def _build_knowledge_listing(
    knowledge_docs: list[dict[str, Any]],
    max_tokens: int = 2048,
) -> str:
    from data_agent_baseline.token_utils import count_tokens, truncate_by_tokens

    parts: list[str] = []
    remaining = max_tokens
    for doc in knowledge_docs:
        if remaining <= 0:
            break
        asset = doc.get("asset_path", "unknown")
        content = doc.get("content", "")
        if not content:
            continue
        wrapper_overhead = max(len(asset) // 4 + 10, 20)
        budget = remaining - wrapper_overhead
        if budget <= 0:
            break
        truncated = truncate_by_tokens(content, budget)
        remaining -= count_tokens(truncated) + wrapper_overhead
        parts.append(f'<document path="{asset}">\n{truncated}\n</document>')
    if not parts:
        return ""
    return "<knowledge_documents>\n" + "\n\n".join(parts) + "\n</knowledge_documents>"


def _format_field_line(
    asset: str, field: dict[str, Any], table_name: str | None = None
) -> str:
    name = field.get("name", "?")
    ftype = field.get("type", "?")
    prefix = f"{asset}.{table_name}" if table_name else asset
    line = f"{prefix}.{name} ({ftype})"

    stats_parts: list[str] = []
    card = field.get("cardinality")
    if card is not None:
        stats_parts.append(f"card={card}")
    missing = field.get("missing_count")
    if missing is not None and missing > 0:
        stats_parts.append(f"miss={missing}")
    if field.get("primary_key"):
        stats_parts.append("pk")
    if stats_parts:
        line += " | " + ", ".join(stats_parts)

    distinct = field.get("distinct_values")
    if isinstance(distinct, list) and distinct:
        sample = [str(v["value"] if isinstance(v, dict) else v) for v in distinct[:5]]
        line += " | vals: " + ", ".join(sample)

    min_v = field.get("min_value")
    max_v = field.get("max_value")
    if min_v is not None and max_v is not None:
        line += f" | [{min_v}, {max_v}]"

    desc = field.get("description")
    if isinstance(desc, str) and desc.strip():
        line += f" — {desc.strip()}"
    note = field.get("note")
    if isinstance(note, str) and note.strip():
        line += f" [{note.strip()}]"
    return line


def _count_total_fields(schemas: list[dict[str, Any]]) -> int:
    total = 0
    for schema in schemas:
        if schema.get("kind") == "sqlite":
            for table in schema.get("tables", []):
                total += len(table.get("fields", []))
        else:
            total += len(schema.get("fields", []))
    return total


def _collect_valid_paths(schemas: list[dict[str, Any]]) -> set[str]:
    paths: set[str] = set()
    for schema in schemas:
        asset = schema.get("asset_path", "")
        if not asset:
            continue
        if schema.get("kind") == "sqlite":
            for table in schema.get("tables", []):
                tname = table.get("name", "")
                for field in table.get("fields", []):
                    fname = field.get("name", "")
                    if fname:
                        paths.add(f"{asset}.{fname}")
                        if tname:
                            paths.add(f"{asset}.{tname}.{fname}")
        else:
            for field in schema.get("fields", []):
                fname = field.get("name", "")
                if fname:
                    paths.add(f"{asset}.{fname}")
    return paths


def _fallback_ambiguity_result() -> dict[str, Any]:
    return {
        "question_intent": {
            "entities": [],
            "filters": [],
            "metrics": [],
            "requested_output": "",
            "grain": "",
            "expected_answer_shape": {
                "row_shape": "unknown",
                "column_hint": "",
            },
        },
        "ambiguities": [],
        "resolved_by_knowledge": [],
        "non_ambiguous_candidates": [],
    }


# ---------------------------------------------------------------------------
# Tiny type-coercion helpers
# ---------------------------------------------------------------------------


def _list_str(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(v).strip() for v in value if isinstance(v, str) and v.strip()]


def _str_or(value: Any, default: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return default
