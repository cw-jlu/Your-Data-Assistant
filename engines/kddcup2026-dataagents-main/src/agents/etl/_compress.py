"""LLM compression and paragraph sampling."""

from __future__ import annotations

import logging
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

from agents.etl._columns import metadata_by_name
from agents.etl._constants import (
    COMPRESS_CHUNK_SIZE,
    FUND_FIELD_DISAMBIGUATION_PROMPT,
    LLM_CALL_TIMEOUT,
    MAX_WORKERS,
    RECORD_LABEL_EXAMPLES,
    SCHEMA_SAMPLE_TOKEN_BUDGET,
)
from agents.etl._detect import detect_sections
from agents.etl._grouping import (
    entity_chunks_by_token_budget,
    group_paragraphs_by_entity,
)
from agents.etl._threading import submit_in_context

if TYPE_CHECKING:
    from agents.llm.types import ModelAdapter

logger = logging.getLogger(__name__)


def _field_type_instruction(field: str, field_type: str) -> str:
    type_text = field_type.lower()
    if "local_record_id" in type_text or "integer_scalar" in type_text:
        return (
            f"- {field}: {field_type}. Extract one scalar integer only. "
            'Convert labels like "档案 <N>" or "Record <N>" to "<N>". '
            'Reject ranges/groups/lists such as "197至217"; if the paragraph only '
            "states missing values for such a range/group, skip the paragraph."
        )
    if ("number" in type_text or "integer" in type_text) and "rank" not in type_text:
        return (
            f"- {field}: {field_type}. Extract only an explicit numeric scalar that "
            "matches this exact field; remove unit suffixes."
        )
    if "date" in type_text:
        return (
            f"- {field}: {field_type}. Extract only an explicit reporting date/period; "
            "use the final corrected value when corrections are described."
        )
    if "time" in type_text:
        return f"- {field}: {field_type}. Extract only an explicit clock/timestamp value."
    if "id" in type_text or "identity" in type_text:
        return (
            f"- {field}: {field_type}. Extract one scalar identity value only; "
            "do not use ranges, grouped labels, or narrative descriptions."
        )
    return f"- {field}: {field_type}. Extract only explicit values of this type."


def _format_schema_type_hint(
    schema_columns: list[str] | None,
    schema_field_types: dict[str, str] | None,
) -> str:
    if not schema_columns or not schema_field_types:
        return ""
    field_types = metadata_by_name(schema_columns, schema_field_types)
    lines = [
        _field_type_instruction(field, field_types[field])
        for field in schema_columns
        if field in field_types
    ]
    if not lines:
        return ""
    return (
        "\nField type constraints (obey strictly; omit values that violate the type):\n"
        + "\n".join(lines)
        + "\n"
    )


def _format_hard_type_rules(
    schema_columns: list[str] | None,
    schema_field_types: dict[str, str] | None,
) -> str:
    lines = [
        "HARD TYPE RULES:",
        "- ALL numeric values: NEVER include thousand separators (commas). Write 7031834, NOT 7,031,834.",
    ]
    if not schema_columns or not schema_field_types:
        return "\n" + "\n".join(lines) + "\n"

    field_types = metadata_by_name(schema_columns, schema_field_types)
    number_fields = [
        field
        for field in schema_columns
        if (
            "number" in field_types.get(field, "").lower()
            or "integer" in field_types.get(field, "").lower()
        )
        and "rank" not in field_types.get(field, "").lower()
    ]
    date_fields = [
        field for field in schema_columns if "date" in field_types.get(field, "").lower()
    ]
    boolean_fields = [
        field for field in schema_columns if "boolean" in field_types.get(field, "").lower()
    ]

    lines.extend(
        [
            "- If a value has the wrong type for a field, leave the value blank.",
            "- Do NOT put names, descriptions, or labels into number/date/boolean fields.",
        ]
    )
    if number_fields:
        lines.append(
            "- number/integer fields: numeric values only (integers or decimals); "
            "strip unit suffixes and thousand separators. A leading ~ marking an "
            "approximate value is allowed (see FUZZY NUMBERS). "
            f"Fields: {', '.join(number_fields)}."
        )
    if date_fields:
        lines.append(
            "- date fields: convert to YYYY-MM-DD (or YYYY-MM / YYYY if day/month unknown); "
            "leave blank if not a recognizable date. "
            f"Fields: {', '.join(date_fields)}."
        )
    if boolean_fields:
        lines.append(
            "- boolean fields: normalize to true or false; "
            "leave blank if the value is not boolean. "
            f"Fields: {', '.join(boolean_fields)}."
        )
    return "\n" + "\n".join(lines) + "\n"


def sample_paragraphs(text: str, first_n: int = 3, stride_n: int = 10) -> str:
    """Sample paragraphs: first few + evenly spaced from the rest of the file.

    This ensures the LLM sees all paragraph types (identity, physical stats,
    demographics, classification, etc.) rather than only the first section.
    """
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paras) <= first_n + stride_n:
        return "\n\n".join(paras)

    samples = paras[:first_n]
    remaining = paras[first_n:]
    step = max(1, len(remaining) // stride_n)
    for i in range(0, len(remaining), step):
        samples.append(remaining[i])
        if len(samples) >= first_n + stride_n:
            break

    return "\n\n".join(samples)


def sample_paragraphs_by_section(text: str, token_limit: int = SCHEMA_SAMPLE_TOKEN_BUDGET) -> str:
    """Sample whole paragraphs for schema field discovery.

    Small documents pass through IN FULL — complete evidence beats any
    sample.  Larger documents get a token budget spread across sections in
    proportion to their size, with CENTERED systematic sampling inside each
    section (k=1 picks the middle paragraph): section-boundary paragraphs
    are intro/outro narrative where field mentions are weakest, while the
    interior carries the value-bearing paragraphs (regression: task_59
    ed_consumerpriceindex — boundary-biased samples fed the schema call
    date narrative only and the CPI fields never entered the schema).
    Documents with <3 sections are sampled the same way as one big section.
    """
    from agents.llm.tokenizer import count_qwen_tokens

    sections = detect_sections(text)
    if not sections or len(sections) < 3:
        sections = [{"title": "", "content": text}]

    sec_paras: list[tuple[str, list[str]]] = []
    for sec in sections:
        paras = [
            p.strip() for p in sec["content"].split("\n\n") if p.strip() and len(p.strip()) > 50
        ]
        if paras:
            sec_paras.append((str(sec["title"]), paras))
    if not sec_paras:
        return ""

    def build(scale: float) -> str:
        parts: list[str] = []
        for title, paras in sec_paras:
            if title:
                parts.append(f"### {title}")
            k = min(len(paras), max(1, round(len(paras) * scale)))
            indices = sorted({round((j + 0.5) * len(paras) / k - 0.5) for j in range(k)})
            parts.extend(paras[i] for i in indices)
        return "\n\n".join(parts)

    sample = build(1.0)
    total_tokens = count_qwen_tokens(sample)
    if total_tokens <= token_limit:
        return sample

    scale = token_limit / total_tokens
    for _ in range(8):
        sample = build(scale)
        if count_qwen_tokens(sample) <= token_limit:
            return sample
        scale *= 0.8

    # One paragraph per section still exceeds the budget: flatten the whole
    # document and stride evenly across it.
    flat = [p for _, paras in sec_paras for p in paras]
    flat_tokens = count_qwen_tokens("\n\n".join(flat))
    keep = max(1, int(len(flat) * token_limit / max(1, flat_tokens)))
    indices = sorted({round((j + 0.5) * len(flat) / keep - 0.5) for j in range(keep)})
    return "\n\n".join(flat[i] for i in indices)


def _generate_compress_guide(
    adapter: ModelAdapter,
    prose_text: str,
    schema_columns: list[str] | None = None,
    schema_field_defs: dict[str, str] | None = None,
    schema_field_types: dict[str, str] | None = None,
) -> str | None:
    """Analyze document structure once, produce a reusable extraction system prompt."""
    if not schema_columns:
        return None

    paras = [p.strip() for p in prose_text.split("\n\n") if p.strip() and len(p.strip()) > 50]
    if not paras:
        return None
    if len(paras) <= 20:
        sample = "\n\n".join(paras)
    else:
        step = max(1, len(paras) // 20)
        sample = "\n\n".join(paras[::step][:20])

    field_defs = metadata_by_name(schema_columns, schema_field_defs)
    field_types = metadata_by_name(schema_columns, schema_field_types)
    defs_lines: list[str] = []
    for f in schema_columns:
        d = field_defs.get(f, "")
        t = field_types.get(f, "")
        parts = [f]
        if t:
            parts.append(f"type={t}")
        if d:
            parts.append(d)
        defs_lines.append("  - " + " | ".join(parts))
    defs_block = "\n".join(defs_lines)
    cols = ", ".join(schema_columns)

    prompt = textwrap.dedent(f"""\
        Below is a representative sample from a large prose data document.
        Each entity has a numeric record identifier (e.g. "档案 29", "记录 29").

        Target schema fields:
        {defs_block}

        DOCUMENT SAMPLE (paragraphs from evenly-spaced positions):
        ---
        {sample}
        ---

        Analyze this document thoroughly and produce a SYSTEM PROMPT that will
        guide another LLM to extract the correct field values from any
        paragraph of this document into pipe-separated key-value lines like:
          {cols}

        Your system prompt MUST cover:

        1. DOCUMENT LAYOUT: describe how the document is organized — are all
           fields for one record in a single paragraph, or does the same record
           appear across multiple paragraphs/sections with different fields
           each time?

        2. FIELD-TO-PROSE MAPPING: for EACH schema field, state the EXACT
           prose phrase template that provides its value. Quote the Chinese or
           English template with a placeholder X. Example:
             field_a ← "在本次交易前，该实体持有 X 股" → X
             field_b ← "此次股权变动涉及的股份数量为 X 股" → X

        3. CRITICAL DISAMBIGUATION: list pairs of fields that look similar
           in the prose and explain precisely how to tell them apart. This is
           the most important part — mistakes here cause wrong data.
           Pay special attention to percentage/ratio fields: a ratio stated
           alongside a QUANTITY (e.g. "涉及 X 股…占转让方持股的 Y") is NOT
           the same as a ratio stated alongside a HOLDING (e.g. "持有 X 股，
           占总股本的 Y%"). Map each ratio to the field whose SHARES COUNT
           appears in the same sentence.
           CRITICAL: when a record spans multiple paragraphs — one describing
           the transaction (transferred shares + ratio to total) and another
           describing the transferor's holding state (held shares + ratio to
           total) — explicitly instruct which paragraph supplies which field.
           "Before/after holding pct" comes ONLY from the holding-state
           paragraph (large share count, large pct ~1-50%), NEVER from the
           transaction paragraph (small transfer count, tiny ratio ~0.001-0.01%).

        4. MULTIPLE NAME FORMS: when the document provides MULTIPLE
           names or abbreviations for the same entity (e.g. a short trading
           name "国投金地ETF" AND a longer official name "国投瑞银沪深300金融
           地产ETF"), instruct the extractor to ALWAYS pick the SHORTEST form
           for abbreviation fields (secuabbr, chinamabbr, etc.). The short
           form is typically labeled "证券简称" or "市场简称"; the longer form
           is "官方简称" or "法定名称". Map each name form to the correct field.

        5. FUZZY NUMBER MARKING: if any values use approximate language
           ("约", "大约", "近", Chinese number words like "六百九十七万"),
           instruct the extractor to convert the numeral mechanically and
           prefix it with ~ (e.g. "约六百九十七万" → ~6970000) — never to
           attempt arithmetic on it; a downstream deterministic pass
           recomputes exact values from sibling fields.

        Your generated guide MUST incorporate these NON-NEGOTIABLE rules
        verbatim — do NOT rephrase, weaken, or omit any of them:

        A. PERCENTAGES: strip the "%" suffix and output the number as-is.
           "2.0%" → 2.0, "-1.55%" → -1.55.
           NEVER divide by 100 to convert to decimal form.

        B. ABBREVIATION FIELDS: fund entities typically have three naming
           tiers, longest → shortest. Map each to its EXACT schema column:
           (1) FULL NAME (全称/正式名称)
               → ChiName / fund_name. The longest, official registered name.
           (2) MARKET NAME (市场简称/常用名/通用简称/官方简称/公开简称)
               → ChiNameAbbr / fund_name_short. Shorter than full name but
               often still includes fund type suffix like "(LOF)".
           (3) TRADING NAME (证券简称/交易简称)
               → SecuAbbr / secuabbr. ALWAYS the shortest name — the label
               used on the exchange trading screen.
           ChiNameAbbr and SecuAbbr are DIFFERENT fields. NEVER copy the
           market name into SecuAbbr when a shorter trading name exists.
           Example: "市场简称为富国天丰强化收益债券(LOF)…交易简称为富国天丰"
           → ChiNameAbbr = "富国天丰强化收益债券(LOF)" (tier 2, longer)
           → SecuAbbr = "富国天丰" (tier 3, shortest). NOT the other way.
           Example: "通用简称为大摩资源优选混合(LOF)，交易代码为大摩资源"
           → ChiNameAbbr = "大摩资源优选混合(LOF)" (tier 2)
           → SecuAbbr = "大摩资源" (tier 3, shortest).
           If tier 3 does not exist, use tier 2 as SecuAbbr fallback.
           NEVER leave SecuAbbr empty when any name form is available.

        C. CORRECTED VALUES: when a value is stated as "initially X,
           confirmed/corrected to Y", output Y only. Never output X.

        D. FIELD TYPE DISCIPLINE: NEVER put a numeric ID/code into a text
           name/abbreviation field, or vice versa. Name/abbreviation fields
           (SecuAbbr, ChiNameAbbr, fund_name, etc.) contain readable text
           (Chinese, English, or mixed), NEVER a pure numeric code.
           Code/ID fields (SecuCode, InnerCode) contain numeric identifiers.
           Source: "证券代码为 512200...交易简称为南华杭州湾区ETF"
           CORRECT: SecuCode=512200, SecuAbbr=南华杭州湾区ETF
           WRONG:   SecuAbbr=512200 (that is the SecuCode, not a name)

        E. LANGUAGE CONSISTENCY: output field values in the SAME language
           as the source document. NEVER translate.
           Chinese source "归类为混合型基金" → "混合型", NOT "Mixed Fund".
           Chinese source "属于常规基金" → "常规基金", NOT "Conventional Fund".
           English source "classified as Bond Fund" → "Bond Fund", NOT "债券型".

        F. VALUE FORMAT PRESERVATION: preserve the complete format of
           structured values as shown in reference data or schema samples.
           If a field stores composite values like "xx/yy" (e.g. rank
           "43/166"), output the full form — NEVER just the numerator.
           The numerator and denominator often appear in DIFFERENT parts
           of the same paragraph. You MUST scan the full paragraph and
           combine them:
           "在包含 162 家机构的同业评比中，其取得了第20位的排名"
             → denominator=162 (from "包含 162 家"), numerator=20 → "20/162"
           "在37家可比机构中位列第9名"
             → denominator=37, numerator=9 → "9/37"
           "取得了 20/117 的排名"
             → already complete → "20/117"
           WRONG: outputting just "20" or "9" without the denominator.

        G. FUZZY DATES: ALWAYS convert natural-language date descriptions
           to YYYY-MM-DD format. NEVER leave a date field empty when the
           text describes it in words.
           "2021年终" / "2021日历年结束" / "2021年末" → 2021-12-31
           "2021年第三季度末" / "截至2021年Q3" → 2021-09-30
           "2021年第二季度收官日" → 2021-06-30
           "2021年第一季度末" → 2021-03-31
           "基于2020日历年结束时的数据" → 2020-12-31
           Only leave a date field empty when the text explicitly says
           data is missing/unavailable — NOT when it uses a verbal date
           description instead of YYYY-MM-DD digits.

        H. FUND FIELD ROUTING: fund documents use "type"/"类型" for several
           distinct dimensions. Route values to the correct column:
           * `Type` (product structure): ETF, LOF, 契约型封闭式,
             开放式(带固定封闭期).
             Source cues: "是一只LOF", "结构为ETF", "类型为LOF".
           * `FundType` (fund classification): 股票型, 混合型, 债券型, 货币型,
             其他型, 基础设施证券投资基金.
             Normalize "XX型基金" → "XX型" (drop "基金").
             Source cues: "归类为XX型基金", "属于XX型基金",
             "资产大类上被归为XX型", "资产类别上被划分为XX型".
           * `InvestmentType` (investment approach): 指数型, 优化指数型,
             成长型—积极成长型, 成长型—稳健成长型, 债券型, 综合型.
             Source cues: "投资类型为指数型", "投资类型被界定为综合型".
           * `InvestStyle` (investment style): 大盘价值, 灵活配置型,
             行业股票-医药, 商品-贵金属, etc.
             Source cues: "投资风格为...", "投资风格专注于...".
           * `FundNature` (fund nature): Conventional Fund (常规基金),
             QDII Fund.
             Source cues: "属于常规基金", "被归类为QDII基金".
           * `FloatType` (trading channel): Exchange-Traded (仅场内),
             Exchange-Traded and Over-the-Counter (场内和场外).
             Source cues: "仅支持场内交易", "支持场内和场外交易".
           * `IfFOF` (boolean): Yes/No.
             Source cues: "基金中基金", "FOF".
           NEVER put ETF/LOF in `FundType`. NEVER put 股票型/混合型 in `Type`.
           NEVER put 指数型/成长型 in `FundType`.

        I. SHARED-ENTITY PARAGRAPHS: when a paragraph declares that
           multiple record IDs share the same entity (e.g. "投资记录 2162
           和 2166 均指向同一标的，即 大成中小盘A。该基金的内部代码为
           34003"), output a SEPARATE line for EACH record ID, and COPY
           all shared identity fields (InnerCode, ChiName, ChiNameAbbr,
           SecuAbbr, SecuCode, etc.) to every line.
           Example output:
             record_id: 2162 | InnerCode: 34003 | ChiName: 大成中小盘混合型...
             record_id: 2166 | InnerCode: 34003 | ChiName: 大成中小盘混合型...
           WRONG: outputting only one line for 2162 and leaving 2166
           without identity fields.

        J. FUND FIELD DISAMBIGUATION:
           {FUND_FIELD_DISAMBIGUATION_PROMPT}

        K. OUTPUT FORMAT: every output line MUST use explicit field names
           with colon separators — "field_name: value | field_name: value".
           NEVER output positional pipe-separated values without field
           names (e.g. "3055|华夏港股通A|160322" is WRONG; correct form is
           "record_id: 3055 | secuabbr: 华夏港股通A | secucode: 160322").
           Your guide MUST repeat this format instruction verbatim.

        Output ONLY the system prompt text (under 1200 words), no wrapping
        or commentary.""")
    from agents.llm.types import ModelMessage

    messages = [
        ModelMessage(
            role="system",
            content="You analyze document structures and write precise data extraction instructions.",
        ),
        ModelMessage(role="user", content=prompt),
    ]
    try:
        response = adapter.complete(messages)
        guide = response.content.strip()
        logger.info("ETL: generated compress guide (%d chars)", len(guide))
        return guide
    except Exception as exc:
        logger.warning("Failed to generate compress guide: %s", exc)
        return None


def _format_schema_def_hint(
    schema_columns: list[str] | None,
    schema_field_defs: dict[str, str] | None,
) -> str:
    if not schema_columns or not schema_field_defs:
        return ""
    field_defs = metadata_by_name(schema_columns, schema_field_defs)
    lines = [f"  - {field}: {field_defs[field]}" for field in schema_columns if field in field_defs]
    if not lines:
        return ""
    return (
        "\nField semantic definitions — match values strictly by these meanings, "
        "NOT by proximity or order of appearance in the text:\n" + "\n".join(lines) + "\n"
    )


def compress_chunk(
    adapter: ModelAdapter,
    chunk_text: str,
    schema_columns: list[str] | None = None,
    primary_key: str | None = None,
    anchor_keys: list[str] | None = None,
    schema_field_types: dict[str, str] | None = None,
    schema_field_defs: dict[str, str] | None = None,
    compress_guide: str | None = None,
    entity_aware: bool = False,
) -> str:
    """Ask LLM to extract labeled key-value pairs from prose paragraphs."""
    if schema_columns and primary_key:
        cols = ", ".join(schema_columns)
        schema_hint = (
            f"\nALLOWED FIELD NAMES (use ONLY these, letter-for-letter): {cols}\n"
            "NEVER rename, abbreviate, or paraphrase a field name. "
            'If the prose says "height" and the allowed name is "admissionheight", '
            'output "admissionheight". Any key not in the allowed list is an error.\n'
            "Map a value to a field only when it explicitly matches that field's "
            "meaning. Do NOT force narrative text into the closest field.\n"
        )
        type_hint = _format_schema_type_hint(schema_columns, schema_field_types)
        hard_type_rules = _format_hard_type_rules(schema_columns, schema_field_types)
        if entity_aware:
            format_hint = (
                f"Format — ONE line per entity, pipe-separated key-value pairs:\n"
                f"  {primary_key}: <value> | field_a: <value> | field_b: <value> | ...\n"
                "Use COLON (:) to separate key from value. NEVER use = as separator.\n"
                "Each entity's paragraphs are grouped between '---' delimiters and "
                "begin with a [RECORD_ID: N] marker. Copy that exact N as the "
                f"{primary_key} value of the output line — NEVER renumber entities "
                "sequentially (1, 2, 3, ...) and NEVER invent IDs. "
                "Combine information from ALL paragraphs of the same entity into "
                "ONE output line. OMIT fields with no suitable value in any of "
                "the entity's paragraphs.\n"
            )
        else:
            format_hint = (
                f"Format — one line per entity-paragraph, pipe-separated key-value pairs:\n"
                f"  {primary_key}: <value> | field_a: <value> | field_b: <value> | ...\n"
                "Use COLON (:) to separate key from value. NEVER use = as separator.\n"
                "ONLY include fields that have actual values in the paragraph. "
                "OMIT non-identity fields with no suitable value; downstream "
                "normalization will fill them as empty cells.\n"
            )
        if anchor_keys and len(anchor_keys) > 1:
            anchor_names = ", ".join(anchor_keys)
            if entity_aware:
                anchor_rule = (
                    f"- Each output line MUST include these identity fields: {anchor_names}. "
                    f"Take {primary_key} from the [RECORD_ID: N] marker; collect the "
                    "other identity values from any paragraph in the entity group."
                )
            else:
                anchor_rule = (
                    f"- Start each line with these identity fields: {anchor_names}. "
                    "Include ALL of them when mentioned in the paragraph. "
                    "If a field's value is not stated in the paragraph, include its "
                    f"key with an empty value. Take {primary_key} from the "
                    "paragraph's own record label; NEVER renumber paragraphs "
                    "sequentially (1, 2, 3, ...) and NEVER invent IDs."
                )
        else:
            anchor_rule = (
                f"- Start each line with {primary_key} as stated in the paragraph. "
                "This field anchors every line."
            )
            if not entity_aware:
                anchor_rule += (
                    " Copy the identifier from the paragraph's own record label "
                    f'(e.g. "Record <N>" → {primary_key}: <N>, or '
                    f'"Registry ID: recXxx" → {primary_key}: recXxx). NEVER renumber '
                    "paragraphs sequentially (1, 2, 3, ...) and NEVER invent IDs — "
                    "if a paragraph states no record identifier at all, SKIP that "
                    "paragraph."
                )
    else:
        schema_hint = ""
        type_hint = ""
        hard_type_rules = ""
        if entity_aware:
            format_hint = (
                "Format — ONE line per entity, pipe-separated key-value pairs:\n"
                "  ID: <value> | entity_name: <name> | <field>: <value> | ...\n"
                "Use COLON (:) to separate key from value. NEVER use = as separator.\n"
                "Each entity's paragraphs are grouped between '---' delimiters and "
                "begin with a [RECORD_ID: N] marker. Copy that exact N as the ID "
                "value — NEVER renumber entities sequentially and NEVER invent IDs. "
                "Combine all paragraphs of the same entity into ONE line.\n"
            )
        else:
            format_hint = (
                "Format — one line per entity-paragraph:\n"
                "  ID: <value> | entity_name: <name> | <field>: <value> | ...\n"
                "Use COLON (:) to separate key from value. NEVER use = as separator.\n"
            )
        anchor_rule = (
            "- Start each line with the entity's ID AND its name/codename/label "
            "as stated in the paragraph. These two fields anchor every line."
        )

    id_field_label = primary_key or "ID"

    sparse = bool(schema_columns and len(schema_columns) > 10)
    if sparse:
        placeholder_rule = (
            "- If a field is not mentioned or its value is None/NaN/placeholder/redacted, "
            "SKIP IT entirely — do NOT include it in the output line."
        )
    else:
        placeholder_rule = (
            "- If a field's ENTIRE value is 0.0, None, NaN, placeholder, or "
            'redacted (with no real data), output it with a "(placeholder)" marker — '
            'e.g. "height_cm: 0.0 (placeholder)". Do NOT mark corrected values as '
            'placeholder — "initially X, corrected to Y" means Y is the real value.'
        )

    def_hint = _format_schema_def_hint(schema_columns, schema_field_defs)

    if entity_aware:
        opening = (
            "Below are groups of paragraphs from a data document, separated by '---'. "
            "Each group contains ALL paragraphs about ONE entity, mixing data with "
            "narrative filler, and starts with its authoritative [RECORD_ID: N] marker."
        )
        task_line = (
            "Task: for each entity group, combine information from all its "
            "paragraphs and extract ONE line of labeled key-value pairs."
        )
        skip_rule = (
            f"- If an entire entity group provides NO extractable field values "
            f"beyond the {id_field_label}, SKIP it."
        )
        shared_para_rule = (
            "- A paragraph may be SHARED by several entities and enumerate "
            'per-entity values (e.g. "... stood at <X> for unit <A>, <Y> for '
            "unit <B>\"): take ONLY the value stated for this group's "
            "[RECORD_ID: N]; NEVER borrow a value stated for another entity. "
            "If such a paragraph only declares values missing/unavailable for "
            "a set of entities, leave those fields empty — that is not a "
            "reason to skip the rest of the entity group.\n"
            "- EXCEPTION: when a paragraph declares that MULTIPLE entities "
            "share the SAME value (e.g. 'units A and B are both classified "
            "as stock-type'), apply that value to EVERY listed entity. "
            "This is NOT borrowing — the text explicitly states the value "
            "for all named entities."
        )
    else:
        opening = (
            "Below are paragraphs from a data document. Each paragraph describes "
            "one entity and mixes data with narrative filler."
        )
        task_line = "Task: for each paragraph, extract data as labeled key-value pairs."
        shared_para_rule = ""
        skip_rule = (
            "- If a paragraph provides NO extractable field values beyond the "
            "record_id (e.g. it only says data is missing/not provided/unavailable), "
            "SKIP that paragraph entirely — do NOT output a row with only identity fields and empty data fields. "
            "However, if the paragraph carries identity/anchor values such as "
            "secucode, companycode, or entity name, DO output a row even when no "
            "other data fields are present — downstream merge needs those identity "
            "values to complete the record."
        )

    prompt = textwrap.dedent(f"""\
        {opening}

        {task_line}
        {schema_hint}
        {def_hint}
        {FUND_FIELD_DISAMBIGUATION_PROMPT}
        {type_hint}
        {hard_type_rules}
        {format_hint}
        Rules:
        {anchor_rule}
        {shared_para_rule}
        - LANGUAGE: output field values in the SAME language as the source \
        document. NEVER translate values. Chinese source → Chinese values; \
        English source → English values.
        - VALUE FORMAT: preserve the complete format of structured values. \
        If the schema expects "xx/yy" (e.g. rank "43/166"), output the full \
        composite form — NEVER just the numerator. The numerator and \
        denominator often appear in different parts of the same paragraph; \
        scan the full context and combine them. \
        "在包含 162 家机构的同业评比中，其取得了第20位的排名" → "20/162". \
        "在37家可比机构中位列第9名" → "9/37". \
        WRONG: outputting just "20" or "9".
        - SHARED-ENTITY: when a paragraph declares multiple record IDs \
        share the same entity (e.g. "记录 2162 和 2166 均指向同一标的"), \
        output a SEPARATE line for EACH record ID with ALL shared identity \
        fields copied. NEVER output only one line and leave the other IDs \
        without identity fields.
        - Every value MUST have a descriptive field label before it.
        - If a schema field has no explicit, suitable value in this {"entity group" if entity_aware else "paragraph"}, \
        leave it empty by omitting that non-identity field. Do NOT substitute \
        background descriptions, business strengths, culture, plans, or prose \
        summaries just because they are loosely related to the field name.
        {skip_rule}
        - Field-name qualifiers are binding. Do NOT map an overall/global \
        total, count, rank, or average into a qualified/category-specific \
        field unless the paragraph explicitly states the same qualifier for \
        that value; omit the field instead.
        - Local document labels such as {RECORD_LABEL_EXAMPLES} are \
        record identifiers. Put only the scalar numeric portion (for example, \
        "<N>") in the {id_field_label} field. \
        NEVER output ranges like "197至217" or grouped labels as a single \
        {id_field_label} value, and never put a local record label into domain ID \
        fields such as personalcode, innercode, or fundcode unless \
        the paragraph explicitly labels it as that domain ID.
        - personalcode/PersonalCode fields are person identifiers. Do NOT \
        fill them with a manager name or with a local document record number; \
        leave personalcode empty unless an explicit personal/manager code is \
        stated.
        - Output numeric values WITHOUT unit suffixes (e.g. "<value> cm" → <value>).
        - PERCENTAGES: the source text may express proportions in TWO forms: \
        percent form ("1.55%") or ratio form ("0.001910" meaning 0.191%). \
        Normalize ALL proportion/percentage values to PERCENT form: \
        "1.55%" → 1.55 (strip %); bare ratio like "0.001910" that \
        describes a proportion (e.g. "占总股本的 0.001910") → multiply \
        by 100 → 0.191. This ensures consistent units within each column.
        - NUMERIC CONSISTENCY (mandatory cross-check after extraction):
          (a) For paired shares+percentage fields representing the SAME state \
        (e.g. pre-transfer shares and pre-transfer pct), verify: \
        pct_value ≈ shares_value / total_capital. If your extracted pct is \
        orders of magnitude off from this ratio, you picked the WRONG number.
          (b) A "before" percentage must be LARGER than the corresponding \
        "after" percentage (transferor lost shares, so pct dropped).
          (c) When a paragraph contains MULTIPLE ratio/decimal values, the one \
        stated in the SAME sentence as the shares count (e.g. "持有 X 股，\
        占总股本的 Y%") is the matching percentage. Ratios stated alongside \
        the TRANSFER QUANTITY (e.g. "涉及 X 股…占转让方持股的 Y") belong \
        to a DIFFERENT field or should be discarded.
          (d) HOLDING vs TRANSACTION disambiguation — this is the #1 error \
        source: documents often describe one record across TWO sections: \
        (i) a transaction section stating the transferred quantity and its \
        ratio to total capital (e.g. "转让 30万股…占总股本 0.00191"); \
        (ii) a holding section stating the pre/post-transfer HOLDING and \
        its ratio to total capital (e.g. "持有 244万股，占总股本 1.55%"). \
        The "before/after holding pct" field MUST come from section (ii) — \
        the ratio stated in the same clause as the HOLDING shares count. \
        The ratio from section (i) describes what fraction of total capital \
        was TRANSFERRED, which is a different semantic and must NEVER fill \
        holding percentage fields. \
        Key signal: match the percentage to its CO-OCCURRING shares figure. \
        "持有 X 股，占总股本的 Y" → Y is the holding pct for X shares. \
        "涉及/转让 X 股…占总股本的 Y" → Y is the transfer-volume ratio, \
        discard for holding fields. \
        Cross-check: holding_pct ≈ holding_shares / total_capital. If the \
        ratio you picked doesn't satisfy this, you grabbed the wrong one.
          If any check fails, re-read the paragraph and pick the value that \
        satisfies all three constraints.
        - CHINESE NUMBERS: NEVER output Chinese number words — ALWAYS \
        convert to Arabic digits. 两=2 一点五=1.5 亿=1e8 万=1e4. \
        "两亿元"→200000000 "一点五亿元"→150000000. Leaving empty is wrong.
        - FUZZY NUMBERS: when a value is approximate — marked with "约", \
        "大约", "近", "左右", "roughly", "approximately", or written in \
        coarse Chinese number words like "约六百九十七万" — convert the \
        numeral mechanically and prefix it with ~ (e.g. "约六百九十七万股" \
        → ~6970000). Do NOT attempt arithmetic to sharpen it, and NEVER \
        omit a value merely because it is approximate — always output it \
        with the ~ prefix: a downstream deterministic pass recomputes exact \
        values from sibling fields. Values stated as exact digits get NO \
        prefix.
        {placeholder_rule}
        - For corrections: output the FINAL corrected value directly — e.g. \
        "label: positive", not "label: positive (placeholder)".
        - Preserve URLs, names, dates, times, categories, classifications.
        - Strip only narrative filler that contains no structured field, schema field, \
        ID, value, date, unit, label, status, or question-relevant fact.
        - If NOTHING in this input qualifies for extraction under these rules, \
        return an EMPTY response. NEVER fabricate demonstration rows and NEVER \
        copy placeholder examples (such as "<N>" or "<value>") from these \
        instructions into the output.
        - Separate entries with ONE blank line. No commentary, no markdown.

        ---
        {chunk_text}
        ---

        REMINDER: output MUST use "field_name: value | field_name: value" format. NEVER omit field names.""")
    from agents.llm.types import ModelMessage

    system_content = compress_guide or "You are a data extraction preprocessor."
    messages = [
        ModelMessage(role="system", content=system_content),
        ModelMessage(role="user", content=prompt),
    ]
    return adapter.complete(messages).content.strip()


def _positional_chunks(text: str) -> list[tuple[str | None, str]]:
    """Original positional chunking: split by section, then by paragraph count."""
    sections = detect_sections(text)
    work: list[tuple[str | None, str]] = []
    if len(sections) < 3:
        paras = [p.strip() for p in text.split("\n\n") if p.strip()]
        for i in range(0, len(paras), COMPRESS_CHUNK_SIZE):
            work.append((None, "\n\n".join(paras[i : i + COMPRESS_CHUNK_SIZE])))
    else:
        for sec in sections:
            paras = [
                p.strip() for p in sec["content"].split("\n\n") if p.strip() and len(p.strip()) > 30
            ]
            first = True
            for i in range(0, len(paras), COMPRESS_CHUNK_SIZE):
                title = sec["title"] if first else None
                work.append((title, "\n\n".join(paras[i : i + COMPRESS_CHUNK_SIZE])))
                first = False
    return work


def compress_prose(
    adapter: ModelAdapter,
    text: str,
    schema_columns: list[str] | None = None,
    primary_key: str | None = None,
    anchor_keys: list[str] | None = None,
    schema_field_types: dict[str, str] | None = None,
    schema_field_defs: dict[str, str] | None = None,
) -> tuple[str, dict[str, list[str]] | None, str | None]:
    """Compress prose by stripping noise. All chunks across all sections run in one pool.

    Returns (compressed_text, entity_groups, compress_guide) where entity_groups
    maps record_id → source paragraphs (None when positional chunking was used)
    and compress_guide is the LLM-generated extraction guide (None when no
    schema was provided).
    """

    # Try entity-aware chunking: group paragraphs by entity (LLM grouping
    # with deterministic verification) so each entity's full context is in
    # one chunk, never split across chunks.
    grouping = group_paragraphs_by_entity(text, adapter=adapter)
    entity_groups: dict[str, list[str]] | None = None
    work: list[tuple[str | None, str]] = []
    chunk_rids: list[list[str]] = []
    if grouping:
        entity_groups, leftover = grouping
        chunk_infos = entity_chunks_by_token_budget(entity_groups)
        work = [(None, c) for c, _ in chunk_infos]
        chunk_rids = [rids for _, rids in chunk_infos]
        # Paragraphs in no verified group (NONE-classified or unassigned)
        # still carry numbers: compress them positionally instead of
        # dropping them — the LLM reads the record id straight from the
        # paragraph text and the downstream PK merge unifies those lines
        # with the entity lines.
        if leftover:
            logger.info(
                "ETL compress: %d numeric paragraphs matched no entity; "
                "processing them positionally",
                len(leftover),
            )
            for i in range(0, len(leftover), COMPRESS_CHUNK_SIZE):
                work.append((None, "\n\n".join(leftover[i : i + COMPRESS_CHUNK_SIZE])))
                chunk_rids.append([])
        logger.info(
            "ETL compress: entity-aware chunking: %d entities → %d chunks",
            len(entity_groups),
            len(work),
        )
    else:
        work = _positional_chunks(text)
        chunk_rids = [[] for _ in work]

    if not work:
        return text, entity_groups, None

    guide = _generate_compress_guide(
        adapter,
        text,
        schema_columns,
        schema_field_defs,
        schema_field_types,
    )

    compressed: list[str | None] = [None] * len(work)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {
            submit_in_context(
                pool,
                compress_chunk,
                adapter,
                item[1],
                schema_columns,
                primary_key,
                anchor_keys,
                schema_field_types,
                schema_field_defs,
                guide,
                entity_aware=bool(chunk_rids[idx]),
            ): idx
            for idx, item in enumerate(work)
        }
        try:
            for fut in as_completed(futs, timeout=LLM_CALL_TIMEOUT * len(work)):
                idx = futs[fut]
                try:
                    compressed[idx] = fut.result(timeout=LLM_CALL_TIMEOUT)
                except TimeoutError:
                    logger.warning("Compress chunk %d timed out", idx)
                except Exception as exc:
                    logger.warning("Compress chunk %d failed: %s", idx, exc)
        except TimeoutError:
            logger.warning("Compress pool timed out, proceeding with partial results")

    # Entity-aware integrity check: every output line's PK must echo one of
    # the [RECORD_ID: N] markers from its own chunk.  A fabricated sequential
    # PK (1, 2, 3, ...) collides across chunks and silently collapses
    # entities in the downstream PK merge — surface it loudly, then retry
    # the genuinely missing entities in a targeted recovery pass.
    recovered_parts: list[str] = []
    if entity_groups and primary_key:

        def _extracted_pks(chunk_text: str) -> set[str]:
            from agents.etl._record import normalize_local_record_ids, parse_kv_text

            table = parse_kv_text(
                chunk_text,
                columns=schema_columns or [],
                primary_key=primary_key,
                anchor_keys=anchor_keys,
                field_types=schema_field_types,
            )
            normalize_local_record_ids(table)
            return {record.pk.strip() for record in table.records if record.pk}

        to_recover: set[str] = set()
        for idx, expected_rids in enumerate(chunk_rids):
            if not expected_rids:
                continue
            chunk_result = compressed[idx]
            if not chunk_result:
                # Whole chunk failed or timed out — recover all its entities.
                to_recover.update(expected_rids)
                continue
            expected = set(expected_rids)
            got = _extracted_pks(chunk_result)
            bogus = got - expected
            missing = expected - got
            if bogus:
                logger.warning(
                    "ETL compress: chunk %d PK values %s not in expected markers %s "
                    "— model ignored [RECORD_ID] markers; downstream merge may "
                    "collapse entities",
                    idx,
                    sorted(bogus),
                    sorted(expected),
                )
            if missing:
                logger.warning(
                    "ETL compress: chunk %d missing entities %s (%d/%d extracted)",
                    idx,
                    sorted(missing),
                    len(expected - missing),
                    len(expected),
                )
                to_recover.update(missing)

        if to_recover:
            subset = {rid: entity_groups[rid] for rid in sorted(to_recover) if rid in entity_groups}
            if subset:
                logger.info(
                    "ETL compress: targeted recovery pass for %d missing entities",
                    len(subset),
                )
                for retry_text, _ in entity_chunks_by_token_budget(subset):
                    try:
                        result = compress_chunk(
                            adapter,
                            retry_text,
                            schema_columns,
                            primary_key,
                            anchor_keys,
                            schema_field_types,
                            schema_field_defs,
                            guide,
                            entity_aware=True,
                        )
                    except Exception as exc:
                        logger.warning("ETL compress: recovery chunk failed: %s", exc)
                        continue
                    if result:
                        recovered_parts.append(result)
                got_all: set[str] = set()
                for part in recovered_parts:
                    got_all |= _extracted_pks(part)
                recovered = sorted(set(subset) & got_all)
                still_missing = sorted(set(subset) - got_all)
                if recovered:
                    logger.info("ETL compress: recovery pass extracted %s", recovered)
                if still_missing:
                    logger.warning(
                        "ETL compress: entities still missing after recovery "
                        "(likely no extractable data): %s",
                        still_missing,
                    )

    parts: list[str] = []
    for idx, (title, _) in enumerate(work):
        if title:
            parts.append(f"### {title}")
        chunk_result = compressed[idx]
        if chunk_result:
            parts.append(chunk_result)
    parts.extend(recovered_parts)
    return "\n\n".join(parts), entity_groups, guide
