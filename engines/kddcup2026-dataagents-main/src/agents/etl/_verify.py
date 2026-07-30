"""Value-level retry, cross-field dedup, and outlier verification.

这三个阶段直接操作 :class:`~agents.etl._record.EntityTable`（in place）：
修正按 ``by_pk[rid].fields[col] = ...`` 精确落地（大小写敏感），LLM 文本经
:func:`~agents.etl._record.sanitize_llm_value` 收口 + ``clean_cell_for_type``
类型清洗后入库——不存在正则回填，值内容不可能破坏行结构。
"""

from __future__ import annotations

import logging
import re
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

from agents.etl._columns import metadata_by_name
from agents.etl._constants import (
    DEDUP_CONTEXT_WINDOW,
    FILL_RATE_THRESHOLD,
    FUND_FIELD_DISAMBIGUATION_PROMPT,
    LLM_CALL_TIMEOUT,
    MAX_RETRY_ENTITIES,
    MAX_VERIFY_ENTITIES,
    MAX_WORKERS,
)
from agents.etl._merge import clean_cell_for_type, strip_reasoning_leak
from agents.etl._record import EntityTable, Record, sanitize_llm_value, set_record_field
from agents.etl._threading import submit_in_context

if TYPE_CHECKING:
    from agents.llm.types import ModelAdapter

logger = logging.getLogger(__name__)


def _records_by_pk(table: EntityTable) -> dict[str, list[Record]]:
    by_pk: dict[str, list[Record]] = {}
    for record in table.records:
        if record.pk:
            by_pk.setdefault(record.pk, []).append(record)
    return by_pk


def _apply_llm_value(table: EntityTable, record: Record, col: str, val: str, writer: str) -> None:
    """LLM 修正值落地的唯一路径：sanitize → 类型清洗 → 入库。"""
    field_types = metadata_by_name(table.columns, table.field_types)
    cleaned = clean_cell_for_type(
        sanitize_llm_value(val),
        field_types.get(col),
        field_name=col,
    )
    set_record_field(record, col, cleaned, writer)


def _detect_gap_cells(
    table: EntityTable,
) -> dict[str, list[str]]:
    """Return {record_id: [missing_col, ...]} for high-fill columns with gaps."""
    primary_key = table.primary_key
    if not table.records or not primary_key:
        return {}

    skip_lower = {a.lower() for a in table.anchor_keys}
    candidate_cols = [c for c in table.columns if c.lower() not in skip_lower]
    if not candidate_cols:
        return {}

    n = len(table.records)
    fill_counts: dict[str, int] = dict.fromkeys(candidate_cols, 0)
    for record in table.records:
        for c in candidate_cols:
            if record.fields.get(c, "").strip():
                fill_counts[c] += 1

    gap_cols = [
        c
        for c in candidate_cols
        if fill_counts[c] / n >= FILL_RATE_THRESHOLD and fill_counts[c] < n
    ]
    if not gap_cols:
        return {}

    gaps: dict[str, list[str]] = {}
    for record in table.records:
        if not record.pk:
            continue
        missing = [c for c in gap_cols if not record.fields.get(c, "").strip()]
        if missing:
            gaps[record.pk] = missing
    return gaps


def _value_search_needles(value_str: str) -> list[str]:
    """Return plain and comma-separated forms of a numeric value string."""
    plain = value_str.replace(",", "")
    if not plain:
        return []
    int_part = plain.split(".")[0].lstrip("-")
    comma_form = ""
    if int_part.isdigit() and len(int_part) > 3:
        chunks: list[str] = []
        while int_part:
            chunks.append(int_part[-3:])
            int_part = int_part[:-3]
        comma_form = ",".join(reversed(chunks))
    needles = [plain]
    if comma_form and comma_form != plain:
        needles.append(comma_form)
    return needles


def _extract_value_context(value_str: str, source_text: str, *, max_hits: int = 3) -> str:
    """Extract source context snippets surrounding *all* occurrences of *value_str*."""
    snippets: list[str] = []
    seen_starts: set[int] = set()
    for needle in _value_search_needles(value_str):
        for m in re.finditer(re.escape(needle), source_text):
            start = max(0, m.start() - DEDUP_CONTEXT_WINDOW)
            if start in seen_starts:
                continue
            seen_starts.add(start)
            end = min(len(source_text), m.end() + DEDUP_CONTEXT_WINDOW)
            snippets.append(source_text[start:end])
            if len(snippets) >= max_hits:
                break
        if len(snippets) >= max_hits:
            break
    return " [...] ".join(snippets)


def dedup_cross_field_copies(
    adapter: ModelAdapter,
    table: EntityTable,
    source_text: str | None,
) -> dict[str, list[str]]:
    """Clear values duplicated across distinct fields by the same entity.

    When compress puts the same numeric value into two columns, one is
    almost certainly wrong (e.g. reserveassets value copied into
    totalassets).  Uses one LLM call to resolve which column each
    duplicate value actually belongs to, then clears the wrong column
    so retry can re-extract it.

    Mutates *table* in place and returns cleared as {pk_val: [col, ...]}.
    """
    primary_key = table.primary_key
    if not primary_key or not source_text or not table.records:
        return {}

    skip_lower = {a.lower() for a in table.anchor_keys}
    data_cols = [c for c in table.columns if c.lower() not in skip_lower]

    dupes: list[tuple[str, str, list[str], str]] = []
    for record in table.records:
        if not record.pk:
            continue

        val_to_cols: dict[str, list[str]] = {}
        for c in data_cols:
            v = record.fields.get(c, "").strip().replace(",", "")
            if not v:
                continue
            try:
                float(v)
            except ValueError:
                continue
            val_to_cols.setdefault(v, []).append(c)

        for val, cols in val_to_cols.items():
            if len(cols) < 2:
                continue
            ctx = _extract_value_context(val, source_text)
            dupes.append((record.pk, val, cols, ctx))

    if not dupes:
        return {}

    from agents.llm.types import ModelMessage

    items: list[str] = []
    for i, (pk_val, val, cols, ctx) in enumerate(dupes):
        items.append(
            f"{i + 1}. {primary_key}={pk_val}, value={val}, "
            f"columns={cols}\n   Source context: ...{ctx}..."
        )

    prompt = textwrap.dedent(f"""\
        The same numeric value appears in multiple columns for each entity
        below. Based on the source context, determine whether the value
        genuinely belongs to ALL listed columns or only ONE of them.

        {chr(10).join(items)}

        For each item output EXACTLY one line:
        <item_number>:BOTH   — if the source text independently states this value for each column
        <item_number>:<column_to_keep>  — if only one column should have this value

        Output nothing else.""")

    messages: list[ModelMessage] = [
        ModelMessage(
            role="system",
            content="Resolve cross-field value duplicates. "
            "Output item_number:column_to_keep or item_number:BOTH lines only.",
        ),
        ModelMessage(role="user", content=prompt),
    ]

    try:
        response = adapter.complete(messages)
    except Exception as exc:
        logger.warning("ETL dedup LLM resolution failed: %s", exc)
        return {}

    raw = response.content.strip()
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    to_clear: dict[str, list[str]] = {}
    for line in raw.splitlines():
        line = line.strip()
        if ":" not in line:
            continue
        idx_str, _, keep_col = line.partition(":")
        try:
            idx = int(idx_str.strip()) - 1
        except ValueError:
            continue
        if idx < 0 or idx >= len(dupes):
            continue
        keep_col = keep_col.strip().lower()
        if keep_col == "both":
            continue
        pk_val, _val, cols, _ctx = dupes[idx]
        matched = [c for c in cols if c.lower() == keep_col]
        if len(matched) != 1:
            continue
        for c in cols:
            if c != matched[0]:
                to_clear.setdefault(pk_val, []).append(c)

    if not to_clear:
        return {}

    by_pk = _records_by_pk(table)
    cleared = 0
    for pk_val, cols in to_clear.items():
        for record in by_pk.get(pk_val, []):
            for col in cols:
                if not record.fields.get(col, "").strip():
                    continue
                set_record_field(record, col, "", "dedup")
                cleared += 1

    if cleared:
        logger.info(
            "ETL dedup: cleared %d cross-field copy values across %d entities",
            cleared,
            len(to_clear),
        )
    return to_clear


def retry_missing_values(
    adapter: ModelAdapter,
    table: EntityTable,
    entity_groups: dict[str, list[str]] | None,
    schema_field_defs: dict[str, str] | None = None,
    forced_gaps: dict[str, list[str]] | None = None,
) -> None:
    """Re-extract missing values for high-fill columns from source paragraphs.

    Mutates *table* in place; only fills cells that are still empty.
    """
    if not table.primary_key or not entity_groups or not table.columns:
        return

    gaps = _detect_gap_cells(table)
    if forced_gaps:
        for pk_val, cols in forced_gaps.items():
            existing = gaps.get(pk_val, [])
            gaps[pk_val] = sorted(set(existing + cols))
    if not gaps:
        return

    retry_ids = sorted(gaps.keys())[:MAX_RETRY_ENTITIES]
    all_missing_cols = sorted({c for cols in gaps.values() for c in cols})

    logger.info(
        "ETL value retry: %d entities missing values in columns %s",
        len(retry_ids),
        all_missing_cols,
    )

    by_pk = _records_by_pk(table)
    patched = 0
    for rid in retry_ids:
        source_paras = entity_groups.get(rid)
        if not source_paras:
            continue
        missing_cols = gaps[rid]
        missing_col_by_lower = {c.lower(): c for c in missing_cols}
        source_text = "\n\n".join(source_paras)

        col_list = ", ".join(missing_cols)
        defs_hint = ""
        field_defs = metadata_by_name(missing_cols, schema_field_defs)
        if field_defs:
            parts: list[str] = []
            for c in missing_cols:
                d = field_defs.get(c, "")
                if d:
                    parts.append(f"  {c}: {d}")
            if parts:
                defs_hint = "\nField definitions:\n" + "\n".join(parts) + "\n"

        fund_hint = ""
        fund_fields = {
            "type",
            "fund_type",
            "fundtype",
            "fundtypename",
            "investmenttype",
            "investstyle",
            "fundnature",
            "floattype",
            "iffof",
        }
        if any(c.lower() in fund_fields for c in missing_cols):
            fund_hint = (
                "\nFund field disambiguation for missing values:\n"
                f"{FUND_FIELD_DISAMBIGUATION_PROMPT}\n"
                "- Do not leave a fund field blank when the source text contains "
                "an explicit value for it.\n"
            )

        prompt = (
            f"Extract ONLY these fields from the text below:\n"
            f"Fields: {col_list}\n"
            f"{defs_hint}"
            f"{fund_hint}"
            f"LANGUAGE: output values in the SAME language as the source text. NEVER translate.\n"
            f"Output format — one line, pipe-separated:\n"
            f"  {' | '.join(f'{c}: <value>' for c in missing_cols)}\n"
            f"If a field's value is not in the text, leave it empty: {missing_cols[0]}: \n"
            f"Output ONLY the single key-value line, nothing else.\n\n"
            f"TEXT:\n{source_text}"
        )

        from agents.llm.types import ModelMessage as _MM

        messages = [_MM(role="user", content=prompt)]
        try:
            response = adapter.complete(messages)
        except Exception:
            logger.debug("ETL value retry: LLM call failed for %s", rid, exc_info=True)
            continue

        reply = response.content.strip()
        if reply.startswith("```"):
            reply = reply.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        extracted: dict[str, str] = {}
        for part in reply.split("|"):
            if ":" not in part:
                continue
            key, _, val = part.partition(":")
            key = missing_col_by_lower.get(key.strip().lower())
            val = val.strip()
            if key and val:
                extracted[key] = val

        if not extracted:
            continue

        applied: list[str] = []
        for record in by_pk.get(rid, []):
            for col, val in extracted.items():
                if record.fields.get(col, "").strip():
                    continue  # 只回填仍为空的格
                _apply_llm_value(table, record, col, val, "retry")
                applied.append(col)
        if applied:
            patched += len(applied)
            logger.info(
                "ETL value retry: patched %d fields for entity %s: %s",
                len(applied),
                rid,
                sorted(set(applied)),
            )

    if patched:
        logger.info(
            "ETL value retry: total %d values patched across %d entities", patched, len(retry_ids)
        )


def _sample_verify_entities(
    table: EntityTable,
    entity_groups: dict[str, list[str]],
) -> list[str]:
    """Select a spread of entity IDs for verification, prioritizing entities
    with more source paragraphs (higher chance of cross-paragraph confusion)."""
    candidates: list[tuple[str, int]] = []
    for record in table.records:
        if record.pk and record.pk in entity_groups:
            candidates.append((record.pk, len(entity_groups[record.pk])))
    candidates.sort(key=lambda x: -x[1])
    return [pk for pk, _ in candidates[:MAX_VERIFY_ENTITIES]]


def verify_field_values(
    adapter: ModelAdapter,
    table: EntityTable,
    entity_groups: dict[str, list[str]] | None,
    schema_field_defs: dict[str, str] | None = None,
    compress_guide: str | None = None,
) -> None:
    """Verify extracted field values against source paragraphs using LLM.

    Mutates *table* in place with validated corrections.
    """
    primary_key = table.primary_key
    if not primary_key or not entity_groups or not table.columns or not table.records:
        return

    verify_ids = _sample_verify_entities(table, entity_groups)
    if not verify_ids:
        return

    skip_lower = {a.lower() for a in table.anchor_keys}
    types = metadata_by_name(table.columns, table.field_types)
    verify_cols = [
        c
        for c in table.columns
        if c.lower() not in skip_lower and "date" not in types.get(c, "").lower()
    ]
    if not verify_cols:
        return

    logger.info(
        "ETL field verify: checking %d entities (%d numeric fields each)",
        len(verify_ids),
        len(verify_cols),
    )

    from agents.llm.types import ModelMessage as _MM

    col_list = ", ".join(verify_cols)
    defs_block = ""
    field_defs = metadata_by_name(verify_cols, schema_field_defs)
    if field_defs:
        parts = [f"  {c}: {field_defs[c]}" for c in verify_cols if c in field_defs]
        if parts:
            defs_block = "\nField definitions:\n" + "\n".join(parts) + "\n"

    guide_block = ""
    if compress_guide:
        guide_block = (
            "\nDOCUMENT EXTRACTION GUIDE (generated during extraction — "
            "use these field-to-prose mappings and disambiguation rules "
            "when judging whether a value is correct):\n"
            f"{compress_guide}\n"
        )

    anchor_lower = {a.lower() for a in table.anchor_keys}
    anchor_lower.add(primary_key.lower())
    by_pk = _records_by_pk(table)

    def _verify_one(rid: str) -> tuple[str, dict[str, str]]:
        source_paras = entity_groups.get(rid)
        if not source_paras:
            return rid, {}
        row = next((r.fields for r in by_pk.get(rid, [])), None)
        if row is None:
            return rid, {}

        current_kv = " | ".join(f"{c}: {row.get(c, '')}" for c in verify_cols)
        source_text = "\n\n".join(source_paras)

        prompt = (
            f"Verify each field value against the source text.\n"
            f"Fields to verify: {col_list}\n"
            f"{defs_block}"
            f"{guide_block}\n"
            f"CURRENT EXTRACTION:\n{current_kv}\n\n"
            f"SOURCE TEXT:\n{source_text}\n\n"
            f"IMPORTANT: Match each value to the EXACT field label in the "
            f"source text. A value labeled 'Reserve Assets' is NOT the same "
            f"as 'Total Assets'. A sub-total or component value must NOT be "
            f"assigned to an aggregate/total field.\n\n"
            f"ABBREVIATION CHECK: if SecuAbbr/secuabbr is among the fields, it "
            f"MUST be the SHORTEST name (readable text, NOT a numeric code). "
            f"Three tiers, longest→shortest:\n"
            f"  (1) FULL NAME (全称/正式名称) → ChiName/fund_name ONLY. "
            f"In SecuAbbr = ALWAYS WRONG.\n"
            f"  (2) MARKET NAME (市场简称/常用名/通用简称/官方简称/"
            f"市场上的简称) → ChiNameAbbr/fund_name_short ONLY. "
            f"In SecuAbbr = WRONG when tier 3 exists.\n"
            f"  (3) TRADING NAME (证券简称/交易简称/交易代码为<非数字文字>)"
            f" → SecuAbbr/secuabbr ONLY. ALWAYS the shortest.\n"
            f"SecuAbbr MUST be the SHORTEST name form. If multiple candidates "
            f"exist, ALWAYS pick the shortest.\n"
            f"If current SecuAbbr is a tier-2 value but a shorter tier-3 name "
            f"exists in the source, it is WRONG — correct it.\n"
            f"If current SecuAbbr is empty but ANY name exists in the source, "
            f"it is WRONG — fill it with the shortest available name.\n"
            f"A PURE NUMERIC value like '512200' in SecuAbbr is ALWAYS "
            f"WRONG — that is a SecuCode, not a name. SecuAbbr must "
            f"contain readable text (e.g. '南华杭州湾区ETF').\n"
            f"Example: SecuAbbr='512200' → WRONG (numeric code, not a name).\n"
            f"Example: current='大摩资源优选混合(LOF)' (tier 2) but source "
            f"has '交易代码为大摩资源' (tier 3) → correct to '大摩资源'.\n\n"
            f"NEVER output NONE for a field that already has a plausible "
            f"value unless the source text explicitly contradicts it. "
            f"Semantic synonyms in field labels (e.g. '内部管理编号' vs "
            f"'内部代码') do NOT count as contradictions.\n\n"
            f"For each field whose extracted value is WRONG, output:\n"
            f"  field_name: correct_value\n"
            f"If the source text does not mention a field at all, output:\n"
            f"  field_name: NONE\n"
            f"Output ONLY wrong fields, one per line. "
            f"NEVER repeat correct values. NEVER output all fields. "
            f"No commentary or analysis.\n"
            f"If all values are correct, output: ALL_CORRECT\n\n"
            f"Example — if only totalprofit is wrong:\n"
            f"GOOD:\n"
            f"  totalprofit: 23890000\n"
            f"BAD (repeats correct values):\n"
            f"  secuabbr: XX | totalprofit: 23890000 | fund_name: YY\n"
            f"BAD (adds commentary):\n"
            f"  totalprofit: 23890000\n"
            f"  **Analysis:** the source says..."
        )

        try:
            response = adapter.complete([_MM(role="user", content=prompt)])
        except Exception:
            logger.debug("ETL field verify failed for %s", rid, exc_info=True)
            return rid, {}

        reply = response.content.strip()
        if reply.startswith("```"):
            reply = reply.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        if "ALL_CORRECT" in reply.upper():
            return rid, {}

        # Cross-field collision guard: reject corrections that duplicate
        # (a) an anchor/identity field value, or
        # (b) a value from a field with a different schema type.
        _anchor_vals: set[str] = set()
        _typed_vals: dict[str, set[str]] = {}
        for c in table.columns:
            v = row.get(c, "").strip()
            if not v:
                continue
            if c.lower() in anchor_lower:
                _anchor_vals.add(v)
            else:
                t = types.get(c, "").lower()
                if t:
                    _typed_vals.setdefault(t, set()).add(v)

        def _is_cross_field_collision(field: str, value: str) -> bool:
            if value in _anchor_vals:
                return True
            ft = types.get(field, "").lower()
            if not ft:
                return False
            return any(value in vs for ot, vs in _typed_vals.items() if ot != ft)

        corrections: dict[str, str] = {}
        col_by_lower = {c.lower(): c for c in verify_cols}

        def _parse_correction(field: str, raw_val: str) -> str | None:
            val = strip_reasoning_leak(raw_val.strip())
            if not val:
                return None
            if val.upper().startswith("NONE"):
                return ""
            type_text = types.get(field, "").lower()
            if ("number" in type_text or "integer" in type_text) and "rank" not in type_text:
                normalized = val.replace(",", "")
                try:
                    float(normalized)
                except ValueError:
                    return None
                return normalized
            return val

        # Strip LLM commentary: stop at blank lines or markdown headers
        clean_lines: list[str] = []
        for line in reply.splitlines():
            stripped = line.strip()
            if not stripped:
                break
            if stripped.startswith(("**", "#")):
                break
            clean_lines.append(line)

        # Expand pipe-separated KV lines into individual segments
        segments: list[str] = []
        for line in clean_lines:
            if "|" in line and ":" in line.split("|", 1)[1]:
                segments.extend(seg.strip() for seg in line.split("|") if seg.strip())
            else:
                segments.append(line)

        for seg in segments:
            if ":" not in seg:
                continue
            key, _, val = seg.partition(":")
            key = col_by_lower.get(key.strip().lower())
            if not key:
                continue
            parsed = _parse_correction(key, val)
            if parsed is None:
                continue
            current = row.get(key, "").strip()
            key_type = types.get(key, "").lower()
            if ("number" in key_type or "integer" in key_type) and "rank" not in key_type:
                current = current.replace(",", "")
            if parsed != current:
                if parsed == "" and current:
                    logger.warning(
                        "ETL field verify: rejecting NONE correction for %s "
                        "(entity %s already has value %r)",
                        key,
                        rid,
                        current,
                    )
                    continue
                if _is_cross_field_collision(key, parsed):
                    logger.warning(
                        "ETL field verify: rejecting correction %s=%r for entity %s "
                        "(collides with anchor or cross-type field value)",
                        key,
                        parsed,
                        rid,
                    )
                    continue
                corrections[key] = parsed
        return rid, corrections

    all_corrections: dict[str, dict[str, str]] = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {submit_in_context(pool, _verify_one, rid): rid for rid in verify_ids}
        try:
            for fut in as_completed(futs, timeout=LLM_CALL_TIMEOUT * len(verify_ids)):
                try:
                    rid, corr = fut.result(timeout=LLM_CALL_TIMEOUT)
                    if corr:
                        all_corrections[rid] = corr
                except Exception:
                    pass
        except TimeoutError:
            logger.warning("ETL field verify: pool timed out")

    if not all_corrections:
        logger.info("ETL field verify: all values confirmed correct")
        return

    patched = 0
    for rid, corrections in all_corrections.items():
        applied = 0
        for record in by_pk.get(rid, []):
            for col, val in corrections.items():
                if val == "":
                    continue  # NONE 修正在解析层已被拦（非空现值）或为 no-op
                _apply_llm_value(table, record, col, val, "verify")
                applied += 1
        if applied:
            patched += applied
            logger.info(
                "ETL field verify: corrected entity %s: %s",
                rid,
                dict(corrections),
            )

    logger.info(
        "ETL field verify: corrected %d values across %d entities",
        patched,
        len(all_corrections),
    )
