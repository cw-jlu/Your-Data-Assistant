"""Schema inference, parsing, and multi-round merging."""

from __future__ import annotations

import logging
import re
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING

from agents.etl._columns import metadata_by_name
from agents.etl._compress import sample_paragraphs_by_section
from agents.etl._constants import (
    AIRTABLE_ID_RE,
    CODE_ID_RE,
    FUND_FIELD_DISAMBIGUATION_PROMPT,
    LOCAL_RECORD_ID_RE,
    RECORD_ID_FIELDS,
    SCHEMA_RESPONSE_PREFIX,
    SCHEMA_SAMPLE_ROUNDS,
)
from agents.etl.knowledge import km_table_fields
from agents.llm import ModelAdapter, ModelMessage, ModelResponse

if TYPE_CHECKING:
    from agents.benchmark.schema import PublicTask

logger = logging.getLogger(__name__)


def _has_local_record_ids(prose_text: str) -> bool:
    """Return whether prose uses a repeated document-local record identifier.

    Checks both keyword-prefixed numeric IDs ("Record 29", "档案 29") and
    alphanumeric IDs (Airtable-style rec*, code-style TR*).
    """
    matches = LOCAL_RECORD_ID_RE.findall(prose_text)
    if len(set(matches)) >= 3:
        return True
    if len(set(AIRTABLE_ID_RE.findall(prose_text))) >= 3:
        return True
    # Code-style IDs count only when one shared letter prefix carries >=3
    # distinct numbers (TR391/TR483/TR512). Mixed prefixes are domain codes
    # (exchange tickers, ISO labels), not a record-id scheme.
    codes_by_prefix: dict[str, set[str]] = {}
    for prefix, number in CODE_ID_RE.findall(prose_text):
        codes_by_prefix.setdefault(prefix, set()).add(number)
    return any(len(numbers) >= 3 for numbers in codes_by_prefix.values())


def _ensure_local_record_anchor(
    prose_text: str,
    columns: list[str],
    anchor_keys: list[str],
    explicit_types: dict[str, str] | None = None,
) -> tuple[list[str], list[str]]:
    """Add a document-local record ID anchor when prose is sectioned by one.

    Some generated prose files describe the same row across multiple sections
    using labels like "档案 353" or "战略单元 353".  Those labels are not domain
    keys such as ``personalcode``; they are a local row key needed to merge the
    partial facts before downstream analysis.
    """
    known_fields = {c.lower() for c in columns} | {a.lower() for a in anchor_keys}

    def _promote_record_id(cols: list[str], ancs: list[str]) -> tuple[list[str], list[str]]:
        """Ensure record_id is first in both columns and anchor_keys."""
        rid = next((c for c in cols if c.lower() in RECORD_ID_FIELDS), "record_id")
        if cols[0].lower() not in RECORD_ID_FIELDS:
            cols = [rid] + [c for c in cols if c != rid]
        if not ancs or ancs[0].lower() not in RECORD_ID_FIELDS:
            ancs = [rid] + [a for a in ancs if a != rid]
        return cols, ancs

    if known_fields & RECORD_ID_FIELDS:
        return _promote_record_id(columns, anchor_keys)
    if explicit_types:
        for field, ftype in explicit_types.items():
            if field.lower() in known_fields and "local_record_id" in ftype.lower():
                return _promote_record_id(columns, anchor_keys)
    if not _has_local_record_ids(prose_text):
        return columns, anchor_keys

    columns = ["record_id", *columns]
    anchor_keys = ["record_id", *anchor_keys]
    logger.info("ETL schema: added local record_id anchor from prose record labels")
    return columns, anchor_keys


def _parse_schema_types_line(line: str) -> dict[str, str]:
    """Parse TYPES: field=type, field=type output from schema extraction."""
    if not line.upper().startswith("TYPES"):
        return {}
    raw = line.split(":", 1)[-1]
    parsed: dict[str, str] = {}
    for item in re.split(r"[,;]", raw):
        item = item.strip()
        if not item:
            continue
        field, sep, field_type = item.partition("=")
        if not sep:
            field, sep, field_type = item.partition(":")
        if sep and field.strip() and field_type.strip():
            parsed[field.strip()] = field_type.strip()
    return parsed


def _parse_kv_line(line: str) -> dict[str, str]:
    """Parse ``field=value, field=value`` from any prefixed line."""
    raw = line.split(":", 1)[-1]
    parsed: dict[str, str] = {}
    for item in re.split(r"[,;]", raw):
        item = item.strip()
        if not item:
            continue
        field, sep, val = item.partition("=")
        if not sep:
            field, sep, val = item.partition(":")
        if sep and field.strip() and val.strip():
            parsed[field.strip()] = val.strip()
    return parsed


def _parse_defs_line(line: str) -> dict[str, str]:
    """Parse ``DEFS: field=definition; field=definition`` output.

    Splits on semicolons only (definitions may contain commas).
    """
    if not line.upper().startswith("DEF"):
        return {}
    raw = line.split(":", 1)[-1]
    parsed: dict[str, str] = {}
    for item in raw.split(";"):
        item = item.strip()
        if not item:
            continue
        field, sep, defn = item.partition("=")
        if sep and field.strip() and defn.strip():
            parsed[field.strip()] = defn.strip()
    return parsed


def _canonicalize_casefold_names(
    names: list[str],
    preferred: dict[str, str] | None = None,
) -> tuple[list[str], list[tuple[str, str]]]:
    """Deduplicate names that differ only by case.

    When *preferred* is provided, its values define the spelling to keep for
    matching lower-case keys.
    """
    preferred_by_lower = {k.lower(): v for k, v in (preferred or {}).items()}
    canonical_by_lower: dict[str, str] = {}
    result: list[str] = []
    remapped: list[tuple[str, str]] = []
    for name in names:
        low = name.lower()
        target = preferred_by_lower.get(low, name)
        existing = canonical_by_lower.get(low)
        if existing is not None:
            if name != existing:
                remapped.append((name, existing))
            continue
        canonical_by_lower[low] = target
        result.append(target)
        if name != target:
            remapped.append((name, target))
    return result, remapped


def _parse_schema_response(
    text: str,
) -> tuple[str, list[str], dict[str, str], list[str], dict[str, str], dict[str, str]] | None:
    """Parse LLM schema output by line prefix, not position.

    Looks for PK:, ANCHORS:, TYPES:, UNITS:, DEFS: prefixes in any order.
    The first non-prefixed line with commas is treated as the column list.

    Returns (pk, anchor_keys, explicit_types, columns, field_units, field_defs)
    or None.
    """
    lines = [line.strip() for line in text.strip().strip("`").splitlines() if line.strip()]
    if len(lines) < 2:
        return None

    pk: str | None = None
    anchor_keys: list[str] = []
    explicit_types: dict[str, str] = {}
    field_units: dict[str, str] = {}
    field_defs: dict[str, str] = {}
    columns: list[str] = []

    for line in lines:
        m = SCHEMA_RESPONSE_PREFIX.match(line)
        if m:
            tag = m.group(1).upper()
            if tag == "PK":
                pk = line.split(":", 1)[-1].strip()
            elif tag.startswith("ANCHOR"):
                raw = [a.strip() for a in line.split(":", 1)[-1].split(",") if a.strip()]
                if raw:
                    anchor_keys = raw
            elif tag.startswith("TYPE"):
                explicit_types = _parse_schema_types_line(line)
            elif tag.startswith("UNIT"):
                field_units = _parse_kv_line(line)
            elif tag.startswith("DEF"):
                field_defs = _parse_defs_line(line)
        elif "," in line and not columns and "=" not in line:
            tokens = [c.strip() for c in line.split(",") if c.strip()]
            if all(len(t.split()) <= 3 for t in tokens):
                columns = tokens

    if not pk or len(columns) < 2:
        return None

    columns, _ = _canonicalize_casefold_names(columns)
    column_by_lower = {c.lower(): c for c in columns}
    pk = column_by_lower.get(pk.lower(), pk)

    if not anchor_keys:
        anchor_keys = [pk]
    elif pk.lower() not in {a.lower() for a in anchor_keys}:
        anchor_keys.insert(0, pk)

    canonical_anchors: list[str] = []
    seen_anchors: set[str] = set()
    for ak in anchor_keys:
        canonical = column_by_lower.get(ak.lower(), ak)
        canonical_lower = canonical.lower()
        if canonical_lower in seen_anchors:
            continue
        canonical_anchors.append(canonical)
        seen_anchors.add(canonical_lower)
        if canonical_lower not in column_by_lower:
            columns.append(canonical)
            column_by_lower[canonical_lower] = canonical

    return pk, canonical_anchors, explicit_types, columns, field_units, field_defs


def _complete_schema_field_defs(
    columns: list[str],
    field_defs: dict[str, str] | None = None,
) -> dict[str, str]:
    """Canonicalize LLM-provided field definitions to match schema column spelling."""
    if not field_defs:
        return {}
    canonical = {column.lower(): column for column in columns}
    result: dict[str, str] = {}
    for field, defn in field_defs.items():
        column = canonical.get(field.lower())
        if column:
            result[column] = defn
    return result


def _complete_schema_field_types(
    columns: list[str],
    explicit_types: dict[str, str] | None = None,
) -> dict[str, str]:
    """Canonicalize LLM-provided field types to match schema column spelling."""
    if not explicit_types:
        return {}
    canonical = {column.lower(): column for column in columns}
    result: dict[str, str] = {}
    for field, field_type in explicit_types.items():
        column = canonical.get(field.lower())
        if column:
            result[column] = field_type
    return result


def merge_prose_schema_into_governance(
    adapter: ModelAdapter,
    prose_text: str,
    gov_columns: list[str],
    gov_defs: dict[str, str],
    gov_types: dict[str, str],
    question: str,
    table_name: str | None = None,
) -> tuple[list[str], dict[str, str], dict[str, str]]:
    """Union governance + prose-inferred schemas with synonym dedup.

    Independently infers a schema from the prose sample (no governance
    context, so it discovers ALL fields without bias), then asks the LLM
    to adjudicate which prose fields are synonyms of governance fields
    and which are genuinely distinct.

    Returns (extra_columns, extra_types, extra_defs) — only the fields
    NOT already covered by a governance synonym.
    """
    inferred = _infer_schema_single(
        adapter, sample_paragraphs_by_section(prose_text), question, table_name=table_name
    )
    if not inferred:
        return [], {}, {}

    prose_cols, _prose_anchors, prose_types_raw, _prose_units, prose_defs_raw = inferred
    prose_types = _complete_schema_field_types(prose_cols, prose_types_raw)
    prose_defs = _complete_schema_field_defs(prose_cols, prose_defs_raw)

    from agents.etl._merge import strip_key

    gov_defs = metadata_by_name(gov_columns, gov_defs)
    gov_norm = {strip_key(c) for c in gov_columns}
    novel = [
        c for c in prose_cols if strip_key(c) not in gov_norm and c.lower() not in RECORD_ID_FIELDS
    ]
    if not novel:
        return [], {}, {}

    gov_block = "\n".join(f"- {c}: {gov_defs.get(c, '(no definition)')}" for c in gov_columns)
    novel_block = "\n".join(f"- {c}: {prose_defs.get(c, '(no definition)')}" for c in novel)

    prompt = textwrap.dedent(f"""\
        A data governance document defines these CANONICAL fields:
        {gov_block}

        An independent prose scan discovered these ADDITIONAL fields:
        {novel_block}

        For each ADDITIONAL field, decide:
        - If it is an EXACT SYNONYM of one canonical field (same concept,
          scope, aggregation level, unit), output: additional=canonical
        - If it is DISTINCT (different concept, sub-total vs grand-total,
          component vs aggregate, different time window, etc.), output:
          additional=DISTINCT

        Sub-totals and aggregates are ALWAYS DISTINCT even when names are
        similar (e.g. reserve_assets = deposits + cash ≠ totalassets = all
        assets). Component fields are DISTINCT from their parent total.

        {FUND_FIELD_DISAMBIGUATION_PROMPT}

        Output one line per additional field, nothing else.""")

    messages = [
        ModelMessage(
            role="system",
            content="Adjudicate whether prose-discovered fields are synonyms "
            "of governance fields. Output mapping lines only.",
        ),
        ModelMessage(role="user", content=prompt),
    ]
    try:
        response: ModelResponse = adapter.complete(messages)
    except Exception as exc:
        logger.warning("Schema synonym adjudication failed: %s", exc)
        return [], {}, {}

    raw = response.content.strip()
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    distinct: list[str] = []
    for line in raw.splitlines():
        line = line.strip().lstrip("- ")
        if "=" not in line:
            continue
        left, _, right = line.partition("=")
        left = left.strip().lower()
        right = right.strip().lower()
        if right == "distinct" and left in {c.lower() for c in novel}:
            orig = next(c for c in novel if c.lower() == left)
            distinct.append(orig)

    extra_cols = distinct
    extra_types = {c: prose_types.get(c, "number") for c in extra_cols}
    extra_defs = {c: prose_defs.get(c, "") for c in extra_cols}
    return extra_cols, extra_types, extra_defs


def _merge_synonyms_into_km(
    adapter: ModelAdapter,
    columns: list[str],
    raw_defs: dict[str, str],
    km_fields: dict[str, str],
) -> list[tuple[str, str]]:
    """Merge LLM-inferred columns that are synonyms of governance columns.

    Columns already in *km_fields* are canonical. Any other column in
    *columns* that the LLM identifies as a synonym of a governance column
    gets removed, and its data merges into the governance column during
    downstream compression.  Mutates *columns* and *raw_defs* in place.
    Returns list of (removed, canonical) pairs.
    """
    km_canon = {c.lower(): c for c in km_fields}
    canonical_columns, merged = _canonicalize_casefold_names(columns, km_canon)
    if merged:
        columns[:] = canonical_columns

    km_lower = set(km_canon)
    rid_fields = {c.lower() for c in RECORD_ID_FIELDS}
    non_km = [c for c in columns if c.lower() not in km_lower and c.lower() not in rid_fields]
    if not non_km:
        return merged

    km_block = "\n".join(f"- {c}: {km_fields[c]}" for c in km_fields)
    extra_block = "\n".join(f"- {c}: {raw_defs.get(c, '(no definition)')}" for c in non_km)
    prompt = textwrap.dedent(f"""\
        A data governance document defines these CANONICAL fields:
        {km_block}

        An LLM schema inference also produced these EXTRA fields:
        {extra_block}

        For each EXTRA field, decide:
        - SYNONYM (output: extra=canonical) when the extra field measures
          the SAME real-world quantity as a canonical field. Merge even if:
          - the names differ (report_date vs enddate — both are the
            reporting period end date)
          - one name is a typo/abbreviation of the other (endate vs
            enddate)
          - the definitions mention different formats or phrasings for the
            same concept (e.g. "raw date text" vs "date in YYYY-MM-DD" —
            same date, different representation)
        - DISTINCT (output: extra=DISTINCT) ONLY when the extra field
          measures a genuinely DIFFERENT quantity. Examples of DISTINCT:
          - different time windows (daily vs weekly growth rate)
          - different aggregation levels (per-fund vs per-company total)
          - component vs aggregate (net_profit vs total_profit)
          - before vs after state (pre-transfer vs post-transfer shares)

        When in doubt, prefer SYNONYM — two columns for the same concept
        causes value split (some rows fill one column, others fill the
        other) which is a critical data quality error.

        Output one line per extra field, nothing else.""")
    messages = [
        ModelMessage(
            role="system",
            content=(
                "Adjudicate whether extra fields are synonyms of canonical "
                "governance fields. Judge by the REAL-WORLD QUANTITY measured, "
                "not by name spelling or definition wording."
            ),
        ),
        ModelMessage(role="user", content=prompt),
    ]
    try:
        response: ModelResponse = adapter.complete(messages)
    except Exception as exc:
        logger.warning("Synonym merge into km failed: %s", exc)
        return []

    raw = response.content.strip()
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    non_km_lower = {c.lower(): c for c in non_km}
    for line in raw.splitlines():
        line = line.strip().lstrip("- ")
        if "=" not in line:
            continue
        left, _, right = line.partition("=")
        left_low = left.strip().lower()
        right_low = right.strip().lower()
        if right_low == "distinct":
            continue
        orig = non_km_lower.get(left_low)
        canon = km_canon.get(right_low)
        if orig and canon and orig in columns:
            columns.remove(orig)
            merged.append((orig, canon))
    return merged


def _remap_schema_metadata(
    merged: list[tuple[str, str]],
    explicit_types: dict[str, str],
    field_units: dict[str, str],
    raw_defs: dict[str, str],
) -> None:
    """Move metadata from removed synonym columns to their canonical columns."""

    def remap_dict(values: dict[str, str], old: str, new: str) -> None:
        old_key = next((key for key in values if key.lower() == old.lower()), None)
        if old_key is None:
            return
        new_key = next(
            (key for key in values if key.lower() == new.lower() and key != old_key),
            None,
        )
        old_value = values.pop(old_key)
        if new_key is None:
            values[new] = old_value

    for old, new in merged:
        remap_dict(explicit_types, old, new)
        remap_dict(field_units, old, new)
        remap_dict(raw_defs, old, new)


def _remap_anchor_keys(anchor_keys: list[str], merged: list[tuple[str, str]]) -> list[str]:
    """Canonicalize anchor keys after synonym columns have been removed."""
    synonym_map = {old.lower(): new for old, new in merged}
    remapped: list[str] = []
    seen: set[str] = set()
    for key in anchor_keys:
        canonical = synonym_map.get(key.lower(), key)
        canonical_lower = canonical.lower()
        if canonical_lower in seen:
            continue
        remapped.append(canonical)
        seen.add(canonical_lower)
    return remapped


def _apply_km_schema_floor(
    columns: list[str],
    raw_defs: dict[str, str],
    km_fields: dict[str, str],
) -> list[str]:
    """Re-add governance-listed columns the LLM omitted from the schema.

    The governance document's field table is authoritative: a column listed
    there must survive into the schema even when the prose sample happens to
    show no values for it (task_53 lost ``totalliabilities`` exactly this
    way).  Mutates ``columns``/``raw_defs`` in place; returns the re-added
    column names.

    Typo/synonym resolution (``endate`` → ``enddate``) is handled by
    ``_merge_synonyms_into_km`` which uses an LLM call — not edit-distance
    heuristics — so short governance names like ``gdp`` or ``type`` cannot
    false-positive against unrelated columns.
    """
    existing = {c.lower() for c in columns}
    readded = [c for c in km_fields if c.lower() not in existing]
    for c in readded:
        columns.append(c)
        raw_defs.setdefault(c, km_fields[c])
    return readded


def _extract_relevant_entity(
    adapter: ModelAdapter,
    km_text: str,
    prose_stem: str,
    prose_sample: str,
) -> str | None:
    """Pre-filter knowledge.md to only the entity section matching the prose file.

    Sends a lightweight LLM call to identify which entity in the governance
    document corresponds to the prose file, and returns only that section.
    This prevents cross-table column contamination in downstream schema inference.
    """
    prompt = textwrap.dedent(f"""\
        Below is a data governance document that describes MULTIPLE database \
        tables / entities. I also provide a prose data file name and a short \
        sample from it.

        Prose file: "{prose_stem}"
        Prose sample (first few lines):
        ---
        {prose_sample[:1500]}
        ---

        Task: find the ONE entity / table section in the governance document \
        that best matches this prose file. Copy that section VERBATIM — \
        include the heading, all column definitions, all footnotes and units \
        that belong to it. Do NOT include sections for other entities.

        If no entity matches, output exactly: NONE

        Governance document:
        {km_text}""")
    messages = [
        ModelMessage(
            role="system",
            content="You extract the relevant section from a governance document. Output only the matched section verbatim, nothing else.",
        ),
        ModelMessage(role="user", content=prompt),
    ]
    try:
        response: ModelResponse = adapter.complete(messages)
    except Exception as exc:
        logger.warning("LLM entity pre-filter failed: %s", exc)
        return None

    raw = response.content.strip()
    if not raw or raw.strip().strip("`").upper() == "NONE":
        return None
    if len(raw) < 20:
        return None
    logger.info(
        "Knowledge pre-filter for '%s': %d → %d chars",
        prose_stem,
        len(km_text),
        len(raw),
    )
    return raw


def parse_knowledge_schema(
    task: PublicTask, prose_path: Path, adapter: ModelAdapter, prose_text: str
) -> tuple[list[str], list[str], dict[str, str], dict[str, str], dict[str, str]] | None:
    """Use LLM to extract column names, anchor keys, units, and definitions from knowledge.md.

    Samples the prose file so the LLM can discover fields present in the data
    but missing from the governance document (e.g. record IDs).

    Returns (columns, anchor_keys, field_types, field_units, field_defs) or None.
    ``anchor_keys[0]`` is the canonical primary key; subsequent entries are
    secondary identity fields (name, label, secondary ID) that can uniquely
    identify an entity when the PK takes different forms across sections.
    """
    km_path = task.context_dir / "knowledge.md"
    if not km_path.is_file():
        return None
    try:
        km_text = km_path.read_text(encoding="utf-8")
    except OSError:
        return None

    prose_stem = prose_path.stem

    # Deterministic gate: a governance document that never mentions this
    # table cannot define its schema.  Skipping the knowledge path outright
    # keeps the LLM entity matcher from force-matching a DIFFERENT table's
    # section (task_59 ed_consumerpriceindex inherited retailvalueofscgoods
    # columns that way) and saves two LLM calls.
    if prose_stem.lower() not in km_text.lower():
        logger.info(
            "ETL schema: '%s' not mentioned in knowledge.md — skipping knowledge path",
            prose_stem,
        )
        return None

    prose_sample = sample_paragraphs_by_section(prose_text) if prose_text else ""

    # Deterministic field floor, parsed from the ORIGINAL governance text —
    # before the LLM pre-filter gets a chance to drop table rows.
    km_fields = km_table_fields(km_text, prose_stem)

    filtered_km = _extract_relevant_entity(adapter, km_text, prose_stem, prose_sample or "")
    if filtered_km and prose_stem.lower() in filtered_km.lower():
        km_text = filtered_km
    elif filtered_km:
        # The filter returned a section that never mentions this table — it
        # matched some other entity.  A real section would carry the table
        # name in its heading; treat as no-match instead of letting the
        # schema call inherit a wrong table's columns.
        logger.warning(
            "ETL schema: knowledge pre-filter for '%s' returned a section "
            "not mentioning it — treating as no-match",
            prose_stem,
        )
        return None

    prose_section = ""
    if prose_sample and not km_fields:
        prose_section = (
            f"\n\nSample paragraphs from EVERY SECTION of the prose file "
            f'"{prose_stem}.md" (use these to discover ALL numeric/categorical '
            f"data fields present in the prose, even if the governance document "
            f"does not list them):\n"
            f"---\n{prose_sample}\n---"
        )

    kb_ref = _build_kb_reference(prose_stem)

    prompt = textwrap.dedent(f"""\
        Below is a data governance document AND a sample from a prose data file.
        Find the entity in the governance document that best matches \
        the prose file "{prose_stem}".

        If NO entity in the governance document matches, output exactly: NONE

        Otherwise output exactly SIX lines:
        Line 1: PK: <primary_key_field_name>
        Line 2: ANCHORS: <comma-separated identity fields that can uniquely \
        identify an entity — includes PK plus name/label/secondary ID fields>
        Line 3: TYPES: <field=type, field=type, ... for every field>
        Line 4: UNITS: <field=unit for each numeric field whose values carry \
        a unit in the prose sample or governance document. Use _default=<unit> \
        for the governance document's declared default monetary unit.>
        Line 5: DEFS: <field=one-sentence semantic definition; field=definition; \
        ... for every field. Use semicolons between entries. Copy the exact \
        semantic definition from the governance document when available; for \
        fields discovered only in the prose sample, write a concise definition \
        that distinguishes this field from similarly-named fields.>
        Line 6: all field names as a comma-separated list (must include all ANCHORS)

        IMPORTANT:
        - CRITICAL: columns listed in the governance document already have \
        canonical database names (e.g. "sumbeforetran", "pctbeforetran", \
        "companycode"). You MUST use these EXACT names verbatim in PK, \
        ANCHORS, TYPES, UNITS, DEFS, and the column list. NEVER rename, \
        translate, expand, or paraphrase them — "sumbeforetran" stays \
        "sumbeforetran", NOT "shares_held_before" or "sum_before_tran".
        - Only invent new field names for fields discovered in the prose sample \
        that are not fully covered by any governance column's definition. \
        New names must be SHORT GENERIC FIELD NAMES (like "event_id", \
        "tran_date"), NEVER a specific data value.
        - ANCHORS should include the PK and any name/label/secondary identifier \
        field that could be used to recognize the same entity across different \
        sections of the document (e.g. a person's name, a record label). \
        Numeric measure/amount fields (GDP, revenue, price, count, ratio) are \
        NEVER anchors — anchors are identifiers, not data values.
        - MANDATORY: include EVERY column listed in the governance document's \
        table for the matched entity — even when the prose sample shows no \
        values for it (the sample covers a fraction of the document; absence \
        from the sample is NOT absence from the data). Omitting a governance \
        column is an error.
        - Use simple type labels such as integer_scalar_id, scalar_id, date, time, \
        number, integer_count, integer_rank, string, category, boolean, url.
        - For rank fields with "xx/yy" format (e.g. "43/166"), use type "string", \
        NOT "integer_rank". "integer_rank" is only for pure numeric ranks.
        - For document-local record_id fields: use type \
        "integer_scalar local_record_id" when the IDs are pure numbers, or \
        "scalar_id local_record_id" when the IDs are alphanumeric \
        (e.g. rec0Si5cQ4rJRVzd6, TR391). \
        Record_id values must be a single scalar, not a range or list.
        - MANDATORY: scan the prose sample for numeric, date, and \
        categorical data fields that have NO matching governance column. \
        For each prose field, check every governance column's \
        definition / description. A governance column covers a prose field \
        ONLY when it matches the same concept AND all binding qualifiers: \
        time window, before/after state, numerator/denominator, entity scope, \
        category, unit, and aggregation level. Opaque database names are \
        canonical when their definitions match exactly; do NOT add translated \
        aliases for them. If a prose field is only related to a governance \
        column but differs in any binding qualifier, treat it as a distinct \
        discovered field and add a short descriptive name. \
        Example — WRONG: governance has "rrintenyear" defined as \
        "近十年累计回报率", prose shows "近十年累计回报率: 166.09" → adding \
        "ten_year_return" creates a duplicate column and splits values. \
        CORRECT: recognise "近十年累计回报率" matches "rrintenyear" → use \
        "rrintenyear", do NOT add "ten_year_return". \
        Example — CORRECT addition: governance has "sumbeforetran" \
        ("转让前持股") but prose ALSO has "转让数量" which no governance \
        column covers → add "transfer_shares" as a new discovered field. \
        Two columns for the same concept causes value split (some rows \
        fill column A, others fill column B) and is a critical schema error.
        - If paragraphs are keyed by repeated document-local labels such as \
        "档案 353", "Record 353", "Registry ID: rec0Si5cQ4rJRVzd6", \
        "identifier recXxx", "designated TR391", AND no REFERENCE SCHEMA \
        below declares a primary key, include a field named "record_id" and \
        put it FIRST in PK and ANCHORS. If a REFERENCE SCHEMA declares a \
        primary key, use that column instead. This local record ID is the \
        row key for merging sections; it is NOT the same as domain IDs such \
        as personalcode, innercode, fundcode, or secucode.
        - Do NOT add narrative/filler fields — only structured data fields.
        {FUND_FIELD_DISAMBIGUATION_PROMPT}
        - For the UNITS line: scan the prose sample for unit suffixes attached \
        to numeric values (e.g. "2478.76亿元" → gdp=亿元, "15.3%" → rate=%). \
        Output a field=unit pair when you find a CONCRETE unit suffix \
        (e.g. 元, 万元, 亿元, %, 万吨, 万人, 公斤) in the actual data values \
        or governance document. Fields whose values appear with "%" in the \
        prose or are described as "percentage" / "比例" in the governance \
        document MUST be tagged with unit=%. \
        A currency denomination statement (e.g. "denominated in Chinese \
        Yuan", "currency: CNY", "以人民币计价") declares the CURRENCY, not \
        the magnitude — it is NOT a unit. NEVER emit field=元 from such a \
        statement; only tag a field 元 when its numeric values carry an \
        explicit 元 suffix. Monetary fields with no visible magnitude suffix \
        are already in the governance document's default unit and must be \
        OMITTED. \
        Dimensionless fields (price indices, ranks, counts without unit) and \
        fields with no visible unit suffix \
        must be OMITTED entirely — do NOT write them with "_default", \
        "currency", "units", "amount", "count", or any placeholder.

        Governance document:
        {km_text}{prose_section}
        {kb_ref}""")
    messages = [
        ModelMessage(
            role="system",
            content=(
                "You extract database schema. "
                "Output PK, ANCHORS, TYPES, UNITS, DEFS, then comma-separated columns."
            ),
        ),
        ModelMessage(role="user", content=prompt),
    ]
    try:
        response: ModelResponse = adapter.complete(messages)
    except Exception as exc:
        logger.warning("LLM schema extraction failed: %s", exc)
        return None

    raw = response.content.strip()
    if not raw or raw.strip().strip("`").upper() == "NONE":
        if prose_stem.lower() in km_text.lower():
            logger.warning(
                "ETL schema: LLM returned NONE for '%s' but knowledge section "
                "contains the table name — retrying with hint",
                prose_stem,
            )
            hint = ModelMessage(
                role="user",
                content=(
                    f"The governance document DOES contain a section for "
                    f'"{prose_stem}" — it appears in the heading. '
                    f"Do NOT output NONE. Extract the schema from that section."
                ),
            )
            try:
                response = adapter.complete(
                    [
                        *messages,
                        ModelMessage(role="assistant", content="NONE"),
                        hint,
                    ]
                )
            except Exception:
                return None
            raw = response.content.strip()
            if not raw or raw.strip().strip("`").upper() == "NONE":
                logger.warning("ETL schema: retry also returned NONE for '%s'", prose_stem)
                return None
        else:
            return None

    parsed = _parse_schema_response(raw)
    if not parsed:
        return None
    _, anchor_keys, explicit_types, columns, field_units, raw_defs = parsed

    if km_fields:
        readded = _apply_km_schema_floor(columns, raw_defs, km_fields)
        if readded:
            logger.warning(
                "ETL schema: LLM omitted %d columns listed in knowledge.md for '%s': %s — re-added",
                len(readded),
                prose_stem,
                readded,
            )
        merged = _merge_synonyms_into_km(adapter, columns, raw_defs, km_fields)
        if merged:
            _remap_schema_metadata(merged, explicit_types, field_units, raw_defs)
            anchor_keys = _remap_anchor_keys(anchor_keys, merged)
            logger.info(
                "ETL schema: merged %d synonym columns into governance names for '%s': %s",
                len(merged),
                prose_stem,
                merged,
            )

    columns, anchor_keys = _ensure_local_record_anchor(
        prose_text,
        columns,
        anchor_keys,
        explicit_types,
    )
    field_types = _complete_schema_field_types(columns, explicit_types)
    field_defs = _complete_schema_field_defs(columns, raw_defs)

    logger.info(
        "Schema from knowledge.md (LLM) for '%s': pk=%s, anchors=%s, types=%s, units=%s, defs=%s, cols=%s",
        prose_stem,
        anchor_keys[0],
        anchor_keys,
        field_types,
        field_units,
        field_defs,
        columns,
    )
    return columns, anchor_keys, field_types, field_units, field_defs


def infer_field_units_from_prose(
    adapter: ModelAdapter,
    prose_sample: str,
    columns: list[str],
    field_defs: dict[str, str],
    existing_units: dict[str, str],
) -> dict[str, str]:
    """Identify units for *existing* schema fields by inspecting prose values.

    Returns {field: unit} for fields where a unit was detected.  Only fields
    already in ``columns`` may appear; no new fields are invented.  Entries
    already present in ``existing_units`` are never overwritten.
    """
    if not prose_sample or not columns:
        return {}
    field_defs_by_col = metadata_by_name(columns, field_defs)
    col_block = "\n".join(f"- {c}: {field_defs_by_col.get(c, '(no definition)')}" for c in columns)
    prompt = textwrap.dedent(f"""\
        Below are sample paragraphs from a prose data file AND a list of \
        known schema fields.

        UNIT EXTRACTION RULES

        1. WHEN TO EXTRACT A UNIT
        Output field=unit ONLY when the unit is visibly attached to that \
        field's numeric value, or the exact field/header explicitly carries \
        a magnitude unit. The unit must be a real magnitude suffix such as \
        元, 万元, 亿元, 百万元, %, 万吨, 万人, or 公斤.

        2. WHEN NOT TO EXTRACT A UNIT
        Do NOT output any unit when the text only states currency, \
        denomination, table scope, data type, or reporting currency. These \
        are currency labels, NOT magnitude units: "denominated in Chinese \
        Yuan", "currency: CNY", "CurrencyName: 人民币元", "RMB-denominated \
        balance sheet", "all monetary values are in Chinese Yuan", "以人民币计价". \
        If the numeric value appears as "CorporateSavings: 326283.0" with \
        no suffix, OMIT CorporateSavings entirely.

        3. EXAMPLES
        GOOD: "Corporate Savings were 326283万元." -> CorporateSavings=万元
        GOOD: "GDP was 2478.76亿元." -> GDP=亿元
        GOOD: "Growth rate was 2.5%." -> GrowthRate=%
        GOOD: field/header "国内生产总值(百万元)" -> 国内生产总值=百万元
        BAD: "Corporate Savings were 326283.0. CurrencyName: Chinese Yuan." \
        -> DO NOT output CorporateSavings=元; output nothing for CorporateSavings.
        BAD: "All figures are denominated in Chinese Yuan. TotalSavings was \
        965459.0." -> DO NOT output TotalSavings=元; output nothing for TotalSavings.

        ONLY output units for the fields listed below — do NOT invent \
        new fields. If no field qualifies, output nothing.

        Schema fields:
        {col_block}

        Prose sample:
        ---
        {prose_sample}
        ---""")
    messages = [
        ModelMessage(
            role="system",
            content=(
                "You identify visible magnitude-unit suffixes for known fields. "
                "Output field=unit lines only. If no numeric value visibly carries "
                "a unit suffix, output nothing. Currency labels such as Yuan/RMB/CNY "
                "are not magnitude units."
            ),
        ),
        ModelMessage(role="user", content=prompt),
    ]
    try:
        response: ModelResponse = adapter.complete(messages)
    except Exception as exc:
        logger.warning("ETL prose unit inference failed: %s", exc)
        return {}

    col_lower = {c.lower(): c for c in columns}
    existing_unit_lower = {field.lower() for field in existing_units}
    result: dict[str, str] = {}
    for line in response.content.strip().splitlines():
        line = line.strip().lstrip("- ")
        if "=" not in line:
            continue
        field, _, unit = line.partition("=")
        field, unit = field.strip(), unit.strip()
        canonical = col_lower.get(field.lower())
        if canonical and unit and canonical.lower() not in existing_unit_lower:
            result[canonical] = unit
    if result:
        logger.info("ETL prose unit inference: %s", result)
    return result


def _build_kb_reference(table_name: str | None) -> str:
    """Build a prompt section from schema KB if table_name matches."""
    if not table_name:
        return ""
    from agents.etl._schema_kb import lookup

    known = lookup(table_name)
    if not known:
        return ""
    col_list = ", ".join(f"{c} ({known.dtypes.get(c, 'string')})" for c in known.columns)
    rows_block = ""
    if known.sample_rows:
        header = " | ".join(known.columns)
        lines: list[str] = [header]
        for row in known.sample_rows[:3]:
            vals = " | ".join(str(row.get(c, "")) for c in known.columns)
            lines.append(vals)
        rows_block = "\n".join(lines)

    pk_note = ""
    if (
        known.columns
        and known.columns[0].lower() == "id"
        and known.dtypes.get(known.columns[0], "") == "integer"
    ):
        pk_note = (
            f'\n        "{known.columns[0]}" is the PRIMARY KEY / row identifier. '
            "Document-local record labels (however phrased) refer to this column. "
            "Use it as PK — do NOT create a separate record_id column."
        )

    return textwrap.dedent(f"""\

        REFERENCE SCHEMA for "{known.table_name}" (from structured data sources):
        Columns: {col_list}{pk_note}
        Sample data:
        {rows_block}

        ALWAYS use these exact column names. Match your output fields to this reference.
        Preserve the VALUE FORMAT shown in the sample data — the sample rows are the
        ground truth for how each field's values should be formatted.
        Example: if a sample value is "43/166", the prose may scatter the parts across
        the paragraph ("在包含 166 家机构中…取得了第43位") — combine them into "43/166".
        NEVER output just "43".
        LANGUAGE NOTE: the sample data language may differ from the source document.
        Column names are fixed (use as-is), but field VALUES must match the source
        document's language. E.g. sample shows "Equity Fund" but source says "股票型"
        → output "股票型".
        Only add extra fields if the prose clearly contains data not covered above.
    """)


def _infer_schema_single(
    adapter: ModelAdapter,
    sample: str,
    question: str,
    table_name: str | None = None,
) -> tuple[list[str], list[str], dict[str, str], dict[str, str], dict[str, str]] | None:
    """One round of schema inference. Returns raw parsed tuple or None."""
    kb_ref = _build_kb_reference(table_name)
    prompt = textwrap.dedent(f"""\
        Below is a sample from a prose data document. Each paragraph describes
        one entity and mixes structured data with narrative filler.

        Task question for context:
        {question}

        Sample (from multiple sections):
        ---
        {sample}
        ---
        {kb_ref}
        Your job:
        1. Identify ALL structured data fields present in the sample
           (numeric values, dates, categories, IDs, codes, statuses).
        2. Ignore narrative filler (descriptions, opinions, context sentences
           that don't carry a reusable data value).
        3. Assign each field a SHORT canonical name (snake_case, e.g.
           "effective_date", "reserve_ratio", "target_entity").
        4. Identify the primary key field (the field that uniquely identifies
           each entity — usually an ID number).
        5. Identify ANCHOR fields — identity fields (including PK) that can
           uniquely identify an entity even when the PK is missing or takes
           a different form (e.g. a name, label, or secondary ID).

        Output exactly SIX lines:
        Line 1: PK: <primary_key_field_name>
        Line 2: ANCHORS: <comma-separated identity fields including PK>
        Line 3: TYPES: <field=type, field=type, ... for every field>
        Line 4: UNITS: <field=unit for each numeric field whose values carry \
        a CONCRETE unit suffix in the prose sample (e.g. gdp=亿元, rate=%). \
        Only use real unit suffixes visible in the data values. Dimensionless \
        fields (price indices, ratios, ranks, counts without unit) and fields \
        with no visible unit suffix must be OMITTED — do NOT write them with \
        "_default", "currency", "units", "amount", or any placeholder.>
        Line 5: DEFS: <field=one-sentence semantic definition; field=definition; \
        ... for every field. Semicolons between entries. Write a concise \
        definition that distinguishes this field from similarly-named fields.>
        Line 6: all field names as a comma-separated list (must include all ANCHORS)

        Rules:
        - Column names must be SHORT GENERIC FIELD NAMES, NEVER specific data values.
        - Only include fields that carry actual structured data values.
        - Use simple type labels such as integer_scalar_id, scalar_id, date, time,
          number, integer_count, integer_rank, string, category, boolean, url.
        - For rank fields that use "xx/yy" format (e.g. "43/166"), use type
          "string", NOT "integer_rank". "integer_rank" is only for pure
          numeric ranks (e.g. 1, 2, 3).
        - If paragraphs are keyed by repeated document-local labels such as
          "档案 353", "Record 353", "Registry ID: rec0Si5cQ4rJRVzd6",
          "identifier recXxx", "designated TR391", include a field named
          "record_id" and put it FIRST in PK and ANCHORS. This local record ID
          is the row key for merging sections; it is NOT the same as domain IDs
          such as personalcode, innercode, fundcode, or secucode.
          Give record_id type "integer_scalar local_record_id" when IDs are
          pure numbers, or "scalar_id local_record_id" when alphanumeric.
          Record_id values must be a single scalar, not a range or list.
        {FUND_FIELD_DISAMBIGUATION_PROMPT}
        - Do NOT include narrative/context/filler fields.""")

    messages = [
        ModelMessage(
            role="system",
            content="You extract database schema from prose samples. Output PK, ANCHORS, TYPES, UNITS, DEFS, then columns.",
        ),
        ModelMessage(role="user", content=prompt),
    ]
    try:
        response: ModelResponse = adapter.complete(messages)
    except Exception as exc:
        logger.warning("LLM schema inference from sample failed: %s", exc)
        return None

    parsed = _parse_schema_response(response.content)
    if not parsed:
        return None
    _pk, anchor_keys, explicit_types, columns, field_units, raw_defs = parsed
    return columns, anchor_keys, explicit_types, field_units, raw_defs


def _merge_multi_round_schemas(
    adapter: ModelAdapter,
    rounds: list[tuple[list[str], list[str], dict[str, str], dict[str, str], dict[str, str]]],
) -> tuple[list[str], list[str], dict[str, str], dict[str, str], dict[str, str]]:
    """Merge schemas from multiple inference rounds: union columns, then LLM-adjudicate synonyms."""
    if len(rounds) == 1:
        return rounds[0]

    all_cols: list[str] = []
    seen_lower: set[str] = set()
    merged_anchors: list[str] = []
    seen_anchor_lower: set[str] = set()
    merged_types: dict[str, str] = {}
    merged_units: dict[str, str] = {}
    merged_defs: dict[str, str] = {}

    for cols, anchors, types, units, defs in rounds:
        for c in cols:
            if c.lower() not in seen_lower and c.lower() not in RECORD_ID_FIELDS:
                seen_lower.add(c.lower())
                all_cols.append(c)
        for a in anchors:
            a_lower = a.lower()
            if a_lower not in seen_anchor_lower and a_lower not in RECORD_ID_FIELDS:
                merged_anchors.append(a)
                seen_anchor_lower.add(a_lower)
        merged_types.update(types)
        merged_units.update(units)
        merged_defs.update(defs)

    if len(all_cols) <= len(rounds[0][0]):
        return all_cols, merged_anchors, merged_types, merged_units, merged_defs

    merged_defs_by_col = metadata_by_name(all_cols, merged_defs)
    col_block = "\n".join(f"- {c}: {merged_defs_by_col.get(c, '')}" for c in all_cols)
    prompt = textwrap.dedent(f"""\
        Multiple schema inference rounds produced these columns (some may be
        synonyms for the same concept):

        {col_block}

        Group columns that refer to EXACTLY the same data field. For each group,
        pick the most descriptive canonical name.

        {FUND_FIELD_DISAMBIGUATION_PROMPT}

        Output one line per group:
          canonical_name = synonym1, synonym2, ...
        For columns with no synonyms, output:
          column_name = (unique)""")

    messages = [
        ModelMessage(
            role="system", content="Merge synonym schema columns. Output mapping lines only."
        ),
        ModelMessage(role="user", content=prompt),
    ]
    try:
        response: ModelResponse = adapter.complete(messages)
    except Exception as exc:
        logger.warning("Schema synonym merge failed: %s", exc)
        return all_cols, merged_anchors, merged_types, merged_units, merged_defs

    content = response.content.strip()
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

    canonical_cols: list[str] = []
    alias_to_canon: dict[str, str] = {}
    for line in content.splitlines():
        line = line.strip().lstrip("- ")
        if "=" not in line:
            continue
        canon, _, rest = line.partition("=")
        canon = canon.strip()
        if not canon:
            continue
        canonical_cols.append(canon)
        for syn in rest.split(","):
            syn = syn.strip().strip("()")
            if syn.lower() != "unique" and syn.lower() != canon.lower():
                alias_to_canon[syn.lower()] = canon

    if not canonical_cols:
        return all_cols, merged_anchors, merged_types, merged_units, merged_defs

    final_anchors: list[str] = []
    final_anchor_lower: set[str] = set()
    for a in merged_anchors:
        resolved = alias_to_canon.get(a.lower(), a)
        resolved_lower = resolved.lower()
        if resolved_lower not in final_anchor_lower:
            final_anchors.append(resolved)
            final_anchor_lower.add(resolved_lower)

    final_types: dict[str, str] = {}
    final_units: dict[str, str] = {}
    final_defs: dict[str, str] = {}
    for old_key, val in merged_types.items():
        new_key = alias_to_canon.get(old_key.lower(), old_key)
        final_types.setdefault(new_key, val)
    for old_key, val in merged_units.items():
        new_key = alias_to_canon.get(old_key.lower(), old_key)
        final_units.setdefault(new_key, val)
    for old_key, val in merged_defs.items():
        new_key = alias_to_canon.get(old_key.lower(), old_key)
        final_defs.setdefault(new_key, val)

    logger.info(
        "Schema multi-round merge: %d union cols → %d canonical (%d synonyms resolved)",
        len(all_cols),
        len(canonical_cols),
        len(alias_to_canon),
    )
    return canonical_cols, final_anchors, final_types, final_units, final_defs


def infer_schema_from_sample(
    adapter: ModelAdapter,
    prose_text: str,
    question: str,
    table_name: str | None = None,
) -> tuple[list[str], list[str], dict[str, str], dict[str, str], dict[str, str]] | None:
    """Infer a unified schema by sampling the raw prose text before compression.

    Runs multiple inference rounds, unions all discovered columns, and merges
    synonyms via LLM adjudication to minimize field loss from single-run variance.

    Returns (columns, anchor_keys, field_types, field_units, field_defs) or None.
    """
    sample = sample_paragraphs_by_section(prose_text)
    if not sample or len(sample) < 50:
        return None

    rounds: list[tuple[list[str], list[str], dict[str, str], dict[str, str], dict[str, str]]] = []
    for _i in range(SCHEMA_SAMPLE_ROUNDS):
        result = _infer_schema_single(adapter, sample, question, table_name=table_name)
        if result:
            cols, anchors, types, units, defs = result
            rounds.append((cols, anchors, types, units, defs))

    if not rounds:
        return None

    columns, anchor_keys, field_types, field_units, field_defs = _merge_multi_round_schemas(
        adapter, rounds
    )

    columns, anchor_keys = _ensure_local_record_anchor(
        prose_text,
        columns,
        anchor_keys,
        field_types,
    )
    field_types = _complete_schema_field_types(columns, field_types)
    field_defs = _complete_schema_field_defs(columns, field_defs)

    logger.info(
        "Schema inferred from sample (%d rounds): pk=%s, anchors=%s, cols=%s",
        len(rounds),
        anchor_keys[0] if anchor_keys else None,
        anchor_keys,
        columns,
    )
    return columns, anchor_keys, field_types, field_units, field_defs
