"""ID conflict resolution, pre-merge, and scalar cell normalization helpers."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from agents.etl._constants import (
    APPROX_TAG,
    DATE_SCALAR,
    NUMERIC_SCALAR,
    TEMP_ANCHOR_PREFIX,
    UNIT_SUFFIX,
)
from agents.etl._types import (
    has_standalone_id,
    is_placeholder,
    normalize_boolean,
    split_approx_tag,
)

if TYPE_CHECKING:
    from agents.etl._record import EntityTable, Record

logger = logging.getLogger(__name__)


def strip_key(s: str) -> str:
    return s.strip().lower().replace("_", "").replace("-", "")


def canonical_key_map(keys: list[str]) -> dict[str, str]:
    """Map normalized field names to their canonical schema spelling."""
    key_map: dict[str, str] = {}
    for key in keys:
        normalized = strip_key(key)
        if normalized in key_map and key_map[normalized] != key:
            logger.warning(
                "ETL schema: columns %r and %r collide after key normalization (%r)",
                key_map[normalized],
                key,
                normalized,
            )
        key_map[normalized] = key
    return key_map


def canonicalize_key(key: str, key_map: dict[str, str]) -> str:
    return key_map.get(strip_key(key), key.strip())


def line_has_kv_key(line: str, key: str) -> bool:
    """Return whether a pipe-separated KV line contains ``key`` as a full field."""
    target = strip_key(key)
    for part in line.split("|"):
        raw_key, sep, _value = part.partition(":")
        if sep and strip_key(raw_key) == target:
            return True
    return False


def resolve_id_conflicts(table: EntityTable) -> int:
    """Detect and fix systematic ID-shift errors in compressed prose.

    Pattern: the prose defines entity names in the first section (identity).
    Later sections sometimes shift IDs by a fixed offset — e.g. entity "X"
    is ID 365 in section 1, but sections 2-5 assign ID 369 to "X" instead
    (a chain shift where every ID is off by the same amount).

    Strategy:
      1. Build the canonical name→ID mapping from the FIRST occurrence of each name.
      2. Scan all lines — if a line's (ID, name) pair disagrees with the canonical
         mapping AND the canonical ID exists for that name, rewrite the ID.
    """
    if table.primary_key:
        return 0
    _placeholder_set = {"-", "none", "nan", "placeholder", "- (placeholder)", ""}

    name_to_first_id: dict[str, int] = {}
    id_to_first_name: dict[int, str] = {}
    for record in table.records:
        raw_id = record.fields.get("ID", "").strip()
        name = record.fields.get("entity_name", "").strip()
        if not raw_id.isdigit():
            continue
        eid = int(raw_id)
        if name.lower() in _placeholder_set:
            continue
        name_to_first_id.setdefault(name, eid)
        id_to_first_name.setdefault(eid, name)

    fixed_count = 0
    for record in table.records:
        raw_id = record.fields.get("ID", "").strip()
        name = record.fields.get("entity_name", "").strip()
        if not raw_id.isdigit() or name.lower() in _placeholder_set:
            continue
        eid = int(raw_id)
        canonical_id = name_to_first_id.get(name)
        if canonical_id is not None and canonical_id != eid:
            record.fields["ID"] = str(canonical_id)
            record.pk = str(canonical_id)
            record.provenance["ID"] = "id_conflict"
            if fixed_count < 10:
                logger.info(
                    "ETL ID-fix: '%s' ID %d → %d",
                    name,
                    eid,
                    canonical_id,
                )
            fixed_count += 1
    if fixed_count:
        logger.info("ETL ID-fix: corrected %d lines total", fixed_count)
    return fixed_count


def _repair_fabricated_pks(
    parsed: list[tuple[str | None, list[tuple[str, str]]]],
    primary_key: str | None,
    anchor_keys: list[str] | None,
) -> list[tuple[str | None, list[tuple[str, str]]]]:
    """Rewrite sequentially fabricated primary keys from an alias column.

    When compression renumbers paragraphs (1, 2, 3, ...) instead of copying
    the stated record label, the true id usually still lands in a sibling
    identity column (e.g. "Strategic Unit 300" → strategic_unit_id).  Two
    deterministic signatures gate the repair:

    - alias column: pk == column on at least 3 lines and on more lines than
      it differs (true dual-id documents, e.g. record_id vs personalcode,
      agree on ~0 lines and never qualify);
    - fabricated run: the disagreeing pk values are integers forming a dense
      run starting at 1 (legitimate disagreeing id pairs such as secucode
      vs companycode are large sparse codes and never qualify).

    Lines whose pk falls inside the run get pk := alias value, so round-1
    grouping folds each fragment into its real entity row.
    """
    if not primary_key:
        return parsed

    # Candidate columns: declared anchors first, then remaining columns in
    # first-seen order — the alias column is not always a declared anchor.
    primary_lower = primary_key.lower()
    candidates: list[str] = [a for a in (anchor_keys or []) if a.lower() != primary_lower]
    for _, pairs in parsed:
        for key, _val in pairs:
            if key.lower() != primary_lower and key not in candidates:
                candidates.append(key)

    for column in candidates:
        agree = 0
        disagree_pks: list[str] = []
        for pk_val, pairs in parsed:
            if not pk_val or is_placeholder(pk_val):
                continue
            col_val = next((v for k, v in pairs if k == column and v.strip()), "")
            if not col_val or is_placeholder(col_val):
                continue
            if pk_val == col_val:
                agree += 1
            else:
                disagree_pks.append(pk_val)
        if agree < 3 or len(disagree_pks) >= agree or not disagree_pks:
            continue
        if not all(pk.isdigit() for pk in disagree_pks):
            continue
        run = {int(pk) for pk in disagree_pks}
        if min(run) != 1 or max(run) > len(run) + max(2, len(run) // 5):
            continue

        repaired = 0
        out: list[tuple[str | None, list[tuple[str, str]]]] = []
        for pk_val, pairs in parsed:
            col_val = next((v for k, v in pairs if k == column and v.strip()), "")
            if (
                pk_val
                and pk_val.isdigit()
                and int(pk_val) in run
                and col_val
                and not is_placeholder(col_val)
                and col_val != pk_val
            ):
                pairs = [(k, col_val if k == primary_key else v) for k, v in pairs]
                pk_val = col_val
                repaired += 1
            out.append((pk_val, pairs))
        logger.info(
            "ETL pre-merge: rewrote %d fabricated sequential PKs (run 1..%d) from alias column %r",
            repaired,
            max(run),
            column,
        )
        return out
    return parsed


def _drop_unanchored_pks(
    parsed: list[tuple[str | None, list[tuple[str, str]]]],
    primary_key: str | None,
    source_text: str | None,
) -> list[tuple[str | None, list[tuple[str, str]]]]:
    """Drop lines whose digit primary key never anchors in the source text.

    Every real record label is stated in the prose by construction; a digit
    pk that appears nowhere as a standalone token (``has_standalone_id``) is
    fabricated — typically a hallucinated line mixing values from several
    entities — so its data attribution cannot be trusted and the whole line
    is dropped. This differs from blank-PK fragments (an honest "unknown",
    handled by the round-3 echo sweep): a fabricated pk also poisons round-2
    anchor uniqueness, blocking legitimate fragments from merging home.

    Non-digit pks and lines without a pk are left alone; no-op without
    ``source_text``.
    """
    if not primary_key or not source_text:
        return parsed
    kept: list[tuple[str | None, list[tuple[str, str]]]] = []
    dropped: list[str] = []
    for pk_val, pairs in parsed:
        if pk_val and pk_val.isdigit() and not has_standalone_id(source_text, pk_val):
            dropped.append(pk_val)
            continue
        kept.append((pk_val, pairs))
    if dropped:
        logger.info(
            "ETL pre-merge: dropped %d lines with source-unanchored pks: %s",
            len(dropped),
            dropped[:10],
        )
    return kept


def _record_pairs_for_merge(record: Record) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for key, val in record.fields.items():
        if val and key in record.approx:
            val = f"{APPROX_TAG}{val}"
        pairs.append((key, val))
    return pairs


def _group_key_from_pairs(
    pk_val: str | None,
    pairs: list[tuple[str, str]],
    anchor_keys: list[str] | None,
) -> str | None:
    """Pick a stable grouping key for one parsed record."""
    if pk_val is not None and not is_placeholder(pk_val):
        return pk_val
    if anchor_keys and len(anchor_keys) > 1:
        pair_map = dict(pairs)
        for anchor in anchor_keys[1:]:
            anchor_val, _approx = split_approx_tag(pair_map.get(anchor, ""))
            anchor_val = anchor_val.strip()
            if anchor_val and not is_placeholder(anchor_val):
                return f"{TEMP_ANCHOR_PREFIX}{anchor}:{anchor_val}"
    return None


def merge_records(
    table: EntityTable,
    source_text: str | None = None,
) -> None:
    """Merge multi-line entity records into one row per unique entity.

    Record-oriented port of the old ``pre_merge_entity_lines`` logic:

    - repair fabricated sequential pks from an alias column;
    - drop digit pks that never appear as standalone source anchors;
    - group by primary key;
    - attach blank-pk fragments via secondary anchors;
    - drop identity-echo fragments that carry no data.

    Mutates ``table.records`` in place.
    """
    from agents.etl._record import Record, set_record_field

    primary_key = table.primary_key
    anchor_keys = table.anchor_keys
    if len(table.records) <= 1:
        return

    parsed = [(record.pk, _record_pairs_for_merge(record)) for record in table.records]
    parsed = _repair_fabricated_pks(parsed, primary_key, anchor_keys)
    parsed = _drop_unanchored_pks(parsed, primary_key, source_text)

    groups: dict[str, Record] = {}
    order: list[str] = []
    dropped_unkeyed = 0

    for pk_val, pairs in parsed:
        group_key = _group_key_from_pairs(pk_val, pairs, anchor_keys)
        if group_key is None:
            dropped_unkeyed += 1
            continue
        if group_key not in groups:
            groups[group_key] = Record()
            order.append(group_key)
        record = groups[group_key]
        for key, raw_val in pairs:
            val, _approx = split_approx_tag(raw_val)
            val = val.strip()
            existing = record.fields.get(key)
            if existing is None or (is_placeholder(existing) and not is_placeholder(val)):
                set_record_field(record, key, raw_val.strip(), "pre_merge")
            elif (
                val
                and existing
                and not is_placeholder(existing)
                and not is_placeholder(val)
                and existing != val
            ):
                table.conflicts += 1
        if primary_key:
            record.pk = record.fields.get(primary_key) or None
        elif "ID" in record.fields:
            record.pk = record.fields.get("ID") or None

    if anchor_keys and len(anchor_keys) > 1:
        groups_before_r2 = len(groups)
        for anchor in anchor_keys[1:]:
            anchor_to_pks: dict[str, list[str]] = {}
            for pk_val in list(groups.keys()):
                a_val = groups[pk_val].fields.get(anchor, "").strip()
                if a_val and not is_placeholder(a_val):
                    anchor_to_pks.setdefault(a_val, []).append(pk_val)

            for _a_val, pk_list in anchor_to_pks.items():
                if len(pk_list) <= 1:
                    continue
                temp_pks = [pk for pk in pk_list if pk.startswith(TEMP_ANCHOR_PREFIX)]
                real_pks = [pk for pk in pk_list if not pk.startswith(TEMP_ANCHOR_PREFIX)]
                if not temp_pks or len(real_pks) != 1:
                    continue

                canonical = real_pks[0]
                for pk in temp_pks:
                    conflict = False
                    for anchor_name in anchor_keys[1:]:
                        t_val = groups[pk].fields.get(anchor_name, "").strip()
                        c_val = groups[canonical].fields.get(anchor_name, "").strip()
                        if (
                            t_val
                            and c_val
                            and not is_placeholder(t_val)
                            and not is_placeholder(c_val)
                            and t_val != c_val
                        ):
                            conflict = True
                            break
                    if conflict:
                        continue
                    for key, val in groups[pk].fields.items():
                        current = groups[canonical].fields.get(key)
                        if (
                            (current is None or is_placeholder(current))
                            and val
                            and not is_placeholder(val)
                        ):
                            if key in groups[pk].approx:
                                val = f"{APPROX_TAG}{val}"
                            set_record_field(groups[canonical], key, val, "pre_merge")
                    del groups[pk]

        groups_after_r2 = len(groups)
        if groups_before_r2 != groups_after_r2:
            logger.info(
                "ETL anchor-merge: %d groups → %d unique entities (anchors: %s)",
                groups_before_r2,
                groups_after_r2,
                anchor_keys,
            )

    real_keys = [key for key in order if key in groups and not key.startswith(TEMP_ANCHOR_PREFIX)]
    temp_keys = [key for key in order if key in groups and key.startswith(TEMP_ANCHOR_PREFIX)]
    if temp_keys and real_keys:
        identity_fields = {key.lower() for key in [primary_key, *anchor_keys] if key}
        real_field_values: dict[str, set[str]] = {}
        for key in real_keys:
            for field, val in groups[key].fields.items():
                val = val.strip()
                if val and not is_placeholder(val):
                    real_field_values.setdefault(field, set()).add(val)

        echo_dropped = 0
        for key in temp_keys:
            informative: list[str] = []
            for field, val in groups[key].fields.items():
                val = val.strip()
                if not val or is_placeholder(val):
                    continue
                if field.lower() in identity_fields:
                    continue
                real_vals = real_field_values.get(field)
                is_constant_echo = (
                    len(real_keys) >= 2 and real_vals is not None and real_vals == {val}
                )
                if not is_constant_echo:
                    informative.append(field)
            if informative:
                logger.warning(
                    "ETL pre-merge: unplaced fragment kept as row — anchors ambiguous, "
                    "data fields: %s",
                    informative,
                )
            else:
                del groups[key]
                echo_dropped += 1
        if echo_dropped:
            logger.info(
                "ETL pre-merge: dropped %d identity-echo fragments with no data fields",
                echo_dropped,
            )

    merged: list[Record] = []
    for key in order:
        record = groups.get(key)
        if record is None:
            continue
        if primary_key:
            record.pk = record.fields.get(primary_key) or None
        elif "ID" in record.fields:
            record.pk = record.fields.get("ID") or None
        merged.append(record)

    before = len(table.records)
    after = len(merged)
    table.records = merged
    if before != after:
        logger.info(
            "ETL pre-merge: %d entity records → %d unique entities",
            before,
            after,
        )
    if dropped_unkeyed:
        logger.info("ETL pre-merge: dropped %d records without a grouping key", dropped_unkeyed)


def clean_cell(val: str) -> str:
    """Normalize a single KV value into a CSV-ready cell."""
    if not val or is_placeholder(val):
        return ""
    m = UNIT_SUFFIX.match(val)
    if m:
        return m.group(1)
    return val


_REASONING_LEAK = re.compile(
    r"\s*\(?"
    r"(?:"
    # English patterns
    r"Source text|However,|Wait,|Let me|Let's|Note:|The (?:text|context|extraction|source)"
    r"|strictly following|implies the|this (?:is|means|suggests)"
    # Chinese patterns
    r"|或根据|原文指代|原文(?:是|写|说|提到)|通常指|然而，|但仔细|但是，"
    r"|如果源文本|这看起来|根据文档|根据上下文|仔细阅读"
    r")"
    r".*",
    re.IGNORECASE | re.DOTALL,
)


_FUND_TYPE_SUFFIX = re.compile(r"^(股票|混合|债券|货币|其他)型基金$")


def _normalize_fund_type(val: str, field_name: str) -> str:
    """Normalize '混合型基金' → '混合型' for fund classification fields."""
    if field_name.lower() not in {"fundtype", "fund_type", "fundtypename"}:
        return val
    m = _FUND_TYPE_SUFFIX.match(val.strip())
    if m:
        return m.group(1) + "型"
    return val


def strip_reasoning_leak(val: str) -> str:
    """Truncate LLM reasoning that leaked into a field value."""
    m = _REASONING_LEAK.search(val)
    if m and m.start() > 0:
        return val[: m.start()].rstrip(" ,;|")
    return val


_NAME_ABBR_FIELDS = frozenset(
    {
        "secuabbr",
        "chiname",
        "chinameabbr",
        "chinamabbr",
        "fund_name",
        "fund_name_short",
        "name",
        "abbr",
        "companyname",
        "abbrchiname",
    }
)


def clean_cell_for_type(
    val: str,
    field_type: str | None = None,
    field_name: str | None = None,
) -> str:
    """Normalize a KV value and enforce lightweight schema field types.

    A leading approximate-value marker (``~6970000``, from the compression
    FUZZY NUMBERS rule) survives on numeric cells so the identity-repair pass
    can recompute the exact value; it is dropped from date/boolean cells.
    """
    bare, approx = split_approx_tag(val)
    cleaned = clean_cell(bare)
    if not cleaned:
        return ""
    cleaned = strip_reasoning_leak(cleaned)
    if field_name:
        cleaned = _normalize_fund_type(cleaned, field_name)
    if field_name and field_name.lower() in _NAME_ABBR_FIELDS:
        stripped_digits = cleaned.replace(",", "").replace(".", "").lstrip("-")
        if stripped_digits.isdigit():
            return ""
    type_text = (field_type or "").lower()
    if ("number" in type_text or "integer" in type_text) and "rank" not in type_text:
        normalized = cleaned.replace(",", "")
        if normalized.endswith("%"):
            normalized = normalized[:-1].strip()
        if not NUMERIC_SCALAR.match(normalized):
            return ""
        return f"{APPROX_TAG}{normalized}" if approx else normalized
    if "date" in type_text:
        return cleaned if DATE_SCALAR.match(cleaned) else ""
    if "boolean" in type_text:
        return normalize_boolean(cleaned) or ""
    normalized = cleaned.replace(",", "")
    if approx and NUMERIC_SCALAR.match(normalized):
        return f"{APPROX_TAG}{normalized}"
    if normalized != cleaned and NUMERIC_SCALAR.match(normalized):
        return normalized
    return clean_cell(val)


def cells_agree(a: str, b: str) -> bool:
    """Two non-empty cells agree when equal as strings or as parsed numbers."""
    if a == b:
        return True
    try:
        return float(a) == float(b)
    except ValueError:
        return False


def column_fill_counts(header: list[str], rows: list[list[str]]) -> dict[str, int]:
    """Count non-empty cells per column."""
    counts = dict.fromkeys(header, 0)
    for row in rows:
        for col, val in zip(header, row, strict=False):
            if val.strip():
                counts[col] += 1
    return counts


def apply_synonym_merges(
    header: list[str],
    rows: list[list[str]],
    merges: list[tuple[str, str]],
    protected: set[str] | None = None,
) -> tuple[list[str], list[list[str]], list[tuple[str, str]]]:
    """Fold synonym source columns into their canonical target columns.

    ``merges`` holds adjudicated ``(source, target)`` pairs — typically a
    populated extraction-discovered column folding into an empty
    governance-canonical column (the value-split failure mode).  The
    adjudication says the two columns *mean* the same thing; this function
    re-checks, per pair and against the CURRENT table state, that the data
    *agrees* with that claim:

    - identity columns named in ``protected`` are never touched, except for
      folding two columns whose names differ only by case;
    - a pair with any row-level conflict (both cells non-empty and not equal
      as strings or numbers) is skipped — semantically-approved but
      data-contradicted merges must not corrupt either column;
    - an all-empty source is a no-op and is skipped.

    Pairs are processed in deterministic order (descending source fill, then
    name), so when two sources target one canonical column the second is
    conflict-checked against the already-merged result.

    Returns ``(header, rows, applied)`` — new copies; inputs are not mutated.
    """
    protected_lower = {p.lower() for p in (protected or set())}
    header = list(header)
    rows = [list(row) for row in rows]
    fill = column_fill_counts(header, rows)

    applied: list[tuple[str, str]] = []
    for src, dst in sorted(merges, key=lambda m: (-fill.get(m[0], 0), m[0])):
        case_only = src.lower() == dst.lower()
        if src == dst or (
            not case_only and (src.lower() in protected_lower or dst.lower() in protected_lower)
        ):
            continue
        if src not in header or dst not in header:
            continue
        si, di = header.index(src), header.index(dst)
        if not any(row[si].strip() for row in rows):
            continue
        conflicts = sum(
            1
            for row in rows
            if row[si].strip()
            and row[di].strip()
            and not cells_agree(row[si].strip(), row[di].strip())
        )
        if conflicts:
            logger.warning(
                "ETL synonym-unify: %d row conflicts between '%s' and '%s' — not merged",
                conflicts,
                src,
                dst,
            )
            continue
        for row in rows:
            if row[si].strip() and not row[di].strip():
                row[di] = row[si]
        header.pop(si)
        for row in rows:
            row.pop(si)
        applied.append((src, dst))
    return header, rows, applied
