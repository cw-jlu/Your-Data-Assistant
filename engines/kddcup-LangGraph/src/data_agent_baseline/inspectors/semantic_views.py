from __future__ import annotations

from typing import Any

from data_agent_baseline.config import DataInspectorSemanticViewConfig


_TECHNICAL_FIELD_NAMES = {"id"}


def _source_key(ref: dict[str, Any]) -> tuple[str, str | None]:
    return (str(ref.get("asset_path", "")), ref.get("table"))


def _field_names(ref: dict[str, Any]) -> list[str]:
    return [str(field) for field in ref.get("fields", [])]


def _optional_int(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _field_type_family(field_type: str) -> str:
    normalized = field_type.lower()
    if any(token in normalized for token in ("date", "time")):
        return "temporal"
    if "int" in normalized:
        return "integer"
    if normalized in {"integer", "number"} or any(
        token in normalized for token in ("real", "float", "double", "numeric", "decimal")
    ):
        return "number"
    if any(token in normalized for token in ("char", "clob", "text", "string", "varchar")):
        return "string"
    return "unknown"


def _is_all_missing(column: dict[str, Any], row_count: Any) -> bool:
    missing = column.get("missing_count")
    if missing is None or row_count is None:
        return False
    try:
        return int(missing) >= int(row_count)
    except (TypeError, ValueError):
        return False


def _cardinality_ratio(column: dict[str, Any], row_count: Any) -> float | None:
    cardinality = _optional_int(column.get("cardinality"))
    rows = _optional_int(row_count)
    if cardinality is None or not rows:
        return None
    return cardinality / max(rows, 1)


def _missing_ratio(column: dict[str, Any], row_count: Any) -> float:
    missing = _optional_int(column.get("missing_count"))
    rows = _optional_int(row_count)
    if missing is None or not rows:
        return 0.0
    return min(max(missing / max(rows, 1), 0.0), 1.0)


def _column_identifiers(column: dict[str, Any]) -> set[str]:
    identifiers = {str(column.get("name", "")).lower()}
    json_path = column.get("json_path")
    if json_path:
        identifiers.add(str(json_path).lower())
    return {identifier for identifier in identifiers if identifier}


def _resolve_column_name(logical_table: dict[str, Any], field: str) -> str | None:
    normalized = str(field).lower()
    if not normalized:
        return None
    for column in logical_table.get("columns", []):
        name = str(column.get("name", ""))
        if name and normalized in _column_identifiers(column):
            return name
    return None


def _resolve_join_fields(
    logical_table: dict[str, Any],
    fields: list[str],
) -> list[str] | None:
    resolved: list[str] = []
    for field in fields:
        column_name = _resolve_column_name(logical_table, field)
        if column_name is None:
            return None
        resolved.append(column_name)
    return resolved


def _is_high_cardinality_numeric(column: dict[str, Any], row_count: Any) -> bool:
    cardinality = _optional_int(column.get("cardinality"))
    if cardinality is None:
        return True
    ratio = _cardinality_ratio(column, row_count)
    if ratio is None:
        return cardinality > 50
    return cardinality > 50 and ratio > 0.30


def _payload_column_score(column: dict[str, Any], row_count: Any) -> float:
    if _is_all_missing(column, row_count):
        return 0.0

    family = _field_type_family(str(column.get("type", "unknown")))
    missing_penalty = _missing_ratio(column, row_count) * 50
    cardinality = _optional_int(column.get("cardinality"))
    ratio = _cardinality_ratio(column, row_count)

    if family == "string":
        low_cardinality_bonus = 30 if ratio is not None and ratio <= 0.30 else 0
        return 300 + low_cardinality_bonus - missing_penalty
    if family == "temporal":
        return 35 - missing_penalty
    if family in {"integer", "number"}:
        if _is_high_cardinality_numeric(column, row_count):
            return 0.0
        low_cardinality_bonus = 40 if ratio is not None and ratio <= 0.20 else 0
        observed_cardinality_bonus = 10 if cardinality is not None and cardinality <= 20 else 0
        return 140 + low_cardinality_bonus + observed_cardinality_bonus - missing_penalty
    if cardinality is not None and (ratio is None or ratio <= 0.30):
        return 80 - missing_penalty
    return 0.0


def score_lookup_payload(
    *,
    logical_table: dict[str, Any],
    join_keys: set[str],
    base_columns: set[str],
) -> list[tuple[float, int, dict[str, Any]]]:
    candidates: list[tuple[float, int, dict[str, Any]]] = []
    row_count = logical_table.get("row_count")
    join_key_lower = {field.lower() for field in join_keys}
    base_lower = {field.lower() for field in base_columns}

    for index, column in enumerate(logical_table.get("columns", [])):
        name = str(column.get("name", ""))
        lowered = name.lower()
        if not name or lowered in _TECHNICAL_FIELD_NAMES:
            continue
        if _column_identifiers(column) & join_key_lower:
            continue
        if lowered in base_lower:
            continue
        score = _payload_column_score(column, row_count)
        if score <= 0:
            continue
        candidates.append((score, -index, column))

    candidates.sort(key=lambda item: (-item[0], -item[1]))
    return candidates


def select_lookup_payload_columns(
    *,
    logical_table: dict[str, Any],
    join_keys: set[str],
    base_column_names: set[str],
    max_fields: int,
) -> list[dict[str, Any]]:
    if max_fields <= 0:
        return []
    candidates = score_lookup_payload(
        logical_table=logical_table,
        join_keys=join_keys,
        base_columns=base_column_names,
    )
    return [item[2] for item in candidates[:max_fields]]


def relationship_is_row_preserving(
    rel: dict[str, Any],
    config: DataInspectorSemanticViewConfig,
) -> bool:
    cardinality = str(rel.get("cardinality", ""))
    allowed_cardinalities = {"many_to_one"}
    if config.allow_one_to_one_enrichment:
        allowed_cardinalities.add("one_to_one")
    if cardinality not in allowed_cardinalities:
        return False

    confidence = _optional_float(rel.get("confidence"))
    if confidence < config.min_confidence:
        return False

    evidence = rel.get("evidence", {})
    distinct_ratio = _optional_float(evidence.get("matched_source_distinct_ratio"))
    target_uniqueness = _optional_float(evidence.get("target_uniqueness_ratio"))
    if distinct_ratio < config.min_distinct_match_ratio:
        return False
    return target_uniqueness + 1e-9 >= config.min_target_uniqueness_ratio


def score_enrichment_candidate(
    rel: dict[str, Any],
    source_logical: dict[str, Any],
    target_logical: dict[str, Any],
    payload_columns: list[dict[str, Any]],
) -> float:
    del source_logical, target_logical
    evidence = rel.get("evidence", {})
    score = _optional_float(rel.get("confidence")) * 1000
    score += _optional_float(evidence.get("matched_source_distinct_ratio")) * 160
    score += _optional_float(evidence.get("matched_source_row_ratio")) * 80
    score += _optional_float(evidence.get("target_uniqueness_ratio")) * 120
    score += min(len(payload_columns), 12) * 12
    if rel.get("cardinality") == "one_to_one":
        score -= 120
    return score


def _unique_dimension_column_name(
    field_name: str,
    *,
    used_names: set[str],
    dimension_table: str,
) -> str:
    if field_name not in used_names:
        used_names.add(field_name)
        return field_name
    prefix = dimension_table[:24].strip("_") or "dimension"
    candidate = f"{prefix}_{field_name}"
    suffix = 2
    while candidate in used_names:
        candidate = f"{prefix}_{field_name}_{suffix}"
        suffix += 1
    used_names.add(candidate)
    return candidate


def _best_row_preserving_lookups_by_base(
    relationships: list[dict[str, Any]],
    *,
    logical_by_source: dict[tuple[str, str | None], dict[str, Any]],
    config: DataInspectorSemanticViewConfig,
) -> dict[tuple[str, str | None], list[tuple[dict[str, Any], list[dict[str, Any]]]]]:
    grouped: dict[
        tuple[str, str | None],
        dict[tuple[str, str | None], tuple[float, dict[str, Any], list[dict[str, Any]]]],
    ] = {}
    for rel in relationships:
        if not relationship_is_row_preserving(rel, config):
            continue
        source = rel.get("source", {})
        target = rel.get("target", {})
        source_key = _source_key(source)
        target_key = _source_key(target)
        source_logical = logical_by_source.get(source_key)
        target_logical = logical_by_source.get(target_key)
        if source_logical is None or target_logical is None:
            continue
        if source_key == target_key:
            continue

        payload_columns = select_lookup_payload_columns(
            logical_table=target_logical,
            join_keys=set(_field_names(target)),
            base_column_names={str(column.get("name", "")) for column in source_logical.get("columns", [])},
            max_fields=config.max_dimension_fields_per_view,
        )
        if len(payload_columns) < config.min_payload_fields:
            continue
        score = score_enrichment_candidate(rel, source_logical, target_logical, payload_columns)

        by_target = grouped.setdefault(source_key, {})
        current = by_target.get(target_key)
        if current is None:
            by_target[target_key] = (score, rel, payload_columns)
            continue
        if score > current[0]:
            by_target[target_key] = (score, rel, payload_columns)

    result: dict[tuple[str, str | None], list[tuple[dict[str, Any], list[dict[str, Any]]]]] = {}
    for source_key, by_target in grouped.items():
        ranked = sorted(by_target.values(), key=lambda item: -item[0])
        result[source_key] = [
            (rel, payload_columns)
            for _, rel, payload_columns in ranked[: config.max_dimensions_per_view]
        ]
    return result


def build_derived_views(
    catalog: dict[str, Any],
    *,
    logical_tables: list[dict[str, Any]],
    config: DataInspectorSemanticViewConfig,
) -> list[dict[str, Any]]:
    if not config.enabled or config.max_views <= 0 or config.max_dimensions_per_view <= 0:
        return []

    logical_by_source = {
        (str(table.get("source_asset_path", "")), table.get("source_table")): table
        for table in logical_tables
    }
    relationships_by_base = _best_row_preserving_lookups_by_base(
        catalog.get("relationships", []),
        logical_by_source=logical_by_source,
        config=config,
    )

    views: list[dict[str, Any]] = []
    for base_key, relationships in sorted(
        relationships_by_base.items(),
        key=lambda item: str(logical_by_source.get(item[0], {}).get("table", "")),
    ):
        if len(views) >= config.max_views:
            break
        base_logical = logical_by_source.get(base_key)
        if base_logical is None:
            continue
        base_table = str(base_logical.get("table", ""))
        if not base_table:
            continue
        view_name = f"v_{base_table}_enriched"
        if not view_name.startswith("v_") or not view_name.endswith("_enriched"):
            continue

        used_names = {str(column.get("name", "")) for column in base_logical.get("columns", [])}
        columns: list[dict[str, Any]] = [
            {
                "name": str(column.get("name", "")),
                "type": str(column.get("type", "unknown")),
                "source_table": base_table,
                "source_field": str(column.get("name", "")),
                "role": "base",
            }
            for column in base_logical.get("columns", [])
            if column.get("name")
        ]
        joins: list[dict[str, Any]] = []
        warnings: list[str] = []

        for rel, selected_columns in relationships:
            target = rel.get("target", {})
            target_key = _source_key(target)
            dimension_logical = logical_by_source.get(target_key)
            if dimension_logical is None:
                continue
            dimension_table = str(dimension_logical.get("table", ""))
            source_fields = _field_names(rel.get("source", {}))
            target_fields = _field_names(target)
            resolved_source_fields = _resolve_join_fields(base_logical, source_fields)
            resolved_target_fields = _resolve_join_fields(dimension_logical, target_fields)
            if resolved_source_fields is None or resolved_target_fields is None:
                warnings.append("join_field_not_found_in_logical_table")
                continue
            evidence = rel.get("evidence", {})
            distinct_ratio = float(evidence.get("matched_source_distinct_ratio", 0) or 0)
            if distinct_ratio < config.strict_distinct_match_ratio:
                warnings.append("dimension_match_below_strict_threshold")
            if not selected_columns:
                continue
            joins.append(
                {
                    "join_type": "left",
                    "dimension_table": dimension_table,
                    "dimension_source": {
                        "asset_path": target_key[0],
                        "table": target_key[1],
                    },
                    "source_fields": resolved_source_fields,
                    "target_fields": resolved_target_fields,
                    "confidence": rel.get("confidence"),
                    "matched_source_distinct_ratio": evidence.get("matched_source_distinct_ratio"),
                    "matched_source_row_ratio": evidence.get("matched_source_row_ratio"),
                    "target_uniqueness_ratio": evidence.get("target_uniqueness_ratio"),
                }
            )
            for column in selected_columns:
                source_field = str(column.get("name", ""))
                column_name = _unique_dimension_column_name(
                    source_field,
                    used_names=used_names,
                    dimension_table=dimension_table,
                )
                columns.append(
                    {
                        "name": column_name,
                        "type": str(column.get("type", "unknown")),
                        "source_table": dimension_table,
                        "source_field": source_field,
                        "role": "dimension",
                    }
                )

        if not joins:
            continue
        views.append(
            {
                "name": view_name,
                "kind": "derived_view",
                "base_table": base_table,
                "base_source": {
                    "asset_path": base_key[0],
                    "table": base_key[1],
                },
                "grain": "same_as_base_table",
                "is_original_table": False,
                "knowledge_authority": "source_fields_only",
                "description": (
                    f"{base_table} records enriched with high-confidence lookup dimensions. "
                    "This is not an original table from knowledge.md."
                ),
                "row_count": base_logical.get("row_count"),
                "columns": columns,
                "joins": joins,
                "warnings": sorted(set(warnings)),
            }
        )
    return views


def semantic_view_names(catalog: dict[str, Any]) -> set[str]:
    return {str(view.get("name", "")) for view in catalog.get("derived_views", []) if view.get("name")}


def find_semantic_view(catalog: dict[str, Any], table_name: str) -> dict[str, Any] | None:
    normalized = table_name.strip().strip('"')
    lowered = normalized.lower()
    for view in catalog.get("derived_views", []):
        name = str(view.get("name", ""))
        if name == normalized or name.lower() == lowered:
            return view
    return None
