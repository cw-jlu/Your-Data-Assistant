"""Synthesize a candidate pandas/SQL filter recipe from explore_video's rules dict.

The video sub-agent's ``findings.extracted_data.rules`` arrives in heterogeneous
shapes across runs — keys observed in production include ``filter_field``,
``threshold_value``, ``ranking_limit``, ``output_fields``, ``aggregation_field``,
``data_source_table``, and many Chinese-suffixed variants. Each run the agent
re-writes the same filter logic from scratch in ``execute_python`` /
``execute_context_sql`` and frequently mistranslates:

- threshold boundary inclusion (``>`` vs ``>=``)
- column choice (e.g. ``ChiName`` vs ``ChiNameAbbr``)
- forgetting top-N, group-by, batch_id, snapshot multiplicity

This module performs a deterministic pre-translation so the downstream agent
starts from a structured recipe instead of free-form rewriting. Output is a
plaintext hint attached to the explore_video observation; agents are explicitly
told to verify exact column names against the actual table schema before use.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, cast

# 视频抽出的 rules 字段在不同 run 里键名漂得很厉害,用别名 tuple 兜底。
# 排在前面的优先;一旦命中即停。
_FIELD_KEYS: tuple[str, ...] = (
    "field",
    "filter_field",
    "metric",
    "column",
    "attribute",
    "aggregation_field",
)
_FIELD_LABEL_KEYS: tuple[str, ...] = ("field_label", "field_cn", "label", "field_name_cn")
_OPERATOR_KEYS: tuple[str, ...] = ("operator", "comparator", "op", "condition_type")
_THRESHOLD_KEYS: tuple[str, ...] = (
    "threshold",
    "threshold_value",
    "value",
    "condition_value",
)
_THRESHOLD_TEXT_KEYS: tuple[str, ...] = ("threshold_text", "value_text")
_UNIT_KEYS: tuple[str, ...] = ("unit", "threshold_unit", "value_unit")
_SORT_FIELD_KEYS: tuple[str, ...] = (
    "sort_field",
    "sort_by",
    "order_by",
    "ranking_field",
)
_TOPN_SORT_FALLBACK_KEYS: tuple[str, ...] = ("aggregation_field", "metric")
_SORT_DIR_KEYS: tuple[str, ...] = (
    "sort_direction",
    "sort_order",
    "order",
    "ranking_order",
    "sort_cn",
)
_TOPN_KEYS: tuple[str, ...] = ("top_n", "ranking_limit", "limit", "n", "top")
_EXPORT_KEYS: tuple[str, ...] = (
    "export_field",
    "export_fields",
    "output_field",
    "output_fields",
    "select",
    "fields_to_export",
    "fields_for_result",
    "result_fields",
    "fields",
)
_GROUPBY_KEYS: tuple[str, ...] = ("group_by", "grouping_dimension", "groupby", "group")
_TABLE_KEYS: tuple[str, ...] = (
    "table",
    "source_table",
    "data_source_table",
    "source_tables",
)
_BATCH_KEYS: tuple[str, ...] = (
    "batch_id",
    "batch_no",
    "archive_id",
    "snapshot_id",
    "scheme_id",
)
_SNAPSHOT_KEYS: tuple[str, ...] = ("snapshot_dates", "reporting_dates", "quarter_ends")
_CANONICAL_RULE_KEYS: set[str] = {
    "filters",
    "logic",
    "metrics",
    "group_by",
    "sort_by",
    "top_n",
    "periods",
    "identifiers",
    "tables",
    "output_fields",
    "snapshot_dates",
}

# Operator 归一化:把英文/中文/符号都映射到 pandas/SQL 表达式里能直接拼的符号串。
# 顺序无关;映射目标永远是 Python 比较运算符。
_OPERATOR_NORMALIZATION: dict[str, str] = {
    ">": ">",
    "greater than": ">",
    "above": ">",
    "exceeds": ">",
    "more than": ">",
    "高于": ">",
    "大于": ">",
    "≥": ">=",
    ">=": ">=",
    "at or above": ">=",
    "no less than": ">=",
    "not less than": ">=",
    "at least": ">=",
    "gte": ">=",
    "大于等于": ">=",
    "不低于": ">=",
    "不少于": ">=",
    "<": "<",
    "less than": "<",
    "below": "<",
    "under": "<",
    "fewer than": "<",
    "低于": "<",
    "小于": "<",
    "≤": "<=",
    "<=": "<=",
    "at or below": "<=",
    "no more than": "<=",
    "not more than": "<=",
    "at most": "<=",
    "lte": "<=",
    "小于等于": "<=",
    "不超过": "<=",
    "不多于": "<=",
    "=": "==",
    "==": "==",
    "equal to": "==",
    "equals": "==",
    "等于": "==",
    "!=": "!=",
    "≠": "!=",
    "not equal to": "!=",
    "not equals": "!=",
    "in": "in",
    "between": "between",
    "contains": "contains",
    "不等于": "!=",
}

# 排序方向:大小写不敏感的子串匹配。
_DESCENDING_TOKENS: tuple[str, ...] = (
    "desc",
    "descending",
    "降序",
    "由高到低",
    "highest first",
    "high to low",
)
_ASCENDING_TOKENS: tuple[str, ...] = (
    "asc",
    "ascending",
    "升序",
    "由低到高",
    "lowest first",
    "low to high",
)

# 视频里的字段名常带括号注释:"流通A股股本 (AFLOATS)" / "本日基准增长率 (dailybenchgr)"。
# 括号里的 ASCII 标识符通常就是源表里的列名 — 优先用它。
_PAREN_IDENT_RE = re.compile(r"\(([A-Za-z_][A-Za-z0-9_]*)\)")
# 不带 ASCII 标识符的情况下,先 trim 括号注释再返回原文。
_PAREN_ANY_RE = re.compile(r"\s*\([^)]*\)\s*$")


@dataclass(frozen=True, slots=True)
class RecipeFilter:
    """One executable filter condition from rules.canonical or legacy rules."""

    field: str | None = None
    field_label: str | None = None
    operator: str | None = None
    threshold: Any | None = None
    threshold_unit: str | None = None
    source_rule_id: str | None = None
    note: str | None = None


@dataclass(frozen=True, slots=True)
class RecipeSort:
    """One sort key from rules.canonical or legacy rules."""

    field: str | None = None
    descending: bool | None = None
    source_rule_id: str | None = None
    note: str | None = None


@dataclass(frozen=True, slots=True)
class FilterRecipe:
    """Canonical filter intent extracted from heterogeneous video rules."""

    field: str | None = None
    field_label: str | None = None  # Chinese label retained for context
    operator: str | None = None  # normalized comparator when known
    threshold: Any | None = None  # raw value; agent decides dtype coercion
    threshold_unit: str | None = None
    sort_field: str | None = None
    sort_descending: bool | None = None
    top_n: int | None = None
    export_fields: tuple[str, ...] = ()
    group_by: str | None = None
    table_hint: str | None = None
    batch_id: str | None = None
    snapshot_dates: tuple[str, ...] = ()
    filters: tuple[RecipeFilter, ...] = ()
    logic: str | None = None
    sorts: tuple[RecipeSort, ...] = ()
    group_by_fields: tuple[str, ...] = ()
    table_hints: tuple[str, ...] = ()

    def is_sufficient(self) -> bool:
        """True when the recipe has enough structure to be worth surfacing.

        Two anchor patterns count:
        1. A filter pair (``field`` + ``operator``) - most common rule shape.
        2. A ranking pair (``sort_field`` + ``top_n``) - covers "Top N by X" videos.
        Plus a fallback: any of group_by / batch_id / snapshot_dates is enough
        on its own because the downstream agent gains structure even without
        a comparator.
        """
        filter_conditions = self.filters or (
            RecipeFilter(
                field=self.field,
                field_label=self.field_label,
                operator=self.operator,
                threshold=self.threshold,
                threshold_unit=self.threshold_unit,
            ),
        )
        sort_conditions = self.sorts or (
            RecipeSort(field=self.sort_field, descending=self.sort_descending),
        )
        has_filter = any(
            item.field is not None and item.operator is not None for item in filter_conditions
        )
        has_ranking = (
            any(item.field is not None for item in sort_conditions) and self.top_n is not None
        )
        has_structural = bool(
            self.group_by or self.group_by_fields or self.batch_id or self.snapshot_dates
        )
        return has_filter or has_ranking or has_structural


def _first_value(source: dict[str, Any], keys: tuple[str, ...]) -> Any:
    """Return the first non-empty value from ``source`` for any of ``keys``."""
    for key in keys:
        if key not in source:
            continue
        value: Any = source[key]
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if isinstance(value, (list, tuple)) and not cast(list[Any], value):
            continue
        return cast(Any, value)
    return None


def _clean_field_name(raw: Any) -> tuple[str | None, str | None]:
    """Extract ``(field, label)`` from a possibly-decorated field string.

    Examples:
        "流通A股股本 (AFLOATS)" → ("AFLOATS", "流通A股股本")
        "dailybenchgr"          → ("dailybenchgr", None)
        "本日基准增长率"        → ("本日基准增长率", None)
    """
    if not isinstance(raw, str):
        return None, None
    text = raw.strip()
    if not text:
        return None, None
    match = _PAREN_IDENT_RE.search(text)
    if match:
        ident = match.group(1)
        label = _PAREN_ANY_RE.sub("", text).strip() or None
        if label == ident:
            label = None
        return ident, label
    cleaned = _PAREN_ANY_RE.sub("", text).strip()
    return (cleaned or text), None


def _normalize_operator(raw: Any) -> str | None:
    if not isinstance(raw, str):
        return None
    key = raw.strip().lower()
    if not key:
        return None
    if key in _OPERATOR_NORMALIZATION:
        return _OPERATOR_NORMALIZATION[key]
    # 也尝试去掉常见尾词:"greater than (strict)" → "greater than"
    for marker in (" (strict)", " (inclusive)", " (exclusive)"):
        if key.endswith(marker):
            stripped = key[: -len(marker)]
            if stripped in _OPERATOR_NORMALIZATION:
                return _OPERATOR_NORMALIZATION[stripped]
    return None


def _normalize_sort_direction(raw: Any) -> bool | None:
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if not text:
        return None
    for token in _DESCENDING_TOKENS:
        if token in text:
            return True
    for token in _ASCENDING_TOKENS:
        if token in text:
            return False
    return None


def _coerce_int(raw: Any) -> int | None:
    if isinstance(raw, bool):  # bool is subclass of int — exclude
        return None
    if isinstance(raw, int):
        return raw if raw > 0 else None
    if isinstance(raw, str):
        text = raw.strip().replace(",", "").replace("，", "")
        if text.isdigit():
            value = int(text)
            return value if value > 0 else None
    return None


def _coerce_export_fields(raw: Any) -> tuple[str, ...]:
    """Normalize a single string or a list of strings into a clean field tuple."""
    if raw is None:
        return ()
    items: list[Any]
    if isinstance(raw, str):
        items = [raw]
    elif isinstance(raw, (list, tuple)):
        items = list(cast(list[Any], raw))
    else:
        return ()
    out: list[str] = []
    for item in items:
        cleaned, _ = _clean_field_name(item)
        if cleaned and cleaned not in out:
            out.append(cleaned)
    return tuple(out)


def _coerce_str(raw: Any) -> str | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        text = raw.strip()
        return text or None
    if isinstance(raw, (int, float)):
        return str(raw)
    return None


def _coerce_threshold_value(raw: Any) -> Any | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        text = raw.strip()
        return text or None
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return str(raw)
    if isinstance(raw, (list, tuple)):
        return [item for item in cast(list[Any], raw) if item is not None]
    return raw


def _coerce_snapshot_dates(raw: Any) -> tuple[str, ...]:
    if not isinstance(raw, (list, tuple)):
        return ()
    items = list(cast(list[Any], raw))
    return tuple(str(item) for item in items if item is not None and str(item).strip())


def _coerce_clean_fields(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        items = [raw]
    elif isinstance(raw, (list, tuple)):
        items = list(cast(list[Any], raw))
    else:
        return ()
    out: list[str] = []
    for item in items:
        cleaned, _ = _clean_field_name(item)
        if cleaned and cleaned not in out:
            out.append(cleaned)
    return tuple(out)


def _coerce_str_tuple(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        items = [raw]
    elif isinstance(raw, (list, tuple)):
        items = list(cast(list[Any], raw))
    else:
        items = [raw]
    out: list[str] = []
    for item in items:
        text = _coerce_str(item)
        if text and text not in out:
            out.append(text)
    return tuple(out)


def _coerce_logic(raw: Any) -> str | None:
    text = _coerce_str(raw)
    if text is None:
        return None
    upper = text.upper()
    return upper if upper in {"AND", "OR"} else text


def _extract_threshold(rules: dict[str, Any]) -> tuple[Any | None, str | None]:
    """Pull threshold value + unit from either flat or nested rule shapes."""
    raw = _first_value(rules, _THRESHOLD_KEYS)
    if isinstance(raw, dict):
        # task_4 shape: threshold = {"value": 2000000, "unit": "shares", ...}
        nested = cast(dict[str, Any], raw)
        value = _coerce_threshold_value(_first_value(nested, ("value", *_THRESHOLD_KEYS)))
        unit = _coerce_str(_first_value(nested, _UNIT_KEYS))
        return value, unit
    value = _coerce_threshold_value(raw)
    if value is None:
        value = _coerce_str(_first_value(rules, _THRESHOLD_TEXT_KEYS))
    unit = _coerce_str(_first_value(rules, _UNIT_KEYS))
    return value, unit


def _extract_operator(rules: dict[str, Any]) -> str | None:
    """Try top-level then nested-threshold for the operator string."""
    direct = _normalize_operator(_first_value(rules, _OPERATOR_KEYS))
    if direct:
        return direct
    raw_threshold = _first_value(rules, _THRESHOLD_KEYS)
    if isinstance(raw_threshold, dict):
        nested = cast(dict[str, Any], raw_threshold)
        return _normalize_operator(_first_value(nested, _OPERATOR_KEYS))
    return None


def _first_dict(raw: Any) -> dict[str, Any] | None:
    if isinstance(raw, dict):
        return cast(dict[str, Any], raw)
    if isinstance(raw, (list, tuple)):
        for item in cast(list[Any], raw):
            if isinstance(item, dict):
                return cast(dict[str, Any], item)
    return None


def _first_scalar(raw: Any) -> Any:
    if isinstance(raw, (list, tuple)):
        for item in cast(list[Any], raw):
            if item is not None and not (isinstance(item, str) and not item.strip()):
                return item
        return None
    return raw


def _dict_items(raw: Any) -> tuple[dict[str, Any], ...]:
    if isinstance(raw, dict):
        return (cast(dict[str, Any], raw),)
    if isinstance(raw, (list, tuple)):
        return tuple(
            cast(dict[str, Any], item) for item in cast(list[Any], raw) if isinstance(item, dict)
        )
    return ()


def _sort_items(raw: Any) -> tuple[dict[str, Any], ...]:
    if isinstance(raw, str):
        return ({"field": raw},)
    return _dict_items(raw)


def _canonical_dict_from_rules(rules: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(rules, dict) or not rules:
        return None
    canonical = rules.get("canonical")
    if isinstance(canonical, dict):
        return cast(dict[str, Any], canonical)
    if set(rules).issubset(_CANONICAL_RULE_KEYS):
        return rules
    return None


def _extract_canonical_filters(canonical: dict[str, Any] | None) -> tuple[RecipeFilter, ...]:
    if canonical is None:
        return ()
    out: list[RecipeFilter] = []
    for item in _dict_items(canonical.get("filters")):
        field_value, field_label_from_paren = _clean_field_name(_first_value(item, _FIELD_KEYS))
        explicit_label = _coerce_str(_first_value(item, _FIELD_LABEL_KEYS))
        threshold, threshold_unit = _extract_threshold(item)
        condition = RecipeFilter(
            field=field_value,
            field_label=field_label_from_paren or explicit_label,
            operator=_extract_operator(item),
            threshold=threshold,
            threshold_unit=threshold_unit,
            source_rule_id=_coerce_str(item.get("source_rule_id")),
            note=_coerce_str(item.get("note")),
        )
        if any(
            (
                condition.field,
                condition.field_label,
                condition.operator,
                condition.threshold is not None,
                condition.threshold_unit,
                condition.source_rule_id,
                condition.note,
            )
        ):
            out.append(condition)
    return tuple(out)


def _extract_canonical_sorts(canonical: dict[str, Any] | None) -> tuple[RecipeSort, ...]:
    if canonical is None:
        return ()
    out: list[RecipeSort] = []
    for item in _sort_items(canonical.get("sort_by")):
        field, _ = _clean_field_name(_first_value(item, ("field", *_SORT_FIELD_KEYS)))
        sort = RecipeSort(
            field=field,
            descending=_normalize_sort_direction(
                _first_value(item, ("direction", *_SORT_DIR_KEYS))
            ),
            source_rule_id=_coerce_str(item.get("source_rule_id")),
            note=_coerce_str(item.get("note")),
        )
        if any((sort.field, sort.descending is not None, sort.source_rule_id, sort.note)):
            out.append(sort)
    return tuple(out)


def _canonical_rules_to_legacy(canonical: dict[str, Any]) -> dict[str, Any]:
    """Translate the new conservative rules.canonical shape for old recipe fields."""
    out: dict[str, Any] = {}

    first_filter = _first_dict(canonical.get("filters"))
    if first_filter is not None:
        if first_filter.get("field") is not None:
            out["field"] = first_filter.get("field")
        if first_filter.get("operator") is not None:
            out["operator"] = first_filter.get("operator")
        if first_filter.get("value") is not None:
            out["threshold"] = first_filter.get("value")
        if first_filter.get("unit") is not None:
            out["unit"] = first_filter.get("unit")

    first_metric = _first_dict(canonical.get("metrics"))
    if first_metric is not None:
        if "field" not in out and first_metric.get("field") is not None:
            out["metric"] = first_metric.get("field")
        if first_metric.get("aggregation") is not None:
            out["aggregation"] = first_metric.get("aggregation")

    first_sort = _first_dict(canonical.get("sort_by"))
    if first_sort is not None:
        if first_sort.get("field") is not None:
            out["sort_field"] = first_sort.get("field")
        if first_sort.get("direction") is not None:
            out["sort_direction"] = first_sort.get("direction")

    if canonical.get("logic") is not None:
        out["logic"] = canonical.get("logic")
    if canonical.get("top_n") is not None:
        out["top_n"] = canonical.get("top_n")
    if canonical.get("group_by"):
        out["group_by"] = _first_scalar(canonical.get("group_by"))
    if canonical.get("tables"):
        out["table"] = _first_scalar(canonical.get("tables"))
    if canonical.get("output_fields"):
        out["output_fields"] = canonical.get("output_fields")
    if canonical.get("snapshot_dates"):
        out["snapshot_dates"] = canonical.get("snapshot_dates")
    if canonical.get("periods"):
        out["statistical_period"] = _first_scalar(canonical.get("periods"))

    identifiers = canonical.get("identifiers")
    if isinstance(identifiers, dict):
        for key, value in cast(dict[str, Any], identifiers).items():
            out.setdefault(key, value)
    return out


def _rules_for_recipe(rules: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return the best backward-compatible rule dict for recipe extraction."""
    if not isinstance(rules, dict) or not rules:
        return None
    if not any(
        key in rules for key in ("canonical", "rule_items", "raw_observations", "task_specific")
    ):
        if set(rules).issubset(_CANONICAL_RULE_KEYS):
            return _canonical_rules_to_legacy(rules)
        return rules

    out: dict[str, Any] = {}
    canonical = rules.get("canonical")
    if isinstance(canonical, dict):
        out.update(_canonical_rules_to_legacy(cast(dict[str, Any], canonical)))

    task_specific = rules.get("task_specific")
    if isinstance(task_specific, dict):
        for key, value in cast(dict[str, Any], task_specific).items():
            out.setdefault(key, value)
    return out or None


def extract_filter_recipe(rules: dict[str, Any] | None) -> FilterRecipe | None:
    """Walk a video rules dict, return a canonical FilterRecipe or None if empty.

    New reports prefer ``rules.canonical`` plus flexible fallbacks. Legacy flat
    rules remain supported because historical traces and tests still carry them.
    """
    canonical = _canonical_dict_from_rules(rules)
    rules = _rules_for_recipe(rules)
    if not isinstance(rules, dict):
        return None
    if not rules:
        return None

    field_value, field_label_from_paren = _clean_field_name(_first_value(rules, _FIELD_KEYS))
    explicit_label = _coerce_str(_first_value(rules, _FIELD_LABEL_KEYS))
    field_label = field_label_from_paren or explicit_label

    operator = _extract_operator(rules)
    threshold, threshold_unit = _extract_threshold(rules)

    top_n = _coerce_int(_first_value(rules, _TOPN_KEYS))
    export_fields = _coerce_export_fields(_first_value(rules, _EXPORT_KEYS))

    sort_field_raw = _first_value(rules, _SORT_FIELD_KEYS)
    sort_field, _ = _clean_field_name(sort_field_raw)
    if sort_field is None and top_n is not None:
        sort_field, _ = _clean_field_name(_first_value(rules, _TOPN_SORT_FALLBACK_KEYS))
    sort_descending = _normalize_sort_direction(_first_value(rules, _SORT_DIR_KEYS))

    group_by_raw = _first_value(rules, _GROUPBY_KEYS)
    group_by, _ = _clean_field_name(group_by_raw)

    table_hint_raw = _first_value(rules, _TABLE_KEYS)
    table_hint: str | None
    if isinstance(table_hint_raw, (list, tuple)):
        first = next(iter(cast(list[Any], table_hint_raw)), None)
        table_hint = _coerce_str(first)
    else:
        table_hint = _coerce_str(table_hint_raw)

    batch_id = _coerce_str(_first_value(rules, _BATCH_KEYS))
    snapshot_dates = _coerce_snapshot_dates(_first_value(rules, _SNAPSHOT_KEYS))

    filters = _extract_canonical_filters(canonical)
    if filters:
        primary_filter = filters[0]
        field_value = field_value or primary_filter.field
        field_label = field_label or primary_filter.field_label
        operator = operator or primary_filter.operator
        if threshold is None:
            threshold = primary_filter.threshold
        threshold_unit = threshold_unit or primary_filter.threshold_unit
    elif any((field_value, field_label, operator, threshold is not None, threshold_unit)):
        filters = (
            RecipeFilter(
                field=field_value,
                field_label=field_label,
                operator=operator,
                threshold=threshold,
                threshold_unit=threshold_unit,
            ),
        )

    sorts = _extract_canonical_sorts(canonical)
    if sorts:
        primary_sort = sorts[0]
        sort_field = sort_field or primary_sort.field
        if sort_descending is None:
            sort_descending = primary_sort.descending
    elif sort_field:
        sorts = (RecipeSort(field=sort_field, descending=sort_descending),)

    logic = _coerce_logic(canonical.get("logic") if canonical is not None else rules.get("logic"))

    group_by_fields = _coerce_clean_fields(
        canonical.get("group_by") if canonical is not None else None
    )
    if group_by_fields:
        group_by = group_by or group_by_fields[0]
    elif group_by:
        group_by_fields = (group_by,)

    table_hints = _coerce_str_tuple(canonical.get("tables") if canonical is not None else None)
    if table_hints:
        table_hint = table_hint or table_hints[0]
    elif table_hint:
        table_hints = (table_hint,)

    recipe = FilterRecipe(
        field=field_value,
        field_label=field_label,
        operator=operator,
        threshold=threshold,
        threshold_unit=threshold_unit,
        sort_field=sort_field,
        sort_descending=sort_descending,
        top_n=top_n,
        export_fields=export_fields,
        group_by=group_by,
        table_hint=table_hint,
        batch_id=batch_id,
        snapshot_dates=snapshot_dates,
        filters=filters,
        logic=logic,
        sorts=sorts,
        group_by_fields=group_by_fields,
        table_hints=table_hints,
    )
    return (
        recipe
        if any(
            (
                recipe.field,
                recipe.operator,
                recipe.threshold is not None,
                recipe.sort_field,
                recipe.top_n,
                recipe.export_fields,
                recipe.group_by,
                recipe.batch_id,
                recipe.snapshot_dates,
                recipe.filters,
                recipe.sorts,
            )
        )
        else None
    )


def _recipe_filters(recipe: FilterRecipe) -> tuple[RecipeFilter, ...]:
    if recipe.filters:
        return recipe.filters
    if any((recipe.field, recipe.field_label, recipe.operator, recipe.threshold is not None)):
        return (
            RecipeFilter(
                field=recipe.field,
                field_label=recipe.field_label,
                operator=recipe.operator,
                threshold=recipe.threshold,
                threshold_unit=recipe.threshold_unit,
            ),
        )
    return ()


def _recipe_sorts(recipe: FilterRecipe) -> tuple[RecipeSort, ...]:
    if recipe.sorts:
        return recipe.sorts
    if recipe.sort_field:
        return (RecipeSort(field=recipe.sort_field, descending=recipe.sort_descending),)
    return ()


def _sort_direction_label(descending: bool | None) -> str:
    return "DESC" if descending else ("ASC" if descending is False else "?")


def _format_filter_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return repr(value)


def _format_recipe_filter(condition: RecipeFilter) -> str:
    field = condition.field or condition.field_label or "?"
    label_tail = (
        f" (label: {condition.field_label})" if condition.field and condition.field_label else ""
    )
    op = condition.operator or "?"
    value_tail = (
        "" if condition.threshold is None else f" {_format_filter_value(condition.threshold)}"
    )
    unit_tail = f" {condition.threshold_unit}" if condition.threshold_unit else ""
    source_tail = f" [source: {condition.source_rule_id}]" if condition.source_rule_id else ""
    note_tail = f" ({condition.note})" if condition.note else ""
    return f"{field}{label_tail} {op}{value_tail}{unit_tail}{source_tail}{note_tail}"


def _format_recipe_sort(sort: RecipeSort) -> str:
    field = sort.field or "?"
    source_tail = f" [source: {sort.source_rule_id}]" if sort.source_rule_id else ""
    note_tail = f" ({sort.note})" if sort.note else ""
    return f"{field} {_sort_direction_label(sort.descending)}{source_tail}{note_tail}"


def _render_filter_expression(condition: RecipeFilter) -> tuple[str | None, str | None]:
    if not condition.field or not condition.operator:
        return None, f"filter incomplete: {_format_recipe_filter(condition)}"
    op = condition.operator
    field_repr = repr(condition.field)
    value = condition.threshold
    if op in {">", ">=", "<", "<=", "==", "!="}:
        if value is None:
            return None, f"filter missing threshold: {_format_recipe_filter(condition)}"
        return f"(df[{field_repr}] {op} {value!r})", None
    if op == "in":
        if value is None:
            return None, f"IN filter missing values: {_format_recipe_filter(condition)}"
        values = cast(list[Any], value) if isinstance(value, list) else [value]
        return f"(df[{field_repr}].isin({values!r}))", None
    if op == "contains":
        if value is None:
            return None, f"contains filter missing text: {_format_recipe_filter(condition)}"
        return f"(df[{field_repr}].astype(str).str.contains({value!r}, na=False))", None
    if op == "between":
        if isinstance(value, list):
            values = cast(list[Any], value)
            if len(values) == 2:
                low, high = values
                return f"(df[{field_repr}].between({low!r}, {high!r}, inclusive='both'))", None
        return None, f"between filter needs [low, high]: {_format_recipe_filter(condition)}"
    return None, f"unsupported operator for template: {_format_recipe_filter(condition)}"


def render_filter_recipe(recipe: FilterRecipe) -> str:
    """Format a FilterRecipe into the plaintext hint surfaced to the agent.

    Output is intentionally machine-readable-ish: a structured bullet list of
    the recipe fields followed by a pandas template skeleton. Every line
    reminds the agent that field/table names must be verified before use.
    """
    lines: list[str] = [
        "[video-rule recipe - VERIFY each name against actual schema before use]",
    ]
    if recipe.field:
        label_tail = f" (label: {recipe.field_label})" if recipe.field_label else ""
        lines.append(f"  field:        {recipe.field}{label_tail}")
    elif recipe.field_label:
        lines.append(f"  field_label:  {recipe.field_label} [no ASCII column name extracted]")
    if recipe.operator:
        lines.append(f"  operator:     {recipe.operator}")
    if recipe.threshold is not None:
        unit_tail = f" (unit: {recipe.threshold_unit})" if recipe.threshold_unit else ""
        lines.append(f"  threshold:    {_format_filter_value(recipe.threshold)}{unit_tail}")

    filters = _recipe_filters(recipe)
    if len(filters) > 1:
        logic = recipe.logic or "?"
        logic_tail = "" if recipe.logic else " (verify AND vs OR)"
        lines.append(f"  filter_logic: {logic}{logic_tail}")
        lines.append("  filters:")
        for index, condition in enumerate(filters, start=1):
            lines.append(f"    {index}. {_format_recipe_filter(condition)}")

    if recipe.sort_field:
        lines.append(
            f"  sort:         {recipe.sort_field} {_sort_direction_label(recipe.sort_descending)}"
        )
    sorts = _recipe_sorts(recipe)
    if len(sorts) > 1:
        lines.append("  sort_by:")
        for index, sort in enumerate(sorts, start=1):
            lines.append(f"    {index}. {_format_recipe_sort(sort)}")
    if recipe.top_n is not None:
        lines.append(f"  top_n:        {recipe.top_n}")

    group_by_fields = recipe.group_by_fields or ((recipe.group_by,) if recipe.group_by else ())
    if len(group_by_fields) > 1:
        lines.append(f"  group_by:     {list(group_by_fields)}")
    elif recipe.group_by:
        lines.append(f"  group_by:     {recipe.group_by}")
    if recipe.export_fields:
        lines.append(f"  output_cols:  {list(recipe.export_fields)}")

    table_hints = recipe.table_hints or ((recipe.table_hint,) if recipe.table_hint else ())
    if len(table_hints) > 1:
        lines.append(f"  table_hints:  {list(table_hints)}")
    elif recipe.table_hint:
        lines.append(f"  table_hint:   {recipe.table_hint}")
    if recipe.batch_id:
        lines.append(f"  batch_id:     {recipe.batch_id}")
    if recipe.snapshot_dates:
        lines.append(f"  snapshot_dates: {list(recipe.snapshot_dates)}  (emit one row per date)")

    template_lines = _render_pandas_template(recipe)
    if template_lines:
        lines.append("")
        lines.append("Pandas template skeleton (adapt column names from inspect_files):")
        lines.extend(template_lines)
    return "\n".join(lines)


def _render_pandas_template(recipe: FilterRecipe) -> list[str]:
    """Emit a minimal pandas snippet that mirrors the recipe's intent."""
    if not recipe.is_sufficient():
        return []
    out: list[str] = ["```python"]
    table_hints = recipe.table_hints or ((recipe.table_hint,) if recipe.table_hint else ())
    if table_hints:
        out.append(f"  df = pd.read_csv('csv/{table_hints[0]}.csv')  # verify exact path")
        if len(table_hints) > 1:
            out.append(f"  # additional tables referenced by the video: {list(table_hints[1:])}")
    else:
        out.append("  df = pd.read_csv(...)  # use the source table identified by inspect_files")

    filters = _recipe_filters(recipe)
    mask_exprs: list[str] = []
    filter_notes: list[str] = []
    for condition in filters:
        expr, note = _render_filter_expression(condition)
        if expr:
            mask_exprs.append(expr)
        if note:
            filter_notes.append(note)
    if mask_exprs:
        out.append("  # coerce thresholds to matching column dtypes before filtering")
        if len(mask_exprs) == 1:
            out.append(f"  mask = {mask_exprs[0]}")
        else:
            if recipe.logic not in {"AND", "OR"}:
                out.append(
                    "  # filter_logic unknown; verify whether these conditions are AND or OR"
                )
            joiner = "|" if recipe.logic == "OR" else "&"
            out.append("  mask = (")
            for index, expr in enumerate(mask_exprs):
                prefix = "" if index == 0 else f"{joiner} "
                out.append(f"      {prefix}{expr}")
            out.append("  )")
        out.append("  result = df[mask]")
    elif filters:
        for note in filter_notes:
            out.append(f"  # {note}")
        out.append("  result = df  # add the comparison once incomplete filters are resolved")
    else:
        out.append("  result = df  # no explicit filter rule extracted")

    sorts = tuple(sort for sort in _recipe_sorts(recipe) if sort.field)
    if len(sorts) == 1:
        sort = sorts[0]
        ascending = sort.descending is False
        out.append(f"  result = result.sort_values('{sort.field}', ascending={ascending})")
        if sort.descending is None:
            out.append("  # verify sort direction; video did not specify ASC/DESC clearly")
    elif sorts:
        fields = [cast(str, sort.field) for sort in sorts]
        ascending = [sort.descending is False for sort in sorts]
        out.append(f"  result = result.sort_values({fields!r}, ascending={ascending!r})")
        if any(sort.descending is None for sort in sorts):
            out.append("  # verify unknown sort directions before executing")
    if recipe.top_n is not None:
        out.append(f"  result = result.head({recipe.top_n})")

    group_by_fields = recipe.group_by_fields or ((recipe.group_by,) if recipe.group_by else ())
    if group_by_fields:
        out.append(
            f"  # group_by: {list(group_by_fields)} - adjust to groupby(...).agg(...) "
            "if the question asks per-group output"
        )
    if recipe.export_fields:
        cols_repr = ", ".join(f"'{c}'" for c in recipe.export_fields)
        out.append(f"  answer_df = result[[{cols_repr}]]  # verify columns are the WH-target")
    else:
        out.append("  answer_df = result[[...]]  # pick the WH-target columns explicitly")
    out.append("```")
    return out


def synthesize_filter_recipe(rules: dict[str, Any] | None) -> str | None:
    """One-shot convenience: rules dict → rendered hint or None.

    Returns None when no recipe can be extracted OR when the extracted recipe
    is too sparse to be worth surfacing — callers can use truthiness to gate.
    """
    recipe = extract_filter_recipe(rules)
    if recipe is None or not recipe.is_sufficient():
        return None
    return render_filter_recipe(recipe)
