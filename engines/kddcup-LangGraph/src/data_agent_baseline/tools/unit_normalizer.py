from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------------------
# Unit registry — maps unit strings to their multiplication factor
# relative to the base unit (factor = 1).
# All values are normalised to unit=1 by multiplying raw_number × factor.
# ---------------------------------------------------------------------------

UNIT_FACTORS: dict[str, float] = {
    # --- Currency: base = 元 (yuan) ---
    "元": 1.0,
    "万元": 10_000.0,
    "百万元": 1_000_000.0,
    "千万元": 10_000_000.0,
    "亿元": 100_000_000.0,
    "十亿元": 1_000_000_000.0,
    "万亿元": 1_000_000_000_000.0,
    "千元": 1_000.0,
    "百元": 100.0,
    "十元": 10.0,
    # Aliases without 元 suffix
    "万": 10_000.0,
    "百万": 1_000_000.0,
    "千万": 10_000_000.0,
    "亿": 100_000_000.0,
    "十亿": 1_000_000_000.0,
    "万亿": 1_000_000_000_000.0,
    "千": 1_000.0,
    "百": 100.0,
    # English aliases
    "million": 1_000_000.0,
    "billion": 1_000_000_000.0,
    "trillion": 1_000_000_000_000.0,
    # --- Percentage: base = 1 (100% = 1.0) ---
    "%": 0.01,
    "百分比": 0.01,
    "％": 0.01,
    # --- Basis points ---
    "基点": 0.0001,
    "bp": 0.0001,
    "bps": 0.0001,
    # --- Explicit base unit ---
    "1": 1.0,
    # --- Shares / financial instruments: base = 股 (1 share) ---
    "股": 1.0,
    "万股": 10_000.0,
    "亿股": 100_000_000.0,
    "shares": 1.0,
    # --- Dimensionless / counts ---
    "次": 1.0,
    "个": 1.0,
    "家": 1.0,
    "人": 1.0,
    "户": 1.0,
    "笔": 1.0,
    "份": 1.0,
    "张": 1.0,
    "只": 1.0,
    "台": 1.0,
    "辆": 1.0,
    "项": 1.0,
    "条": 1.0,
    "件": 1.0,
    "吨": 1.0,
    "公斤": 1.0,
    "千克": 1.0,
    "克": 1.0,
    "升": 1.0,
    "毫升": 1.0,
    "米": 1.0,
    "公里": 1.0,
    "平方米": 1.0,
    "公顷": 1.0,
}

# Canonical unit name → factor (a normalised subset for easy lookup)
_CANONICAL_UNIT_TO_FACTOR: dict[str, float] = {
    "元": 1.0,
    "万元": 10_000.0,
    "百万元": 1_000_000.0,
    "千万元": 10_000_000.0,
    "亿元": 100_000_000.0,
    "十亿元": 1_000_000_000.0,
    "万亿元": 1_000_000_000_000.0,
    "千元": 1_000.0,
    "百元": 100.0,
    "%": 0.01,
    "基点": 0.0001,
    "1": 1.0,
}

# ---------------------------------------------------------------------------
# Unit alias normalisation — maps variant spellings to canonical forms.
# ---------------------------------------------------------------------------

_UNIT_ALIAS_MAP: list[tuple[str, str]] = [
    # Bare scale aliases → canonical currency unit
    ("万亿", "万亿元"),
    ("十亿", "十亿元"),
    ("千万", "千万元"),
    ("百万", "百万元"),
    ("亿", "亿元"),
    ("万", "万元"),
    ("千", "千元"),
    ("百", "百元"),
    # Other
    ("百分比", "%"),
    ("％", "%"),
    ("基点", "基点"),
    ("bps", "基点"),
    ("bp", "基点"),
    # English unit aliases
    ("million", "百万元"),
    ("billion", "十亿元"),
    ("trillion", "万亿元"),
]

# ---------------------------------------------------------------------------
# Compound-unit prefixes — tried longest-first so "万亿" matches before "万".
# Used to decompose units like "万股" → prefix "万" × suffix "股".
# ---------------------------------------------------------------------------

_COMPOUND_PREFIXES: list[tuple[str, float]] = [
    ("万亿", 1_000_000_000_000.0),
    ("十亿", 1_000_000_000.0),
    ("千万", 10_000_000.0),
    ("百万", 1_000_000.0),
    ("亿", 100_000_000.0),
    ("万", 10_000.0),
    ("千", 1_000.0),
    ("百", 100.0),
    ("十", 10.0),
    # English
    ("trillion", 1_000_000_000_000.0),
    ("billion", 1_000_000_000.0),
    ("million", 1_000_000.0),
]

# Fuzzy keyword → factor for last-resort matching when the unit string
# contains a scale keyword but didn't match any exact/compound pattern.
_FUZZY_KEYWORD_FACTORS: list[tuple[str, float]] = [
    ("万亿", 1_000_000_000_000.0),
    ("十亿", 1_000_000_000.0),
    ("千万", 10_000_000.0),
    ("百万", 1_000_000.0),
    ("亿", 100_000_000.0),
    ("万", 10_000.0),
    ("千", 1_000.0),
    ("百", 100.0),
    ("trillion", 1_000_000_000_000.0),
    ("billion", 1_000_000_000.0),
    ("million", 1_000_000.0),
]


def normalize_unit(raw_unit: str | None) -> str | None:
    """Canonicalise a unit string to its standard form.

    Returns the canonical unit name (e.g. ``"万元"``, ``"%"``) or
    ``None`` when the input is ``None`` / empty.
    """
    if raw_unit is None:
        return None
    unit = raw_unit.strip()
    if not unit:
        return None
    if unit in _CANONICAL_UNIT_TO_FACTOR:
        return unit
    for pattern, canonical in _UNIT_ALIAS_MAP:
        if unit == pattern:
            return canonical
    cleaned = re.sub(r"[（(][^)）]*[)）]$", "", unit).strip()
    if cleaned and cleaned != unit:
        return normalize_unit(cleaned)
    if unit in UNIT_FACTORS:
        return unit
    # --- compound-unit decomposition: "万X" → prefix × suffix ---
    for prefix, prefix_factor in _COMPOUND_PREFIXES:
        if unit.startswith(prefix) and len(unit) > len(prefix):
            suffix = unit[len(prefix):]
            suffix_canonical = normalize_unit(suffix)
            if suffix_canonical is not None:
                compound = prefix + suffix_canonical
                compound_factor = prefix_factor * UNIT_FACTORS[suffix_canonical]
                UNIT_FACTORS[compound] = compound_factor
                return compound
    return None


def get_factor(unit: str | None) -> float:
    """Return the multiplication factor for *unit*.

    Raises :exc:`KeyError` if the unit is unknown (including ``None``).
    """
    if unit is None:
        raise KeyError("unit is None — cannot determine conversion factor")
    canonical = normalize_unit(unit)
    if canonical is None:
        # Fuzzy fallback: scan the unit string for scale keywords (万/亿/million/...).
        # This catches non-standard unit strings like "shares (万)" or "万 shares".
        for keyword, factor in _FUZZY_KEYWORD_FACTORS:
            if keyword in unit:
                return factor
        raise KeyError(
            f"Unknown unit {unit!r}. "
            f"Known units: {sorted(UNIT_FACTORS.keys())}"
        )
    factor = UNIT_FACTORS.get(canonical)
    if factor is None:
        raise KeyError(
            f"Unit {unit!r} normalised to {canonical!r} but no factor registered."
        )
    return factor


def normalize_value(raw_number: float, raw_unit: str | None) -> float:
    """Convert *raw_number* from *raw_unit* to the base unit (unit=1).

    The formula is simply::

        result = raw_number × factor(raw_unit)

    When *raw_unit* is ``None`` the raw number is returned as-is (pass-through).

    Examples
    --------
    >>> normalize_value(1.5, "万元")
    15000.0
    >>> normalize_value(21.01965, "百万元")
    21019650.0
    >>> normalize_value(15, "%")
    0.15
    >>> normalize_value(42, None)
    42.0
    """
    if raw_unit is None:
        return float(raw_number)
    return raw_number * get_factor(raw_unit)


def _is_raw_value_dict(value: Any) -> bool:
    """Check whether *value* has the ``{raw_number, raw_unit}`` shape."""
    return isinstance(value, dict) and "raw_number" in value


def normalize_facts(
    facts: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Normalise raw ``{raw_number, raw_unit}`` values in *facts* to bare numbers.

    Every fact value with the ``{raw_number, raw_unit}`` shape is converted
    to a bare number by multiplying ``raw_number × factor(raw_unit)``.
    The target is always the base unit (unit=1).

    Returns
    -------
    ``(normalised_facts, conversion_log)`` where *conversion_log* is a list
    of per-conversion event dicts suitable for structured logging.
    """
    normalised: list[dict[str, Any]] = []
    conversion_log: list[dict[str, Any]] = []

    for fact in facts:
        values = fact.get("values", {})
        if not isinstance(values, dict):
            normalised.append(fact)
            continue

        new_values: dict[str, Any] = {}
        for field, value in values.items():
            if not _is_raw_value_dict(value):
                new_values[field] = value
                continue

            raw_number = value["raw_number"]
            raw_unit = value.get("raw_unit")

            try:
                converted = normalize_value(raw_number, raw_unit)
            except KeyError as exc:
                converted = float(raw_number)
                conversion_log.append({
                    "line_id": fact.get("line_id"),
                    "field": field,
                    "raw_number": raw_number,
                    "raw_unit": raw_unit,
                    "converted": converted,
                    "warning": str(exc),
                })
            else:
                conversion_log.append({
                    "line_id": fact.get("line_id"),
                    "field": field,
                    "raw_number": raw_number,
                    "raw_unit": raw_unit,
                    "converted": converted,
                })

            new_values[field] = converted

        normalised.append({**fact, "values": new_values})

    return normalised, conversion_log


def generate_normalization_rule(
    expected_source_units: list[str] | None = None,
) -> str:
    """Generate a human-readable normalisation rule string."""
    rule = "Values are normalised to the base unit (unit=1) by the unit normaliser."
    if expected_source_units:
        rule += (
            f" Expected source units: {expected_source_units}. "
            "Each value is multiplied by its unit factor to reach the base unit."
        )
    else:
        rule += " Source unit detected from document context."
    return rule
