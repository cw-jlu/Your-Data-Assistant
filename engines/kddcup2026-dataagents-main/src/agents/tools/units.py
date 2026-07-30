"""Column-conversion metadata from ETL _units.json caches."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ColumnConversion:
    field: str
    source: str
    target: str
    factor: float
    label: str


def load_conversions(units_path: Path) -> list[ColumnConversion]:
    """Parse a ``_units.json`` into validated column conversions.

    Returns an empty list on I/O errors, parse errors, or when no
    actionable conversions exist (identity factor, non-numeric, etc.).
    """
    try:
        units = json.loads(units_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    result: list[ColumnConversion] = []
    by_lower = {str(key).lower(): value for key, value in units.items()}
    for field, source_unit in units.items():
        if field.startswith("_"):
            continue
        target = by_lower.get(f"_target_{field}".lower())
        factor_raw = by_lower.get(f"_factor_{field}".lower())
        if not target or not factor_raw or str(source_unit) == str(target):
            continue
        try:
            factor = float(factor_raw)
        except (ValueError, TypeError):
            continue
        if factor == 0 or factor == 1:
            continue
        result.append(
            ColumnConversion(
                field=field,
                source=str(source_unit),
                target=str(target),
                factor=factor,
                label=f"CONVERT: ×{factor_raw} ({source_unit} → {target})",
            )
        )
    return result
