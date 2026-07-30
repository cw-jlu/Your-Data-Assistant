"""Unit conversion, proportion convention inference, and units sidecar I/O."""

from __future__ import annotations

import contextlib
import csv
import json
import logging
import math
import re
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

from agents.config import ETL_SCRATCH_ROOT
from agents.etl._constants import (
    PROPORTION_SAMPLE_LIMIT,
    PROPORTION_SUBFAMILIES,
    UNIT_SCALE_FAMILIES,
)
from agents.llm import ModelMessage, ModelResponse

if TYPE_CHECKING:
    from agents.benchmark.schema import PublicTask
    from agents.llm import ModelAdapter

logger = logging.getLogger(__name__)


def _deterministic_factor(source_unit: str, target_unit: str) -> float | None:
    """Compute the exact source→target multiplier when both units share a scale family.

    Returns None when the pair is not exactly computable (unknown unit, or the
    units belong to different families).
    """
    src = source_unit.strip()
    tgt = target_unit.strip()
    for family in UNIT_SCALE_FAMILIES:
        if src in family and tgt in family:
            return family[src] / family[tgt]
    return None


def vote_target_units(
    adapter: ModelAdapter,
    km_text: str,
    field_units: dict[str, str],
    question: str = "",
) -> dict[str, tuple[str, float]]:
    """Ask the LLM for each field's target unit and conversion factor.

    Returns {field: (target_unit, factor)} where factor is the multiplier
    to convert source values to target values (e.g. 亿元→万元 = 10000).
    Fields whose source unit already matches the target are excluded.
    """
    source_fields = {
        f: u
        for f, u in field_units.items()
        if not f.startswith("_") and u and not u.startswith("_")
    }
    if not source_fields:
        return {}

    field_list = ", ".join(f"{f}={u}" for f, u in source_fields.items())
    prompt = (
        "Below is knowledge.md and a list of fields with their "
        "current units as found in the source data.\n\n"
        f"Fields: {field_list}\n\n"
        "For each field, determine the TARGET unit that knowledge.md "
        "EXPLICITLY declares. The declared unit is AUTHORITATIVE — NEVER "
        "override it with reporting conventions, domain knowledge, or common "
        "practice. If knowledge.md says '百万元', the target is 百万元 even "
        "when the source is 亿元. If the document declares a default unit "
        "(e.g. '百万元'), apply it to all monetary fields. If a field's source "
        "unit already matches the target, repeat it unchanged. "
        "Dimensionless fields (price indices, ranks, %) must keep their "
        "source unit as-is — do NOT replace with '_default' or any placeholder.\n\n"
        "PERCENTAGE RULE:\n"
        "1. If the QUESTION asks for a '比例' (proportion/ratio) and the "
        "field's source unit is '%', convert to 'ratio' (factor 0.01). "
        "In Chinese financial data, '比例' means a decimal fraction "
        "(e.g. 0.0155 = 1.55%), matching the canonical database storage.\n"
        "2. Otherwise, only convert to 'ratio' when knowledge.md "
        "EXPLICITLY uses the word 'ratio' or 'decimal ratio' "
        "in that field's definition. If knowledge.md merely says '%', "
        "'percentage', '百分比', or the field is just a return/rate expressed "
        "in percent, keep the source unit as '%' — do NOT convert.\n\n"
        "For each field where source ≠ target, also output the numeric "
        "multiplication factor: target_value = source_value × factor. "
        "Mind the direction — converting to a LARGER unit needs a factor "
        "BELOW 1.\n\n"
        "Output format — one field per entry, comma-separated:\n"
        "  field*factor=target_unit\n"
        "If source == target (no conversion needed), omit the *factor part:\n"
        "  field=target_unit\n\n"
        "Examples (derive actual units from knowledge.md, not from these examples):\n"
        "  field_a*0.0001=万元, field_b=%\n"
        "  (field_a: source was 元, target is 万元, factor=1/10000)\n"
        "  (field_b: source was %, target is %, no conversion)\n\n"
        f"knowledge.md:\n{km_text}"
    )
    if question:
        prompt += f"\n\nQUESTION (for context): {question}"
    messages = [
        ModelMessage(
            role="system",
            content=(
                "Map each field to its target unit per knowledge.md and the question. "
                "Output field=unit*factor pairs. Omit *factor when source==target."
            ),
        ),
        ModelMessage(role="user", content=prompt),
    ]

    try:
        response: ModelResponse = adapter.complete(messages)
    except Exception:
        return {}

    return _parse_unit_factor_response(response.content.strip(), source_fields)


def _parse_unit_factor_response(
    text: str,
    source_fields: dict[str, str],
) -> dict[str, tuple[str, float]]:
    """Parse ``field=unit*factor`` response into {field: (unit, factor)}.

    Only returns entries where source_unit != target_unit and factor is valid.
    For unit pairs within a known scale family the factor is computed
    deterministically, overriding whatever the LLM produced — the LLM is
    trusted only for choosing the target unit, not the arithmetic.
    """
    result: dict[str, tuple[str, float]] = {}
    source_by_lower = {field.lower(): field for field in source_fields}
    raw = text.split(":", 1)[-1] if ":" in text else text
    for item in re.split(r"[,;\n]", raw):
        item = item.strip()
        if not item:
            continue
        field, sep, rest = item.partition("=")
        if not sep:
            continue
        field = field.strip()
        rest = rest.strip()
        # Support both formats:
        #   new: field*factor=target_unit  (field already has * stripped)
        #   old: field=target_unit*factor
        if "*" in field:
            field, _, factor_part = field.rpartition("*")
            field = field.strip()
            unit_part = rest
            star = True
        else:
            unit_part, star_sep, factor_part = rest.partition("*")
            star = bool(star_sep)
            unit_part = unit_part.strip()
        canonical_field = source_by_lower.get(field.lower())
        if canonical_field is None:
            continue
        if not unit_part or unit_part.startswith("_"):
            continue
        if unit_part == source_fields[canonical_field]:
            continue
        exact = _deterministic_factor(source_fields[canonical_field], unit_part)
        if exact is not None:
            if exact == 1:
                continue
            if star:
                with contextlib.suppress(ValueError, TypeError):
                    if float(factor_part.strip()) != exact:
                        logger.warning(
                            "Unit factor for %s (%s→%s) corrected: LLM gave %s, exact is %s",
                            field,
                            source_fields[canonical_field],
                            unit_part,
                            factor_part.strip(),
                            exact,
                        )
            result[canonical_field] = (unit_part, exact)
            continue
        if not star:
            continue
        try:
            factor = float(factor_part.strip())
        except (ValueError, TypeError):
            continue
        if factor == 0 or factor == 1:
            continue
        result[canonical_field] = (unit_part, factor)
    return result


def write_units_sidecar(
    task: PublicTask,
    prose_stem: str,
    field_units: dict[str, str],
) -> None:
    """Write per-field unit metadata to a sidecar JSON in _cache/."""
    if not field_units:
        return
    cache_dir = ETL_SCRATCH_ROOT / task.task_id / "_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    units_path = cache_dir / f"{prose_stem}_units.json"
    units_path.write_text(json.dumps(field_units, ensure_ascii=False), encoding="utf-8")


def _proportion_subfamily(name: str) -> str | None:
    """The proportion naming subfamily a column belongs to, or None."""
    for subfamily, pattern in PROPORTION_SUBFAMILIES:
        if pattern.search(name):
            return subfamily
    return None


def _classify_proportion_values(values: list[float]) -> str | None:
    """'ratio' when every |v| <= 1, 'percent' when any |v| > 1, None when empty."""
    finite = [abs(v) for v in values if math.isfinite(v)]
    if not finite:
        return None
    return "ratio" if max(finite) <= 1.0 else "percent"


def infer_proportion_conventions(context_dir: Path) -> dict[str, str]:
    """Observe per-subfamily how sibling structured sources store proportions."""
    verdicts: dict[str, set[str]] = {}

    def _record(column_name: str, values: list[float]) -> None:
        subfamily = _proportion_subfamily(column_name)
        verdict = _classify_proportion_values(values)
        if subfamily and verdict:
            verdicts.setdefault(subfamily, set()).add(verdict)

    for db_path in sorted(context_dir.glob("db/*.sqlite")) + sorted(context_dir.glob("db/*.db")):
        with contextlib.suppress(sqlite3.Error, OSError, ValueError):
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            try:
                tables = [
                    row[0]
                    for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
                ]
                for table in tables:
                    columns = [row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')]
                    for column in columns:
                        if not _proportion_subfamily(column):
                            continue
                        with contextlib.suppress(sqlite3.Error, TypeError, ValueError):
                            rows = conn.execute(
                                f'SELECT "{column}" FROM "{table}" '
                                f'WHERE "{column}" IS NOT NULL '
                                f"LIMIT {PROPORTION_SAMPLE_LIMIT}"
                            ).fetchall()
                            values: list[float] = []
                            for (raw,) in rows:
                                with contextlib.suppress(TypeError, ValueError):
                                    values.append(float(raw))
                            _record(column, values)
            finally:
                conn.close()

    for csv_file in sorted(context_dir.glob("csv/*.csv")):
        with (
            contextlib.suppress(OSError, csv.Error, UnicodeDecodeError),
            csv_file.open(encoding="utf-8", errors="ignore") as f,
        ):
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                continue
            targets = [i for i, name in enumerate(header) if _proportion_subfamily(name)]
            if not targets:
                continue
            samples: dict[int, list[float]] = {i: [] for i in targets}
            for row_num, row in enumerate(reader):
                if row_num >= PROPORTION_SAMPLE_LIMIT:
                    break
                for i in targets:
                    if i < len(row) and row[i].strip():
                        with contextlib.suppress(ValueError):
                            samples[i].append(float(row[i].strip().replace(",", "")))
            for i in targets:
                _record(header[i], samples[i])

    return {
        subfamily: next(iter(votes)) for subfamily, votes in verdicts.items() if len(votes) == 1
    }


def apply_proportion_convention(
    field_units: dict[str, str],
    conventions: dict[str, str] | None,
) -> None:
    """Align %-sourced fields with their subfamily's storage convention, in place."""
    if not conventions:
        return
    for field, source_unit in list(field_units.items()):
        if field.startswith("_") or str(source_unit) != "%":
            continue
        subfamily = _proportion_subfamily(field)
        convention = conventions.get(subfamily) if subfamily else None
        if convention == "ratio":
            field_units[f"_target_{field}"] = "ratio"
            field_units[f"_factor_{field}"] = "0.01"
        elif convention == "percent":
            field_units.pop(f"_target_{field}", None)
            field_units.pop(f"_factor_{field}", None)
