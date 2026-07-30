"""Shared helpers for parsing ETL governance metadata from ``knowledge.md``."""

from __future__ import annotations

import re

from agents.etl._constants import KM_FIELD_CELL, KM_HEADING, KM_TABLE_SEPARATOR


def _km_section_table_fields(lines: list[str], start: int, end: int) -> dict[str, str]:
    """Parse ``| column | definition |`` table rows inside one km section."""
    fields: dict[str, str] = {}
    in_table = False
    header_seen = False
    col_idx = 0
    for ln in lines[start:end]:
        stripped = ln.strip()
        if not stripped.startswith("|"):
            in_table, header_seen = False, False
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if cells and all(KM_TABLE_SEPARATOR.fullmatch(c) for c in cells if c):
            in_table = True
            continue
        if not in_table and not header_seen:
            header_seen = True
            col_idx = next(
                (k for k, c in enumerate(cells) if re.search(r"column|field|字段|列", c, re.I)),
                0,
            )
            continue
        if not in_table or col_idx >= len(cells):
            continue
        cell = cells[col_idx].strip().strip("`").strip()
        if KM_FIELD_CELL.fullmatch(cell):
            fields[cell] = cells[col_idx + 1].strip() if col_idx + 1 < len(cells) else ""
    return fields


def _km_table_rows_by_stem(lines: list[str], prose_stem: str) -> dict[str, str]:
    """Extract fields from table rows whose first column matches *prose_stem*.

    Handles knowledge.md formats where the table name lives inside the table
    body (e.g. ``| `mf_mainfinancialindexq` | `enddate` | 报告期截止日期 |``)
    rather than in the section heading.
    """
    fields: dict[str, str] = {}
    stem_pattern = re.compile(rf"`{re.escape(prose_stem)}`|\b{re.escape(prose_stem)}\b")
    in_table = False
    for ln in lines:
        stripped = ln.strip()
        if not stripped.startswith("|"):
            in_table = False
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if cells and all(KM_TABLE_SEPARATOR.fullmatch(c) for c in cells if c):
            in_table = True
            continue
        if not in_table:
            continue
        if len(cells) >= 3 and stem_pattern.search(cells[0]):
            field = cells[1].strip().strip("`").strip()
            defn = cells[2].strip()
            if field and KM_FIELD_CELL.fullmatch(field):
                fields[field] = defn
    return fields


def km_table_fields(km_text: str, prose_stem: str) -> dict[str, str]:
    """Deterministically parse the governance field table for ``prose_stem``.

    First tries heading-level matching (the stem appears in a section heading).
    Falls back to row-level matching (the stem appears in the first column of
    a markdown table row, e.g. ``| `table_name` | `field` | definition |``).
    """
    lines = km_text.splitlines()
    starts: list[tuple[int, int]] = []
    for i, ln in enumerate(lines):
        m = KM_HEADING.match(ln)
        if m and re.search(rf"`{re.escape(prose_stem)}`|\b{re.escape(prose_stem)}\b", m.group(2)):
            starts.append((i, len(m.group(1))))
    for start, level in starts:
        end = len(lines)
        for j in range(start + 1, len(lines)):
            m = KM_HEADING.match(lines[j])
            if m and len(m.group(1)) <= level:
                end = j
                break
        fields = _km_section_table_fields(lines, start, end)
        if fields:
            return fields
    return _km_table_rows_by_stem(lines, prose_stem)
