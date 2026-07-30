"""Additive-identity mining and approximate-numeral repair."""

from __future__ import annotations

import csv
import logging
import math
from pathlib import Path

from agents.etl._constants import (
    MIN_IDENTITY_SUPPORT,
    NUMERIC_EQ_ABS,
    NUMERIC_EQ_REL,
    NUMERIC_SCALAR,
)
from agents.etl._types import is_placeholder, split_approx_tag

logger = logging.getLogger(__name__)


def _numbers_close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=NUMERIC_EQ_REL, abs_tol=NUMERIC_EQ_ABS)


def _parse_cell_number(raw: str) -> tuple[float | None, bool]:
    """Parse a CSV cell into ``(value, is_approx)``; ``(None, False)`` if non-numeric."""
    bare, approx = split_approx_tag(raw)
    if not bare:
        return None, False
    try:
        return float(bare.replace(",", "")), approx
    except ValueError:
        return None, False


def _stated_granularity(raw: str) -> float:
    """The precision a stated numeral carries, from its own digits.

    ``6970000`` (六百九十七万) was stated at 万 granularity → 1e4; a value
    with no trailing zeros, a decimal point, or an exponent carries full
    precision and returns a granularity that disables coarse matching.
    """
    bare, _approx = split_approx_tag(raw)
    bare = bare.replace(",", "")
    if not bare or not NUMERIC_SCALAR.match(bare):
        return 0.0
    if "." in bare or "e" in bare.lower():
        return 0.0
    digits = bare.lstrip("+-")
    stripped = digits.rstrip("0")
    if not stripped:  # the value 0 itself
        return 0.0
    zeros = len(digits) - len(stripped)
    return float(10**zeros) if zeros else 0.0


def _coarse_consistent(stated: float, raw: str, solved: float) -> bool:
    """Whether ``solved`` equals the stated value at the value's own granularity.

    This is the acceptance rule for replacing a stated-approximate number with
    an identity-derived exact one: the stated numeral's trailing zeros define
    the band it claims ("约六百九十七万" claims 万-precision), and the solved
    value must quantize back to the same numeral — no tunable tolerance.
    """
    granularity = _stated_granularity(raw)
    if granularity <= 1.0:
        return False
    if _numbers_close(solved, stated):
        return False
    return round(solved / granularity) == round(stated / granularity)


def _parse_numeric_table(
    header: list[str],
    rows: list[list[str]],
) -> list[list[tuple[float | None, bool]] | None]:
    """Parse cells per column as ``(value, is_approx)``; None for non-numeric columns.

    A column qualifies as numeric when every non-empty cell parses as a number
    (approx tags allowed) and at least one cell is non-empty.
    """
    parsed: list[list[tuple[float | None, bool]] | None] = []
    for j in range(len(header)):
        col: list[tuple[float | None, bool]] = []
        numeric_ok = True
        nonempty = 0
        for row in rows:
            raw = row[j].strip()
            if not raw:
                col.append((None, False))
                continue
            num, approx = _parse_cell_number(raw)
            if num is None:
                numeric_ok = False
                break
            nonempty += 1
            col.append((num, approx))
        parsed.append(col if numeric_ok and nonempty else None)
    return parsed


def _blamable_term(stated: float, raw: str, solved: float, support_grains: list[float]) -> bool:
    """Whether an untagged violating term may absorb the row's discrepancy.

    Two independent conditions, both data-derived:

    1. coarse consistency — the identity-solved value quantizes back to the
       stated numeral at the numeral's own granularity;
    2. granularity anomaly — the numeral is strictly coarser than EVERY
       numeral this column shows on identity-supporting rows. In a column
       where round numbers are the norm (share lots trade in 万-multiples),
       roundness carries no approximation signal and never implicates a cell;
       in a full-precision column, a uniquely round numeral is exactly the
       fuzzy-numeral signature ("约六百九十七万" → 6970000).
    """
    if not _coarse_consistent(stated, raw, solved):
        return False
    if len(support_grains) < 2:
        return False
    return _stated_granularity(raw) > max(support_grains)


def _scan_identity(
    parsed: list[list[tuple[float | None, bool]] | None],
    rows: list[list[str]],
    triple: tuple[int, int, int],
) -> tuple[int, int, bool, dict[int, list[float]]]:
    """Evaluate one candidate triple ``a == b + c`` over all rows.

    Pass 1 splits fully-exact (untagged, all-three-present) rows into
    supporters and violators and collects the supporters' stated
    granularities per column. Pass 2 classifies each violator: repairable
    (some term is blamable per `_blamable_term`) or a hard violation.

    Returns ``(support, suspects, hard_violation, support_grains)``.
    """
    a, b, c = triple
    col_a, col_b, col_c = parsed[a], parsed[b], parsed[c]
    assert col_a is not None and col_b is not None and col_c is not None

    supporters: list[int] = []
    violators: list[int] = []
    for i in range(len(rows)):
        va, ta = col_a[i]
        vb, tb = col_b[i]
        vc, tc = col_c[i]
        if va is None or vb is None or vc is None or ta or tb or tc:
            continue
        if _numbers_close(va, vb + vc):
            supporters.append(i)
        else:
            violators.append(i)

    grains = {j: [_stated_granularity(rows[i][j].strip()) for i in supporters] for j in triple}

    suspects = 0
    for i in violators:
        va = col_a[i][0]
        vb = col_b[i][0]
        vc = col_c[i][0]
        assert va is not None and vb is not None and vc is not None
        blamable = [
            j
            for j, stated, solved in ((a, va, vb + vc), (b, vb, va - vc), (c, vc, va - vb))
            if _blamable_term(stated, rows[i][j].strip(), solved, grains[j])
        ]
        if not blamable:
            return len(supporters), suspects, True, grains
        suspects += 1
    return len(supporters), suspects, False, grains


def mine_additive_identities(
    header: list[str],
    rows: list[list[str]],
) -> list[tuple[int, int, int]]:
    """Mine column triples satisfying ``col_a == col_b + col_c`` row-wise.

    Deterministic invariant mining over numeric columns. A triple is accepted
    when its fully-exact rows support the identity (within float-dust
    tolerance) on at least MIN_IDENTITY_SUPPORT rows, exact support strictly
    dominates the repair candidates, and NO exact row violates it beyond
    repair — a violating row is forgiven only when some term is blamable
    under `_blamable_term` (coarse-consistent solved value AND column-level
    granularity anomaly).

    Returns ``(a, b, c)`` column-index triples with ``b < c``.
    """
    parsed = _parse_numeric_table(header, rows)
    numeric_cols = [j for j in range(len(header)) if parsed[j] is not None]
    identities: list[tuple[int, int, int]] = []
    for a in numeric_cols:
        for bi, b in enumerate(numeric_cols):
            if b == a:
                continue
            for c in numeric_cols[bi + 1 :]:
                if c in (a, b):
                    continue
                support, suspects, hard_violation, _grains = _scan_identity(parsed, rows, (a, b, c))
                if not hard_violation and support >= MIN_IDENTITY_SUPPORT and support > suspects:
                    identities.append((a, b, c))
    return identities


def _format_repaired_number(value: float) -> str:
    if math.isclose(value, round(value), rel_tol=0, abs_tol=NUMERIC_EQ_ABS):
        return str(round(value))
    return format(value, ".12g")


def repair_numeric_identities(
    header: list[str],
    rows: list[list[str]],
) -> tuple[list[list[str]], list[tuple[int, str, str, str]]]:
    """Recompute approximate numerals from mined additive identities.

    Two repair triggers, both solved from the other two terms of a validated
    identity and both accepted only under granularity consistency
    (`_coarse_consistent` — the solved value must quantize back to the stated
    numeral at the numeral's own precision):

    - cells tagged approximate by compression (``~6970000``) — the tag is
      explicit evidence, so plain granularity consistency suffices;
    - untagged cells whose row violates a validated identity and which are
      the UNIQUE blamable term of that row per `_blamable_term`
      (coarse-consistent AND anomalously round for the column — compression
      missed the tag);
    - empty cells that are the ONLY missing term of a validated identity in
      their row (compression dropped a stated value entirely). A validated
      identity implies definitional coupling, so the missing term is
      mathematically determined. Rows missing 2+ terms — the genuine
      source-null pattern ("转让方信息缺失") — can never be solved and are
      left untouched; null patterns are data.

    A cell repairs only when all candidate identities agree on a single
    solved value. All approx tags are stripped from the returned rows
    regardless of repair outcome. Returns ``(rows, repairs)`` with repairs
    as ``(row_index, column, old, new)``; inputs are not mutated.
    """
    rows = [list(row) for row in rows]
    identities = mine_additive_identities(header, rows)

    parsed_snapshot = _parse_numeric_table(header, rows)
    identity_grains = {
        identity: _scan_identity(parsed_snapshot, rows, identity)[3] for identity in identities
    }

    by_col: dict[int, list[tuple[int, int, int]]] = {}
    for a, b, c in identities:
        by_col.setdefault(a, []).append((a, b, c))
        by_col.setdefault(b, []).append((a, b, c))
        by_col.setdefault(c, []).append((a, b, c))

    def _solve(identity: tuple[int, int, int], target: int, row: list[str]) -> float | None:
        a, b, c = identity
        nums: dict[int, float] = {}
        for j in (a, b, c):
            if j == target:
                continue
            num, approx = _parse_cell_number(row[j].strip())
            if num is None or approx:
                return None
            nums[j] = num
        if target == a:
            return nums[b] + nums[c]
        if target == b:
            return nums[a] - nums[c]
        return nums[a] - nums[b]

    repairs: list[tuple[int, str, str, str]] = []
    for i, row in enumerate(rows):
        proposals: dict[int, list[float]] = {}

        for j in range(len(header)):
            raw = row[j].strip()
            num, approx = _parse_cell_number(raw)
            if num is None or not approx:
                continue
            for identity in by_col.get(j, []):
                solved = _solve(identity, j, row)
                if solved is None:
                    continue
                if _coarse_consistent(num, raw, solved) or _numbers_close(solved, num):
                    proposals.setdefault(j, []).append(solved)

        for identity in identities:
            a, b, c = identity
            empty_cols: list[int] = []
            cells: list[tuple[int, float | None, bool]] = []
            unusable = False
            for j in (a, b, c):
                raw = row[j].strip()
                if not raw:
                    empty_cols.append(j)
                    continue
                num, approx = _parse_cell_number(raw)
                if num is None or approx:
                    unusable = True
                    break
                cells.append((j, num, approx))
            if unusable:
                continue

            if len(empty_cols) == 1:
                solved_missing = _solve(identity, empty_cols[0], row)
                if solved_missing is not None:
                    proposals.setdefault(empty_cols[0], []).append(solved_missing)
                continue
            if empty_cols:
                continue

            va, vb, vc = cells[0][1], cells[1][1], cells[2][1]
            assert va is not None and vb is not None and vc is not None
            if _numbers_close(va, vb + vc):
                continue
            grains = identity_grains[identity]
            blamable = [
                (j, solved)
                for j, stated, solved in ((a, va, vb + vc), (b, vb, va - vc), (c, vc, va - vb))
                if _blamable_term(stated, row[j].strip(), solved, grains[j])
            ]
            if len(blamable) == 1:
                j, solved = blamable[0]
                proposals.setdefault(j, []).append(solved)

        for j, solved_values in proposals.items():
            distinct: list[float] = []
            for value in solved_values:
                if not any(_numbers_close(value, seen) for seen in distinct):
                    distinct.append(value)
            if len(distinct) != 1:
                continue
            old = row[j].strip()
            stated, _approx = _parse_cell_number(old)
            if stated is not None and _numbers_close(distinct[0], stated):
                continue
            new = _format_repaired_number(distinct[0])
            row[j] = new
            repairs.append((i, header[j], old, new))

    for row in rows:
        for j in range(len(header)):
            raw = row[j].strip()
            num, approx = _parse_cell_number(raw)
            if num is not None and approx:
                bare, _ = split_approx_tag(raw)
                row[j] = bare

    return rows, repairs


def validate_csv(csv_path: Path) -> tuple[list[str], int] | None:
    """Read a CSV and return (header, row_count), or None if invalid.

    Rejects (returns None) instead of raising so a truncated or corrupted
    cache file triggers re-extraction rather than poisoning every run:
    - undecodable bytes (e.g. a multi-byte char cut in half by truncation)
    - empty column names in the header
    - ragged data rows (width != header width, the truncated-tail signature)
    - all-placeholder data rows (no entity or measured value survived)
    """
    if not csv_path.is_file():
        return None
    try:
        with csv_path.open(encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                return None
            if any(not col.strip() for col in header):
                return None
            row_count = 0
            for row in reader:
                if len(row) != len(header):
                    return None
                if not any(cell.strip() and not is_placeholder(cell) for cell in row):
                    return None
                row_count += 1
        return (header, row_count) if row_count > 0 else None
    except (csv.Error, OSError, UnicodeDecodeError):
        return None
