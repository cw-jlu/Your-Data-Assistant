"""Behavior tests for source-grounded, subfamily-scoped proportion conventions."""

from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

import pytest

from agents.etl._units import (
    _classify_proportion_values,
    apply_proportion_convention,
    infer_proportion_conventions,
)


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([0.88, 0.27, 1.0], "ratio"),
        ([88.02, 27.5], "percent"),
        ([-0.052, 0.31], "ratio"),  # signed change rates classify by magnitude
        ([-5.2, 0.3], "percent"),
        ([], None),
        ([float("nan")], None),
    ],
)
def test_classify_proportion_values(values: list[float], expected: str | None) -> None:
    assert _classify_proportion_values(values) == expected


def _make_context(tmp_path: Path) -> Path:
    context = tmp_path / "context"
    (context / "db").mkdir(parents=True)
    (context / "csv").mkdir()
    return context


def _make_db(path: Path, columns: dict[str, list[float]]) -> None:
    conn = sqlite3.connect(path)
    col_defs = ", ".join(f'"{name}" REAL' for name in columns)
    conn.execute(f"CREATE TABLE t (id INTEGER, {col_defs})")
    n_rows = max(len(vals) for vals in columns.values())
    for i in range(n_rows):
        vals = [cols[i] if i < len(cols) else None for cols in columns.values()]
        placeholders = ", ".join("?" for _ in vals)
        conn.execute(f"INSERT INTO t VALUES (?, {placeholders})", [i, *vals])
    conn.commit()
    conn.close()


def test_infer_conventions_are_subfamily_scoped(tmp_path: Path) -> None:
    """task_15 shape: pct columns store fractions while ratio columns store
    per-10-share counts — verdicts must not bleed across subfamilies."""
    context = _make_context(tmp_path)
    _make_db(
        context / "db" / "sub_db.sqlite",
        {
            "AccuPCTOfPled": [0.880238, 0.2703],
            "PCTOfFullShares": [0.6142, 0.1],
            "TransferPlaRatio": [10.0, 3.0],
        },
    )

    assert infer_proportion_conventions(context) == {"pct": "ratio", "ratio": "percent"}


def test_infer_conventions_merge_csv_evidence(tmp_path: Path) -> None:
    context = _make_context(tmp_path)
    with (context / "csv" / "lc_relatedsh.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["code", "HoldingPCT", "note"])
        writer.writerows([["1", "0.8065", "a"], ["2", "0.2750", "b"]])

    assert infer_proportion_conventions(context) == {"pct": "ratio"}


def test_infer_conventions_mixed_subfamily_is_omitted(tmp_path: Path) -> None:
    context = _make_context(tmp_path)
    _make_db(
        context / "db" / "sub_db.sqlite",
        {
            "AccuProportion": [0.5926],
            "Proportion1": [899.7, 100.0],
            "HoldingPCT": [0.8],
        },
    )

    assert infer_proportion_conventions(context) == {"pct": "ratio"}


def test_infer_conventions_without_proportion_columns(tmp_path: Path) -> None:
    context = _make_context(tmp_path)
    _make_db(context / "db" / "sub_db.sqlite", {"amount": [0.5, 2.0]})

    assert infer_proportion_conventions(context) == {}


def test_apply_convention_matches_field_to_own_subfamily() -> None:
    units = {
        "pctbeforetran": "%",
        "transfer_pct_holder": "%",
        "bonus_share_ratio": "%",
        "growth": "%",  # no proportion token → heuristic stays
        "gdp": "亿元",
        "_target_gdp": "万元",
        "_factor_gdp": "10000",
        "_target_growth": "ratio",
        "_factor_growth": "0.01",
    }
    apply_proportion_convention(units, {"pct": "ratio", "ratio": "percent"})

    assert units["_target_pctbeforetran"] == "ratio"
    assert units["_factor_pctbeforetran"] == "0.01"
    assert units["_target_transfer_pct_holder"] == "ratio"
    # ratio-subfamily verdict 'percent' cancels nothing here (no prior target),
    # and never installs a conversion
    assert "_target_bonus_share_ratio" not in units
    # untouched: no-subfamily field keeps its heuristic target, non-% keeps voted
    assert units["_target_growth"] == "ratio"
    assert units["_target_gdp"] == "万元"


def test_apply_convention_percent_cancels_heuristic_targets() -> None:
    units = {
        "reserve_ratio": "%",
        "_target_reserve_ratio": "ratio",
        "_factor_reserve_ratio": "0.01",
    }
    apply_proportion_convention(units, {"ratio": "percent"})

    assert "_target_reserve_ratio" not in units
    assert "_factor_reserve_ratio" not in units


def test_apply_convention_empty_is_noop() -> None:
    units = {"pctbeforetran": "%", "_target_pctbeforetran": "ratio"}
    before = dict(units)
    apply_proportion_convention(units, {})
    apply_proportion_convention(units, None)
    assert units == before
