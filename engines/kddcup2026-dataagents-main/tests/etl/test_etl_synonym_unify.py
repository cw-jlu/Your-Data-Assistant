"""Behavior tests for value-split synonym column unification.

Covers the deterministic merge gates (`apply_synonym_merges`), adjudication
response parsing, and the in-memory table repair orchestration
(`unify_table_synonyms`) with a stubbed adapter.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agents.etl._merge import apply_synonym_merges, column_fill_counts
from agents.etl._reconcile import _parse_alias_response, unify_table_synonyms
from agents.llm.types import ModelResponse


def _table(
    header: list[str],
    rows: list[list[str]],
) -> tuple[list[str], list[list[str]]]:
    return list(header), [list(r) for r in rows]


# ---------------------------------------------------------------------------
# apply_synonym_merges — deterministic gates
# ---------------------------------------------------------------------------


def test_merge_fills_empty_canonical_and_drops_source() -> None:
    header, rows = _table(
        ["record_id", "sumbeforetran", "transferor_shares_before"],
        [["29", "", "2441732"], ["30", "", "5469268"], ["31", "", ""]],
    )

    new_header, new_rows, applied = apply_synonym_merges(
        header,
        rows,
        [("transferor_shares_before", "sumbeforetran")],
    )

    assert applied == [("transferor_shares_before", "sumbeforetran")]
    assert new_header == ["record_id", "sumbeforetran"]
    assert [r[1] for r in new_rows] == ["2441732", "5469268", ""]
    assert len(new_rows) == 3


def test_merge_unions_row_disjoint_values() -> None:
    header, rows = _table(
        ["id", "canon", "alias"],
        [["1", "10", ""], ["2", "", "20"], ["3", "", ""]],
    )

    _new_header, new_rows, applied = apply_synonym_merges(header, rows, [("alias", "canon")])

    assert applied == [("alias", "canon")]
    assert [r[1] for r in new_rows] == ["10", "20", ""]


def test_merge_skips_pair_with_row_conflict() -> None:
    header, rows = _table(
        ["id", "canon", "alias"],
        [["1", "10", "11"], ["2", "", "20"]],
    )

    new_header, new_rows, applied = apply_synonym_merges(header, rows, [("alias", "canon")])

    assert applied == []
    assert new_header == header
    assert new_rows == rows


@pytest.mark.parametrize(
    ("canon_val", "alias_val"),
    [
        ("1.55", "1.550"),
        ("2441732", "2441732.0"),
        ("0.50", ".5"),
    ],
)
def test_merge_treats_numerically_equal_overlap_as_agreement(
    canon_val: str, alias_val: str
) -> None:
    header, rows = _table(
        ["id", "canon", "alias"],
        [["1", canon_val, alias_val], ["2", "", "20"]],
    )

    _new_header, new_rows, applied = apply_synonym_merges(header, rows, [("alias", "canon")])

    assert applied == [("alias", "canon")]
    # Overlapping cell keeps the canonical spelling; disjoint cell moves over.
    assert [r[1] for r in new_rows] == [canon_val, "20"]


def test_merge_two_sources_one_canonical_second_conflicts_after_first() -> None:
    # big fills rows 1-2; small claims row 1 with a DIFFERENT value: once big
    # is merged (larger fill goes first), small must be vetoed by conflict.
    header, rows = _table(
        ["id", "canon", "big", "small"],
        [["1", "", "10", "99"], ["2", "", "20", ""]],
    )

    new_header, new_rows, applied = apply_synonym_merges(
        header,
        rows,
        [("small", "canon"), ("big", "canon")],
    )

    assert applied == [("big", "canon")]
    assert "small" in new_header
    assert [r[new_header.index("canon")] for r in new_rows] == ["10", "20"]


def test_merge_never_touches_protected_columns() -> None:
    header, rows = _table(
        ["record_id", "canon", "alias"],
        [["1", "", "10"]],
    )

    for pair in [("alias", "record_id"), ("record_id", "canon")]:
        new_header, new_rows, applied = apply_synonym_merges(
            header, rows, [pair], protected={"record_id"}
        )
        assert applied == []
        assert new_header == header
        assert new_rows == rows


def test_merge_allows_case_only_protected_column_fold() -> None:
    header, rows = _table(
        ["record_id", "secucode", "SecuCode"],
        [["1", "", "601908"], ["2", "601908", "601908"]],
    )

    new_header, new_rows, applied = apply_synonym_merges(
        header,
        rows,
        [("SecuCode", "secucode")],
        protected={"record_id", "secucode"},
    )

    assert applied == [("SecuCode", "secucode")]
    assert new_header == ["record_id", "secucode"]
    assert [row[1] for row in new_rows] == ["601908", "601908"]


def test_merge_skips_all_empty_source_and_missing_columns() -> None:
    header, rows = _table(
        ["id", "canon", "alias"],
        [["1", "", ""]],
    )

    for pair in [("alias", "canon"), ("ghost", "canon"), ("alias", "ghost")]:
        new_header, _new_rows, applied = apply_synonym_merges(header, rows, [pair])
        assert applied == []
        assert new_header == header


def test_column_fill_counts_ignores_whitespace_cells() -> None:
    counts = column_fill_counts(["a", "b"], [["x", " "], ["", "y"], ["z", ""]])
    assert counts == {"a": 2, "b": 1}


# ---------------------------------------------------------------------------
# _parse_alias_response — format tolerance, unknown-name rejection
# ---------------------------------------------------------------------------

_DISCOVERED = ["transferor_shares_before", "transferor_pct_before", "tran_date"]
_CANONICALS = ["sumbeforetran", "pctbeforetran"]


@pytest.mark.parametrize(
    "text",
    [
        # one per line
        "transferor_shares_before=sumbeforetran\n"
        "transferor_pct_before=pctbeforetran\n"
        "tran_date=DISTINCT",
        # comma separated
        "transferor_shares_before=sumbeforetran, transferor_pct_before=pctbeforetran, tran_date=DISTINCT",
        # echoed prefix + mixed case + trailing period
        "ALIASES: Transferor_Shares_Before=SumBeforeTran; transferor_pct_before=pctbeforetran.; tran_date=distinct",
    ],
)
def test_parse_alias_response_format_tolerance(text: str) -> None:
    mapping = _parse_alias_response(text, _DISCOVERED, _CANONICALS)
    assert mapping == {
        "transferor_shares_before": "sumbeforetran",
        "transferor_pct_before": "pctbeforetran",
    }


def test_parse_alias_response_rejects_unknown_names() -> None:
    text = (
        "made_up_column=sumbeforetran\n"  # unknown discovered → ignored
        "transferor_shares_before=not_a_canonical\n"  # unknown canonical → ignored
        "transferor_pct_before=pctbeforetran"
    )
    mapping = _parse_alias_response(text, _DISCOVERED, _CANONICALS)
    assert mapping == {"transferor_pct_before": "pctbeforetran"}


def test_parse_alias_response_unlisted_discovered_defaults_to_distinct() -> None:
    mapping = _parse_alias_response(
        "transferor_shares_before=sumbeforetran", _DISCOVERED, _CANONICALS
    )
    assert "transferor_pct_before" not in mapping
    assert "tran_date" not in mapping


# ---------------------------------------------------------------------------
# unify_table_synonyms — in-memory orchestration (unit sidecar on disk)
# ---------------------------------------------------------------------------


class _StubAdapter:
    def __init__(self, content: str | Exception) -> None:
        self._content = content
        self.calls = 0

    def complete(self, messages: list[Any]) -> ModelResponse:
        self.calls += 1
        if isinstance(self._content, Exception):
            raise self._content
        return ModelResponse(content=self._content)


_HEADER = [
    "record_id",
    "secucode",
    "sumbeforetran",
    "pctbeforetran",
    "transferor_shares_before",
    "transferor_pct_before",
    "transferee_shares_before",
]

_ROWS = [
    ["29", "300707", "", "", "2441732", "1.55", ""],
    ["30", "300707", "", "", "5469268", "3.48", ""],
    ["48", "300707", "", "", "", "", "1565000"],
]

_FIELD_DEFS = {
    "sumbeforetran": "Total shares held before transfer",
    "pctbeforetran": "Shareholding percentage before transfer",
    "transferor_shares_before": "Transferor's shares before transfer",
    "transferor_pct_before": "Transferor's percentage before transfer",
    "transferee_shares_before": "Transferee's shares before transfer",
}


def _run_unify(
    tmp_path: Path,
    adapter: _StubAdapter,
    rows: list[list[str]] | None = None,
    units: dict[str, str] | None = None,
) -> tuple[list[str], list[list[str]], Path, list[tuple[str, str]]]:
    units_path = tmp_path / "lc_sharetransfer_units.json"
    if units is not None:
        units_path.write_text(json.dumps(units, ensure_ascii=False), encoding="utf-8")
    header, out_rows, applied = unify_table_synonyms(
        adapter,  # type: ignore[arg-type]
        list(_HEADER),
        [list(r) for r in (rows if rows is not None else _ROWS)],
        "lc_sharetransfer",
        ["sumbeforetran", "pctbeforetran", "secucode"],
        ["record_id", "secucode"],
        _FIELD_DEFS,
        "显示300707的出让前持股数量和出让前持股比例",
        units_path,
    )
    return header, out_rows, units_path, applied


def test_unify_repairs_value_split_and_moves_unit_metadata(tmp_path: Path) -> None:
    adapter = _StubAdapter(
        "transferor_shares_before=sumbeforetran\n"
        "transferor_pct_before=pctbeforetran\n"
        "transferee_shares_before=DISTINCT"
    )
    units = {
        "pctbeforetran": "%",
        "transferor_pct_before": "%",
        "_target_pctbeforetran": "ratio",
        "_factor_pctbeforetran": "0.01",
    }

    header, rows, units_path, applied = _run_unify(tmp_path, adapter, units=units)

    assert sorted(applied) == [
        ("transferor_pct_before", "pctbeforetran"),
        ("transferor_shares_before", "sumbeforetran"),
    ]
    assert "transferor_shares_before" not in header
    assert "transferor_pct_before" not in header
    assert "transferee_shares_before" in header  # DISTINCT survives
    sum_idx, pct_idx = header.index("sumbeforetran"), header.index("pctbeforetran")
    assert [r[sum_idx] for r in rows] == ["2441732", "5469268", ""]
    assert [r[pct_idx] for r in rows] == ["1.55", "3.48", ""]

    meta = json.loads(units_path.read_text(encoding="utf-8"))
    assert meta["_factor_pctbeforetran"] == "0.01"
    assert meta["pctbeforetran"] == "%"
    assert "transferor_pct_before" not in meta


def test_unify_with_complete_canonicals_makes_no_llm_call(tmp_path: Path) -> None:
    adapter = _StubAdapter("transferor_shares_before=sumbeforetran")
    filled_rows = [
        ["29", "300707", "2441732", "1.55", "2441732", "1.55", ""],
        ["30", "300707", "5469268", "3.48", "5469268", "3.48", ""],
    ]

    header, _rows, _units_path, applied = _run_unify(tmp_path, adapter, rows=filled_rows)

    assert applied == []
    assert adapter.calls == 0
    assert header == _HEADER


def test_unify_folds_case_only_governance_duplicates_without_llm(tmp_path: Path) -> None:
    units_path = tmp_path / "qt_dailyquote_units.json"
    units_path.write_text(
        json.dumps({"SecuCode": "code", "_target_TradingDay": "date"}, ensure_ascii=False),
        encoding="utf-8",
    )
    adapter = _StubAdapter("SHOULD_NOT_BE_USED")

    header, rows, applied = unify_table_synonyms(
        adapter,  # type: ignore[arg-type]
        ["record_id", "secucode", "SecuCode", "tradingday", "TradingDay", "turnoverdeals"],
        [
            ["2706", "", "601908", "", "2021-01-08", "122610"],
            ["2718", "601908", "601908", "2021-01-26", "2021-01-26", "71041"],
        ],
        "qt_dailyquote",
        ["secucode", "tradingday", "turnoverdeals"],
        ["record_id", "secucode", "tradingday"],
        {},
        "show trading volume by trading day",
        units_path,
    )

    assert applied == [("SecuCode", "secucode"), ("TradingDay", "tradingday")]
    assert adapter.calls == 0
    assert header == ["record_id", "secucode", "tradingday", "turnoverdeals"]
    assert rows == [
        ["2706", "601908", "2021-01-08", "122610"],
        ["2718", "601908", "2021-01-26", "71041"],
    ]
    meta = json.loads(units_path.read_text(encoding="utf-8"))
    assert meta["secucode"] == "code"
    assert meta["_target_tradingday"] == "date"
    assert "SecuCode" not in meta
    assert "_target_TradingDay" not in meta


def test_unify_repairs_partial_split(tmp_path: Path) -> None:
    # values routed under two names across rows: canonical holds row 29,
    # the discovered synonym holds rows 30/48 — union must complete the column
    rows = [
        ["29", "300707", "2441732", "1.55", "", "", ""],
        ["30", "300707", "", "", "5469268", "3.48", ""],
        ["48", "300707", "", "", "3487568", "2.2186", ""],
    ]
    adapter = _StubAdapter(
        "transferor_shares_before=sumbeforetran\n"
        "transferor_pct_before=pctbeforetran\n"
        "transferee_shares_before=DISTINCT"
    )

    header, out_rows, _units_path, applied = _run_unify(tmp_path, adapter, rows=rows)

    assert sorted(applied) == [
        ("transferor_pct_before", "pctbeforetran"),
        ("transferor_shares_before", "sumbeforetran"),
    ]
    sum_idx = header.index("sumbeforetran")
    assert [r[sum_idx] for r in out_rows] == ["2441732", "5469268", "3487568"]


def test_unify_conflicting_overlap_blocks_pair_before_llm(tmp_path: Path) -> None:
    # the only gapped canonical's only candidate DISAGREES on the overlapping
    # row — no structurally viable pair anywhere, so no LLM call is spent
    rows = [
        ["29", "300707", "2441732", "1.55", "9999999", "", ""],
        ["30", "300707", "", "3.48", "", "", ""],
    ]
    adapter = _StubAdapter("transferor_shares_before=sumbeforetran")

    header, _rows, _units_path, applied = _run_unify(tmp_path, adapter, rows=rows)

    assert applied == []
    assert adapter.calls == 0
    assert header == _HEADER


def test_unify_adapter_failure_leaves_table_untouched(tmp_path: Path) -> None:
    adapter = _StubAdapter(RuntimeError("boom"))

    header, rows, _units_path, applied = _run_unify(tmp_path, adapter)

    assert applied == []
    assert header == _HEADER
    assert rows == _ROWS


def test_unify_collision_on_one_canonical_merges_nothing_into_it(tmp_path: Path) -> None:
    adapter = _StubAdapter(
        "transferor_shares_before=sumbeforetran\n"
        "transferee_shares_before=sumbeforetran\n"
        "transferor_pct_before=pctbeforetran"
    )

    header, _rows, _units_path, applied = _run_unify(tmp_path, adapter)

    assert applied == [("transferor_pct_before", "pctbeforetran")]
    assert "transferor_shares_before" in header
    assert "transferee_shares_before" in header


def test_unify_unit_family_mismatch_blocks_merge(tmp_path: Path) -> None:
    adapter = _StubAdapter(
        "transferor_shares_before=sumbeforetran\ntransferor_pct_before=pctbeforetran"
    )
    units = {"PctBeforeTran": "ratio", "TRANSFEROR_PCT_BEFORE": "%"}

    header, _rows, _units_path, applied = _run_unify(tmp_path, adapter, units=units)

    assert applied == [("transferor_shares_before", "sumbeforetran")]
    assert "transferor_pct_before" in header


def test_unify_never_offers_anchor_columns(tmp_path: Path) -> None:
    # secucode is governance-listed AND an anchor; even with an empty anchor
    # column the unifier must not offer or merge identity columns.
    rows = [
        ["29", "", "", "", "2441732", "1.55", ""],
        ["30", "", "", "", "5469268", "3.48", ""],
    ]
    adapter = _StubAdapter(
        "transferor_shares_before=sumbeforetran\n"
        "transferor_pct_before=pctbeforetran\n"
        "transferee_shares_before=DISTINCT"
    )

    header, _rows, _units_path, applied = _run_unify(tmp_path, adapter, rows=rows)

    assert sorted(applied) == [
        ("transferor_pct_before", "pctbeforetran"),
        ("transferor_shares_before", "sumbeforetran"),
    ]
    assert "secucode" in header
