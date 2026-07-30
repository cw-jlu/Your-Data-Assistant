"""Unit tests for the structured Record/EntityTable intermediate representation.

用例直接取自 2026-07 ETL 审查的复现集：前导空格键、PK 非首列、行内重复键、
`~` 模糊标记出带/回注、未匹配键保留、`ID:` 无 schema 路径、CRLF、`=` 分隔、
rejects 分类、sanitize_llm_value、atomic_write_text 覆盖语义。
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

import pytest

from agents.etl._record import (
    EntityTable,
    Record,
    atomic_write_text,
    normalize_records,
    parse_kv_text,
    records_to_rows,
    sanitize_llm_value,
    table_to_kv_text,
    to_csv_text,
)

_COLS = ["record_id", "name", "score"]


def test_parse_pk_not_first_column_with_leading_space_keys() -> None:
    """`| record_id: 1` 的键带前导空格，PK 非首列时记录仍须被识别。"""
    table = parse_kv_text(
        "name: Alice | record_id: 1 | score: 10\nname: Bob | record_id: 2 | score: 20",
        columns=_COLS,
        primary_key="record_id",
        anchor_keys=["record_id"],
    )

    assert len(table.records) == 2
    assert table.rejects == []
    assert table.records[0].pk == "1"
    assert table.records[0].fields == {"name": "Alice", "record_id": "1", "score": "10"}


def test_parse_canonicalizes_key_spelling_to_schema() -> None:
    table = parse_kv_text(
        "Record-ID: 7 | NAME: X",
        columns=_COLS,
        primary_key="record_id",
    )

    assert table.records[0].fields == {"record_id": "7", "name": "X"}
    assert table.records[0].provenance == {"record_id": "parse", "name": "parse"}


def test_parse_duplicate_key_keeps_first_value_and_counts_conflict() -> None:
    table = parse_kv_text(
        "record_id: 1 | score: 10 | score: 99 | name: A | name: A",
        columns=_COLS,
        primary_key="record_id",
    )

    record = table.records[0]
    assert record.fields["score"] == "10"  # first non-placeholder wins
    assert table.conflicts == 1  # 同值重复不计冲突
    assert table.stats()["conflicts"] == 1


def test_parse_duplicate_key_fills_initial_empty_value() -> None:
    table = parse_kv_text(
        "record_id: 1 | score: | score: 10",
        columns=_COLS,
        primary_key="record_id",
    )

    record = table.records[0]
    assert record.fields["score"] == "10"
    assert table.conflicts == 1


def test_parse_duplicate_key_fills_initial_placeholder_and_preserves_approx() -> None:
    table = parse_kv_text(
        "record_id: 1 | score: none | score: ~6970000",
        columns=_COLS,
        primary_key="record_id",
    )

    record = table.records[0]
    assert record.fields["score"] == "6970000"
    assert record.approx == {"score"}
    assert table.conflicts == 1


def test_parse_strips_approx_tag_into_record_approx() -> None:
    table = parse_kv_text(
        "record_id: 1 | score: ~6970000 | name: X",
        columns=_COLS,
        primary_key="record_id",
    )

    record = table.records[0]
    assert record.fields["score"] == "6970000"  # 表内存裸值
    assert record.approx == {"score"}


def test_records_to_rows_reinjects_approx_prefix() -> None:
    """~ 出带存储、序列化时回注——repair_numeric_identities 接口不变。"""
    table = parse_kv_text(
        "record_id: 1 | score: ～123 | name: X",
        columns=_COLS,
        primary_key="record_id",
    )

    header, rows = records_to_rows(table)

    assert header == _COLS
    assert rows == [["1", "X", "~123"]]


def test_parse_keeps_unmatched_keys_until_projection() -> None:
    table = parse_kv_text(
        "record_id: 1 | mystery_field: 42",
        columns=_COLS,
        primary_key="record_id",
    )

    assert table.records[0].fields["mystery_field"] == "42"

    header, rows = records_to_rows(table)
    assert header == _COLS
    assert rows == [["1", "", ""]]  # 投影时才丢未匹配键


def test_parse_no_schema_uses_entity_line_convention() -> None:
    """无 schema 时沿用 `ID:` 行首约定，pk 键名为 ID。"""
    table = parse_kv_text(
        "ID: 5 | entity_name: X\nnot an entity: 3",
        columns=[],
        primary_key=None,
    )

    assert len(table.records) == 1
    assert table.records[0].pk == "5"
    assert [r.reason for r in table.rejects] == ["no_pk"]


def test_parse_crlf_input() -> None:
    table = parse_kv_text(
        "record_id: 1 | name: A\r\nrecord_id: 2 | name: B\r\n",
        columns=_COLS,
        primary_key="record_id",
    )

    assert [r.pk for r in table.records] == ["1", "2"]
    assert table.records[1].fields["name"] == "B"


def test_parse_equals_separator_tolerance() -> None:
    table = parse_kv_text(
        "record_id=3 | name=Carol | score: 7",
        columns=_COLS,
        primary_key="record_id",
    )

    assert table.records[0].fields == {"record_id": "3", "name": "Carol", "score": "7"}


def test_parse_equals_separator_preserves_colon_in_value() -> None:
    table = parse_kv_text(
        "record_id=4 | name=https://example.com/a:b | score=8",
        columns=_COLS,
        primary_key="record_id",
    )

    assert table.records[0].fields == {
        "record_id": "4",
        "name": "https://example.com/a:b",
        "score": "8",
    }


def test_parse_skips_markdown_headings_and_classifies_data_rejects() -> None:
    table = parse_kv_text(
        "\n".join(
            [
                "record_id: 1 | name: A",
                "## Section heading",  # trace/section metadata, not malformed data
                "name: orphan | score: 5",  # KV 但缺 pk
                "",  # 空行静默跳过
            ]
        ),
        columns=_COLS,
        primary_key="record_id",
    )

    assert len(table.records) == 1
    assert [(r.reason, r.line) for r in table.rejects] == [
        ("no_pk", "name: orphan | score: 5"),
    ]


def test_parse_blank_pk_value_yields_none_pk() -> None:
    table = parse_kv_text(
        "record_id:  | name: NoId",
        columns=_COLS,
        primary_key="record_id",
    )

    assert len(table.records) == 1  # 键在即实体行（空值留给 pre_merge 处理）
    assert table.records[0].pk is None


def test_stats_counts_records_cells_rejects() -> None:
    table = parse_kv_text(
        "record_id: 1 | name: A | score: \nnarrative line",
        columns=_COLS,
        primary_key="record_id",
    )

    assert table.stats() == {
        "records": 1,
        "nonempty_cells": 2,
        "rejects": 1,
        "conflicts": 0,
    }


def test_records_to_rows_drops_all_empty_rows() -> None:
    table = EntityTable(
        columns=_COLS,
        primary_key="record_id",
        anchor_keys=["record_id"],
        field_types={},
        records=[Record(fields={"record_id": "1"}), Record(fields={"name": ""})],
        rejects=[],
    )

    _header, rows = records_to_rows(table)

    assert rows == [["1", "", ""]]


def test_normalize_records_blanks_null_placeholders_before_projection() -> None:
    table = parse_kv_text(
        "\n".join(
            [
                "record_id: 1 | name: null | score: 10",
                "record_id: null | name: null | score: null",
            ]
        ),
        columns=_COLS,
        primary_key="record_id",
        anchor_keys=["record_id"],
    )

    normalize_records(table)
    _header, rows = records_to_rows(table)

    assert rows == [["1", "", "10"]]


def test_to_csv_text_quotes_embedded_delimiters() -> None:
    text = to_csv_text(["a", "b"], [['x,"y', "line1\nline2"]])

    parsed = list(csv.reader(io.StringIO(text)))
    assert parsed == [["a", "b"], ['x,"y', "line1\nline2"]]


def test_table_to_kv_text_renders_schema_and_extra_keys() -> None:
    table = parse_kv_text(
        "record_id: 1 | score: ~5 | extra: e",
        columns=_COLS,
        primary_key="record_id",
    )

    text = table_to_kv_text(table)

    assert text == "record_id: 1 | name:  | score: ~5 | extra: e"


def test_table_to_kv_text_without_schema_uses_field_order() -> None:
    table = parse_kv_text("ID: 9 | foo: bar", columns=[], primary_key=None)

    assert table_to_kv_text(table) == "ID: 9 | foo: bar"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (r"\1", r"\1"),  # regex 组引用按字面存储
        ("line1\nline2", "line1 line2"),  # 换行折叠——不再劈断 KV 行
        ("val\x00\x08ue", "val ue"),  # 控制字符
        ("  spaced   out  ", "spaced out"),
    ],
)
def test_sanitize_llm_value(raw: str, expected: str) -> None:
    assert sanitize_llm_value(raw) == expected


def test_sanitize_llm_value_truncates_overlong_and_reasoning_leak() -> None:
    assert len(sanitize_llm_value("x" * 2000)) == 500
    assert sanitize_llm_value("123 However, the text implies otherwise") == "123"


def test_atomic_write_text_overwrites_existing_file(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    target = out_dir / "out.csv"
    target.write_text("old content", encoding="utf-8")

    atomic_write_text(target, "new content")

    assert target.read_text(encoding="utf-8") == "new content"
    assert list(out_dir.iterdir()) == [target]  # 无残留 tmp 文件
