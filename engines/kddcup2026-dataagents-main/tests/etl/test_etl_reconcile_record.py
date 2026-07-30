"""Record-level malformed-line repair and field-name reconciliation tests."""

from __future__ import annotations

from typing import Any

import pytest

from agents.etl._merge import merge_records
from agents.etl._reconcile import fix_malformed_lines, reconcile_field_names
from agents.etl._record import parse_kv_text, records_to_rows
from agents.llm.types import ModelMessage, ModelResponse


class _ReplyAdapter:
    def __init__(self, reply: str | Exception) -> None:
        self.reply = reply
        self.messages: list[list[ModelMessage]] = []

    def complete(
        self, messages: list[ModelMessage], *, tools: object | None = None, **kwargs: Any
    ) -> ModelResponse:
        del tools, kwargs
        self.messages.append(messages)
        if isinstance(self.reply, Exception):
            raise self.reply
        return ModelResponse(content=self.reply)


def test_fix_malformed_lines_uses_numbered_protocol() -> None:
    table = parse_kv_text(
        "id: 1 | value: ok\nbare malformed value 2",
        columns=["id", "value"],
        primary_key="id",
        anchor_keys=["id"],
    )
    assert [reject.reason for reject in table.rejects] == ["no_kv"]
    adapter = _ReplyAdapter("1) id: 2 | value: fixed")

    recovered = fix_malformed_lines(adapter, table, schema_field_defs=None)

    assert recovered == 1
    assert table.rejects == []
    assert [record.fields for record in table.records] == [
        {"id": "1", "value": "ok"},
        {"id": "2", "value": "fixed"},
    ]
    prompt = adapter.messages[0][1].content
    assert "1) bare malformed value 2" in prompt


def test_fix_malformed_lines_discards_unnumbered_reply() -> None:
    table = parse_kv_text(
        "bad row",
        columns=["id", "value"],
        primary_key="id",
        anchor_keys=["id"],
    )

    recovered = fix_malformed_lines(
        _ReplyAdapter("id: 2 | value: fixed"),
        table,
        schema_field_defs=None,
    )

    assert recovered == 0
    assert table.records == []
    assert [reject.line for reject in table.rejects] == ["bad row"]


def test_reconcile_field_names_renames_case_variants_once() -> None:
    table = parse_kv_text(
        "id: 1 | Alias: 10\nid: 2 | alias: 20\nid: 3 | known: ok",
        columns=["id", "known"],
        primary_key="id",
        anchor_keys=["id"],
    )

    applied = reconcile_field_names(
        _ReplyAdapter("Alias=known"),
        table,
        schema_field_defs={"known": "canonical value"},
    )

    assert applied == {"Alias": "known", "alias": "known"}
    assert [record.fields.get("known") for record in table.records] == ["10", "20", "ok"]
    assert all(
        "Alias" not in record.fields and "alias" not in record.fields for record in table.records
    )


@pytest.mark.parametrize("target_value", ["", "none", "- (placeholder)"])
def test_reconcile_field_names_fills_empty_existing_target(target_value: str) -> None:
    table = parse_kv_text(
        f"id: 1 | known: {target_value} | Alias: 10",
        columns=["id", "known"],
        primary_key="id",
        anchor_keys=["id"],
    )

    applied = reconcile_field_names(
        _ReplyAdapter("Alias=known"),
        table,
        schema_field_defs=None,
    )

    assert applied == {"Alias": "known"}
    assert table.records[0].fields["known"] == "10"
    assert "Alias" not in table.records[0].fields
    assert records_to_rows(table) == (["id", "known"], [["1", "10"]])


def test_reconcile_field_names_refreshes_primary_key_before_pre_merge() -> None:
    table = parse_kv_text(
        "id:  | alias_id: 1 | name: A\nid: 2 | name: B",
        columns=["id", "name"],
        primary_key="id",
        anchor_keys=["id"],
    )

    applied = reconcile_field_names(
        _ReplyAdapter("alias_id=id"),
        table,
        schema_field_defs=None,
    )
    merge_records(table, source_text="source records mention 1 and 2")

    assert applied == {"alias_id": "id"}
    assert records_to_rows(table) == (["id", "name"], [["1", "A"], ["2", "B"]])


def test_reconcile_field_names_skips_existing_target() -> None:
    table = parse_kv_text(
        "id: 1 | known: keep | Alias: 10",
        columns=["id", "known"],
        primary_key="id",
        anchor_keys=["id"],
    )

    applied = reconcile_field_names(
        _ReplyAdapter("Alias=known"),
        table,
        schema_field_defs=None,
    )

    assert applied == {}
    assert table.records[0].fields["known"] == "keep"
    assert table.records[0].fields["Alias"] == "10"
