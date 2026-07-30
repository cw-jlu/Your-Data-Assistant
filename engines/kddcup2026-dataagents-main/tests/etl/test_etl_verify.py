"""Regression tests for ETL value retry and field verification (EntityTable API)."""

from __future__ import annotations

from typing import Any, cast

from agents.etl._record import EntityTable, parse_kv_text
from agents.etl._verify import retry_missing_values, verify_field_values
from agents.llm.types import ModelResponse


class _ReplyAdapter:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.messages: list[object] = []

    def complete(self, messages: object) -> ModelResponse:
        self.messages.append(messages)
        return ModelResponse(content=self.reply)


def _table(
    text: str,
    columns: list[str],
    primary_key: str,
    field_types: dict[str, str] | None = None,
    anchor_keys: list[str] | None = None,
) -> EntityTable:
    return parse_kv_text(
        text,
        columns=columns,
        primary_key=primary_key,
        anchor_keys=anchor_keys or [primary_key],
        field_types=field_types,
    )


def test_retry_missing_values_matches_mixed_case_reply_keys() -> None:
    table = _table("RecordID: 1 | PersonalCode: ", ["RecordID", "PersonalCode"], "RecordID")

    retry_missing_values(
        cast(Any, _ReplyAdapter("personalcode: 123")),
        table,
        entity_groups={"1": ["RecordID 1 has PersonalCode 123."]},
        forced_gaps={"1": ["PersonalCode"]},
    )

    assert table.records[0].fields["PersonalCode"] == "123"
    assert table.records[0].provenance["PersonalCode"] == "retry"


def test_retry_missing_values_never_overwrites_existing_value() -> None:
    table = _table("RecordID: 1 | PersonalCode: 999", ["RecordID", "PersonalCode"], "RecordID")

    retry_missing_values(
        cast(Any, _ReplyAdapter("personalcode: 123")),
        table,
        entity_groups={"1": ["RecordID 1 has PersonalCode 123."]},
        forced_gaps={"1": ["PersonalCode"]},
    )

    assert table.records[0].fields["PersonalCode"] == "999"


def test_verify_field_values_matches_mixed_case_reply_keys() -> None:
    table = _table("RecordID: 1 | PersonalCode: 123", ["RecordID", "PersonalCode"], "RecordID")

    verify_field_values(
        cast(Any, _ReplyAdapter("personalcode: 456")),
        table,
        entity_groups={"1": ["RecordID 1 has PersonalCode 456."]},
    )

    assert table.records[0].fields["PersonalCode"] == "456"
    assert table.records[0].provenance["PersonalCode"] == "verify"


def test_verify_field_values_matches_metadata_case_insensitively() -> None:
    adapter = _ReplyAdapter("ALL_CORRECT")
    table = _table(
        "RecordID: 1 | TradingDay: 2021-01-08 | Amount: 10",
        ["RecordID", "TradingDay", "Amount"],
        "RecordID",
        field_types={"tradingday": "date", "amount": "number"},
    )

    verify_field_values(
        cast(Any, adapter),
        table,
        entity_groups={"1": ["RecordID 1 traded on 2021-01-08 with amount 10."]},
        schema_field_defs={"amount": "Reported amount."},
    )

    assert adapter.messages
    messages = cast(list[Any], adapter.messages[0])
    prompt = messages[0].content
    assert "Fields to verify: Amount" in prompt
    assert "TradingDay" not in prompt.split("CURRENT EXTRACTION:", 1)[0]
    assert "Amount: Reported amount." in prompt


def test_verify_field_values_rejects_none_when_value_exists() -> None:
    """NONE correction must not clear an already-populated field."""
    table = _table(
        "RecordID: 1 | TotalAssets: 123",
        ["RecordID", "TotalAssets"],
        "RecordID",
        field_types={"TotalAssets": "number"},
    )

    verify_field_values(
        cast(Any, _ReplyAdapter("totalassets: NONE")),
        table,
        entity_groups={"1": ["RecordID 1 does not mention TotalAssets."]},
    )

    assert table.records[0].fields["TotalAssets"] == "123"


def test_verify_field_values_allows_none_when_value_empty() -> None:
    """NONE on an already-empty field is a no-op (no change)."""
    table = _table(
        "RecordID: 1 | TotalAssets: ",
        ["RecordID", "TotalAssets"],
        "RecordID",
        field_types={"TotalAssets": "number"},
    )

    verify_field_values(
        cast(Any, _ReplyAdapter("totalassets: NONE")),
        table,
        entity_groups={"1": ["RecordID 1 does not mention TotalAssets."]},
    )

    assert table.records[0].fields["TotalAssets"] == ""


def test_retry_missing_values_treats_backreference_like_reply_literally() -> None:
    r"""LLM 值含 \1 时必须按字面落地，不得被解释为 regex 组引用。"""
    table = _table("RecordID: 1 | Code: ", ["RecordID", "Code"], "RecordID")

    retry_missing_values(
        cast(Any, _ReplyAdapter(r"code: \1")),
        table,
        entity_groups={"1": [r"RecordID 1 has Code \1."]},
        forced_gaps={"1": ["Code"]},
    )

    assert table.records[0].fields["Code"] == "\\1"


def test_verify_field_values_survive_windows_path_correction() -> None:
    r"""值含 C:\Users\x 不得抛异常或损坏相邻字段。"""
    table = _table(
        "RecordID: 1 | Path: old | Status: open",
        ["RecordID", "Path", "Status"],
        "RecordID",
        field_types={"Path": "string"},
    )

    verify_field_values(
        cast(Any, _ReplyAdapter(r"path: C:\Users\x")),
        table,
        entity_groups={"1": [r"RecordID 1 stores files at C:\Users\x."]},
    )

    assert table.records[0].fields["Path"] == "C:\\Users\\x"
    assert table.records[0].fields["Status"] == "open"


def test_verify_correction_with_pipes_and_newlines_cannot_split_records() -> None:
    """历史事故输入：修正值带竖线/换行——入库经 sanitize，行结构不可能再被破坏。"""
    table = _table(
        "RecordID: 1 | Name: old | Status: open",
        ["RecordID", "Name", "Status"],
        "RecordID",
        field_types={"Name": "string"},
    )

    verify_field_values(
        cast(Any, _ReplyAdapter("name: first\nsecond")),
        table,
        entity_groups={"1": ["RecordID 1 is named first second."]},
    )

    assert len(table.records) == 1
    assert table.records[0].fields["Name"] == "first"
    assert table.records[0].fields["Status"] == "open"


def test_verify_field_values_accepts_string_corrections() -> None:
    table = _table(
        "RecordID: 1 | Status: open",
        ["RecordID", "Status"],
        "RecordID",
        field_types={"Status": "category"},
    )

    verify_field_values(
        cast(Any, _ReplyAdapter("status: closed")),
        table,
        entity_groups={"1": ["RecordID 1 has status closed."]},
    )

    assert table.records[0].fields["Status"] == "closed"
