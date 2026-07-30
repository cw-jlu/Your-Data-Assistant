from __future__ import annotations

import csv
import io

import pytest

from agents.etl._merge import (
    canonical_key_map,
    canonicalize_key,
    line_has_kv_key,
    merge_records,
)
from agents.etl._record import (
    normalize_records,
    parse_kv_text,
    records_to_rows,
    table_to_kv_text,
    to_csv_text,
)


def _kv_to_csv(
    text: str,
    schema_columns: list[str],
    primary_key: str | None,
    schema_field_types: dict[str, str] | None = None,
) -> str | None:
    """新管道尾部的组合：parse → normalize → records_to_rows → to_csv_text。"""
    table = parse_kv_text(
        text,
        columns=schema_columns,
        primary_key=primary_key,
        field_types=schema_field_types,
    )
    normalize_records(table)
    header, rows = records_to_rows(table)
    if not rows:
        return None
    return to_csv_text(header, rows)


def _merge_text(
    text: str,
    primary_key: str | None,
    anchor_keys: list[str] | None = None,
    source_text: str | None = None,
) -> str:
    table = parse_kv_text(
        text,
        columns=[],
        primary_key=primary_key,
        anchor_keys=anchor_keys or ([] if primary_key is None else [primary_key]),
    )
    merge_records(table, source_text=source_text)
    return table_to_kv_text(table)


def test_strip_key_canonicalization_ignores_surrounding_whitespace() -> None:
    """KV 行里 `| record_id: 5` 的键带前导空格，规范化必须命中。"""
    assert line_has_kv_key("name: X | record_id: 5", "record_id") is True
    key_map = canonical_key_map(["Height_CM", "record_id"])
    assert canonicalize_key(" Height-CM", key_map) == "Height_CM"


def test_kv_to_csv_pk_not_first_column() -> None:
    """PK 非首列时（键带前导空格）整文件不得产出 None。"""
    text = "\n".join(
        [
            "name: Alice | record_id: 1 | score: 10",
            "name: Bob | record_id: 2 | score: 20",
        ]
    )
    out = _kv_to_csv(text, ["record_id", "name", "score"], "record_id")
    assert out is not None
    rows = list(csv.reader(io.StringIO(out)))
    assert rows[0] == ["record_id", "name", "score"]
    assert rows[1] == ["1", "Alice", "10"]
    assert rows[2] == ["2", "Bob", "20"]


def test_pre_merge_uses_secondary_anchor_when_primary_key_is_blank() -> None:
    text = "\n".join(
        [
            "personalcode: 101001204 | chinesename: 柳军 | equityfundnv: 851.187055",
            "personalcode:  | chinesename: 韩海平 | totalfundnv: 182.488480",
            "personalcode:  | chinesename: 柳军 | totalfundnv: 883.586211",
        ]
    )

    merged = _merge_text(
        text,
        primary_key="personalcode",
        anchor_keys=["personalcode", "chinesename"],
    )

    assert (
        "personalcode: 101001204 | chinesename: 柳军 | equityfundnv: 851.187055 "
        "| totalfundnv: 883.586211"
    ) in merged
    assert "chinesename: 韩海平 | totalfundnv: 182.488480" in merged


def test_pre_merge_does_not_merge_distinct_primary_keys_by_shared_secondary_anchor() -> None:
    text = "\n".join(
        [
            ("record_id: 150 | province_name: 黑龙江省 | primary_industry_gdp: 684.60"),
            "record_id: 197 | province_name: 黑龙江省 | gdp: 10235.00",
        ]
    )

    merged = _merge_text(
        text,
        primary_key="record_id",
        anchor_keys=["record_id", "province_name", "enddate"],
    )

    assert ("record_id: 150 | province_name: 黑龙江省 | primary_industry_gdp: 684.60") in merged
    assert "record_id: 197 | province_name: 黑龙江省 | gdp: 10235.00" in merged
    assert "record_id: 197 | province_name: 黑龙江省 | gdp: 10235.00 | primary" not in merged


def test_pre_merge_blocks_anchor_merge_when_another_anchor_conflicts() -> None:
    """Temp group shares one secondary anchor but conflicts on another — must not merge."""
    text = "\n".join(
        [
            "personalcode: 101001204 | chinesename: 柳军 | dept: A",
            "personalcode:  | chinesename: 柳军 | dept: B | bonus: 500",
        ]
    )

    merged = _merge_text(
        text,
        primary_key="personalcode",
        anchor_keys=["personalcode", "chinesename", "dept"],
    )

    lines = [row.strip() for row in merged.splitlines() if row.strip()]
    assert len(lines) == 2, f"Expected 2 separate groups, got {len(lines)}"
    real_line = next(row for row in lines if "personalcode: 101001204" in row)
    assert "bonus" not in real_line, "bonus should not leak into the real group"
    assert "dept: A" in real_line


def test_kv_to_csv_matches_schema_keys_case_insensitively() -> None:
    text = (
        "ID: 1688094 | pid: 017-30133 | height_cm: 173.0 "
        "| admission_weight_kg: 86.10 | weight_loss_kg: 1.2 "
        "| discharge_weight_kg: 84.90 | origin: Emergency Department "
        "| discharge_status: Alive"
    )
    schema = [
        "id",
        "pid",
        "height_cm",
        "admission_weight_kg",
        "weight_loss_kg",
        "discharge_weight_kg",
        "origin",
        "discharge_status",
    ]

    csv_text = _kv_to_csv(text, schema, "id")
    assert csv_text is not None

    row = next(csv.DictReader(io.StringIO(csv_text)))
    assert row == {
        "id": "1688094",
        "pid": "017-30133",
        "height_cm": "173.0",
        "admission_weight_kg": "86.10",
        "weight_loss_kg": "1.2",
        "discharge_weight_kg": "84.90",
        "origin": "Emergency Department",
        "discharge_status": "Alive",
    }


def test_kv_to_csv_matches_field_types_case_insensitively() -> None:
    csv_text = _kv_to_csv(
        "RecordID: 1 | Amount: 1,234 | IsActive: 是",
        ["RecordID", "Amount", "IsActive"],
        primary_key="RecordID",
        schema_field_types={
            "amount": "number",
            "isactive": "boolean",
        },
    )

    assert csv_text is not None
    assert list(csv.reader(io.StringIO(csv_text))) == [
        ["RecordID", "Amount", "IsActive"],
        ["1", "1234", "true"],
    ]


def test_primary_key_detection_does_not_match_key_substrings() -> None:
    schema = ["id", "pid", "height_cm"]
    csv_text = _kv_to_csv(
        "pid: 017-30133 | height_cm: 173.0",
        schema,
        primary_key="id",
    )

    assert csv_text is None


def test_kv_to_csv_preserves_rows_without_semantic_filtering() -> None:
    schema = [
        "record_id",
        "province_name",
        "enddate",
        "gdp",
        "primary_industry_gdp",
        "second_industry_gdp",
        "third_industry_gdp",
    ]
    text = "\n".join(
        [
            (
                "record_id: 197 | province_name: 黑龙江、甘肃、广东、陕西、"
                "湖南和云南 | enddate: | gdp: | primary_industry_gdp: | "
                "second_industry_gdp: | third_industry_gdp:"
            ),
            (
                "record_id: 286 | province_name: 山西省 | enddate: | gdp: | "
                "primary_industry_gdp: 302.48 | second_industry_gdp: 4265.77 | "
                "third_industry_gdp: 2370.48"
            ),
        ]
    )

    csv_text = _kv_to_csv(
        text,
        schema,
        primary_key="record_id",
    )
    assert csv_text is not None

    rows = list(csv.DictReader(io.StringIO(csv_text)))
    assert len(rows) == 2
    assert rows[0]["record_id"] == "197"
    assert rows[0]["province_name"] == "黑龙江、甘肃、广东、陕西、湖南和云南"
    assert rows[1]["record_id"] == "286"
    assert rows[1]["primary_industry_gdp"] == "302.48"


def test_kv_to_csv_handles_cleaned_record_id_and_text_spacing() -> None:
    csv_text = _kv_to_csv(
        "record_id: 286 | fund_name: 华夏 A",
        ["record_id", "fund_name"],
        primary_key="record_id",
        schema_field_types={"record_id": "integer_scalar local_record_id", "fund_name": "string"},
    )
    assert csv_text is not None

    row = next(csv.DictReader(io.StringIO(csv_text)))
    assert row["record_id"] == "286"
    assert row["fund_name"] == "华夏A"


def test_kv_to_csv_blanks_values_with_wrong_schema_type() -> None:
    schema = [
        "fundcode",
        "lowestsumsubscribing",
        "lowestsumredemption",
        "foundedsize",
        "establishmentdate",
        "iffof",
        "ifinitiatingfund",
    ]
    csv_text = _kv_to_csv(
        (
            "fundcode: 159001 | lowestsumsubscribing: 万家上证50交易型开放式指数证券投资基金 "
            "| lowestsumredemption: 900000.0 | foundedsize: 1,234.50 "
            "| establishmentdate: 股票型 | iffof: 万家上证50交易型开放式指数证券投资基金 "
            "| ifinitiatingfund: 否"
        ),
        schema,
        primary_key="fundcode",
        schema_field_types={
            "fundcode": "scalar_id",
            "lowestsumsubscribing": "number",
            "lowestsumredemption": "number",
            "foundedsize": "number",
            "establishmentdate": "date",
            "iffof": "boolean",
            "ifinitiatingfund": "boolean",
        },
    )
    assert csv_text is not None

    row = next(csv.DictReader(io.StringIO(csv_text)))
    assert row == {
        "fundcode": "159001",
        "lowestsumsubscribing": "",
        "lowestsumredemption": "900000.0",
        "foundedsize": "1234.50",
        "establishmentdate": "",
        "iffof": "",
        "ifinitiatingfund": "false",
    }


def test_pre_merge_repairs_fabricated_sequential_pks() -> None:
    """Fragments renumbered 1..N by the LLM fold back into their real rows.

    Regression: task_53 in_meansofproductionpi — value-bearing paragraphs
    got record_id 1..29 while the true id sat in strategic_unit_id.
    """
    text = "\n".join(
        [
            "record_id: 300 | strategic_unit_id: 300 | scope: National",
            "record_id: 338 | strategic_unit_id: 338 | scope: National",
            "record_id: 532 | strategic_unit_id: 532 | scope: National",
            "record_id: 1 | strategic_unit_id: 300 | indexvalue: 101.2",
            "record_id: 2 | strategic_unit_id: 338 | indexvalue: 100.1",
        ]
    )
    merged = _merge_text(
        text,
        primary_key="record_id",
        anchor_keys=["record_id", "strategic_unit_id"],
    )
    lines = [ln for ln in merged.splitlines() if ln.strip()]

    assert len(lines) == 3
    row300 = next(ln for ln in lines if ln.startswith("record_id: 300"))
    assert "indexvalue: 101.2" in row300
    assert not any(ln.startswith("record_id: 1 ") for ln in lines)


def test_pre_merge_alias_column_need_not_be_declared_anchor() -> None:
    """The alias column is discovered from the data even without anchor_keys."""
    text = "\n".join(
        [
            "record_id: 300 | strategic_unit_id: 300 | scope: National",
            "record_id: 338 | strategic_unit_id: 338 | scope: National",
            "record_id: 532 | strategic_unit_id: 532 | scope: National",
            "record_id: 1 | strategic_unit_id: 532 | indexvalue: 99.8",
        ]
    )
    merged = _merge_text(text, primary_key="record_id")
    lines = [ln for ln in merged.splitlines() if ln.strip()]

    assert len(lines) == 3
    row532 = next(ln for ln in lines if ln.startswith("record_id: 532"))
    assert "indexvalue: 99.8" in row532


def test_pre_merge_keeps_legitimate_id_disagreements() -> None:
    """Sparse large-code disagreements (secucode vs companycode) stay intact."""
    text = "\n".join(
        [
            "record_id: 600001 | companycode: 600001 | label: a",
            "record_id: 600002 | companycode: 600002 | label: b",
            "record_id: 600003 | companycode: 600003 | label: c",
            "record_id: 600009 | companycode: 81143 | label: d",
        ]
    )
    merged = _merge_text(
        text,
        primary_key="record_id",
        anchor_keys=["record_id", "companycode"],
    )
    lines = [ln for ln in merged.splitlines() if ln.strip()]

    assert len(lines) == 4
    assert any("record_id: 600009 | companycode: 81143" in ln for ln in lines)


def test_pre_merge_keeps_genuine_small_sequential_pks() -> None:
    """Docs whose real pk space is 1..N with an unrelated id column are untouched."""
    text = "\n".join(
        [
            "record_id: 1 | personalcode: 101001 | label: a",
            "record_id: 2 | personalcode: 101002 | label: b",
            "record_id: 3 | personalcode: 101003 | label: c",
        ]
    )
    merged = _merge_text(
        text,
        primary_key="record_id",
        anchor_keys=["record_id", "personalcode"],
    )
    lines = [ln for ln in merged.splitlines() if ln.strip()]

    assert len(lines) == 3
    assert any(ln.startswith("record_id: 1 ") for ln in lines)


@pytest.mark.parametrize(
    "fragment",
    [
        # anchor + non-anchor column whose value is constant across real rows
        "record_id:  | secucode: 300707 | companyname: 无锡威唐工业技术股份有限公司",
        # anchor values only
        "record_id:  | secucode: 300707",
    ],
)
def test_pre_merge_drops_identity_echo_fragment(fragment: str) -> None:
    """Blank-PK fragments carrying only identity echoes never become rows."""
    text = "\n".join(
        [
            "record_id: 29 | secucode: 300707 | companyname: 无锡威唐工业技术股份有限公司 | transfer_shares: 300000",
            "record_id: 30 | secucode: 300707 | companyname: 无锡威唐工业技术股份有限公司 | transfer_shares: 785000",
            "record_id: 31 | secucode: 300707 | companyname: 无锡威唐工业技术股份有限公司 | transfer_shares: 40000",
            fragment,
        ]
    )
    merged = _merge_text(
        text,
        primary_key="record_id",
        anchor_keys=["record_id", "secucode"],
    )
    lines = [ln for ln in merged.splitlines() if ln.strip()]

    assert len(lines) == 3
    assert all(ln.split("|")[0].strip() != "record_id:" for ln in lines)


def test_pre_merge_keeps_data_carrying_fragment_with_ambiguous_anchor() -> None:
    """An unplaced fragment holding real data survives as its own row."""
    text = "\n".join(
        [
            "record_id: 29 | secucode: 300707 | transfer_shares: 300000",
            "record_id: 30 | secucode: 300707 | transfer_shares: 785000",
            "record_id:  | secucode: 300707 | transfer_shares: 99999",
        ]
    )
    merged = _merge_text(
        text,
        primary_key="record_id",
        anchor_keys=["record_id", "secucode"],
    )
    lines = [ln for ln in merged.splitlines() if ln.strip()]

    assert len(lines) == 3
    assert any("transfer_shares: 99999" in ln for ln in lines)


def test_pre_merge_single_real_row_disables_constant_echo_rule() -> None:
    """With one real record, column constancy is no evidence — fragment stays."""
    text = "\n".join(
        [
            "personalcode: 101 | chinesename: 柳军 | dept: A | city: X",
            "personalcode:  | chinesename: 柳军 | dept: B | city: X",
        ]
    )
    merged = _merge_text(
        text,
        primary_key="personalcode",
        anchor_keys=["personalcode", "chinesename", "dept"],
    )
    lines = [ln for ln in merged.splitlines() if ln.strip()]

    assert len(lines) == 2
    assert any("city: X" in ln and "dept: B" in ln for ln in lines)


_UNIT_SOURCE = (
    "Strategic Unit 112 had liabilities of 888,100 to the central bank. "
    "Strategic Unit 7 reported total assets of 600. "
    "Strategic Unit 9 reported total assets of 700."
)


def test_pre_merge_drops_source_unanchored_pk_and_heals_anchor_merge() -> None:
    """A fabricated pk is dropped, restoring anchor uniqueness so the
    legitimate blank-PK fragment merges into its real record."""
    text = "\n".join(
        [
            "record_id: 112 | asset_id: 112 | totalassets: 500",
            "record_id: 7 | asset_id: 7 | totalassets: 600",
            "record_id: 9 | asset_id: 9 | totalassets: 700",
            # fabricated line: pk 0 never appears in the source text
            "record_id: 0 | asset_id: 112 | liabilities_cb: 946200",
            # honest fragment: blank pk, real anchor, real data
            "record_id:  | asset_id: 112 | liabilities_cb: 888100",
        ]
    )
    merged = _merge_text(
        text,
        primary_key="record_id",
        anchor_keys=["record_id", "asset_id"],
        source_text=_UNIT_SOURCE,
    )
    lines = [ln for ln in merged.splitlines() if ln.strip()]

    assert len(lines) == 3
    assert not any("record_id: 0 " in ln or "946200" in ln for ln in lines)
    unit_112 = next(ln for ln in lines if ln.startswith("record_id: 112"))
    assert "liabilities_cb: 888100" in unit_112


def test_pre_merge_drops_pk_only_embedded_in_larger_numbers() -> None:
    """Digits appearing only inside formatted numbers do not anchor a pk."""
    text = "\n".join(
        [
            "record_id: 112 | asset_id: 112 | totalassets: 500",
            "record_id: 7 | asset_id: 7 | totalassets: 600",
            "record_id: 408 | asset_id: 408 | totalassets: 900",
        ]
    )
    merged = _merge_text(
        text,
        primary_key="record_id",
        anchor_keys=["record_id", "asset_id"],
        source_text=_UNIT_SOURCE + " The ledger total reached 8,408,793 yuan.",
    )
    lines = [ln for ln in merged.splitlines() if ln.strip()]

    assert len(lines) == 2
    assert not any("record_id: 408" in ln for ln in lines)


def test_pre_merge_unanchored_gate_skips_non_digit_pks_and_no_source() -> None:
    text = "\n".join(
        [
            "record_id: TR06 | label: a",
            "record_id: TR07 | label: b",
        ]
    )
    # non-digit pks survive even though absent from source_text
    merged = _merge_text(
        text,
        primary_key="record_id",
        anchor_keys=["record_id"],
        source_text=_UNIT_SOURCE,
    )
    assert len([ln for ln in merged.splitlines() if ln.strip()]) == 2

    # without source_text the gate is a no-op for digit pks too
    digit_text = "record_id: 9991 | label: a\nrecord_id: 9992 | label: b"
    merged = _merge_text(
        digit_text,
        primary_key="record_id",
        anchor_keys=["record_id"],
    )
    assert len([ln for ln in merged.splitlines() if ln.strip()]) == 2
