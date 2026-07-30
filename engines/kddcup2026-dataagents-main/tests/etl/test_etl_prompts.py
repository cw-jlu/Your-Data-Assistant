from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from agents.etl._compress import (
    _format_schema_def_hint,
    _format_schema_type_hint,
    _generate_compress_guide,
)
from agents.etl._record import (
    normalize_local_record_ids,
    normalize_records,
    parse_kv_text,
    table_to_kv_text,
)
from agents.etl._schema import (
    _ensure_local_record_anchor,
    _has_local_record_ids,
    _merge_multi_round_schemas,
    _merge_synonyms_into_km,
    _parse_schema_response,
    infer_field_units_from_prose,
)
from agents.etl._units import _deterministic_factor, _parse_unit_factor_response
from agents.llm.types import ModelResponse


def _record_clean(
    text: str,
    columns: list[str],
    field_types: dict[str, str],
    primary_key: str,
) -> str:
    table = parse_kv_text(
        text,
        columns=columns,
        primary_key=primary_key,
        anchor_keys=[primary_key],
        field_types=field_types,
    )
    normalize_local_record_ids(table)
    normalize_records(table)
    return table_to_kv_text(table)


def test_record_pass_drops_grouped_record_ids_before_clean_markdown() -> None:
    cleaned = _record_clean(
        "\n".join(
            [
                (
                    "record_id: 档案 197至217 | province_name: 黑龙江、甘肃、广东、陕西、"
                    "湖南和云南 | enddate: | gdp:"
                ),
                ("record_id: 档案 286 | province_name: 山西省 | primary_industry_gdp: 302.48"),
            ]
        ),
        columns=["record_id", "province_name", "enddate", "gdp", "primary_industry_gdp"],
        field_types={
            "record_id": "integer_scalar local_record_id",
            "province_name": "string",
            "enddate": "date",
            "gdp": "number",
            "primary_industry_gdp": "number",
        },
        primary_key="record_id",
    )

    assert "197至217" not in cleaned
    assert "record_id: 286 | province_name: 山西省" in cleaned
    assert "primary_industry_gdp: 302.48" in cleaned


def test_record_clean_blanks_values_with_wrong_schema_type() -> None:
    cleaned = _record_clean(
        (
            "fundcode: 159001 | lowestsumsubscribing: 万家上证50交易型开放式指数证券投资基金 "
            "| lowestsumredemption: 900000.0 | foundedsize: 1,234.50 "
            "| establishmentdate: 股票型 | iffof: 万家上证50交易型开放式指数证券投资基金 "
            "| ifinitiatingfund: 否"
        ),
        columns=[
            "fundcode",
            "lowestsumsubscribing",
            "lowestsumredemption",
            "foundedsize",
            "establishmentdate",
            "iffof",
            "ifinitiatingfund",
        ],
        field_types={
            "fundcode": "scalar_id",
            "lowestsumsubscribing": "number",
            "lowestsumredemption": "number",
            "foundedsize": "number",
            "establishmentdate": "date",
            "iffof": "boolean",
            "ifinitiatingfund": "boolean",
        },
        primary_key="fundcode",
    )

    assert "lowestsumsubscribing:  | lowestsumredemption: 900000.0" in cleaned
    assert "foundedsize: 1234.50" in cleaned
    assert "establishmentdate:  | iffof:  | ifinitiatingfund: false" in cleaned


def test_record_clean_preserves_approx_tag_on_numeric_values() -> None:
    """~ 是恒等式修复的显式证据，压缩清洗规范数字但不得剥掉标记。"""
    cleaned = _record_clean(
        "record_id: 1 | gdp: ~1,234.5 | population: ～99 | area: 10",
        columns=["record_id", "gdp", "population", "area"],
        field_types={
            "record_id": "integer_scalar local_record_id",
            "gdp": "number",
            "population": "number",
            "area": "number",
        },
        primary_key="record_id",
    )

    assert cleaned == "record_id: 1 | gdp: ~1234.5 | population: ~99 | area: 10"


def test_compress_hints_match_metadata_case_insensitively() -> None:
    columns = ["SecuCode", "TradingDay"]

    type_hint = _format_schema_type_hint(
        columns,
        {"secucode": "scalar_id", "tradingday": "date"},
    )
    def_hint = _format_schema_def_hint(
        columns,
        {"secucode": "Stock ticker.", "tradingday": "Trading session date."},
    )

    assert "SecuCode: scalar_id" in type_hint
    assert "TradingDay: date" in type_hint
    assert "SecuCode: Stock ticker." in def_hint
    assert "TradingDay: Trading session date." in def_hint


def test_compress_clean_matches_schema_types_case_insensitively() -> None:
    cleaned = _record_clean(
        "RecordID: 档案 42 | Amount: 1,234",
        columns=["RecordID", "Amount"],
        field_types={
            "recordid": "integer_scalar local_record_id",
            "amount": "number",
        },
        primary_key="RecordID",
    )

    assert cleaned == "RecordID: 42 | Amount: 1234"


def test_compress_guide_matches_metadata_case_insensitively() -> None:
    adapter = _CapturingAdapter("guide")

    result = _generate_compress_guide(
        cast(Any, adapter),
        "This paragraph is deliberately long enough to be sampled for guide generation. "
        "It says SecuCode 600000 traded on 2021-01-08 with many details.",
        schema_columns=["SecuCode", "TradingDay"],
        schema_field_defs={
            "secucode": "Stock ticker.",
            "tradingday": "Trading session date.",
        },
        schema_field_types={"secucode": "scalar_id", "tradingday": "date"},
    )

    assert result == "guide"
    prompt = adapter.messages[0][1].content
    assert "SecuCode | type=scalar_id | Stock ticker." in prompt
    assert "TradingDay | type=date | Trading session date." in prompt


def test_schema_adds_record_id_for_repeated_local_record_labels() -> None:
    columns, anchors = _ensure_local_record_anchor(
        "档案 21 记录一个经理。\n\n战略单元 353 记录另一个经理。\n\n记录 418 继续。",
        ["personalcode", "manager_name"],
        ["personalcode", "manager_name"],
    )

    assert columns == ["record_id", "personalcode", "manager_name"]
    assert anchors == ["record_id", "personalcode", "manager_name"]


@pytest.mark.parametrize(
    ("prose", "expected"),
    [
        # 关键词 + 数字：档案/Record 等标签方案
        pytest.param("档案 21。\n战略单元 353。\n记录 418。", True, id="cjk-labels"),
        pytest.param("Record 1, Record 2, Record 3.", True, id="english-labels"),
        # 关键词作为长词内嵌子串不算（profile 含 file、briefcase 含 case）
        pytest.param(
            "profile 29 and profile 30 and profile 31 exist.", False, id="profile-not-file"
        ),
        pytest.param(
            "briefcase 12, briefcase 13, briefcase 14 shipped.", False, id="briefcase-not-case"
        ),
        # Airtable 风格：rec 后首字符为大写/数字
        pytest.param(
            "rec0Si5cQ4rJRVzd6 then rec7SRmmw3oovHndK then recXq0Zz1Ab2Cd3Ef.",
            True,
            id="airtable-ids",
        ),
        # 驼峰英文词 rec 后是小写，不算
        pytest.param(
            "recordIdentifier, receivablesTurnover, reconciliationBalance ok; "
            "recordIdentifier2, receivablesTurnoverX, reconciliationRatioY too.",
            False,
            id="camelcase-words",
        ),
        # code 风格：同一前缀 >=3 个编号
        pytest.param("TR391 TR483 TR512 are specimens.", True, id="single-prefix-codes"),
        # 混合前缀（交易所代码 + ISO 标准号）不算记录方案
        pytest.param("SZ300707, SH600519 and ISO9001 are mentioned.", False, id="mixed-prefixes"),
        pytest.param("SZ300707 and SZ300708 only.", False, id="two-codes-insufficient"),
    ],
)
def test_has_local_record_ids_detection(prose: str, expected: bool) -> None:
    assert _has_local_record_ids(prose) is expected


def test_schema_skips_record_id_for_governance_prose_with_tickers() -> None:
    """含 3 个交易所代码的治理 schema 不得被注入幽灵 record_id。"""
    columns, anchors = _ensure_local_record_anchor(
        "公司 SZ300707 的董事会与 SH600519、SZ000858 的治理结构对比。",
        ["companycode", "board_size"],
        ["companycode"],
    )

    assert columns == ["companycode", "board_size"]
    assert anchors == ["companycode"]


@pytest.mark.parametrize(
    ("raw", "pk", "anchors", "types", "cols"),
    [
        pytest.param(
            "PK: fundcode\n"
            "ANCHORS: fundcode, fundname\n"
            "TYPES: fundcode=scalar_id, fundname=string, nav=number\n"
            "fundcode, fundname, nav",
            "fundcode",
            ["fundcode", "fundname"],
            {"fundcode": "scalar_id", "fundname": "string", "nav": "number"},
            ["fundcode", "fundname", "nav"],
            id="normal-order",
        ),
        # 各前缀行乱序出现也按前缀识别，不按位置
        pytest.param(
            "TYPES: id=integer, score=number\nANCHORS: id\nPK: id\nid, score, label",
            "id",
            ["id"],
            {"id": "integer", "score": "number"},
            ["id", "score", "label"],
            id="out-of-order",
        ),
        # 只有 PK 行：anchors 回退为 [pk]，types 为空
        pytest.param(
            "PK: patient_id\npatient_id, age, weight",
            "patient_id",
            ["patient_id"],
            {},
            ["patient_id", "age", "weight"],
            id="missing-anchors-and-types",
        ),
        # 前后的 prose 行被忽略，不污染列清单
        pytest.param(
            "Here is the schema:\n"
            "PK: record_id\n"
            "ANCHORS: record_id, province_name\n"
            "TYPES: record_id=integer_scalar local_record_id, gdp=number\n"
            "record_id, province_name, gdp\n"
            "Note: gdp is in billions.",
            "record_id",
            ["record_id", "province_name"],
            {"record_id": "integer_scalar local_record_id", "gdp": "number"},
            ["record_id", "province_name", "gdp"],
            id="extra-commentary-ignored",
        ),
        # 声明为 ANCHOR 但缺席列清单的字段被补到 columns 尾部
        pytest.param(
            "PK: id\nANCHORS: id, name\nid, score",
            "id",
            ["id", "name"],
            {},
            ["id", "score", "name"],
            id="anchor-appended-to-columns",
        ),
    ],
)
def test_parse_schema_response_variants(
    raw: str, pk: str, anchors: list[str], types: dict[str, str], cols: list[str]
) -> None:
    result = _parse_schema_response(raw)
    assert result is not None
    got_pk, got_anchors, got_types, got_cols, units, defs = result
    assert (got_pk, got_anchors, got_types, got_cols) == (pk, anchors, types, cols)
    assert units == {}
    assert defs == {}


@pytest.mark.parametrize(
    "raw",
    [
        "NONE",
        "I cannot determine the schema.",
        "PK: x\nonly_one_column",
    ],
)
def test_parse_schema_response_returns_none_on_garbage(raw: str) -> None:
    assert _parse_schema_response(raw) is None


def test_parse_schema_response_remaps_anchor_case_to_existing_columns() -> None:
    result = _parse_schema_response(
        "PK: id\n"
        "ANCHORS: id, secucode, tradingday\n"
        "TYPES: id=integer_scalar_id, SecuCode=scalar_id, TradingDay=date\n"
        "id,SecuCode,TradingDay,TurnoverDeals"
    )

    assert result is not None
    pk, anchors, _types, columns, _units, _defs = result
    assert pk == "id"
    assert anchors == ["id", "SecuCode", "TradingDay"]
    assert columns == ["id", "SecuCode", "TradingDay", "TurnoverDeals"]


@pytest.mark.parametrize(
    ("source", "target", "expected"),
    [
        # Within the CNY family the factor is scale(source)/scale(target).
        ("亿元", "万元", 10000.0),
        ("元", "万元", 0.0001),
        ("万元", "元", 10000.0),
        ("万元", "亿元", 0.0001),
        ("元", "亿元", 1e-8),
        ("%", "ratio", 0.01),
        ("ratio", "%", 100.0),
        # Different families or unknown units are not computable.
        ("元", "%", None),
        ("万吨", "吨", None),
        ("万元", "USD", None),
    ],
)
def test_deterministic_factor(source: str, target: str, expected: float | None) -> None:
    assert _deterministic_factor(source, target) == expected


@pytest.mark.parametrize(
    ("raw", "source_units", "expected"),
    [
        # 元→万元 must divide by 10000 even when the LLM echoes the *10000 example.
        # Regression: ed_chinafibalancesheetrmb (task_47) — the LLM pattern-matched
        # the 亿元→万元 example and emitted *10000 for every 元-sourced field.
        pytest.param(
            "corporatesavings=万元*10000, totalloans=万元*10000",
            {"corporatesavings": "元", "totalloans": "元"},
            {"corporatesavings": ("万元", 0.0001), "totalloans": ("万元", 0.0001)},
            id="corrects-inverted-direction",
        ),
        # A known unit pair no longer depends on the LLM emitting *factor.
        pytest.param(
            "gdp=万元",
            {"gdp": "亿元"},
            {"gdp": ("万元", 10000.0)},
            id="fills-missing-factor-for-known-pair",
        ),
        pytest.param(
            "amount=万元",
            {"Amount": "元"},
            {"Amount": ("万元", 0.0001)},
            id="matches-field-case-insensitively",
        ),
        pytest.param(
            "weight=吨*10000",
            {"weight": "万吨"},
            {"weight": ("吨", 10000.0)},
            id="keeps-llm-factor-for-unknown-pair",
        ),
        pytest.param(
            "rate=ratio*0.01",
            {"rate": "%"},
            {"rate": ("ratio", 0.01)},
            id="percent-spelling-converts",
        ),
        # Same-scale units (factor 1) must not produce a conversion entry.
        pytest.param("gdp=万元*1", {"gdp": "万元"}, {}, id="equal-scale-spelling-dropped"),
    ],
)
def test_parse_unit_factor_response(
    raw: str, source_units: dict[str, str], expected: dict[str, tuple[str, float]]
) -> None:
    assert _parse_unit_factor_response(raw, source_units) == expected


_KM_FIXTURE = """\
# Governance

### 2.1 Balance Sheet (`ed_bs`)

| Column | Semantic Definition |
|---|---|
| `totalassets` | Aggregate assets |
| `totalliabilities` | Aggregate liabilities |

Key Relationship: identity.

### 2.9 Cross-Table Disambiguation

| Pair | Rule |
|---|---|
| `totalassets` vs `netabroadassets` | Never substitute |

### 3.1 Example Queries on `ed_bs`

No field table here, only prose.
"""


def test_km_table_fields_parses_backticked_table() -> None:
    from agents.etl.knowledge import km_table_fields

    fields = km_table_fields(_KM_FIXTURE, "ed_bs")
    assert fields == {
        "totalassets": "Aggregate assets",
        "totalliabilities": "Aggregate liabilities",
    }


def test_km_table_fields_parses_chinese_table() -> None:
    from agents.etl.knowledge import km_table_fields

    km = (
        "### 2.2 国内生产总值 (Gross Domestic Product)\n\n"
        "| Column | Semantic Definition |\n"
        "|--------|---------------------|\n"
        "| 国内生产总值(百万元) | China's GDP in millions of RMB. |\n"
        "| 人均国内生产总值(元/人) | Per capita GDP. |\n"
    )
    fields = km_table_fields(km, "国内生产总值")
    assert "国内生产总值(百万元)" in fields
    assert "人均国内生产总值(元/人)" in fields


def test_km_table_fields_rejects_substring_stem_anchor() -> None:
    """`mf_netvalue` must not anchor inside `mf_netvalueperformancehis`."""
    from agents.etl.knowledge import km_table_fields

    km = (
        "### 2.8 Net Value Performance History (`mf_netvalueperformancehis`)\n\n"
        "| Column | Semantic Definition |\n"
        "|---|---|\n"
        "| `nvdailygrowthrate` | Daily growth rate |\n"
    )
    assert km_table_fields(km, "mf_netvalue") == {}


def test_km_table_fields_retries_next_heading_and_skips_junk_cells() -> None:
    """A table-less heading falls through to the next matching one; prose-like
    cells (`a` vs `b`) never become columns."""
    from agents.etl.knowledge import km_table_fields

    km = "### 1.1 Overview of `ed_bs`\n\nProse only here.\n\n" + _KM_FIXTURE
    fields = km_table_fields(km, "ed_bs")
    assert sorted(fields) == ["totalassets", "totalliabilities"]
    # the disambiguation table's "`totalassets` vs `netabroadassets`" cell
    # must never parse as a column even if that section were matched
    assert not any("vs" in c for c in fields)


def test_apply_km_schema_floor_readds_omitted_columns() -> None:
    from agents.etl._schema import _apply_km_schema_floor

    columns = ["record_id", "totalassets"]
    raw_defs = {"totalassets": "from llm"}
    km_fields = {
        "totalassets": "Aggregate assets",
        "totalliabilities": "Aggregate liabilities",
    }

    readded = _apply_km_schema_floor(columns, raw_defs, km_fields)

    assert readded == ["totalliabilities"]
    assert columns == ["record_id", "totalassets", "totalliabilities"]
    # existing LLM definition wins; the km definition fills the gap
    assert raw_defs["totalassets"] == "from llm"
    assert raw_defs["totalliabilities"] == "Aggregate liabilities"


class _FailingAdapter:
    """Asserts the knowledge path is skipped before any LLM call."""

    def complete(self, messages: list[Any], *, tools: object | None = None) -> ModelResponse:
        raise AssertionError("LLM must not be called when km lacks the table")


class _SequenceAdapter:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls = 0

    def complete(self, messages: list[Any], *, tools: object | None = None) -> ModelResponse:
        self.calls += 1
        return ModelResponse(content=self._responses.pop(0))


class _CapturingAdapter:
    def __init__(self, response: str = "") -> None:
        self.response = response
        self.messages: list[list[Any]] = []

    def complete(self, messages: list[Any], *, tools: object | None = None) -> ModelResponse:
        self.messages.append(messages)
        return ModelResponse(content=self.response)


def test_merge_synonyms_into_km_canonicalizes_case_only_governance_fields() -> None:
    columns = ["record_id", "SecuCode", "TradingDay", "TurnoverDeals"]
    raw_defs = {
        "SecuCode": "Stock ticker code.",
        "TradingDay": "Trading session date.",
        "TurnoverDeals": "Trading volume.",
    }
    km_fields = {
        "secucode": "Stock ticker identifying the security.",
        "tradingday": "The calendar date of the trading session.",
        "turnoverdeals": "Trading volume for the given trading day.",
    }

    merged = _merge_synonyms_into_km(
        cast(Any, _FailingAdapter()),
        columns,
        raw_defs,
        km_fields,
    )

    assert merged == [
        ("SecuCode", "secucode"),
        ("TradingDay", "tradingday"),
        ("TurnoverDeals", "turnoverdeals"),
    ]
    assert columns == ["record_id", "secucode", "tradingday", "turnoverdeals"]


def test_merge_multi_round_schemas_deduplicates_anchor_case() -> None:
    columns, anchors, types, _units, _defs = _merge_multi_round_schemas(
        cast(Any, _FailingAdapter()),
        [
            (["SecuCode"], ["SecuCode"], {"SecuCode": "scalar_id"}, {}, {}),
            (["secucode"], ["secucode"], {"secucode": "scalar_id"}, {}, {}),
        ],
    )

    assert columns == ["SecuCode"]
    assert anchors == ["SecuCode"]
    assert types["SecuCode"] == "scalar_id"


def test_infer_field_units_prompt_separates_currency_from_magnitude() -> None:
    adapter = _CapturingAdapter("")

    infer_field_units_from_prose(
        cast(Any, adapter),
        "Corporate Savings were 326283.0. CurrencyName: Chinese Yuan.",
        ["CorporateSavings"],
        {"CorporateSavings": "period-end balance of corporate/enterprise deposits"},
        {},
    )

    assert adapter.messages
    system = adapter.messages[0][0].content
    prompt = adapter.messages[0][1].content

    assert "WHEN TO EXTRACT A UNIT" in prompt
    assert "WHEN NOT TO EXTRACT A UNIT" in prompt
    assert "EXAMPLES" in prompt
    assert "CurrencyName: 人民币元" in prompt
    assert "DO NOT output CorporateSavings=元" in prompt
    assert "If no field qualifies, output nothing" in prompt
    assert "Currency labels such as Yuan/RMB/CNY are not magnitude units" in system


def test_parse_knowledge_schema_skips_unmentioned_table(tmp_path) -> None:
    """Gate A: km never mentions the table → no LLM call, fall to inference.

    Regression: task_59 ed_consumerpriceindex — the entity matcher
    force-matched ed_retailvalueofscgoods and the schema inherited a wrong
    table's columns, blanking every CPI value.
    """
    from types import SimpleNamespace

    from agents.etl._schema import parse_knowledge_schema

    (tmp_path / "knowledge.md").write_text(_KM_FIXTURE, encoding="utf-8")
    task = SimpleNamespace(context_dir=tmp_path)

    result = parse_knowledge_schema(
        cast(Any, task),
        Path("doc/ed_consumerpriceindex.md"),
        cast(Any, _FailingAdapter()),
        "Strategic Unit 3 reported a QoqCPI of 100.1 for the period.",
    )

    assert result is None


def test_parse_knowledge_schema_rejects_filter_output_without_stem(tmp_path) -> None:
    """Gate B: the pre-filter returning another entity's section is no-match."""
    from types import SimpleNamespace

    from agents.etl._schema import parse_knowledge_schema

    km = _KM_FIXTURE + "\nCross-reference: see also `ed_cpi` for deflators.\n"
    (tmp_path / "knowledge.md").write_text(km, encoding="utf-8")
    task = SimpleNamespace(context_dir=tmp_path)
    adapter = _SequenceAdapter(
        ["### 2.1 Balance Sheet (`ed_bs`)\n\n| Column | Def |\n|---|---|\n| `totalassets` | x |"]
    )

    result = parse_knowledge_schema(
        cast(Any, task),
        Path("doc/ed_cpi.md"),
        cast(Any, adapter),
        "Unit 3 reported values for the period in question here.",
    )

    assert result is None
    assert adapter.calls == 1  # only the pre-filter ran; main call skipped


def test_parse_knowledge_schema_applies_km_floor(tmp_path) -> None:
    """Happy path: LLM omits a governance column; the floor re-adds it."""
    from types import SimpleNamespace

    from agents.etl._schema import parse_knowledge_schema

    (tmp_path / "knowledge.md").write_text(_KM_FIXTURE, encoding="utf-8")
    task = SimpleNamespace(context_dir=tmp_path)
    section = (
        "### 2.1 Balance Sheet (`ed_bs`)\n\n"
        "| Column | Semantic Definition |\n|---|---|\n"
        "| `totalassets` | Aggregate assets |\n"
        "| `totalliabilities` | Aggregate liabilities |\n"
    )
    schema_reply = (
        "PK: record_id\n"
        "ANCHORS: record_id\n"
        "TYPES: record_id=integer_scalar local_record_id, totalassets=number\n"
        "record_id, totalassets"
    )
    adapter = _SequenceAdapter([section, schema_reply])

    result = parse_knowledge_schema(
        cast(Any, task),
        Path("doc/ed_bs.md"),
        cast(Any, adapter),
        "Record 3 of ed_bs held assets. Record 4 held more. Record 5 held some.",
    )

    assert result is not None
    columns, _anchors, _types, _units, defs = result
    assert "totalliabilities" in columns  # floor re-added the omitted column
    assert defs.get("totalliabilities") == "Aggregate liabilities"


def test_parse_knowledge_schema_remaps_anchor_when_synonym_merged(tmp_path) -> None:
    """If a PK synonym is merged into a km column, anchors must follow it."""
    from types import SimpleNamespace

    from agents.etl._schema import parse_knowledge_schema

    km = (
        "### 1.1 Reporting Dates (`ed_dates`)\n\n"
        "| Column | Semantic Definition |\n|---|---|\n"
        "| `enddate` | Reporting period end date |\n"
        "| `amount` | Reported amount |\n"
    )
    (tmp_path / "knowledge.md").write_text(km, encoding="utf-8")
    task = SimpleNamespace(context_dir=tmp_path)
    schema_reply = (
        "PK: report_date\n"
        "ANCHORS: report_date\n"
        "TYPES: report_date=date, amount=number\n"
        "UNITS: amount=万元\n"
        "DEFS: report_date=Date phrase from prose; amount=Reported amount\n"
        "report_date, amount"
    )
    adapter = _SequenceAdapter([km, schema_reply, "report_date=enddate"])

    result = parse_knowledge_schema(
        cast(Any, task),
        Path("doc/ed_dates.md"),
        cast(Any, adapter),
        "The ed_dates prose describes reporting dates and amounts.",
    )

    assert result is not None
    columns, anchors, types, units, defs = result
    assert "report_date" not in columns
    assert "enddate" in columns
    assert anchors == ["enddate"]
    assert types["enddate"] == "date"
    assert units["amount"] == "万元"
    assert defs["enddate"] == "Reporting period end date"


def test_parse_knowledge_schema_canonicalizes_case_only_km_fields(tmp_path) -> None:
    """Knowledge field spelling wins when schema output differs only by case."""
    from types import SimpleNamespace

    from agents.etl._schema import parse_knowledge_schema

    km = (
        "### 2.3 Daily Market Quotations — `qt_dailyquote`\n\n"
        "| Field | Semantic Definition |\n|---|---|\n"
        "| `secucode` | Stock ticker identifying the security. |\n"
        "| `turnoverdeals` | Trading volume for the given trading day. |\n"
        "| `tradingday` | The calendar date of the trading session. |\n"
    )
    (tmp_path / "knowledge.md").write_text(km, encoding="utf-8")
    task = SimpleNamespace(context_dir=tmp_path)
    schema_reply = (
        "PK: record_id\n"
        "ANCHORS: record_id, secucode, tradingday\n"
        "TYPES: record_id=integer_scalar local_record_id, SecuCode=scalar_id, "
        "TradingDay=date, TurnoverDeals=integer_count\n"
        "DEFS: SecuCode=Stock ticker; TradingDay=Trading date; "
        "TurnoverDeals=Trading volume\n"
        "record_id,SecuCode,TradingDay,TurnoverDeals"
    )
    adapter = _SequenceAdapter([km, schema_reply])

    result = parse_knowledge_schema(
        cast(Any, task),
        Path("doc/qt_dailyquote.md"),
        cast(Any, adapter),
        "档案 2706 的 SecuCode 为 601908，交易日为 2021-01-08。",
    )

    assert result is not None
    columns, anchors, types, _units, defs = result
    assert columns == ["record_id", "secucode", "tradingday", "turnoverdeals"]
    assert anchors == ["record_id", "secucode", "tradingday"]
    assert types["secucode"] == "scalar_id"
    assert types["tradingday"] == "date"
    assert defs["turnoverdeals"] == "Trading volume"
