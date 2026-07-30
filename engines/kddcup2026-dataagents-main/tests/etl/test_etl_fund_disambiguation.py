from __future__ import annotations

from typing import Any

from agents.etl._compress import _generate_compress_guide
from agents.etl._record import parse_kv_text
from agents.etl._schema import merge_prose_schema_into_governance
from agents.etl._verify import retry_missing_values
from agents.llm.types import ModelMessage, ModelResponse


class _SequenceAdapter:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.prompts: list[str] = []

    def complete(
        self, messages: list[ModelMessage], *, tools: object | None = None, **kwargs: Any
    ) -> ModelResponse:
        del tools, kwargs
        content = messages[-1].content
        assert isinstance(content, str)
        self.prompts.append(content)
        if not self._responses:
            raise AssertionError("no scripted responses left")
        text = self._responses.pop(0)
        return ModelResponse(content=text)


def test_fund_schema_merge_disambiguates_fund_type_from_product_structure() -> None:
    adapter = _SequenceAdapter(
        [
            "\n".join(
                [
                    "PK: record_id",
                    "ANCHORS: record_id, innercode",
                    "TYPES: record_id=integer_scalar local_record_id, innercode=scalar_id, fund_type=category, product_structure=category",
                    "UNITS:",
                    "DEFS: fund_type=fund classification such as 股票型 or 混合型; product_structure=ETF or LOF trading/product structure",
                    "record_id, innercode, fund_type, product_structure",
                ]
            ),
            "fund_type=type\nproduct_structure=DISTINCT",
        ]
    )

    extra_cols, extra_types, extra_defs = merge_prose_schema_into_governance(
        adapter,
        (
            "档案 69 的内部编码为 12461。它是一只LOF，"
            "最终备案文件将其修正为积极配置-大盘成长，这使其归类为混合型基金。"
        ),
        ["innercode", "type"],
        {
            "innercode": "Internal fund identifier.",
            "type": "Fund classification by investment strategy.",
        },
        {"innercode": "scalar_id", "type": "category"},
        "group all funds by fund type",
    )

    assert extra_cols == ["product_structure"]
    assert extra_types == {"product_structure": "category"}
    assert extra_defs == {"product_structure": "ETF or LOF trading/product structure"}
    assert "MUTUAL FUND DOMAIN DISAMBIGUATION" in adapter.prompts[0]
    assert "NEVER merge values across these columns" in adapter.prompts[1]
    assert "FundType" in adapter.prompts[1]


def test_compress_guide_prompt_disambiguates_fund_classification() -> None:
    adapter = _SequenceAdapter(["guide"])

    guide = _generate_compress_guide(
        adapter,
        (
            "档案 69 的内部编码为 12461。它是一只LOF，属于常规基金。"
            "最终备案文件将其修正为积极配置-大盘成长，这使其归类为混合型基金。"
        ),
        ["record_id", "innercode", "type"],
        {
            "record_id": "Document-local archive id.",
            "innercode": "Internal fund identifier.",
            "type": "Fund classification by investment strategy.",
        },
        {
            "record_id": "integer_scalar local_record_id",
            "innercode": "scalar_id",
            "type": "category",
        },
    )

    assert guide == "guide"
    assert len(adapter.prompts) == 1
    prompt = adapter.prompts[0]
    assert "MUTUAL FUND DOMAIN DISAMBIGUATION" in prompt
    assert "ETF/LOF" in prompt
    assert "FundType" in prompt or "fund classification" in prompt.lower()
    assert "股票型" in prompt
    assert "混合型" in prompt


def test_retry_missing_type_prompt_uses_fund_classification_triggers() -> None:
    adapter = _SequenceAdapter(["type: 股票型"])

    table = parse_kv_text(
        "record_id: 748 | secuabbr: 工银创新药ETF | type: ",
        columns=["record_id", "secuabbr", "type"],
        primary_key="record_id",
        anchor_keys=["record_id"],
    )
    retry_missing_values(
        adapter,  # type: ignore[arg-type]
        table,
        entity_groups={
            "748": [
                "战略单元 748（工银创新药ETF）是一只ETF，属于常规基金。"
                "其投资类型为指数型，投资风格专注于行业股票-医药领域，"
                "属于股票型基金，仅支持场内交易。"
            ]
        },
        schema_field_defs={"type": "Fund classification by investment strategy."},
        forced_gaps={"748": ["type"]},
    )

    assert table.records[0].fields["type"] == "股票型"
    assert len(adapter.prompts) == 1
    prompt = adapter.prompts[0]
    assert "Fund field disambiguation" in prompt
    assert "MUTUAL FUND DOMAIN DISAMBIGUATION" in prompt
    assert "基础设施证券投资基金" in prompt
