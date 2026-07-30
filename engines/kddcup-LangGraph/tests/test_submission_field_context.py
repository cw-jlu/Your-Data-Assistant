from __future__ import annotations

import json
from pathlib import Path

from data_agent_baseline.agents.submission_field_context import build_submission_field_context


def _catalog() -> dict[str, object]:
    return {
        "schemas": [
            {
                "kind": "sqlite",
                "asset_path": "stock.sqlite",
                "tables": [
                    {
                        "name": "lc_sharefp",
                        "row_count": 10,
                        "fields": [
                            {"name": "id", "type": "BIGINT", "primary_key": True},
                            {"name": "SecuCode", "type": "VARCHAR"},
                            {"name": "FPSHName", "type": "VARCHAR"},
                            {"name": "InvolvedSum", "type": "DOUBLE"},
                            {"name": "CompanyCode", "type": "BIGINT"},
                        ],
                    },
                    {
                        "name": "配股大股东认配状况",
                        "row_count": 10,
                        "fields": [
                            {"name": "序号", "type": "BIGINT", "primary_key": True},
                            {"name": "公司代码", "type": "BIGINT"},
                            {"name": "股东名称", "type": "VARCHAR"},
                            {"name": "应配股数(股)", "type": "DOUBLE"},
                        ],
                    },
                    {
                        "name": "lc_business",
                        "row_count": 3,
                        "fields": [
                            {"name": "CompanyCode", "type": "BIGINT"},
                            {"name": "IndustryName", "type": "VARCHAR"},
                        ],
                    },
                ],
            }
        ],
        "derived_views": [
            {
                "name": "v_lc_sharefp_enriched",
                "kind": "derived_view",
                "base_table": "lc_sharefp",
                "columns": [
                    {
                        "name": "FPSHName",
                        "type": "VARCHAR",
                        "source_table": "lc_sharefp",
                        "source_field": "FPSHName",
                    },
                    {
                        "name": "IndustryName",
                        "type": "VARCHAR",
                        "source_table": "lc_business",
                        "source_field": "IndustryName",
                    },
                ],
                "joins": [
                    {
                        "dimension_table": "lc_business",
                        "source_fields": ["CompanyCode"],
                        "target_fields": ["CompanyCode"],
                    }
                ],
            }
        ],
    }


def _lineage_for(context: dict[str, object], column: str) -> dict[str, object]:
    for item in context["output_lineage"]:
        if item["output_column"] == column:
            return item
    raise AssertionError(f"missing lineage for {column}")


def _write_structured_doc_manifest(
    context_dir: Path,
    *,
    registered_table: str = "qt_dailyquote",
) -> None:
    manifest_dir = context_dir / ".generated" / "structured_doc"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "manifest.json").write_text(
        json.dumps(
            {
                "tables": [
                    {
                        "source_path": "doc/qt_dailyquote.md",
                        "target_table": "qt_dailyquote",
                        "registered_table": registered_table,
                        "columns": ["secucode", "turnoverdeals", "tradingday"],
                        "primary_key_field": "secucode",
                        "file": f"{registered_table}.jsonl",
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_probe_query_context_binds_aliases_to_source_fields() -> None:
    context = build_submission_field_context(
        {
            "source_tool": "execute_probe_query",
            "source_tool_args": {
                "queries": [
                    "SELECT FPSHName AS shareholder_name, "
                    "InvolvedSum AS shares_involved "
                    "FROM lc_sharefp WHERE SecuCode = '600180'"
                ]
            },
        },
        {"columns": ["shareholder_name", "shares_involved"], "rows": []},
        catalog=_catalog(),
    )

    assert context["status"] == "complete"
    assert any(table["table"] == "lc_sharefp" for table in context["source_tables"])
    assert any(
        field["name"] == "FPSHName"
        for universe in context["field_universe"]
        for field in universe["fields"]
    )
    assert _lineage_for(context, "shareholder_name")["sources"][0]["field"] == "FPSHName"
    assert _lineage_for(context, "shares_involved")["sources"][0]["field"] == "InvolvedSum"


def test_execute_python_context_extracts_static_query_rows_sql() -> None:
    code = (
        "rows = query_rows(\"SELECT DISTINCT 股东名称 FROM 配股大股东认配状况 "
        "WHERE \\\"应配股数(股)\\\" > 2000000 ORDER BY 股东名称\")\n"
        "print(rows)"
    )
    context = build_submission_field_context(
        {"source_tool": "execute_python", "source_tool_args": {"code": code}},
        {"columns": ["股东名称"], "rows": []},
        catalog=_catalog(),
    )

    assert context["status"] == "complete"
    assert any(table["table"] == "配股大股东认配状况" for table in context["source_tables"])
    assert _lineage_for(context, "股东名称")["sources"][0]["field"] == "股东名称"
    assert any(field["field"] == "应配股数(股)" for field in context["filter_fields"])


def test_implicit_join_generates_join_edge() -> None:
    context = build_submission_field_context(
        {
            "source_tool": "execute_probe_query",
            "source_tool_args": {
                "queries": [
                    "SELECT s.FPSHName, b.IndustryName "
                    "FROM lc_sharefp s, lc_business b "
                    "WHERE s.CompanyCode = b.CompanyCode"
                ]
            },
        },
        {"columns": ["FPSHName", "IndustryName"], "rows": []},
        catalog=_catalog(),
    )

    assert context["status"] == "complete"
    assert context["join_edges"][0]["left"]["qualified_name"] == "s.CompanyCode"
    assert context["join_edges"][0]["right"]["qualified_name"] == "b.CompanyCode"


def test_derived_view_context_uses_view_fields_and_embedded_joins() -> None:
    context = build_submission_field_context(
        {
            "source_tool": "execute_probe_query",
            "source_tool_args": {
                "queries": ["SELECT FPSHName, IndustryName FROM v_lc_sharefp_enriched"]
            },
        },
        {"columns": ["FPSHName", "IndustryName"], "rows": []},
        catalog=_catalog(),
    )

    universe = context["field_universe"][0]
    assert universe["table"] == "v_lc_sharefp_enriched"
    assert universe["kind"] == "derived_view"
    assert universe["joins"]
    assert {field["name"] for field in universe["fields"]} == {"FPSHName", "IndustryName"}


def test_dynamic_python_sql_returns_partial_warning() -> None:
    context = build_submission_field_context(
        {
            "source_tool": "execute_python",
            "source_tool_args": {
                "code": "table = 'lc_sharefp'\nrows = query_rows(f'SELECT FPSHName FROM {table}')"
            },
        },
        {"columns": ["FPSHName"], "rows": []},
        catalog=_catalog(),
    )

    assert context["status"] == "partial"
    assert any(warning["kind"] == "dynamic_python_sql" for warning in context["warnings"])


def test_structured_doc_manifest_supplies_registered_table_fields(tmp_path: Path) -> None:
    _write_structured_doc_manifest(tmp_path)

    context = build_submission_field_context(
        {
            "source_tool": "execute_probe_query",
            "source_tool_args": {
                "queries": [
                    "SELECT tradingday, turnoverdeals "
                    "FROM qt_dailyquote WHERE secucode = '601908'"
                ]
            },
        },
        {"columns": ["tradingday", "turnoverdeals"], "rows": []},
        catalog={"schemas": []},
        context_dir=tmp_path,
    )

    assert context["status"] == "complete"
    universe = context["field_universe"][0]
    assert universe["table"] == "qt_dailyquote"
    assert universe["kind"] == "structured_doc_table"
    assert universe["source_path"] == "doc/qt_dailyquote.md"
    assert universe["target_table"] == "qt_dailyquote"
    assert {field["name"] for field in universe["fields"]} == {
        "secucode",
        "turnoverdeals",
        "tradingday",
    }
    assert _lineage_for(context, "turnoverdeals")["sources"][0]["field"] == "turnoverdeals"


def test_structured_doc_manifest_handles_extracted_conflict_table(tmp_path: Path) -> None:
    _write_structured_doc_manifest(tmp_path, registered_table="qt_dailyquote_extracted")

    context = build_submission_field_context(
        {
            "source_tool": "execute_probe_query",
            "source_tool_args": {
                "queries": ["SELECT turnoverdeals FROM qt_dailyquote_extracted"]
            },
        },
        {"columns": ["turnoverdeals"], "rows": []},
        catalog={"schemas": []},
        context_dir=tmp_path,
    )

    assert context["status"] == "complete"
    assert context["field_universe"][0]["table"] == "qt_dailyquote_extracted"
    assert context["field_universe"][0]["registered_table"] == "qt_dailyquote_extracted"


def test_structured_doc_manifest_invalid_json_does_not_raise(tmp_path: Path) -> None:
    manifest_dir = tmp_path / ".generated" / "structured_doc"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "manifest.json").write_text("{", encoding="utf-8")

    context = build_submission_field_context(
        {
            "source_tool": "execute_probe_query",
            "source_tool_args": {"queries": ["SELECT turnoverdeals FROM qt_dailyquote"]},
        },
        {"columns": ["turnoverdeals"], "rows": []},
        catalog={"schemas": []},
        context_dir=tmp_path,
    )

    assert context["status"] == "partial"
    assert any(
        warning["kind"] == "structured_doc_manifest_read_error"
        for warning in context["warnings"]
    )
    assert any(
        warning["kind"] == "source_table_not_found_in_catalog"
        for warning in context["warnings"]
    )


def test_structured_doc_manifest_invalid_tables_shape_does_not_raise(tmp_path: Path) -> None:
    manifest_dir = tmp_path / ".generated" / "structured_doc"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "manifest.json").write_text('{"tables": {}}', encoding="utf-8")

    context = build_submission_field_context(
        {
            "source_tool": "execute_probe_query",
            "source_tool_args": {"queries": ["SELECT turnoverdeals FROM qt_dailyquote"]},
        },
        {"columns": ["turnoverdeals"], "rows": []},
        catalog={"schemas": []},
        context_dir=tmp_path,
    )

    assert context["status"] == "partial"
    assert any(
        warning["kind"] == "structured_doc_manifest_invalid" for warning in context["warnings"]
    )


def test_structured_doc_table_can_join_catalog_table(tmp_path: Path) -> None:
    _write_structured_doc_manifest(tmp_path)

    context = build_submission_field_context(
        {
            "source_tool": "execute_probe_query",
            "source_tool_args": {
                "queries": [
                    "SELECT q.tradingday, b.IndustryName "
                    "FROM qt_dailyquote q "
                    "JOIN lc_business b ON q.secucode = b.CompanyCode"
                ]
            },
        },
        {"columns": ["tradingday", "IndustryName"], "rows": []},
        catalog=_catalog(),
        context_dir=tmp_path,
    )

    assert context["status"] == "complete"
    assert {universe["kind"] for universe in context["field_universe"]} == {
        "structured_doc_table",
        "sqlite",
    }
    assert context["join_edges"][0]["left"]["qualified_name"] == "q.secucode"
    assert context["join_edges"][0]["right"]["qualified_name"] == "b.CompanyCode"
