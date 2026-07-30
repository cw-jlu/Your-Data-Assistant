"""Explorer registry：video_tool 注入开关 + report 运行时校验 + 定向模式字段默认。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents.benchmark.schema import PublicTask, TaskAssets, TaskRecord
from agents.explorer.registry import ReportInput, create_explorer_tool_registry
from agents.tools.registry import FunctionTool, ToolExecutionResult


def _make_task(tmp_path: Path, *, question: str = "q") -> PublicTask:
    task_dir = tmp_path / "task_1"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    return PublicTask(
        record=TaskRecord(task_id="task_1", difficulty="easy", question=question),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )


def _stub_video_tool() -> FunctionTool:
    return FunctionTool(
        name="explore_video",
        description="stub",
        json_schema=None,
        handler=lambda _t, _a: ToolExecutionResult(ok=True, content={}),
    )


def test_video_tool_absent_by_default() -> None:
    registry = create_explorer_tool_registry()
    assert "explore_video" not in registry.definitions


def test_video_tool_injected_when_provided() -> None:
    registry = create_explorer_tool_registry(video_tool=_stub_video_tool())
    assert "explore_video" in registry.definitions
    assert registry.definitions["report"].is_terminal is True


def test_execute_context_sql_description_matches_local_index_contract() -> None:
    registry = create_explorer_tool_registry()
    description = registry.definitions["execute_context_sql"].description.lower()

    assert "virtual-context sqlite database copy" in description
    assert "create index if not exists" in description
    assert "read-only sql query" not in description


def test_inspect_files_description_mentions_pdf_entity_group_samples() -> None:
    registry = create_explorer_tool_registry()
    description = registry.definitions["inspect_files"].description

    assert "page_count/paragraph_count/entity_groups_sample for PDF" in description
    assert "head for docs" in description


def test_preview_file_description_matches_sqlite_row_count_shape() -> None:
    registry = create_explorer_tool_registry()
    description = registry.definitions["preview_file"].description

    assert "{name, create_sql, row_count}" in description
    assert "PDF → {page_count, paragraph_count, entity_groups_sample}" in description
    assert "inspect_files already returns this PDF entity-group shape" in description
    assert "targeted single-file re-check" in description


def test_preview_file_is_exposed_to_explorer_openai_tools() -> None:
    registry = create_explorer_tool_registry()
    tools = registry.to_openai_tools()
    by_name = {tool["function"]["name"]: tool["function"] for tool in tools}

    assert "preview_file" in by_name
    preview_tool = by_name["preview_file"]
    assert (
        "PDF → {page_count, paragraph_count, entity_groups_sample}" in preview_tool["description"]
    )
    parameters = preview_tool["parameters"]
    assert parameters["type"] == "object"
    assert parameters["required"] == ["path"]
    assert "path" in parameters["properties"]


def test_report_now_validated_at_registry_boundary(tmp_path: Path) -> None:
    registry = create_explorer_tool_registry()
    assert registry.definitions["report"].input_model is ReportInput
    with pytest.raises(ValueError, match="report: extra field"):
        registry.execute(_make_task(tmp_path), "report", {"bogus_field": 1})


def test_report_files_and_schema_map_default_empty(tmp_path: Path) -> None:
    # 定向模式允许精简报告：files / schema_map 不再必填
    registry = create_explorer_tool_registry()
    result = registry.execute(_make_task(tmp_path), "report", {"knowledge": {"answer_hint": "x"}})
    assert result.ok is True and result.is_terminal is True


@pytest.mark.parametrize("field", ["schema_map", "knowledge", "value_samples"])
def test_report_rejects_scalar_dict_fields(tmp_path: Path, field: str) -> None:
    registry = create_explorer_tool_registry()
    with pytest.raises(ValueError, match=rf"report: field '{field}'"):
        registry.execute(_make_task(tmp_path), "report", {field: "not a dict"})


def test_report_value_samples_preserve_scalar_types(tmp_path: Path) -> None:
    registry = create_explorer_tool_registry()
    result = registry.execute(
        _make_task(tmp_path),
        "report",
        {
            "value_samples": {
                "db/sub_db.sqlite.mf_fmretscaleanalysis.TotalAUM": [0.171053, 405.178231],
                "csv/foo.category": ["本科", "硕士"],
                "csv/foo.count": [1, 2],
            }
        },
    )

    assert result.ok is True and result.is_terminal is True
    findings = json.loads(result.content["findings"])
    assert findings["value_samples"] == {
        "db/sub_db.sqlite.mf_fmretscaleanalysis.TotalAUM": [0.171053, 405.178231],
        "csv/foo.category": ["本科", "硕士"],
        "csv/foo.count": [1, 2],
    }


def test_report_etl_sources_preserved(tmp_path: Path) -> None:
    registry = create_explorer_tool_registry()
    result = registry.execute(
        _make_task(tmp_path),
        "report",
        {
            "etl_sources": [
                {
                    "path": "doc/profiles.pdf",
                    "reason": "contains required profile records",
                }
            ]
        },
    )

    assert result.ok is True and result.is_terminal is True
    findings = json.loads(result.content["findings"])
    assert findings["etl_sources"] == [
        {
            "path": "doc/profiles.pdf",
            "reason": "contains required profile records",
        }
    ]


def test_report_rejects_missing_same_stem_doc_etl_for_knowledge_table(tmp_path: Path) -> None:
    question = "管理基金规模超过100亿的基金经理最高学历分布情况"
    task = _make_task(tmp_path, question=question)
    (task.context_dir / "knowledge.md").write_text(
        "### Use Case 3\n"
        f"Business Question: {question}\n"
        "Formula: `mf_fmscaleanalysisn.totalfundnv`\n"
        "### Use Case 4\n"
        "Business Question: 任职天数超过1000天的基金经理所管理的基金名称和基金风险等级\n"
        "Formula: `mf_fundmanagernew.managementtime`",
        encoding="utf-8",
    )
    doc_dir = task.context_dir / "doc"
    doc_dir.mkdir()
    (doc_dir / "mf_fmscaleanalysisn.pdf").write_bytes(b"%PDF-1.4\n")
    (doc_dir / "mf_fundmanagernew.md").write_text("unrelated use case", encoding="utf-8")

    registry = create_explorer_tool_registry()
    result = registry.execute(
        task,
        "report",
        {
            "files": [
                {
                    "path": "json/mf_fmretscaleanalysis.json",
                    "format": "JSON",
                    "row_count": 6041,
                }
            ],
            "schema_map": {"mf_fmretscaleanalysis": {"columns": {}}},
            "knowledge": {
                "knowledge_md_evidence": [
                    {
                        "source": "knowledge.md",
                        "kind": "example_sql",
                        "text": (
                            "SELECT a.education, COUNT(*) FROM mf_personalinfo AS a "
                            "JOIN mf_fmscaleanalysisn AS b ON a.personalcode = "
                            "b.personalcode WHERE b.totalfundnv > 100 GROUP BY "
                            "a.education;"
                        ),
                    }
                ]
            },
            "etl_sources": [],
        },
    )

    assert result.ok is False and result.is_terminal is False
    assert "report.etl_sources is missing required same-stem document" in result.content["error"]
    assert result.content["required_etl_sources"] == [
        {
            "path": "doc/mf_fmscaleanalysisn.pdf",
            "reason": (
                "knowledge.md references table mf_fmscaleanalysisn, no exact "
                "structured source named mf_fmscaleanalysisn was reported, and "
                "this same-stem document contains the table records"
            ),
        }
    ]


def test_report_accepts_required_same_stem_doc_etl_for_knowledge_table(tmp_path: Path) -> None:
    question = "管理基金规模超过100亿的基金经理最高学历分布情况"
    task = _make_task(tmp_path, question=question)
    (task.context_dir / "knowledge.md").write_text(
        "### Use Case 3\n"
        f"Business Question: {question}\n"
        "Formula: `mf_fmscaleanalysisn.totalfundnv`\n",
        encoding="utf-8",
    )
    doc_dir = task.context_dir / "doc"
    doc_dir.mkdir()
    (doc_dir / "mf_fmscaleanalysisn.pdf").write_bytes(b"%PDF-1.4\n")

    registry = create_explorer_tool_registry()
    result = registry.execute(
        task,
        "report",
        {
            "files": [{"path": "json/mf_fmretscaleanalysis.json", "format": "JSON"}],
            "schema_map": {"mf_fmretscaleanalysis": {"columns": {}}},
            "knowledge": {
                "knowledge_md_evidence": [
                    {
                        "source": "knowledge.md",
                        "kind": "formula",
                        "text": "`mf_fmscaleanalysisn.totalfundnv`",
                    }
                ]
            },
            "etl_sources": [
                {
                    "path": "doc/mf_fmscaleanalysisn.pdf",
                    "reason": "contains scale-analysis records for mf_fmscaleanalysisn",
                }
            ],
        },
    )

    assert result.ok is True and result.is_terminal is True
    findings = json.loads(result.content["findings"])
    assert findings["etl_sources"] == [
        {
            "path": "doc/mf_fmscaleanalysisn.pdf",
            "reason": "contains scale-analysis records for mf_fmscaleanalysisn",
        }
    ]
