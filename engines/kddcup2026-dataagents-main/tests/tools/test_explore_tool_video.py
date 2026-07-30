"""explore 工具：explore_video 按任务动态注入、video_findings 注入。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import agents.explorer.runner as runner_mod
from agents.benchmark.schema import PublicTask, TaskAssets, TaskRecord
from agents.explorer.runner import ExplorerResult
from agents.tools.explore import ExploreInput, create_explore_tool_definition
from agents.tools.registry import FunctionTool, ToolExecutionResult


class _Adapter:
    def complete(self, messages, *, tools=None, **kwargs):  # pragma: no cover
        raise AssertionError("not called")


def _video_tool() -> FunctionTool:
    return FunctionTool(
        name="explore_video",
        description="stub",
        json_schema=None,
        handler=lambda _t, _a: ToolExecutionResult(ok=True, content={}),
    )


def _make_task(tmp_path: Path, *, with_video: bool) -> PublicTask:
    task_dir = tmp_path / "task_1"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    if with_video:
        (context_dir / "demo.mp4").write_bytes(b"\x00")
    return PublicTask(
        record=TaskRecord(task_id="task_1", difficulty="easy", question="What is X?"),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )


def test_explore_tool_description_matches_discovery_contract() -> None:
    tool = create_explore_tool_definition(_Adapter())
    description = tool.description.lower()

    assert "discovery-only" in description
    assert "read-only sub-agent" not in description
    assert "pdf entity_groups_sample" in description
    assert "etl_sources" in description


@pytest.fixture
def recorded(monkeypatch: pytest.MonkeyPatch) -> dict:
    seen: dict = {}

    def fake_run_explorer(task, question, model, *, video_tool=None, **kwargs):
        seen.update(question=question, model=model, video_tool=video_tool)
        return ExplorerResult(
            findings='{"files": [], "schema_map": {}}',
            steps_used=2,
            success=True,
            video_findings=({"video_status": "ok", "findings": {"summary": "v"}},),
        )

    monkeypatch.setattr(runner_mod, "run_explorer", fake_run_explorer)
    return seen


@pytest.mark.parametrize("task_has_video", [True, False], ids=["video-task", "plain-task"])
@pytest.mark.parametrize("pass_video_tool", [True, False], ids=["with-tool", "no-tool"])
def test_video_tool_injected_only_when_task_has_video(
    tmp_path: Path, recorded: dict, task_has_video: bool, pass_video_tool: bool
) -> None:
    adapter = _Adapter()
    tool = create_explore_tool_definition(
        adapter, video_tool=_video_tool() if pass_video_tool else None
    )
    task = _make_task(tmp_path, with_video=task_has_video)

    result = tool.handler(task, ExploreInput())

    assert result.ok is True
    assert recorded["model"] is adapter
    if task_has_video and pass_video_tool:
        assert recorded["video_tool"] is not None
        assert result.content["video_findings"][0]["findings"]["summary"] == "v"
    else:
        assert recorded["video_tool"] is None
        assert "video_findings" not in result.content


def test_explore_tool_marks_fallback_findings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run_explorer(task, question, model, *, video_tool=None, **kwargs):
        del task, question, model, video_tool, kwargs
        return ExplorerResult(
            findings='{"files": []}',
            steps_used=11,
            success=True,
            fallback_used=True,
        )

    monkeypatch.setattr(runner_mod, "run_explorer", fake_run_explorer)
    tool = create_explore_tool_definition(_Adapter())
    task = _make_task(tmp_path, with_video=False)

    result = tool.handler(task, ExploreInput())

    assert result.ok is True
    assert result.content["explorer_steps_used"] == 11
    assert result.content["explorer_fallback_used"] is True


def test_explore_tool_forces_same_stem_doc_etl_for_missing_knowledge_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run_explorer(task, question, model, *, video_tool=None, **kwargs):
        del task, question, model, video_tool, kwargs
        return ExplorerResult(
            findings=json.dumps(
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
                                    "SELECT a.education, COUNT(*) FROM "
                                    "`mf_personalinfo` AS a JOIN "
                                    "`mf_fmscaleanalysisn` AS b "
                                    "WHERE b.totalfundnv > 100"
                                ),
                            }
                        ]
                    },
                    "etl_sources": [],
                }
            ),
            steps_used=4,
            success=True,
        )

    monkeypatch.setattr(runner_mod, "run_explorer", fake_run_explorer)
    task = _make_task(tmp_path, with_video=False)
    (task.context_dir / "knowledge.md").write_text(
        "### Use Case 3\n"
        "Business Question: 管理基金规模超过100亿的基金经理最高学历分布情况\n"
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

    tool = create_explore_tool_definition(_Adapter())
    result = tool.handler(task, ExploreInput())

    assert result.ok is True
    assert result.content["etl_sources"] == [
        {
            "path": "doc/mf_fmscaleanalysisn.pdf",
            "reason": (
                "knowledge.md references table mf_fmscaleanalysisn, no exact "
                "structured source named mf_fmscaleanalysisn was reported, and "
                "this same-stem document contains the table records"
            ),
        }
    ]


def test_explore_tool_does_not_force_doc_etl_when_exact_structured_table_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run_explorer(task, question, model, *, video_tool=None, **kwargs):
        del task, question, model, video_tool, kwargs
        return ExplorerResult(
            findings=json.dumps(
                {
                    "files": [
                        {
                            "path": "db/sub_db.sqlite",
                            "format": "SQLite",
                            "tables": ["mf_fmscaleanalysisn"],
                        }
                    ],
                    "knowledge": {
                        "knowledge_md_evidence": [
                            {
                                "source": "knowledge.md",
                                "kind": "formula",
                                "text": "`mf_fmscaleanalysisn.totalfundnv`",
                            }
                        ]
                    },
                    "etl_sources": [],
                }
            ),
            steps_used=3,
            success=True,
        )

    monkeypatch.setattr(runner_mod, "run_explorer", fake_run_explorer)
    task = _make_task(tmp_path, with_video=False)
    doc_dir = task.context_dir / "doc"
    doc_dir.mkdir()
    (doc_dir / "mf_fmscaleanalysisn.pdf").write_bytes(b"%PDF-1.4\n")

    tool = create_explore_tool_definition(_Adapter())
    result = tool.handler(task, ExploreInput())

    assert result.ok is True
    assert result.content["etl_sources"] == []
