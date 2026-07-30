"""run_etl tool tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import agents.tools.run_etl as run_etl_mod
from agents.benchmark.schema import PublicTask, TaskAssets, TaskRecord
from agents.etl._detect import ETLResult
from agents.llm import ModelMessage, ModelResponse
from agents.tools.registry import ToolRegistry
from agents.tools.run_etl import create_run_etl_tool_definition


def _make_task(tmp_path: Path) -> PublicTask:
    task_dir = tmp_path / "task_1"
    context_dir = task_dir / "context"
    (context_dir / "doc").mkdir(parents=True)
    return PublicTask(
        record=TaskRecord(task_id="task_1", difficulty="easy", question="?"),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )


class _FakeModelAdapter:
    def complete(
        self, messages: list[ModelMessage], *, tools: object | None = None, **kwargs: Any
    ) -> ModelResponse:
        del messages, tools, kwargs
        raise AssertionError("run_etl_for_task is monkeypatched in these tests")


def test_run_etl_exposes_selected_csv_under_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = _make_task(tmp_path)
    source_doc = task.context_dir / "doc" / "source.md"
    source_doc.write_text("id: 1\n", encoding="utf-8")
    csv_path = tmp_path / "scratch" / "task_1" / "_etl" / "source.csv"
    csv_path.parent.mkdir(parents=True)
    csv_path.write_text("id,value\n1,a\n", encoding="utf-8")
    model = _FakeModelAdapter()
    selected_seen: list[Path] = []

    def fake_run_etl_for_task(
        _task: PublicTask,
        seen_model: object,
        *,
        selected_prose_files: list[Path] | None = None,
    ) -> list[ETLResult]:
        assert seen_model is model
        selected_seen.extend(selected_prose_files or [])
        return [
            ETLResult(
                source_file="source.md",
                csv_path=str(csv_path),
                columns=["id", "value"],
                row_count=1,
            )
        ]

    monkeypatch.setattr(run_etl_mod, "run_etl_for_task", fake_run_etl_for_task)
    registry = ToolRegistry(definitions={"run_etl": create_run_etl_tool_definition(lambda: model)})

    result = registry.execute(task, "run_etl", {"paths": ["doc/source.md"]})

    assert result.ok is True
    assert selected_seen == [source_doc.resolve()]
    assert result.content["status"] == "ok"
    assert result.content["csv_files"] == [
        {
            "source_file": "source.md",
            "csv_path": "csv/source.csv",
            "columns": ["id", "value"],
            "row_count": 1,
        }
    ]
    assert (task.context_dir / "csv" / "source.csv").read_text(encoding="utf-8") == (
        "id,value\n1,a\n"
    )


def test_run_etl_reports_no_output_for_selected_document(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = _make_task(tmp_path)
    (task.context_dir / "doc" / "source.md").write_text("incidental prose\n", encoding="utf-8")

    def fake_run_etl_for_task(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(run_etl_mod, "run_etl_for_task", fake_run_etl_for_task)
    registry = ToolRegistry(
        definitions={"run_etl": create_run_etl_tool_definition(lambda: _FakeModelAdapter())}
    )

    result = registry.execute(task, "run_etl", {"paths": ["doc/source.md"]})

    assert result.ok is True
    assert result.content["status"] == "no_output"
    assert result.content["csv_files"] == []
    assert result.content["skipped_sources"] == ["source.md"]


def test_run_etl_rejects_non_prose_paths(tmp_path: Path) -> None:
    task = _make_task(tmp_path)
    (task.context_dir / "records.csv").write_text("id\n1\n", encoding="utf-8")
    registry = ToolRegistry(
        definitions={"run_etl": create_run_etl_tool_definition(lambda: _FakeModelAdapter())}
    )

    with pytest.raises(ValueError, match="not an ETL prose document"):
        registry.execute(task, "run_etl", {"paths": ["records.csv"]})


def test_run_etl_rejects_knowledge_md(tmp_path: Path) -> None:
    task = _make_task(tmp_path)
    (task.context_dir / "knowledge.md").write_text("# Knowledge\n", encoding="utf-8")
    registry = ToolRegistry(
        definitions={"run_etl": create_run_etl_tool_definition(lambda: _FakeModelAdapter())}
    )

    with pytest.raises(ValueError, match=r"knowledge\.md is schema guidance"):
        registry.execute(task, "run_etl", {"paths": ["knowledge.md"]})
