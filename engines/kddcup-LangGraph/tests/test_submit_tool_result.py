"""Tests for submit_tool_result tool."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from data_agent_baseline.benchmark.schema import PublicTask, TaskRecord, TaskAssets
from data_agent_baseline.tools.registry import (
    ToolRuntimeContext,
    create_default_tool_registry,
    _extract_answer_from_probe_query,
    _extract_answer_from_python,
    _extract_answer_from_context_sql,
    _submit_tool_result,
)
from data_agent_baseline.tools.python_exec import TaskContextWorkspace


# ---------------------------------------------------------------------------
# Extractor unit tests
# ---------------------------------------------------------------------------


def test_extract_answer_from_probe_query_last_successful():
    content = {
        "ok": True,
        "results": [
            {"ok": True, "columns": ["a"], "rows": [[1]], "row_count": 1},
            {"ok": True, "columns": ["x", "y"], "rows": [[10, 20], [30, 40]], "row_count": 2},
        ],
        "query_count": 2,
    }
    columns, rows = _extract_answer_from_probe_query(content)
    assert columns == ["x", "y"]
    assert rows == [[10, 20], [30, 40]]


def test_extract_answer_from_probe_query_skips_failed():
    content = {
        "ok": False,
        "results": [
            {"ok": False, "error": "table not found", "sql": "SELECT * FROM bad"},
            {"ok": True, "columns": ["name"], "rows": [["Alice"]], "row_count": 1},
        ],
        "query_count": 2,
    }
    columns, rows = _extract_answer_from_probe_query(content)
    assert columns == ["name"]
    assert rows == [["Alice"]]


def test_extract_answer_from_probe_query_all_failed():
    content = {
        "ok": False,
        "results": [{"ok": False, "error": "bad"}],
        "query_count": 1,
    }
    with pytest.raises(ValueError, match="No successful query result"):
        _extract_answer_from_probe_query(content)


def test_extract_answer_from_python_valid_json():
    output = json.dumps({"columns": ["id", "name"], "rows": [[1, "Alice"], [2, "Bob"]]})
    content = {"success": True, "output": output, "stderr": ""}
    columns, rows = _extract_answer_from_python(content)
    assert columns == ["id", "name"]
    assert rows == [[1, "Alice"], [2, "Bob"]]


def test_extract_answer_from_python_with_extra_output():
    content = {
        "success": True,
        "output": "Processing...\nDone.\n" + json.dumps({"columns": ["v"], "rows": [[42]]}),
        "stderr": "",
    }
    columns, rows = _extract_answer_from_python(content)
    assert columns == ["v"]
    assert rows == [[42]]


def test_extract_answer_from_python_with_nested_json_cell():
    output = json.dumps(
        {
            "columns": ["id", "payload"],
            "rows": [[1, {"nested": "value"}]],
        },
        ensure_ascii=False,
    )
    content = {"success": True, "output": "debug\n" + output, "stderr": ""}
    columns, rows = _extract_answer_from_python(content)
    assert columns == ["id", "payload"]
    assert rows == [[1, {"nested": "value"}]]


def test_extract_answer_from_python_rejects_dict_rows():
    output = json.dumps(
        {
            "columns": ["education", "count"],
            "rows": [{"education": "Master's degree", "count": 33}],
        },
        ensure_ascii=False,
    )
    content = {"success": True, "output": output, "stderr": ""}

    with pytest.raises(ValueError, match="rows must be list\\[list\\]"):
        _extract_answer_from_python(content)


def test_extract_answer_from_python_no_json():
    content = {"success": True, "output": "Just some text without JSON", "stderr": ""}
    with pytest.raises(ValueError, match="does not contain a valid JSON"):
        _extract_answer_from_python(content)


def test_extract_answer_from_python_invalid_json():
    content = {"success": True, "output": "{bad json}", "stderr": ""}
    with pytest.raises(ValueError, match="Failed to parse JSON"):
        _extract_answer_from_python(content)


def test_extract_answer_from_python_missing_keys():
    content = {"success": True, "output": json.dumps({"data": []}), "stderr": ""}
    with pytest.raises(ValueError, match="must contain 'columns'"):
        _extract_answer_from_python(content)


def test_extract_answer_from_context_sql():
    content = {"columns": ["a", "b"], "rows": [[1, 2], [3, 4]]}
    columns, rows = _extract_answer_from_context_sql(content)
    assert columns == ["a", "b"]
    assert rows == [[1, 2], [3, 4]]


def test_extract_answer_from_context_sql_missing():
    with pytest.raises(ValueError, match="did not return columns/rows"):
        _extract_answer_from_context_sql({})


# ---------------------------------------------------------------------------
# Handler integration test (using a mock registry)
# ---------------------------------------------------------------------------


def _create_task(tmp_path: Path) -> PublicTask:
    context_dir = tmp_path / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    (context_dir / "students.csv").write_text(
        "name,score\nAlice,95\nBob,87\n",
        encoding="utf-8",
    )
    return PublicTask(
        record=TaskRecord(task_id="test_task", difficulty="easy", question="test?"),
        assets=TaskAssets(task_dir=tmp_path, context_dir=context_dir),
    )


def _create_large_csv_task(tmp_path: Path, row_count: int = 300) -> PublicTask:
    context_dir = tmp_path / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    rows = "\n".join(f"{i},val{i}" for i in range(row_count))
    (context_dir / "big.csv").write_text(f"id,value\n{rows}\n", encoding="utf-8")
    return PublicTask(
        record=TaskRecord(task_id="large_task", difficulty="easy", question="test?"),
        assets=TaskAssets(task_dir=tmp_path, context_dir=context_dir),
    )


def test_submit_tool_result_unsupported_tool(tmp_path: Path):
    task = _create_task(tmp_path)
    workspace = TaskContextWorkspace(task.context_dir)
    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(task=task, python_workspace=workspace, registry=registry)

    result = _submit_tool_result(
        runtime_context,
        {"tool_name": "nonexistent_tool", "tool_args": {}},
    )
    assert result.ok is False
    assert "Unsupported source tool" in result.content["error"]


def test_submit_tool_result_probe_query_does_not_need_registry(tmp_path: Path):
    task = _create_task(tmp_path)
    workspace = TaskContextWorkspace(task.context_dir)
    runtime_context = ToolRuntimeContext(task=task, python_workspace=workspace, registry=None)

    result = _submit_tool_result(
        runtime_context,
        {"tool_name": "execute_probe_query", "tool_args": {"queries": ["SELECT 1"]}},
    )
    assert result.ok is True
    assert result.answer is not None
    assert result.answer.rows == [[1]]


def test_submit_tool_result_column_count_mismatch(tmp_path: Path):
    """When the user specifies columns with wrong count, it should error."""
    task = _create_task(tmp_path)
    workspace = TaskContextWorkspace(task.context_dir)
    runtime_context = ToolRuntimeContext(task=task, python_workspace=workspace)

    result = _submit_tool_result(
        runtime_context,
        {
            "tool_name": "execute_probe_query",
            "tool_args": {"queries": ["SELECT name, score FROM students"]},
            "columns": ["only_one"],  # Wrong count
        },
    )
    assert result.ok is False
    assert "Column count mismatch" in result.content["error"]


def test_submit_tool_result_success(tmp_path: Path):
    """End-to-end test returning a valid answer."""
    task = _create_task(tmp_path)
    workspace = TaskContextWorkspace(task.context_dir)
    runtime_context = ToolRuntimeContext(task=task, python_workspace=workspace)

    result = _submit_tool_result(
        runtime_context,
        {
            "tool_name": "execute_probe_query",
            "tool_args": {"queries": ["SELECT name, score FROM students ORDER BY score DESC"], "limit": 200},
        },
    )

    assert result.ok is True
    assert result.is_terminal is True
    assert result.answer is not None
    assert result.answer.columns == ["name", "score"]
    assert result.answer.rows == [["Alice", 95], ["Bob", 87]]
    assert result.content["status"] == "submitted"
    assert result.content["source_tool"] == "execute_probe_query"
    assert result.content["column_count"] == 2
    assert result.content["row_count"] == 2
    assert result.answer_submission == {
        "submission_tool": "submit_tool_result",
        "source_tool": "execute_probe_query",
        "source_tool_args": {
            "queries": ["SELECT name, score FROM students ORDER BY score DESC"],
            "limit": 200,
        },
        "column_override": None,
        "source_output_columns": ["name", "score"],
        "final_columns": ["name", "score"],
        "selected_query_index": 0,
    }


def test_submit_tool_result_with_column_override(tmp_path: Path):
    """Test that columns can be renamed via the columns parameter."""
    task = _create_task(tmp_path)
    workspace = TaskContextWorkspace(task.context_dir)
    runtime_context = ToolRuntimeContext(task=task, python_workspace=workspace)

    result = _submit_tool_result(
        runtime_context,
        {
            "tool_name": "execute_probe_query",
            "tool_args": {"queries": ["SELECT name AS first_name, CAST(score AS VARCHAR) AS last_name FROM students WHERE name = 'Alice'"]},
            "columns": ["given_name", "family_name"],
        },
    )

    assert result.ok is True
    assert result.answer is not None
    assert result.answer.columns == ["given_name", "family_name"]
    assert result.answer.rows == [["Alice", "95"]]
    assert result.answer_submission["source_output_columns"] == ["first_name", "last_name"]
    assert result.answer_submission["final_columns"] == ["given_name", "family_name"]
    assert result.answer_submission["selected_query_index"] == 0


def test_submit_tool_result_rejects_extract_structured_doc_source_tool(tmp_path: Path):
    task = _create_task(tmp_path)
    registry = create_default_tool_registry()
    acquired: list[str] = []

    class FakeGate:
        def acquire(self, tool_name: str, **metadata: object) -> None:
            acquired.append(tool_name)

        def release(self, tool_name: str) -> None:
            acquired.append(f"release:{tool_name}")

    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(task.context_dir),
        registry=registry,
        tool_gate=FakeGate(),
    )

    result = _submit_tool_result(
        runtime_context,
        {
            "tool_name": "extract_structured_doc",
            "tool_args": {
                "path": "doc/managers.md",
                "target_table": "managers",
                "fields": ["personalcode"],
                "max_model_calls": 4,
            },
        },
    )

    assert result.ok is False
    assert result.answer is None
    assert "Unsupported source tool: 'extract_structured_doc'" in result.content["error"]
    assert "execute_probe_query" in result.content["error"]
    assert "execute_python" in result.content["error"]
    assert acquired == []


def test_submit_tool_result_probe_query_ignores_preview_limit(tmp_path: Path):
    task = _create_large_csv_task(tmp_path, row_count=300)
    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(task.context_dir),
    )

    preview = registry.execute(
        runtime_context,
        "execute_probe_query",
        {"queries": ["SELECT * FROM big ORDER BY id"], "limit": 200},
    )
    assert preview.ok is True
    assert preview.content["results"][0]["row_count"] == 200
    assert preview.content["results"][0]["truncated"] is True

    result = _submit_tool_result(
        runtime_context,
        {
            "tool_name": "execute_probe_query",
            "tool_args": {"queries": ["SELECT * FROM big ORDER BY id"], "limit": 3},
        },
    )

    assert result.ok is True
    assert result.answer is not None
    assert result.answer.columns == ["id", "value"]
    assert len(result.answer.rows) == 300
    assert result.answer.rows[0] == [0, "val0"]
    assert result.answer.rows[-1] == [299, "val299"]




def test_submit_tool_result_execute_python_can_submit_query_helper_output(tmp_path: Path):
    task = _create_task(tmp_path)
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(task.context_dir),
    )
    code = (
        "import json\n"
        "print(json.dumps(query('SELECT name, score FROM students ORDER BY score DESC'), "
        "ensure_ascii=False))"
    )

    result = _submit_tool_result(
        runtime_context,
        {
            "tool_name": "execute_python",
            "tool_args": {"code": code},
        },
    )

    assert result.ok is True
    assert result.answer is not None
    assert result.answer.columns == ["name", "score"]
    assert result.answer.rows == [["Alice", 95], ["Bob", 87]]
    assert result.answer_submission == {
        "submission_tool": "submit_tool_result",
        "source_tool": "execute_python",
        "source_tool_args": {"code": code},
        "column_override": None,
        "source_output_columns": ["name", "score"],
        "final_columns": ["name", "score"],
        "selected_query_index": None,
    }
