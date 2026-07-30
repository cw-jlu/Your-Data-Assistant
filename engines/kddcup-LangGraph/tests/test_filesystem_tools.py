from __future__ import annotations

from pathlib import Path

from data_agent_baseline.benchmark.schema import PublicTask, TaskAssets, TaskRecord
from data_agent_baseline.tools.filesystem import list_context_tree, read_doc_preview, resolve_context_path


def _create_task(tmp_path: Path, task_id: str = "task_demo") -> PublicTask:
    task_dir = tmp_path / task_id
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    (context_dir / "notes.md").write_text("# Notes\nhello\n", encoding="utf-8")
    return PublicTask(
        record=TaskRecord(task_id=task_id, difficulty="easy", question="Read notes."),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )


def test_context_paths_accept_context_prefix_and_return_normalized_preview_path(tmp_path: Path) -> None:
    task = _create_task(tmp_path)

    resolved_path = resolve_context_path(task, "context/notes.md")
    preview = read_doc_preview(task, "context/notes.md")

    assert resolved_path == task.context_dir / "notes.md"
    assert preview["path"] == "notes.md"
    assert "hello" in str(preview["preview"])


def test_list_context_tree_uses_relative_root_and_guidance(tmp_path: Path) -> None:
    task = _create_task(tmp_path)

    payload = list_context_tree(task)

    assert payload["root"] == "."
    assert "relative to the context directory" in str(payload["path_convention"])
    assert "context/" in str(payload["path_convention"])
    assert payload["entries"] == [
        {
            "path": "notes.md",
            "kind": "file",
            "size": (task.context_dir / "notes.md").stat().st_size,
        }
    ]
