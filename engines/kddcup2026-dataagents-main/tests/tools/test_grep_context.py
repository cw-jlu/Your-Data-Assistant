"""Regression tests for grep_context."""

from __future__ import annotations

import contextlib
import sqlite3
from pathlib import Path

from agents.benchmark.schema import PublicTask, TaskAssets, TaskRecord
from agents.tools.grep_context import run_grep_context


def _make_task(tmp_path: Path) -> PublicTask:
    task_dir = tmp_path / "task_grep"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    return PublicTask(
        record=TaskRecord(task_id="task_grep", difficulty="easy", question="?"),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )


def test_grep_context_sqlite_uses_regex_semantics(tmp_path: Path) -> None:
    task = _make_task(tmp_path)
    db_path = task.context_dir / "main.sqlite"
    with contextlib.closing(sqlite3.connect(db_path)) as conn:
        conn.execute("CREATE TABLE notes (id INTEGER, label TEXT)")
        conn.executemany(
            "INSERT INTO notes VALUES (?, ?)",
            [
                (1, "全国合计"),
                (2, "中国数据"),
                (3, "province only"),
            ],
        )
        conn.commit()

    result = run_grep_context(task, pattern="全国|中国", path_filter="main.sqlite")

    assert result["match_count"] == 1
    hit = result["results"][0]
    assert hit["file"] == "main.sqlite"
    assert hit["table"] == "notes"
    assert hit["column"] == "label"
    assert hit["total_matches"] == 2
    assert hit["samples"] == ["全国合计", "中国数据"]
