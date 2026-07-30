from __future__ import annotations

from pathlib import Path

import pytest

from agents.benchmark.schema import PublicTask, TaskAssets, TaskRecord
from agents.tools import context as context_mod


def _make_task(tmp_path: Path) -> PublicTask:
    task_dir = tmp_path / "input" / "task_ctx"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    return PublicTask(
        record=TaskRecord(task_id="task_ctx", difficulty="hard", question="?"),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )


def test_prose_without_etl_csv_stays_visible(tmp_path: Path, monkeypatch) -> None:
    """ETL 失败/跳过的 prose 文件必须回退暴露给 agent（fail-open）。"""
    task = _make_task(tmp_path)
    doc_dir = task.context_dir / "doc"
    doc_dir.mkdir()
    (doc_dir / "covered.md").write_text("etl ok", encoding="utf-8")
    (doc_dir / "orphan.md").write_text("etl failed for this one", encoding="utf-8")

    scratch_root = tmp_path / "scratch"
    monkeypatch.setattr(context_mod, "ETL_SCRATCH_ROOT", scratch_root)
    etl_dir = scratch_root / task.task_id / "_etl"
    etl_dir.mkdir(parents=True)
    (etl_dir / "covered.csv").write_text("id,value\n1,a\n", encoding="utf-8")

    virtual_context = context_mod.build_virtual_context(task)

    assert not (virtual_context / "doc" / "covered.md").exists()
    assert (virtual_context / "doc" / "orphan.md").exists()
    assert (virtual_context / "csv" / "covered.csv").exists()


def test_prose_with_invalid_etl_csv_stays_visible(tmp_path: Path, monkeypatch) -> None:
    """ETL 产出了 CSV 但内容无效（空/仅表头）时，源 prose 不得被隐藏。"""
    task = _make_task(tmp_path)
    doc_dir = task.context_dir / "doc"
    doc_dir.mkdir()
    (doc_dir / "broken.md").write_text("prose", encoding="utf-8")
    (doc_dir / "broken.pdf").write_bytes(b"%PDF-1.4\n")

    scratch_root = tmp_path / "scratch"
    monkeypatch.setattr(context_mod, "ETL_SCRATCH_ROOT", scratch_root)
    etl_dir = scratch_root / task.task_id / "_etl"
    etl_dir.mkdir(parents=True)
    (etl_dir / "broken.csv").write_text("id,value\n", encoding="utf-8")

    virtual_context = context_mod.build_virtual_context(task)

    assert (virtual_context / "doc" / "broken.md").exists()
    assert (virtual_context / "doc" / "broken.pdf").exists()
    assert not (virtual_context / "csv" / "broken.csv").exists()
    assert not (virtual_context / "csv").exists()


@pytest.mark.parametrize("etl_dir_exists", [False, True])
def test_prose_visible_when_no_etl_output(
    tmp_path: Path, monkeypatch, etl_dir_exists: bool
) -> None:
    task = _make_task(tmp_path)
    (task.context_dir / "notes.md").write_text("prose", encoding="utf-8")

    scratch_root = tmp_path / "scratch"
    monkeypatch.setattr(context_mod, "ETL_SCRATCH_ROOT", scratch_root)
    if etl_dir_exists:
        (scratch_root / task.task_id / "_etl").mkdir(parents=True)

    virtual_context = context_mod.build_virtual_context(task)

    assert (virtual_context / "notes.md").exists()
