"""Tests for detect_prose_files."""

from __future__ import annotations

from pathlib import Path

from agents.benchmark.schema import PublicTask, TaskAssets, TaskRecord
from agents.etl._detect import detect_prose_files


def _make_task(tmp_path: Path) -> PublicTask:
    task_dir = tmp_path / "task_etl"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    return PublicTask(
        record=TaskRecord(task_id="task_etl", difficulty="easy", question="?"),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )


def _write_lines(path: Path, n: int) -> None:
    path.write_text("\n".join(f"line {i}" for i in range(n)))


def test_txt_in_doc_dir_detected(tmp_path: Path) -> None:
    task = _make_task(tmp_path)
    doc_dir = task.context_dir / "doc"
    doc_dir.mkdir()
    _write_lines(doc_dir / "data.txt", 5)
    result = detect_prose_files(task)
    assert any(p.name == "data.txt" for p in result)


def test_non_utf8_txt_excluded(tmp_path: Path) -> None:
    task = _make_task(tmp_path)
    bad = task.context_dir / "bad.txt"
    bad.write_bytes(b"line\nline\n\xff")

    result = detect_prose_files(task)

    assert bad not in result


def test_md_in_context_root_detected(tmp_path: Path) -> None:
    task = _make_task(tmp_path)
    _write_lines(task.context_dir / "heroes.md", 3)
    result = detect_prose_files(task)
    assert any(p.name == "heroes.md" for p in result)


def test_utf8_chinese_md_detected(tmp_path: Path) -> None:
    task = _make_task(tmp_path)
    zh = task.context_dir / "heroes_zh.md"
    zh.write_text("姓名：阿青\n门派：越女剑\n", encoding="utf-8")
    result = detect_prose_files(task)
    assert zh in result


def test_knowledge_md_excluded_everywhere(tmp_path: Path) -> None:
    task = _make_task(tmp_path)
    doc_dir = task.context_dir / "doc"
    doc_dir.mkdir()
    _write_lines(task.context_dir / "knowledge.md", 10)
    _write_lines(doc_dir / "knowledge.md", 10)
    result = detect_prose_files(task)
    assert not any(p.name.lower() == "knowledge.md" for p in result)


def test_short_file_included(tmp_path: Path) -> None:
    task = _make_task(tmp_path)
    _write_lines(task.context_dir / "short.md", 3)
    result = detect_prose_files(task)
    assert any(p.name == "short.md" for p in result)


def test_empty_file_excluded(tmp_path: Path) -> None:
    task = _make_task(tmp_path)
    empty = task.context_dir / "empty.md"
    _write_lines(empty, 0)
    result = detect_prose_files(task)
    assert empty not in result


def test_no_duplicates_when_file_in_both(tmp_path: Path) -> None:
    """doc/ takes priority; same file should not appear twice."""
    task = _make_task(tmp_path)
    doc_dir = task.context_dir / "doc"
    doc_dir.mkdir()
    _write_lines(doc_dir / "data.md", 5)
    result = detect_prose_files(task)
    names = [p.name for p in result]
    assert names.count("data.md") == 1
