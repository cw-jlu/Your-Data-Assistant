"""inspect_files 工具的单元测试。

覆盖点：
- mixed context（csv / json / md / sqlite / xlsx 占位 / 未知扩展）下的统一输出形态
- CSV dtype 推断（int/float/str）
- sqlite tables → columns 抽取
- supported=false 占位（xlsx + 未知扩展）
- file_count / unsupported_formats / truncated 字段语义
"""

from __future__ import annotations

import contextlib
import csv
import json
import sqlite3
from pathlib import Path

import pytest

from agents.benchmark.schema import PublicTask, TaskAssets, TaskRecord
from agents.tools import constants
from agents.tools.inspect_files import inspect_context_files


def _make_task(tmp_path: Path) -> PublicTask:
    task_dir = tmp_path / "task_x"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    return PublicTask(
        record=TaskRecord(task_id="task_x", difficulty="easy", question="?"),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _write_sqlite(path: Path) -> None:
    with contextlib.closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE patients (id INTEGER PRIMARY KEY, name TEXT)")
        conn.execute("CREATE TABLE visits (id INTEGER, patient_id INTEGER, date TEXT)")
        conn.execute("INSERT INTO patients (id, name) VALUES (1, 'a')")
        conn.commit()


def _write_pdf(path: Path, pages: list[list[str]]) -> None:
    import pymupdf  # pyright: ignore[reportMissingImports]

    path.parent.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open()  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
    try:
        for page_lines in pages:
            page = doc.new_page()  # pyright: ignore[reportUnknownMemberType]
            y = 72
            for text in page_lines:
                page.insert_text((72, y), text, fontsize=11)  # pyright: ignore[reportUnknownMemberType]
                y += 36
        doc.save(str(path))  # pyright: ignore[reportUnknownMemberType]
    finally:
        doc.close()  # pyright: ignore[reportUnknownMemberType]


@pytest.fixture
def task_with_mixed_context(tmp_path: Path) -> PublicTask:
    task = _make_task(tmp_path)
    context = task.context_dir
    # CSV with int/float/str columns — dtype 推断三类样本
    _write_csv(
        context / "people.csv",
        header=["age", "score", "name"],
        rows=[
            ["12", "3.14", "alice"],
            ["34", "2.71", "bob"],
            ["56", "1.41", "carol"],
        ],
    )
    # JSON object with keys
    (context / "config.json").write_text(json.dumps({"alpha": 1, "beta": "x"}))
    # markdown doc
    (context / "knowledge.md").write_text("# Topic\nline 2\nline 3\nline 4")
    # sqlite database
    _write_sqlite(context / "main.sqlite")
    # xlsx 占位 — 已识别但本工具未原生支持
    (context / "data.xlsx").write_bytes(b"\x50\x4b\x03\x04")
    # 未知扩展 — 也走 supported=false 分支
    (context / "blob.bin").write_bytes(b"\x00\x01\x02")
    return task


def test_inspect_files_returns_one_entry_per_file(task_with_mixed_context: PublicTask) -> None:
    result = inspect_context_files(task_with_mixed_context)
    assert result["file_count"] == 6
    assert result["truncated"] is False
    paths = {entry["path"] for entry in result["files"]}
    assert paths == {
        "people.csv",
        "config.json",
        "knowledge.md",
        "main.sqlite",
        "data.xlsx",
        "blob.bin",
    }


def test_inspect_files_csv_dtype_inference(task_with_mixed_context: PublicTask) -> None:
    result = inspect_context_files(task_with_mixed_context)
    csv_entry = next(entry for entry in result["files"] if entry["path"] == "people.csv")
    assert csv_entry["format"] == "csv"
    assert csv_entry["supported"] is True
    schema = csv_entry["schema"]
    assert schema["columns"] == ["age", "score", "name"]
    assert schema["dtypes"] == ["int", "float", "str"]
    assert schema["row_count"] == 3
    assert schema["dtypes_confidence"] == "approx"
    # Step 1: column profile assertions
    assert "profile" in schema
    assert schema["profile_sample_rows"] == 3
    profile = schema["profile"]
    assert len(profile) == 3
    # int column: min/max
    assert profile[0]["null_rate"] == 0.0
    assert profile[0]["min"] == 12
    assert profile[0]["max"] == 56
    # float column: min/max
    assert profile[1]["null_rate"] == 0.0
    assert profile[1]["min"] == 1.41
    assert profile[1]["max"] == 3.14
    # str column with <=20 unique: values list
    assert profile[2]["null_rate"] == 0.0
    assert profile[2]["values"] == ["alice", "bob", "carol"]


def test_inspect_files_sqlite_tables(task_with_mixed_context: PublicTask) -> None:
    result = inspect_context_files(task_with_mixed_context)
    sqlite_entry = next(entry for entry in result["files"] if entry["path"] == "main.sqlite")
    assert sqlite_entry["format"] == "sqlite"
    assert sqlite_entry["supported"] is True
    tables = sqlite_entry["schema"]["tables"]
    assert set(tables.keys()) == {"patients", "visits"}
    assert tables["patients"]["columns"] == ["id", "name"]
    assert tables["patients"]["row_count"] == 1
    # patients has rows → profiling present
    assert tables["patients"]["dtypes"] == ["int", "str"]
    assert "profile" in tables["patients"]
    assert tables["patients"]["profile_sample_rows"] == 1
    # visits has 0 rows → no profiling
    assert tables["visits"]["columns"] == ["id", "patient_id", "date"]
    assert tables["visits"]["row_count"] == 0
    assert "dtypes" not in tables["visits"]
    assert "profile" not in tables["visits"]


def test_inspect_files_xlsx_marked_unsupported(task_with_mixed_context: PublicTask) -> None:
    result = inspect_context_files(task_with_mixed_context)
    xlsx_entry = next(entry for entry in result["files"] if entry["path"] == "data.xlsx")
    assert xlsx_entry["supported"] is False
    assert xlsx_entry["format"] == "xlsx"
    assert "execute_python" in xlsx_entry["reason"]
    assert "xlsx" in result["unsupported_formats"]


def test_inspect_files_video_hints_explore_video(tmp_path: Path) -> None:
    """视频文件的 reason 必须明确指向 `explore_video`，绝不能误导走 execute_python。

    旧默认文案 "use execute_python or another preview tool" 让模型试着 pandas/csv 读视频
    必然失败，至少烧 1 步、最坏 2–3 步才换工具。explorer 的工具集里只有 `explore_video`
    能处理视频，hint 必须直接点名。

    覆盖全部 `agents.runtime.media.VIDEO_EXTENSIONS` 也是同步契约——任何新加的扩展名都
    要在 constants 里登记 hint，否则该视频会回落到默认文案上。
    """
    from agents.runtime.media import VIDEO_EXTENSIONS

    task = _make_task(tmp_path)
    for ext in sorted(VIDEO_EXTENSIONS):
        (task.context_dir / f"clip{ext}").write_bytes(b"\x00")

    result = inspect_context_files(task)

    by_path = {entry["path"]: entry for entry in result["files"]}
    for ext in VIDEO_EXTENSIONS:
        entry = by_path[f"clip{ext}"]
        assert entry["supported"] is False, ext
        assert entry["format"] == ext.lstrip("."), ext
        # 关键：不能让模型读到任何引导它走 execute_python 的字眼
        assert "execute_python" not in entry["reason"], ext
        assert "explore_video" in entry["reason"], ext


def test_unsupported_hint_covers_every_video_extension() -> None:
    """`UNSUPPORTED_HINT_BY_EXT` 必须覆盖 `media.VIDEO_EXTENSIONS` 的全集。

    单测两份常量的同步契约——任何一方增减扩展名，本断言会立刻提醒另一方。
    防止下次有人在 `media.py` 加 `.ogv` 但忘了在 `constants.py` 登记 hint。
    """
    from agents.runtime.media import VIDEO_EXTENSIONS

    missing = VIDEO_EXTENSIONS - set(constants.UNSUPPORTED_HINT_BY_EXT)
    assert not missing, f"video extensions without inspect_files hint: {sorted(missing)}"
    for ext in VIDEO_EXTENSIONS:
        assert "explore_video" in constants.UNSUPPORTED_HINT_BY_EXT[ext], ext


def test_inspect_files_unknown_extension_marked_unsupported(
    task_with_mixed_context: PublicTask,
) -> None:
    result = inspect_context_files(task_with_mixed_context)
    bin_entry = next(entry for entry in result["files"] if entry["path"] == "blob.bin")
    assert bin_entry["supported"] is False
    assert bin_entry["format"] == "bin"
    assert isinstance(bin_entry["reason"], str) and bin_entry["reason"]


def test_inspect_files_json_object_summary(task_with_mixed_context: PublicTask) -> None:
    result = inspect_context_files(task_with_mixed_context)
    json_entry = next(entry for entry in result["files"] if entry["path"] == "config.json")
    assert json_entry["format"] == "json"
    schema = json_entry["schema"]
    assert schema["kind"] == "object"
    assert set(schema["keys"]) == {"alpha", "beta"}
    assert schema["key_count"] == 2


def test_inspect_files_markdown_head(task_with_mixed_context: PublicTask) -> None:
    result = inspect_context_files(task_with_mixed_context)
    md_entry = next(entry for entry in result["files"] if entry["path"] == "knowledge.md")
    assert md_entry["format"] == "markdown"
    schema = md_entry["schema"]
    assert schema["line_count"] == 4
    assert schema["head"] == ["# Topic", "line 2", "line 3"]
    # knowledge.md < 32KB → full_text preloaded
    assert schema["knowledge_preloaded"] is True
    assert "# Topic" in schema["full_text"]


def test_inspect_files_pdf_uses_entity_group_samples(tmp_path: Path) -> None:
    task = _make_task(tmp_path)
    _write_pdf(
        task.context_dir / "doc" / "profiles.pdf",
        [
            [
                "Record 101 management scale is",
            ],
            [
                "123.45 billion yuan.",
                "Record 102 management scale is 88.00 billion yuan.",
                "Record 103 management scale is 9.50 billion yuan.",
            ],
        ],
    )

    result = inspect_context_files(task)

    entry = next(e for e in result["files"] if e["path"] == "doc/profiles.pdf")
    assert entry["format"] == "pdf"
    assert entry["supported"] is True
    schema = entry["schema"]
    assert "head" not in schema
    assert "page_char_counts" not in schema
    assert schema["page_count"] == 2
    assert schema["paragraph_count"] == 3
    assert schema["entity_grouping"] == "record_id_contiguous_paragraphs"
    assert schema["entity_group_count"] == 3
    assert schema["entity_sample_size"] == 3
    groups = {group["entity_id"]: group for group in schema["entity_groups_sample"]}
    assert set(groups) == {"101", "102", "103"}
    record_101_text = "\n\n".join(paragraph["text"] for paragraph in groups["101"]["paragraphs"])
    assert "Record 101 management scale is 123.45 billion yuan." in record_101_text


def test_inspect_files_non_knowledge_md_no_fulltext(tmp_path: Path) -> None:
    """Non-knowledge .md files do not get full_text preloaded."""
    task = _make_task(tmp_path)
    (task.context_dir / "notes.md").write_text("some notes")
    result = inspect_context_files(task)
    entry = next(e for e in result["files"] if e["path"] == "notes.md")
    assert "full_text" not in entry["schema"]
    assert "knowledge_preloaded" not in entry["schema"]


def test_inspect_files_oversized_knowledge_md_no_fulltext(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """knowledge.md exceeding size limit does not get preloaded."""
    monkeypatch.setattr(constants, "KNOWLEDGE_MD_PRELOAD_LIMIT", 10)
    task = _make_task(tmp_path)
    (task.context_dir / "knowledge.md").write_text("x" * 100)
    result = inspect_context_files(task)
    entry = next(e for e in result["files"] if e["path"] == "knowledge.md")
    assert "full_text" not in entry["schema"]
    assert "knowledge_preloaded" not in entry["schema"]


def test_inspect_files_truncates_when_above_limit(tmp_path: Path) -> None:
    task = _make_task(tmp_path)
    # 创建 105 个空 .txt 占位文件，触发上限截断
    for index in range(105):
        (task.context_dir / f"file_{index:03d}.txt").write_text("")
    result = inspect_context_files(task)
    assert result["truncated"] is True
    assert result["file_count"] == 100


def test_inspect_files_empty_context(tmp_path: Path) -> None:
    task = _make_task(tmp_path)
    result = inspect_context_files(task)
    assert result["file_count"] == 0
    assert result["files"] == []
    assert result["unsupported_formats"] == []
    assert result["truncated"] is False


def test_inspect_files_sqlite_profile_values(tmp_path: Path) -> None:
    """SQLite profiling: multi-row table gets min/max for int, values for categorical."""
    task = _make_task(tmp_path)
    db_path = task.context_dir / "test.db"
    with contextlib.closing(sqlite3.connect(db_path)) as conn:
        conn.execute("CREATE TABLE items (id INTEGER, category TEXT, price REAL)")
        conn.executemany(
            "INSERT INTO items VALUES (?, ?, ?)",
            [(1, "A", 10.5), (2, "B", 20.0), (3, "A", 15.5)],
        )
        conn.commit()
    result = inspect_context_files(task)
    entry = next(e for e in result["files"] if e["path"] == "test.db")
    tbl = entry["schema"]["tables"]["items"]
    assert tbl["dtypes"] == ["int", "str", "float"]
    assert tbl["profile_sample_rows"] == 3
    assert tbl["profile"][0]["min"] == 1
    assert tbl["profile"][0]["max"] == 3
    assert tbl["profile"][1]["values"] == ["A", "B"]
    assert tbl["profile"][2]["min"] == 10.5
    assert tbl["profile"][2]["max"] == 20.0


def test_inspect_files_sqlite_zero_rows_no_profile(tmp_path: Path) -> None:
    """Empty table: no dtypes/profile keys."""
    task = _make_task(tmp_path)
    db_path = task.context_dir / "empty.sqlite"
    with contextlib.closing(sqlite3.connect(db_path)) as conn:
        conn.execute("CREATE TABLE t (a INTEGER, b TEXT)")
        conn.commit()
    result = inspect_context_files(task)
    entry = next(e for e in result["files"] if e["path"] == "empty.sqlite")
    tbl = entry["schema"]["tables"]["t"]
    assert tbl["row_count"] == 0
    assert "dtypes" not in tbl
    assert "profile" not in tbl


def test_inspect_files_oversized_json_via_streaming(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """超过 100MB 上限的 JSON：inspect_files 走 ijson 流式分支，不再返回 kind="unknown"。

    回归 run 20260430-025 task_2：旧契约对 7.4MB zip_code.json 返回零结构信息，
    模型在 execute_python 里瞎猜 `for k in data.keys()[:3]: print(data[k])` 印爆 stdout。
    新契约下 schema.kind == "object"，keys 字段给出真实顶层 keys。
    """
    monkeypatch.setattr(constants, "INSPECT_FILES_JSON_SIZE_LIMIT", 50)
    task = _make_task(tmp_path)
    (task.context_dir / "zip_code.json").write_text(
        json.dumps({"table": "zip_code", "records": [{"id": 1}, {"id": 2}]})
    )

    result = inspect_context_files(task)
    entry = next(e for e in result["files"] if e["path"] == "zip_code.json")

    assert entry["format"] == "json"
    assert entry["supported"] is True
    schema = entry["schema"]
    assert schema["kind"] == "object"
    assert schema["streamed"] is True
    assert schema["keys"] == ["table", "records"]
    assert schema["key_count"] == 2
    # 旧 truncated_at 字段已废弃
    assert "truncated_at" not in schema
