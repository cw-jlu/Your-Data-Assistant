from __future__ import annotations

from pathlib import Path

import pytest

from agents.benchmark.schema import PublicTask, TaskAssets, TaskRecord
from agents.tools.read_doc import read_doc_preview


@pytest.fixture
def task(tmp_path: Path) -> PublicTask:
    task_dir = tmp_path / "task_1"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    return PublicTask(
        record=TaskRecord(
            task_id="task_1",
            difficulty="easy",
            question="Read the document.",
        ),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )


def write_doc(task: PublicTask, name: str, lines: list[str]) -> None:
    (task.context_dir / name).write_text("\n".join(lines))


def numbered_lines(count: int) -> list[str]:
    return [f"line-{line_number}" for line_number in range(1, count + 1)]


def test_read_doc_returns_requested_line_page(task: PublicTask) -> None:
    write_doc(task, "knowledge.md", numbered_lines(100))

    result = read_doc_preview(task, "knowledge.md", start_line=21, line_count=10)

    assert result == {
        "total_lines": 100,
        "returned_range": {"start_line": 21, "end_line": 30},
        "has_more": True,
        "keyword_match": None,
        "preview": "\n".join(f"line-{line_number}" for line_number in range(21, 31)),
    }


def test_read_doc_returns_last_page_without_has_more(task: PublicTask) -> None:
    write_doc(task, "knowledge.md", numbered_lines(100))

    result = read_doc_preview(task, "knowledge.md", start_line=95, line_count=20)

    assert result["returned_range"] == {"start_line": 95, "end_line": 100}
    assert result["has_more"] is False
    assert result["preview"] == "\n".join(f"line-{line_number}" for line_number in range(95, 101))


def test_read_doc_normalizes_line_arguments(task: PublicTask) -> None:
    write_doc(task, "knowledge.md", numbered_lines(400))

    result = read_doc_preview(task, "knowledge.md", start_line=-10, line_count=500)

    assert result["returned_range"] == {"start_line": 1, "end_line": 300}
    assert result["has_more"] is True
    assert result["preview"].splitlines()[0] == "line-1"
    assert result["preview"].splitlines()[-1] == "line-300"


def test_read_doc_keyword_returns_matching_paragraph(task: PublicTask) -> None:
    """keyword 模式：返回包含关键词的整个段落（空行分隔），不再是固定行窗口。"""
    lines = numbered_lines(100)
    lines[49] = "line-50 special Thrombosis finding"
    write_doc(task, "knowledge.md", lines)

    result = read_doc_preview(
        task,
        "knowledge.md",
        keyword="thrombosis",
    )

    # paging 字段在 keyword 模式下被清空——避免模型把它们当成"已读到这里"
    assert result["returned_range"] is None
    assert result["has_more"] is False
    assert result["preview"] is None
    # 只有一个段落（整文件都是非空行连续），里面包含命中行
    km = result["keyword_match"]
    assert km["keyword"] == "thrombosis"
    assert km["found"] is True
    assert km["match_count"] == 1
    assert km["matches_truncated"] is False
    assert len(km["matches"]) == 1
    assert "line-50 special Thrombosis finding" in km["matches"][0]["text"]


def test_read_doc_keyword_returns_all_matching_paragraphs(task: PublicTask) -> None:
    """多匹配段落：每个匹配段落独立返回，按行号顺序。

    本 case 模拟 superhero.md 形态——blank-line 分隔的"一段一记录"prose 数据库。
    旧实现只返回首个匹配前后窗口；新实现一次性把 3 段都返回，省去翻页。
    """
    paragraphs_text = (
        "Riddler entry: ID 577 male\n"
        "\n"
        "Hawkman entry: ID 334\n"
        "\n"
        "Riddler sequel: ID 999\n"
        "\n"
        "Joker entry: ID 451\n"
        "\n"
        "Riddler crossover: ID 123"
    )
    (task.context_dir / "doc.md").write_text(paragraphs_text)

    result = read_doc_preview(task, "doc.md", keyword="riddler")

    km = result["keyword_match"]
    assert km["found"] is True
    assert km["match_count"] == 3
    assert km["matches_truncated"] is False
    matches = km["matches"]
    assert [m["start_line"] for m in matches] == [1, 5, 9]
    assert "ID 577" in matches[0]["text"]
    assert "ID 999" in matches[1]["text"]
    assert "ID 123" in matches[2]["text"]
    # paragraph_truncated 默认 False（这些段都在 2KB 以内）
    assert all(m["paragraph_truncated"] is False for m in matches)


def test_read_doc_keyword_ignores_start_line(task: PublicTask) -> None:
    """keyword 模式下 start_line 被忽略——全文搜索，不再是"从 start_line 起的首匹配"。

    替换旧 `test_read_doc_keyword_search_starts_at_start_line`：
    旧契约强制模型同时管理分页 + 关键词的相对位置，新契约把它简化为"一次拿到全部"。
    """
    paragraphs_text = (
        "First chunk Thrombosis early\n\nSecond chunk unrelated\n\nThird chunk Thrombosis later"
    )
    (task.context_dir / "doc.md").write_text(paragraphs_text)

    result = read_doc_preview(
        task,
        "doc.md",
        start_line=3,  # 故意从中间开始：旧实现会忽略第 1 段，新实现两段都返回
        line_count=12,
        keyword="thrombosis",
    )

    km = result["keyword_match"]
    assert km["match_count"] == 2
    matches = km["matches"]
    assert "Thrombosis early" in matches[0]["text"]
    assert "Thrombosis later" in matches[1]["text"]


def test_read_doc_keyword_miss_returns_empty_matches(task: PublicTask) -> None:
    """keyword 未命中：matches=[], match_count=0，不再回退到分页。"""
    write_doc(task, "knowledge.md", numbered_lines(100))

    result = read_doc_preview(
        task,
        "knowledge.md",
        start_line=21,
        line_count=10,
        keyword="Thrombosis",
    )

    assert result["preview"] is None
    assert result["returned_range"] is None
    assert result["keyword_match"] == {
        "keyword": "Thrombosis",
        "found": False,
        "match_count": 0,
        "matches": [],
        "matches_truncated": False,
    }


def test_read_doc_keyword_match_count_cap(task: PublicTask) -> None:
    """匹配数超 20 段：返回前 20 段，match_count 反映总数，matches_truncated=True。"""
    paragraphs = [f"hit {i} target" for i in range(30)]
    (task.context_dir / "doc.md").write_text("\n\n".join(paragraphs))

    result = read_doc_preview(task, "doc.md", keyword="target")

    km = result["keyword_match"]
    assert km["match_count"] == 30
    assert len(km["matches"]) == 20
    assert km["matches_truncated"] is True


def test_read_doc_keyword_byte_cap_stops_early(task: PublicTask) -> None:
    """累计 > 8KB：不再追加新段，matches_truncated=True；已收集的段保留。"""
    # 每段 ~600B，~14 段就会撞 8KB 上限
    paragraphs = ["target " + ("y" * 600) for _ in range(20)]
    (task.context_dir / "doc.md").write_text("\n\n".join(paragraphs))

    result = read_doc_preview(task, "doc.md", keyword="target")

    km = result["keyword_match"]
    assert km["match_count"] == 20
    assert len(km["matches"]) < 20
    assert km["matches_truncated"] is True


def test_read_doc_keyword_paragraph_byte_truncation(task: PublicTask) -> None:
    """单段超 2KB：截断到 2KB 并追加 "\\n..." 标记，挂 paragraph_truncated=True。"""
    big_paragraph = "target " + ("z" * 5000)
    (task.context_dir / "doc.md").write_text(big_paragraph)

    result = read_doc_preview(task, "doc.md", keyword="target")

    km = result["keyword_match"]
    assert len(km["matches"]) == 1
    match = km["matches"][0]
    assert match["paragraph_truncated"] is True
    assert match["text"].endswith("\n...")
    # 截断后文本主体长度 == cap；尾部 "\n..." 4 字节
    assert len(match["text"]) == 2 * 1024 + len("\n...")


def test_read_doc_empty_file_returns_stable_empty_result(task: PublicTask) -> None:
    write_doc(task, "empty.md", [])

    result = read_doc_preview(task, "empty.md")

    assert result == {
        "total_lines": 0,
        "returned_range": None,
        "has_more": False,
        "keyword_match": None,
        "preview": "",
    }


def test_read_doc_rejects_path_escape(task: PublicTask) -> None:
    with pytest.raises(ValueError, match="Path escapes context dir"):
        read_doc_preview(task, "../secret.md")
