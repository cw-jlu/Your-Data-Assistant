"""PDF ETL edge-case tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agents.benchmark.schema import PublicTask, TaskAssets, TaskRecord
from agents.etl import extractor
from agents.etl._pdf import (
    _page_to_paragraphs,
    _stitch_page_paragraphs,
    _unwrap_soft_line_breaks,
)
from agents.llm import ModelMessage, ModelResponse


class ExplodingAdapter:
    def complete(
        self, messages: list[ModelMessage], *, tools: object | None = None, **kwargs: Any
    ) -> ModelResponse:
        raise AssertionError("empty extracted text should not call the LLM")


class StaticAdapter:
    def complete(
        self, messages: list[ModelMessage], *, tools: object | None = None, **kwargs: Any
    ) -> ModelResponse:
        return ModelResponse(content="ID: 1 | value: fresh")


class _FakePage:
    def __init__(self, lines: list[tuple[float, float, float, float, str]]) -> None:
        self.lines = lines

    def get_text(self, mode: str | None = None) -> object:
        if mode == "dict":
            return {
                "blocks": [
                    {
                        "lines": [
                            {
                                "bbox": (x0, y0, x1, y1),
                                "spans": [{"text": text}],
                            }
                            for x0, y0, x1, y1, text in self.lines
                        ]
                    }
                ]
            }
        return "\n".join(line[4] for line in self.lines)


def _make_task(tmp_path: Path) -> PublicTask:
    task_dir = tmp_path / "task_pdf_etl"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    return PublicTask(
        record=TaskRecord(task_id="task_pdf_etl", difficulty="easy", question="?"),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )


def test_extract_prose_file_returns_none_for_textless_pdf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pymupdf

    task = _make_task(tmp_path)
    monkeypatch.setattr(extractor, "ETL_SCRATCH_ROOT", tmp_path / "etl")

    pdf_path = task.context_dir / "blank.pdf"
    doc = pymupdf.open()
    doc.new_page()
    doc.save(str(pdf_path))
    doc.close()

    result = extractor.extract_prose_file(ExplodingAdapter(), pdf_path, task, "")

    assert result is None


def test_extract_prose_file_reuses_existing_csv_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = _make_task(tmp_path)
    monkeypatch.setattr(extractor, "ETL_SCRATCH_ROOT", tmp_path / "etl")

    prose_path = task.context_dir / "patient.md"
    prose_path.write_text("ID 1 has value cached.", encoding="utf-8")

    etl_dir = tmp_path / "etl" / task.task_id / "_etl"
    etl_dir.mkdir(parents=True)
    csv_path = etl_dir / "patient.csv"
    csv_path.write_text("id,value\n1,cached\n", encoding="utf-8")

    cached = extractor.extract_prose_file(
        ExplodingAdapter(),
        prose_path,
        task,
        prose_path.read_text(encoding="utf-8"),
        schema_columns=["id", "value"],
        anchor_keys=["id"],
    )

    assert cached is not None
    assert cached.row_count == 1
    assert csv_path.read_text(encoding="utf-8").strip() == "id,value\n1,cached"


def test_extract_prose_file_extracts_when_no_csv_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = _make_task(tmp_path)
    monkeypatch.setattr(extractor, "ETL_SCRATCH_ROOT", tmp_path / "etl")

    prose_path = task.context_dir / "patient.md"
    prose_path.write_text("ID 1 has value fresh.", encoding="utf-8")

    result = extractor.extract_prose_file(
        StaticAdapter(),
        prose_path,
        task,
        prose_path.read_text(encoding="utf-8"),
        schema_columns=["id", "value"],
        anchor_keys=["id"],
    )

    assert result is not None
    csv_path = tmp_path / "etl" / task.task_id / "_etl" / "patient.csv"
    assert csv_path.is_file()


def test_unwrap_soft_line_breaks_joins_prose_inside_entity_paragraph() -> None:
    text = "\n".join(
        [
            "关于档案条目 266 的审查，初步记录显示其内部代码为 4170，但经过数据校准后，",
            "最终确认为 4177。该金融工具在二级市场的交易代码被指定为 162207。值得注意",
            "的是，该基金的年度报告提交截止日期已根据监管要求进行了调整。",
        ]
    )

    unwrapped = _unwrap_soft_line_breaks(text)

    assert "数据校准后，最终确认为 4177" in unwrapped
    assert "值得注意的是，该基金" in unwrapped
    assert "\n" not in unwrapped


def test_unwrap_soft_line_breaks_joins_english_with_spaces_and_hyphenation() -> None:
    text = "\n".join(
        [
            "The portfolio manager reviewed the invest-",
            "ment strategy and confirmed",
            "the final benchmark exposure.",
        ]
    )

    unwrapped = _unwrap_soft_line_breaks(text)

    assert unwrapped == (
        "The portfolio manager reviewed the investment strategy and confirmed "
        "the final benchmark exposure."
    )


def test_unwrap_soft_line_breaks_does_not_infer_paragraphs_from_entity_words() -> None:
    text = "\n".join(
        [
            "档案 21 的记录显示其个人代码为 101013239。",
            "该经理仍在任。",
            "关于档案条目 266 的审查，初步记录显示其内部代码为 4170。",
            "最终确认为 4177。",
        ]
    )

    unwrapped = _unwrap_soft_line_breaks(text)

    assert (
        unwrapped == "档案 21 的记录显示其个人代码为 101013239。该经理仍在任。"
        "关于档案条目 266 的审查，初步记录显示其内部代码为 4170。最终确认为 4177。"
    )


def test_unwrap_soft_line_breaks_preserves_markdown_structures() -> None:
    text = "\n".join(
        [
            "| field | value |",
            "| --- | --- |",
            "| code | 101 |",
            "",
            "- item one",
            "- item two",
            "",
            "普通段落第一行，",
            "普通段落第二行。",
        ]
    )

    unwrapped = _unwrap_soft_line_breaks(text)

    assert "| field | value |\n| --- | --- |\n| code | 101 |" in unwrapped
    assert "- item one\n- item two" in unwrapped
    assert "普通段落第一行，普通段落第二行。" in unwrapped


def test_page_to_paragraphs_uses_layout_gaps_as_paragraphs() -> None:
    page = _FakePage(
        [
            (0, 50, 100, 60, "第二段第一行，"),
            (0, 64, 100, 74, "第二段第二行。"),
            (0, 10, 100, 20, "第一段第一行，"),
            (0, 24, 100, 34, "第一段第二行。"),
        ]
    )

    paragraphs = _page_to_paragraphs(page)

    assert paragraphs == ["第一段第一行，第一段第二行。", "第二段第一行，第二段第二行。"]


def test_stitch_page_paragraphs_joins_incomplete_sentence_across_pages() -> None:
    stitched = _stitch_page_paragraphs(
        [
            (1, ["上一页最后一段停在新的数"]),
            (2, ["据可视化软件。该段继续。", "下一段。"]),
        ]
    )

    assert stitched == [
        (1, ["上一页最后一段停在新的数据可视化软件。该段继续。"]),
        (2, ["下一段。"]),
    ]


def test_stitch_page_paragraphs_joins_english_across_pages() -> None:
    stitched = _stitch_page_paragraphs(
        [
            (1, ["The manager reviewed the invest-"]),
            (2, ["ment policy in detail.", "The next paragraph starts here."]),
        ]
    )

    assert stitched == [
        (1, ["The manager reviewed the investment policy in detail."]),
        (2, ["The next paragraph starts here."]),
    ]


def test_stitch_page_paragraphs_keeps_complete_sentences_separate() -> None:
    stitched = _stitch_page_paragraphs(
        [
            (1, ["上一页最后一段已经结束。"]),
            (2, ["下一页第一段。"]),
        ]
    )

    assert stitched == [(1, ["上一页最后一段已经结束。"]), (2, ["下一页第一段。"])]
