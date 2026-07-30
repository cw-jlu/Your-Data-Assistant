from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import fitz
import numpy as np
from langchain_core.messages import AIMessage

from data_agent_baseline.benchmark.schema import PublicTask, TaskAssets, TaskRecord
from data_agent_baseline.run.context_preprocessor import (
    _coalesce_markdown_lines,
    pdf_to_markdown,
    prepare_task_context,
)
from data_agent_baseline.run.video_preprocessor import (
    GeneratedVideoAsset,
    VideoPreprocessResult,
    extract_stable_frames,
    repair_transcript_mojibake,
    simplify_chinese_transcript,
    transcribe_video_audio,
)
from data_agent_baseline.run.video_understanding_agent import (
    add_video_understanding_summaries,
)
from data_agent_baseline.tools.filesystem import list_context_tree, read_doc_preview
from data_agent_baseline.tools.python_exec import TaskContextWorkspace


def _write_pdf(path: Path, lines: list[tuple[float, str]], *, toc: list[list[object]] | None = None) -> None:
    document = fitz.open()
    page = document.new_page()
    for y, text in lines:
        page.insert_text((72, y), text, fontsize=12)
    if toc:
        document.set_toc(toc)
    document.save(path)
    document.close()


def _write_multipage_pdf(path: Path, pages: list[list[tuple[float, str]]]) -> None:
    document = fitz.open()
    for lines in pages:
        page = document.new_page()
        for y, text in lines:
            page.insert_text((72, y), text, fontsize=12)
    document.save(path)
    document.close()


def _task_with_context(task_dir: Path) -> PublicTask:
    context_dir = task_dir / "context"
    return PublicTask(
        record=TaskRecord(task_id=task_dir.name, difficulty="easy", question="question"),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )


def _write_synthetic_video(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        5.0,
        (160, 90),
    )
    assert writer.isOpened()
    scenes = [
        ((20, 20, 180), "A"),
        ((20, 160, 20), "B"),
        ((180, 20, 20), "C"),
    ]
    try:
        for color, label in scenes:
            frame = np.full((90, 160, 3), color, dtype=np.uint8)
            cv2.putText(frame, label, (55, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
            for _ in range(10):
                writer.write(frame)
    finally:
        writer.release()


def test_pdf_to_markdown_uses_pdf_toc_as_headings(tmp_path: Path) -> None:
    pdf_path = tmp_path / "with_toc.pdf"
    _write_pdf(
        pdf_path,
        [
            (72, "Main Report"),
            (96, "The first paragraph continues"),
            (120, "without punctuation"),
            (168, "Detailed Section"),
            (192, "Section content."),
        ],
        toc=[[1, "Main Report", 1], [2, "Detailed Section", 1]],
    )

    markdown = pdf_to_markdown(pdf_path)

    assert "# Main Report" in markdown
    assert "## Detailed Section" in markdown
    assert "The first paragraph continues without punctuation" in markdown


def test_pdf_to_markdown_uses_visual_spacing_as_paragraph_boundaries(tmp_path: Path) -> None:
    pdf_path = tmp_path / "visual_paragraphs.pdf"
    _write_pdf(
        pdf_path,
        [
            (72, "First paragraph continues"),
            (96, "on the same visual block."),
            (156, "Second visual paragraph."),
        ],
    )

    markdown = pdf_to_markdown(pdf_path)

    assert markdown.splitlines() == [
        "First paragraph continues on the same visual block.",
        "Second visual paragraph.",
    ]


def test_pdf_to_markdown_does_not_infer_headings_without_toc(tmp_path: Path) -> None:
    pdf_path = tmp_path / "without_toc.pdf"
    _write_pdf(
        pdf_path,
        [
            (72, "Capital Market Audit Summary"),
            (96, "Body text."),
        ],
    )

    markdown = pdf_to_markdown(pdf_path)

    assert "Capital Market Audit Summary" in markdown
    assert "#" not in markdown


def test_pdf_to_markdown_repairs_cjk_hard_line_wraps(tmp_path: Path) -> None:
    del tmp_path

    markdown = _coalesce_markdown_lines(
        [
            "其适用对象明确为城市信",
            "用社。此举旨在调节该类机构。",
        ]
    )

    assert "城市信用社" in markdown


def test_pdf_to_markdown_repairs_cjk_wrap_before_decimal_number(tmp_path: Path) -> None:
    pdf_path = tmp_path / "decimal_wrap.pdf"
    _write_pdf(
        pdf_path,
        [
            (72, "The manager keeps"),
            (96, "11.0 equity funds in the portfolio."),
        ],
    )

    markdown = pdf_to_markdown(pdf_path)

    assert "keeps 11.0 equity funds" in markdown


def test_pdf_to_markdown_merges_unfinished_paragraph_across_pages(tmp_path: Path) -> None:
    pdf_path = tmp_path / "cross_page_unfinished.pdf"
    _write_multipage_pdf(
        pdf_path,
        [
            [(760, "The department is using a new")],
            [(72, "analytics tool. Approval is complete.")],
        ]
    )

    markdown = pdf_to_markdown(pdf_path)

    assert markdown.splitlines() == [
        "The department is using a new analytics tool. Approval is complete."
    ]


def test_pdf_to_markdown_does_not_merge_finished_paragraph_across_pages(tmp_path: Path) -> None:
    pdf_path = tmp_path / "cross_page_finished.pdf"
    _write_multipage_pdf(
        pdf_path,
        [
            [(760, "Approval is complete.")],
            [(72, "The next page starts a new record.")],
        ],
    )

    markdown = pdf_to_markdown(pdf_path)

    assert markdown.splitlines() == ["Approval is complete.", "The next page starts a new record."]


def test_pdf_to_markdown_does_not_merge_short_page_end_across_pages(tmp_path: Path) -> None:
    pdf_path = tmp_path / "cross_page_short_end.pdf"
    _write_multipage_pdf(
        pdf_path,
        [
            [
                (720, "This longer line establishes the right edge of the text area for the page"),
                (760, "Short title"),
            ],
            [(72, "The next page starts a new record.")],
        ],
    )

    markdown = pdf_to_markdown(pdf_path)

    assert markdown.splitlines() == [
        "This longer line establishes the right edge of the text area for the page",
        "Short title",
        "The next page starts a new record.",
    ]


def test_prepare_task_context_converts_pdfs_and_preserves_md_collisions(
    tmp_path: Path,
) -> None:
    task_dir = tmp_path / "task_1"
    context_dir = task_dir / "context"
    (context_dir / "doc").mkdir(parents=True)
    (context_dir / "doc" / "report.md").write_text("# Existing\n", encoding="utf-8")
    _write_pdf(context_dir / "doc" / "report.pdf", [(72, "PDF body.")])
    task = _task_with_context(task_dir)

    preprocessed = prepare_task_context(task, tmp_path / "output" / "task_1")

    manifest = json.loads(
        (tmp_path / "output" / "task_1" / "context_preprocessing_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    manifest_by_visible_path = {entry["visible_path"]: entry for entry in manifest["entries"]}
    assert manifest_by_visible_path["doc/report.md"]["action"] == "source"
    assert manifest_by_visible_path["doc/report_pdf.md"]["action"] == "pdf_to_markdown"
    assert manifest_by_visible_path["doc/report_pdf.md"]["generated"] is True
    assert not (tmp_path / "output" / "task_1" / "context").exists()
    assert (preprocessed.generated_context_dir / "doc" / "report_pdf.md").exists()
    assert not (preprocessed.generated_context_dir / "doc" / "report.md").exists()
    assert preprocessed.task.context_dir == context_dir
    listed_paths = {
        entry["path"]
        for entry in list_context_tree(preprocessed.task)["entries"]
        if entry["kind"] == "file"
    }
    assert listed_paths == {"doc/report.md", "doc/report_pdf.md"}
    assert read_doc_preview(preprocessed.task, "doc/report.md")["preview"] == "# Existing\n"


def test_extract_stable_frames_detects_synthetic_stable_screens(tmp_path: Path) -> None:
    video_path = tmp_path / "synthetic.avi"
    _write_synthetic_video(video_path)

    manifest = extract_stable_frames(
        video_path,
        tmp_path / "frames",
        sample_fps=2.5,
        min_stable_duration=0.5,
    )

    assert manifest["duration_sec"] == 6.0
    assert manifest["stable_segment_count"] == 3
    assert manifest["saved_image_count"] == 3
    assert len(list((tmp_path / "frames").glob("stable_*.jpg"))) == 3


def test_repair_transcript_mojibake_keeps_normal_text_and_repairs_gbk_mojibake() -> None:
    assert repair_transcript_mojibake("正常中文 and English") == "正常中文 and English"

    repaired = repair_transcript_mojibake("鎴戝�戝厛閬庣��涓�闋� 瑷烘柗杩借工")

    assert "我" in repaired
    assert "診斷追蹤" in repaired


def test_simplify_chinese_transcript_only_changes_detected_chinese() -> None:
    assert simplify_chinese_transcript("欄位與設定", language="zh") == "栏位与设定"
    assert simplify_chinese_transcript("欄位與設定", language="zh-TW") == "栏位与设定"
    assert simplify_chinese_transcript("欄位與設定", language="en") == "欄位與設定"


def test_transcribe_video_audio_writes_simplified_chinese_segments(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class FakeWhisperModel:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def transcribe(self, _video_path: str):
            segments = [
                SimpleNamespace(start=0.0, end=1.0, text=" 欄位與設定 "),
                SimpleNamespace(start=1.0, end=2.0, text=" 導出結果 "),
            ]
            info = SimpleNamespace(
                language="zh",
                language_probability=0.99,
                duration=2.0,
            )
            return iter(segments), info

    monkeypatch.setitem(
        sys.modules,
        "faster_whisper",
        SimpleNamespace(WhisperModel=FakeWhisperModel),
    )

    transcript = transcribe_video_audio(tmp_path / "briefing.mp4")

    assert transcript["language"] == "zh"
    assert transcript["segments"] == [
        {"start_sec": 0.0, "end_sec": 1.0, "text": "栏位与设定"},
        {"start_sec": 1.0, "end_sec": 2.0, "text": "导出结果"},
    ]
    assert transcript["text"] == "栏位与设定 导出结果"


def test_prepare_task_context_replaces_video_with_timeline_and_stable_frames(
    tmp_path: Path,
    monkeypatch,
) -> None:  # noqa: ANN001
    task_dir = tmp_path / "task_1"
    context_dir = task_dir / "context"
    video_dir = context_dir / "video"
    video_dir.mkdir(parents=True)
    (video_dir / "briefing.mp4").write_bytes(b"fake video")
    task = _task_with_context(task_dir)

    def fake_preprocess_video(  # noqa: ANN001
        *, video_path, source_relative_path, generated_context_dir, config, **_kwargs
    ):
        del video_path, config
        timeline_path = generated_context_dir / "video" / "briefing_timeline.md"
        image_path = generated_context_dir / "video" / "briefing_stable_frames" / "stable_001.jpg"
        timeline_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.parent.mkdir(parents=True, exist_ok=True)
        timeline_path.write_text("# Video Timeline\n\nspeech\n", encoding="utf-8")
        image_path.write_bytes(b"fake image")
        return VideoPreprocessResult(
            source_path=source_relative_path,
            timeline=GeneratedVideoAsset(
                visible_path="video/briefing_timeline.md",
                physical_path=timeline_path,
                action="video_timeline",
            ),
            stable_frames=(
                GeneratedVideoAsset(
                    visible_path="video/briefing_stable_frames/stable_001.jpg",
                    physical_path=image_path,
                    action="video_stable_frame",
                ),
            ),
            manifest={
                "source_path": source_relative_path,
                "stable_frame_extraction": {
                    "saved_image_count": 1,
                    "stable_segment_count": 1,
                },
                "transcript": {"language": "en", "segments": [], "text": "speech"},
            },
        )

    monkeypatch.setattr(
        "data_agent_baseline.run.context_preprocessor.preprocess_video",
        fake_preprocess_video,
    )

    preprocessed = prepare_task_context(task, tmp_path / "output" / "task_1")
    listed_paths = {
        entry["path"]
        for entry in list_context_tree(preprocessed.task)["entries"]
        if entry["kind"] == "file"
    }

    assert listed_paths == {
        "video/briefing_timeline.md",
        "video/briefing_stable_frames/stable_001.jpg",
    }
    assert "video/briefing.mp4" not in listed_paths

    workspace = TaskContextWorkspace(
        source_root=preprocessed.task.context_dir,
        context_view=preprocessed.context_view,
    )
    workspace_root = workspace.materialize()

    assert (workspace_root / "video" / "briefing_timeline.md").exists()
    assert (workspace_root / "video" / "briefing_stable_frames" / "stable_001.jpg").exists()
    assert not (workspace_root / "video" / "briefing.mp4").exists()
    workspace.cleanup()

    task_output_dir = tmp_path / "output" / "task_1"
    video_artifact_dir = task_output_dir / "video_preprocessing" / "video__briefing.mp4"
    assert (video_artifact_dir / "timeline.md").read_text(encoding="utf-8").startswith(
        "# Video Timeline"
    )
    assert (video_artifact_dir / "transcript.json").exists()
    assert (video_artifact_dir / "video_preprocessing_manifest.json").exists()
    assert (video_artifact_dir / "stable_frames" / "stable_001.jpg").exists()

    manifest = json.loads(
        (task_output_dir / "context_preprocessing_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["video_preprocessing_dir"] == str(task_output_dir / "video_preprocessing")
    timeline_entry = next(
        entry for entry in manifest["entries"] if entry["action"] == "video_timeline"
    )
    assert timeline_entry["video_preprocessing"]["artifact_bundle"]["artifact_dir"] == str(
        video_artifact_dir
    )


def test_add_video_understanding_summary_keeps_timeline_and_frames(
    tmp_path: Path,
    monkeypatch,
) -> None:  # noqa: ANN001
    task_dir = tmp_path / "task_1"
    context_dir = task_dir / "context"
    video_dir = context_dir / "video"
    video_dir.mkdir(parents=True)
    (video_dir / "briefing.mp4").write_bytes(b"fake video")
    task = _task_with_context(task_dir)

    def fake_preprocess_video(  # noqa: ANN001
        *, video_path, source_relative_path, generated_context_dir, config, **_kwargs
    ):
        del video_path, config
        timeline_path = generated_context_dir / "video" / "briefing_timeline.md"
        image_path = generated_context_dir / "video" / "briefing_stable_frames" / "stable_001.jpg"
        timeline_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.parent.mkdir(parents=True, exist_ok=True)
        timeline_path.write_text("# Video Timeline\n\nfull transcript text\n", encoding="utf-8")
        image_path.write_bytes(b"fake image")
        return VideoPreprocessResult(
            source_path=source_relative_path,
            timeline=GeneratedVideoAsset(
                visible_path="video/briefing_timeline.md",
                physical_path=timeline_path,
                action="video_timeline",
            ),
            stable_frames=(
                GeneratedVideoAsset(
                    visible_path="video/briefing_stable_frames/stable_001.jpg",
                    physical_path=image_path,
                    action="video_stable_frame",
                ),
            ),
            manifest={
                "source_path": source_relative_path,
                "stable_frame_extraction": {
                    "saved_image_count": 1,
                    "stable_segment_count": 1,
                },
                "transcript": {"language": "en", "segments": [], "text": "full transcript text"},
            },
        )

    class SummaryModel:
        def __init__(self) -> None:
            self.invocations = []

        def invoke(self, messages):  # noqa: ANN001
            self.invocations.append(list(messages))
            return AIMessage(
                content=(
                    "The video shows threshold 100 and year 2020 at "
                    "`video/briefing_stable_frames/stable_001.jpg`."
                )
            )

    monkeypatch.setattr(
        "data_agent_baseline.run.context_preprocessor.preprocess_video",
        fake_preprocess_video,
    )

    task_output_dir = tmp_path / "output" / "task_1"
    preprocessed = prepare_task_context(task, task_output_dir)
    model = SummaryModel()
    enriched = add_video_understanding_summaries(
        preprocessed_context=preprocessed,
        task_output_dir=task_output_dir,
        model=model,
        timeout_seconds=None,
    )

    listed_paths = {
        entry["path"]
        for entry in list_context_tree(enriched.task)["entries"]
        if entry["kind"] == "file"
    }
    assert listed_paths == {
        "video/briefing_timeline.md",
        "video/briefing_stable_frames/stable_001.jpg",
        "video/briefing_video_summary.md",
    }
    assert len(model.invocations) == 1
    user_content = model.invocations[0][1].content
    assert isinstance(user_content, list)
    assert sum(1 for part in user_content if part.get("type") == "image_url") == 1
    assert "full transcript text" in user_content[0]["text"]

    summary_text = read_doc_preview(enriched.task, "video/briefing_video_summary.md")["preview"]
    assert "The video shows threshold 100 and year 2020" in str(summary_text)
    assert "video/briefing_timeline.md" in str(summary_text)
    assert "video/briefing_stable_frames/stable_001.jpg" in str(summary_text)
    assert "may be used directly as observed video evidence" not in str(summary_text)
    assert "navigation and planning aid" in str(summary_text)

    debug_summary_path = (
        task_output_dir / "video_understanding" / "video__briefing.mp4" / "summary.md"
    )
    assert debug_summary_path.read_text(encoding="utf-8") == (
        preprocessed.generated_context_dir / "video" / "briefing_video_summary.md"
    ).read_text(encoding="utf-8")

    manifest = json.loads((task_output_dir / "context_preprocessing_manifest.json").read_text(encoding="utf-8"))
    summary_entry = next(entry for entry in manifest["entries"] if entry["action"] == "video_summary")
    assert summary_entry["visible_path"] == "video/briefing_video_summary.md"
    assert summary_entry["video_understanding"]["status"] == "ok"
    assert summary_entry["video_understanding"]["frame_count"] == 1
    assert summary_entry["video_understanding"]["debug_summary_path"] == str(debug_summary_path)


def test_add_video_understanding_summary_failure_instructs_main_agent(
    tmp_path: Path,
    monkeypatch,
) -> None:  # noqa: ANN001
    task_dir = tmp_path / "task_1"
    context_dir = task_dir / "context"
    video_dir = context_dir / "video"
    video_dir.mkdir(parents=True)
    (video_dir / "briefing.mp4").write_bytes(b"fake video")
    task = _task_with_context(task_dir)

    def fake_preprocess_video(  # noqa: ANN001
        *, video_path, source_relative_path, generated_context_dir, config, **_kwargs
    ):
        del video_path, config
        timeline_path = generated_context_dir / "video" / "briefing_timeline.md"
        image_path = generated_context_dir / "video" / "briefing_stable_frames" / "stable_001.jpg"
        timeline_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.parent.mkdir(parents=True, exist_ok=True)
        timeline_path.write_text("# Video Timeline\n\nspeech\n", encoding="utf-8")
        image_path.write_bytes(b"fake image")
        return VideoPreprocessResult(
            source_path=source_relative_path,
            timeline=GeneratedVideoAsset(
                visible_path="video/briefing_timeline.md",
                physical_path=timeline_path,
                action="video_timeline",
            ),
            stable_frames=(
                GeneratedVideoAsset(
                    visible_path="video/briefing_stable_frames/stable_001.jpg",
                    physical_path=image_path,
                    action="video_stable_frame",
                ),
            ),
            manifest={
                "source_path": source_relative_path,
                "stable_frame_extraction": {
                    "saved_image_count": 1,
                    "stable_segment_count": 1,
                },
                "transcript": {"language": "en", "segments": [], "text": "speech"},
            },
        )

    class FailingSummaryModel:
        def invoke(self, messages):  # noqa: ANN001
            del messages
            raise RuntimeError("synthetic video summary failure")

    monkeypatch.setattr(
        "data_agent_baseline.run.context_preprocessor.preprocess_video",
        fake_preprocess_video,
    )

    task_output_dir = tmp_path / "output" / "task_1"
    preprocessed = prepare_task_context(task, task_output_dir)
    enriched = add_video_understanding_summaries(
        preprocessed_context=preprocessed,
        task_output_dir=task_output_dir,
        model=FailingSummaryModel(),
        timeout_seconds=None,
    )

    summary_text = str(
        read_doc_preview(enriched.task, "video/briefing_video_summary.md")["preview"]
    )
    assert "Video Understanding Summary Unavailable" in summary_text
    assert "synthetic video summary failure" in summary_text
    assert "video/briefing_timeline.md" in summary_text
    assert "read_context_image" in summary_text
    assert "do not rely on this failed summary" in summary_text
    assert (
        task_output_dir / "video_understanding" / "video__briefing.mp4" / "summary.md"
    ).read_text(encoding="utf-8").strip() == summary_text.strip()

    manifest = json.loads((task_output_dir / "context_preprocessing_manifest.json").read_text(encoding="utf-8"))
    summary_entry = next(entry for entry in manifest["entries"] if entry["action"] == "video_summary")
    assert summary_entry["video_understanding"]["status"] == "failed"
    assert "synthetic video summary failure" in summary_entry["video_understanding"]["error"]


def test_task_context_workspace_materializes_overlay_without_pdfs(tmp_path: Path) -> None:
    task_dir = tmp_path / "task_1"
    context_dir = task_dir / "context"
    (context_dir / "csv").mkdir(parents=True)
    (context_dir / "doc").mkdir()
    (context_dir / "csv" / "data.csv").write_text("id,value\n1,A\n", encoding="utf-8")
    _write_pdf(context_dir / "doc" / "report.pdf", [(72, "PDF body.")])
    task = _task_with_context(task_dir)
    preprocessed = prepare_task_context(task, tmp_path / "output" / "task_1")

    workspace = TaskContextWorkspace(
        source_root=preprocessed.task.context_dir,
        context_view=preprocessed.context_view,
    )
    workspace_root = workspace.materialize()

    assert (workspace_root / "csv" / "data.csv").read_text(encoding="utf-8") == "id,value\n1,A\n"
    assert (workspace_root / "doc" / "report.md").exists()
    assert not (workspace_root / "doc" / "report.pdf").exists()
    workspace.cleanup()
    assert workspace.path is None
    assert not workspace_root.exists()
