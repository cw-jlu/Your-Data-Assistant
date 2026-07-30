from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz

from data_agent_baseline.benchmark.schema import ContextAsset, ContextView, PublicTask, TaskAssets
from data_agent_baseline.config import VideoPreprocessingConfig
from data_agent_baseline.run.video_preprocessor import (
    GeneratedVideoAsset,
    is_video_path,
    preprocess_video,
    render_video_preprocessing_error_markdown,
    visible_path_for_video_timeline,
)


@dataclass(frozen=True, slots=True)
class PreprocessedContext:
    task: PublicTask
    context_view: ContextView
    generated_context_dir: Path
    manifest: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _PdfLine:
    page_index: int
    x: float
    x_end: float
    y: float
    y_end: float
    page_height: float
    block_index: int
    text: str


@dataclass(frozen=True, slots=True)
class _MarkdownBlock:
    page_index: int
    y: float
    y_end: float
    page_height: float
    text: str
    is_heading: bool = False
    last_line_x_end: float = 0.0
    page_text_x_end: float = 0.0


@dataclass(frozen=True, slots=True)
class _TocHeading:
    level: int
    title: str
    page_index: int
    y: float


_CJK_RE = re.compile(r"[\u3400-\u9fff]")
_SENTENCE_END_RE = re.compile(r"[。！？；：.!?;:]$")
_PARAGRAPH_START_RE = re.compile(r"^(?:[#>*\-+]|(?:\d+(?:\)|、|\.(?!\d))))\s*")


def _normalize_for_match(text: str) -> str:
    return re.sub(r"\s+", "", text.strip()).lower()


def _is_cjk_text(text: str) -> bool:
    return bool(_CJK_RE.search(text))


def _join_lines(previous: str, current: str) -> str:
    previous = previous.rstrip()
    current = current.lstrip()
    if not previous:
        return current
    if not current:
        return previous
    if _is_cjk_text(previous) or _is_cjk_text(current):
        return previous + current
    if previous.endswith("-") and current and current[0].islower():
        return previous[:-1] + current
    return previous + " " + current


def _should_join(previous: str, current: str) -> bool:
    if not previous or not current:
        return False
    if previous.lstrip().startswith("#") or current.lstrip().startswith("#"):
        return False
    if _PARAGRAPH_START_RE.match(current):
        return False
    return not _SENTENCE_END_RE.search(previous.rstrip())


def _markdown_heading(level: int, title: str) -> str:
    bounded_level = min(max(level, 1), 6)
    return f"{'#' * bounded_level} {title.strip()}"


def _extract_pdf_lines(document: fitz.Document) -> list[_PdfLine]:
    lines: list[_PdfLine] = []
    for page_index, page in enumerate(document):
        page_height = float(page.rect.height)
        page_dict = page.get_text("dict")
        for block_index, block in enumerate(page_dict.get("blocks", [])):
            if block.get("type", 0) != 0:
                continue
            for line in block.get("lines", []):
                text = "".join(str(span.get("text", "")) for span in line.get("spans", [])).strip()
                if not text:
                    continue
                bbox = line.get("bbox", [0, 0, 0, 0])
                lines.append(
                    _PdfLine(
                        page_index=page_index,
                        x=float(bbox[0]),
                        x_end=float(bbox[2]),
                        y=float(bbox[1]),
                        y_end=float(bbox[3]),
                        page_height=page_height,
                        block_index=block_index,
                        text=text,
                    )
                )
    return lines


def _extract_toc_headings(document: fitz.Document) -> list[_TocHeading]:
    headings: list[_TocHeading] = []
    for row in document.get_toc(simple=False):
        if len(row) < 3:
            continue
        level = int(row[0])
        title = str(row[1]).strip()
        page_number = int(row[2])
        if not title or page_number < 1:
            continue
        page_index = min(max(page_number - 1, 0), max(document.page_count - 1, 0))
        y = 0.0
        if len(row) >= 4 and isinstance(row[3], dict):
            destination = row[3]
            point = destination.get("to")
            if point is not None and hasattr(point, "y"):
                y = float(point.y)
            if isinstance(destination.get("page"), int):
                page_index = min(max(int(destination["page"]), 0), max(document.page_count - 1, 0))
        headings.append(_TocHeading(level=level, title=title, page_index=page_index, y=y))
    return headings


def _line_indexes_replaced_by_headings(
    page_lines: list[_PdfLine],
    page_headings: list[_TocHeading],
) -> set[int]:
    consumed: set[int] = set()
    for heading in page_headings:
        normalized_title = _normalize_for_match(heading.title)
        if not normalized_title:
            continue
        best_index: int | None = None
        best_distance = float("inf")
        for index, line in enumerate(page_lines):
            if index in consumed:
                continue
            if _normalize_for_match(line.text) != normalized_title:
                continue
            distance = abs(line.y - heading.y)
            if distance < best_distance:
                best_index = index
                best_distance = distance
        if best_index is not None and best_distance <= 60:
            consumed.add(best_index)
    return consumed


def _estimate_same_paragraph_gap(page_lines: list[_PdfLine]) -> float:
    gaps: list[float] = []
    heights: list[float] = []
    for previous, current in zip(page_lines, page_lines[1:]):
        gap = current.y - previous.y_end
        if gap >= 0:
            gaps.append(gap)
        height = previous.y_end - previous.y
        if height > 0:
            heights.append(height)
    if page_lines:
        last_height = page_lines[-1].y_end - page_lines[-1].y
        if last_height > 0:
            heights.append(last_height)

    median_height = sorted(heights)[len(heights) // 2] if heights else 16.0
    height_threshold = max(8.0, median_height * 0.75)
    if len(gaps) < 2:
        return height_threshold

    sorted_gaps = sorted(gaps)
    best_split: tuple[float, float] | None = None
    best_ratio = 1.0
    for left, right in zip(sorted_gaps, sorted_gaps[1:]):
        if left <= 0:
            continue
        ratio = right / left
        if ratio > best_ratio:
            best_ratio = ratio
            best_split = (left, right)
    if best_split is not None and best_ratio >= 1.8:
        clustered_threshold = sum(best_split) / 2
        return min(height_threshold, clustered_threshold)
    return height_threshold


def _paragraph_text(lines: list[_PdfLine]) -> str:
    paragraph = ""
    for line in lines:
        paragraph = _join_lines(paragraph, line.text)
    return paragraph.strip()


def _paragraph_blocks_for_page(page_lines: list[_PdfLine]) -> list[_MarkdownBlock]:
    if not page_lines:
        return []
    page_lines = sorted(page_lines, key=lambda line: (line.y, line.x, line.block_index))
    gap_threshold = _estimate_same_paragraph_gap(page_lines)
    page_text_x_end = max(line.x_end for line in page_lines)

    grouped: list[list[_PdfLine]] = []
    current: list[_PdfLine] = []
    for line in page_lines:
        if current and line.y - current[-1].y_end > gap_threshold:
            grouped.append(current)
            current = []
        current.append(line)
    if current:
        grouped.append(current)

    return [
        _MarkdownBlock(
            page_index=group[0].page_index,
            y=group[0].y,
            y_end=group[-1].y_end,
            page_height=group[-1].page_height,
            text=_paragraph_text(group),
            last_line_x_end=group[-1].x_end,
            page_text_x_end=page_text_x_end,
        )
        for group in grouped
    ]


def _blocks_with_toc_headings(lines: list[_PdfLine], headings: list[_TocHeading]) -> list[_MarkdownBlock]:
    output: list[_MarkdownBlock] = []
    page_count = max(
        [line.page_index for line in lines] + [heading.page_index for heading in headings],
        default=-1,
    ) + 1

    for page_index in range(page_count):
        page_lines = [line for line in lines if line.page_index == page_index]
        page_headings = [heading for heading in headings if heading.page_index == page_index]
        page_lines.sort(key=lambda line: (line.y, line.block_index))
        page_headings.sort(key=lambda heading: (heading.y, heading.level, heading.title))
        consumed_line_indexes = _line_indexes_replaced_by_headings(page_lines, page_headings)

        events: list[tuple[float, int, _MarkdownBlock]] = []
        for heading in page_headings:
            events.append(
                (
                    heading.y,
                    0,
                    _MarkdownBlock(
                        page_index=page_index,
                        y=heading.y,
                        y_end=heading.y,
                        page_height=page_lines[0].page_height if page_lines else 0.0,
                        text=_markdown_heading(heading.level, heading.title),
                        is_heading=True,
                    ),
                )
            )
        remaining_lines = [
            line for index, line in enumerate(page_lines) if index not in consumed_line_indexes
        ]
        for block in _paragraph_blocks_for_page(remaining_lines):
            events.append((block.y, 1, block))
        for _, _, block in sorted(events, key=lambda item: (item[0], item[1])):
            output.append(block)
    return output


def _blocks_without_toc_headings(lines: list[_PdfLine]) -> list[_MarkdownBlock]:
    output: list[_MarkdownBlock] = []
    page_indexes = sorted({line.page_index for line in lines})
    for page_index in page_indexes:
        output.extend(_paragraph_blocks_for_page([line for line in lines if line.page_index == page_index]))
    return output


def _should_merge_across_pages(previous: _MarkdownBlock, current: _MarkdownBlock) -> bool:
    if previous.is_heading or current.is_heading:
        return False
    if current.page_index != previous.page_index + 1:
        return False
    if _SENTENCE_END_RE.search(previous.text.rstrip()):
        return False
    near_page_bottom = previous.y_end >= previous.page_height * 0.84
    near_page_top = current.y <= current.page_height * 0.16
    reaches_right_edge = previous.last_line_x_end >= previous.page_text_x_end - 40
    return near_page_bottom and near_page_top and reaches_right_edge


def _render_markdown_blocks(blocks: list[_MarkdownBlock]) -> str:
    if not blocks:
        return ""
    output: list[str] = []
    previous_block: _MarkdownBlock | None = None
    for block in blocks:
        if output and previous_block is not None and _should_merge_across_pages(previous_block, block):
            output[-1] = _join_lines(output[-1], block.text)
        else:
            output.append(block.text.strip())
        previous_block = _MarkdownBlock(
            page_index=block.page_index,
            y=block.y,
            y_end=block.y_end,
            page_height=block.page_height,
            text=output[-1],
            is_heading=block.is_heading,
            last_line_x_end=block.last_line_x_end,
            page_text_x_end=block.page_text_x_end,
        )
    text = "\n".join(line for line in output if line).strip()
    return text + "\n" if text else ""


def _coalesce_markdown_lines(lines: list[str]) -> str:
    output: list[str] = []
    current_paragraph = ""

    def flush_paragraph() -> None:
        nonlocal current_paragraph
        if current_paragraph:
            output.append(current_paragraph.strip())
            current_paragraph = ""

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            flush_paragraph()
            continue
        if line.startswith("#"):
            flush_paragraph()
            if output:
                output.append("")
            output.append(line)
            output.append("")
            continue
        if current_paragraph and _should_join(current_paragraph, line):
            current_paragraph = _join_lines(current_paragraph, line)
        else:
            flush_paragraph()
            current_paragraph = line
    flush_paragraph()

    text = "\n".join(output)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text + "\n" if text else ""


def pdf_to_markdown(pdf_path: Path) -> str:
    with fitz.open(pdf_path) as document:
        lines = _extract_pdf_lines(document)
        headings = _extract_toc_headings(document)
        blocks = (
            _blocks_with_toc_headings(lines, headings)
            if headings
            else _blocks_without_toc_headings(lines)
        )
    return _render_markdown_blocks(blocks)


def _visible_path_for_pdf(source_path: Path, source_context_dir: Path) -> str:
    relative_pdf_path = source_path.relative_to(source_context_dir)
    candidate = relative_pdf_path.with_suffix(".md")
    source_md_peer = source_path.with_suffix(".md")
    if source_md_peer.exists():
        candidate = candidate.with_name(f"{candidate.stem}_pdf.md")
    return candidate.as_posix()


def _add_generated_asset(
    *,
    visible_assets: list[ContextAsset],
    manifest_entries: list[dict[str, Any]],
    generated_asset: GeneratedVideoAsset,
    source_path: str,
    extra_manifest: dict[str, Any] | None = None,
) -> None:
    visible_assets.append(
        ContextAsset(
            visible_path=generated_asset.visible_path,
            physical_path=generated_asset.physical_path,
            source_path=source_path,
            action=generated_asset.action,
            generated=True,
        )
    )
    manifest_entry: dict[str, Any] = {
        "source_path": source_path,
        "visible_path": generated_asset.visible_path,
        "physical_path": str(generated_asset.physical_path),
        "action": generated_asset.action,
        "generated": True,
    }
    if extra_manifest:
        manifest_entry.update(extra_manifest)
    manifest_entries.append(manifest_entry)


def _video_artifact_dir_name(source_relative_path: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", source_relative_path).strip("_") or "video"


def _save_video_artifact_bundle(
    *,
    task_video_artifacts_dir: Path,
    source_path: str,
    result: Any,
) -> dict[str, Any]:
    artifact_dir = task_video_artifacts_dir / _video_artifact_dir_name(source_path)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    timeline_artifact_path = artifact_dir / "timeline.md"
    shutil.copy2(result.timeline.physical_path, timeline_artifact_path)

    manifest_artifact_path = artifact_dir / "video_preprocessing_manifest.json"
    manifest_artifact_path.write_text(
        json.dumps(result.manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    transcript_artifact_path = artifact_dir / "transcript.json"
    transcript_artifact_path.write_text(
        json.dumps(result.manifest.get("transcript", {}), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    stable_frames_artifact_dir = artifact_dir / "stable_frames"
    stable_frames_source_dir = (
        result.stable_frames[0].physical_path.parent if result.stable_frames else None
    )
    if stable_frames_source_dir is not None and stable_frames_source_dir.exists():
        shutil.copytree(
            stable_frames_source_dir,
            stable_frames_artifact_dir,
            dirs_exist_ok=True,
        )
    else:
        stable_frames_artifact_dir.mkdir(parents=True, exist_ok=True)

    return {
        "artifact_dir": str(artifact_dir),
        "timeline_artifact_path": str(timeline_artifact_path),
        "manifest_artifact_path": str(manifest_artifact_path),
        "transcript_artifact_path": str(transcript_artifact_path),
        "stable_frames_artifact_dir": str(stable_frames_artifact_dir),
    }


def _save_failed_video_artifact_bundle(
    *,
    task_video_artifacts_dir: Path,
    source_path: str,
    timeline_path: Path,
    error: str,
) -> dict[str, Any]:
    artifact_dir = task_video_artifacts_dir / _video_artifact_dir_name(source_path)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    timeline_artifact_path = artifact_dir / "timeline.md"
    shutil.copy2(timeline_path, timeline_artifact_path)
    manifest_artifact_path = artifact_dir / "video_preprocessing_manifest.json"
    manifest_artifact_path.write_text(
        json.dumps(
            {
                "source_path": source_path,
                "status": "failed",
                "error": error,
                "timeline_artifact_path": str(timeline_artifact_path),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "artifact_dir": str(artifact_dir),
        "timeline_artifact_path": str(timeline_artifact_path),
        "manifest_artifact_path": str(manifest_artifact_path),
    }


def prepare_task_context_view(
    task: PublicTask,
    task_output_dir: Path,
    *,
    video_config: VideoPreprocessingConfig | None = None,
    prompt_model: Any | None = None,
    prompt_model_name: str | None = None,
) -> PreprocessedContext:
    source_context_dir = task.context_dir
    effective_video_config = video_config or VideoPreprocessingConfig()
    generated_context_dir = task_output_dir / "generated_context"
    if generated_context_dir.exists():
        shutil.rmtree(generated_context_dir)
    generated_context_dir.mkdir(parents=True, exist_ok=True)

    task_video_artifacts_dir = task_output_dir / "video_preprocessing"
    if task_video_artifacts_dir.exists():
        shutil.rmtree(task_video_artifacts_dir)

    # Remove stale full-context mirrors produced by older preprocessing runs.
    legacy_context_dir = task_output_dir / "context"
    if legacy_context_dir.exists():
        shutil.rmtree(legacy_context_dir)

    manifest_entries: list[dict[str, Any]] = []
    visible_assets: list[ContextAsset] = []
    knowledge_path = source_context_dir / "knowledge.md"
    knowledge_text = (
        knowledge_path.read_text(encoding="utf-8", errors="replace")
        if knowledge_path.is_file()
        else ""
    )
    for source_path in sorted(source_context_dir.rglob("*")):
        if source_path.is_dir():
            continue
        relative_path = source_path.relative_to(source_context_dir)
        relative_path_text = relative_path.as_posix()
        if source_path.suffix.lower() == ".pdf":
            visible_path = _visible_path_for_pdf(source_path, source_context_dir)
            target_path = generated_context_dir / visible_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            markdown = pdf_to_markdown(source_path)
            target_path.write_text(markdown, encoding="utf-8")
            visible_assets.append(
                ContextAsset(
                    visible_path=visible_path,
                    physical_path=target_path,
                    source_path=relative_path.as_posix(),
                    action="pdf_to_markdown",
                    generated=True,
                )
            )
            manifest_entries.append(
                {
                    "source_path": relative_path_text,
                    "visible_path": visible_path,
                    "physical_path": str(target_path),
                    "action": "pdf_to_markdown",
                    "generated": True,
                }
            )
            continue
        if is_video_path(source_path):
            timeline_visible_path = visible_path_for_video_timeline(relative_path_text)
            try:
                if not effective_video_config.enabled:
                    raise RuntimeError("Video preprocessing is disabled by configuration.")
                result = preprocess_video(
                    video_path=source_path,
                    source_relative_path=relative_path_text,
                    generated_context_dir=generated_context_dir,
                    config=effective_video_config,
                    question=task.question,
                    knowledge_text=knowledge_text,
                    prompt_model=prompt_model,
                    prompt_model_name=prompt_model_name,
                )
                artifact_bundle = _save_video_artifact_bundle(
                    task_video_artifacts_dir=task_video_artifacts_dir,
                    source_path=relative_path_text,
                    result=result,
                )
                _add_generated_asset(
                    visible_assets=visible_assets,
                    manifest_entries=manifest_entries,
                    generated_asset=result.timeline,
                    source_path=relative_path_text,
                    extra_manifest={
                        "video_preprocessing": {
                            "saved_image_count": result.manifest[
                                "stable_frame_extraction"
                            ].get("saved_image_count", 0),
                            "stable_segment_count": result.manifest[
                                "stable_frame_extraction"
                            ].get("stable_segment_count", 0),
                            "language": result.manifest["transcript"].get("language"),
                            "artifact_bundle": artifact_bundle,
                        }
                    },
                )
                for frame_asset in result.stable_frames:
                    _add_generated_asset(
                        visible_assets=visible_assets,
                        manifest_entries=manifest_entries,
                        generated_asset=frame_asset,
                        source_path=relative_path_text,
                    )
            except Exception as exc:  # noqa: BLE001
                target_path = generated_context_dir / timeline_visible_path
                target_path.parent.mkdir(parents=True, exist_ok=True)
                target_path.write_text(
                    render_video_preprocessing_error_markdown(relative_path_text, str(exc)),
                    encoding="utf-8",
                )
                artifact_bundle = _save_failed_video_artifact_bundle(
                    task_video_artifacts_dir=task_video_artifacts_dir,
                    source_path=relative_path_text,
                    timeline_path=target_path,
                    error=str(exc),
                )
                _add_generated_asset(
                    visible_assets=visible_assets,
                    manifest_entries=manifest_entries,
                    generated_asset=GeneratedVideoAsset(
                        visible_path=timeline_visible_path,
                        physical_path=target_path,
                        action="video_preprocessing_failed",
                    ),
                    source_path=relative_path_text,
                    extra_manifest={"error": str(exc), "artifact_bundle": artifact_bundle},
                )
            continue
        visible_path = relative_path.as_posix()
        visible_assets.append(
            ContextAsset(
                visible_path=visible_path,
                physical_path=source_path,
                source_path=visible_path,
                action="source",
                generated=False,
            )
        )
        manifest_entries.append(
            {
                "source_path": visible_path,
                "visible_path": visible_path,
                "physical_path": str(source_path),
                "action": "source",
                "generated": False,
            }
        )

    context_view = ContextView(
        source_context_dir=source_context_dir,
        generated_context_dir=generated_context_dir,
        assets=tuple(sorted(visible_assets, key=lambda asset: asset.visible_path)),
    )
    manifest = {
        "task_id": task.task_id,
        "source_context_dir": str(source_context_dir),
        "generated_context_dir": str(generated_context_dir),
        "video_preprocessing_dir": str(task_video_artifacts_dir),
        "entries": manifest_entries,
    }
    (task_output_dir / "context_preprocessing_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    preprocessed_task = PublicTask(
        record=task.record,
        assets=TaskAssets(
            task_dir=task.task_dir,
            context_dir=source_context_dir,
            context_view=context_view,
        ),
    )
    return PreprocessedContext(
        task=preprocessed_task,
        context_view=context_view,
        generated_context_dir=generated_context_dir,
        manifest=manifest,
    )


def prepare_task_context(
    task: PublicTask,
    task_output_dir: Path,
    *,
    video_config: VideoPreprocessingConfig | None = None,
    prompt_model: Any | None = None,
    prompt_model_name: str | None = None,
) -> PreprocessedContext:
    return prepare_task_context_view(
        task,
        task_output_dir,
        video_config=video_config,
        prompt_model=prompt_model,
        prompt_model_name=prompt_model_name,
    )
