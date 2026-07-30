from __future__ import annotations

import csv
import json
import threading
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import cv2
import numpy as np

from data_agent_baseline.config import VideoPreprocessingConfig
from data_agent_baseline.asr_runtime import AsrPipelineRuntime, AsrRuntimeSettings

VIDEO_EXTENSIONS = frozenset({".mp4", ".m4v", ".mov", ".webm", ".mkv", ".avi"})
_MOJIBAKE_HINT_CHARS = frozenset("鎴戝閫欐槸鐩搁棞瑷烘柗鐨勬暣楂旈噺绱滄湁鍊嬮厤缃")
_OPENCC_T2S: Any | None = None
_ASR_RUNTIMES: dict[AsrRuntimeSettings, AsrPipelineRuntime] = {}
_ASR_RUNTIMES_LOCK = threading.Lock()


@dataclass(frozen=True, slots=True)
class GeneratedVideoAsset:
    visible_path: str
    physical_path: Path
    action: str


@dataclass(frozen=True, slots=True)
class VideoPreprocessResult:
    source_path: str
    timeline: GeneratedVideoAsset
    stable_frames: tuple[GeneratedVideoAsset, ...]
    manifest: dict[str, Any]


def is_video_path(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTENSIONS


def preprocess_frame_for_diff(frame: np.ndarray, width: int = 320) -> np.ndarray:
    h, w = frame.shape[:2]
    if w != width:
        new_h = max(1, int(h * width / w))
        frame = cv2.resize(frame, (width, new_h), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (3, 3), 0)


def changed_pixel_ratio(prev: np.ndarray, curr: np.ndarray, pixel_delta: int = 25) -> float:
    diff = cv2.absdiff(prev, curr)
    return float(np.mean(diff > pixel_delta))


def read_frame_at(cap: cv2.VideoCapture, time_sec: float) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0, time_sec) * 1000)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Could not read frame at {time_sec:.3f}s")
    return frame


def group_consecutive(indices: list[int]) -> list[list[int]]:
    if not indices:
        return []
    groups = [[indices[0]]]
    for idx in indices[1:]:
        if idx == groups[-1][-1] + 1:
            groups[-1].append(idx)
        else:
            groups.append([idx])
    return groups


def extract_stable_frames(
    video_path: Path,
    output_dir: Path,
    sample_fps: float = 2.0,
    diff_threshold: float = 0.025,
    pixel_delta: int = 25,
    min_stable_duration: float = 1.0,
    resize_width: int = 320,
    jpg_quality: int = 95,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = frame_count / fps if fps > 0 else 0
        sample_every = max(1, int(round(fps / sample_fps))) if fps > 0 else 15
        actual_sample_fps = fps / sample_every if fps > 0 else sample_fps
        sample_interval = 1.0 / actual_sample_fps if actual_sample_fps > 0 else 0.5

        times: list[float] = []
        preps: list[np.ndarray] = []
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % sample_every == 0:
                times.append(frame_idx / fps if fps > 0 else len(times) / sample_fps)
                preps.append(preprocess_frame_for_diff(frame, resize_width))
            frame_idx += 1

        if len(preps) < 2:
            raise RuntimeError("Too few sampled frames to analyze.")

        diffs = [0.0]
        for i in range(1, len(preps)):
            diffs.append(changed_pixel_ratio(preps[i - 1], preps[i], pixel_delta))

        stable_indices: list[int] = []
        for i in range(len(preps)):
            prev_ok = (i == 0) or (diffs[i] <= diff_threshold)
            next_ok = (i == len(preps) - 1) or (diffs[i + 1] <= diff_threshold)
            if prev_ok and next_ok:
                stable_indices.append(i)

        runs = group_consecutive(stable_indices)
        segments: list[dict[str, Any]] = []
        saved_count = 0

        for run in runs:
            start_t = times[run[0]]
            end_t = times[run[-1]] + sample_interval
            stable_duration = end_t - start_t
            if stable_duration < min_stable_duration:
                continue

            rep_idx = run[len(run) // 2]
            rep_t = times[rep_idx]
            frame = read_frame_at(cap, rep_t)

            item = {
                "segment_index": len(segments) + 1,
                "start_sec": round(start_t, 3),
                "end_sec": round(end_t, 3),
                "duration_sec": round(stable_duration, 3),
                "representative_sec": round(rep_t, 3),
                "saved": True,
                "filename": "",
            }

            saved_count += 1
            filename = f"stable_{saved_count:03d}_t{rep_t:07.2f}s.jpg"
            cv2.imwrite(
                str(output_dir / filename),
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality],
            )
            item["filename"] = filename

            segments.append(item)

        with (output_dir / "frame_diff_scores.csv").open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(["sample_index", "time_sec", "changed_pixel_ratio", "is_transition"])
            for i, (t, d) in enumerate(zip(times, diffs)):
                writer.writerow([i, round(t, 3), round(d, 6), int(d > diff_threshold)])

        manifest = {
            "video": str(video_path),
            "fps": fps,
            "frame_count": frame_count,
            "duration_sec": round(duration, 3),
            "sample_fps_requested": sample_fps,
            "sample_fps_actual": round(actual_sample_fps, 3),
            "diff_threshold": diff_threshold,
            "pixel_delta": pixel_delta,
            "min_stable_duration": min_stable_duration,
            "resize_width": resize_width,
            "sample_count": len(times),
            "stable_segment_count": len(segments),
            "saved_image_count": saved_count,
            "segments": segments,
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return manifest
    finally:
        cap.release()


def transcribe_video_audio(
    video_path: Path,
    *,
    model_name: str = "base",
    device: str = "cpu",
    compute_type: str = "int8",
    cpu_threads: int = 4,
    num_workers: int = 1,
    stage: str = "medium-baseline",
    detector_model_name: str = "tiny",
    language_threshold: float = 0.5,
    prompt_min_terms: int = 8,
    prompt_max_terms: int = 15,
    ui_terms: tuple[str, ...] = (),
    question: str = "",
    knowledge_text: str = "",
    prompt_model: Any | None = None,
    prompt_model_name: str | None = None,
    runtime: AsrPipelineRuntime | None = None,
) -> dict[str, Any]:
    settings = AsrRuntimeSettings(
        model_name=model_name,
        detector_model_name=detector_model_name,
        device=device,
        compute_type=compute_type,
        cpu_threads=cpu_threads,
        num_workers=num_workers,
        language_threshold=language_threshold,
        prompt_min_terms=prompt_min_terms,
        prompt_max_terms=prompt_max_terms,
        ui_terms=ui_terms or AsrRuntimeSettings().ui_terms,
    )
    effective_runtime = runtime
    if effective_runtime is None:
        with _ASR_RUNTIMES_LOCK:
            effective_runtime = _ASR_RUNTIMES.get(settings)
            if effective_runtime is None:
                effective_runtime = AsrPipelineRuntime(settings)
                _ASR_RUNTIMES[settings] = effective_runtime
    result = effective_runtime.transcribe(
        video_path,
        stage=stage,
        question=question,
        knowledge_text=knowledge_text,
        prompt_model=prompt_model,
        prompt_model_name=prompt_model_name,
    )
    return {
        "language": result["language"],
        "language_probability": result["language_probability"],
        "duration_sec": result["model_duration_seconds"],
        "segments": result["segments"],
        "text": result["text"],
        "asr": {
            "stage": result["stage"],
            "settings": {
                "model": model_name,
                "detector_model": detector_model_name,
                "device": device,
                "compute_type": compute_type,
                "cpu_threads": cpu_threads,
                "num_workers": num_workers,
                "language_threshold": language_threshold,
            },
            "language_detection": result["language_detection"],
            "prompt": result["prompt"],
            "timings": result["timings"],
        },
    }


def _mojibake_score(text: str) -> float:
    if not text:
        return 0.0
    replacement_count = text.count("\ufffd")
    hint_count = sum(1 for char in text if char in _MOJIBAKE_HINT_CHARS)
    return replacement_count * 4.0 + hint_count


def repair_transcript_mojibake(text: str) -> str:
    """Repair common UTF-8 Chinese text decoded through GBK/CP936 by ASR stacks."""
    original = text.strip()
    if not original:
        return original
    original_score = _mojibake_score(original)
    if original_score < 3:
        return original

    candidates = [original]
    for encoding in ("gbk", "cp936"):
        try:
            candidates.append(
                original.encode(encoding, errors="replace").decode("utf-8", errors="replace")
            )
        except UnicodeError:
            continue

    return min(candidates, key=_mojibake_score)


def simplify_chinese_transcript(text: str, *, language: object) -> str:
    """Convert detected Chinese ASR text to simplified Chinese."""
    normalized_language = str(language or "").strip().lower().replace("_", "-")
    if normalized_language != "zh" and not normalized_language.startswith("zh-"):
        return text

    global _OPENCC_T2S
    if _OPENCC_T2S is None:
        try:
            from opencc import OpenCC
        except ImportError as exc:
            raise RuntimeError(
                "opencc-python-reimplemented is required to normalize Chinese ASR output."
            ) from exc
        _OPENCC_T2S = OpenCC("t2s")
    return str(_OPENCC_T2S.convert(text))


def _format_time(seconds: float | int | None) -> str:
    if seconds is None:
        return "unknown"
    total_ms = int(round(float(seconds) * 1000))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
    return f"{m:02d}:{s:02d}.{ms:03d}"


def _overlaps(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    return max(a_start, b_start) <= min(a_end, b_end)


def _segments_for_window(
    transcript_segments: list[dict[str, Any]],
    start_sec: float,
    end_sec: float,
) -> list[dict[str, Any]]:
    return [
        segment
        for segment in transcript_segments
        if _overlaps(
            float(segment["start_sec"]),
            float(segment["end_sec"]),
            start_sec,
            end_sec,
        )
    ]


def render_video_timeline_markdown(
    *,
    source_path: str,
    stable_frames_visible_dir: str,
    stable_manifest: dict[str, Any],
    transcript: dict[str, Any],
) -> str:
    transcript_segments = list(transcript.get("segments") or [])
    lines: list[str] = [
        f"# Video Timeline: {source_path}",
        "",
        f"Source video: `{source_path}`",
        f"Duration: {_format_time(stable_manifest.get('duration_sec'))}",
        f"Detected language: {transcript.get('language') or 'unknown'}",
        f"Stable visual segments: {stable_manifest.get('stable_segment_count', 0)}",
        f"Saved stable frames: {stable_manifest.get('saved_image_count', 0)}",
        "",
        "Use the stable-frame images and transcript together. "
        "Each image path is relative to the task context directory.",
        "",
        "## Visual And Speech Timeline",
        "",
    ]

    for segment in stable_manifest.get("segments", []):
        start_sec = float(segment["start_sec"])
        end_sec = float(segment["end_sec"])
        image_name = segment.get("filename")
        image_path = (
            f"{stable_frames_visible_dir}/{image_name}"
            if image_name
            else "not saved; duplicate or unavailable"
        )
        lines.extend(
            [
                (
                    f"### Segment {segment['segment_index']}: "
                    f"{_format_time(start_sec)} - {_format_time(end_sec)}"
                ),
                "",
                f"- Representative time: {_format_time(segment.get('representative_sec'))}",
                f"- Stable frame: `{image_path}`",
            ]
        )
        window_segments = _segments_for_window(transcript_segments, start_sec, end_sec)
        if window_segments:
            lines.append("- Transcript in this visual segment:")
            for speech in window_segments:
                lines.append(
                    "  - "
                    f"[{_format_time(speech['start_sec'])} - {_format_time(speech['end_sec'])}] "
                    f"{speech['text']}"
                )
        else:
            lines.append("- Transcript in this visual segment: none detected")
        lines.append("")

    lines.extend(["## Full Transcript", ""])
    if transcript_segments:
        for speech in transcript_segments:
            lines.append(
                f"- [{_format_time(speech['start_sec'])} - {_format_time(speech['end_sec'])}] "
                f"{speech['text']}"
            )
    else:
        lines.append("No speech was transcribed.")
    lines.append("")
    return "\n".join(lines)


def render_video_preprocessing_error_markdown(source_path: str, error: str) -> str:
    return (
        f"# Video Timeline: {source_path}\n\n"
        "Video preprocessing failed for this source video.\n\n"
        f"- Source video: `{source_path}`\n"
        f"- Error: {error}\n\n"
        "The original video is intentionally not attached to the model context. "
        "Use the other available task context if this video could not be processed.\n"
    )


def visible_dir_for_stable_frames(source_relative_path: str) -> str:
    source = PurePosixPath(source_relative_path)
    parent = "" if str(source.parent) == "." else source.parent.as_posix()
    stable_dir = f"{source.stem}_stable_frames"
    return f"{parent}/{stable_dir}" if parent else stable_dir


def visible_path_for_video_timeline(source_relative_path: str) -> str:
    source = PurePosixPath(source_relative_path)
    parent = "" if str(source.parent) == "." else source.parent.as_posix()
    filename = f"{source.stem}_timeline.md"
    return f"{parent}/{filename}" if parent else filename


def preprocess_video(
    *,
    video_path: Path,
    source_relative_path: str,
    generated_context_dir: Path,
    config: VideoPreprocessingConfig,
    question: str = "",
    knowledge_text: str = "",
    prompt_model: Any | None = None,
    prompt_model_name: str | None = None,
) -> VideoPreprocessResult:
    stable_frames_visible_dir = visible_dir_for_stable_frames(source_relative_path)
    stable_frames_dir = generated_context_dir / stable_frames_visible_dir
    timeline_visible_path = visible_path_for_video_timeline(source_relative_path)
    timeline_path = generated_context_dir / timeline_visible_path
    timeline_path.parent.mkdir(parents=True, exist_ok=True)

    stable_manifest = extract_stable_frames(
        video_path=video_path,
        output_dir=stable_frames_dir,
        sample_fps=config.sample_fps,
        diff_threshold=config.diff_threshold,
        pixel_delta=config.pixel_delta,
        min_stable_duration=config.min_stable_duration,
        resize_width=config.resize_width,
        jpg_quality=config.jpg_quality,
    )
    transcript = transcribe_video_audio(
        video_path,
        model_name=config.asr_model,
        device=config.asr_device,
        compute_type=config.asr_compute_type,
        cpu_threads=config.asr_cpu_threads,
        num_workers=config.asr_num_workers,
        stage=config.asr_stage,
        detector_model_name=config.asr_language_detector_model,
        language_threshold=config.asr_language_threshold,
        prompt_min_terms=config.asr_prompt_min_terms,
        prompt_max_terms=config.asr_prompt_max_terms,
        ui_terms=config.asr_ui_terms,
        question=question,
        knowledge_text=knowledge_text,
        prompt_model=prompt_model,
        prompt_model_name=prompt_model_name,
    )
    timeline_path.write_text(
        render_video_timeline_markdown(
            source_path=source_relative_path,
            stable_frames_visible_dir=stable_frames_visible_dir,
            stable_manifest=stable_manifest,
            transcript=transcript,
        ),
        encoding="utf-8",
    )

    frame_assets = []
    for segment in stable_manifest.get("segments", []):
        filename = segment.get("filename")
        if not filename:
            continue
        visible_path = f"{stable_frames_visible_dir}/{filename}"
        frame_assets.append(
            GeneratedVideoAsset(
                visible_path=visible_path,
                physical_path=generated_context_dir / visible_path,
                action="video_stable_frame",
            )
        )

    manifest = {
        "source_path": source_relative_path,
        "timeline_visible_path": timeline_visible_path,
        "stable_frames_visible_dir": stable_frames_visible_dir,
        "stable_frame_extraction": stable_manifest,
        "transcript": transcript,
    }
    (stable_frames_dir / "video_preprocessing_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return VideoPreprocessResult(
        source_path=source_relative_path,
        timeline=GeneratedVideoAsset(
            visible_path=timeline_visible_path,
            physical_path=timeline_path,
            action="video_timeline",
        ),
        stable_frames=tuple(frame_assets),
        manifest=manifest,
    )
