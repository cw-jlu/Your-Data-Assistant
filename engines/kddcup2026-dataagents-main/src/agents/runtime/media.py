"""Task media helpers: video discovery and video-agent content."""

from __future__ import annotations

import base64
import logging
import mimetypes
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".m4v", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}
VIDEO_MAX_BYTES = 100 * 1024 * 1024
DASHSCOPE_COMPRESS_THRESHOLD = 7 * 1024 * 1024


def find_videos(context_dir: Path) -> list[Path]:
    """Recurse into ``context/`` and return all video files, sorted for determinism."""
    return sorted(
        p for p in context_dir.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )


def _compress_video(src: Path, target_bytes: int = DASHSCOPE_COMPRESS_THRESHOLD) -> Path:
    """Compress video with ffmpeg to fit under *target_bytes*."""
    if not shutil.which("ffmpeg"):
        logger.warning("ffmpeg not found; skipping video compression")
        return src

    fd, tmp_name = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    tmp = Path(tmp_name)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-vcodec",
        "libx264",
        "-crf",
        "28",
        "-preset",
        "medium",
        "-vf",
        "scale='min(640,iw)':'min(480,ih)':force_original_aspect_ratio=decrease",
        "-an",
        str(tmp),
    ]
    try:
        subprocess.run(cmd, capture_output=True, timeout=60, check=True)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        logger.warning("Video compression failed: %s", exc)
        tmp.unlink(missing_ok=True)
        return src

    compressed_size = tmp.stat().st_size
    logger.info(
        "Video compressed: %s → %s (%d → %d bytes, %.0f%%)",
        src.name,
        tmp.name,
        src.stat().st_size,
        compressed_size,
        compressed_size / src.stat().st_size * 100,
    )
    return tmp


def build_video_user_content(
    text: str,
    video_path: Path,
    *,
    backend_kind: str | None = None,
) -> list[dict[str, Any]]:
    """Build multimodal user content for the video sub-agent: text + base64 data URI."""
    actual_path = video_path
    if backend_kind == "dashscope" and video_path.stat().st_size >= DASHSCOPE_COMPRESS_THRESHOLD:
        actual_path = _compress_video(video_path)

    try:
        mime_type = mimetypes.guess_type(str(actual_path))[0] or "video/mp4"
        video_b64 = base64.standard_b64encode(actual_path.read_bytes()).decode("ascii")
    finally:
        if actual_path != video_path:
            actual_path.unlink(missing_ok=True)

    return [
        {"type": "text", "text": text},
        {"type": "video_url", "video_url": {"url": f"data:{mime_type};base64,{video_b64}"}},
    ]
