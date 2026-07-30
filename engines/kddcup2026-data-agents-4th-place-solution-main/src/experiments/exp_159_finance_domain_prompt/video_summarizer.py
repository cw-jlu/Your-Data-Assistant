"""exp_159 lever ③ — pre-extract a structured summary of the briefing video.

A focused, single-purpose vision pass watches context/video/briefing.mp4 ONCE
(native video_url base64, which Qwen3.5 supports) and transcribes the
task-relevant facts. The summary is injected into the preamble as text, so the
main agent reads a reliable transcription instead of watching the video itself
mid-loop (where it tends to fabricate thresholds).

No gold literals: the prompt only asks the model to transcribe what the video
actually shows. Universal across tasks.
"""
from __future__ import annotations

import base64
from pathlib import Path

from kobushi_core.model import ModelAdapter, ModelMessage

_VIDEO_EXTS = (".mp4", ".webm", ".mov", ".avi", ".mkv")

_SUMMARY_PROMPT = """A short briefing/dashboard video is attached to this message.
Transcribe ONLY what the video actually shows — never invent numbers or labels.
Report in compact bullets (under 220 words), quoting on-screen text EXACTLY
(including any non-English text):

- CRITERIA: any rule, filter, threshold, time period, or scope the video defines,
  VERBATIM, each with its on-screen label and an mm:ss timestamp. Write
  "CRITERIA: none" if the video states no selection rule.
- DISPLAYED ANSWER: if the video shows a final result directly (any value, list,
  ranking, or entity), transcribe it VERBATIM, in order. Write "DISPLAYED ANSWER:
  none" if the answer must be computed from the database.
- DASHBOARD: one line naming the panel/view shown and the metric it tracks."""


def find_video(context_dir) -> Path | None:
    if context_dir is None:
        return None
    cd = Path(context_dir)
    if not cd.is_dir():
        return None
    for ext in _VIDEO_EXTS:
        for p in sorted(cd.glob(f"**/*{ext}")):
            return p
    return None


def summarize_video(task, model: ModelAdapter) -> str | None:
    """Return a text summary of the task's briefing video, or None if no video.

    On any failure returns a short error string (so the caller can still note a
    video exists) rather than raising."""
    vp = find_video(getattr(task, "context_dir", None))
    if vp is None:
        return None
    try:
        b64 = base64.b64encode(vp.read_bytes()).decode()
        content = [
            {"type": "text", "text": _SUMMARY_PROMPT},
            {"type": "video_url",
             "video_url": {"url": f"data:video/mp4;base64,{b64}"}},
        ]
        out = model.complete([ModelMessage(role="user", content=content)])
        return (out or "").strip() or None
    except Exception as exc:  # noqa: BLE001
        return f"[video summary unavailable: {exc!r}]"
