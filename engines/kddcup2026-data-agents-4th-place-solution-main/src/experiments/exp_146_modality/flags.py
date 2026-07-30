"""exp_146 per-lever ablation flags.

Each lever is independently toggled by an env var so the bench can measure
its isolated effect. ALL flags off ⇒ exp_146 behaves identically to exp_143
(the sanity baseline).

  EXP146_ANSWER_SHAPE=1  → lever ①  DISTINCT→JOIN-fanout-only + NULL/BLANK preserve
  EXP146_PROSE=1         → lever ②  pypdf read + prose_docs inventory +
                                     catalog-404→doc redirect + prose-parse protocol
  EXP146_VIDEO=1         → lever ③  pre-extract video summary into preamble
                                     (and stop attaching the raw video to the loop)
  EXP146_KEYFRAMES=1     → attach extracted unique video keyframes as images
                                     (and stop attaching the raw video to the loop)
  EXP146_WATCH_VIDEO=1   → expose a watch_video tool instead of auto-attaching
                                     raw video/keyframes to the first message
  EXP146_PDF_PREPROCESS=1 → use pre-extracted PDF text cache for prose docs
                                     and surface PDF doc previews in preamble
  EXP146_PROSE_INDEX=1 → surface cached Qwen sub-agent prose indexes in preamble
  EXP146_SOURCE_ROUTER=1 → classify whether SQL should be primary/support/avoided
                                     from the pre-answer exploration trace
  EXP146_PREFIX_CACHE=0 → disable task-scoped X-Prefix-Cache-Key runtime header
                                     (enabled by default; not an experiment lever)

Flags are read LIVE from the environment (functions, not import-time constants)
so they are robust to import order — the bench sets the env, then runs.
"""
from __future__ import annotations

import os

_TRUE = {"1", "true", "yes", "on"}


def _on(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in _TRUE


def answer_shape_on() -> bool:
    return _on("EXP146_ANSWER_SHAPE")


def prose_on() -> bool:
    return _on("EXP146_PROSE")


def video_on() -> bool:
    return _on("EXP146_VIDEO")


def keyframes_on() -> bool:
    return _on("EXP146_KEYFRAMES")


def watch_video_on() -> bool:
    return _on("EXP146_WATCH_VIDEO")


def pdf_preprocess_on() -> bool:
    return _on("EXP146_PDF_PREPROCESS")


def prose_index_on() -> bool:
    return _on("EXP146_PROSE_INDEX")


def source_router_on() -> bool:
    return _on("EXP146_SOURCE_ROUTER")


def active_levers() -> list[str]:
    out = []
    if answer_shape_on():
        out.append("answer_shape")
    if prose_on():
        out.append("prose")
    if video_on():
        out.append("video")
    if keyframes_on():
        out.append("keyframes")
    if watch_video_on():
        out.append("watch_video")
    if pdf_preprocess_on():
        out.append("pdf_preprocess")
    if prose_index_on():
        out.append("prose_index")
    if source_router_on():
        out.append("source_router")
    return out
