"""exp_163 per-lever ablation flags.

exp_163 forks the exp_144 source-router profile used in the recent 60-task
benches. The production/default profile enables the Phase 2 v1 candidate
levers; env vars can still override each lever.

  EXP163_ANSWER_SHAPE=1  → lever ①  DISTINCT→JOIN-fanout-only + NULL/BLANK preserve
  EXP163_PROSE=1         → lever ②  pypdf read + prose_docs inventory +
                                     catalog-404→doc redirect + prose-parse protocol
  EXP163_VIDEO=1         → lever ③  pre-extract video summary into preamble
                                     (and stop attaching the raw video to the loop)
  EXP163_KEYFRAMES=1     → attach extracted unique video keyframes as images
                                     (and stop attaching the raw video to the loop)
  EXP163_VIDEO_KEYFRAME_NOTE=1 → extract visible task-relevant keyframe text
                                     into preamble + source router payload
  EXP163_WATCH_VIDEO=1   → expose a watch_video tool instead of auto-attaching
                                     raw video/keyframes to the first message
  EXP163_PDF_PREPROCESS=1 → use pre-extracted PDF text cache for prose docs
                                     and surface PDF doc previews in preamble
  EXP163_PROSE_INDEX=1 → surface cached Qwen sub-agent prose indexes in preamble
  EXP163_SOURCE_ROUTER=1 → classify whether SQL should be primary/support/avoided
                                     from the pre-answer exploration trace
  EXP163_ANTI_AGG=1 → classify lookup/list answer shape and suppress
                                     row-collapsing math hints
  EXP163_ANTI_AGG_SQL_GUARD=1 → POC: reject final SQL whose output SELECT list
                                     aggregates when anti_agg says NO_AGG
  EXP163_FINAL_SQL_GUARD=1 → classify and rewrite high-risk final SQL operations:
                                     unnecessary DISTINCT and NULL/blank filters
  EXP163_UNIT_PROMPT=1 → POC: answer-phase prompt nudge to preserve explicit
                                     units/scales when materializing final values
  EXP163_AUDIO_ASR=1 → transcribe task video audio into a cautious preamble hint
                                     (raw video attachment behavior is unchanged)
  EXP163_PREFIX_CACHE=0 → disable task-scoped X-Prefix-Cache-Key runtime header
                                     (enabled by default; not an experiment lever)

Flags are read LIVE from the environment (functions, not import-time constants)
so they are robust to import order — the bench sets the env, then runs.
"""
from __future__ import annotations

import os

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}
_DEFAULT_ON = {
    "EXP163_ANSWER_SHAPE",
    "EXP163_PROSE",
    "EXP163_VIDEO_KEYFRAME_NOTE",
    "EXP163_KEYFRAMES",
    "EXP163_VISUAL_ATTACH",
    "EXP163_PDF_PREPROCESS",
    "EXP163_SOURCE_ROUTER",
    "EXP163_ANTI_AGG",
    "EXP163_ANTI_AGG_SQL_GUARD",
    "EXP163_AUDIO_ASR",
}


def _on(name: str) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return name in _DEFAULT_ON
    value = raw.strip().lower()
    if value in _FALSE:
        return False
    return value in _TRUE


def answer_shape_on() -> bool:
    return _on("EXP163_ANSWER_SHAPE")


def prose_on() -> bool:
    return _on("EXP163_PROSE")


def video_on() -> bool:
    return _on("EXP163_VIDEO")


def keyframes_on() -> bool:
    return _on("EXP163_KEYFRAMES")


def visual_attach_on() -> bool:
    """Whether ANY visual (keyframe images or raw video) is attached to the
    agent's first message. Default ON. Set EXP163_VISUAL_ATTACH=0 for a
    note-only ablation: the text keyframe note still goes in the preamble, but
    no images / raw video are attached, so we can isolate the value of vision."""
    return _on("EXP163_VISUAL_ATTACH")


def video_keyframe_note_on() -> bool:
    return _on("EXP163_VIDEO_KEYFRAME_NOTE")


def watch_video_on() -> bool:
    return _on("EXP163_WATCH_VIDEO")


def pdf_preprocess_on() -> bool:
    return _on("EXP163_PDF_PREPROCESS")


def prose_index_on() -> bool:
    return _on("EXP163_PROSE_INDEX")


def source_router_on() -> bool:
    return _on("EXP163_SOURCE_ROUTER")


def anti_agg_on() -> bool:
    return _on("EXP163_ANTI_AGG")


def anti_agg_sql_guard_on() -> bool:
    return _on("EXP163_ANTI_AGG_SQL_GUARD")


def final_sql_guard_on() -> bool:
    return _on("EXP163_FINAL_SQL_GUARD")


def unit_prompt_on() -> bool:
    return _on("EXP163_UNIT_PROMPT")


def audio_asr_on() -> bool:
    return _on("EXP163_AUDIO_ASR")


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
    if video_keyframe_note_on():
        out.append("video_keyframe_note")
    if watch_video_on():
        out.append("watch_video")
    if pdf_preprocess_on():
        out.append("pdf_preprocess")
    if prose_index_on():
        out.append("prose_index")
    if source_router_on():
        out.append("source_router")
    if anti_agg_on():
        out.append("anti_agg")
    if anti_agg_sql_guard_on():
        out.append("anti_agg_sql_guard")
    if final_sql_guard_on():
        out.append("final_sql_guard")
    if unit_prompt_on():
        out.append("unit_prompt")
    if audio_asr_on():
        out.append("audio_asr")
    return out
