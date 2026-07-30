"""exp_159 per-lever ablation flags.

exp_159 forks the exp_144 source-router profile used in the recent 60-task
benches. The production/default profile enables the Phase 2 v1 candidate
levers; env vars can still override each lever.

  EXP159_ANSWER_SHAPE=1  → lever ①  DISTINCT→JOIN-fanout-only + NULL/BLANK preserve
  EXP159_PROSE=1         → lever ②  pypdf read + prose_docs inventory +
                                     catalog-404→doc redirect + prose-parse protocol
  EXP159_VIDEO=1         → lever ③  pre-extract video summary into preamble
                                     (and stop attaching the raw video to the loop)
  EXP159_KEYFRAMES=1     → attach extracted unique video keyframes as images
                                     (and stop attaching the raw video to the loop)
  EXP159_VIDEO_KEYFRAME_NOTE=1 → extract visible task-relevant keyframe text
                                     into preamble + source router payload
  EXP159_WATCH_VIDEO=1   → expose a watch_video tool instead of auto-attaching
                                     raw video/keyframes to the first message
  EXP159_PDF_PREPROCESS=1 → use pre-extracted PDF text cache for prose docs
                                     and surface PDF doc previews in preamble
  EXP159_PROSE_INDEX=1 → surface cached Qwen sub-agent prose indexes in preamble
  EXP159_SOURCE_ROUTER=1 → classify whether SQL should be primary/support/avoided
                                     from the pre-answer exploration trace
  EXP159_ANTI_AGG=1 → classify lookup/list answer shape and suppress
                                     row-collapsing math hints
  EXP159_ANTI_AGG_SQL_GUARD=1 → POC: reject final SQL whose output SELECT list
                                     aggregates when anti_agg says NO_AGG
  EXP159_FINAL_SQL_GUARD=1 → classify and rewrite high-risk final SQL operations:
                                     unnecessary DISTINCT and NULL/blank filters
  EXP159_UNIT_PROMPT=1 → POC: answer-phase prompt nudge to preserve explicit
                                     units/scales when materializing final values
  EXP159_AUDIO_ASR=1 → transcribe task video audio into a cautious preamble hint
                                     (raw video attachment behavior is unchanged)
  EXP159_FINANCE_DOMAIN_PROMPT=1 → classify task domain and inject a compact
                                     finance-specific note for fund/stock/macro
  EXP159_PREFIX_CACHE=0 → disable task-scoped X-Prefix-Cache-Key runtime header
                                     (enabled by default; not an experiment lever)

Flags are read LIVE from the environment (functions, not import-time constants)
so they are robust to import order — the bench sets the env, then runs.
"""
from __future__ import annotations

import os

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}
_DEFAULT_ON = {
    "EXP159_ANSWER_SHAPE",
    "EXP159_PROSE",
    "EXP159_VIDEO_KEYFRAME_NOTE",
    "EXP159_PDF_PREPROCESS",
    "EXP159_SOURCE_ROUTER",
    "EXP159_ANTI_AGG",
    "EXP159_ANTI_AGG_SQL_GUARD",
    "EXP159_AUDIO_ASR",
    "EXP159_FINANCE_DOMAIN_PROMPT",
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
    return _on("EXP159_ANSWER_SHAPE")


def prose_on() -> bool:
    return _on("EXP159_PROSE")


def video_on() -> bool:
    return _on("EXP159_VIDEO")


def keyframes_on() -> bool:
    return _on("EXP159_KEYFRAMES")


def video_keyframe_note_on() -> bool:
    return _on("EXP159_VIDEO_KEYFRAME_NOTE")


def watch_video_on() -> bool:
    return _on("EXP159_WATCH_VIDEO")


def pdf_preprocess_on() -> bool:
    return _on("EXP159_PDF_PREPROCESS")


def prose_index_on() -> bool:
    return _on("EXP159_PROSE_INDEX")


def source_router_on() -> bool:
    return _on("EXP159_SOURCE_ROUTER")


def anti_agg_on() -> bool:
    return _on("EXP159_ANTI_AGG")


def anti_agg_sql_guard_on() -> bool:
    return _on("EXP159_ANTI_AGG_SQL_GUARD")


def final_sql_guard_on() -> bool:
    return _on("EXP159_FINAL_SQL_GUARD")


def unit_prompt_on() -> bool:
    return _on("EXP159_UNIT_PROMPT")


def audio_asr_on() -> bool:
    return _on("EXP159_AUDIO_ASR")


def finance_domain_prompt_on() -> bool:
    return _on("EXP159_FINANCE_DOMAIN_PROMPT")


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
    if finance_domain_prompt_on():
        out.append("finance_domain_prompt")
    return out
