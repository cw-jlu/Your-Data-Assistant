"""exp_149 per-lever ablation flags.

exp_149 forks the exp_144 source-router profile used in the recent 60-task
benches. The production/default profile enables the Phase 2 v1 candidate
levers; env vars can still override each lever.

  EXP173_ANSWER_SHAPE=1  → lever ①  DISTINCT→JOIN-fanout-only + NULL/BLANK preserve
  EXP173_PROSE=1         → lever ②  pypdf read + prose_docs inventory +
                                     catalog-404→doc redirect + prose-parse protocol
  EXP173_VIDEO=1         → lever ③  pre-extract video summary into preamble
                                     (and stop attaching the raw video to the loop)
  EXP173_KEYFRAMES=1     → attach extracted unique video keyframes as images
                                     (and stop attaching the raw video to the loop)
  EXP173_VIDEO_KEYFRAME_NOTE=1 → extract visible task-relevant keyframe text
                                     into preamble + source router payload
  EXP173_WATCH_VIDEO=1   → expose a watch_video tool instead of auto-attaching
                                     raw video/keyframes to the first message
  EXP173_PDF_PREPROCESS=1 → use pre-extracted PDF text cache for prose docs
                                     and surface PDF doc previews in preamble
  EXP173_PROSE_INDEX=1 → surface cached Qwen sub-agent prose indexes in preamble
  EXP173_SOURCE_ROUTER=1 → classify whether SQL should be primary/support/avoided
                                     from the pre-answer exploration trace
  EXP173_ANTI_AGG=1 → classify lookup/list answer shape and suppress
                                     row-collapsing math hints
  EXP173_ANTI_AGG_SQL_GUARD=1 → POC: reject final SQL whose output SELECT list
                                     aggregates when anti_agg says NO_AGG
  EXP173_MATH_OTHER_ONLY=1 → fire the math-formula advisor ONLY when the domain
                                     router returns 'other' (BULL/EHR have ~0%
                                     complex calc); off → fire on every task
  EXP173_EXPLORE_SHAPE=1 → also inject the DISTINCT-fanout-only + NULL/BLANK-
                                     preservation rules ("the final answer
                                     includes NULL and duplicate rows; do not add
                                     WHERE <col> IS NOT NULL or a spurious
                                     DISTINCT") into the EXPLORE phase, where the
                                     agent builds/tests its result SQL (the
                                     answer_shape override only covers ANSWER/
                                     VERIFY). Default off for clean A/B vs v7;
                                     promote to on if it nets positive.
  EXP173_PREFIX_CACHE=0 → disable task-scoped X-Prefix-Cache-Key runtime header
                                     (enabled by default; not an experiment lever)

Flags are read LIVE from the environment (functions, not import-time constants)
so they are robust to import order — the bench sets the env, then runs.
"""
from __future__ import annotations

import os

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}
_DEFAULT_ON = {
    "EXP173_ANSWER_SHAPE",
    "EXP173_PROSE",
    "EXP173_VIDEO_KEYFRAME_NOTE",
    "EXP173_PDF_PREPROCESS",
    "EXP173_SOURCE_ROUTER",
    "EXP173_ANTI_AGG",
    "EXP173_ANTI_AGG_SQL_GUARD",
    "EXP173_DOMAIN_ROUTER",
    # EXP173_MATH_OTHER_ONLY intentionally NOT default-on: gating the math
    # advisor to domain='other' regressed 4 finance tasks (task_13/22/26/46) in
    # exp_167 run_002 — the formula acts as an aggregation-structure anchor that
    # suppresses agent cross-check drift, not just a complex-calc helper. Default
    # OFF = formula fires on every task (matches the exp_149/166 bench behavior).
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
    return _on("EXP173_ANSWER_SHAPE")


def prose_on() -> bool:
    return _on("EXP173_PROSE")


def video_on() -> bool:
    return _on("EXP173_VIDEO")


def keyframes_on() -> bool:
    return _on("EXP173_KEYFRAMES")


def video_keyframe_note_on() -> bool:
    return _on("EXP173_VIDEO_KEYFRAME_NOTE")


def watch_video_on() -> bool:
    return _on("EXP173_WATCH_VIDEO")


def pdf_preprocess_on() -> bool:
    return _on("EXP173_PDF_PREPROCESS")


def prose_index_on() -> bool:
    return _on("EXP173_PROSE_INDEX")


def source_router_on() -> bool:
    return _on("EXP173_SOURCE_ROUTER")


def anti_agg_on() -> bool:
    return _on("EXP173_ANTI_AGG")


def anti_agg_sql_guard_on() -> bool:
    return _on("EXP173_ANTI_AGG_SQL_GUARD")


def domain_router_on() -> bool:
    return _on("EXP173_DOMAIN_ROUTER")


def math_other_only_on() -> bool:
    return _on("EXP173_MATH_OTHER_ONLY")


def explore_shape_on() -> bool:
    return _on("EXP173_EXPLORE_SHAPE")


def prose_extract_on() -> bool:
    return _on("EXP173_PROSE_EXTRACT")


def prose_gate_on() -> bool:
    """exp_171: deterministic prose-needed gate in front of select+extract.
    OFF == exp_170 (always-extract). Only meaningful with prose_extract_on()."""
    return _on("EXP173_PROSE_GATE")


def ehr_distinct_on() -> bool:
    """exp_172: for EHR domains (eicu/mimic/ehr), drop the finance-tuned anti-DISTINCT
    guard and allow SELECT DISTINCT on set-valued questions ("how is X delivered / what
    routes/methods"). OFF == exp_171. Finance/other untouched."""
    return _on("EXP173_EHR_DISTINCT")


def icl_fewshot_on() -> bool:
    """exp_173: retrieve top-K similar (question, gold SQL) from the external per-domain
    corpus (BULL finance / EHRSQL) and inject them as few-shot demonstrations in the
    preamble. Per-domain (deterministic router picks the corpus); domain='other' is a
    no-op. Self-match (near-identical) neighbours are dropped so we teach the pattern,
    not the exact answer. OFF == exp_172/v9 behaviour."""
    return _on("EXP173_ICL_FEWSHOT")


def groupby_dim_on() -> bool:
    """exp_173: append a GROUP-BY-dimension teaching block to the fund/stock domain
    notes. Teaches (procedure, not answer): the GROUP BY key is the dimension named
    right after the grouping cue (不同/各/按…分 / different/by/per), NOT the entity being
    counted; and when several look-alike columns exist, pick by sampling distinct VALUES,
    not by column-name similarity. No specific column mapping (no leak); macro excluded
    (0% GROUP BY in BULL). OFF == exp_172."""
    return _on("EXP173_GROUPBY_DIM")


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
    if domain_router_on():
        out.append("domain_router")
    if math_other_only_on():
        out.append("math_other_only")
    if explore_shape_on():
        out.append("explore_shape")
    if prose_extract_on():
        out.append("prose_extract")
    if prose_gate_on():
        out.append("prose_gate")
    if ehr_distinct_on():
        out.append("ehr_distinct")
    if groupby_dim_on():
        out.append("groupby_dim")
    if icl_fewshot_on():
        out.append("icl_fewshot")
    return out
