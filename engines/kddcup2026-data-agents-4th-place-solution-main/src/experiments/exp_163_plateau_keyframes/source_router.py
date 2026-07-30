from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import Any

from kobushi_core.model import ModelAdapter, ModelMessage

from experiments.exp_163_plateau_keyframes.runtime import StepRecord


ROUTES = {"SQL_PRIMARY", "GREP_LITERAL", "DOC_VALUES", "VIDEO_GROUNDED", "MIXED_SOURCE"}
SQL_ROLES = {"primary", "support_only", "avoid"}

_NOTE_SUPPORT_ONLY_PATTERNS = (
    "retrieve the full",
    "full list from the database",
    "from the database",
    "return to the database",
    "query the database",
    "results not visible",
    "not visible",
    "not explicitly stated",
    "results are pending",
    "results pending",
    "export pending",
    "query is still pending",
    "回到数据库",
    "查询完整",
    "完整列表",
    "结果未显示",
    "未在屏幕上显示",
)


_SYSTEM = """You are a source-routing sub-agent for a data-answering ReAct agent.
Decide whether the proposed final SQL should be the primary source of truth,
only a support/materialization step, or avoided.

Return ONLY compact JSON:
{"route":"SQL_PRIMARY|GREP_LITERAL|DOC_VALUES|VIDEO_GROUNDED|MIXED_SOURCE",
 "sql_role":"primary|support_only|avoid",
 "reason":"one short sentence"}

Definitions:
- SQL_PRIMARY: structured DB/CSV/JSON data is the main source; SQL should compute/filter/aggregate.
- GREP_LITERAL: grep found an exact literal answer; preserve that text, SQL should not normalize it.
- DOC_VALUES: prose/PDF records are the main source; SQL may only materialize extracted literal VALUES.
- VIDEO_GROUNDED: video/keyframes provide source evidence.
  * If video/keyframes show the final displayed value/list/ranking, you MUST set sql_role=avoid.
  * If video/keyframes only define criteria/thresholds/fields and SQL must find matching rows, set sql_role=support_only.
- MIXED_SOURCE: doc/video evidence and SQL evidence both matter or conflict; SQL is usually support_only.

sql_role:
- primary: broad SQL over structured tables is allowed to produce the answer.
- support_only: SQL can map entities, verify attributes, or apply source-provided criteria, but must not contradict source evidence.
- avoid: final evidence is already literal/displayed; prefer SELECT/VALUES literals. Do not query a broad structured table to recompute/normalize it.
Do NOT output support_only when your reason says the video/grep/prose displays the final answer,
final list, or final ranking. In that case support_only is wrong; use avoid.

Generic examples:
- Video says a dashboard statistic is a displayed numeric value and the question asks that statistic:
  {"route":"VIDEO_GROUNDED","sql_role":"avoid","reason":"Video displays the final numeric answer."}
- Video shows a ranked list and the question asks for that displayed list:
  {"route":"VIDEO_GROUNDED","sql_role":"avoid","reason":"Video displays the final ranking/list."}
- Video shows a threshold/operator, while prose/SQL must enumerate all matching rows:
  {"route":"VIDEO_GROUNDED","sql_role":"support_only","reason":"Video defines criteria, but rows must be extracted."}
- Prose/PDF contains row values and SQL only materializes UNION ALL/VALUES:
  {"route":"DOC_VALUES","sql_role":"support_only","reason":"SQL is only materializing extracted document values."}

Prefer conservative routing. If video, grep, or prose/PDF evidence directly answers the question,
do not let SQL recompute a different answer.

If video_keyframe_note is provided, treat it as visible video evidence extracted
from keyframes, not as a routing decision. If it contains filters, thresholds,
dates, selected options, fields, or instructions such as retrieving a full list
from the database, SQL may be needed to apply those visible constraints. Do not
treat visible rows as complete final answers unless the note explicitly quotes a
screen saying they are complete/final.

If video_keyframe_note says results are not visible/pending, or says to retrieve
the full list from the database, choose sql_role=support_only rather than avoid."""


def _clip(value: Any, limit: int = 900) -> str:
    text = str(value).replace("\n", " | ")
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def _step_content(step: StepRecord) -> str:
    obs = step.observation or {}
    return _clip(obs.get("content", ""), 1200)


def _tool_counts(steps: list[StepRecord]) -> Counter[str]:
    return Counter(step.action for step in steps)


def _has_video_signal(steps: list[StepRecord]) -> bool:
    parts: list[str] = []
    for step in steps:
        parts.extend([step.thought or "", step.raw_response or "", _step_content(step)])
    hay = " ".join(parts).lower()
    return any(token in hay for token in ("watching video", "video shows", "[video]", "keyframe"))


def _has_prose_redirect(steps: list[StepRecord]) -> bool:
    return any("PROSE_DOC_REDIRECT" in _step_content(step) for step in steps)


def _important_evidence(steps: list[StepRecord], *, max_items: int = 12) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for step in steps:
        if step.action not in {"grep", "read_doc", "execute_sql", "describe_data", "list_context"}:
            continue
        item = {
            "tool": step.action,
            "thought": _clip(step.thought, 450),
            "observation": _step_content(step),
        }
        out.append(item)
    if len(out) <= max_items:
        return out
    # Keep the first source-discovery items and the latest evidence.
    return out[:3] + out[-(max_items - 3):]


def _deterministic_route(
    *,
    counts: Counter[str],
    saw_video: bool,
    saw_prose_redirect: bool,
    sql: str,
) -> dict[str, str]:
    read_doc_n = counts.get("read_doc", 0)
    grep_n = counts.get("grep", 0)
    execute_n = counts.get("execute_sql", 0)
    sql_lc = sql.lower()

    if saw_video:
        return {
            "route": "VIDEO_GROUNDED",
            "sql_role": "support_only",
            "reason": "Video evidence appeared in the trace; SQL should support or materialize it rather than override displayed criteria or answers.",
        }
    if saw_prose_redirect:
        return {
            "route": "DOC_VALUES",
            "sql_role": "support_only",
            "reason": "A catalog miss redirected the agent to a prose/PDF document; SQL should materialize extracted document values.",
        }
    if grep_n > 0 and read_doc_n == 0:
        return {
            "route": "GREP_LITERAL",
            "sql_role": "avoid",
            "reason": "The trace used grep without document paging, which usually indicates an exact literal lookup.",
        }
    if read_doc_n > 0 and execute_n > 0:
        return {
            "route": "MIXED_SOURCE",
            "sql_role": "support_only",
            "reason": "Both document reading and SQL exploration were used; verify source priority before confirming.",
        }
    if read_doc_n > 0:
        return {
            "route": "DOC_VALUES",
            "sql_role": "support_only",
            "reason": "The trace depends on prose/PDF reads; SQL should only materialize extracted values.",
        }
    if re.search(r"\b(values|union\s+all|select\s+['\"])", sql_lc):
        return {
            "route": "DOC_VALUES",
            "sql_role": "support_only",
            "reason": "The proposed SQL appears to materialize literal values rather than query structured tables.",
        }
    return {
        "route": "SQL_PRIMARY",
        "sql_role": "primary",
        "reason": "The trace does not show direct video/prose literal evidence, so structured SQL can be the primary source.",
    }


def _parse_route(text: str) -> dict[str, str] | None:
    try:
        data = json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
        except Exception:
            return None
    if not isinstance(data, dict):
        return None
    route = str(data.get("route", "")).strip()
    sql_role = str(data.get("sql_role", "")).strip()
    reason = str(data.get("reason", "")).strip()
    if route not in ROUTES or sql_role not in SQL_ROLES or not reason:
        return None
    return {"route": route, "sql_role": sql_role, "reason": _clip(reason, 500)}


def _note_prefers_support_only(note: str | None) -> bool:
    if not note:
        return False
    hay = note.lower()
    return any(pattern in hay for pattern in _NOTE_SUPPORT_ONLY_PATTERNS)


def _apply_video_note_guard(route: dict[str, Any], note: str | None) -> dict[str, Any]:
    if route.get("sql_role") != "avoid" or not _note_prefers_support_only(note):
        return route
    guarded = dict(route)
    guarded["route"] = "VIDEO_GROUNDED"
    guarded["sql_role"] = "support_only"
    guarded["reason"] = (
        "Video keyframe note provides criteria or visible records but says final results "
        "are pending/not visible or must be retrieved from the database; SQL should apply "
        "the visible constraints rather than be avoided."
    )
    guarded["video_note_guarded"] = True
    return guarded


def route_answer_source(
    *,
    question: str,
    steps: list[StepRecord],
    sql: str,
    model: ModelAdapter | None = None,
    video_keyframe_note: str | None = None,
) -> dict[str, Any]:
    counts = _tool_counts(steps)
    saw_video = _has_video_signal(steps)
    saw_prose_redirect = _has_prose_redirect(steps)
    fallback = _deterministic_route(
        counts=counts, saw_video=saw_video, saw_prose_redirect=saw_prose_redirect, sql=sql
    )
    fallback = _apply_video_note_guard(fallback, video_keyframe_note)
    usage = {
        "tool_counts": dict(counts),
        "saw_video_signal": saw_video,
        "saw_prose_redirect": saw_prose_redirect,
    }

    if model is None or os.environ.get("EXP163_SOURCE_ROUTER_MODEL", "1").strip() == "0":
        return {**fallback, **usage, "method": "deterministic"}

    payload = {
        "question": question,
        "proposed_sql": sql,
        "tool_counts": dict(counts),
        "saw_video_signal": saw_video,
        "saw_prose_redirect": saw_prose_redirect,
        "evidence": _important_evidence(steps),
    }
    if video_keyframe_note:
        payload["video_keyframe_note"] = _clip(video_keyframe_note, 2200)
    try:
        response = model.complete(
            [
                ModelMessage(role="system", content=_SYSTEM),
                ModelMessage(role="user", content=json.dumps(payload, ensure_ascii=False)),
            ],
            enable_thinking=False,
            max_tokens=256,
        )
        routed = _parse_route(response)
        if routed:
            routed = _apply_video_note_guard(routed, video_keyframe_note)
            return {**routed, **usage, "method": "model", "fallback": fallback}
    except Exception as exc:
        return {**fallback, **usage, "method": "deterministic_fallback", "router_error": str(exc)[:200]}
    return {**fallback, **usage, "method": "deterministic_fallback", "router_error": "invalid model JSON"}
