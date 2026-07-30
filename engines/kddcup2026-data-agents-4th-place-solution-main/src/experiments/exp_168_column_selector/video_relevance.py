from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any

from kobushi_core.benchmark.schema import PublicTask
from kobushi_core.model import ModelAdapter, ModelMessage


_SYSTEM = """You are a video-source relevance sub-agent.
Given a data question and attached video keyframes, decide whether the video
contains key information needed to answer the question.

Return ONLY compact JSON:
{"video_has_key_info":true|false,
 "key_info_type":"none|criteria|displayed_answer|both",
 "sql_role_hint":"primary|support_only|avoid",
 "reason":"one short sentence",
 "evidence":["short quoted on-screen evidence", "..."]}

Definitions:
- none: video has no useful task-specific information.
- criteria: video gives filters/thresholds/time window/scope/fields, but rows or
  aggregates still need SQL/prose.
- displayed_answer: video displays the final value/list/ranking/entities asked by
  the question.
- both: video gives criteria and also displays part/all of the final answer.

sql_role_hint:
- primary: SQL may be primary because video has no key information.
- support_only: video gives criteria/source constraints; SQL may enumerate rows.
- avoid: video displays the final answer/list/ranking; do not recompute it from SQL.

Use only the attached keyframes. Do not infer from gold labels or prior runs."""


def _parse_json(text: str) -> dict[str, Any] | None:
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
    key_info_type = str(data.get("key_info_type", "")).strip()
    sql_role_hint = str(data.get("sql_role_hint", "")).strip()
    if key_info_type not in {"none", "criteria", "displayed_answer", "both"}:
        return None
    if sql_role_hint not in {"primary", "support_only", "avoid"}:
        return None
    if key_info_type == "none":
        video_has_key_info = False
        sql_role_hint = "primary"
    elif key_info_type == "criteria":
        video_has_key_info = True
        sql_role_hint = "support_only"
    else:
        video_has_key_info = True
        sql_role_hint = "avoid"
    return {
        "video_has_key_info": video_has_key_info,
        "key_info_type": key_info_type,
        "sql_role_hint": sql_role_hint,
        "reason": str(data.get("reason", "")).strip()[:500],
        "evidence": data.get("evidence") if isinstance(data.get("evidence"), list) else [],
    }


def assess_keyframe_relevance(
    *,
    task: PublicTask,
    keyframes: list[Path],
    model: ModelAdapter,
    max_images: int = 18,
) -> dict[str, Any]:
    frames = list(keyframes[:max_images])
    if not frames:
        return {
            "video_has_key_info": False,
            "key_info_type": "none",
            "sql_role_hint": "primary",
            "reason": "No keyframes were provided.",
            "evidence": [],
            "method": "no_keyframes",
        }

    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": json.dumps(
                {
                    "task_id": task.task_id,
                    "question": task.question,
                    "instruction": "Inspect the attached keyframes and classify video-source relevance.",
                },
                ensure_ascii=False,
            ),
        }
    ]
    for path in frames:
        b64 = base64.b64encode(path.read_bytes()).decode()
        content.append({"type": "text", "text": f"Keyframe: {path.name}"})
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            }
        )

    response = model.complete(
        [
            ModelMessage(role="system", content=_SYSTEM),
            ModelMessage(role="user", content=content),
        ],
        enable_thinking=False,
        max_tokens=512,
    )
    parsed = _parse_json(response)
    if parsed is None:
        return {
            "video_has_key_info": False,
            "key_info_type": "none",
            "sql_role_hint": "primary",
            "reason": "Invalid model JSON.",
            "evidence": [],
            "method": "invalid_json",
            "raw_response": response[:1000],
        }
    parsed["method"] = "model_keyframes"
    parsed["n_keyframes"] = len(frames)
    return parsed
