"""Explore tool: wraps the Explorer sub-agent as a main-agent tool.

The `explore` tool replaces `inspect_files` in production: a single call
runs the Explorer sub-agent (3–10 discovery steps) and returns a complete
data map — file inventory, schemas, PDF entity group samples, knowledge
extraction, ETL sources, join paths, and warnings — so the main agent can
skip manual multi-step discovery.

Heavy imports (``agents.explorer.runner``) are deferred to call-time to
break the circular import chain: ``registry`` → ``explore`` → ``explorer``
→ ``agents.agent`` → ``runtime`` → ``tools`` → ``registry``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel, ConfigDict

from agents.benchmark.schema import PublicTask
from agents.explorer.etl_sources import ensure_knowledge_doc_etl_sources
from agents.tools.registry import ToolExecutionResult
from agents.tools.schema_normalize import normalize_for_vllm
from agents.tools.units import load_conversions


class ExploreInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


if TYPE_CHECKING:
    from agents.llm import ModelAdapter
    from agents.tools.registry import FunctionTool

logger = logging.getLogger(__name__)

EXPLORE_SCHEMA = normalize_for_vllm(ExploreInput.model_json_schema())  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]


def _ensure_column_conversions(task: PublicTask, content: dict[str, Any]) -> None:
    """Deterministically inject column_conversions from ``_units.json``.

    1. Fills missing ``column_conversions`` in ``schema_map`` entries.
    2. Appends a top-level ``unit_conversion`` array as a prominent reminder.
    """
    from agents.config import ETL_SCRATCH_ROOT

    cache_dir = ETL_SCRATCH_ROOT / task.task_id / "_cache"
    if not cache_dir.is_dir():
        return

    schema_map = cast(dict[str, Any], content.get("schema_map", {}) or {})
    top_level: list[dict[str, Any]] = []

    for units_path in sorted(cache_dir.glob("*_units.json")):
        conversions = load_conversions(units_path)
        if not conversions:
            continue

        col_conversions = {c.field: c.label for c in conversions}
        conv_by_field = {c.field: c for c in conversions}
        stem = units_path.stem.removesuffix("_units")

        for key, raw_entry in schema_map.items():
            if not isinstance(raw_entry, dict):
                continue
            entry = cast(dict[str, Any], raw_entry)
            key_stem = Path(key).stem if "/" in key else key
            if key_stem != stem:
                continue
            existing = cast(dict[str, str], entry.get("column_conversions", {}) or {})
            existing_lower = {field.lower() for field in existing}
            for col, note in col_conversions.items():
                if col.lower() not in existing_lower:
                    existing[col] = note
                    existing_lower.add(col.lower())
            if existing:
                entry["column_conversions"] = existing
            columns_dict: dict[str, Any] | None = entry.get("columns")  # type: ignore[assignment]
            if isinstance(columns_dict, dict):
                columns_by_lower = {str(name).lower(): str(name) for name in columns_dict}
                for col, conv in conv_by_field.items():
                    col_key = columns_by_lower.get(col.lower(), col)
                    col_info: dict[str, Any] | None = columns_dict.get(col_key)  # type: ignore[assignment]
                    if not isinstance(col_info, dict):
                        continue
                    suffix = (
                        f" [UNIT: stored as {conv.source},"
                        f" submit as {conv.target} — multiply by {conv.factor}]"
                    )
                    for desc_key in ("description", "semantic"):
                        if desc_key in col_info:
                            col_info[desc_key] = str(col_info[desc_key]) + suffix

        by_group: dict[tuple[str, str, float], list[str]] = {}
        for c in conversions:
            by_group.setdefault((c.source, c.target, c.factor), []).append(c.field)
        for (from_unit, to_unit, factor), columns in by_group.items():
            top_level.append(
                {
                    "file": stem,
                    "columns": columns,
                    "from": from_unit,
                    "to": to_unit,
                    "factor": factor,
                    "instruction": (
                        f"Multiply {', '.join(columns)} by {factor} "
                        f"to convert from {from_unit} to {to_unit} before submitting."
                    ),
                }
            )

    if top_level:
        content["unit_conversion"] = top_level


def create_explore_tool_definition(
    explorer_model: ModelAdapter,
    *,
    video_tool: FunctionTool | None = None,
) -> FunctionTool:
    """Build a FunctionTool that runs the Explorer sub-agent.

    `explore_video` 按任务动态注入：含视频任务把 ``video_tool`` 传给
    `run_explorer`，registry 在其内部组装一次——既是执行分发表，也是该 run
    每轮请求下推的工具广告源，广告与执行结构性同源。
    """
    from agents.tools.registry import FunctionTool as _FunctionTool

    def handler(task: PublicTask, _args: Any) -> ToolExecutionResult:
        import agents.explorer.runner as explorer_runner
        from agents.runtime.media import find_videos

        has_video = video_tool is not None and bool(find_videos(task.context_dir))
        result = explorer_runner.run_explorer(
            task, task.question, explorer_model, video_tool=video_tool if has_video else None
        )
        content: dict[str, Any]
        if result.success:
            try:
                parsed: Any = json.loads(result.findings)
                content = (
                    cast(dict[str, Any], parsed)
                    if isinstance(parsed, dict)
                    else {"findings": result.findings}
                )
            except (json.JSONDecodeError, TypeError):
                content = {"findings": result.findings}
            content["explorer_steps_used"] = result.steps_used
            content["explorer_fallback_used"] = result.fallback_used
            if has_video and result.video_findings:
                # 确定性透传：视频精确数值不经 explorer 二次转写直达主 agent
                content["video_findings"] = list(result.video_findings)
            _ensure_column_conversions(task, content)
            ensure_knowledge_doc_etl_sources(task, content)
        else:
            content = result.to_tool_content()
        return ToolExecutionResult(ok=result.success, content=content)

    return _FunctionTool(
        name="explore",
        description=(
            "Use FIRST to bootstrap context exploration. Launches a discovery-only "
            "sub-agent that inspects every file under context (columns, dtypes, "
            "row counts, table schemas, PDF entity_groups_sample, "
            "column_conversions), reads knowledge.md, samples key values, "
            "identifies document ETL needs and join paths, and analyzes videos "
            "if present — all in one call. Returns {files, schema_map, knowledge, "
            "etl_sources, join_paths, value_samples, warnings, unit_conversion}. "
            "No parameters "
            "needed. This replaces manual multi-step discovery. "
            "Example: explore({})"
        ),
        json_schema=EXPLORE_SCHEMA,
        handler=handler,
    )
