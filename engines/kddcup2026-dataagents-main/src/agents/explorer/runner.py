"""Explorer sub-agent runner.

Spawns a short ReAct loop with discovery tools to explore the data
landscape for a given task+question, then returns the findings.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, cast

from agents.agent import ReActAgent, ReActAgentConfig
from agents.benchmark.schema import PublicTask
from agents.explorer.prompt import EXPLORER_SYSTEM_PROMPT
from agents.explorer.registry import REPORT_KEYS, ReportInput, create_explorer_tool_registry
from agents.llm import ModelAdapter
from agents.runtime import StepCallback
from agents.tools.registry import FunctionTool

logger = logging.getLogger(__name__)

EXPLORER_MAX_STEPS = 16


@dataclass(frozen=True, slots=True)
class ExplorerResult:
    """Structured output from an Explorer run."""

    findings: str
    steps_used: int
    success: bool
    failure_reason: str | None = None
    fallback_used: bool = False
    # explore_video 成功 observation 的确定性透传（None = 本次运行没有视频调用）
    video_findings: tuple[dict[str, Any], ...] | None = None

    def to_tool_content(self) -> dict[str, Any]:
        """Format as a tool observation for the main agent."""
        if not self.success:
            return {
                "explorer_status": "failed",
                "failure_reason": self.failure_reason or "unknown",
                "steps_used": self.steps_used,
            }
        return {
            "explorer_status": "ok",
            "findings": self.findings,
            "steps_used": self.steps_used,
        }


def _maybe_trace(model: ModelAdapter, registry: Any) -> tuple[ModelAdapter, Any]:
    """Wrap model and registry with tracing if a trace context is active."""
    from agents.tracing.create import get_current_trace
    from agents.tracing.instrument import TracedModelAdapter, TracedToolRegistry

    if get_current_trace() is not None:
        if not isinstance(model, TracedModelAdapter):
            model = TracedModelAdapter(model)
        if not isinstance(registry, TracedToolRegistry):
            registry = TracedToolRegistry(registry)
    return model, registry


def run_explorer(
    task: PublicTask,
    question: str,  # reserved for future prompt override
    model: ModelAdapter,
    *,
    max_steps: int = EXPLORER_MAX_STEPS,
    step_callback: StepCallback | None = None,
    video_tool: FunctionTool | None = None,
) -> ExplorerResult:
    """Run the Explorer sub-agent on a task and return findings.

    The Explorer uses a discovery tool subset (no execute_python, no answer)
    and terminates via a `report` tool. It aims to complete in 3-6 steps.
    """
    registry = create_explorer_tool_registry(video_tool=video_tool)
    traced_model, traced_registry = _maybe_trace(model, registry)

    agent = ReActAgent(
        model=traced_model,
        tools=traced_registry,
        config=ReActAgentConfig(
            max_steps=max_steps,
            max_empty_tool_call_retries=3,
            stop_after_final_step_retries=True,
        ),
        system_prompt=EXPLORER_SYSTEM_PROMPT,
        step_callback=step_callback,
    )

    result = agent.run(task)
    video_findings = _extract_video_findings(result.steps) or None

    # Extract findings from steps: find the report tool call
    findings = _extract_findings(result.steps)

    if findings is not None:
        return ExplorerResult(
            findings=findings,
            steps_used=len(result.steps),
            success=True,
            fallback_used=False,
            video_findings=video_findings,
        )

    # Fallback: model didn't call answer — synthesize findings from observations
    synthesized = _synthesize_findings(result.steps)
    if synthesized:
        logger.info("Explorer fallback: synthesized findings from %d steps", len(result.steps))
        return ExplorerResult(
            findings=synthesized,
            steps_used=len(result.steps),
            success=True,
            fallback_used=True,
            video_findings=video_findings,
        )

    return ExplorerResult(
        findings="",
        steps_used=len(result.steps),
        success=False,
        failure_reason=result.failure_reason or "Explorer did not submit a report",
    )


def _synthesize_findings(steps: list[Any]) -> str:
    """Build a findings string from successful tool observations when model
    didn't call answer. Extracts the most useful data from each step."""
    import json

    sections: dict[str, Any] = {}
    for step in steps:
        action: str = getattr(step, "action", "")
        ok: bool = getattr(step, "ok", False)
        if not ok or action == "__error__":
            continue
        obs_raw: Any = getattr(step, "observation", {})
        if not isinstance(obs_raw, dict):
            continue
        obs = cast(dict[str, Any], obs_raw)
        raw: Any = obs.get("content", obs)
        content = cast(dict[str, Any], raw) if isinstance(raw, dict) else obs

        if action == "inspect_files":
            files_summary: list[dict[str, Any]] = []
            file_list: list[Any] = cast(list[Any], content.get("files", []))
            for f_item in file_list:
                f = cast(dict[str, Any], f_item)
                entry: dict[str, Any] = {
                    "path": f.get("path", ""),
                    "format": f.get("format", ""),
                }
                raw_schema: Any = f.get("schema", {})
                if isinstance(raw_schema, dict):
                    schema = cast(dict[str, Any], raw_schema)
                    if "columns" in schema:
                        entry["columns"] = schema["columns"]
                    if "row_count" in schema:
                        entry["row_count"] = schema["row_count"]
                    if "column_conversions" in schema:
                        entry["column_conversions"] = schema["column_conversions"]
                    if "tables" in schema:
                        tables = cast(dict[str, Any], schema["tables"])
                        entry["tables"] = {
                            t: {
                                "columns": cast(dict[str, Any], info).get("columns", []),
                                "row_count": cast(dict[str, Any], info).get("row_count"),
                            }
                            for t, info in tables.items()
                            if isinstance(info, dict)
                        }
                files_summary.append(entry)
            sections["files"] = files_summary

        elif action == "preview_file":
            action_input = cast(dict[str, Any], getattr(step, "action_input", {}))
            key = str(action_input.get("path", "file"))
            if "columns" in content:
                sections.setdefault("csv_previews", {})[key] = {
                    "columns": content.get("columns", []),
                    "dtypes": content.get("dtypes", []),
                    "row_count": content.get("row_count"),
                }
            elif "tables" in content:
                sections.setdefault("sqlite_schemas", {})[key] = content["tables"]
            elif "kind" in content:
                json_preview_keys = (
                    "kind",
                    "key_count",
                    "keys",
                    "keys_truncated",
                    "value_preview",
                    "length",
                    "head",
                    "head_truncation",
                    "first_item_kind",
                    "sample_value_kinds",
                    "value",
                    "value_kind",
                    "value_truncated",
                    "streamed",
                    "truncated",
                    "reason",
                )
                sections.setdefault("json_previews", {})[key] = {
                    k: content[k] for k in json_preview_keys if k in content
                }
            elif content.get("format") == "text" or "total_lines" in content:
                text_preview = {
                    k: content[k]
                    for k in ("total_lines", "head", "lines", "has_more")
                    if k in content
                }
                sections.setdefault("text_previews", {})[key] = text_preview

        elif action == "grep_context":
            sections.setdefault("grep_results", []).append(
                {
                    "pattern": content.get("pattern"),
                    "match_count": content.get("match_count"),
                    "results": cast(list[Any], content.get("results", []))[:10],
                }
            )

        elif action == "execute_context_sql":
            sections.setdefault("sql_results", []).append(
                {
                    "columns": content.get("columns", []),
                    "row_count": content.get("row_count"),
                    "rows": cast(list[Any], content.get("rows", []))[:5],
                }
            )

    return json.dumps(sections, ensure_ascii=False) if sections else ""


def _extract_findings(steps: list[Any]) -> str | None:
    """Walk steps backward to find the report tool's structured findings."""
    import json as _json

    for step in reversed(steps):
        if getattr(step, "action", None) == "report" and getattr(step, "ok", False):
            raw_args: Any = getattr(step, "action_input", None)
            if not isinstance(raw_args, dict):
                continue
            report = ReportInput.model_validate(raw_args)
            assembled = {
                k: getattr(report, k, {} if k in ("schema_map", "knowledge") else [])
                for k in REPORT_KEYS
            }
            return _json.dumps(assembled, ensure_ascii=False)
    return None


def _extract_video_findings(steps: list[Any]) -> tuple[dict[str, Any], ...]:
    """Collect successful explore_video observations in step order (deterministic)."""
    found: list[dict[str, Any]] = []
    for step in steps:
        if getattr(step, "action", None) != "explore_video" or not getattr(step, "ok", False):
            continue
        obs: Any = getattr(step, "observation", None)
        if isinstance(obs, dict):
            obs_dict = cast(dict[str, Any], obs)
            content: Any = obs_dict.get("content", obs_dict)
            found.append(cast(dict[str, Any], content if isinstance(content, dict) else obs_dict))
    return tuple(found)
