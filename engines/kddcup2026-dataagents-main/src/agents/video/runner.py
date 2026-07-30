"""Video sub-agent runner: one short ReAct loop, report-only registry."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agents.agent import ReActAgent, ReActAgentConfig
from agents.benchmark.schema import PublicTask
from agents.llm import ModelAdapter
from agents.runtime import StepCallback
from agents.runtime.media import build_video_user_content
from agents.video.prompt import VIDEO_AGENT_SYSTEM_PROMPT, build_video_task_text
from agents.video.registry import VideoReportInput, create_video_tool_registry

logger = logging.getLogger(__name__)

VIDEO_AGENT_MAX_STEPS = 3


@dataclass(frozen=True, slots=True)
class VideoAgentResult:
    """Structured output from a video agent run."""

    findings: str
    steps_used: int
    success: bool
    failure_reason: str | None = None

    def to_tool_content(self) -> dict[str, Any]:
        """Format the failure case as a tool observation for the explorer."""
        return {
            "video_status": "failed",
            "failure_reason": self.failure_reason or "unknown",
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


def _extract_report(steps: list[Any]) -> str | None:
    """Walk steps backward for the validated report args.

    与 explorer 同款模式：report 终止后 `state.answer` 仍为 None，AgentRunResult 会
    被误填 "did not submit an answer" 的 failure_reason——成功判定只看 steps。
    """
    for step in reversed(steps):
        if getattr(step, "action", None) != "report" or not getattr(step, "ok", False):
            continue
        raw: Any = getattr(step, "action_input", None)
        if isinstance(raw, dict):
            report = VideoReportInput.model_validate(raw)
            return json.dumps(report.model_dump(exclude_none=True), ensure_ascii=False)
    return None


def run_video_agent(
    task: PublicTask,
    video_path: Path,
    instructions: str,
    model: ModelAdapter,
    *,
    max_steps: int = VIDEO_AGENT_MAX_STEPS,
    step_callback: StepCallback | None = None,
) -> VideoAgentResult:
    """Run the video sub-agent on one video file and return its report."""
    registry = create_video_tool_registry()
    traced_model, traced_registry = _maybe_trace(model, registry)

    agent = ReActAgent(
        model=traced_model,
        tools=traced_registry,
        config=ReActAgentConfig(max_steps=max_steps, max_empty_tool_call_retries=1),
        system_prompt=VIDEO_AGENT_SYSTEM_PROMPT,
        step_callback=step_callback,
    )
    source: Any = model
    while hasattr(source, "inner"):
        source = source.inner
    backend_kind: str | None = getattr(source, "backend_kind", None)
    content = build_video_user_content(
        build_video_task_text(task.question, instructions),
        video_path,
        backend_kind=backend_kind,
    )
    result = agent.run(task, initial_user_content=content)

    findings = _extract_report(result.steps)
    if findings is not None:
        return VideoAgentResult(findings=findings, steps_used=len(result.steps), success=True)
    return VideoAgentResult(
        findings="",
        steps_used=len(result.steps),
        success=False,
        failure_reason=result.failure_reason or "Video agent did not submit a report",
    )
