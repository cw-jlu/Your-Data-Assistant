"""explore_video tool: wraps the video sub-agent as an explorer-only tool.

与 `tools/explore.py` 同款 agent-as-tool 模式；重 import（agents.video.runner）
推迟到调用期以断开 registry → video → agent → runtime → tools 的环。
该工具的全部使用指引都在 description 里：registry 不含本工具时模型完全无感知
（`build_system_message` 不渲染工具文档，说明只经请求体 tools=[...] 下推）。
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

from agents.benchmark.schema import PublicTask
from agents.tools._fields import path_field
from agents.tools.registry import ToolExecutionResult
from agents.tools.schema_normalize import normalize_for_vllm


class ExploreVideoInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: Annotated[
        str,
        path_field(file_kind="video file", examples=("video.mp4", "media/demo.mov")),
    ]
    instructions: str = Field(
        description=(
            "Analysis brief for the video agent: what the task needs from this video — "
            "which charts, tables, values, or text to transcribe, plus any IDs or labels "
            "to look for."
        ),
    )


EXPLORE_VIDEO_SCHEMA = normalize_for_vllm(ExploreVideoInput.model_json_schema())

if TYPE_CHECKING:
    from agents.llm import ModelAdapter
    from agents.tools.registry import FunctionTool

logger = logging.getLogger(__name__)


def create_explore_video_tool_definition(video_model: ModelAdapter) -> FunctionTool:
    """Build the FunctionTool that runs the video sub-agent on one context video."""
    from agents.tools.registry import FunctionTool as _FunctionTool

    def handler(task: PublicTask, args: ExploreVideoInput) -> ToolExecutionResult:
        import agents.runtime.media as media
        import agents.video.runner as video_runner
        from agents.tools.context import resolve_context_path

        try:
            video_path = resolve_context_path(task, args.path)
        except (ValueError, FileNotFoundError) as exc:
            return ToolExecutionResult(ok=False, content={"error": str(exc)})
        if video_path.suffix.lower() not in media.VIDEO_EXTENSIONS:
            return ToolExecutionResult(
                ok=False, content={"error": f"{args.path} is not a video file"}
            )
        size = video_path.stat().st_size
        if size > media.VIDEO_MAX_BYTES:
            return ToolExecutionResult(
                ok=False,
                content={
                    "error": (
                        f"{args.path} is {size} bytes, which exceeds the "
                        f"{media.VIDEO_MAX_BYTES} byte limit"
                    )
                },
            )

        result = video_runner.run_video_agent(task, video_path, args.instructions, video_model)
        if not result.success:
            return ToolExecutionResult(ok=False, content=result.to_tool_content())
        try:
            findings: Any = json.loads(result.findings)
        except (json.JSONDecodeError, TypeError):
            findings = result.findings
        content: dict[str, Any] = {
            "video_status": "ok",
            "path": args.path,
            "findings": findings,
            "video_steps_used": result.steps_used,
        }
        # 把 video sub-agent 抽出的 rules 立刻翻译成可执行 pandas 模板,贴在
        # observation 里 — 下游 agent 自由写 SQL/python 时经常误用规则(列名
        # 选错、阈值边界、忘 top_n / group_by / batch_id),这一步把"翻译"
        # 这一段从 LLM 主观判断变成 deterministic 预处理。
        from typing import cast as _cast

        from agents.video.rule_templater import synthesize_filter_recipe

        rules: dict[str, Any] | None = None
        if isinstance(findings, dict):
            findings_dict = _cast(dict[str, Any], findings)
            extracted = findings_dict.get("extracted_data")
            if isinstance(extracted, dict):
                extracted_dict = _cast(dict[str, Any], extracted)
                rules_raw = extracted_dict.get("rules")
                if isinstance(rules_raw, dict):
                    rules = _cast(dict[str, Any], rules_raw)
        recipe_text = synthesize_filter_recipe(rules)
        if recipe_text is not None:
            content["filter_recipe"] = recipe_text
        return ToolExecutionResult(ok=True, content=content)

    return _FunctionTool(
        name="explore_video",
        description=(
            "Launch the video analysis sub-agent on one video file from context. "
            "Returns {video_status, path, findings, video_steps_used, "
            "filter_recipe}. Use ONLY on video files listed by inspect_files. "
            "Write instructions as an analysis brief tied to the task question: "
            "which charts, tables, values, or text to transcribe. Call once per "
            "video. Findings pass through to the main agent verbatim — reference "
            "key facts in your report's knowledge field but do NOT re-transcribe "
            "numbers. "
            'Example: explore_video({"path": "video.mp4", '
            '"instructions": "Extract the bar chart values for Q1-Q4 revenue"})'
        ),
        json_schema=EXPLORE_VIDEO_SCHEMA,
        handler=handler,
        input_model=ExploreVideoInput,
    )
