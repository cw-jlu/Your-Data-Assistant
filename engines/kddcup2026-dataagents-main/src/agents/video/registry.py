"""Video sub-agent tool registry: a single terminal `report` tool.

校验经 `FunctionTool.input_model` 在 registry 边界强制——不依赖 vLLM
constrained decoding（`adapt_model_response` 的 recovered 调用路径会绕过它）。
"""

from __future__ import annotations

import json
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from agents.benchmark.schema import PublicTask
from agents.tools.registry import FunctionTool, ToolExecutionResult, ToolRegistry
from agents.tools.schema_normalize import normalize_for_vllm

# 报告序列化总量兜底（≈ schema 密集型 explorer findings 实测体量的 2 倍）：
# observation 会被逐字回放进 explorer 后续每一轮消息且消息层无截断，必须在执行边界拦截。
VIDEO_REPORT_MAX_BYTES = 64 * 1024


def _empty_filters() -> list[VideoCanonicalFilter]:
    return []


def _empty_metrics() -> list[VideoCanonicalMetric]:
    return []


def _empty_sorts() -> list[VideoCanonicalSort]:
    return []


def _empty_rule_items() -> list[VideoRuleItem]:
    return []


class VideoCanonicalFilter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str | None = Field(
        default=None,
        description="Source field or column this filter applies to.",
    )
    operator: str | None = Field(
        default=None,
        description="Comparator such as >, >=, <, <=, =, !=, in, between, contains.",
    )
    value: Any | None = Field(
        default=None,
        description="Threshold or comparison value. Preserve the exact visible value.",
    )
    unit: str | None = Field(
        default=None,
        description="Unit shown with the value, e.g. %, shares, yuan.",
    )
    source_rule_id: str | None = Field(
        default=None,
        description="ID of the rule_item this canonical filter came from, when known.",
    )
    source_time: str | None = Field(
        default=None,
        description="Video timestamp for this rule, when known.",
    )
    note: str | None = Field(
        default=None,
        description="Short caveat if the filter is incomplete or needs schema mapping.",
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_common_aliases(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        source = cast(dict[str, Any], value)
        out: dict[str, Any] = dict(source)
        op = out.pop("op", None)
        if "operator" not in out and op is not None:
            out["operator"] = op
        if "value" not in out:
            for key in ("threshold", "threshold_value", "condition_value"):
                if key in out:
                    out["value"] = out.pop(key)
                    break
        for key in ("threshold", "threshold_value", "condition_value"):
            out.pop(key, None)
        return out


class VideoCanonicalMetric(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str | None = Field(default=None, description="Metric field or column.")
    aggregation: str | None = Field(
        default=None,
        description="Aggregation such as count, sum, avg, min, max, or none.",
    )
    output_name: str | None = Field(
        default=None,
        description="Desired output label, if the video specifies one.",
    )
    source_rule_id: str | None = Field(default=None)
    note: str | None = Field(default=None)

    @model_validator(mode="before")
    @classmethod
    def _accept_common_aliases(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        source = cast(dict[str, Any], value)
        out: dict[str, Any] = dict(source)
        agg = out.pop("agg", None)
        if "aggregation" not in out and agg is not None:
            out["aggregation"] = agg
        metric = out.pop("metric", None)
        if "field" not in out and metric is not None:
            out["field"] = metric
        return out


class VideoCanonicalSort(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str | None = Field(default=None, description="Field to sort by.")
    direction: str | None = Field(
        default=None,
        description="Sort direction, usually asc or desc.",
    )
    source_rule_id: str | None = Field(default=None)
    note: str | None = Field(default=None)

    @model_validator(mode="before")
    @classmethod
    def _accept_common_aliases(cls, value: Any) -> Any:
        if isinstance(value, str):
            return {"field": value}
        if not isinstance(value, dict):
            return value
        source = cast(dict[str, Any], value)
        out: dict[str, Any] = dict(source)
        sort_field = out.pop("sort_field", None)
        if "field" not in out and sort_field is not None:
            out["field"] = sort_field
        sort_direction = out.pop("sort_direction", None)
        if "direction" not in out and sort_direction is not None:
            out["direction"] = sort_direction
        return out


class VideoCanonicalRules(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filters: list[VideoCanonicalFilter] = Field(
        default_factory=_empty_filters,
        description=(
            "Only fully representable executable filters. Leave empty for rules "
            "with exceptions, formulas, visual-only logic, or ambiguous mapping."
        ),
    )
    logic: str | None = Field(
        default=None,
        description="How filters combine, e.g. AND or OR. Omit when unknown.",
    )
    metrics: list[VideoCanonicalMetric] = Field(default_factory=_empty_metrics)
    group_by: list[str] = Field(default_factory=list)
    sort_by: list[VideoCanonicalSort] = Field(default_factory=_empty_sorts)
    top_n: int | str | None = Field(default=None)
    periods: list[Any] = Field(
        default_factory=list,
        description="Time windows, years, quarters, or period rules.",
    )
    identifiers: dict[str, Any] = Field(
        default_factory=dict,
        description="Batch IDs, scheme IDs, security codes, plan names, and similar IDs.",
    )
    tables: list[str] = Field(default_factory=list)
    output_fields: list[str] = Field(default_factory=list)
    snapshot_dates: list[str] = Field(default_factory=list)

    @field_validator("group_by", "tables", "output_fields", "snapshot_dates", mode="before")
    @classmethod
    def _string_to_list(cls, value: Any) -> Any:
        if value is None:
            return []
        if isinstance(value, str):
            text = value.strip()
            return [text] if text else []
        return value

    @field_validator("filters", "metrics", "sort_by", mode="before")
    @classmethod
    def _object_to_list(cls, value: Any) -> Any:
        if value is None:
            return []
        if isinstance(value, (dict, str)):
            return [value]
        return value


class VideoRuleItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str | None = Field(default=None, description="Stable local ID, e.g. r1.")
    kind: str | None = Field(
        default=None,
        description="Rule type such as filter, formula, exception_rule, ranking, grouping.",
    )
    description: str = Field(
        default="",
        description="Plain-language rule copied from the video as faithfully as possible.",
    )
    fields: list[str] = Field(default_factory=list)
    source_time: str | None = Field(default=None)
    confidence: str | None = Field(default=None, description="high, medium, or low.")
    structured: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional rule-specific structured payload that does not fit canonical.",
    )

    @field_validator("fields", mode="before")
    @classmethod
    def _field_to_list(cls, value: Any) -> Any:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return value


class VideoRules(BaseModel):
    model_config = ConfigDict(extra="forbid")

    canonical: VideoCanonicalRules = Field(
        default_factory=VideoCanonicalRules,
        description=(
            "Conservative executable rules. Fill only when the rule can be represented "
            "completely and safely; otherwise leave canonical empty and use rule_items."
        ),
    )
    rule_items: list[VideoRuleItem] = Field(
        default_factory=_empty_rule_items,
        description="Flexible rule descriptions that preserve complex or task-specific logic.",
    )
    raw_observations: list[str] = Field(
        default_factory=list,
        description="Verbatim or near-verbatim rule text visible in the video.",
    )
    task_specific: dict[str, Any] = Field(
        default_factory=dict,
        description="Question-relevant rule details that do not fit canonical fields.",
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_rules(cls, value: Any) -> Any:
        if value is None:
            return {}
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return {}
            try:
                value = json.loads(text)
            except json.JSONDecodeError:
                return value
        if not isinstance(value, dict):
            return value
        rules = cast(dict[str, Any], value)
        if not rules:
            return {}
        wrapper_keys = {"canonical", "rule_items", "raw_observations", "task_specific"}
        if wrapper_keys.intersection(rules):
            return rules
        canonical_keys = {
            "filters",
            "logic",
            "metrics",
            "group_by",
            "sort_by",
            "top_n",
            "periods",
            "identifiers",
            "tables",
            "output_fields",
            "snapshot_dates",
        }
        if set(rules).issubset(canonical_keys):
            return {"canonical": rules}
        return {"task_specific": rules}

    @field_validator("rule_items", mode="before")
    @classmethod
    def _rule_item_to_list(cls, value: Any) -> Any:
        if value is None:
            return []
        if isinstance(value, dict):
            return [value]
        return value

    @field_validator("raw_observations", mode="before")
    @classmethod
    def _raw_observations_to_list(cls, value: Any) -> Any:
        if value is None:
            return []
        if isinstance(value, str):
            text = value.strip()
            return [text] if text else []
        return value


class VideoExtractedData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rules: VideoRules = Field(
        default_factory=VideoRules,
        description=(
            "Configuration the video defines. Use `canonical` for complete executable "
            "query rules (filters, grouping, sorting, top-N, IDs, periods, tables). "
            "Use `rule_items`, `raw_observations`, and `task_specific` for complex "
            "or unusual rules so no information is lost."
        ),
    )
    displayed_samples: dict[str, Any] | list[Any] = Field(
        default_factory=dict,
        description=(
            "Numbers, distribution snapshots, TOP-N panels, preview rows, and other "
            "values the screen merely shows. These are not direct answer values. Use "
            "an object for keyed samples or an array for natural TOP-N/list panels."
        ),
    )


class VideoReportInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = Field(
        max_length=4000,
        description="What the video shows overall: content type, scenes, purpose.",
    )
    timeline: list[dict[str, Any]] = Field(  # pyright: ignore[reportUnknownVariableType]
        default_factory=list,
        description=(
            'Timestamped events, e.g. [{"time": "00:42", "event": "bar chart of Q3 '
            'revenue by category"}]. Cover the whole video. MUST be an array; use an '
            "empty array when there are no events. FORBIDDEN: null or one plain string."
        ),
    )
    extracted_data: VideoExtractedData = Field(
        default_factory=VideoExtractedData,
        description=(
            "Two top-level keys REQUIRED, even if one is empty: "
            "`rules` = query specification to apply against real data. Use "
            "rules.canonical only for complete executable structure; use "
            "rule_items, raw_observations, and task_specific for complex or "
            "unusual rules. "
            "`displayed_samples` = numbers, distribution snapshots, TOP-N panels, "
            "preview rows the screen merely SHOWS - NOT the answer. "
            "MUST be a structured object OR array. FORBIDDEN: quoted JSON text. "
            "Transcribe all numbers exactly as shown. Same value may appear in BOTH "
            "keys with a `warnings` note if you cannot tell which it is."
        ),
    )
    coverage: list[str] = Field(
        default_factory=list,
        description=(
            "Content you saw but did NOT fully transcribe (one item per entry), so the "
            "caller can request a targeted follow-up. MUST be an array of strings; use "
            "an empty array when there is no omitted content. FORBIDDEN: null or one "
            "plain string."
        ),
    )
    warnings: list[str] = Field(
        default_factory=list,
        description=(
            "Blurry / truncated / uncertain readings, one atomic fact each. MUST be an "
            "array of strings; use an empty array when there are no warnings. FORBIDDEN: "
            "null or one plain string."
        ),
    )

    @field_validator("extracted_data", mode="before")
    @classmethod
    def _parse_stringified_extracted_data(cls, value: Any) -> Any:
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return {}
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return value
        return value

    @field_validator("timeline", mode="before")
    @classmethod
    def _none_timeline_to_empty(cls, value: Any) -> Any:
        if value is None:
            return []
        return value

    @field_validator("coverage", "warnings", mode="before")
    @classmethod
    def _normalize_text_list_fields(cls, value: Any) -> Any:
        if value is None:
            return []
        if isinstance(value, str):
            text = value.strip()
            return [text] if text else []
        return value


VIDEO_REPORT_SCHEMA = normalize_for_vllm(VideoReportInput.model_json_schema())


def handle_video_report(_task: PublicTask, args: VideoReportInput) -> ToolExecutionResult:
    findings = json.dumps(args.model_dump(exclude_none=True), ensure_ascii=False)
    if len(findings.encode("utf-8")) > VIDEO_REPORT_MAX_BYTES:
        return ToolExecutionResult(
            ok=False,
            content={
                "error": (
                    f"report payload exceeds {VIDEO_REPORT_MAX_BYTES} bytes; resubmit "
                    "with only question-relevant data and move omitted content into "
                    "`coverage` entries"
                )
            },
        )
    return ToolExecutionResult(
        ok=True,
        content={"status": "reported", "findings": findings},
        is_terminal=True,
    )


def create_video_tool_registry() -> ToolRegistry:
    report = FunctionTool(
        name="report",
        description=(
            "Submit your video analysis findings. This is the ONLY way to "
            "complete the task — call it exactly once with complete findings. "
            "Pass each field as a separate parameter: summary, timeline, "
            "extracted_data (with tables/rules/text), coverage, warnings. "
            "Nested fields MUST stay structured objects/arrays. FORBIDDEN: "
            "stringified objects or stringified arrays."
        ),
        json_schema=VIDEO_REPORT_SCHEMA,
        handler=handle_video_report,
        is_terminal=True,
        input_model=VideoReportInput,
    )
    return ToolRegistry(definitions={report.name: report})
