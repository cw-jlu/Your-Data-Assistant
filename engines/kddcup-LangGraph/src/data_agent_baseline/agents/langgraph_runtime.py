from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from collections.abc import Callable
from typing import Any, get_args, get_origin
from uuid import uuid4

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph

from data_agent_baseline.agents.answer_validator import validate_answer as invoke_answer_validator
from data_agent_baseline.agents.multimodal import build_initial_user_content
from data_agent_baseline.agents.process_validator import (
    validate_process as invoke_process_validator,
)
from data_agent_baseline.agents.prompt import build_system_prompt
from data_agent_baseline.agents.prompt2 import build_system_prompt_v2
from data_agent_baseline.agents.prompt3 import build_system_prompt_v3
from data_agent_baseline.agents.submission_risk_detector import (
    detect_submission_risks,
)
from data_agent_baseline.agents.submission_field_context import (
    build_submission_field_context,
)
from data_agent_baseline.agents.ambiguity_analyzer import analyze_ambiguity
from data_agent_baseline.agents.runtime import AgentRunResult, StepRecord
from data_agent_baseline.agents.state import AgentGraphState
from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.config import DataInspectorConfig, ProcessValidatorConfig
from data_agent_baseline.inspectors import DataUnderstandingAgent
from data_agent_baseline.model_retry import invoke_model_with_retries, summarize_model_retry_events
from data_agent_baseline.tools.python_exec import TaskContextWorkspace
from data_agent_baseline.tools.registry import ToolRegistry, ToolRuntimeContext
from data_agent_baseline.tools.truncation import (
    TRUNCATION_SUFFIX,
    truncate_answer_content,
    truncate_content,
)

logger = logging.getLogger(__name__)


TraceCallback = Callable[[dict[str, Any]], None]
REASONING_HISTORY_DERIVED_CONTENT_KEY = "_dab_reasoning_history_derived_content"
ANSWER_VALIDATOR_MAX_PREVIEW_ROWS = 50
ANSWER_VALIDATOR_DISTINCT_EXAMPLES_PER_COLUMN = 10
_DEFAULT_EVIDENCE_MAX_ITEMS = 12
_DEFAULT_EVIDENCE_MAX_STR_TOKENS = 300
_DEFAULT_EVIDENCE_MAX_LIST_ITEMS = 5


@dataclass(frozen=True, slots=True)
class LangGraphAgentConfig:
    max_steps: int = 16
    model_request_timeout_seconds: int | None = 1800
    empty_stop_retry_limit: int = 2
    # Maximum number of times answer validation can reject and return to the main agent.
    validation_retry_limit: int = 2
    enable_answer_validator: bool = True
    enable_process_validator: bool = False
    enable_data_inspector: bool = False
    enable_ambiguity_analysis: bool = False
    strip_reasoning_history: bool = False
    reasoning_history_limit: int | None = None
    data_inspector: DataInspectorConfig = field(default_factory=DataInspectorConfig)
    process_validator: ProcessValidatorConfig = field(default_factory=ProcessValidatorConfig)
    prompt_version: int = 1
    max_attached_video_frames: int = 16
    compress_used_image_messages: bool = True
    compressed_image_note_chars: int = 600

    def __post_init__(self) -> None:
        if self.reasoning_history_limit is not None and self.reasoning_history_limit < 0:
            raise ValueError("reasoning_history_limit must be None or a non-negative integer.")
        if self.max_attached_video_frames < 0:
            raise ValueError("max_attached_video_frames must be non-negative.")
        if self.compressed_image_note_chars < 0:
            raise ValueError("compressed_image_note_chars must be non-negative.")


EMPTY_STOP_REPAIR_PROMPT = (
    "Your previous response did not call a tool. In the next turn, immediately call a tool."
)
INVALID_TOOL_CALL_REPAIR_PROMPT = (
    "Your previous tool call was discarded because its arguments were not valid JSON. "
    "Immediately call one tool again. The arguments must be one complete JSON object: "
    "use double-quoted keys and strings, escape literal double quotes inside string values, "
    "and put no commentary inside the tool arguments."
)
FORCE_ANSWER_PROMPT = (
    "You have reached the maximum number of model steps for this task. "
    "Do not call any exploratory tools or continue analysis. "
    "Use the information already gathered in the conversation and immediately call "
    "`submit_tool_result` with your best final answer table. If the evidence is "
    "incomplete, submit the best answer you can infer from the available evidence."
)

PSEUDO_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<function=([A-Za-z_][A-Za-z0-9_]*)>\s*(.*?)\s*</function>\s*</tool_call>",
    re.DOTALL,
)
PSEUDO_FUNCTION_TAG_RE = re.compile(
    r"<function=([A-Za-z_][A-Za-z0-9_]*)>\s*(.*?)\s*</function>",
    re.DOTALL,
)
PSEUDO_TOOL_CALL_BLOCK_RE = re.compile(
    r"<tool_call\b[^>]*>.*?</tool_call>",
    re.DOTALL | re.IGNORECASE,
)
PSEUDO_TOOL_PARAMETER_RE = re.compile(
    r"<parameter=([A-Za-z_][A-Za-z0-9_]*)>\s*(.*?)\s*</parameter>",
    re.DOTALL,
)
IMAGE_FROM_TEXT_RE = re.compile(r"Image from `([^`]+)`:")


@dataclass(frozen=True, slots=True)
class RecoveredToolCall:
    source: str
    tool_name: str
    tool_call: dict[str, Any]


def _redact_data_url(url: str) -> str:
    if not url.startswith("data:"):
        return url
    header, separator, payload = url.partition(",")
    if separator:
        return f"{header},<redacted {len(payload)} chars>"
    return f"data:<redacted {len(url)} chars>"


def _redact_multimodal_part(part: Any) -> Any:
    if not isinstance(part, dict):
        return part
    redacted = dict(part)
    image_url = redacted.get("image_url")
    if isinstance(image_url, dict):
        redacted_image_url = dict(image_url)
        url = redacted_image_url.get("url")
        if isinstance(url, str):
            redacted_image_url["url"] = _redact_data_url(url)
        redacted["image_url"] = redacted_image_url
    video_url = redacted.get("video_url")
    if isinstance(video_url, dict):
        redacted_video_url = dict(video_url)
        url = redacted_video_url.get("url")
        if isinstance(url, str):
            redacted_video_url["url"] = _redact_data_url(url)
        redacted["video_url"] = redacted_video_url
    return redacted


def _render_message_content(content: Any) -> str | None:
    if content in (None, ""):
        return None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return json.dumps([_redact_multimodal_part(part) for part in content], ensure_ascii=False)
    return json.dumps(content, ensure_ascii=False)


def _extract_schemas_list(global_data_profile: str) -> list[dict[str, Any]] | None:
    if not global_data_profile.strip():
        return None
    try:
        profile = json.loads(global_data_profile)
    except json.JSONDecodeError:
        return None
    schemas = profile.get("schemas") if isinstance(profile, dict) else None
    if not isinstance(schemas, list) or not schemas:
        return None
    return schemas


def _extract_knowledge_documents(
    full_schemas: list[dict[str, Any]],
) -> list[dict[str, Any]] | None:
    """Extract document schemas with full text content (knowledge.md files)."""
    docs = [
        s
        for s in full_schemas
        if isinstance(s, dict)
        and s.get("kind") == "document"
        and isinstance(s.get("content"), str)
        and s["content"].strip()
    ]
    return docs if docs else None


def _extract_knowledge_documents_from_profile(profile: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Extract knowledge documents from either full or lightweight catalog profiles."""
    knowledge_documents = profile.get("knowledge_documents")
    if isinstance(knowledge_documents, list):
        docs = [
            doc
            for doc in knowledge_documents
            if isinstance(doc, dict)
            and isinstance(doc.get("content"), str)
            and doc["content"].strip()
        ]
        if docs:
            return docs

    schemas = profile.get("schemas")
    if isinstance(schemas, list) and schemas:
        return _extract_knowledge_documents(schemas)
    return None


def _extract_video_summaries(task: Any) -> list[dict[str, Any]] | None:
    """Extract video summary content from the task's context view assets.

    Returns a list of dicts with a ``"content"`` key (same shape as
    ``_extract_knowledge_documents``), or None if no video summaries exist.
    """
    from data_agent_baseline.benchmark.context_view import iter_context_file_assets

    try:
        assets = [
            a for a in iter_context_file_assets(task)
            if getattr(a, "action", None) == "video_summary"
        ]
    except Exception:
        return None

    if not assets:
        return None

    result: list[dict[str, Any]] = []
    for asset in assets:
        try:
            text = asset.physical_path.read_text(encoding="utf-8", errors="replace")
            if text.strip():
                result.append({"content": text.strip()})
        except OSError:
            continue

    return result if result else None


def _coerce_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _submitted_answer_fingerprint(answer: dict[str, Any]) -> str:
    payload = json.dumps(answer, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _canonical_fingerprint(value: Any) -> str:
    """Fingerprint a runtime payload for binding, never for validator caching."""
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _submitted_answer_row_count(answer: dict[str, Any]) -> int:
    rows = answer.get("rows")
    return len(rows) if isinstance(rows, list) else 0


def _submitted_answer_scalar_preview(answer: dict[str, Any] | None) -> tuple[bool, Any]:
    if not isinstance(answer, dict):
        return False, None
    columns = answer.get("columns")
    rows = answer.get("rows")
    if not isinstance(columns, list) or len(columns) != 1:
        return False, None
    if not isinstance(rows, list) or len(rows) != 1:
        return False, None
    row = rows[0]
    if not isinstance(row, list) or len(row) != 1:
        return False, None
    value = row[0]
    if value is None or isinstance(value, bool | int | float | str):
        return True, value
    return False, None


def _cell_kind(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int | float):
        return "number"
    if isinstance(value, str):
        return "empty_string" if value == "" else "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _stable_sample_indices(item_count: int, sample_size: int, fingerprint: str) -> list[int]:
    if item_count <= 0 or sample_size <= 0:
        return []
    sample_size = min(sample_size, item_count)
    seed = int(fingerprint[:16], 16) if fingerprint else 0
    remaining = list(range(item_count))
    selected: list[int] = []
    # Deterministic Fisher-Yates prefix without importing random.
    for offset in range(sample_size):
        pick = offset + (seed + offset * 1103515245) % (item_count - offset)
        remaining[offset], remaining[pick] = remaining[pick], remaining[offset]
        selected.append(remaining[offset])
    return sorted(selected)


def _distinct_value_examples(
    values: list[Any],
    *,
    max_examples: int,
    max_str_tokens: int,
    max_list_items: int,
    fingerprint: str,
) -> tuple[list[Any], bool]:
    unique_values: list[Any] = []
    seen: set[str] = set()
    for value in values:
        try:
            key = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        except TypeError:
            key = repr(value)
        if key in seen:
            continue
        seen.add(key)
        unique_values.append(value)

    if len(unique_values) <= max_examples:
        selected_indices = list(range(len(unique_values)))
    else:
        head_count = min(3, max_examples)
        tail_count = min(3, max_examples - head_count)
        middle_count = max_examples - head_count - tail_count
        head_indices = list(range(head_count))
        tail_indices = list(range(len(unique_values) - tail_count, len(unique_values)))
        middle_candidates = [
            index
            for index in range(head_count, len(unique_values) - tail_count)
            if index >= 0
        ]
        sampled_middle = [
            middle_candidates[index]
            for index in _stable_sample_indices(
                len(middle_candidates),
                middle_count,
                fingerprint,
            )
        ]
        selected_indices = sorted(set(head_indices + sampled_middle + tail_indices))

    examples: list[Any] = []
    values_truncated = False
    for index in selected_indices[:max_examples]:
        value = unique_values[index]
        bounded_value = truncate_content(
            value,
            max_str_tokens=max_str_tokens,
            max_list_items=max_list_items,
        )
        values_truncated = values_truncated or bounded_value != value
        examples.append(bounded_value)

    examples_truncated = len(unique_values) > len(examples) or values_truncated
    return examples, examples_truncated


def _build_answer_validator_context(
    answer: dict[str, Any],
    *,
    max_str_tokens: int,
    max_list_items: int,
    distinct_examples_per_column: int = ANSWER_VALIDATOR_DISTINCT_EXAMPLES_PER_COLUMN,
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    """Build bounded answer context without row samples."""
    legacy_preview = truncate_answer_content(
        answer,
        max_str_tokens=max_str_tokens,
        max_list_items=min(max_list_items, ANSWER_VALIDATOR_MAX_PREVIEW_ROWS),
    )
    legacy_preview_truncated = legacy_preview != answer

    columns = answer.get("columns")
    rows = answer.get("rows")
    columns_list = list(columns) if isinstance(columns, list) else []
    rows_list = list(rows) if isinstance(rows, list) else []
    row_count = len(rows_list)
    column_count = len(columns_list)
    fingerprint = _submitted_answer_fingerprint(answer)

    row_length_counts: dict[str, int] = {}
    seen_rows: set[str] = set()
    duplicate_row_count = 0
    for row in rows_list:
        if not isinstance(row, list):
            row_length = "invalid"
        else:
            row_length = str(len(row))
        row_length_counts[row_length] = row_length_counts.get(row_length, 0) + 1
        try:
            row_key = json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)
        except TypeError:
            row_key = repr(row)
        if row_key in seen_rows:
            duplicate_row_count += 1
        else:
            seen_rows.add(row_key)

    column_profiles: list[dict[str, Any]] = []
    examples_truncated = False
    for col_index, column in enumerate(columns_list):
        value_types: set[str] = set()
        column_values: list[Any] = []
        for row in rows_list:
            if not isinstance(row, list) or col_index >= len(row):
                continue
            else:
                value = row[col_index]
                if value is None:
                    continue
                value_types.add(_cell_kind(value))
                column_values.append(value)
        distinct_examples, profile_examples_truncated = _distinct_value_examples(
            column_values,
            max_examples=max(0, distinct_examples_per_column),
            max_str_tokens=max_str_tokens,
            max_list_items=max_list_items,
            fingerprint=f"{fingerprint}:{col_index}",
        )
        examples_truncated = examples_truncated or profile_examples_truncated
        column_profiles.append(
            {
                "name": column,
                "index": col_index,
                "value_types": sorted(value_types),
                "distinct_value_examples": distinct_examples,
                "examples_truncated": profile_examples_truncated,
            }
        )

    structure_overview = {
        "columns": columns_list,
        "row_count": row_count,
        "column_count": column_count,
        "row_length_counts": row_length_counts,
        "duplicate_row_count": duplicate_row_count,
        "column_profiles": column_profiles,
    }

    context_bounded = examples_truncated or legacy_preview_truncated
    return legacy_preview, structure_overview, context_bounded


def _submission_context_for_validator(answer_submission: Any) -> dict[str, Any] | None:
    if not isinstance(answer_submission, dict):
        return None
    if answer_submission.get("submission_tool") != "submit_tool_result":
        return None
    return dict(answer_submission)


def _submission_field_context_trace_preview(
    submission_field_context: dict[str, Any],
    *,
    max_fields_per_table: int = 40,
) -> dict[str, Any]:
    """Build a compact trace-facing preview of source-field context."""
    field_universe_preview: list[dict[str, Any]] = []
    for universe in submission_field_context.get("field_universe", []):
        if not isinstance(universe, dict):
            continue
        fields = universe.get("fields", [])
        field_names = [
            str(field.get("name"))
            for field in fields
            if isinstance(field, dict) and field.get("name") is not None
        ]
        entry = {
            "table": universe.get("table"),
            "alias": universe.get("alias"),
            "kind": universe.get("kind"),
            "field_count": len(field_names),
            "fields": field_names[:max_fields_per_table],
            "fields_truncated": len(field_names) > max_fields_per_table,
            **({"joins": universe.get("joins")} if universe.get("joins") else {}),
        }
        for key in ("source_path", "target_table", "registered_table"):
            if universe.get(key) is not None:
                entry[key] = universe.get(key)
        field_universe_preview.append(entry)

    return {
        "status": submission_field_context.get("status"),
        "source_tool": submission_field_context.get("source_tool"),
        "source_tables": submission_field_context.get("source_tables", []),
        "field_universe": field_universe_preview,
        "output_lineage": submission_field_context.get("output_lineage", []),
        "join_edges": submission_field_context.get("join_edges", []),
        "filter_fields": submission_field_context.get("filter_fields", []),
        "warnings": submission_field_context.get("warnings", []),
    }


def _contains_truncation_marker(value: Any) -> bool:
    if isinstance(value, str):
        return TRUNCATION_SUFFIX in value
    if isinstance(value, list):
        return any(_contains_truncation_marker(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_truncation_marker(item) for item in value.values())
    return False


def _evidence_source(args: dict[str, Any], tool_name: str) -> dict[str, str]:
    path = args.get("path") or args.get("knowledge_path")
    table = args.get("target_table") or args.get("table")
    if isinstance(path, str) and path:
        kind = "image" if tool_name in {"read_context_image", "record_visual_evidence"} else "document"
        return {"kind": kind, "path": path}
    if isinstance(table, str) and table:
        return {"kind": "table", "name": table}
    return {"kind": "tool_result", "name": tool_name}


def _evidence_capabilities(tool_name: str, args: dict[str, Any] | None = None) -> list[str]:
    mapping = {
        "get_table_profile": ["table_schema"],
        "get_field_profile": ["field_definition"],
        "get_table_relationships": ["candidate_join_only"],
        "get_column_distinct_values": ["field_value_observation"],
        "execute_probe_query": ["source_row_observation"],
        "execute_python": ["programmatic_observation"],
        "search_doc": ["document_match_locator"],
        "read_doc": ["document_text_fact"],
        "lookup_doc_outline": ["document_navigation"],
        "inspect_doc_structure": ["document_schema", "structured_field_coverage"],
        "extract_structured_doc": ["structured_document_rows"],
        "read_context_image": ["visual_fact"],
        "record_visual_evidence": ["visual_fact_receipt"],
        "search_semantic_catalog": ["source_discovery"],
        "list_context": ["asset_discovery"],
    }
    caps = mapping.get(tool_name, ["tool_observation"])
    if tool_name == "read_doc" and isinstance(args, dict):
        path = str(args.get("path", "")).replace("\\", "/").lower()
        if path.endswith("_video_summary.md"):
            caps = ["video_narrative_context"]
    return caps


def _build_supporting_source_evidence(
    steps: list[dict[str, Any]],
    submission_context: dict[str, Any] | None,
    *,
    max_items: int = _DEFAULT_EVIDENCE_MAX_ITEMS,
    max_str_tokens: int = _DEFAULT_EVIDENCE_MAX_STR_TOKENS,
    max_list_items: int = _DEFAULT_EVIDENCE_MAX_LIST_ITEMS,
) -> dict[str, Any]:
    """Create a bounded, tool-only evidence capsule for process validation."""
    selected: list[dict[str, Any]] = []
    omitted: list[dict[str, str]] = []
    seen: set[str] = set()

    for step in reversed(steps):
        if step.get("node") != "tool":
            continue
        calls = step.get("tool_calls")
        results = step.get("tool_results")
        if not isinstance(calls, list) or not isinstance(results, list):
            continue
        for call, result in reversed(list(zip(calls, results))):
            if not isinstance(call, dict) or not isinstance(result, dict):
                continue
            tool_name = str(call.get("name") or result.get("tool") or "")
            tool_call_id = str(call.get("id") or tool_name)
            args = _coerce_dict(call.get("args"))
            source = _evidence_source(args, tool_name)
            source_label = str(source.get("path") or source.get("name") or tool_name)
            if result.get("ok") is not True:
                omitted.append({"tool": tool_name, "source": source_label, "reason": "not_successful"})
                continue
            content = result.get("content")
            if _contains_truncation_marker(content):
                omitted.append({"tool": tool_name, "source": source_label, "reason": "truncated_result"})
                continue
            identity = _canonical_fingerprint({"tool": tool_name, "args": args, "source": source})
            if identity in seen:
                continue
            seen.add(identity)
            if len(selected) >= max_items:
                omitted.append({"tool": tool_name, "source": source_label, "reason": "evidence_budget"})
                continue
            locator = dict(source)
            if tool_name == "read_context_image" and isinstance(args.get("path"), str):
                locator = {"frame_path": args["path"]}
            excerpt = truncate_content(
                content,
                max_str_tokens=max_str_tokens,
                max_list_items=max_list_items,
            )
            selected.append(
                {
                    "id": f"step_{step.get('step_index')}:call_{tool_call_id}",
                    "trace_step_index": step.get("step_index"),
                    "tool_call_id": tool_call_id,
                    "tool": tool_name,
                    "capabilities": _evidence_capabilities(tool_name, args),
                    "source": source,
                    "request": args,
                    "observation": {
                        "status": "complete",
                        "truncated": False,
                        "locator": locator,
                        "evidence_excerpt": excerpt,
                    },
                }
            )

    selected.reverse()
    omitted.reverse()
    return {
        "schema_version": 2,
        "evidence_items": selected,
        "omitted_or_unusable": omitted[:max_items],
    }


def _summarize_validation_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for entry in history:
        summary.append(
            {
                "answer_fingerprint": entry.get("answer_fingerprint"),
                "answer_columns": entry.get("answer_columns"),
                "answer_row_count": entry.get("answer_row_count"),
                "valid": entry.get("valid"),
                "rationale": entry.get("rationale"),
                "issues": entry.get("issues", []),
                "validator_error": entry.get("validator_error"),
            }
        )
    return summary


def _validation_history_entry(
    *,
    answer_fingerprint: str,
    answer: dict[str, Any],
    validation_result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "answer_fingerprint": answer_fingerprint,
        "answer_columns": answer.get("columns"),
        "answer_row_count": _submitted_answer_row_count(answer),
        "valid": bool(validation_result.get("valid", True)),
        "rationale": validation_result.get("rationale"),
        "issues": list(validation_result.get("issues", [])),
        "validator_error": validation_result.get("validator_error"),
        "raw_response": validation_result.get("raw_response"),
    }


def _normalize_tool_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for tool_call in tool_calls:
        normalized.append(
            {
                "id": tool_call.get("id"),
                "name": tool_call.get("name"),
                "args": _coerce_dict(tool_call.get("args")),
            }
        )
    return normalized


def _summarize_invalid_tool_calls(
    invalid_tool_calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for call in invalid_tool_calls:
        summaries.append(
            {
                "id": call.get("id"),
                "name": call.get("name"),
                "args": _preview_text(str(call.get("args", "")), limit=800),
                "error": _preview_text(str(call.get("error", "")), limit=1200),
            }
        )
    return summaries


def _build_invalid_tool_call_repair_prompt(
    invalid_tool_call_errors: list[dict[str, Any]] | None,
) -> str:
    if not invalid_tool_call_errors:
        return INVALID_TOOL_CALL_REPAIR_PROMPT

    details = json.dumps(invalid_tool_call_errors, ensure_ascii=False, indent=2)
    return (
        f"{INVALID_TOOL_CALL_REPAIR_PROMPT}\n\n"
        "The discarded malformed tool call details were:\n"
        f"{details}\n\n"
        "Repair the exact intended call. If you are calling `submit_tool_result` "
        "with `execute_python`, prefer a short single-print program such as: "
        "{\"tool_name\":\"execute_python\",\"tool_args\":{\"code\":\"import json\\n"
        "print(json.dumps({\\\"columns\\\":[\\\"column\\\"],\\\"rows\\\":[[\\\"value\\\"]]}))\"}}"
    )


def _preview_text(text: str | None, *, limit: int = 180) -> str | None:
    if text in (None, ""):
        return None
    return text if len(text) <= limit else f"{text[:limit]}..."


def _summarize_message(message: BaseMessage) -> dict[str, Any]:
    rendered_content = _render_message_content(message.content)
    payload = {
        "type": message.type,
        "name": getattr(message, "name", None),
        "content_preview": _preview_text(rendered_content),
        "content_length": 0 if rendered_content is None else len(rendered_content),
    }
    if isinstance(message.content, list):
        payload["content_part_types"] = [
            part.get("type")
            for part in message.content
            if isinstance(part, dict) and part.get("type") is not None
        ]
        payload["video_part_count"] = sum(
            1
            for part in message.content
            if isinstance(part, dict) and part.get("type") == "video_url"
        )
        payload["image_part_count"] = sum(
            1
            for part in message.content
            if isinstance(part, dict) and part.get("type") == "image_url"
        )
    if isinstance(message, AIMessage):
        payload["tool_call_names"] = [
            call["name"] for call in _normalize_tool_calls(message.tool_calls)
        ]
    if isinstance(message, ToolMessage):
        payload["tool_call_id"] = message.tool_call_id
        payload["status"] = getattr(message, "status", None)
    return payload


def _compressed_image_message_summaries(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        rendered_content = _render_message_content(message.content)
        if not isinstance(rendered_content, str):
            continue
        if not rendered_content.startswith("Compressed image observation:"):
            continue
        summaries.append(
            {
                "message_index": index,
                "type": message.type,
                "content": rendered_content,
                "content_length": len(rendered_content),
            }
        )
    return summaries


def _summarize_model_request(
    *,
    messages: list[BaseMessage],
    tools: list[BaseTool],
    tool_choice: str,
    parallel_tool_calls: bool,
) -> dict[str, Any]:
    last_message = messages[-1] if messages else None
    payload = {
        "message_count": len(messages),
        "last_message": None if last_message is None else _summarize_message(last_message),
        "tool_names": [tool.name for tool in tools],
        "tool_choice": tool_choice,
        "parallel_tool_calls": parallel_tool_calls,
    }
    compressed_image_messages = _compressed_image_message_summaries(messages)
    if compressed_image_messages:
        payload["compressed_image_messages"] = compressed_image_messages
    return payload


def _summarize_ai_message(ai_message: AIMessage) -> dict[str, Any]:
    response_metadata = _coerce_dict(getattr(ai_message, "response_metadata", None))
    usage_metadata = _coerce_dict(getattr(ai_message, "usage_metadata", None))
    token_usage = _coerce_dict(response_metadata.get("token_usage"))
    completion_token_details = _coerce_dict(token_usage.get("completion_tokens_details"))
    output_token_details = _coerce_dict(usage_metadata.get("output_token_details"))
    rendered_content = _render_message_content(ai_message.content)
    reasoning_content = _render_message_content(
        ai_message.additional_kwargs.get("reasoning_content")
    )
    payload = {
        "response_id": response_metadata.get("id") or ai_message.id,
        "model_name": response_metadata.get("model_name"),
        "finish_reason": response_metadata.get("finish_reason"),
        "tool_call_names": [call["name"] for call in _normalize_tool_calls(ai_message.tool_calls)],
        "content_preview": _preview_text(rendered_content),
        "content_length": 0 if rendered_content is None else len(rendered_content),
        "input_tokens": usage_metadata.get("input_tokens", token_usage.get("prompt_tokens")),
        "output_tokens": usage_metadata.get("output_tokens", token_usage.get("completion_tokens")),
        "reasoning_tokens": output_token_details.get(
            "reasoning",
            completion_token_details.get("reasoning_tokens"),
        ),
    }
    if reasoning_content is not None:
        payload["reasoning_content"] = reasoning_content
        payload["reasoning_content_length"] = len(reasoning_content)
    return payload


def _ai_reasoning_content(ai_message: AIMessage) -> str | None:
    return _render_message_content(ai_message.additional_kwargs.get("reasoning_content"))


def _strip_pseudo_tool_call_blocks(text: str) -> str:
    # Strip from outer to inner: <tool_call> wrappers first, then bare <function>
    # blocks, then leftover <parameter> fragments.  The parser in
    # _parse_pseudo_tool_call already handles all three nesting levels, so the
    # cleaner must mirror that coverage.
    cleaned = PSEUDO_TOOL_CALL_BLOCK_RE.sub("", text)
    cleaned = PSEUDO_FUNCTION_TAG_RE.sub("", cleaned)
    cleaned = PSEUDO_TOOL_PARAMETER_RE.sub("", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _reasoning_as_history_content(ai_message: AIMessage) -> tuple[str, bool]:
    """Expose useful assistant reasoning in history after removing pseudo tool-call text."""
    visible_content = _render_message_content(ai_message.content)
    if visible_content:
        return _strip_pseudo_tool_call_blocks(visible_content), False
    reasoning_content = _ai_reasoning_content(ai_message)
    if reasoning_content:
        return _strip_pseudo_tool_call_blocks(reasoning_content), True
    return "", False


def _with_clean_reasoning_history_content(
    ai_message: AIMessage,
    *,
    strip_reasoning: bool = False,
) -> AIMessage:
    if strip_reasoning:
        visible_content = _render_message_content(ai_message.content)
        content = _strip_pseudo_tool_call_blocks(visible_content) if visible_content else ""
        additional_kwargs = dict(ai_message.additional_kwargs)
        additional_kwargs.pop("reasoning_content", None)
        additional_kwargs.pop(REASONING_HISTORY_DERIVED_CONTENT_KEY, None)
        update = {"content": content, "additional_kwargs": additional_kwargs}
    else:
        content, derived_from_reasoning = _reasoning_as_history_content(ai_message)
        additional_kwargs = dict(ai_message.additional_kwargs)
        if derived_from_reasoning:
            additional_kwargs[REASONING_HISTORY_DERIVED_CONTENT_KEY] = True
        else:
            additional_kwargs.pop(REASONING_HISTORY_DERIVED_CONTENT_KEY, None)
        update = {"content": content, "additional_kwargs": additional_kwargs}
    if hasattr(ai_message, "model_copy"):
        return ai_message.model_copy(update=update)
    return ai_message.copy(update=update)


def _strip_reasoning_from_history_message(ai_message: AIMessage) -> AIMessage:
    additional_kwargs = dict(ai_message.additional_kwargs)
    derived_content = bool(additional_kwargs.pop(REASONING_HISTORY_DERIVED_CONTENT_KEY, False))
    additional_kwargs.pop("reasoning_content", None)
    update: dict[str, Any] = {"additional_kwargs": additional_kwargs}
    if derived_content:
        update["content"] = ""
    if hasattr(ai_message, "model_copy"):
        return ai_message.model_copy(update=update)
    return ai_message.copy(update=update)


def _remove_reasoning_history_marker(ai_message: AIMessage) -> AIMessage:
    if REASONING_HISTORY_DERIVED_CONTENT_KEY not in ai_message.additional_kwargs:
        return ai_message
    additional_kwargs = dict(ai_message.additional_kwargs)
    additional_kwargs.pop(REASONING_HISTORY_DERIVED_CONTENT_KEY, None)
    update = {"additional_kwargs": additional_kwargs}
    if hasattr(ai_message, "model_copy"):
        return ai_message.model_copy(update=update)
    return ai_message.copy(update=update)


def _message_with_content(message: BaseMessage, content: Any) -> BaseMessage:
    update = {"content": content}
    if hasattr(message, "model_copy"):
        return message.model_copy(update=update)
    return message.copy(update=update)


def _image_message_parts(message: BaseMessage) -> list[dict[str, Any]] | None:
    if not isinstance(message, HumanMessage) or not isinstance(message.content, list):
        return None
    image_parts = [
        part
        for part in message.content
        if isinstance(part, dict) and part.get("type") == "image_url"
    ]
    return image_parts or None


def _image_paths_from_message(message: HumanMessage) -> list[str]:
    paths: list[str] = []
    if not isinstance(message.content, list):
        return paths
    for part in message.content:
        if not isinstance(part, dict) or part.get("type") != "text":
            continue
        text = part.get("text")
        if not isinstance(text, str):
            continue
        paths.extend(match.group(1) for match in IMAGE_FROM_TEXT_RE.finditer(text))
    return paths


def _image_details_from_parts(image_parts: list[dict[str, Any]]) -> list[str]:
    details: list[str] = []
    for part in image_parts:
        image_url = part.get("image_url")
        detail = "auto"
        if isinstance(image_url, dict):
            raw_detail = image_url.get("detail")
            if raw_detail not in (None, ""):
                detail = str(raw_detail)
        details.append(detail)
    return details


def _truncate_text_by_chars(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def _assistant_note_after_image(
    messages: list[BaseMessage],
    image_message_index: int,
    *,
    max_chars: int,
) -> str | None:
    for message in messages[image_message_index + 1 :]:
        if not isinstance(message, AIMessage):
            continue
        rendered = _render_message_content(message.content) or ""
        note = _strip_pseudo_tool_call_blocks(rendered)
        return _truncate_text_by_chars(note, max_chars)
    return None


def _compressed_image_message_content(
    message: HumanMessage,
    image_parts: list[dict[str, Any]],
    *,
    assistant_note: str,
) -> str:
    paths = _image_paths_from_message(message)
    details = _image_details_from_parts(image_parts)
    image_count = max(len(paths), len(details), len(image_parts))
    lines = [
        "Compressed image observation:",
        "Images:",
    ]
    for index in range(image_count):
        path = paths[index] if index < len(paths) else f"<image {index + 1}>"
        detail = details[index] if index < len(details) else "auto"
        lines.append(f"- path: {path}")
        lines.append(f"  detail: {detail}")

    note = assistant_note.strip() or (
        "No assistant observation text was recorded; call read_context_image again if needed."
    )
    lines.extend(
        [
            "",
            "Assistant note after viewing:",
            note,
        ]
    )
    return "\n".join(lines)


def _compress_used_image_messages(
    messages: list[BaseMessage],
    *,
    enabled: bool,
    compressed_image_note_chars: int,
) -> list[BaseMessage]:
    if not enabled:
        return messages

    compressed: list[BaseMessage] = []
    for index, message in enumerate(messages):
        image_parts = _image_message_parts(message)
        if image_parts is None or not isinstance(message, HumanMessage):
            compressed.append(message)
            continue

        assistant_note = _assistant_note_after_image(
            messages,
            index,
            max_chars=compressed_image_note_chars,
        )
        if assistant_note is None:
            compressed.append(message)
            continue

        compressed.append(
            _message_with_content(
                message,
                _compressed_image_message_content(
                    message,
                    image_parts,
                    assistant_note=assistant_note,
                ),
            )
        )

    return compressed


def _prepare_messages_for_model(
    messages: list[BaseMessage],
    *,
    strip_reasoning_history: bool,
    reasoning_history_limit: int | None,
    compress_used_image_messages: bool,
    compressed_image_note_chars: int,
) -> list[BaseMessage]:
    keep_remaining: int | None = None
    if not strip_reasoning_history:
        keep_remaining = reasoning_history_limit

    prepared: list[BaseMessage] = []
    for message in reversed(messages):
        if not isinstance(message, AIMessage):
            prepared.append(message)
            continue

        reasoning_content = _ai_reasoning_content(message)
        should_strip = strip_reasoning_history
        if not should_strip and reasoning_content is not None:
            if keep_remaining is None:
                should_strip = False
            elif keep_remaining > 0:
                keep_remaining -= 1
                should_strip = False
            else:
                should_strip = True

        prepared.append(
            _strip_reasoning_from_history_message(message)
            if should_strip
            else _remove_reasoning_history_marker(message)
        )

    prepared.reverse()
    return _compress_used_image_messages(
        prepared,
        enabled=compress_used_image_messages,
        compressed_image_note_chars=compressed_image_note_chars,
    )


def _schema_field_annotations(tool_schemas: dict[str, type[Any]], tool_name: str) -> dict[str, Any]:
    schema = tool_schemas.get(tool_name)
    if schema is None:
        return {}
    fields = getattr(schema, "model_fields", None) or getattr(schema, "__fields__", {})
    annotations: dict[str, Any] = {}
    for name, model_field in fields.items():
        annotations[name] = getattr(model_field, "annotation", None) or getattr(
            model_field, "outer_type_", None
        )
    return annotations


def _schema_has_required_fields(tool_schema: type[Any]) -> bool:
    """Check whether the tool schema has any required (no-default) fields."""
    fields = getattr(tool_schema, "model_fields", None) or getattr(tool_schema, "__fields__", {})
    return any(
        getattr(field_info, "is_required", lambda: False)() for field_info in fields.values()
    )


def _coerce_pseudo_tool_value(value: str, annotation: Any) -> Any:
    cleaned = value.strip()
    origin = get_origin(annotation)
    target = origin or annotation
    if target is bool:
        lowered = cleaned.lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
        return cleaned
    if target is int:
        try:
            return int(cleaned)
        except ValueError:
            return cleaned
    if target is float:
        try:
            return float(cleaned)
        except ValueError:
            return cleaned
    if target in {list, dict} or cleaned.startswith(("[", "{")):
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return cleaned
    if annotation is not None and get_args(annotation):
        for nested_annotation in get_args(annotation):
            coerced = _coerce_pseudo_tool_value(cleaned, nested_annotation)
            if not isinstance(coerced, str) or nested_annotation is str:
                return coerced
    return cleaned


def _parse_pseudo_tool_call(
    *,
    text: str,
    source: str,
    available_tool_names: set[str],
    tool_schemas: dict[str, type[Any]],
) -> RecoveredToolCall | None:
    match = PSEUDO_TOOL_CALL_RE.search(text)
    if match is None:
        match = PSEUDO_FUNCTION_TAG_RE.search(text)
    if match is None:
        return None
    tool_name = match.group(1)
    if tool_name not in available_tool_names:
        return None
    parameter_block = match.group(2)
    field_annotations = _schema_field_annotations(tool_schemas, tool_name)
    args: dict[str, Any] = {}
    for parameter_match in PSEUDO_TOOL_PARAMETER_RE.finditer(parameter_block):
        name = parameter_match.group(1)
        if field_annotations and name not in field_annotations:
            continue
        args[name] = _coerce_pseudo_tool_value(
            parameter_match.group(2), field_annotations.get(name)
        )
    if not args:
        # Allow empty args only when the tool schema has zero required fields.
        tool_schema = tool_schemas.get(tool_name)
        if tool_schema is not None and _schema_has_required_fields(tool_schema):
            return None
    return RecoveredToolCall(
        source=source,
        tool_name=tool_name,
        tool_call={
            "name": tool_name,
            "args": args,
            "id": f"recovered_{uuid4().hex[:24]}",
            "type": "tool_call",
        },
    )


def _recover_pseudo_tool_call(
    ai_message: AIMessage,
    *,
    available_tool_names: set[str],
    tool_schemas: dict[str, type[Any]],
) -> tuple[AIMessage, RecoveredToolCall | None]:
    if ai_message.tool_calls:
        return ai_message, None
    candidate_sources = [
        ("reasoning_content", _ai_reasoning_content(ai_message)),
        ("content", _render_message_content(ai_message.content)),
    ]
    for source, text in candidate_sources:
        if not text:
            continue
        recovered = _parse_pseudo_tool_call(
            text=text,
            source=source,
            available_tool_names=available_tool_names,
            tool_schemas=tool_schemas,
        )
        if recovered is None:
            continue
        if hasattr(ai_message, "model_copy"):
            return ai_message.model_copy(update={"tool_calls": [recovered.tool_call]}), recovered
        return ai_message.copy(update={"tool_calls": [recovered.tool_call]}), recovered
    return ai_message, None


def _discard_invalid_tool_calls(ai_message: AIMessage) -> AIMessage:
    """Remove malformed provider tool calls before replaying message history.

    Providers may return a response with ``finish_reason='tool_calls'`` while
    LangChain places an unparsable call in ``invalid_tool_calls``.  Sending that
    message back on the next request makes DashScope reject the *history* with
    a 400 before the model can regenerate a valid call.
    """
    if not ai_message.invalid_tool_calls:
        return ai_message

    additional_kwargs = dict(ai_message.additional_kwargs)
    # Provider adapters can retain the raw malformed payload here even after
    # LangChain has moved it to ``invalid_tool_calls``.
    additional_kwargs.pop("tool_calls", None)
    additional_kwargs.pop("function_call", None)

    return ai_message.model_copy(
        update={
            "tool_calls": [],
            "invalid_tool_calls": [],
            "additional_kwargs": additional_kwargs,
        }
    )


def _is_non_action_stop(ai_message: AIMessage) -> bool:
    return not ai_message.tool_calls


def _build_context_table_summary(context_dir: Path) -> str:
    """Build a concise table-name reference for inject into execute_probe_query description.

    Scans the context directory for data files and returns a formatted list of
    available table names the model must use in SQL queries.
    """
    lines: list[str] = []
    if not context_dir.is_dir():
        return ""
    for entry in sorted(context_dir.rglob("*"), key=lambda p: (p.is_dir(), p.suffix, p.name)):
        if entry.is_dir():
            continue
        suffix = entry.suffix.lower()
        rel_path = entry.relative_to(context_dir).as_posix()
        stem = entry.stem
        if suffix == ".csv":
            lines.append(f"  - {stem} (CSV: {rel_path})")
        elif suffix == ".json":
            lines.append(f"  - {stem} (JSON: {rel_path})")
        elif suffix in (".db", ".sqlite", ".sqlite3"):
            try:
                uri = f"file:{entry.resolve().as_posix()}?mode=ro"
                conn = sqlite3.connect(uri, uri=True)
                try:
                    rows = conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
                    ).fetchall()
                    for (table_name,) in rows:
                        lines.append(f"  - {table_name} (SQLite: {rel_path})")
                finally:
                    conn.close()
            except Exception:
                continue
    if not lines:
        return ""
    return "Available tables from the data catalog:\n" + "\n".join(lines)


class LangGraphAgent:
    def __init__(
        self,
        *,
        model: BaseChatModel | Any,
        tools: ToolRegistry,
        config: LangGraphAgentConfig | None = None,
        trace_callback: TraceCallback | None = None,
        tool_gate: Any | None = None,
    ) -> None:
        self.model = model
        self.tools = tools
        self.config = config or LangGraphAgentConfig()
        self.trace_callback = trace_callback
        self.tool_gate = tool_gate

    def run(self, task: PublicTask) -> AgentRunResult:
        python_workspace = TaskContextWorkspace(
            task.context_dir,
            context_view=task.assets.context_view,
        )
        trace_dir = None
        trace_callback_owner = getattr(self.trace_callback, "__self__", None)
        trace_path = getattr(trace_callback_owner, "trace_path", None)
        if trace_path is not None:
            trace_dir = trace_path.parent
        runtime_context = ToolRuntimeContext(
            task=task,
            python_workspace=python_workspace,
            budget=self.config.data_inspector.sample_budget,
            semantic_view_config=self.config.data_inspector.semantic_views,
            model=self.model,
            trace_dir=trace_dir,
            tool_gate=self.tool_gate,
        )
        bound_tools = self.tools.bind(runtime_context)
        langchain_tools = bound_tools.langchain_tools()
        available_tool_names = {tool.name for tool in langchain_tools}
        tool_schemas = {tool.name: getattr(tool, "args_schema", None) for tool in langchain_tools}
        final_submission_tool_names = {"submit_tool_result"}
        final_submission_tools = [
            tool for tool in langchain_tools if tool.name in final_submission_tool_names
        ]
        final_submission_tool_schemas = {
            tool.name: getattr(tool, "args_schema", None) for tool in final_submission_tools
        }
        tool_choice = "auto"
        parallel_tool_calls = False
        model_with_tools = self.model.bind_tools(
            langchain_tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
        )
        # thinking 模式下 DashScope 拒绝 object/required 形式的 tool_choice；
        # force_answer 只绑定了 submit_tool_result，配合强提示用 auto 即可可靠触发提交。
        forced_submission_tool_choice = "auto"

        def trace_timestamp() -> str:
            return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        def answer_payload(answer: Any) -> dict[str, Any] | None:
            if answer is None:
                return None
            if hasattr(answer, "to_dict"):
                return answer.to_dict()
            return dict(answer) if isinstance(answer, dict) else None

        def emit_trace(
            state: AgentGraphState,
            update: AgentGraphState | None = None,
            *,
            partial: bool = True,
        ) -> None:
            if self.trace_callback is None:
                return
            update = update or {}
            answer = update.get("answer", state.get("answer"))
            failure_reason = update.get("failure_reason", state.get("failure_reason"))
            payload = {
                "task_id": task.task_id,
                "answer": answer_payload(answer),
                "steps": [*list(state.get("steps", [])), *list(update.get("steps", []))],
                "failure_reason": failure_reason,
                "succeeded": answer is not None and failure_reason is None,
                "inspector": update.get("inspector", state.get("inspector")),
                "semantic_ledger": update.get("semantic_ledger", state.get("semantic_ledger")),
                "partial": partial,
                "started_at": state.get("started_at"),
                "updated_at": trace_timestamp(),
            }
            global_data_profile = update.get(
                "global_data_profile", state.get("global_data_profile")
            )
            if isinstance(global_data_profile, str) and global_data_profile.strip():
                payload["global_data_profile"] = global_data_profile
            self.trace_callback(payload)

        def emit_in_progress_trace(
            state: AgentGraphState,
            *,
            node: str,
            assistant_message: str | None = None,
            tool_calls: list[dict[str, Any]] | None = None,
            tool_results: list[dict[str, Any]] | None = None,
            model_request: dict[str, Any] | None = None,
            model_response: dict[str, Any] | None = None,
        ) -> None:
            step_payload = StepRecord(
                step_index=next_step_index(state),
                node=node,
                assistant_message=assistant_message,
                tool_calls=tool_calls or [],
                tool_results=tool_results or [{"ok": None, "status": "in_progress"}],
                ok=False,
                model_request=model_request,
                model_response=model_response,
            ).to_dict()
            step_payload["status"] = "in_progress"
            step_payload["started_at"] = trace_timestamp()
            emit_trace(state, {"steps": [step_payload]})

        def next_step_index(state: AgentGraphState) -> int:
            return len(state.get("steps", [])) + 1

        def init_state(_: AgentGraphState) -> AgentGraphState:
            _PROMPT_BUILDERS = {
                1: build_system_prompt,
                2: build_system_prompt_v2,
                3: build_system_prompt_v3,
            }
            builder = _PROMPT_BUILDERS.get(self.config.prompt_version, build_system_prompt)
            catalog_top_n = self.config.data_inspector.sample_budget.catalog_top_distinct_values
            try:
                system_prompt = builder(catalog_top_n=catalog_top_n)
            except TypeError:
                system_prompt = builder()
            return {
                "task_id": task.task_id,
                "messages": [
                    SystemMessage(content=system_prompt),
                ],
                "step_count": 0,
                "empty_stop_retry_count": 0,
                "last_model_had_invalid_tool_calls": False,
                "last_invalid_tool_call_errors": [],
                "validation_retry_count": 0,
                "answer_validation_history": [],
                "process_validation_retry_count": 0,
                "last_process_validated_model_count": 0,
                "process_validation_passed": False,
                "semantic_ledger": None,
                "answer": None,
                "answer_submission": None,
                "failure_reason": None,
                "steps": [],
                "tool_events": [],
                "temp_workspace": None,
                "inspector": None,
                "global_data_profile": None,
                "started_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }

        def global_data_exploration(state: AgentGraphState) -> AgentGraphState:
            if not self.config.enable_data_inspector:
                return {}
            _step_start = perf_counter()
            _step_started_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            emit_in_progress_trace(
                state,
                node="global_data_exploration",
                assistant_message="Global data profiling is in progress.",
                tool_results=[
                    {"ok": None, "status": "in_progress", "phase": "global_data_exploration"}
                ],
            )
            try:
                understanding_agent = DataUnderstandingAgent(
                    config=self.config.data_inspector,
                )
                exploration_kwargs: dict[str, object] = {
                    "context_dir": task.context_dir,
                    "task_id": task.task_id,
                }
                if task.assets.context_view is not None:
                    exploration_kwargs["context_view"] = task.assets.context_view
                profile, catalog = understanding_agent.explore_data_globally(**exploration_kwargs)
                profile_preview = _preview_text(profile, limit=500) or ""
                step_record = StepRecord(
                    step_index=next_step_index(state),
                    node="global_data_exploration",
                    assistant_message=profile_preview,
                    tool_calls=[],
                    tool_results=[{"ok": True, "content": {"profile_length": len(profile)}}],
                    ok=True,
                    model_request=None,
                    model_response=None,
                    started_at=_step_started_at,
                    elapsed_seconds=round(perf_counter() - _step_start, 3),
                )
                update: AgentGraphState = {
                    "global_data_profile": profile,
                    "inspector": {"semantic_catalog": catalog},
                    "steps": [step_record.to_dict()],
                }
                runtime_context._catalog_cache = catalog
                emit_trace(state, update)
                return update
            except Exception as exc:  # noqa: BLE001
                step_record = StepRecord(
                    step_index=next_step_index(state),
                    node="global_data_exploration",
                    assistant_message=None,
                    tool_calls=[],
                    tool_results=[{"ok": False, "error": str(exc)}],
                    ok=False,
                    model_request=None,
                    model_response=None,
                    started_at=_step_started_at,
                    elapsed_seconds=round(perf_counter() - _step_start, 3),
                )
                update = {
                    "global_data_profile": f"Global profiling failed: {exc}",
                    "steps": [step_record.to_dict()],
                }
                emit_trace(state, update)
                return update

        def analyze_ambiguity_step(state: AgentGraphState) -> AgentGraphState:
            if not self.config.enable_ambiguity_analysis:
                return {}
            _step_start = perf_counter()
            _step_started_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            emit_in_progress_trace(
                state,
                node="analyze_ambiguity",
                assistant_message="Ambiguity analysis is in progress.",
                tool_results=[{"ok": None, "status": "in_progress", "phase": "ambiguity_analysis"}],
            )
            try:
                schemas: list[dict[str, Any]] | None = None
                knowledge_docs: list[dict[str, Any]] | None = None
                profile_str = state.get("global_data_profile") or ""
                if profile_str.strip():
                    try:
                        profile = json.loads(profile_str)
                        if isinstance(profile, dict):
                            schemas = profile.get("schemas")
                            knowledge_docs = _extract_knowledge_documents_from_profile(profile)
                    except json.JSONDecodeError:
                        pass
                result = analyze_ambiguity(
                    model=self.model,
                    question=task.question,
                    schemas=schemas,
                    knowledge_docs=knowledge_docs,
                )
                analysis_preview = json.dumps(result, ensure_ascii=False)[:500]
                step_record = StepRecord(
                    step_index=next_step_index(state),
                    node="analyze_ambiguity",
                    assistant_message=analysis_preview,
                    tool_calls=[],
                    tool_results=[{"ok": True, "content": result}],
                    ok=True,
                    model_request={"question": task.question},
                    model_response=result,
                    started_at=_step_started_at,
                    elapsed_seconds=round(perf_counter() - _step_start, 3),
                )
                update: AgentGraphState = {
                    "ambiguity_analysis": result,
                    "steps": [step_record.to_dict()],
                }
                emit_trace(state, update)
                return update
            except Exception as exc:  # noqa: BLE001
                empty_ambiguity = {
                    "question_intent": {
                        "entities": [],
                        "filters": [],
                        "metrics": [],
                        "requested_output": "",
                        "grain": "",
                    },
                    "ambiguities": [],
                    "resolved_by_knowledge": [],
                    "non_ambiguous_candidates": [],
                }
                step_record = StepRecord(
                    step_index=next_step_index(state),
                    node="analyze_ambiguity",
                    assistant_message=None,
                    tool_calls=[],
                    tool_results=[{"ok": False, "error": str(exc)}],
                    ok=False,
                    model_request=None,
                    model_response=None,
                    started_at=_step_started_at,
                    elapsed_seconds=round(perf_counter() - _step_start, 3),
                )
                update = {
                    "ambiguity_analysis": empty_ambiguity,
                    "steps": [step_record.to_dict()],
                }
                emit_trace(state, update)
                return update

        def receive_problem(state: AgentGraphState) -> AgentGraphState:
            content_parts: list[str] = [
                f"<user_query>\nUser Question: {task.question}\n</user_query>"
            ]

            ambiguity_analysis = state.get("ambiguity_analysis") or {}
            global_data_profile = state.get("global_data_profile") or ""
            has_catalog = self.config.enable_data_inspector and global_data_profile.strip()
            has_analysis = bool(ambiguity_analysis)

            if has_catalog or has_analysis:
                preamble_parts: list[str] = []

                if has_catalog and has_analysis:
                    preamble_parts.append(
                        "To help you answer the <user_query>, here are the data "
                        "lightweight catalog and the prior ambiguity analysis.  The "
                        "lightweight catalog contains query_surfaces, documents, "
                        "knowledge text, and media paths. Query surfaces are the "
                        "recommended SQL entry points: derived surfaces are query "
                        "conveniences, not original tables from knowledge.md, while "
                        "original_table surfaces are direct logical tables. Field "
                        "meanings for derived surfaces come from each field's "
                        "source_table/source_field. Use semantic "
                        "catalog tools for full field profiles, top distinct values, "
                        "min/max ranges, and relationship evidence.\n\n"
                        "The ambiguity analysis section identifies semantic risks "
                        "that could lead to wrong answers.  It does NOT field-bind "
                        "or resolve ambiguities — it surfaces what you should "
                        "verify.  Resolve each ambiguity by probing real data, "
                        "then verify non-ambiguous candidates, and output an "
                        "ambiguity resolution log before computing the final answer."
                    )
                elif has_catalog:
                    preamble_parts.append(
                        "To help you answer the <user_query>, here is the lightweight "
                        "catalog. It contains query_surfaces, documents, knowledge "
                        "text, and media paths. Start from query_surfaces when "
                        "choosing SQL entry points. Treat kind=derived_view surfaces "
                        "as derived query conveniences, not original tables from "
                        "knowledge.md; kind=original_table surfaces are direct "
                        "logical tables. Field meanings for derived surfaces come "
                        "from each field's source_table/source_field; derived "
                        "surfaces do not apply filters, aggregation, deduplication, "
                        "latest-record rules, or unit conversions. Use semantic catalog tools "
                        "when you need full field profiles, top distinct values, "
                        "min/max ranges, or relationship evidence."
                    )
                elif has_analysis:
                    preamble_parts.append(
                        "To help you answer the <user_query>, here is the prior "
                        "ambiguity analysis — a pre-risk identification checklist.  "
                        "It identifies semantic ambiguities that could lead to wrong "
                        "answers.  It does NOT field-bind or resolve ambiguities.  "
                        "Resolve each ambiguity with actual data probes before "
                        "computing."
                    )
                context_parts: list[str] = list(preamble_parts)

                if has_catalog:
                    context_parts.append(
                        f"<lightweight_catalog>\n{global_data_profile}\n</lightweight_catalog>"
                    )

                if has_analysis:
                    context_parts.append(
                        "<ambiguity_analysis>\n"
                        f"{json.dumps(ambiguity_analysis, ensure_ascii=False, indent=2)}\n"
                        "</ambiguity_analysis>"
                    )

                context_body = "\n\n".join(context_parts)
                content_parts.append(f"<context_injection>\n{context_body}\n</context_injection>")

                action_target = "the provided context"
                if has_analysis and has_catalog:
                    action_target = "the <ambiguity_analysis> and <lightweight_catalog>"
                elif has_analysis:
                    action_target = "the <ambiguity_analysis>"
                elif has_catalog:
                    action_target = "the <lightweight_catalog>"
            else:
                action_target = "the user question"

            ambiguities_list = ambiguity_analysis.get("ambiguities", []) if has_analysis else []
            if ambiguities_list:
                # Concrete per-ambiguity resolution strategy
                amb_items: list[str] = []
                for amb in ambiguities_list:
                    cq = amb.get("clarifying_question", "")
                    rv = amb.get("required_verification", [])
                    rv_text = "; ".join(rv) if rv else "probe real data"
                    amb_items.append(
                        f'- {amb["id"]} ({amb["type"]}): "{amb["phrase"]}"\n'
                        f"  Clarifying question: {cq}\n"
                        f"  Required verification: {rv_text}"
                    )
                action_parts: list[str] = [
                    "<action_trigger>",
                    "CRITICAL: You MUST resolve ALL ambiguities below BEFORE computing the final answer.",
                    "",
                    "For each ambiguity:",
                    "  1. Probe real data to answer the clarifying question.",
                    "  2. Use the required_verification steps as a starting point.",
                    "  3. Explicitly record which interpretation was chosen and why.",
                    "",
                    "Ambiguities to resolve:",
                    *amb_items,
                    "",
                    "Resolution strategy:",
                    "  Step 1: Identify candidate fields for each ambiguity from the lightweight catalog above.",
                    "  Step 2: Inspect semantic profiles/relationships and probe real data for each ambiguity.",
                    "  Step 3: Output an ambiguity resolution log — one line per ambiguity, stating the chosen interpretation and the data evidence.",
                    "  Step 4: Only after ALL ambiguities are resolved, proceed to compute the final answer.",
                    "</action_trigger>",
                ]
                content_parts.append("\n".join(action_parts))
            elif has_analysis and has_catalog:
                content_parts.append(
                    "<action_trigger>\n"
                    "No blocking ambiguities were identified.  Verify the "
                    "non_ambiguous_candidates against the catalog, then proceed "
                    "to compute the answer.\n"
                    "</action_trigger>"
                )
            else:
                content_parts.append(
                    "<action_trigger>\n"
                    f"Based on {action_target}, formulate your first thought and "
                    "execute the most appropriate tool to begin solving the "
                    "<user_query>.\n"
                    "</action_trigger>"
                )
            return {
                "messages": [
                    HumanMessage(
                        content=build_initial_user_content(
                            task,
                            "\n\n".join(content_parts),
                            max_attached_frames=self.config.max_attached_video_frames,
                        )
                    )
                ]
            }

        def model_step(state: AgentGraphState) -> AgentGraphState:
            if state.get("failure_reason") is not None or state.get("answer") is not None:
                return {}
            if state.get("step_count", 0) >= self.config.max_steps:
                return {"failure_reason": "Agent did not submit an answer within max_steps."}

            _step_start = perf_counter()
            _step_started_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            request_messages = _prepare_messages_for_model(
                list(state["messages"]),
                strip_reasoning_history=self.config.strip_reasoning_history,
                reasoning_history_limit=self.config.reasoning_history_limit,
                compress_used_image_messages=self.config.compress_used_image_messages,
                compressed_image_note_chars=self.config.compressed_image_note_chars,
            )
            request_payload = _summarize_model_request(
                messages=request_messages,
                tools=langchain_tools,
                tool_choice=tool_choice,
                parallel_tool_calls=parallel_tool_calls,
            )
            emit_in_progress_trace(
                state,
                node="model",
                tool_results=[{"ok": None, "status": "in_progress", "phase": "model_request"}],
                model_request=request_payload,
            )
            retry_events: list[dict[str, Any]] = []

            def record_model_retry(event: dict[str, Any]) -> None:
                retry_events.append(dict(event))
                retry_status = "retrying" if event.get("will_retry") else "failed"
                emit_in_progress_trace(
                    state,
                    node="model",
                    tool_results=[
                        {
                            "ok": False,
                            "status": retry_status,
                            "phase": "model_request",
                            "attempt": event.get("attempt"),
                            "max_attempts": event.get("max_attempts"),
                            "error": event.get("error"),
                            "next_retry_delay_seconds": event.get("next_retry_delay_seconds"),
                        }
                    ],
                    model_request=request_payload,
                    model_response={
                        "request_retry": summarize_model_retry_events(retry_events),
                    },
                )

            try:
                ai_message = invoke_model_with_retries(
                    model_with_tools,
                    request_messages,
                    on_retry_event=record_model_retry,
                    timeout_seconds=self.config.model_request_timeout_seconds,
                )
            except Exception as exc:
                # When the model emits a tool call whose function.arguments is
                # not valid JSON the provider returns a 400 that is not
                # retryable.  Inject escalating repair prompts and retry up to
                # JSON_REPAIR_MAX_ATTEMPTS times before giving up.
                error_text = str(exc)
                if (
                    "function.arguments" in error_text
                    and "JSON format" in error_text
                ):
                    JSON_REPAIR_MAX_ATTEMPTS = 3
                    JSON_REPAIR_PROMPTS = [
                        # Attempt 1 — gentle, diagnostic
                        (
                            "Your previous response contained a tool call with "
                            "malformed arguments that were not valid JSON.\n\n"
                            "Common causes:\n"
                            "- Unescaped double-quotes inside a JSON string value\n"
                            "- Trailing text or commentary after the closing brace\n"
                            "- The arguments block was cut off mid-JSON (truncation)\n\n"
                            "Re-generate your tool call with correctly formatted JSON "
                            "arguments. Double-check that every \" inside a string "
                            "value is escaped as \\\". If the arguments are complex, "
                            "use a simpler tool or break them into smaller calls."
                        ),
                        # Attempt 2 — prescriptive, command-style
                        (
                            "Your tool call arguments were STILL not valid JSON.\n\n"
                            "This is a blocking error — the system CANNOT process "
                            "your response until the JSON is valid.\n\n"
                            "Follow these rules exactly:\n"
                            "1. Output ONLY the tool call — no extra text before or after.\n"
                            "2. Use double quotes for ALL keys and string values.\n"
                            "3. Escape every literal double-quote inside a string "
                            "value as backslash-quote.\n"
                            "4. Ensure braces and brackets are balanced.\n"
                            "5. Do NOT put markdown, reasoning, or commentary "
                            "inside the arguments block.\n\n"
                            "Call a DIFFERENT, simpler tool if necessary."
                        ),
                        # Attempt 3 — minimal, failsafe
                        (
                            "STILL invalid JSON. This is your LAST attempt.\n\n"
                            "Call submit_tool_result with the simplest possible "
                            "tool_args. If the SQL query contains special "
                            "characters, call execute_python instead and have "
                            "the Python code print the result as CSV. Then "
                            "call submit_tool_result with tool_name='execute_python' "
                            "and tool_args={'code': 'print(...)'}.\n\n"
                            "The tool call MUST have valid JSON arguments. "
                            "No exceptions, no commentary in the arguments field."
                        ),
                    ]

                    repair_attempt = 0
                    repair_succeeded = False
                    last_repair_error: Exception | None = None

                    while repair_attempt < JSON_REPAIR_MAX_ATTEMPTS and not repair_succeeded:
                        repair_attempt += 1
                        repair_prompt = HumanMessage(
                            content=JSON_REPAIR_PROMPTS[repair_attempt - 1]
                        )
                        try:
                            request_messages_repair = _prepare_messages_for_model(
                                [*request_messages, repair_prompt],
                                strip_reasoning_history=self.config.strip_reasoning_history,
                                reasoning_history_limit=self.config.reasoning_history_limit,
                                compress_used_image_messages=self.config.compress_used_image_messages,
                                compressed_image_note_chars=self.config.compressed_image_note_chars,
                            )
                            ai_message = invoke_model_with_retries(
                                model_with_tools,
                                request_messages_repair,
                                on_retry_event=record_model_retry,
                                timeout_seconds=self.config.model_request_timeout_seconds,
                            )
                            # Inject the repair prompt into history so the trace
                            # shows the recovery step.
                            state["messages"].append(repair_prompt)
                            repair_succeeded = True
                        except Exception as exc_repair:
                            last_repair_error = exc_repair
                            # If this error is also a JSON format error, try
                            # the next repair prompt; otherwise give up.
                            exc_repair_text = str(exc_repair)
                            if not (
                                "function.arguments" in exc_repair_text
                                and "JSON format" in exc_repair_text
                            ):
                                break
                            # Add the failed repair prompt to history so the
                            # model sees the escalating instructions.
                            state["messages"].append(repair_prompt)

                    if not repair_succeeded:
                        model_response = {
                            "error": str(last_repair_error) if last_repair_error else error_text,
                            "json_repair": {
                                "attempted": True,
                                "attempts": repair_attempt,
                                "max_attempts": JSON_REPAIR_MAX_ATTEMPTS,
                            },
                        }
                        request_retry = summarize_model_retry_events(
                            retry_events, succeeded=False
                        )
                        if request_retry is not None:
                            model_response["request_retry"] = request_retry
                        step_record = StepRecord(
                            step_index=next_step_index(state),
                            node="model",
                            assistant_message=None,
                            tool_calls=[],
                            tool_results=[
                                {
                                    "ok": False,
                                    "error": str(last_repair_error) if last_repair_error else error_text,
                                }
                            ],
                            ok=False,
                            model_request=request_payload,
                            model_response=model_response,
                            started_at=_step_started_at,
                            elapsed_seconds=round(perf_counter() - _step_start, 3),
                        )
                        update = {
                            "failure_reason": (
                                f"Model request failed after {repair_attempt} JSON repair "
                                f"attempt(s): {last_repair_error}"
                            ),
                            "steps": [step_record.to_dict()],
                        }
                        emit_trace(state, update)
                        return update
                else:
                    model_response = {"error": error_text}
                    request_retry = summarize_model_retry_events(
                        retry_events, succeeded=False
                    )
                    if request_retry is not None:
                        model_response["request_retry"] = request_retry
                    step_record = StepRecord(
                        step_index=next_step_index(state),
                        node="model",
                        assistant_message=None,
                        tool_calls=[],
                        tool_results=[{"ok": False, "error": error_text}],
                        ok=False,
                        model_request=request_payload,
                        model_response=model_response,
                        started_at=_step_started_at,
                        elapsed_seconds=round(perf_counter() - _step_start, 3),
                    )
                    update = {
                        "failure_reason": f"Model request failed: {exc}",
                        "steps": [step_record.to_dict()],
                    }
                    emit_trace(state, update)
                    return update

            invalid_tool_calls = list(ai_message.invalid_tool_calls)
            safe_ai_message = _discard_invalid_tool_calls(ai_message)
            recovered_ai_message, recovered_tool_call = _recover_pseudo_tool_call(
                safe_ai_message,
                available_tool_names=available_tool_names,
                tool_schemas=tool_schemas,
            )
            history_ai_message = _with_clean_reasoning_history_content(
                recovered_ai_message,
                strip_reasoning=self.config.strip_reasoning_history,
            )
            model_response = _summarize_ai_message(ai_message)
            invalid_tool_call_errors = _summarize_invalid_tool_calls(invalid_tool_calls)
            if invalid_tool_calls:
                model_response["invalid_tool_call_count"] = len(invalid_tool_calls)
                model_response["invalid_tool_calls"] = invalid_tool_call_errors
            if recovered_tool_call is not None:
                model_response["recovered_tool_call"] = True
                model_response["recovered_tool_call_source"] = recovered_tool_call.source
                model_response["recovered_tool_call_name"] = recovered_tool_call.tool_name
            request_retry = summarize_model_retry_events(retry_events, succeeded=True)
            if request_retry is not None:
                model_response["request_retry"] = request_retry
            step_record = StepRecord(
                step_index=next_step_index(state),
                node="model",
                assistant_message=_render_message_content(history_ai_message.content),
                tool_calls=_normalize_tool_calls(history_ai_message.tool_calls),
                tool_results=[],
                ok=True,
                model_request=request_payload,
                model_response=model_response,
                started_at=_step_started_at,
                elapsed_seconds=round(perf_counter() - _step_start, 3),
            )
            update = {
                "messages": [history_ai_message],
                "step_count": state.get("step_count", 0) + 1,
                "last_model_had_invalid_tool_calls": bool(invalid_tool_calls),
                "last_invalid_tool_call_errors": invalid_tool_call_errors,
                "steps": [step_record.to_dict()],
            }
            emit_trace(state, update)
            return update

        def tool_step(state: AgentGraphState) -> AgentGraphState:
            last_message = state["messages"][-1]
            if not isinstance(last_message, AIMessage):
                return {
                    "failure_reason": "Tool execution requested without a preceding AI tool call."
                }

            _step_start = perf_counter()
            _step_started_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            emit_in_progress_trace(
                state,
                node="tool",
                tool_calls=_normalize_tool_calls(last_message.tool_calls),
                tool_results=[{"ok": None, "status": "in_progress", "phase": "tool_execution"}],
            )

            tool_results: list[dict[str, Any]] = []
            tool_messages: list[ToolMessage] = []
            model_attachment_parts: list[dict[str, Any]] = []
            terminal_answer = state.get("answer")
            terminal_answer_submission = state.get("answer_submission")
            overall_ok = True

            for tool_call in last_message.tool_calls:
                tool_name = str(tool_call.get("name"))
                tool_args = _coerce_dict(tool_call.get("args"))
                tool_call_id = str(tool_call.get("id"))
                try:
                    if tool_name not in available_tool_names:
                        raise KeyError(f"Tool is not available to the model: {tool_name}")
                    result = bound_tools.execute(tool_name, tool_args)
                    payload = self.tools.format_result(tool_name, result)
                    payload["tool"] = tool_name
                    if result.answer is not None:
                        terminal_answer = result.answer
                        terminal_answer_submission = result.answer_submission
                    if result.model_content_parts:
                        model_attachment_parts.extend(result.model_content_parts)
                    overall_ok = overall_ok and result.ok
                except Exception as exc:
                    payload = {
                        "ok": False,
                        "tool": tool_name,
                        "error": str(exc),
                    }
                    overall_ok = False

                tool_results.append(payload)
                tool_messages.append(
                    ToolMessage(
                        content=json.dumps(payload, ensure_ascii=False),
                        name=tool_name,
                        tool_call_id=tool_call_id,
                        status="success" if payload.get("ok") else "error",
                    )
                )

            step_record = StepRecord(
                step_index=next_step_index(state),
                node="tool",
                assistant_message=None,
                tool_calls=_normalize_tool_calls(last_message.tool_calls),
                tool_results=tool_results,
                ok=overall_ok,
                model_response=None,
                started_at=_step_started_at,
                elapsed_seconds=round(perf_counter() - _step_start, 3),
            )

            update: AgentGraphState = {
                "messages": [
                    *tool_messages,
                    *(
                        [HumanMessage(content=model_attachment_parts)]
                        if model_attachment_parts
                        else []
                    ),
                ],
                "empty_stop_retry_count": 0,
                "last_model_had_invalid_tool_calls": False,
                "steps": [step_record.to_dict()],
                "tool_events": list(tool_results),
                "temp_workspace": runtime_context.temp_workspace,
            }
            if terminal_answer is not None:
                update["answer"] = terminal_answer
                update["answer_submission"] = terminal_answer_submission
            emit_trace(state, update)
            return update

        def force_answer_step(state: AgentGraphState) -> AgentGraphState:
            if state.get("failure_reason") is not None or state.get("answer") is not None:
                return {}
            if state.get("forced_answer_attempted", False):
                return {"failure_reason": "Agent did not submit an answer within max_steps."}
            if not final_submission_tools:
                return {
                    "forced_answer_attempted": True,
                    "failure_reason": "Final submission tool is not available.",
                }

            _step_start = perf_counter()
            _step_started_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            force_prompt = HumanMessage(content=FORCE_ANSWER_PROMPT)
            request_messages = _prepare_messages_for_model(
                [*list(state["messages"]), force_prompt],
                strip_reasoning_history=self.config.strip_reasoning_history,
                reasoning_history_limit=self.config.reasoning_history_limit,
                compress_used_image_messages=self.config.compress_used_image_messages,
                compressed_image_note_chars=self.config.compressed_image_note_chars,
            )
            request_payload = _summarize_model_request(
                messages=request_messages,
                tools=final_submission_tools,
                tool_choice=forced_submission_tool_choice,
                parallel_tool_calls=parallel_tool_calls,
            )
            request_payload["forced_answer"] = True
            emit_in_progress_trace(
                state,
                node="force_answer",
                tool_results=[{"ok": None, "status": "in_progress", "phase": "model_request"}],
                model_request=request_payload,
            )
            retry_events: list[dict[str, Any]] = []

            def record_model_retry(event: dict[str, Any]) -> None:
                retry_events.append(dict(event))
                retry_status = "retrying" if event.get("will_retry") else "failed"
                emit_in_progress_trace(
                    state,
                    node="force_answer",
                    tool_results=[
                        {
                            "ok": False,
                            "status": retry_status,
                            "phase": "model_request",
                            "attempt": event.get("attempt"),
                            "max_attempts": event.get("max_attempts"),
                            "error": event.get("error"),
                            "next_retry_delay_seconds": event.get("next_retry_delay_seconds"),
                        }
                    ],
                    model_request=request_payload,
                    model_response={
                        "request_retry": summarize_model_retry_events(retry_events),
                    },
                )

            try:
                model_with_final_submission_tool = self.model.bind_tools(
                    final_submission_tools,
                    tool_choice=forced_submission_tool_choice,
                    parallel_tool_calls=parallel_tool_calls,
                )
                ai_message = invoke_model_with_retries(
                    model_with_final_submission_tool,
                    request_messages,
                    on_retry_event=record_model_retry,
                    timeout_seconds=self.config.model_request_timeout_seconds,
                )
            except Exception as exc:
                model_response = {"error": str(exc)}
                request_retry = summarize_model_retry_events(retry_events, succeeded=False)
                if request_retry is not None:
                    model_response["request_retry"] = request_retry
                step_record = StepRecord(
                    step_index=next_step_index(state),
                    node="force_answer",
                    assistant_message=None,
                    tool_calls=[],
                    tool_results=[{"ok": False, "error": str(exc)}],
                    ok=False,
                    model_request=request_payload,
                    model_response=model_response,
                    started_at=_step_started_at,
                    elapsed_seconds=round(perf_counter() - _step_start, 3),
                )
                update = {
                    "forced_answer_attempted": True,
                    "failure_reason": f"Model request failed during forced final answer: {exc}",
                    "steps": [step_record.to_dict()],
                }
                emit_trace(state, update)
                return update

            recovered_ai_message, recovered_tool_call = _recover_pseudo_tool_call(
                ai_message,
                available_tool_names=final_submission_tool_names,
                tool_schemas=final_submission_tool_schemas,
            )
            history_ai_message = _with_clean_reasoning_history_content(
                recovered_ai_message,
                strip_reasoning=self.config.strip_reasoning_history,
            )
            model_response = _summarize_ai_message(ai_message)
            model_response["forced_answer"] = True
            if recovered_tool_call is not None:
                model_response["recovered_tool_call"] = True
                model_response["recovered_tool_call_source"] = recovered_tool_call.source
                model_response["recovered_tool_call_name"] = recovered_tool_call.tool_name
            request_retry = summarize_model_retry_events(retry_events, succeeded=True)
            if request_retry is not None:
                model_response["request_retry"] = request_retry
            step_record = StepRecord(
                step_index=next_step_index(state),
                node="force_answer",
                assistant_message=_render_message_content(history_ai_message.content),
                tool_calls=_normalize_tool_calls(history_ai_message.tool_calls),
                tool_results=[],
                ok=True,
                model_request=request_payload,
                model_response=model_response,
                started_at=_step_started_at,
                elapsed_seconds=round(perf_counter() - _step_start, 3),
            )
            update = {
                "messages": [force_prompt, history_ai_message],
                "forced_answer_attempted": True,
                "steps": [step_record.to_dict()],
            }
            if not history_ai_message.tool_calls:
                update["failure_reason"] = "Agent did not submit an answer within max_steps."
            emit_trace(state, update)
            return update

        def repair_step(state: AgentGraphState) -> AgentGraphState:
            _step_start = perf_counter()
            _step_started_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            prompt = (
                _build_invalid_tool_call_repair_prompt(
                    state.get("last_invalid_tool_call_errors", [])
                )
                if state.get("last_model_had_invalid_tool_calls", False)
                else EMPTY_STOP_REPAIR_PROMPT
            )
            step_record = StepRecord(
                step_index=next_step_index(state),
                node="repair",
                assistant_message=prompt,
                tool_calls=[],
                tool_results=[],
                ok=True,
                model_request=None,
                model_response=None,
                started_at=_step_started_at,
                elapsed_seconds=round(perf_counter() - _step_start, 3),
            )
            update = {
                "messages": [HumanMessage(content=prompt)],
                "empty_stop_retry_count": state.get("empty_stop_retry_count", 0) + 1,
                "last_model_had_invalid_tool_calls": False,
                "last_invalid_tool_call_errors": [],
                "steps": [step_record.to_dict()],
            }
            emit_trace(state, update)
            return update

        def validate_process_step(state: AgentGraphState) -> AgentGraphState:
            """Validate recent process evidence before continuing or accepting an answer."""
            if not self.config.enable_process_validator:
                return {}

            failure_reason = state.get("failure_reason")
            if failure_reason is not None:
                return {}

            _step_start = perf_counter()
            _step_started_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

            if not isinstance(self.model, BaseChatModel):
                return {}

            answer = state.get("answer")
            if hasattr(answer, "to_dict"):
                answer_dict = answer.to_dict()
            elif isinstance(answer, dict):
                answer_dict = dict(answer)
            else:
                answer_dict = None

            recent_step_limit = self.config.process_validator.recent_step_limit
            recent_steps = list(state.get("steps", []))[-recent_step_limit:]
            semantic_ledger = state.get("semantic_ledger") or {}
            submission_context = _submission_context_for_validator(state.get("answer_submission"))
            pv = self.config.process_validator
            supporting_source_evidence = _build_supporting_source_evidence(
                list(state.get("steps", [])), submission_context,
                max_items=pv.evidence_max_items,
                max_str_tokens=pv.evidence_max_str_tokens,
                max_list_items=pv.evidence_max_list_items,
            )
            knowledge_docs: list[dict[str, Any]] | None = None
            profile_str = state.get("global_data_profile") or ""
            if profile_str.strip():
                try:
                    profile = json.loads(profile_str)
                    if isinstance(profile, dict):
                        knowledge_docs = _extract_knowledge_documents_from_profile(profile)
                except json.JSONDecodeError:
                    pass
            video_summaries: list[dict[str, Any]] | None = None
            try:
                video_summaries = _extract_video_summaries(task)
            except Exception:
                logger.warning(
                    "[%s] Failed to extract video summaries for process validator.",
                    task.task_id,
                    exc_info=True,
                )
            submission_risk_report = detect_submission_risks(submission_context)
            current_model_count = state.get("step_count", 0)
            current_retry = state.get("process_validation_retry_count", 0)
            answer_scalar_preview_available, answer_scalar_preview = _submitted_answer_scalar_preview(answer_dict)
            validation_request = {
                "question": task.question,
                "has_answer": answer_dict is not None,
                "answer_columns": answer_dict.get("columns") if answer_dict else None,
                "answer_row_count": _submitted_answer_row_count(answer_dict) if answer_dict else 0,
                "answer_scalar_preview_available": answer_scalar_preview_available,
                "answer_scalar_preview": answer_scalar_preview,
                "recent_step_count": len(recent_steps),
                "model_count": current_model_count,
                "last_process_validated_model_count": state.get(
                    "last_process_validated_model_count", 0
                ),
                "semantic_ledger_keys": sorted(semantic_ledger.keys()),
                "has_submission_context": submission_context is not None,
                "supporting_evidence_count": len(supporting_source_evidence["evidence_items"]),
                "submission_risk_kinds": [
                    detection.get("kind")
                    for detection in submission_risk_report.get("detected", [])
                    if isinstance(detection, dict)
                ],
                "strict_v3_video_evidence": self.config.prompt_version == 3,
            }

            if state.get("process_validation_passed", False):
                step_record = StepRecord(
                    step_index=next_step_index(state),
                    node="validate_process",
                    assistant_message=(
                        "Process validation skipped because a prior process validation passed."
                    ),
                    tool_calls=[],
                    tool_results=[
                        {
                            "ok": True,
                            "skipped": True,
                            "reason": "prior_process_validation_passed",
                            "prior_process_validation_passed": True,
                            "retry_limit_reached": False,
                        }
                    ],
                    ok=True,
                    model_request=validation_request,
                    model_response=None,
                    started_at=_step_started_at,
                    elapsed_seconds=round(perf_counter() - _step_start, 3),
                )
                update: AgentGraphState = {
                    "last_process_validated_model_count": current_model_count,
                    "steps": [step_record.to_dict()],
                }
                emit_trace(state, update)
                return update

            # A positive retry budget permits one corrected submission to be checked after
            # the last rejection. A zero budget retains the existing "never invoke" mode.
            retry_budget_exhausted = (
                self.config.process_validator.retry_limit == 0
                or current_retry > self.config.process_validator.retry_limit
            )
            if retry_budget_exhausted:
                step_record = StepRecord(
                    step_index=next_step_index(state),
                    node="validate_process",
                    assistant_message="Process validation skipped because retry limit was reached.",
                    tool_calls=[],
                    tool_results=[
                        {
                            "ok": True,
                            "skipped": True,
                            "reason": "retry_limit_reached",
                            "retry_limit_reached": True,
                        }
                    ],
                    ok=True,
                    model_request=validation_request,
                    model_response=None,
                    started_at=_step_started_at,
                    elapsed_seconds=round(perf_counter() - _step_start, 3),
                )
                update: AgentGraphState = {
                    "last_process_validated_model_count": current_model_count,
                    "steps": [step_record.to_dict()],
                }
                emit_trace(state, update)
                return update

            emit_in_progress_trace(
                state,
                node="validate_process",
                assistant_message="Process validator is checking recent work.",
                tool_results=[{"ok": None, "status": "in_progress", "phase": "process_validation"}],
                model_request=validation_request,
            )

            try:
                validation_result = invoke_process_validator(
                    model=self.model,
                    question=task.question,
                    answer=answer_dict,
                    submission_context=submission_context,
                    supporting_source_evidence=supporting_source_evidence,
                    submission_risk_report=submission_risk_report,
                    ambiguity_analysis=state.get("ambiguity_analysis"),
                    recent_steps=recent_steps,
                    semantic_ledger=semantic_ledger,
                    strict_video_evidence=self.config.prompt_version == 3,
                    knowledge_docs=knowledge_docs,
                    video_summaries=video_summaries,
                )
                is_valid = bool(validation_result.get("valid", True))
                issues = list(validation_result.get("issues", []))
                required_next_actions = list(validation_result.get("required_next_actions", []))
                next_ledger = _coerce_dict(validation_result.get("semantic_ledger"))
                validator_error = validation_result.get("validator_error")
                validation_response = {
                    "valid": is_valid,
                    "issues": issues,
                    "required_next_actions": required_next_actions,
                    "semantic_ledger": next_ledger,
                    "raw_response": validation_result.get("raw_response"),
                }

                if is_valid:
                    step_record = StepRecord(
                        step_index=next_step_index(state),
                        node="validate_process",
                        assistant_message="Process validation passed.",
                        tool_calls=[],
                        tool_results=[
                            {
                                "ok": True,
                                "valid": True,
                                "issues": [],
                                "required_next_actions": [],
                                "validator_error": validator_error,
                                "retry_limit_reached": False,
                            }
                        ],
                        ok=True,
                        model_request=validation_request,
                        model_response=validation_response,
                        started_at=_step_started_at,
                        elapsed_seconds=round(perf_counter() - _step_start, 3),
                    )
                    update: AgentGraphState = {
                        "last_process_validated_model_count": current_model_count,
                        "process_validation_passed": True,
                        "semantic_ledger": next_ledger,
                        "steps": [step_record.to_dict()],
                    }
                    emit_trace(state, update)
                    return update

                issues_text = "\n".join(f"- {issue}" for issue in issues)
                actions_text = "\n".join(f"- {action}" for action in required_next_actions)
                feedback_message = (
                    "Your recent work did NOT pass the process validation check. "
                    "The checker found high-confidence risks in the evidence chain:\n"
                    f"{issues_text or '- Process evidence is insufficient.'}\n\n"
                    "Before submitting an answer, take these next actions:\n"
                    f"{actions_text or '- Run concrete data probes to verify the disputed assumptions.'}\n\n"
                    "Then continue solving and submit a corrected answer with `submit_tool_result`."
                )
                step_record = StepRecord(
                    step_index=next_step_index(state),
                    node="validate_process",
                    assistant_message=f"Process validation failed:\n{issues_text}",
                    tool_calls=[],
                    tool_results=[
                        {
                            "ok": False,
                            "valid": False,
                            "issues": issues,
                            "required_next_actions": required_next_actions,
                            "retry_limit_reached": False,
                        }
                    ],
                    ok=False,
                    model_request=validation_request,
                    model_response=validation_response,
                    started_at=_step_started_at,
                    elapsed_seconds=round(perf_counter() - _step_start, 3),
                )
                update = {
                    "answer": None,
                    "answer_submission": None,
                    "failure_reason": None,
                    "messages": [HumanMessage(content=feedback_message)],
                    "process_validation_retry_count": current_retry + 1,
                    "last_process_validated_model_count": current_model_count,
                    "semantic_ledger": next_ledger,
                    "steps": [step_record.to_dict()],
                }
                emit_trace(state, update)
                return update
            except Exception as exc:  # noqa: BLE001
                logger.warning("[%s] Process validator failed; continuing: %s", task.task_id, exc)
                step_record = StepRecord(
                    step_index=next_step_index(state),
                    node="validate_process",
                    assistant_message=None,
                    tool_calls=[],
                    tool_results=[{"ok": False, "error": str(exc)}],
                    ok=False,
                    started_at=_step_started_at,
                    elapsed_seconds=round(perf_counter() - _step_start, 3),
                )
                update = {
                    "last_process_validated_model_count": current_model_count,
                    "steps": [step_record.to_dict()],
                }
                emit_trace(state, update)
                return update

        def finalize(state: AgentGraphState) -> AgentGraphState:
            failure_reason = state.get("failure_reason")
            if state.get("answer") is None and failure_reason is None:
                last_message = state["messages"][-1] if state.get("messages") else None
                if state.get("step_count", 0) >= self.config.max_steps:
                    failure_reason = "Agent did not submit an answer within max_steps."
                elif isinstance(last_message, AIMessage) and not last_message.tool_calls:
                    failure_reason = "Model did not request a tool or submit an answer."
                else:
                    failure_reason = "Agent finished without submitting an answer."

            update = {
                "failure_reason": failure_reason,
                "temp_workspace": runtime_context.temp_workspace,
            }
            emit_trace(state, update, partial=False)
            return update

        def validate_answer_step(state: AgentGraphState) -> AgentGraphState:
            """Validate a submitted answer and optionally return control to the main agent."""
            if not self.config.enable_answer_validator:
                return {}

            answer = state.get("answer")
            failure_reason = state.get("failure_reason")

            if answer is None or failure_reason is not None:
                return {}

            _step_start = perf_counter()
            _step_started_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

            # Unit-test fakes in this repo are not BaseChatModel instances. Real runtime
            # models are, so production runs still get the validation node behavior.
            if not isinstance(self.model, BaseChatModel):
                return {}

            current_retry = state.get("validation_retry_count", 0)
            if current_retry >= self.config.validation_retry_limit:
                logger.info(
                    "[%s] Answer validation reached retry limit (%d); accepting answer.",
                    task.task_id,
                    self.config.validation_retry_limit,
                )
                step_record = StepRecord(
                    step_index=next_step_index(state),
                    node="validate_answer",
                    assistant_message="Answer validation retry limit reached; accepting current answer.",
                    tool_calls=[],
                    tool_results=[{"ok": True, "skipped": True, "reason": "max_retries_reached"}],
                    ok=True,
                    started_at=_step_started_at,
                    elapsed_seconds=round(perf_counter() - _step_start, 3),
                )
                update = {"steps": [step_record.to_dict()]}
                emit_trace(state, update)
                return update

            if hasattr(answer, "to_dict"):
                answer_dict_full = answer.to_dict()
            elif isinstance(answer, dict):
                answer_dict_full = dict(answer)
            else:
                return {}

            (
                answer_dict_for_validator,
                answer_structure_overview_for_validator,
                answer_truncated_for_validator,
            ) = _build_answer_validator_context(
                answer_dict_full,
                max_str_tokens=self.tools.tool_config.max_output_tokens,
                max_list_items=min(
                    self.tools.tool_config.max_list_items,
                    ANSWER_VALIDATOR_MAX_PREVIEW_ROWS,
                ),
            )
            answer_fingerprint = _submitted_answer_fingerprint(answer_dict_full)
            validation_history = list(state.get("answer_validation_history", []))
            submission_ctx = _submission_context_for_validator(state.get("answer_submission"))
            sr_report = detect_submission_risks(submission_ctx)
            submission_field_context = build_submission_field_context(
                submission_ctx,
                answer_dict_full,
                catalog=runtime_context._catalog_cache,
                context_dir=runtime_context.python_workspace.path,
            )
            submission_field_context_preview = _submission_field_context_trace_preview(
                submission_field_context
            )
            pv = self.config.process_validator
            knowledge_docs: list[dict[str, Any]] | None = None
            profile_str = state.get("global_data_profile") or ""
            if profile_str.strip():
                try:
                    profile = json.loads(profile_str)
                    if isinstance(profile, dict):
                        knowledge_docs = _extract_knowledge_documents_from_profile(profile)
                except json.JSONDecodeError:
                    pass
            supporting_source_evidence = _build_supporting_source_evidence(
                list(state.get("steps", [])),
                submission_ctx,
                max_items=pv.evidence_max_items,
                max_str_tokens=pv.evidence_max_str_tokens,
                max_list_items=pv.evidence_max_list_items,
            )
            evidence_items = supporting_source_evidence["evidence_items"]
            has_video_evidence = any(
                any(
                    capability
                    in {
                        "video_narrative_context",
                        "visual_fact",
                        "visual_fact_receipt",
                    }
                    for capability in item.get("capabilities", [])
                )
                for item in evidence_items
                if isinstance(item, dict)
            )
            submission_risk_kinds = [
                d.get("kind") for d in sr_report.get("detected", []) if isinstance(d, dict)
            ]
            validation_request = {
                "question": task.question,
                "answer_fingerprint": answer_fingerprint,
                "answer_columns": answer_dict_full.get("columns"),
                "answer_row_count": _submitted_answer_row_count(answer_dict_full),
                "validator_answer_truncated": answer_truncated_for_validator,
                "validation_history_count": len(validation_history),
                "submission_risk_kinds": submission_risk_kinds,
                "supporting_evidence_count": len(evidence_items),
                "has_video_evidence": has_video_evidence,
                "knowledge_doc_count": len(knowledge_docs or []),
                "submission_field_context_status": submission_field_context.get("status"),
                "submission_field_source_table_count": len(
                    submission_field_context.get("source_tables", [])
                    if isinstance(submission_field_context, dict)
                    else []
                ),
                "submission_field_warning_kinds": [
                    warning.get("kind")
                    for warning in submission_field_context.get("warnings", [])
                    if isinstance(warning, dict)
                ],
                "submission_field_context_preview": submission_field_context_preview,
            }
            logger.info(
                "[%s] Answer validator is checking submitted answer (attempt %d)...",
                task.task_id,
                current_retry + 1,
            )
            emit_in_progress_trace(
                state,
                node="validate_answer",
                assistant_message="Answer validator is checking the submitted answer.",
                tool_results=[{"ok": None, "status": "in_progress", "phase": "answer_validation"}],
            )

            try:
                history_update: list[dict[str, Any]] = []
                cached = False
                validation_result = invoke_answer_validator(
                    model=self.model,
                    question=task.question,
                    answer=answer_dict_for_validator,
                    validation_history=_summarize_validation_history(validation_history),
                    answer_truncated=answer_truncated_for_validator,
                    answer_row_count=_submitted_answer_row_count(answer_dict_full),
                    preview_row_limit=ANSWER_VALIDATOR_MAX_PREVIEW_ROWS,
                    answer_structure_overview=answer_structure_overview_for_validator,
                    submission_risk_kinds=submission_risk_kinds,
                    submission_source=submission_ctx,
                    submission_field_context=submission_field_context,
                    supporting_source_evidence=supporting_source_evidence,
                    knowledge_docs=knowledge_docs,
                )
                history_update = [
                    _validation_history_entry(
                        answer_fingerprint=answer_fingerprint,
                        answer=answer_dict_full,
                        validation_result=validation_result,
                    )
                ]

                is_valid = bool(validation_result.get("valid", True))
                issues = list(validation_result.get("issues", []))
                validator_error = validation_result.get("validator_error")
                rationale = str(validation_result.get("rationale") or "")
                validation_response = {
                    "valid": is_valid,
                    "rationale": rationale,
                    "issues": issues,
                    "raw_response": validation_result.get("raw_response"),
                    "cached": cached,
                }

                if is_valid:
                    logger.info("[%s] Answer validation passed.", task.task_id)
                    step_record = StepRecord(
                        step_index=next_step_index(state),
                        node="validate_answer",
                        assistant_message="Answer format validation passed.",
                        tool_calls=[],
                        tool_results=[
                            {
                                "ok": True,
                                "valid": True,
                                "rationale": rationale,
                                "issues": [],
                                "validator_error": validator_error,
                                "cached": cached,
                            }
                        ],
                        ok=True,
                        model_request=validation_request,
                        model_response=validation_response,
                        started_at=_step_started_at,
                        elapsed_seconds=round(perf_counter() - _step_start, 3),
                    )
                    update = {"steps": [step_record.to_dict()]}
                    if history_update:
                        update["answer_validation_history"] = history_update
                    emit_trace(state, update)
                    return update

                issues_text = "\n".join(f"- {issue}" for issue in issues)

                # 校验未过，但若已无重试余量（步数耗尽 / 处于强制答案阶段），
                # 丢弃答案只会让 finalize 报 "did not submit" 输出零预测。
                # 此时接受当前已提交答案作为 best-effort（列可能正确，仍有 recall 机会），
                # 与上面「retry 上限已到则接受答案」的处理保持一致。
                no_retry_budget = (
                    state.get("step_count", 0) >= self.config.max_steps
                    or state.get("forced_answer_attempted", False)
                )
                if no_retry_budget:
                    logger.info(
                        "[%s] Answer validation failed but no retry budget remains; "
                        "accepting current answer as best-effort:\n%s",
                        task.task_id,
                        issues_text,
                    )
                    step_record = StepRecord(
                        step_index=next_step_index(state),
                        node="validate_answer",
                        assistant_message=(
                            "Answer validation failed but no retry budget remains; "
                            "accepting current answer as best-effort."
                        ),
                        tool_calls=[],
                        tool_results=[
                            {
                                "ok": True,
                                "valid": False,
                                "accepted_best_effort": True,
                                "rationale": rationale,
                                "issues": issues,
                                "cached": cached,
                            }
                        ],
                        ok=True,
                        model_request=validation_request,
                        model_response=validation_response,
                        started_at=_step_started_at,
                        elapsed_seconds=round(perf_counter() - _step_start, 3),
                    )
                    update = {"steps": [step_record.to_dict()]}
                    if history_update:
                        update["answer_validation_history"] = history_update
                    emit_trace(state, update)
                    return update

                feedback_message = (
                    "Your submitted answer did NOT pass the answer validation check. "
                    "The following issues were found:\n"
                    f"{issues_text}\n\n"
                    "Your previous answer, which has been rejected:\n"
                    "(shown as a bounded validator-context structure overview with no row samples; "
                    "the stored submitted answer remains complete)\n"
                    f"```json\n{json.dumps({'answer_structure_overview': answer_structure_overview_for_validator}, ensure_ascii=False, indent=2)}\n```\n\n"
                    "Please fix the issues above and re-submit by calling "
                    "`submit_tool_result` again. "
                    "This validator only checks delivery. If the issue says the process receipt "
                    "is stale, submit again so process validation can re-approve the semantic path. "
                    "Key formatting rules:\n"
                    "1. Dates must be ISO 8601 format with zero-padding, e.g. "
                    "'2024-03-01', not '2024-3-1'.\n"
                    "2. DateTime with timezone must be converted to UTC ending with 'Z'.\n"
                    "3. Only include columns that the question asks for.\n"
                    "4. String values are case-sensitive; do not change their case.\n"
                    "You may directly call `submit_tool_result` with the corrected table."
                )
                logger.info(
                    "[%s] Answer validation failed with %d issue(s); returning to main agent:\n%s",
                    task.task_id,
                    len(issues),
                    issues_text,
                )
                step_record = StepRecord(
                    step_index=next_step_index(state),
                    node="validate_answer",
                    assistant_message=f"Answer validation failed:\n{issues_text}",
                    tool_calls=[],
                    tool_results=[
                        {
                            "ok": False,
                            "valid": False,
                            "rationale": rationale,
                            "issues": issues,
                            "cached": cached,
                        }
                    ],
                    ok=False,
                    model_request=validation_request,
                    model_response=validation_response,
                    started_at=_step_started_at,
                    elapsed_seconds=round(perf_counter() - _step_start, 3),
                )
                # 走到这里说明仍有重试余量（no_retry_budget 已在上面提前返回），
                # 清空答案并把校验反馈交回主循环重新提交。
                update: AgentGraphState = {
                    "answer": None,
                    "answer_submission": None,
                    "failure_reason": None,
                    "messages": [HumanMessage(content=feedback_message)],
                    "validation_retry_count": current_retry + 1,
                    "steps": [step_record.to_dict()],
                }
                if history_update:
                    update["answer_validation_history"] = history_update
                emit_trace(state, update)
                return update
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[%s] Answer validator failed; accepting original answer: %s", task.task_id, exc
                )
                step_record = StepRecord(
                    step_index=next_step_index(state),
                    node="validate_answer",
                    assistant_message=None,
                    tool_calls=[],
                    tool_results=[{"ok": False, "error": str(exc)}],
                    ok=False,
                    started_at=_step_started_at,
                    elapsed_seconds=round(perf_counter() - _step_start, 3),
                )
                update = {"steps": [step_record.to_dict()]}
                emit_trace(state, update)
                return update

        def route_after_model(state: AgentGraphState) -> str:
            if state.get("failure_reason") is not None:
                return "finalize"
            if state.get("answer") is not None:
                return (
                    "validate_process"
                    if self.config.enable_process_validator
                    and not state.get("process_validation_passed", False)
                    else "validate_answer"
                )
            last_message = state["messages"][-1]
            if isinstance(last_message, AIMessage) and last_message.tool_calls:
                return "tool_step"
            if state.get("step_count", 0) >= self.config.max_steps:
                return "finalize" if state.get("forced_answer_attempted", False) else "force_answer"
            if (
                isinstance(last_message, AIMessage)
                and _is_non_action_stop(last_message)
                and state.get("empty_stop_retry_count", 0) < self.config.empty_stop_retry_limit
            ):
                return "repair_step"
            return "finalize"

        def route_after_tool(state: AgentGraphState) -> str:
            if state.get("failure_reason") is not None:
                return "finalize"
            if state.get("answer") is not None:
                if state.get("forced_answer_attempted", False):
                    return "finalize"
                return (
                    "validate_process"
                    if self.config.enable_process_validator
                    and not state.get("process_validation_passed", False)
                    else "validate_answer"
                )
            if state.get("step_count", 0) >= self.config.max_steps:
                return "finalize" if state.get("forced_answer_attempted", False) else "force_answer"
            if (
                self.config.enable_process_validator
                and not state.get("process_validation_passed", False)
                and state.get("step_count", 0) - state.get("last_process_validated_model_count", 0)
                >= self.config.process_validator.checkpoint_model_interval
            ):
                return "validate_process"
            return "model_step"

        def route_after_process_validation(state: AgentGraphState) -> str:
            if state.get("failure_reason") is not None:
                return "finalize"
            if state.get("answer") is not None:
                if state.get("forced_answer_attempted", False):
                    return "finalize"
                return "validate_answer"
            if state.get("step_count", 0) >= self.config.max_steps:
                return "finalize" if state.get("forced_answer_attempted", False) else "force_answer"
            return "model_step"

        def route_after_validation(state: AgentGraphState) -> str:
            if state.get("answer") is None and state.get("failure_reason") is None:
                if (
                    state.get("forced_answer_attempted", False)
                    and state.get("step_count", 0) >= self.config.max_steps
                ):
                    return "finalize"
                return "model_step"
            return "finalize"

        def route_after_force_answer(state: AgentGraphState) -> str:
            if state.get("failure_reason") is not None:
                return "finalize"
            if state.get("answer") is not None:
                return "finalize"
            last_message = state["messages"][-1]
            if isinstance(last_message, AIMessage) and last_message.tool_calls:
                return "tool_step"
            return "finalize"

        graph_builder = StateGraph(AgentGraphState)
        graph_builder.add_node("init_state", init_state)
        graph_builder.add_node("global_data_exploration", global_data_exploration)
        graph_builder.add_node("analyze_ambiguity", analyze_ambiguity_step)
        graph_builder.add_node("receive_problem", receive_problem)
        graph_builder.add_node("model_step", model_step)
        graph_builder.add_node("force_answer", force_answer_step)
        graph_builder.add_node("tool_step", tool_step)
        graph_builder.add_node("repair_step", repair_step)
        graph_builder.add_node("finalize", finalize)
        graph_builder.add_node("validate_process", validate_process_step)
        graph_builder.add_node("validate_answer", validate_answer_step)
        graph_builder.add_edge(START, "init_state")
        graph_builder.add_edge("init_state", "global_data_exploration")
        graph_builder.add_edge("global_data_exploration", "analyze_ambiguity")
        graph_builder.add_edge("analyze_ambiguity", "receive_problem")
        graph_builder.add_edge("receive_problem", "model_step")
        graph_builder.add_conditional_edges(
            "model_step",
            route_after_model,
            {
                "tool_step": "tool_step",
                "repair_step": "repair_step",
                "finalize": "finalize",
                "validate_process": "validate_process",
                "validate_answer": "validate_answer",
                "force_answer": "force_answer",
            },
        )
        graph_builder.add_edge("repair_step", "model_step")
        graph_builder.add_conditional_edges(
            "tool_step",
            route_after_tool,
            {
                "model_step": "model_step",
                "finalize": "finalize",
                "validate_process": "validate_process",
                "validate_answer": "validate_answer",
                "force_answer": "force_answer",
            },
        )
        graph_builder.add_conditional_edges(
            "validate_process",
            route_after_process_validation,
            {
                "model_step": "model_step",
                "finalize": "finalize",
                "validate_answer": "validate_answer",
                "force_answer": "force_answer",
            },
        )
        graph_builder.add_conditional_edges(
            "force_answer",
            route_after_force_answer,
            {
                "tool_step": "tool_step",
                "finalize": "finalize",
            },
        )
        graph_builder.add_conditional_edges(
            "validate_answer",
            route_after_validation,
            {
                "model_step": "model_step",
                "finalize": "finalize",
            },
        )
        graph_builder.add_edge("finalize", END)
        graph = graph_builder.compile()

        try:
            final_state = graph.invoke({})
        finally:
            python_workspace.cleanup()

        steps = [StepRecord(**step_payload) for step_payload in final_state.get("steps", [])]
        return AgentRunResult(
            task_id=task.task_id,
            answer=final_state.get("answer"),
            steps=steps,
            failure_reason=final_state.get("failure_reason"),
            inspector=final_state.get("inspector"),
            global_data_profile=final_state.get("global_data_profile"),
            ambiguity_analysis=final_state.get("ambiguity_analysis"),
            semantic_ledger=final_state.get("semantic_ledger"),
        )
