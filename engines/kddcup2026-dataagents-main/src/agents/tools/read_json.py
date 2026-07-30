"""read_json tool: structured JSON preview."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, cast

from agents.benchmark.schema import PublicTask
from agents.tools import constants
from agents.tools._fields import path_field
from agents.tools.context import resolve_context_path
from agents.tools.decorator import function_tool
from agents.tools.registry import ToolExecutionResult


def _truncate_string_value(text: str) -> tuple[str, bool]:
    """Cap string preview values and report whether truncation happened."""
    if len(text) <= constants.READ_JSON_STRING_VALUE_CAP:
        return text, False
    return text[: constants.READ_JSON_STRING_VALUE_CAP], True


def _shape_stub_for_array_item(item: Any) -> dict[str, Any]:
    """Return structure-only shape for array items that are too large to emit."""
    if isinstance(item, dict):
        keys: list[str] = list(cast(dict[str, Any], item).keys())
        return {
            "kind": "object",
            "key_count": len(keys),
            "keys": keys[:10],
        }
    if isinstance(item, list):
        item_list = cast(list[Any], item)
        first_kind: str | None = None
        if item_list:
            first_kind = type(item_list[0]).__name__
        return {
            "kind": "array",
            "length": len(item_list),
            "first_item_kind": first_kind,
        }
    return {"kind": "scalar", "value_kind": type(item).__name__}


def _array_head_with_caps(
    items: list[Any],
    *,
    item_cap: int = 0,
    aggregate_bytes: int = 0,
) -> tuple[list[Any], str | None]:
    """Build array head preview with per-item and aggregate byte caps."""
    if item_cap == 0:
        item_cap = constants.READ_JSON_ARRAY_HEAD_CAP
    if aggregate_bytes == 0:
        aggregate_bytes = constants.READ_JSON_ARRAY_HEAD_BYTES
    head: list[Any] = []
    cumulative_bytes = 0
    item_cap_hit = False
    byte_cap_hit = False
    for item in items[:item_cap]:
        serialized = json.dumps(item, ensure_ascii=False)
        if len(serialized) > constants.READ_JSON_ITEM_BYTE_LIMIT:
            stub = _shape_stub_for_array_item(item)
            stub_bytes = len(json.dumps(stub, ensure_ascii=False))
            if cumulative_bytes + stub_bytes > aggregate_bytes and head:
                byte_cap_hit = True
                break
            head.append(stub)
            cumulative_bytes += stub_bytes
            item_cap_hit = True
            continue
        if cumulative_bytes + len(serialized) > aggregate_bytes and head:
            byte_cap_hit = True
            break
        head.append(item)
        cumulative_bytes += len(serialized)
    truncation: str | None = None
    if byte_cap_hit:
        truncation = "byte_cap"
    elif item_cap_hit:
        truncation = "item_cap"
    return head, truncation


def _summarize_value_for_preview(value: Any) -> Any:
    """Shallow object value preview: scalars as-is, containers as shape summaries."""
    if isinstance(value, dict):
        return {"kind": "object", "key_count": len(cast(dict[Any, Any], value))}
    if isinstance(value, list):
        value_list = cast(list[Any], value)
        length = len(value_list)
        head, head_truncation = _array_head_with_caps(
            value_list,
            item_cap=constants.READ_JSON_VALUE_PREVIEW_HEAD_CAP,
            aggregate_bytes=constants.READ_JSON_VALUE_PREVIEW_HEAD_BYTES,
        )
        return {
            "kind": "array",
            "length": length,
            "head": head,
            "head_truncation": head_truncation,
            "truncated": length > len(head) or head_truncation is not None,
        }
    if isinstance(value, str):
        truncated_text, was_truncated = _truncate_string_value(value)
        if was_truncated:
            return {"value": truncated_text, "value_truncated": True}
        return truncated_text
    return value


def _streaming_json_schema(path: Path) -> dict[str, Any]:
    """Extract top-level JSON shape for files exceeding the normal load cap."""
    import ijson  # pyright: ignore[reportMissingTypeStubs]

    with path.open("rb") as file:
        first_event = next(iter(ijson.parse(file)), None)
    if first_event is None:
        return {
            "kind": "unknown",
            "streamed": True,
            "reason": "Empty file",
        }
    _, event, _ = first_event

    if event == "start_map":
        keys: list[str] = []
        sample_value_kinds: dict[str, str] = {}
        with path.open("rb") as file:
            for key, value in ijson.kvitems(file, ""):
                keys.append(key)
                if len(sample_value_kinds) < constants.READ_JSON_OBJECT_VALUE_PREVIEW_CAP:
                    sample_value_kinds[key] = type(value).__name__
        return {
            "kind": "object",
            "keys": keys[: constants.READ_JSON_OBJECT_KEYS_CAP],
            "key_count": len(keys),
            "sample_value_kinds": sample_value_kinds,
            "keys_truncated": len(keys) > constants.READ_JSON_OBJECT_KEYS_CAP,
            "streamed": True,
            "truncated": True,
        }

    if event == "start_array":
        length = 0
        first_item_kind: str | None = None
        with path.open("rb") as file:
            for item in ijson.items(file, "item"):
                if first_item_kind is None:
                    first_item_kind = type(item).__name__
                length += 1
        return {
            "kind": "array",
            "length": length,
            "first_item_kind": first_item_kind,
            "streamed": True,
            "truncated": True,
        }

    payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    return {
        "kind": "scalar",
        "value_kind": type(payload).__name__,
        "streamed": True,
    }


def read_json_preview(task: PublicTask, relative_path: str) -> dict[str, Any]:
    """Return structured JSON preview keyed by top-level kind."""
    path = resolve_context_path(task, relative_path)
    size = path.stat().st_size
    if size > constants.READ_JSON_SIZE_LIMIT:
        return _streaming_json_schema(path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(payload, dict):
        payload_dict = cast(dict[str, Any], payload)
        keys: list[str] = list(payload_dict.keys())
        key_count = len(keys)
        listed_keys = keys[: constants.READ_JSON_OBJECT_KEYS_CAP]
        value_preview: dict[str, Any] = {}
        for key in keys[: constants.READ_JSON_OBJECT_VALUE_PREVIEW_CAP]:
            value_preview[key] = _summarize_value_for_preview(payload_dict[key])
        keys_truncated = key_count > constants.READ_JSON_OBJECT_KEYS_CAP
        return {
            "kind": "object",
            "key_count": key_count,
            "keys": listed_keys,
            "keys_truncated": keys_truncated,
            "value_preview": value_preview,
            "truncated": keys_truncated or key_count > constants.READ_JSON_OBJECT_VALUE_PREVIEW_CAP,
        }

    if isinstance(payload, list):
        payload_list = cast(list[Any], payload)
        length = len(payload_list)
        head, head_truncation = _array_head_with_caps(payload_list)
        truncated = length > len(head) or head_truncation is not None
        return {
            "kind": "array",
            "length": length,
            "head": head,
            "head_truncation": head_truncation,
            "truncated": truncated,
        }

    if isinstance(payload, str):
        truncated_text, was_truncated = _truncate_string_value(payload)
        result: dict[str, Any] = {
            "kind": "scalar",
            "value": truncated_text,
        }
        if was_truncated:
            result["value_truncated"] = True
        return result
    return {
        "kind": "scalar",
        "value": payload,
    }


def summarize_json(path: Path) -> dict[str, Any]:
    """Extract JSON top-level shape and shallow schema for inspect_files."""
    size = path.stat().st_size
    if size > constants.INSPECT_FILES_JSON_SIZE_LIMIT:
        return _streaming_json_schema(path)
    payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    if isinstance(payload, dict):
        payload_dict = cast(dict[str, Any], payload)
        keys: list[str] = list(payload_dict.keys())
        sample_kinds = {key: type(payload_dict[key]).__name__ for key in keys[:10]}
        return {
            "kind": "object",
            "keys": keys[:20],
            "key_count": len(keys),
            "sample_value_kinds": sample_kinds,
        }
    if isinstance(payload, list):
        payload_list = cast(list[Any], payload)
        first_kind: str | None = None
        if payload_list:
            first_kind = type(payload_list[0]).__name__
        return {
            "kind": "array",
            "length": len(payload_list),
            "first_item_kind": first_kind,
        }
    return {
        "kind": "scalar",
        "value_kind": type(payload).__name__,
    }


@function_tool
def read_json(
    task: PublicTask,
    path: Annotated[
        str,
        path_field(file_kind="JSON file", examples=("json/Patient.json", "zip_code.json")),
    ],
) -> ToolExecutionResult:
    """Preview a single JSON file. Returns a structured summary keyed by kind:
    array → {length, head (first 20 items)}, object → {key_count, keys
    (first 50), value_preview (first 20)}, scalar → {value}. This is a
    SAMPLE preview, not full data access. For nested field extraction,
    filtering, cross-file joins, or deep traversal, use execute_python.
    Files >100MB return a streamed schema without full load.
    Example: read_json({"path": "json/Patient.json"})"""
    return ToolExecutionResult(ok=True, content=read_json_preview(task, path))
