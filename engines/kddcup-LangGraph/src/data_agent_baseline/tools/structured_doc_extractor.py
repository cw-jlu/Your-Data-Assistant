from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from time import perf_counter
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.config import StructuredDocToolConfig
from data_agent_baseline.inspectors.semantic_catalog import iter_logical_tables
from data_agent_baseline.model_retry import invoke_model_with_retries
from data_agent_baseline.tools.doc_structure import STRUCTURE_VERSION, load_doc_structure
from data_agent_baseline.tools.filesystem import normalize_context_relative_path, resolve_context_path
from data_agent_baseline.tools.python_exec import TaskContextWorkspace
from data_agent_baseline.tools.unit_normalizer import (
    generate_normalization_rule,
    normalize_facts,
)


GENERATED_STRUCTURED_DOC_DIR = ".generated/structured_doc"
VISIBLE_STRUCTURED_DOC_DIR = "structured_doc"
STRUCTURED_DOC_MANIFEST = "manifest.json"
PLAN_VERSION = 6
CHUNKING_VERSION = 1
LINE_ID_KEY = "line_id"


@dataclass(frozen=True, slots=True)
class StructuredDocExtraction:
    columns: list[str]
    rows: list[list[Any]]
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ExtractionPlan:
    target_fields: list[dict[str, str]]
    entity_key_fields: list[str]
    fallback_entity_key: str
    merge_grain: str
    field_hints: dict[str, str]
    field_value_specs: dict[str, dict[str, str | None]]


@dataclass(frozen=True, slots=True)
class MergeResult:
    rows: list[list[Any]]
    fact_count: int
    merged_row_count: int
    dropped_empty_fact_count: int
    column_non_null_counts: dict[str, int]
    field_conflicts: list[dict[str, Any]]
    missing_entity_key_count: int


@dataclass(frozen=True, slots=True)
class ChunkPlan:
    chunks: list[list[dict[str, Any]]]
    chunking_reason: str
    repair_budget: int


@dataclass(frozen=True, slots=True)
class StructuredDocChunkEstimate:
    priority_chunk_count: int | None
    selected_line_count: int | None
    priority_source: str
    error: str | None = None


class StructuredDocExtractionError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        log_summary: dict[str, Any] | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.log_summary = log_summary
        self.details = details or {}


class StructuredDocLogger:
    def __init__(
        self,
        workspace_root: Path,
        registered_table: str,
        *,
        log_dir: Path | None = None,
    ) -> None:
        self._started_at = perf_counter()
        self._events: list[dict[str, Any]] = []
        self.relative_log_file: str | None = None
        self.path: Path | None = None
        if log_dir is not None:
            visible_dir = log_dir / VISIBLE_STRUCTURED_DOC_DIR
            visible_dir.mkdir(parents=True, exist_ok=True)
            self.relative_log_file = (
                f"{VISIBLE_STRUCTURED_DOC_DIR}/structured_doc_{registered_table}.log.jsonl"
            )
            self.path = log_dir / self.relative_log_file
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("", encoding="utf-8")

    def emit(self, event: str, **details: Any) -> None:
        payload = {
            "event": event,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "elapsed_seconds": round(perf_counter() - self._started_at, 3),
            "details": details,
        }
        self._events.append(payload)
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def summary(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        chunks: list[dict[str, Any]] = []
        schema_fields: list[str] = []
        quality_warnings: list[str] = []
        merge_summary: dict[str, Any] | None = None
        model_call_count = 0
        max_model_calls: int | None = None
        budget_exhausted = False
        for event in self._events:
            name = str(event.get("event", ""))
            counts[name] = counts.get(name, 0) + 1
            details = event.get("details", {})
            if isinstance(details, dict):
                raw_model_call_count = details.get("model_call_count")
                if isinstance(raw_model_call_count, int):
                    model_call_count = max(model_call_count, raw_model_call_count)
                raw_max_model_calls = details.get("max_model_calls")
                if isinstance(raw_max_model_calls, int):
                    max_model_calls = raw_max_model_calls
                budget_exhausted = budget_exhausted or bool(details.get("budget_exhausted"))
            if name == "schema_done" and isinstance(details, dict):
                schema_fields = [str(field) for field in details.get("fields", [])]
            if name in {
                "chunk_done",
                "chunk_failed",
                "chunk_repair_done",
                "chunk_repair_failed",
            } and isinstance(details, dict):
                chunks.append(
                    {
                        "event": name,
                        "chunk_index": details.get("chunk_index"),
                        "line_start": details.get("line_start"),
                        "line_end": details.get("line_end"),
                        "input_line_count": details.get("input_line_count"),
                        "fact_count": details.get("fact_count"),
                        "elapsed_seconds": details.get("elapsed_seconds"),
                        "error": details.get("error"),
                        "attempt_number": details.get("attempt_number"),
                        "repair_attempt_number": details.get("repair_attempt_number"),
                        "model_call_count": details.get("model_call_count"),
                        "remaining_model_calls": details.get("remaining_model_calls"),
                    }
                )
            if name == "merge_done" and isinstance(details, dict):
                merge_summary = {
                    "fact_count": details.get("fact_count"),
                    "merged_row_count": details.get("merged_row_count"),
                    "column_non_null_counts": details.get("column_non_null_counts"),
                    "field_conflict_count": details.get("field_conflict_count"),
                    "dropped_empty_fact_count": details.get("dropped_empty_fact_count"),
                    "missing_entity_key_count": details.get("missing_entity_key_count"),
                }
            if name == "quality_warning" and isinstance(details, dict):
                quality_warnings.append(str(details.get("warning", "")))
        return {
            "log_file": self.relative_log_file,
            "event_count": len(self._events),
            "events": counts,
            "elapsed_seconds": round(perf_counter() - self._started_at, 3),
            "schema_fields": schema_fields,
            "chunk_events": chunks,
            "merge_summary": merge_summary,
            "quality_warnings": quality_warnings,
            "failed_chunk_count": counts.get("chunk_failed", 0),
            "repair_count": counts.get("chunk_repair_done", 0),
            "failed_repair_count": counts.get("chunk_repair_failed", 0),
            "model_call_count": model_call_count,
            "max_model_calls": max_model_calls,
            "budget_exhausted": budget_exhausted,
        }


def _short_error(exc: Exception | str, *, limit: int = 300) -> str:
    text = str(exc).replace("\n", " ").strip()
    return text if len(text) <= limit else text[:limit] + "..."


def _message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content)


def _last_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()

    # Strip markdown code fences before attempting JSON parse.
    # Model responses often wrap JSON in ```json / ``` blocks.
    fence_pattern = re.compile(r"```(?:json)?\s*\n?(.*?)\n?```", re.DOTALL)
    fence_match = fence_pattern.search(text)
    if fence_match:
        text = fence_match.group(1)

    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            candidate, end = decoder.raw_decode(stripped)
            if isinstance(candidate, dict) and not stripped[end:].strip():
                return candidate
        except json.JSONDecodeError:
            pass
    parsed: dict[str, Any] | None = None
    # Recognised top-level keys shared with doc_structure._last_json_object.
    _root_keys = frozenset({"facts", "entity_key_fields", "blocks"})
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if not isinstance(candidate, dict):
            continue
        if _root_keys & candidate.keys():
            parsed = candidate
        elif parsed is None:
            parsed = candidate
    if parsed is None:
        raise ValueError("Model response did not contain a JSON object.")
    return parsed


def _normalize_field_name(value: str) -> str:
    return value.strip().strip("`").strip()


def _normalize_key_value(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _split_record_lines(text: str) -> list[dict[str, Any]]:
    lines: list[dict[str, Any]] = []
    for index, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if stripped:
            lines.append({"line_id": index, "text": stripped})
    return lines



def _load_doc_structure_blocks(
    workspace_root: Path,
    *,
    path: str,
) -> tuple[list[dict[str, Any]], str | None, dict[str, Any]]:
    doc_stem = PurePosixPath(path).stem
    structure = load_doc_structure(workspace_root, doc_stem)
    if structure is None:
        return [], None, {}
    blocks = [block for block in structure.get("blocks", []) if isinstance(block, dict)]
    structure_hash = None
    raw_metadata = structure.get("structure")
    metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
    if isinstance(raw_metadata, dict):
        value = raw_metadata.get("structure_hash")
        structure_hash = None if value is None else str(value)
    return blocks, structure_hash, metadata


def _prepend_effective_field(
    requested_fields: list[str] | None,
    primary_key_field: str | None,
) -> list[str] | None:
    if requested_fields is None:
        return None
    fields: list[str] = []
    seen: set[str] = set()
    for value in [primary_key_field, *requested_fields]:
        if value in (None, ""):
            continue
        field = _normalize_field_name(str(value))
        if not field:
            continue
        key = field.lower()
        if key in seen:
            # Prefer the requested field's casing over the primary_key_field's
            # casing when they differ only in case.  The document's actual field
            # names are a better match than the knowledge.md normalisation.
            for idx in range(len(fields)):
                if fields[idx].lower() == key and fields[idx] != field:
                    fields[idx] = field
            continue
        seen.add(key)
        fields.append(field)
    return fields


def _block_candidate_fields(block: dict[str, Any]) -> set[str]:
    raw_fields = block.get("candidate_fields", [])
    if not isinstance(raw_fields, list):
        return set()
    return {_normalize_field_name(str(field)) for field in raw_fields if str(field).strip()}


def _auto_select_blocks_for_fields(
    blocks: list[dict[str, Any]],
    *,
    requested_fields: list[str] | None,
) -> dict[str, Any]:
    field_to_blocks: dict[str, list[str]] = {}
    selected_ids: set[str] = set()
    requested = [_normalize_field_name(field) for field in requested_fields or []]

    if requested:
        for field in requested:
            matches = [
                str(block.get("block_id"))
                for block in blocks
                if field in _block_candidate_fields(block)
            ]
            field_to_blocks[field] = matches
            selected_ids.update(matches)
        missing_fields = [field for field, matches in field_to_blocks.items() if not matches]
    else:
        missing_fields = []
        for block in blocks:
            block_id = str(block.get("block_id") or "").strip()
            if block_id and _block_candidate_fields(block):
                selected_ids.add(block_id)

    selected_blocks = [
        block for block in blocks if str(block.get("block_id") or "").strip() in selected_ids
    ]
    selected_block_ids = [str(block.get("block_id")) for block in selected_blocks]
    return {
        "selected_blocks": selected_blocks,
        "selected_block_ids": selected_block_ids,
        "field_to_blocks": field_to_blocks,
        "missing_fields": missing_fields,
    }


def _selected_line_context(
    lines: list[dict[str, Any]],
    *,
    selected_blocks: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]], list[tuple[int, int]]]:
    ranges: list[tuple[int, int]] = []
    line_context: dict[int, dict[str, Any]] = {}
    for block in selected_blocks:
        start = int(block["data_start_line"])
        end = int(block["data_end_line"])
        ranges.append((start, end))
        for line_id in range(start, end + 1):
            line_context[line_id] = block
    if not ranges:
        return lines, line_context, []
    selected: list[dict[str, Any]] = []
    for line in lines:
        line_id = int(line["line_id"])
        in_range = any(start <= line_id <= end for start, end in ranges)
        if not in_range:
            continue
        copied = dict(line)
        block = line_context.get(line_id)
        if block is not None:
            copied["block_id"] = block.get("block_id")
            copied["section_scope"] = block.get("section_scope") or block.get("scope_id")
            copied["scope_name"] = block.get("scope_name")
            copied["candidate_fields"] = block.get("candidate_fields", [])
        selected.append(copied)
    return selected, line_context, ranges


def _sample_lines_for_plan(
    lines: list[dict[str, Any]], target_count: int = 30
) -> list[dict[str, Any]]:
    """Sample lines evenly across blocks so the LLM can see block→field relationships.

    Returns a list of block summaries, each containing block metadata and
    evenly-spaced sample lines from that block.
    """
    # Group lines by block_id (or "_ungrouped" when no block attribution exists)
    groups: dict[str, dict[str, Any]] = {}
    ordered_block_ids: list[str] = []
    for line in lines:
        block_id = str(line.get("block_id") or "_ungrouped")
        if block_id not in groups:
            groups[block_id] = {
                "block_id": line.get("block_id"),
                "scope_id": line.get("section_scope") or line.get("scope_id"),
                "scope_name": line.get("scope_name"),
                "candidate_fields": line.get("candidate_fields", []),
                "all_lines": [],
            }
            ordered_block_ids.append(block_id)
        groups[block_id]["all_lines"].append(line)

    total_lines = len(lines)
    if total_lines <= target_count:
        # Small enough — include every line grouped by block.
        return [
            {
                "block_id": groups[bid]["block_id"],
                "scope_id": groups[bid]["scope_id"],
                "scope_name": groups[bid]["scope_name"],
                "candidate_fields": groups[bid]["candidate_fields"],
                "sample_lines": [
                    {"line_id": int(ln["line_id"]), "text": str(ln.get("text", ""))}
                    for ln in groups[bid]["all_lines"]
                ],
            }
            for bid in ordered_block_ids
        ]

    # Allocate sample quota per block proportional to its share of total lines.
    block_count = len(ordered_block_ids)
    allocated: list[int] = []
    for i, bid in enumerate(ordered_block_ids):
        share = len(groups[bid]["all_lines"]) / total_lines
        quota = max(1, round(target_count * share))
        allocated.append(quota)

    # Adjust so total matches target_count (may be off due to rounding).
    while sum(allocated) < target_count:
        # Give extra to the largest block that is under its line count.
        for i in sorted(
            range(block_count),
            key=lambda j: len(groups[ordered_block_ids[j]]["all_lines"]),
            reverse=True,
        ):
            bid = ordered_block_ids[i]
            if allocated[i] < len(groups[bid]["all_lines"]):
                allocated[i] += 1
                if sum(allocated) >= target_count:
                    break
    while sum(allocated) > target_count:
        # Shrink the largest-quota block that still has room above 1.
        for i in sorted(range(block_count), key=lambda j: allocated[j], reverse=True):
            if allocated[i] > 1:
                allocated[i] -= 1
                if sum(allocated) <= target_count:
                    break

    # Evenly sample within each block.
    result: list[dict[str, Any]] = []
    for i, bid in enumerate(ordered_block_ids):
        block_lines = groups[bid]["all_lines"]
        k = min(allocated[i], len(block_lines))
        if k <= 0:
            continue
        if k == 1:
            indices = [0]
        else:
            step = (len(block_lines) - 1) / (k - 1) if k > 1 else 1.0
            indices = [min(len(block_lines) - 1, round(j * step)) for j in range(k)]
        result.append(
            {
                "block_id": groups[bid]["block_id"],
                "scope_id": groups[bid]["scope_id"],
                "scope_name": groups[bid]["scope_name"],
                "candidate_fields": groups[bid]["candidate_fields"],
                "sample_lines": [
                    {"line_id": int(block_lines[idx]["line_id"]), "text": str(block_lines[idx].get("text", ""))}
                    for idx in indices
                ],
            }
        )
    return result


def _lines_from_sizes(lines: list[dict[str, Any]], sizes: list[int]) -> list[list[dict[str, Any]]]:
    chunks: list[list[dict[str, Any]]] = []
    offset = 0
    for size in sizes:
        if size <= 0:
            continue
        chunks.append(lines[offset:offset + size])
        offset += size
    return chunks


def _balanced_chunk_sizes(line_count: int, chunk_count: int) -> list[int]:
    base_size, extra = divmod(line_count, chunk_count)
    return [base_size + (1 if index < extra else 0) for index in range(chunk_count)]


def _chunk_lines(
    lines: list[dict[str, Any]],
    available_calls: int,
    *,
    config: StructuredDocToolConfig | None = None,
) -> ChunkPlan:
    structured_doc_config = config or StructuredDocToolConfig()
    if not lines:
        return ChunkPlan(chunks=[], chunking_reason="empty_input", repair_budget=max(0, available_calls))
    if available_calls <= 0:
        raise ValueError("No model calls remain for fact extraction.")

    line_count = len(lines)
    min_chunk_lines = structured_doc_config.min_chunk_lines
    max_chunk_lines = structured_doc_config.max_chunk_lines
    required_chunks = max(1, math.ceil(line_count / max_chunk_lines))
    if required_chunks > available_calls:
        raise ValueError(
            "Selected document lines require more extraction chunks than the model-call "
            f"budget allows: selected_line_count={line_count}, "
            f"required_chunks={required_chunks}, available_model_calls={available_calls}, "
            f"max_chunk_lines={max_chunk_lines}."
        )

    if line_count <= max_chunk_lines:
        chunk_count = 1
    else:
        preferred_chunks = max(1, available_calls // 2) if available_calls > 1 else 1
        max_useful_chunks = max(1, math.ceil(line_count / min_chunk_lines))
        chunk_count = min(max(preferred_chunks, required_chunks), max_useful_chunks, line_count)
        if chunk_count < required_chunks:
            chunk_count = required_chunks

    if chunk_count == 1:
        sizes = [line_count]
        reason = "single_chunk"
    elif line_count < chunk_count * min_chunk_lines:
        sizes = [min_chunk_lines] * (chunk_count - 1)
        sizes.append(line_count - sum(sizes))
        reason = "min_chunk_with_remainder"
    else:
        sizes = _balanced_chunk_sizes(line_count, chunk_count)
        reason = "balanced_by_line_count"

    if any(size > max_chunk_lines for size in sizes):
        raise ValueError(
            "Chunk planner produced an oversized chunk: "
            f"chunk_sizes={sizes}, max_chunk_lines={max_chunk_lines}."
        )
    repair_budget = max(0, available_calls - len(sizes))
    return ChunkPlan(
        chunks=_lines_from_sizes(lines, sizes),
        chunking_reason=reason,
        repair_budget=repair_budget,
    )


_FIELD_VALUE_TYPES = {"string", "integer", "number"}
_UNIT_SOURCES = {"knowledge", "document_dominant", "none"}


def _parse_field_value_specs(
    raw_specs: Any,
    *,
    target_fields: list[str],
) -> dict[str, dict[str, str | None]]:
    if not isinstance(raw_specs, dict):
        raise ValueError("Extraction plan must contain field_value_specs object.")
    expected_fields = set(target_fields)
    provided_fields = {str(field) for field in raw_specs}
    missing = sorted(expected_fields - provided_fields)
    unexpected = sorted(provided_fields - expected_fields)
    if missing or unexpected:
        raise ValueError(
            "field_value_specs must contain exactly the target fields: "
            f"missing={missing}, unexpected={unexpected}."
        )

    specs: dict[str, dict[str, str | None]] = {}
    for field in target_fields:
        raw_spec = raw_specs.get(field)
        if not isinstance(raw_spec, dict):
            raise ValueError(f"field_value_specs[{field!r}] must be an object.")
        value_type = str(raw_spec.get("value_type") or "").strip().lower()
        if value_type not in _FIELD_VALUE_TYPES:
            raise ValueError(
                f"field_value_specs[{field!r}].value_type must be one of "
                f"{sorted(_FIELD_VALUE_TYPES)}."
            )

        unit_source = str(raw_spec.get("unit_source") or "").strip().lower()
        if unit_source not in _UNIT_SOURCES:
            raise ValueError(
                f"field_value_specs[{field!r}].unit_source must be one of "
                f"{sorted(_UNIT_SOURCES)}."
            )

        # --- String fields must not have a unit ---
        if value_type == "string":
            unit_source = "none"

        # --- Resolve expected_source_units ---
        expected_source_units: list[str] | None = None
        raw_expected = raw_spec.get("expected_source_units")
        if isinstance(raw_expected, list):
            expected_source_units = [str(u).strip() for u in raw_expected if str(u).strip()]

        # --- Normalization rule ---
        normalization_rule = str(raw_spec.get("normalization_rule") or "").strip()
        if not normalization_rule:
            normalization_rule = generate_normalization_rule(expected_source_units)

        specs[field] = {
            "value_type": value_type,
            "unit_source": unit_source,
            "normalization_rule": normalization_rule,
            "expected_source_units": expected_source_units,
        }
    return specs


def _validate_fact_value_types(
    values: dict[str, Any],
    *,
    line_id: int,
    field_value_specs: dict[str, dict[str, str | None]],
) -> None:
    for field, value in values.items():
        if value is None:
            continue
        spec = field_value_specs.get(field)
        if spec is None:
            raise ValueError(f"Fact {line_id} has no value contract for field {field!r}.")
        value_type = spec["value_type"]
        # Accept intermediate {raw_number, raw_unit} format for numeric fields
        if isinstance(value, dict) and "raw_number" in value:
            if value_type == "string":
                raise ValueError(
                    f"Fact {line_id} field {field!r} is string type but got "
                    f"raw_number/raw_unit dict {value!r}."
                )
            raw_number = value["raw_number"]
            raw_unit = value.get("raw_unit")
            valid_number = (
                isinstance(raw_number, (int, float))
                and not isinstance(raw_number, bool)
                and math.isfinite(raw_number)
            )
            if not valid_number:
                raise ValueError(
                    f"Fact {line_id} field {field!r} raw_number must be a finite number; "
                    f"got {raw_number!r}."
                )
            if raw_unit is not None and not isinstance(raw_unit, str):
                raise ValueError(
                    f"Fact {line_id} field {field!r} raw_unit must be a string or null; "
                    f"got {raw_unit!r}."
                )
            continue
        # Legacy / normalised bare value path
        if value_type == "string":
            valid = isinstance(value, str)
        elif value_type == "integer":
            valid = isinstance(value, int) and not isinstance(value, bool)
        else:
            valid = (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
            )
        if not valid:
            canonical_unit = spec.get("canonical_unit")
            unit_context = f" in canonical unit {canonical_unit!r}" if canonical_unit else ""
            raise ValueError(
                f"Fact {line_id} field {field!r} must be {value_type}{unit_context}; "
                f"got {value!r}."
            )


def _build_extraction_plan(
    model: Any,
    *,
    target_table: str,
    requested_fields: list[str] | None,
    knowledge_text: str,
    sample_blocks: list[dict[str, Any]],
    candidate_field_names: list[str] | None = None,
) -> tuple[ExtractionPlan, int]:
    # Target fields are determined programmatically from requested_fields
    # (or candidate_field_names when requested_fields is None).
    # The LLM only needs to design entity_key_fields, fallback, merge_grain,
    # field_hints, and a type/unit contract for the fixed target fields.
    if requested_fields:
        target_fields = [
            {"name": field, "description": ""} for field in requested_fields
        ]
    elif candidate_field_names:
        target_fields = [
            {"name": field, "description": ""} for field in candidate_field_names
        ]
    else:
        raise ValueError("No target fields available for extraction plan.")

    field_names_for_prompt = [f["name"] for f in target_fields]

    prompt = {
        "target_table": target_table,
        "target_fields": field_names_for_prompt,
        "plan_version": PLAN_VERSION,
        "instruction": (
            "Design the entity-key and merge strategy for extracting these "
            "target_fields from a markdown/text document. Return only JSON with: "
            "entity_key_fields=[\"field1\", \"field2\"], fallback_entity_key, "
            "merge_grain, field_hints={field:hint}, and field_value_specs={field:{value_type,unit_source,expected_source_units,normalization_rule}}. "
            "entity_key_fields MUST be a flat array of plain strings, e.g. [\"record_id\"]. "
            "Do NOT wrap each entry in an object like {\"name\": \"...\"}; use raw strings only. "
            "entity_key_fields are internal merge keys and do not need to be final output "
            "columns. When target fields for the same entity are spread across selected "
            "blocks/sections, choose source-document entity anchors visible across those "
            "blocks, such as archive_id/doc id/record id/item id. Do not choose a final "
            "output field as the primary entity key if that field is visible only in one "
            "kind of block. Put answer fields in values; put cross-line/cross-section "
            "linking identifiers in entity_key. If no stable natural entity key is "
            "visible, use line_id fallback. "
            "Use the knowledge document as the authority for target field semantics, "
            "units, and value shape. Field hints must describe only the value that "
            "belongs in that field. Never broaden a field definition beyond what the "
            "knowledge document states. "
            "field_value_specs MUST contain exactly one object for every target field. "
            "Each object MUST have value_type (one of string, integer, number), "
            "unit_source (knowledge, document_dominant, or none), "
            "expected_source_units (an array of unit strings likely to appear in the "
            "document for this field, e.g. ['元','万元','百万元'] for currency fields, "
            "['%'] for percentage fields), and normalization_rule (a brief human-readable "
            "description). "
            "IMPORTANT: All values will be automatically normalised to the BASE UNIT "
            "(unit=1) by a deterministic Python normaliser. Currency fields are always "
            "normalised to 元, percentage fields to decimal (15% → 0.15). You do NOT "
            "need to specify a target unit — the normaliser always converts to unit=1. "
            "expected_source_units should list the unit strings that the extraction step "
            "should look for in the document context. "
            "Use the knowledge document to identify expected source units. "
            "If neither knowledge nor document establishes a unit, use "
            "expected_source_units=[] and unit_source=none. "
            "Identifier-like fields, including digit-only identifiers, must use "
            "value_type=string, unit_source=none, expected_source_units=[]. "
            "Do not broaden a scalar target field into an object or combine adjacent "
            "metrics into one field unless the knowledge document explicitly defines "
            "that field as a composite value."
        ),
        "knowledge": knowledge_text,
        "entity_key_rules": [
            "entity_key_fields are for deterministic merging only; they may be non-output source identifiers.",
            "Prefer a source identifier that appears in every selected block type needed for the requested fields.",
            "Reject keys that are visible only in identity blocks, metric blocks, or any single block family.",
            "Do not put target answer values into entity_key; target fields belong in values.",
            "candidate_fields metadata limits values extraction only; it must not prevent extracting entity_key.",
        ],
        "sample_blocks": sample_blocks,
    }
    response = invoke_model_with_retries(
        model,
        [
            SystemMessage(content="You design compact machine-readable fact extraction plans."),
            HumanMessage(content=json.dumps(prompt, ensure_ascii=False)),
        ],
    )
    parsed = _last_json_object(_message_text(response))
    raw_key_fields = parsed.get("entity_key_fields", [])
    entity_key_fields: list[str] = []
    if isinstance(raw_key_fields, list):
        for value in raw_key_fields:
            if isinstance(value, dict):
                name = value.get("name")
                if isinstance(name, str) and name.strip():
                    normalized = _normalize_field_name(name)
                    if normalized:
                        entity_key_fields.append(normalized)
            elif isinstance(value, str):
                normalized = _normalize_field_name(value)
                if normalized:
                    entity_key_fields.append(normalized)
    field_hints_raw = parsed.get("field_hints", {})
    field_hints = {
        str(key): str(value)
        for key, value in field_hints_raw.items()
    } if isinstance(field_hints_raw, dict) else {}
    field_value_specs = _parse_field_value_specs(
        parsed.get("field_value_specs"),
        target_fields=field_names_for_prompt,
    )
    fallback = _normalize_field_name(str(parsed.get("fallback_entity_key") or LINE_ID_KEY))
    if not fallback:
        fallback = LINE_ID_KEY
    return (
        ExtractionPlan(
            target_fields=target_fields,
            entity_key_fields=entity_key_fields,
            fallback_entity_key=fallback,
            merge_grain=str(parsed.get("merge_grain") or ""),
            field_hints=field_hints,
            field_value_specs=field_value_specs,
        ),
        1,
    )


def _extract_chunk_facts(
    model: Any,
    *,
    target_table: str,
    plan: ExtractionPlan,
    lines: list[dict[str, Any]],
    repair_error: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "target_table": target_table,
        "target_fields": plan.target_fields,
        "entity_key_fields": plan.entity_key_fields,
        "fallback_entity_key": plan.fallback_entity_key,
        "merge_grain": plan.merge_grain,
        "field_hints": plan.field_hints,
        "field_value_specs": plan.field_value_specs,
        "lines": lines,
        "rules": [
            "Extract facts visible in each source line; do not infer missing values.",
            "A source line may contribute only some target fields for an entity.",
            "Return only facts with at least one visible target field or useful entity key.",
            "Use null only when a visible fact explicitly has no value; omit absent fields.",
            # --- Unit handling: extract raw number + unit; Python normaliser does the math ---
            "Extract the exact numeric value as written in the source line. Remove thousand-separator commas (1,817,970 → 1817970) but do NOT apply any unit conversion or scale adjustment. Output the raw number as a plain float/int.",
            "CRITICAL — raw_unit rules: raw_unit MUST only be set when the source line contains an EXPLICIT unit label directly attached to or immediately following the numeric value. Explicit labels include: a unit word after the number ('275.9168 百万元', '3.2 亿元'), a percent sign ('1.5%'), or a Chinese numeral suffix that encodes the unit ('六百九十七万'). When the line has a bare number with no unit label (e.g. '21,019,650.00 in DepositsWithCentralBank', 'total reserves of 1,986,140'), set raw_unit to null — do NOT guess or infer the unit from section headers, knowledge documents, field names, or the magnitude of the number. A bare number followed by 'in <fieldname>' or 'of <fieldname>' has NO unit label — raw_unit must be null.",
            "expected_source_units is a hint about what unit labels MAY appear in the document. Use it to recognise unit strings when they ARE present, but do NOT force raw_unit to match one of them when no label is visible.",
            "Convert Chinese numerals to Arabic numbers, placing the scale word in raw_unit. Example: '六百九十七万元' → raw_number: 697, raw_unit: '万元'. '一点五亿元' → raw_number: 1.5, raw_unit: '亿元'. '三千' → raw_number: 3000, raw_unit: null (no explicit unit label, only a scale prefix integrated into the numeral).",
            "When the text includes 约/大约/约等 fuzzy modifiers, extract the stated numeric value without the modifier.",
            "For percentage values: '1.5%' → raw_number: 1.5, raw_unit: '%'. If a line says 'ratio of 15 percent' → raw_number: 15, raw_unit: '%'.",
            "If a line contains multiple numeric values for different fields, determine raw_unit independently for each field based on that specific value's immediate context.",
            # --- Entity key rules ---
            "If a line contains earlier mistaken values and a final confirmed/corrected value, choose the final confirmed/corrected value.",
            "Use entity_key to link facts about the same entity across different lines/sections. You MUST use ONLY the exact field names listed in entity_key_fields as the entity_key keys. When entity_key_fields is [\"record_linkage_id\"], use {\"record_linkage_id\": \"...\"} — never invent names like \"archive_id\" or \"item_ref\". Do not add extra keys beyond entity_key_fields.",
            "entity_key is an internal merge key and may be a source identifier that is not one of the target output fields.",
            "Even when a block only allows one target value field, still extract the same cross-block entity anchor into entity_key when it is visible.",
            "Do not put target answer values into entity_key; target fields belong in values.",
            "When a line includes block_id/section_scope/candidate_fields metadata, extract only values compatible with that block's candidate_fields.",
            "candidate_fields restrict values only; they do not restrict entity_key extraction.",
            "When a source line labels an entity with a Chinese/English prefix followed by a numeric identifier (e.g. 档案 29, 记录 30, 条目 48, Archive 5, Record 12), extract ONLY the numeric/alphanumeric identifier as the entity_key value — strip the label prefix. For example: 档案 29 → entity_key value '29', 记录 30 → '30', 条目 1052 → '1052'.",
            "CRITICAL: entity_key key names must exactly match entity_key_fields. Never change the key name across chunks — use the same entity_key_fields key names in every chunk for the same document.",
        ],
        "output_format": (
            "Return only JSON: {\"facts\":[{\"line_id\":1,\"is_fact\":true,"
            "\"entity_key\":{\"record_linkage_id\":\"...\"},\"values\":{\"numeric_field\":{\"raw_number\":275.9168,\"raw_unit\":\"百万元\"},\"string_field\":\"plain_value\"},"
            "\"evidence_fields\":[\"numeric_field\"]}]}."
            " Numeric target fields with a unit MUST use the {raw_number, raw_unit} object form. "
            "String fields and entity_key fields MUST use plain values (no wrapper)."
        ),
    }
    if repair_error:
        payload["repair_error"] = repair_error
    response = invoke_model_with_retries(
        model,
        [
            SystemMessage(
                content=(
                    "You are a structured fact extractor for markdown/text documents. "
                    "Your output MUST be a single JSON object — no explanations, no markdown "
                    "wrappers, no natural language. "
                    "Return exactly: {\"facts\":[{\"line_id\":<int>,\"is_fact\":true|false,"
                    "\"entity_key\":{...},\"values\":{...},\"evidence_fields\":[...]}]}. "
                    "If no facts are visible, return {\"facts\":[]}."
                )
            ),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
        ],
    )
    return _last_json_object(_message_text(response))


def _validate_facts(
    payload: dict[str, Any],
    *,
    fields: list[str],
    expected_line_ids: set[int],
    fallback_entity_key: str,
    line_context: dict[int, dict[str, Any]] | None = None,
    entity_key_fields: list[str] | None = None,
    field_value_specs: dict[str, dict[str, str | None]] | None = None,
) -> list[dict[str, Any]]:
    raw_facts = payload.get("facts")
    if not isinstance(raw_facts, list):
        raise ValueError("Chunk response must contain facts list.")
    allowed_fields = set(fields)
    facts: list[dict[str, Any]] = []
    for item in raw_facts:
        if not isinstance(item, dict):
            raise ValueError("Each fact item must be an object.")
        line_id = item.get("line_id")
        if not isinstance(line_id, int) or line_id not in expected_line_ids:
            raise ValueError(f"Invalid or unexpected line_id: {line_id!r}.")
        if item.get("is_fact") is False:
            continue
        values = item.get("values", {})
        if values is None:
            values = {}
        if not isinstance(values, dict):
            raise ValueError(f"Fact {line_id} must contain values object.")
        unknown = set(values) - allowed_fields
        if unknown:
            raise ValueError(f"Fact {line_id} returned unknown fields: {sorted(unknown)}.")
        _validate_fact_value_types(
            values,
            line_id=line_id,
            field_value_specs=field_value_specs or {},
        )
        entity_key = item.get("entity_key", {})
        if entity_key is None:
            entity_key = {}
        if not isinstance(entity_key, dict):
            raise ValueError(f"Fact {line_id} must contain entity_key object.")
        normalized_entity_key = {
            _normalize_field_name(str(key)): str(value).strip()
            for key, value in entity_key.items()
            if _normalize_field_name(str(key)) and _normalize_key_value(value) is not None
        }
        if entity_key_fields:
            allowed_entity_keys = set(entity_key_fields)
            unexpected = [k for k in normalized_entity_key if k not in allowed_entity_keys]
            for key in unexpected:
                del normalized_entity_key[key]
        if not normalized_entity_key:
            normalized_entity_key = {fallback_entity_key: str(line_id)}
        evidence_fields = item.get("evidence_fields", [])
        if not isinstance(evidence_fields, list):
            evidence_fields = []
        facts.append(
            {
                "line_id": line_id,
                "entity_key": normalized_entity_key,
                "values": values,
                "evidence_fields": [str(field) for field in evidence_fields],
                "block_id": (line_context or {}).get(line_id, {}).get("block_id"),
                "section_scope": (
                    (line_context or {}).get(line_id, {}).get("section_scope")
                    or (line_context or {}).get(line_id, {}).get("scope_id")
                ),
            }
        )
    return facts


def _filter_facts_by_scope(
    facts: list[dict[str, Any]],
    *,
    line_context: dict[int, dict[str, Any]],
    plan_field_names: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not line_context:
        return facts, []
    plan_set: set[str] = set(plan_field_names or [])
    filtered_facts: list[dict[str, Any]] = []
    filtered_values: list[dict[str, Any]] = []
    for fact in facts:
        line_id = int(fact["line_id"])
        block = line_context.get(line_id)
        values = dict(fact.get("values", {}))
        if not block:
            filtered_facts.append(fact)
            continue
        candidate_fields = {
            str(field)
            for field in block.get("candidate_fields", [])
            if str(field).strip()
        }
        if not candidate_fields:
            if values:
                allowed = sorted(plan_set or values)
                disallowed = sorted(field for field in values if field not in plan_set)
                if disallowed:
                    for field in disallowed:
                        values.pop(field, None)
                filtered_values.append(
                    {
                        "line_id": line_id,
                        "block_id": block.get("block_id"),
                        "section_scope": block.get("section_scope") or block.get("scope_id"),
                        "filtered_fields": disallowed,
                        "allowed_fields": allowed,
                        "reason": "block has no candidate_fields",
                    }
                )
            else:
                values = {}
        else:
            allowed_set = candidate_fields
            disallowed = sorted(field for field in values if field not in allowed_set)
            if disallowed:
                for field in disallowed:
                    values.pop(field, None)
                filtered_values.append(
                    {
                        "line_id": line_id,
                        "block_id": block.get("block_id"),
                        "section_scope": block.get("section_scope") or block.get("scope_id"),
                        "filtered_fields": disallowed,
                        "allowed_fields": sorted(allowed_set),
                    }
                )
        copied = dict(fact)
        copied["values"] = values
        copied["evidence_fields"] = [
            field for field in fact.get("evidence_fields", []) if field in values
        ]
        filtered_facts.append(copied)
    return filtered_facts, filtered_values


def _facts_for_log(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    logged: list[dict[str, Any]] = []
    for fact in facts:
        item = {
            "line_id": fact.get("line_id"),
            "entity_key": fact.get("entity_key", {}),
            "values": fact.get("values", {}),
            "evidence_fields": fact.get("evidence_fields", []),
        }
        if fact.get("block_id") is not None:
            item["block_id"] = fact.get("block_id")
        if fact.get("section_scope") is not None:
            item["section_scope"] = fact.get("section_scope")
        logged.append(item)
    return logged


def _fact_scope_counts(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[tuple[str, str], int] = {}
    for fact in facts:
        key = (
            str(fact.get("block_id") or ""),
            str(fact.get("section_scope") or ""),
        )
        counts[key] = counts.get(key, 0) + 1
    return [
        {"block_id": block_id or None, "section_scope": scope or None, "fact_count": count}
        for (block_id, scope), count in sorted(counts.items())
    ]


def _entity_sort_key(entity_key: dict[str, str], entity_key_fields: list[str]) -> tuple[tuple[str, str], ...]:
    if entity_key_fields:
        ordered = [
            (field, entity_key[field])
            for field in entity_key_fields
            if field in entity_key
        ]
        remaining = sorted((key, value) for key, value in entity_key.items() if key not in entity_key_fields)
        return tuple(ordered + remaining)
    return tuple(sorted(entity_key.items()))


def _normalize_extracted_units(
    facts: list[dict[str, Any]],
    logger: Any,
) -> tuple[list[dict[str, Any]], int]:
    """Convert raw {raw_number, raw_unit} values to bare numbers in unit=1.

    Called after all chunks have been extracted and before merging.
    Uses the deterministic unit normaliser — target is always the base unit (unit=1).
    The LLM never does math.
    """
    if not facts:
        return facts, 0
    normalised, conversion_log = normalize_facts(facts)
    conversion_count = len(conversion_log)
    if conversion_log:
        logger.emit(
            "unit_normalization_done",
            conversion_count=conversion_count,
            conversions=conversion_log[:100],  # cap for log size
        )
    # Log any warnings from failed conversions
    warnings = [c for c in conversion_log if "warning" in c]
    for w in warnings[:20]:
        logger.emit("unit_normalization_warning", **w)
    return normalised, conversion_count


def _merge_facts(
    facts: list[dict[str, Any]],
    *,
    columns: list[str],
    entity_key_fields: list[str],
) -> MergeResult:
    grouped: dict[tuple[tuple[str, str], ...], dict[str, Any]] = {}
    entity_order: list[tuple[tuple[str, str], ...]] = []
    field_conflicts: list[dict[str, Any]] = []
    dropped_empty_fact_count = 0
    missing_entity_key_count = 0
    allowed_columns = set(columns)
    entity_key_set = set(entity_key_fields)
    non_key_columns = [c for c in allowed_columns if c not in entity_key_set]
    for fact in sorted(facts, key=lambda item: int(item["line_id"])):
        values = {
            key: value
            for key, value in fact.get("values", {}).items()
            if key in allowed_columns and value is not None
        }
        if not values:
            dropped_empty_fact_count += 1
            continue
        # When the extraction plan defines both entity-key fields and
        # non-key output columns, a fact that contributes only entity-key
        # values (e.g. from a narrative summary block that mentions an
        # entity ID) would merge into a row of nulls.  Drop it early.
        if non_key_columns:
            if not {k for k in values if k not in entity_key_set}:
                dropped_empty_fact_count += 1
                continue
        entity_key = fact.get("entity_key", {})
        if not isinstance(entity_key, dict) or not entity_key:
            entity_key = {LINE_ID_KEY: str(fact["line_id"])}
            missing_entity_key_count += 1
        normalized_entity_key = {str(key): str(value) for key, value in entity_key.items()}
        key = _entity_sort_key(normalized_entity_key, entity_key_fields)
        if key not in grouped:
            grouped[key] = {"entity_key": normalized_entity_key, "values": {}, "line_ids": []}
            entity_order.append(key)
        row_values = grouped[key]["values"]
        grouped[key]["line_ids"].append(fact["line_id"])
        for field, value in values.items():
            old_value = row_values.get(field)
            if old_value is not None and old_value != value:
                field_conflicts.append(
                    {
                        "entity_key": normalized_entity_key,
                        "field": field,
                        "old_value": old_value,
                        "new_value": value,
                        "line_id": fact["line_id"],
                    }
                )
                continue
            if old_value is None:
                row_values[field] = value
    rows = [
        [grouped[key]["values"].get(column) for column in columns]
        for key in entity_order
    ]
    column_non_null_counts = {
        column: sum(row[index] is not None for row in rows)
        for index, column in enumerate(columns)
    }
    return MergeResult(
        rows=rows,
        fact_count=len(facts),
        merged_row_count=len(rows),
        dropped_empty_fact_count=dropped_empty_fact_count,
        column_non_null_counts=column_non_null_counts,
        field_conflicts=field_conflicts,
        missing_entity_key_count=missing_entity_key_count,
    )


def _quality_warnings(
    *,
    columns: list[str],
    merge_result: MergeResult,
    entity_key_fields: list[str],
) -> list[str]:
    warnings: list[str] = []
    if not entity_key_fields:
        warnings.append("No natural entity key was identified; line_id fallback was used.")
    if merge_result.merged_row_count == 0:
        warnings.append("No valid merged rows were produced from extracted facts.")
    for column in columns:
        non_null = merge_result.column_non_null_counts.get(column, 0)
        if merge_result.merged_row_count >= 5 and non_null == 0:
            warnings.append(f"Column {column!r} is null for every merged row.")
    if merge_result.missing_entity_key_count:
        warnings.append(
            f"{merge_result.missing_entity_key_count} facts used fallback entity keys."
        )
    return warnings


def _catalog_table_names(catalog: dict[str, Any]) -> set[str]:
    return {str(table.get("table", "")) for table in iter_logical_tables(catalog) if table.get("table")}


def _registered_table_name(path: str, catalog: dict[str, Any]) -> str:
    stem = PurePosixPath(path).stem
    existing = {name.lower() for name in _catalog_table_names(catalog)}
    if stem.lower() in existing:
        return f"{stem}_extracted"
    return stem


def _manifest_path(workspace_root: Path) -> Path:
    return workspace_root / GENERATED_STRUCTURED_DOC_DIR / STRUCTURED_DOC_MANIFEST


def _load_manifest(workspace_root: Path) -> dict[str, Any]:
    path = _manifest_path(workspace_root)
    if not path.exists():
        return {"tables": []}
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"tables": []}
    if not isinstance(manifest, dict) or not isinstance(manifest.get("tables"), list):
        return {"tables": []}
    return manifest


def _write_manifest(workspace_root: Path, entry: dict[str, Any]) -> None:
    path = _manifest_path(workspace_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(workspace_root)
    tables = [
        item for item in manifest.get("tables", [])
        if isinstance(item, dict) and item.get("registered_table") != entry["registered_table"]
    ]
    tables.append(entry)
    manifest["tables"] = tables
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _cache_key(
    *,
    path: str,
    knowledge_path: str,
    target_table: str,
    fields: list[str],
    doc_hash: str,
    knowledge_hash: str,
    plan_version: int,
    block_ids: list[str] | None = None,
    line_ranges: list[tuple[int, int]] | None = None,
    doc_structure_hash: str | None = None,
    structure_version: int | None = None,
    chunking_version: int | None = None,
    chunking_config: dict[str, Any] | None = None,
    target_fields: list[str] | None = None,
    entity_key_fields: list[str] | None = None,
) -> str:
    payload = {
        "path": path,
        "knowledge_path": knowledge_path,
        "target_table": target_table,
        "fields": fields,
        "doc_hash": doc_hash,
        "knowledge_hash": knowledge_hash,
        "plan_version": plan_version,
        "block_ids": block_ids or [],
        "line_ranges": line_ranges or [],
        "doc_structure_hash": doc_structure_hash,
        "structure_version": structure_version,
        "chunking_version": chunking_version,
        "chunking_config": chunking_config or {},
        "target_fields": target_fields or [],
        "entity_key_fields": entity_key_fields or [],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _chunking_config_for_cache(config: StructuredDocToolConfig) -> dict[str, Any]:
    return {
        "min_chunk_lines": config.min_chunk_lines,
        "max_chunk_lines": config.max_chunk_lines,
        "max_selected_lines_for_llm_extraction": config.max_selected_lines_for_llm_extraction,
        "default_max_model_calls": config.default_max_model_calls,
        "hard_max_model_calls": config.hard_max_model_calls,
        "llm": {
            "temperature": config.llm.temperature,
            "top_p": config.llm.top_p,
            "repetition_penalty": config.llm.repetition_penalty,
        },
    }


def _bind_extraction_model(
    model: Any, config: StructuredDocToolConfig, *, repair: bool = False
) -> Any:
    """Apply the extraction-only sampling profile without mutating the shared model.

    When ``repair`` is True, use a higher temperature to help the model escape
    a deterministic failure path (e.g. consistently skipping JSON output).
    """

    bind = getattr(model, "bind", None)
    if not callable(bind):
        return model
    return bind(
        temperature=config.llm.repair_temperature if repair else config.llm.temperature,
        top_p=config.llm.top_p,
        max_tokens=config.llm.max_tokens,
        extra_body={"repetition_penalty": config.llm.repetition_penalty},
    )


def estimate_structured_doc_chunk_count(
    *,
    task: PublicTask,
    workspace: TaskContextWorkspace,
    catalog: dict[str, Any],
    path: str,
    knowledge_path: str = "knowledge.md",
    target_table: str | None = None,
    fields: list[str] | None = None,
    max_model_calls: int | None = None,
    structured_doc_config: StructuredDocToolConfig | None = None,
) -> StructuredDocChunkEstimate:
    del catalog, knowledge_path, target_table
    config = structured_doc_config or StructuredDocToolConfig()
    if max_model_calls is None:
        max_model_calls = config.default_max_model_calls
    try:
        max_calls = min(max(1, int(max_model_calls)), config.hard_max_model_calls)
        normalized_path = normalize_context_relative_path(path)
        resolved_doc = resolve_context_path(task, normalized_path)
        doc_text = resolved_doc.read_text(encoding="utf-8", errors="replace")
        requested_fields = [_normalize_field_name(field) for field in fields] if fields else None
        workspace_root = workspace.materialize()
        structure_blocks, _doc_structure_hash, doc_structure_metadata = _load_doc_structure_blocks(
            workspace_root,
            path=normalized_path,
        )
        if not structure_blocks:
            return StructuredDocChunkEstimate(
                priority_chunk_count=None,
                selected_line_count=None,
                priority_source="unknown:missing_doc_structure",
                error="missing_doc_structure",
            )
        primary_key_field = None
        raw_primary_key = doc_structure_metadata.get("primary_key_field")
        if raw_primary_key not in (None, ""):
            primary_key_field = _normalize_field_name(str(raw_primary_key))
        effective_requested_fields = _prepend_effective_field(requested_fields, primary_key_field)
        auto_block_selection = _auto_select_blocks_for_fields(
            structure_blocks,
            requested_fields=effective_requested_fields,
        )
        if auto_block_selection["missing_fields"]:
            return StructuredDocChunkEstimate(
                priority_chunk_count=None,
                selected_line_count=None,
                priority_source="unknown:missing_fields",
                error="missing_fields",
            )
        selected_blocks = list(auto_block_selection["selected_blocks"])
        if not selected_blocks:
            return StructuredDocChunkEstimate(
                priority_chunk_count=None,
                selected_line_count=None,
                priority_source="unknown:no_selected_blocks",
                error="no_selected_blocks",
            )
        all_lines = _split_record_lines(doc_text)
        lines, _line_context, _selected_ranges = _selected_line_context(
            all_lines,
            selected_blocks=selected_blocks,
        )
        selected_line_count = len(lines)
        if selected_line_count > config.max_selected_lines_for_llm_extraction:
            return StructuredDocChunkEstimate(
                priority_chunk_count=None,
                selected_line_count=selected_line_count,
                priority_source="unknown:input_too_large",
                error="input_too_large",
            )
        chunk_plan = _chunk_lines(lines, max_calls - 1, config=config)
        return StructuredDocChunkEstimate(
            priority_chunk_count=len(chunk_plan.chunks),
            selected_line_count=selected_line_count,
            priority_source="estimated_chunk_count",
        )
    except Exception as exc:  # noqa: BLE001
        return StructuredDocChunkEstimate(
            priority_chunk_count=None,
            selected_line_count=None,
            priority_source="unknown:estimate_failed",
            error=str(exc),
        )


def _read_cached_extraction(
    workspace_root: Path,
    *,
    preplan_cache_key: str,
    logger: StructuredDocLogger,
    visible_dir: Path | None = None,
) -> StructuredDocExtraction | None:
    manifest = _load_manifest(workspace_root)
    for entry in manifest.get("tables", []):
        if not isinstance(entry, dict) or entry.get("preplan_cache_key") != preplan_cache_key:
            continue
        file_name = str(entry.get("file") or "")
        data_path = workspace_root / GENERATED_STRUCTURED_DOC_DIR / file_name
        if not data_path.exists():
            continue
        columns = [str(column) for column in entry.get("columns", [])]
        rows: list[list[Any]] = []
        for line in data_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            rows.append([record.get(column) for column in columns])
        metadata = {
            "registered_table": entry.get("registered_table"),
            "source_path": entry.get("source_path"),
            "knowledge_path": entry.get("knowledge_path"),
            "target_table": entry.get("target_table"),
            "requested_fields": entry.get("requested_fields"),
            "effective_requested_fields": entry.get("effective_requested_fields"),
            "primary_key_field": entry.get("primary_key_field"),
            "row_count": len(rows),
            "model_call_count": 0,
            "cache_hit": True,
            "log_file": logger.relative_log_file,
            "visible_data_file": (
                f"{VISIBLE_STRUCTURED_DOC_DIR}/{file_name}" if visible_dir is not None else None
            ),
            "visible_manifest_file": (
                f"{VISIBLE_STRUCTURED_DOC_DIR}/{STRUCTURED_DOC_MANIFEST}"
                if visible_dir is not None
                else None
            ),
            "plan_version": entry.get("plan_version"),
            "selected_blocks": entry.get("selected_blocks", []),
            "selected_line_ranges": entry.get("selected_line_ranges", []),
            "auto_selected_blocks": entry.get("auto_selected_blocks", False),
            "auto_block_selection": entry.get("auto_block_selection"),
            "block_scopes": entry.get("block_scopes", []),
            "doc_structure_hash": entry.get("doc_structure_hash"),
            "structure_version": entry.get("structure_version"),
            "scope_filtered_fact_count": entry.get("scope_filtered_fact_count", 0),
            "entity_key_fields": entry.get("entity_key_fields", []),
            "merge_grain": entry.get("merge_grain"),
            "field_value_specs": entry.get("field_value_specs", {}),
            "fact_count": entry.get("fact_count"),
            "merged_row_count": entry.get("merged_row_count"),
            "column_non_null_counts": entry.get("column_non_null_counts", {}),
            "field_conflict_count": entry.get("field_conflict_count", 0),
        }
        logger.emit(
            "cache_hit",
            cache_key=entry.get("cache_key"),
            preplan_cache_key=preplan_cache_key,
            registered_table=entry.get("registered_table"),
            row_count=len(rows),
        )
        if visible_dir is not None:
            visible_dir.mkdir(parents=True, exist_ok=True)
            (visible_dir / file_name).write_text(
                data_path.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            (visible_dir / STRUCTURED_DOC_MANIFEST).write_text(
                json.dumps(_load_manifest(workspace_root), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        metadata["log_summary"] = logger.summary()
        return StructuredDocExtraction(columns=columns, rows=rows, metadata=metadata)
    return None


def _persist_extraction(
    workspace_root: Path,
    *,
    entry: dict[str, Any],
    columns: list[str],
    rows: list[list[Any]],
    visible_dir: Path | None = None,
) -> None:
    output_dir = workspace_root / GENERATED_STRUCTURED_DOC_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / str(entry["file"])
    with data_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            record = {column: value for column, value in zip(columns, row, strict=False)}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    _write_manifest(workspace_root, entry)
    if visible_dir is not None:
        visible_dir.mkdir(parents=True, exist_ok=True)
        visible_data_path = visible_dir / str(entry["file"])
        visible_data_path.write_text(data_path.read_text(encoding="utf-8"), encoding="utf-8")
        manifest_payload = _load_manifest(workspace_root)
        (visible_dir / STRUCTURED_DOC_MANIFEST).write_text(
            json.dumps(manifest_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def extract_structured_doc(
    *,
    task: PublicTask,
    workspace: TaskContextWorkspace,
    catalog: dict[str, Any],
    model: Any,
    path: str,
    knowledge_path: str = "knowledge.md",
    target_table: str | None = None,
    fields: list[str] | None = None,
    max_model_calls: int | None = None,
    structured_doc_config: StructuredDocToolConfig | None = None,
    log_dir: Path | None = None,
) -> StructuredDocExtraction:
    if model is None:
        raise ValueError("extract_structured_doc requires an available model.")
    config = structured_doc_config or StructuredDocToolConfig()
    extraction_model = _bind_extraction_model(model, config)
    repair_model = _bind_extraction_model(model, config, repair=True)
    if max_model_calls is None:
        max_model_calls = config.default_max_model_calls
    max_calls = min(max(1, int(max_model_calls)), config.hard_max_model_calls)
    normalized_path = normalize_context_relative_path(path)
    normalized_knowledge_path = normalize_context_relative_path(knowledge_path)
    resolved_doc = resolve_context_path(task, normalized_path)
    resolved_knowledge = resolve_context_path(task, normalized_knowledge_path)
    doc_text = resolved_doc.read_text(encoding="utf-8", errors="replace")
    knowledge_text = resolved_knowledge.read_text(encoding="utf-8", errors="replace")
    target = target_table or PurePosixPath(normalized_path).stem
    requested_fields = [_normalize_field_name(field) for field in fields] if fields else None

    doc_hash = hashlib.sha256(doc_text.encode("utf-8")).hexdigest()
    knowledge_hash = hashlib.sha256(knowledge_text.encode("utf-8")).hexdigest()
    workspace_root = workspace.materialize()
    registered_table = _registered_table_name(normalized_path, catalog)
    logger = StructuredDocLogger(workspace_root, registered_table, log_dir=log_dir)
    visible_structured_doc_dir = (
        log_dir / VISIBLE_STRUCTURED_DOC_DIR if log_dir is not None else None
    )
    logger.emit(
        "start",
        path=normalized_path,
        knowledge_path=normalized_knowledge_path,
        target_table=target,
        requested_fields=requested_fields,
        max_model_calls=max_calls,
        registered_table=registered_table,
        plan_version=PLAN_VERSION,
        chunking_version=CHUNKING_VERSION,
        structured_doc_config=_chunking_config_for_cache(config),
        llm_sampling={
            "temperature": config.llm.temperature,
            "top_p": config.llm.top_p,
            "repetition_penalty": config.llm.repetition_penalty,
        },
    )

    structure_blocks, doc_structure_hash, doc_structure_metadata = _load_doc_structure_blocks(
        workspace_root,
        path=normalized_path,
    )
    if not structure_blocks:
        error = (
            "Document structure cache is required before extraction. "
            "Call inspect_doc_structure for this path first."
        )
        logger.emit(
            "missing_doc_structure",
            path=normalized_path,
            recommendation="Call inspect_doc_structure(path, target_table, fields) before extract_structured_doc.",
        )
        logger.emit("failed", error=error, error_code="missing_doc_structure", model_call_count=0)
        raise StructuredDocExtractionError(
            error,
            log_summary=logger.summary(),
            details={
                "error_code": "missing_doc_structure",
                "recommendation": "Call inspect_doc_structure for this document before extract_structured_doc.",
            },
        )
    logger.emit(
        "doc_structure_cache_loaded",
        path=normalized_path,
        block_count=len(structure_blocks),
        doc_structure_hash=doc_structure_hash,
    )
    primary_key_field = None
    raw_primary_key = doc_structure_metadata.get("primary_key_field")
    if raw_primary_key not in (None, ""):
        primary_key_field = _normalize_field_name(str(raw_primary_key))
    effective_requested_fields = _prepend_effective_field(requested_fields, primary_key_field)
    field_names_for_cache = effective_requested_fields or []
    if effective_requested_fields != requested_fields:
        logger.emit(
            "primary_key_field_applied",
            primary_key_field=primary_key_field,
            requested_fields=requested_fields,
            effective_requested_fields=effective_requested_fields,
        )
    auto_block_selection = _auto_select_blocks_for_fields(
        structure_blocks,
        requested_fields=effective_requested_fields,
    )
    selected_blocks = list(auto_block_selection["selected_blocks"])
    selected_block_ids = list(auto_block_selection["selected_block_ids"])
    logger.emit(
        "auto_block_select_done",
        requested_fields=requested_fields,
        effective_requested_fields=effective_requested_fields,
        primary_key_field=primary_key_field,
        selected_block_ids=selected_block_ids,
        field_to_blocks=auto_block_selection["field_to_blocks"],
        missing_fields=auto_block_selection["missing_fields"],
        selected_block_count=len(selected_blocks),
    )
    if auto_block_selection["missing_fields"]:
        missing_fields = list(auto_block_selection["missing_fields"])
        available_fields = sorted(
            {
                field
                for block in structure_blocks
                for field in (block.get("candidate_fields", []) or [])
                if field
            }
        )
        error = (
            f"Fields {missing_fields} not found in document. "
            f"Available candidate fields: {available_fields}. "
            "Retry with corrected field names."
        )
        logger.emit(
            "failed",
            error=error,
            error_code="missing_fields",
            missing_fields=missing_fields,
            available_fields=available_fields,
            model_call_count=0,
        )
        raise StructuredDocExtractionError(
            error,
            log_summary=logger.summary(),
            details={
                "error_code": "missing_fields",
                "missing_fields": missing_fields,
                "available_fields": available_fields,
                "field_to_blocks": auto_block_selection["field_to_blocks"],
                "recommendation": "Retry with field names from available_fields.",
            },
        )
    if not selected_blocks:
        error = (
            "Document structure cache did not select any blocks for extraction. "
            "Call inspect_doc_structure to review candidate_fields."
        )
        logger.emit("failed", error=error, error_code="no_selected_blocks", model_call_count=0)
        raise StructuredDocExtractionError(
            error,
            log_summary=logger.summary(),
            details={
                "error_code": "no_selected_blocks",
                "field_to_blocks": auto_block_selection["field_to_blocks"],
            },
        )

    all_lines = _split_record_lines(doc_text)
    lines, line_context, selected_ranges = _selected_line_context(
        all_lines,
        selected_blocks=selected_blocks,
    )
    preplan_key = _cache_key(
        path=normalized_path,
        knowledge_path=normalized_knowledge_path,
        target_table=target,
        fields=field_names_for_cache,
        doc_hash=doc_hash,
        knowledge_hash=knowledge_hash,
        plan_version=PLAN_VERSION,
        block_ids=selected_block_ids,
        line_ranges=selected_ranges,
        doc_structure_hash=doc_structure_hash,
        structure_version=STRUCTURE_VERSION if selected_block_ids else None,
        chunking_version=CHUNKING_VERSION,
        chunking_config=_chunking_config_for_cache(config),
    )
    if selected_ranges:
        logger.emit(
            "structure_selected",
            selected_blocks=[
                {
                    "block_id": block.get("block_id"),
                    "scope_id": block.get("scope_id") or block.get("section_scope"),
                    "scope_name": block.get("scope_name"),
                    "candidate_fields": block.get("candidate_fields", []),
                    "data_start_line": block.get("data_start_line"),
                    "data_end_line": block.get("data_end_line"),
                }
                for block in selected_blocks
            ],
            selected_line_ranges=selected_ranges,
            input_line_count=len(all_lines),
            selected_line_count=len(lines),
            auto_selected_blocks=True,
        )
    cached = _read_cached_extraction(
        workspace_root,
        preplan_cache_key=preplan_key,
        logger=logger,
        visible_dir=visible_structured_doc_dir,
    )
    if cached is not None:
        return cached
    logger.emit("cache_miss", preplan_cache_key=preplan_key, registered_table=registered_table)

    available_calls_before_plan = max_calls - 1
    logger.emit(
        "input_size_check",
        selected_line_count=len(lines),
        max_selected_lines_for_llm_extraction=config.max_selected_lines_for_llm_extraction,
        max_model_calls=max_calls,
        available_extraction_calls=available_calls_before_plan,
        min_chunk_lines=config.min_chunk_lines,
        max_chunk_lines=config.max_chunk_lines,
    )
    if len(lines) > config.max_selected_lines_for_llm_extraction:
        error = (
            "Selected document range is too large for LLM structured extraction: "
            f"selected_line_count={len(lines)}, "
            f"max_selected_lines_for_llm_extraction={config.max_selected_lines_for_llm_extraction}. "
            "Use inspect_doc_structure with narrower block_ids/line_ranges, or use "
            "read_doc/search_doc plus execute_python for regex/programmatic parsing."
        )
        logger.emit(
            "input_too_large",
            selected_line_count=len(lines),
            max_selected_lines_for_llm_extraction=config.max_selected_lines_for_llm_extraction,
            recommendation=(
                "Narrow block_ids/line_ranges, or parse the document with execute_python "
                "using regex/programmatic logic."
            ),
        )
        logger.emit("failed", error=error, model_call_count=0)
        raise StructuredDocExtractionError(error, log_summary=logger.summary())
    try:
        _chunk_lines(lines, available_calls_before_plan, config=config)
    except Exception as exc:
        error = (
            "Selected document range cannot fit the configured chunk/model-call budget "
            f"before schema planning: {exc}"
        )
        logger.emit("failed", error=_short_error(error), model_call_count=0)
        raise StructuredDocExtractionError(error, log_summary=logger.summary()) from exc

    model_calls = 0
    plan_start = perf_counter()
    logger.emit(
        "schema_start",
        target_table=target,
        requested_fields=requested_fields,
        effective_requested_fields=effective_requested_fields,
        primary_key_field=primary_key_field,
    )
    collected_candidate_fields = sorted(
        {
            field
            for block in selected_blocks
            for field in (block.get("candidate_fields", []) or [])
            if field
        }
    )
    plan = None
    schema_error = None
    try:
        model_calls += 1
        plan, _ = _build_extraction_plan(
            extraction_model,
            target_table=target,
            requested_fields=effective_requested_fields,
            knowledge_text=knowledge_text,
            sample_blocks=_sample_lines_for_plan(lines),
            candidate_field_names=collected_candidate_fields,
        )
    except Exception as exc:
        schema_error = exc
    if plan is None and model_calls < max_calls:
        try:
            model_calls += 1
            repair_plan, _ = _build_extraction_plan(
                repair_model,
                target_table=target,
                requested_fields=effective_requested_fields,
                knowledge_text=knowledge_text,
                sample_blocks=_sample_lines_for_plan(lines),
                candidate_field_names=collected_candidate_fields,
            )
            plan = repair_plan
            schema_error = None
            logger.emit(
                "schema_repair_done",
                elapsed_seconds=round(perf_counter() - plan_start, 3),
                model_call_count=model_calls,
            )
        except Exception as repair_exc:
            schema_error = repair_exc
    if plan is None:
        logger.emit(
            "schema_failed",
            error=_short_error(schema_error),
            model_call_count=model_calls,
        )
        logger.emit("failed", error=_short_error(schema_error), model_call_count=model_calls)
        raise StructuredDocExtractionError(
            str(schema_error), log_summary=logger.summary()
        ) from schema_error
    columns = [field["name"] for field in plan.target_fields]
    if not columns:
        raise ValueError(f"No fields were found for target table {target!r}.")
    logger.emit(
        "schema_done",
        field_count=len(columns),
        fields=columns,
        entity_key_fields=plan.entity_key_fields,
        fallback_entity_key=plan.fallback_entity_key,
        merge_grain=plan.merge_grain,
        field_value_specs=plan.field_value_specs,
        elapsed_seconds=round(perf_counter() - plan_start, 3),
        model_call_count=model_calls,
    )
    logger.emit(
        "extraction_plan",
        target_fields=columns,
        entity_key_fields=plan.entity_key_fields,
        fallback_entity_key=plan.fallback_entity_key,
        merge_grain=plan.merge_grain,
        field_hints=plan.field_hints,
        field_value_specs=plan.field_value_specs,
    )
    if model_calls >= max_calls:
        raise ValueError("Model call budget exhausted before fact extraction.")

    available_calls = max_calls - model_calls
    chunk_plan = _chunk_lines(lines, available_calls, config=config)
    chunks = chunk_plan.chunks
    chunk_sizes = [len(chunk) for chunk in chunks]
    logger.emit(
        "chunk_plan",
        input_line_count=len(lines),
        chunk_count=len(chunks),
        chunk_sizes=chunk_sizes,
        available_model_calls=available_calls,
        reserved_repair_budget=chunk_plan.repair_budget,
        repair_budget=chunk_plan.repair_budget,
        min_chunk_lines=config.min_chunk_lines,
        max_chunk_lines=config.max_chunk_lines,
        chunking_version=CHUNKING_VERSION,
        chunking_reason=chunk_plan.chunking_reason,
    )
    facts: list[dict[str, Any]] = []
    errors: list[str] = []
    scope_filtered_fact_count = 0
    for chunk_index, chunk in enumerate(chunks, start=1):
        if model_calls >= max_calls:
            exc = ValueError("Model call budget exhausted during fact extraction.")
            logger.emit(
                "failed",
                error=_short_error(exc),
                model_call_count=model_calls,
                max_model_calls=max_calls,
                budget_exhausted=True,
            )
            raise StructuredDocExtractionError(str(exc), log_summary=logger.summary()) from exc
        expected_line_ids = {int(line["line_id"]) for line in chunk}
        line_ids = sorted(expected_line_ids)
        line_start = line_ids[0] if line_ids else None
        line_end = line_ids[-1] if line_ids else None
        logger.emit(
            "chunk_start",
            chunk_index=chunk_index,
            line_start=line_start,
            line_end=line_end,
            input_line_count=len(chunk),
            model_call_count=model_calls,
        )
        attempt_number = 0
        repair_attempt_number = 0
        last_error: Exception | None = None
        while model_calls < max_calls:
            attempt_number += 1
            is_repair = attempt_number > 1
            if is_repair:
                repair_attempt_number += 1
                logger.emit(
                    "chunk_repair_start",
                    chunk_index=chunk_index,
                    line_start=line_start,
                    line_end=line_end,
                    input_line_count=len(chunk),
                    attempt_number=attempt_number,
                    repair_attempt_number=repair_attempt_number,
                    model_call_count=model_calls,
                    remaining_model_calls=max_calls - model_calls,
                )
            attempt_start = perf_counter()
            # Consume budget before invoking the model so malformed responses and
            # validation failures are counted as real requests.
            model_calls += 1
            try:
                payload = _extract_chunk_facts(
                    repair_model if is_repair else extraction_model,
                    target_table=target,
                    plan=plan,
                    lines=chunk,
                    repair_error=str(last_error) if is_repair and last_error else None,
                )
                chunk_facts = _validate_facts(
                    payload,
                    fields=columns,
                    expected_line_ids=expected_line_ids,
                    fallback_entity_key=plan.fallback_entity_key or LINE_ID_KEY,
                    line_context=line_context,
                    entity_key_fields=plan.entity_key_fields,
                    field_value_specs=plan.field_value_specs,
                )
                chunk_facts, filtered_values = _filter_facts_by_scope(
                    chunk_facts,
                    line_context=line_context,
                    plan_field_names=columns,
                )
                scope_filtered_fact_count += len(filtered_values)
                facts.extend(chunk_facts)
                for filtered in filtered_values[:20]:
                    logger.emit("scope_filtered_fact", **filtered)
                logger.emit(
                    "chunk_repair_done" if is_repair else "chunk_done",
                    chunk_index=chunk_index,
                    line_start=line_start,
                    line_end=line_end,
                    input_line_count=len(chunk),
                    fact_count=len(chunk_facts),
                    scope_filtered_fact_count=len(filtered_values),
                    extracted_facts=_facts_for_log(chunk_facts),
                    elapsed_seconds=round(perf_counter() - attempt_start, 3),
                    attempt_number=attempt_number,
                    repair_attempt_number=repair_attempt_number,
                    model_call_count=model_calls,
                    remaining_model_calls=max_calls - model_calls,
                )
                break
            except Exception as exc:
                last_error = exc
                errors.append(str(exc))
                logger.emit(
                    "chunk_repair_failed" if is_repair else "chunk_failed",
                    chunk_index=chunk_index,
                    line_start=line_start,
                    line_end=line_end,
                    input_line_count=len(chunk),
                    elapsed_seconds=round(perf_counter() - attempt_start, 3),
                    attempt_number=attempt_number,
                    repair_attempt_number=repair_attempt_number,
                    model_call_count=model_calls,
                    remaining_model_calls=max_calls - model_calls,
                    error=_short_error(exc),
                )
        else:
            error = (
                "Model call budget exhausted while repairing chunk "
                f"{chunk_index} after {attempt_number} attempts "
                f"(max_model_calls={max_calls}): {last_error}"
            )
            logger.emit(
                "failed",
                error=_short_error(error),
                chunk_index=chunk_index,
                attempt_number=attempt_number,
                repair_attempt_number=repair_attempt_number,
                model_call_count=model_calls,
                max_model_calls=max_calls,
                budget_exhausted=True,
            )
            raise StructuredDocExtractionError(error, log_summary=logger.summary()) from last_error

    # --- Unit normalisation: convert {raw_number, raw_unit} → bare numbers in unit=1 ---
    norm_start = perf_counter()
    facts, unit_conversion_count = _normalize_extracted_units(
        facts,
        logger=logger,
    )
    if unit_conversion_count:
        logger.emit(
            "unit_normalization_summary",
            elapsed_seconds=round(perf_counter() - norm_start, 3),
            conversion_count=unit_conversion_count,
        )

    merge_result = _merge_facts(facts, columns=columns, entity_key_fields=plan.entity_key_fields)
    logger.emit(
        "merge_done",
        fact_count=merge_result.fact_count,
        merged_row_count=merge_result.merged_row_count,
        column_non_null_counts=merge_result.column_non_null_counts,
        field_conflict_count=len(merge_result.field_conflicts),
        field_conflicts=merge_result.field_conflicts[:50],
        dropped_empty_fact_count=merge_result.dropped_empty_fact_count,
        missing_entity_key_count=merge_result.missing_entity_key_count,
        scope_filtered_fact_count=scope_filtered_fact_count,
        block_scope_counts=_fact_scope_counts(facts),
    )
    warnings = _quality_warnings(columns=columns, merge_result=merge_result, entity_key_fields=plan.entity_key_fields)
    for warning in warnings:
        logger.emit("quality_warning", warning=warning)
    if not merge_result.rows:
        error = "No valid merged rows were produced from extracted facts."
        logger.emit("failed", error=error, model_call_count=model_calls)
        raise StructuredDocExtractionError(error, log_summary=logger.summary())

    file_name = f"{registered_table}.jsonl"
    log_file = logger.relative_log_file
    full_cache_key = _cache_key(
        path=normalized_path,
        knowledge_path=normalized_knowledge_path,
        target_table=target,
        fields=field_names_for_cache,
        doc_hash=doc_hash,
        knowledge_hash=knowledge_hash,
        plan_version=PLAN_VERSION,
        block_ids=selected_block_ids,
        line_ranges=selected_ranges,
        doc_structure_hash=doc_structure_hash,
        structure_version=STRUCTURE_VERSION if selected_block_ids else None,
        chunking_version=CHUNKING_VERSION,
        chunking_config=_chunking_config_for_cache(config),
        target_fields=columns,
        entity_key_fields=plan.entity_key_fields,
    )
    block_scopes = [
        {
            "block_id": block.get("block_id"),
            "scope_id": block.get("scope_id") or block.get("section_scope"),
            "scope_name": block.get("scope_name"),
            "candidate_fields": block.get("candidate_fields", []),
            "data_start_line": block.get("data_start_line"),
            "data_end_line": block.get("data_end_line"),
        }
        for block in selected_blocks
    ]
    entry = {
        "source_path": normalized_path,
        "knowledge_path": normalized_knowledge_path,
        "target_table": target,
        "registered_table": registered_table,
        "columns": columns,
        "requested_fields": requested_fields,
        "effective_requested_fields": effective_requested_fields,
        "primary_key_field": primary_key_field,
        "row_count": len(merge_result.rows),
        "doc_hash": doc_hash,
        "knowledge_hash": knowledge_hash,
        "preplan_cache_key": preplan_key,
        "cache_key": full_cache_key,
        "file": file_name,
        "log_file": log_file,
        "visible_data_file": (
            f"{VISIBLE_STRUCTURED_DOC_DIR}/{file_name}"
            if visible_structured_doc_dir is not None
            else None
        ),
        "visible_manifest_file": (
            f"{VISIBLE_STRUCTURED_DOC_DIR}/{STRUCTURED_DOC_MANIFEST}"
            if visible_structured_doc_dir is not None
            else None
        ),
        "plan_version": PLAN_VERSION,
        "selected_blocks": selected_block_ids,
        "selected_line_ranges": selected_ranges,
        "auto_selected_blocks": True,
        "auto_block_selection": auto_block_selection,
        "block_scopes": block_scopes,
        "doc_structure_hash": doc_structure_hash,
        "structure_version": STRUCTURE_VERSION if selected_block_ids else None,
        "chunking_version": CHUNKING_VERSION,
        "chunking_config": _chunking_config_for_cache(config),
        "scope_filtered_fact_count": scope_filtered_fact_count,
        "entity_key_fields": plan.entity_key_fields,
        "merge_grain": plan.merge_grain,
        "field_value_specs": plan.field_value_specs,
        "fact_count": merge_result.fact_count,
        "merged_row_count": merge_result.merged_row_count,
        "column_non_null_counts": merge_result.column_non_null_counts,
        "field_conflict_count": len(merge_result.field_conflicts),
    }
    _persist_extraction(
        workspace_root,
        entry=entry,
        columns=columns,
        rows=merge_result.rows,
        visible_dir=visible_structured_doc_dir,
    )
    logger.emit(
        "persist_done",
        data_file=f"{GENERATED_STRUCTURED_DOC_DIR}/{file_name}",
        manifest_file=f"{GENERATED_STRUCTURED_DOC_DIR}/{STRUCTURED_DOC_MANIFEST}",
        visible_data_file=(
            f"{VISIBLE_STRUCTURED_DOC_DIR}/{file_name}"
            if visible_structured_doc_dir is not None
            else None
        ),
        visible_manifest_file=(
            f"{VISIBLE_STRUCTURED_DOC_DIR}/{STRUCTURED_DOC_MANIFEST}"
            if visible_structured_doc_dir is not None
            else None
        ),
        log_file=log_file,
        registered_table=registered_table,
        row_count=len(merge_result.rows),
    )
    logger.emit("done", row_count=len(merge_result.rows), model_call_count=model_calls)
    metadata = {
        "registered_table": registered_table,
        "source_path": normalized_path,
        "knowledge_path": normalized_knowledge_path,
        "target_table": target,
        "input_line_count": len(lines),
        "requested_fields": requested_fields,
        "effective_requested_fields": effective_requested_fields,
        "primary_key_field": primary_key_field,
        "row_count": len(merge_result.rows),
        "skipped_line_count": max(0, len(lines) - merge_result.fact_count),
        "model_call_count": model_calls,
        "chunk_count": len(chunks),
        "cache_hit": False,
        "errors": errors,
        "log_file": log_file,
        "visible_data_file": (
            f"{VISIBLE_STRUCTURED_DOC_DIR}/{file_name}"
            if visible_structured_doc_dir is not None
            else None
        ),
        "visible_manifest_file": (
            f"{VISIBLE_STRUCTURED_DOC_DIR}/{STRUCTURED_DOC_MANIFEST}"
            if visible_structured_doc_dir is not None
            else None
        ),
        "plan_version": PLAN_VERSION,
        "selected_blocks": selected_block_ids,
        "selected_line_ranges": selected_ranges,
        "auto_selected_blocks": True,
        "auto_block_selection": auto_block_selection,
        "block_scopes": block_scopes,
        "doc_structure_hash": doc_structure_hash,
        "structure_version": STRUCTURE_VERSION if selected_block_ids else None,
        "chunking_version": CHUNKING_VERSION,
        "chunking_config": _chunking_config_for_cache(config),
        "scope_filtered_fact_count": scope_filtered_fact_count,
        "entity_key_fields": plan.entity_key_fields,
        "merge_grain": plan.merge_grain,
        "field_value_specs": plan.field_value_specs,
        "fact_count": merge_result.fact_count,
        "merged_row_count": merge_result.merged_row_count,
        "column_non_null_counts": merge_result.column_non_null_counts,
        "field_conflict_count": len(merge_result.field_conflicts),
        "quality_warnings": warnings,
    }
    metadata["log_summary"] = logger.summary()
    return StructuredDocExtraction(columns=columns, rows=merge_result.rows, metadata=metadata)
