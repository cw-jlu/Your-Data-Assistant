from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from time import perf_counter
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.config import StructuredDocToolConfig
from data_agent_baseline.model_retry import invoke_model_with_retries
from data_agent_baseline.tools.filesystem import normalize_context_relative_path, resolve_context_path
from data_agent_baseline.tools.python_exec import TaskContextWorkspace


GENERATED_DOC_STRUCTURE_DIR = ".generated/doc_structure"
VISIBLE_DOC_STRUCTURE_DIR = "doc_structure"
STRUCTURE_VERSION = 8


@dataclass(frozen=True, slots=True)
class DocStructure:
    blocks: list[dict[str, Any]]
    metadata: dict[str, Any]


class DocStructureLogger:
    def __init__(self, doc_stem: str, *, log_dir: Path | None = None) -> None:
        self._started_at = perf_counter()
        self._events: list[dict[str, Any]] = []
        self.relative_log_file: str | None = None
        self.path: Path | None = None
        if log_dir is not None:
            visible_dir = log_dir / VISIBLE_DOC_STRUCTURE_DIR
            visible_dir.mkdir(parents=True, exist_ok=True)
            self.relative_log_file = f"{VISIBLE_DOC_STRUCTURE_DIR}/doc_structure_{doc_stem}.log.jsonl"
            self.path = log_dir / self.relative_log_file
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
        for event in self._events:
            name = str(event.get("event", ""))
            counts[name] = counts.get(name, 0) + 1
        return {
            "log_file": self.relative_log_file,
            "event_count": len(self._events),
            "events": counts,
            "elapsed_seconds": round(perf_counter() - self._started_at, 3),
        }


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


# Keys that identify a top-level doc-structure / extraction response dict.
# When the model output is truncated the scanning fallback may encounter
# inner block/fact objects first; preferring dicts that carry a recognised
# root key keeps _last_json_object from silently returning an inner object.
_JSON_ROOT_KEYS = frozenset({"blocks", "facts", "entity_key_fields"})


def _last_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            candidate, end = decoder.raw_decode(stripped)
            if isinstance(candidate, dict) and not stripped[end:].strip():
                return candidate
        except json.JSONDecodeError:
            pass
    parsed: dict[str, Any] | None = None
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if not isinstance(candidate, dict):
            continue
        if _JSON_ROOT_KEYS & candidate.keys():
            parsed = candidate
        elif parsed is None:
            parsed = candidate
    if parsed is None:
        raise ValueError("Model response did not contain a JSON object.")
    return parsed


def _split_non_empty_lines(text: str) -> list[dict[str, Any]]:
    lines: list[dict[str, Any]] = []
    for index, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if stripped:
            lines.append({"line_id": index, "text": stripped})
    return lines


def _numeric_tokens(text: str) -> list[str]:
    return re.findall(r"\d+(?:[.,]\d+)*", text)


def _boundary_reason(text: str) -> str | None:
    if re.match(r"^#{1,6}\s+", text):
        return "markdown_heading"
    digit_count = sum(ch.isdigit() for ch in text)
    if digit_count == 0:
        return "no_digit_text"
    tokens = _numeric_tokens(text)
    if tokens and max(len(token.replace(".", "").replace(",", "")) for token in tokens) >= 3:
        return None
    if digit_count / max(len(text), 1) > 0.03:
        return None
    first_digit_index = next((index for index, char in enumerate(text) if char.isdigit()), -1)
    if 0 <= first_digit_index < 20:
        return None
    return "sparse_numeric_transition"


def _is_boundary_line(text: str, *, first_line: bool = False) -> bool:
    _ = first_line
    return _boundary_reason(text) is not None


def _candidate_blocks(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not lines:
        return []
    boundary_reasons: dict[int, str] = {}
    for index, line in enumerate(lines):
        reason = _boundary_reason(str(line["text"]))
        if reason is not None:
            boundary_reasons[index] = reason
    boundary_indexes = sorted(boundary_reasons)
    if 0 not in boundary_indexes:
        boundary_indexes.insert(0, 0)
        boundary_reasons[0] = "document_start"
    blocks: list[dict[str, Any]] = []
    for block_number, boundary_index in enumerate(boundary_indexes, start=1):
        next_boundary_index = (
            boundary_indexes[block_number]
            if block_number < len(boundary_indexes)
            else len(lines)
        )
        title_line = int(lines[boundary_index]["line_id"])
        data_lines = lines[boundary_index + 1:next_boundary_index]
        if not data_lines:
            data_start_line = title_line
            data_end_line = title_line
            sample_lines = [lines[boundary_index]]
        else:
            data_start_line = int(data_lines[0]["line_id"])
            data_end_line = int(data_lines[-1]["line_id"])
            # Pick the 3 lines with the most numeric tokens as samples,
            # so the LLM sees the richest data rows when classifying the block.
            sample_lines = sorted(
                data_lines,
                key=lambda ln: len(_numeric_tokens(str(ln["text"]))),
                reverse=True,
            )[:3]
        blocks.append(
            {
                "block_id": f"B{block_number:03d}",
                "title_line": title_line,
                "data_start_line": data_start_line,
                "data_end_line": data_end_line,
                "boundary_text": str(lines[boundary_index]["text"]),
                "boundary_reason": boundary_reasons[boundary_index],
                "sample_lines": sample_lines,
                "has_numeric_data": any(
                    any(ch.isdigit() for ch in str(line["text"]))
                    for line in data_lines
                ),
            }
        )
    return blocks


def _structure_file(workspace_root: Path, doc_stem: str) -> Path:
    return workspace_root / GENERATED_DOC_STRUCTURE_DIR / f"{doc_stem}.json"


def load_doc_structure(workspace_root: Path, doc_stem: str) -> dict[str, Any] | None:
    path = _structure_file(workspace_root, doc_stem)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _persist_doc_structure(workspace_root: Path, doc_stem: str, payload: dict[str, Any]) -> None:
    path = _structure_file(workspace_root, doc_stem)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _persist_visible_doc_structure(
    log_dir: Path | None,
    doc_stem: str,
    payload: dict[str, Any],
) -> str | None:
    if log_dir is None:
        return None
    visible_dir = log_dir / VISIBLE_DOC_STRUCTURE_DIR
    visible_dir.mkdir(parents=True, exist_ok=True)
    relative_path = f"{VISIBLE_DOC_STRUCTURE_DIR}/{doc_stem}.json"
    (log_dir / relative_path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return relative_path


def _normalize_fields(fields: list[str] | None) -> list[str]:
    return [str(field).strip().strip("`") for field in fields or [] if str(field).strip()]


def _normalize_primary_key_field(value: Any, knowledge_text: str) -> str | None:
    """Normalize the primary key field name chosen by the LLM.

    The LLM reads the full knowledge document and selects the best primary key.
    We only normalize the name (strip whitespace/backticks); the LLM's judgment
    is trusted.
    """
    if value in (None, ""):
        return None
    normalized_values = _normalize_fields([str(value)])
    if not normalized_values:
        return None
    return normalized_values[0]


def _field_identity(value: Any) -> str:
    """Return a comparison key for field names supplied by the model."""
    normalized_values = _normalize_fields([str(value)])
    if not normalized_values:
        return ""
    return "".join(normalized_values[0].split()).casefold()


def _validate_primary_key_coverage(
    blocks: list[dict[str, Any]],
    primary_key_field: str | None,
) -> None:
    """Require a non-null document primary key to be selectable for extraction."""
    if primary_key_field is None:
        return
    primary_key_identity = _field_identity(primary_key_field)
    matching_block_ids = [
        str(block.get("block_id") or "")
        for block in blocks
        if any(
            _field_identity(field) == primary_key_identity
            for field in block.get("candidate_fields", [])
        )
    ]
    if matching_block_ids:
        return
    available_fields = sorted(
        {
            str(field)
            for block in blocks
            for field in block.get("candidate_fields", [])
            if _field_identity(field)
        },
        key=str.casefold,
    )
    raise ValueError(
        f"Primary key field {primary_key_field!r} was not included in any block "
        f"candidate_fields. Available candidate fields: {available_fields}. "
        "Mark every block that directly states this primary key, or return "
        "primary_key_field as null when the document has no direct primary-key evidence."
    )


def _cache_key(
    *,
    path: str,
    knowledge_path: str,
    target_table: str,
    fields: list[str],
    doc_hash: str,
    knowledge_hash: str,
) -> str:
    payload = {
        "path": path,
        "knowledge_path": knowledge_path,
        "target_table": target_table,
        "fields": fields,
        "doc_hash": doc_hash,
        "knowledge_hash": knowledge_hash,
        "structure_version": STRUCTURE_VERSION,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _normalize_blocks(raw_blocks: Any, candidate_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(raw_blocks, list):
        raise ValueError("Document structure response must contain blocks list.")
    blocks: list[dict[str, Any]] = []
    for raw in raw_blocks:
        if not isinstance(raw, dict):
            continue
        block_id = str(raw.get("block_id") or "").strip()
        candidate = candidate_by_id.get(block_id)
        if candidate is None:
            continue
        candidate_fields = _normalize_fields(raw.get("candidate_fields"))
        scope_id = str(raw.get("scope_id") or raw.get("section_scope") or "").strip()
        scope_name = str(raw.get("scope_name") or "").strip()
        blocks.append(
            {
                "block_id": block_id,
                "title_line": int(candidate["title_line"]),
                "data_start_line": int(candidate["data_start_line"]),
                "data_end_line": int(candidate["data_end_line"]),
                "boundary_text": str(candidate["boundary_text"]),
                "boundary_reason": str(candidate.get("boundary_reason") or ""),
                "scope_id": scope_id,
                "section_scope": scope_id,
                "scope_name": scope_name,
                "candidate_fields": candidate_fields,
                "confidence": raw.get("confidence"),
                "evidence": str(raw.get("evidence") or ""),
            }
        )
    if not blocks:
        raise ValueError("Document structure response did not classify any candidate blocks.")
    return blocks


def _short_error(exc: Exception | str, *, limit: int = 300) -> str:
    text = str(exc).replace("\n", " ").strip()
    return text if len(text) <= limit else text[:limit] + "..."


def _short_text(text: str, *, limit: int = 800) -> str:
    normalized = text.replace("\n", " ").strip()
    return normalized if len(normalized) <= limit else normalized[:limit] + "..."


def inspect_doc_structure(
    *,
    task: PublicTask,
    workspace: TaskContextWorkspace,
    model: Any,
    path: str,
    knowledge_path: str = "knowledge.md",
    target_table: str | None = None,
    fields: list[str] | None = None,
    max_model_calls: int | None = None,
    structured_doc_config: StructuredDocToolConfig | None = None,
    log_dir: Path | None = None,
) -> DocStructure:
    if model is None:
        raise ValueError("inspect_doc_structure requires an available model.")
    normalized_path = normalize_context_relative_path(path)
    normalized_knowledge_path = normalize_context_relative_path(knowledge_path)
    resolved_doc = resolve_context_path(task, normalized_path)
    resolved_knowledge = resolve_context_path(task, normalized_knowledge_path)
    doc_text = resolved_doc.read_text(encoding="utf-8", errors="replace")
    knowledge_text = resolved_knowledge.read_text(encoding="utf-8", errors="replace")
    target = target_table or PurePosixPath(normalized_path).stem
    requested_fields = _normalize_fields(fields)
    doc_stem = PurePosixPath(normalized_path).stem
    workspace_root = workspace.materialize()
    logger = DocStructureLogger(doc_stem, log_dir=log_dir)
    doc_hash = hashlib.sha256(doc_text.encode("utf-8")).hexdigest()
    knowledge_hash = hashlib.sha256(knowledge_text.encode("utf-8")).hexdigest()
    cache_key = _cache_key(
        path=normalized_path,
        knowledge_path=normalized_knowledge_path,
        target_table=target,
        fields=requested_fields,
        doc_hash=doc_hash,
        knowledge_hash=knowledge_hash,
    )
    logger.emit(
        "start",
        path=normalized_path,
        knowledge_path=normalized_knowledge_path,
        target_table=target,
        requested_fields=requested_fields,
        max_model_calls=max_model_calls,
        structure_version=STRUCTURE_VERSION,
    )
    visible_structure_file = (
        f"{VISIBLE_DOC_STRUCTURE_DIR}/{doc_stem}.json" if log_dir is not None else None
    )
    cached = load_doc_structure(workspace_root, doc_stem)
    if cached and cached.get("cache_key") == cache_key:
        if log_dir is not None:
            _persist_visible_doc_structure(log_dir, doc_stem, cached)
        logger.emit("cache_hit", cache_key=cache_key, block_count=len(cached.get("blocks", [])))
        metadata = dict(cached.get("structure", {}))
        metadata.update(
            {
                "cache_hit": True,
                "model_call_count": 0,
                "log_file": logger.relative_log_file,
                "visible_structure_file": visible_structure_file,
            }
        )
        metadata["log_summary"] = logger.summary()
        return DocStructure(blocks=list(cached.get("blocks", [])), metadata=metadata)
    lines = _split_non_empty_lines(doc_text)
    candidates = _candidate_blocks(lines)
    boundary_reason_counts: dict[str, int] = {}
    for candidate in candidates:
        reason = str(candidate.get("boundary_reason") or "")
        boundary_reason_counts[reason] = boundary_reason_counts.get(reason, 0) + 1
    logger.emit(
        "candidate_blocks",
        candidate_count=len(candidates),
        boundary_reason_counts=boundary_reason_counts,
    )
    if not candidates:
        raise ValueError("No candidate document blocks were found.")
    config = structured_doc_config or StructuredDocToolConfig()
    if max_model_calls is None:
        max_model_calls = config.inspect_doc_structure_max_model_calls
    max_calls = min(max(1, int(max_model_calls)), config.inspect_doc_structure_max_model_calls)
    if max_calls < 1:
        raise ValueError("inspect_doc_structure requires at least one model call.")
    payload = {
        "path": normalized_path,
        "target_table": target,
        "requested_fields": requested_fields or None,
        "structure_version": STRUCTURE_VERSION,
        "instruction": (
            "Classify candidate natural-language document blocks for structured "
            "data extraction. You must return a single JSON object with a top-level "
            '"blocks" key whose value is an array of block objects, plus top-level '
            '"primary_key_field" and "primary_key_evidence" keys.\n\n'

            "## Controlled scope_id vocabulary\n\n"
            "Every block MUST use exactly one of these four scope_id values:\n\n"
            "- `identity`: A paragraph whose primary purpose is to declare entity "
            "identity — stock code, fund code, company name, person name, or other "
            "identifier. It anchors the entity. candidate_fields MAY be non-empty; "
            "include the identifier field(s) directly stated.\n"
            "- `data`: A paragraph that lists individual records, transactions, "
            "events, or line items with quantitative/numeric values that are the "
            "primary subject of the requested fields. Each line typically introduces "
            "a new record. candidate_fields MAY be non-empty; include only fields "
            "whose values are directly stated as the primary subject of the block.\n"
            "- `context`: A paragraph that provides background, narration, analysis, "
            "transition, explanation, or discussion. It may mention field values in "
            "passing or as context for other facts, but its primary purpose is NOT "
            "to introduce the values themselves. candidate_fields MUST be [] (empty).\n"
            "- `summary`: A paragraph that summarises, concludes, reviews, looks "
            "ahead, gives recommendations, or discusses implications. It may repeat "
            "entity identifiers or field values already stated in earlier blocks, "
            "but it does NOT introduce new data records. candidate_fields MUST be "
            "[] (empty).\n\n"

            "## candidate_fields rules\n\n"
            "0. **Knowledge-bound**: Every field in candidate_fields MUST be defined "
            "in the knowledge document for this table. Do NOT invent field names "
            "from document text (e.g. do not create 'unit_id' just because the "
            "document mentions 'unit 16'). If a concept appears in the document "
            "but not in the knowledge schema, it is NOT a valid field. Only the "
            "knowledge document defines the field vocabulary.\n"
            "1. `identity` and `data` blocks: include fields whose values are the "
            "primary subject of the block (restricted to knowledge-defined fields "
            "per rule 0). Do not mark a field merely because its "
            "value appears as an entity identifier or side reference while the block "
            "primarily describes another field.\n"
            "2. `context` and `summary` blocks: candidate_fields MUST be []. Even if "
            "a field value is mentioned, the block is not a data source — it is "
            "discussion or recap.\n"
            "3. If consecutive blocks directly repeat values for the same field, mark "
            "that field in every IDENTITY or DATA block that directly states the "
            "value; do not omit repeated direct evidence merely to avoid redundancy. "
            "Never place the same field in candidate_fields for more than two blocks "
            "total. If more than two blocks are eligible, keep only the two whose "
            "direct evidence makes that field most likely to be a primary subject.\n"
            "4. Mark a field ONLY when the block's data lines directly state the "
            "value. Do not mark a field for related, adjacent, component, change, "
            "effect, post-event, or derived measures; do not mark fields whose values "
            "would require arithmetic, inference, or reconstruction from another "
            "measure.\n\n"

            "## Output format\n\n"
            "Each block object must contain: block_id, scope_id, scope_name, "
            "candidate_fields, confidence, evidence. scope_name is a short "
            "human-readable label in the document's language (e.g. '主体身份识别', "
            "'转让方持股分析', '结论综述').\n"
            "Read the knowledge document to identify the primary key field for the "
            "target table. The primary key MUST be chosen from fields defined in "
            "the knowledge document for this table. Choose the best table-level key "
            "or entity/filter anchor among knowledge-defined fields. Return null "
            "when the knowledge document does not define any field suitable as a "
            "primary key.\n"
            'Example: {"blocks":[{"block_id":"B001","scope_id":"identity",'
            '"scope_name":"基本信息","candidate_fields":["产品代码"],'
            '"confidence":0.9,"evidence":"..."}],'
            '"primary_key_field":"产品代码","primary_key_evidence":"..."}\n'
            "Do NOT return a bare JSON array — it must be wrapped in an object "
            'with a "blocks" key.'
        ),
        "knowledge": knowledge_text,
        "candidate_blocks": candidates,
    }
    started = perf_counter()
    logger.emit("classify_start", model_call_count=0, max_model_calls=max_calls)
    blocks: list[dict[str, Any]] | None = None
    primary_key_field: str | None = None
    primary_key_evidence = ""
    last_error: Exception | None = None
    last_response_text = ""
    candidate_by_id = {block["block_id"]: block for block in candidates}

    # Bind extraction-friendly sampling (higher max_tokens) so the
    # classification response is not truncated when the document has many
    # candidate blocks.
    classify_model = model
    if structured_doc_config is not None:
        bind = getattr(model, "bind", None)
        if callable(bind):
            classify_model = bind(
                temperature=structured_doc_config.llm.temperature,
                top_p=structured_doc_config.llm.top_p,
                max_tokens=structured_doc_config.llm.max_tokens,
                extra_body={"repetition_penalty": structured_doc_config.llm.repetition_penalty},
            )

    for attempt_index in range(1, max_calls + 1):
        attempt_started = perf_counter()
        logger.emit(
            "classify_attempt_start",
            attempt_index=attempt_index,
            model_call_count=attempt_index - 1,
        )
        attempt_payload = dict(payload)
        if last_error is not None:
            attempt_payload["repair"] = {
                "repair_instruction": (
                    "The previous response did not satisfy the required document "
                    "structure contract. scope_id must be one of: identity, data, "
                    "context, summary. context and summary blocks must have "
                    "candidate_fields: []. Every field in candidate_fields MUST be "
                    "defined in the knowledge document for this table — do not invent "
                    "fields from document text. A non-null primary_key_field must be "
                    "chosen from knowledge-defined fields and must appear in "
                    "candidate_fields for one to at most two identity or data blocks "
                    "that directly state its value. No field may appear in "
                    "candidate_fields for more than two blocks; if more blocks are "
                    "eligible, retain only the two with the strongest direct evidence "
                    "that the field is their primary subject. Return "
                    "primary_key_field as null when the knowledge document defines no "
                    "suitable primary key. Return only a JSON object with top-level "
                    "blocks, primary_key_field, and primary_key_evidence."
                ),
                "previous_error": _short_error(last_error),
                "previous_response_preview": _short_text(last_response_text),
                "required_schema": {
                    "blocks": [
                        {
                            "block_id": "B001",
                            "scope_id": "identity",
                            "scope_name": "human readable section name",
                            "candidate_fields": ["field_name"],
                            "confidence": 0.9,
                            "evidence": "brief evidence",
                        }
                    ],
                    "primary_key_field": "field_name_or_null",
                    "primary_key_evidence": "brief evidence or empty string",
                },
            }
        messages = [
            SystemMessage(
                    content=(
                        "You classify document sections into extraction scopes. "
                        "Use only the four controlled scope_id values: identity, data, "
                        "context, summary. context and summary blocks must have empty "
                        "candidate_fields. Every field in candidate_fields and "
                        "primary_key_field must be defined in the knowledge document "
                        "for this table — never invent field names from document text."
                    )
                ),
            HumanMessage(content=json.dumps(attempt_payload, ensure_ascii=False)),
        ]
        try:
            response = invoke_model_with_retries(classify_model, messages)
            last_response_text = _message_text(response)
            parsed = _last_json_object(last_response_text)
            normalized_blocks = _normalize_blocks(parsed.get("blocks"), candidate_by_id)
            normalized_primary_key_field = _normalize_primary_key_field(
                parsed.get("primary_key_field"),
                knowledge_text,
            )
            _validate_primary_key_coverage(normalized_blocks, normalized_primary_key_field)
            normalized_primary_key_evidence = (
                str(parsed.get("primary_key_evidence") or "")
                if normalized_primary_key_field is not None else ""
            )
            blocks = normalized_blocks
            primary_key_field = normalized_primary_key_field
            primary_key_evidence = normalized_primary_key_evidence
            logger.emit(
                "classify_attempt_done",
                attempt_index=attempt_index,
                block_count=len(blocks),
                primary_key_field=primary_key_field,
                elapsed_seconds=round(perf_counter() - attempt_started, 3),
                model_call_count=attempt_index,
            )
            break
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.emit(
                "classify_attempt_failed",
                attempt_index=attempt_index,
                elapsed_seconds=round(perf_counter() - attempt_started, 3),
                model_call_count=attempt_index,
                error=_short_error(exc),
                response_preview=_short_text(last_response_text),
            )
    if blocks is None:
        error = last_error or ValueError("Document structure classification failed.")
        logger.emit("failed", error=_short_error(error), model_call_count=max_calls)
        raise error
    logger.emit(
        "classify_done",
        block_count=len(blocks),
        elapsed_seconds=round(perf_counter() - started, 3),
        model_call_count=attempt_index,
    )
    structure_payload = {
        "blocks": blocks,
        "primary_key_field": primary_key_field,
    }
    structure_hash = hashlib.sha256(
        json.dumps(structure_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    metadata = {
        "source_path": normalized_path,
        "knowledge_path": normalized_knowledge_path,
        "target_table": target,
        "requested_fields": requested_fields,
        "input_line_count": len(lines),
        "candidate_block_count": len(candidates),
        "block_count": len(blocks),
        "doc_hash": doc_hash,
        "knowledge_hash": knowledge_hash,
        "primary_key_field": primary_key_field,
        "primary_key_evidence": primary_key_evidence,
        "structure_hash": structure_hash,
        "structure_version": STRUCTURE_VERSION,
        "model_call_count": attempt_index,
        "cache_hit": False,
        "log_file": logger.relative_log_file,
        "visible_structure_file": visible_structure_file,
    }
    payload_to_persist = {
        "cache_key": cache_key,
        "blocks": blocks,
        "structure": metadata,
    }
    _persist_doc_structure(workspace_root, doc_stem, payload_to_persist)
    visible_structure_file = _persist_visible_doc_structure(log_dir, doc_stem, payload_to_persist)
    if visible_structure_file is not None:
        metadata["visible_structure_file"] = visible_structure_file
        payload_to_persist["structure"] = metadata
        _persist_doc_structure(workspace_root, doc_stem, payload_to_persist)
        _persist_visible_doc_structure(log_dir, doc_stem, payload_to_persist)
    logger.emit(
        "persist_done",
        structure_file=f"{GENERATED_DOC_STRUCTURE_DIR}/{doc_stem}.json",
        visible_structure_file=visible_structure_file,
        block_count=len(blocks),
    )
    logger.emit("done", block_count=len(blocks), model_call_count=attempt_index)
    metadata["log_summary"] = logger.summary()
    return DocStructure(blocks=blocks, metadata=metadata)
