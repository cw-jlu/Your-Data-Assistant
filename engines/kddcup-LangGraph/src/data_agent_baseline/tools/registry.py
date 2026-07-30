from __future__ import annotations

import base64
from copy import deepcopy
import json
from collections.abc import Callable
from dataclasses import dataclass, field
import mimetypes
from pathlib import PurePosixPath
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel

from data_agent_baseline.benchmark.schema import AnswerTable, PublicTask
from data_agent_baseline.config import (
    DataInspectorSampleBudget,
    DataInspectorSemanticViewConfig,
    ToolConfig,
)
from data_agent_baseline.inspectors.semantic_catalog import (
    build_semantic_catalog,
    iter_logical_tables,
)
from data_agent_baseline.inspectors.semantic_views import find_semantic_view
from data_agent_baseline.tools.filesystem import (
    list_context_tree,
    normalize_context_relative_path,
    read_doc_preview,
    resolve_context_path,
    search_doc_text,
)
from data_agent_baseline.tools.langgraph_tools import (
    ExecuteProbeQueryArgs,
    ExecutePythonArgs,
    ExtractStructuredDocArgs,
    GetColumnDistinctValuesArgs,
    GetFieldProfileArgs,
    GetTableProfileArgs,
    GetTableRelationshipsArgs,
    InspectDocStructureArgs,
    ListContextArgs,
    LookupDocOutlineArgs,
    RecordVisualEvidenceArgs,
    ReadContextImageArgs,
    ReadDocArgs,
    SearchDocArgs,
    SearchSemanticCatalogArgs,
    SubmitToolResultArgs,
    create_structured_tool,
)
from data_agent_baseline.tools.doc_structure import inspect_doc_structure
from data_agent_baseline.tools.probe_engine import (
    execute_probe_query,
    get_column_distinct_values,
)
from data_agent_baseline.tools.python_exec import TaskContextWorkspace, execute_python_code
from data_agent_baseline.tools.structured_doc_extractor import (
    StructuredDocExtractionError,
    estimate_structured_doc_chunk_count,
    extract_structured_doc,
)
from data_agent_baseline.tools.truncation import truncate_answer_content, truncate_content

# Python 执行工具的固定超时时间，避免模型生成的脚本长时间卡住。
EXECUTE_PYTHON_TIMEOUT_SECONDS = 30


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    args_schema: type[BaseModel]


@dataclass(frozen=True, slots=True)
class ToolExecutionResult:
    ok: bool
    content: dict[str, Any]
    is_terminal: bool = False
    answer: AnswerTable | None = None
    answer_submission: dict[str, Any] | None = None
    model_content_parts: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class ToolRuntimeContext:
    task: PublicTask
    python_workspace: TaskContextWorkspace
    budget: DataInspectorSampleBudget = field(default_factory=DataInspectorSampleBudget)
    semantic_view_config: DataInspectorSemanticViewConfig = field(
        default_factory=DataInspectorSemanticViewConfig
    )
    _catalog_cache: dict[str, Any] | None = field(default=None, repr=False)
    model: object | None = field(default=None, repr=False)
    registry: "ToolRegistry | None" = field(default=None, repr=False)
    trace_dir: Any | None = field(default=None, repr=False)
    tool_gate: Any | None = field(default=None, repr=False)
    inspected_image_paths: set[str] = field(default_factory=set, repr=False)

    @property
    def temp_workspace(self) -> str | None:
        return None if self.python_workspace.path is None else str(self.python_workspace.path)


ToolHandler = Callable[[ToolRuntimeContext, dict[str, Any]], ToolExecutionResult]


def _ensure_catalog(runtime_context: ToolRuntimeContext) -> dict[str, Any]:
    if runtime_context._catalog_cache is None:
        runtime_context._catalog_cache = build_semantic_catalog(
            runtime_context.task,
            budget=runtime_context.budget,
            semantic_view_config=runtime_context.semantic_view_config,
            max_depth=20,
            include_relationships=True,
        )
    return runtime_context._catalog_cache


def _effective_context_dir(runtime_context: ToolRuntimeContext) -> Any:
    return runtime_context.python_workspace.path or runtime_context.task.context_dir


def _logical_tables_by_name(catalog: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(table["table"]): table for table in iter_logical_tables(catalog)}


def _resolve_logical_table(catalog: dict[str, Any], table_name: str) -> dict[str, Any] | None:
    normalized = table_name.strip().strip('"')
    semantic_view = find_semantic_view(catalog, normalized)
    if semantic_view is not None:
        return {
            "table": semantic_view["name"],
            "source_kind": "derived_view",
            "row_count": semantic_view.get("row_count"),
        }
    tables = _logical_tables_by_name(catalog)
    if normalized in tables:
        return tables[normalized]
    lowered = normalized.lower()
    for table, entry in tables.items():
        if table.lower() == lowered:
            return entry
    return None


DOCUMENT_RECOMMENDED_TOOLS = ["search_doc", "read_doc"]


def _normalize_resource_lookup_name(value: Any) -> str:
    text = str(value or "").strip().strip('"').strip("'").replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    if text.startswith("context/"):
        text = text[len("context/") :]
    if not text:
        return ""
    name = PurePosixPath(text).name or text
    if "." in name:
        name = PurePosixPath(name).stem
    return name.lower()


def _candidate_rank(query: str, candidate_values: list[str]) -> int | None:
    normalized_values = [value for value in candidate_values if value]
    if not query or not normalized_values:
        return None
    if query in normalized_values:
        return 0
    if any(query in value or value in query for value in normalized_values):
        return 1
    return None


def _suggest_logical_tables(
    catalog: dict[str, Any], requested_name: str, *, limit: int = 5
) -> list[dict[str, Any]]:
    query = _normalize_resource_lookup_name(requested_name)
    ranked: list[tuple[int, int, dict[str, Any]]] = []
    for index, logical in enumerate(iter_logical_tables(catalog)):
        table_name = str(logical.get("table", ""))
        rank = _candidate_rank(query, [_normalize_resource_lookup_name(table_name)])
        if rank is None:
            continue
        ranked.append(
            (
                rank,
                index,
                {
                    "table": table_name,
                    "row_count": logical.get("row_count"),
                    "source_kind": logical.get("source_kind"),
                },
            )
        )
    ranked.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in ranked[:limit]]


def _iter_document_suggestions(catalog: dict[str, Any]) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    for asset in catalog.get("assets", []):
        if asset.get("kind") != "document":
            continue
        path = str(asset.get("asset_path", ""))
        stem = PurePosixPath(path).stem
        documents.append(
            {
                "path": path,
                "document_id": stem,
                "stem": stem,
                "kind": "document",
                "size": asset.get("size"),
                "recommended_tools": list(DOCUMENT_RECOMMENDED_TOOLS),
            }
        )
    return documents


def _suggest_documents(
    catalog: dict[str, Any], requested_name: str, *, limit: int = 5
) -> list[dict[str, Any]]:
    query = _normalize_resource_lookup_name(requested_name)
    ranked: list[tuple[int, int, dict[str, Any]]] = []
    for index, document in enumerate(_iter_document_suggestions(catalog)):
        path = str(document.get("path", ""))
        stem = str(document.get("stem", ""))
        rank = _candidate_rank(
            query,
            [
                _normalize_resource_lookup_name(path),
                _normalize_resource_lookup_name(stem),
            ],
        )
        if rank is None:
            continue
        ranked.append((rank, index, document))
    ranked.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in ranked[:limit]]


def _unknown_logical_table_content(catalog: dict[str, Any], table_name: str) -> dict[str, Any]:
    table_suggestions = _suggest_logical_tables(catalog, table_name)
    document_suggestions = _suggest_documents(catalog, table_name)
    query = _normalize_resource_lookup_name(table_name)
    exact_document = next(
        (
            document
            for document in document_suggestions
            if query
            and query
            in {
                _normalize_resource_lookup_name(document.get("path")),
                _normalize_resource_lookup_name(document.get("stem")),
            }
        ),
        None,
    )

    if exact_document is not None:
        hint = (
            "Requested name matched a document, not a structured logical table. "
            f"Bind {exact_document['path']!r} as the ONLY authorized source for information "
            f"attributed to {table_name!r}. Do NOT query, profile, infer from, or substitute "
            "any same-named or similarly named SQL table or view. Your next action MUST inspect "
            f"that document with search_doc or read_doc using path {exact_document['path']!r}. "
            "If the document contains structured entities or metrics, call inspect_doc_structure "
            "before extract_structured_doc; only the table produced by that extraction may later "
            "be queried as this source."
        )
    elif document_suggestions:
        hint = (
            "No structured logical table matched. Candidate documents were found; "
            "prioritize inspecting those documents with search_doc or read_doc before trying "
            "similarly named SQL tables. If a candidate document contains structured entities "
            "or metrics, call inspect_doc_structure before extract_structured_doc."
        )
    elif table_suggestions:
        hint = "No exact logical table matched. Use one of the table_suggestions if appropriate."
    else:
        hint = (
            "No matching logical table or document was found. Use list_context or "
            "search_semantic_catalog with scope='all' to discover available resources."
        )

    return {
        "error": f"Unknown logical table: {table_name}",
        "requested_table": table_name,
        "table_suggestions": table_suggestions,
        "document_suggestions": document_suggestions,
        "hint": hint,
    }


def _strip_source_details(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_source_details(item)
            for key, item in value.items()
            if key not in {"asset_path", "source_asset_path"}
        }
    if isinstance(value, list):
        return [_strip_source_details(item) for item in value]
    return value


def _schema_for_logical_table(
    catalog: dict[str, Any], logical_table: dict[str, Any]
) -> dict[str, Any] | None:
    asset_path = logical_table.get("source_asset_path")
    source_table = logical_table.get("source_table")
    for schema in catalog.get("schemas", []):
        if schema.get("asset_path") != asset_path:
            continue
        if schema.get("kind") == "sqlite":
            for table in schema.get("tables", []):
                if table.get("name") == source_table:
                    return {
                        "table": logical_table["table"],
                        "kind": "sqlite",
                        "row_count": table.get("row_count"),
                        "fields": table.get("fields", []),
                    }
            return None
        return {
            "table": logical_table["table"],
            "kind": schema.get("kind"),
            "row_count": schema.get("row_count"),
            "json_structure": schema.get("json_structure"),
            "fields": schema.get("fields", []),
        }
    return None


def _search_semantic_catalog(
    runtime_context: ToolRuntimeContext, action_input: dict[str, Any]
) -> ToolExecutionResult:
    catalog = _ensure_catalog(runtime_context)
    query = str(action_input["query"]).strip().lower()
    scope = str(action_input.get("scope", "all")).strip().lower()
    limit = max(1, min(int(action_input.get("limit", 20)), 100))
    matches: list[dict[str, Any]] = []

    def in_scope(name: str) -> bool:
        return scope in {"all", name}

    if in_scope("tables") or in_scope("fields"):
        for view in catalog.get("derived_views", []):
            table_name = str(view.get("name", ""))
            if in_scope("tables") and query in table_name.lower():
                matches.append(
                    {
                        "type": "semantic_view",
                        "table": table_name,
                        "base_table": view.get("base_table"),
                        "row_count": view.get("row_count"),
                        "warnings": view.get("warnings", []),
                    }
                )
            if in_scope("fields"):
                for column in view.get("columns", []):
                    column_name = str(column.get("name", ""))
                    if query in column_name.lower() or query in table_name.lower():
                        matches.append(
                            {
                                "type": "semantic_view_field",
                                "table": table_name,
                                "column": column_name,
                                "field_type": column.get("type"),
                                "source_table": column.get("source_table"),
                                "source_field": column.get("source_field"),
                                "role": column.get("role"),
                            }
                        )
        for logical in iter_logical_tables(catalog):
            table_name = str(logical["table"])
            if in_scope("tables") and query in table_name.lower():
                matches.append(
                    {"type": "table", "table": table_name, "row_count": logical.get("row_count")}
                )
            if in_scope("fields"):
                for column in logical.get("columns", []):
                    column_name = str(column.get("name", ""))
                    if query in column_name.lower() or query in table_name.lower():
                        matches.append(
                            {
                                "type": "field",
                                "table": table_name,
                                "column": column_name,
                                "field_type": column.get("type"),
                            }
                        )

    if in_scope("documents"):
        for asset in catalog.get("assets", []):
            if asset.get("kind") != "document":
                continue
            path = str(asset.get("asset_path", ""))
            if query in path.lower():
                matches.append({"type": "document", "path": path, "size": asset.get("size")})

    if in_scope("relationships"):
        table_lookup = _logical_tables_by_name(catalog)
        source_to_table: dict[tuple[str, str | None], str] = {}
        for name, logical in table_lookup.items():
            source_to_table[
                (str(logical.get("source_asset_path")), logical.get("source_table"))
            ] = name
        for rel in catalog.get("relationships", []):
            source = rel.get("source", {})
            target = rel.get("target", {})
            source_table = source_to_table.get((str(source.get("asset_path")), source.get("table")))
            target_table = source_to_table.get((str(target.get("asset_path")), target.get("table")))
            text = " ".join(
                [
                    str(source_table or ""),
                    str(target_table or ""),
                    " ".join(str(v) for v in source.get("fields", [])),
                    " ".join(str(v) for v in target.get("fields", [])),
                    str(rel.get("relationship_type", "")),
                ]
            ).lower()
            if query in text:
                matches.append(
                    {
                        "type": "relationship",
                        "source_table": source_table,
                        "source_fields": source.get("fields", []),
                        "target_table": target_table,
                        "target_fields": target.get("fields", []),
                        "relationship_type": rel.get("relationship_type"),
                        "confidence": rel.get("confidence"),
                    }
                )

    if in_scope("uncertainties"):
        for item in catalog.get("semantic_uncertainties", []):
            text = " ".join(str(v) for v in item.values()).lower()
            if query in text:
                matches.append({"type": "uncertainty", **item})

    return ToolExecutionResult(
        ok=True,
        content={
            "query": query,
            "scope": scope,
            "matches": matches[:limit],
            "match_count": len(matches),
        },
    )


def _get_table_profile(
    runtime_context: ToolRuntimeContext, action_input: dict[str, Any]
) -> ToolExecutionResult:
    catalog = _ensure_catalog(runtime_context)
    table_name = str(action_input["table"])
    semantic_view = find_semantic_view(catalog, table_name)
    if semantic_view is not None:
        return ToolExecutionResult(
            ok=True,
            content={
                "table": semantic_view.get("name"),
                "kind": "derived_view",
                "base_table": semantic_view.get("base_table"),
                "base_source": semantic_view.get("base_source"),
                "grain": semantic_view.get("grain"),
                "is_original_table": False,
                "knowledge_authority": semantic_view.get("knowledge_authority"),
                "description": semantic_view.get("description"),
                "row_count": semantic_view.get("row_count"),
                "fields": semantic_view.get("columns", []),
                "joins": semantic_view.get("joins", []),
                "warnings": semantic_view.get("warnings", []),
            },
        )
    logical = _resolve_logical_table(catalog, table_name)
    if logical is None:
        return ToolExecutionResult(
            ok=False, content=_unknown_logical_table_content(catalog, table_name)
        )
    schema = _schema_for_logical_table(catalog, logical)
    return ToolExecutionResult(ok=schema is not None, content=_strip_source_details(schema or {}))


def _get_field_profile(
    runtime_context: ToolRuntimeContext, action_input: dict[str, Any]
) -> ToolExecutionResult:
    table_result = _get_table_profile(runtime_context, {"table": action_input["table"]})
    if not table_result.ok:
        return table_result
    column = str(action_input["column"]).strip().lower()
    for field_profile in table_result.content.get("fields", []):
        names = {
            str(field_profile.get("name", "")).lower(),
            str(field_profile.get("json_path", "")).lower(),
        }
        if column in names:
            return ToolExecutionResult(
                ok=True,
                content={
                    "table": table_result.content.get("table"),
                    "field": field_profile,
                },
            )
    return ToolExecutionResult(
        ok=False,
        content={
            "error": f"Unknown column {action_input['column']!r} on table {action_input['table']!r}"
        },
    )


def _get_table_relationships(
    runtime_context: ToolRuntimeContext, action_input: dict[str, Any]
) -> ToolExecutionResult:
    catalog = _ensure_catalog(runtime_context)
    table_name = str(action_input["table"])
    semantic_view = find_semantic_view(catalog, table_name)
    if semantic_view is not None:
        base_table = str(semantic_view.get("base_table", ""))
        base_relationships: list[dict[str, Any]] = []
        if base_table:
            base_result = _get_table_relationships(runtime_context, {"table": base_table})
            if base_result.ok:
                base_relationships = base_result.content.get("relationships", [])
        return ToolExecutionResult(
            ok=True,
            content={
                "table": semantic_view.get("name"),
                "kind": "derived_view",
                "embedded_joins": semantic_view.get("joins", []),
                "base_table": base_table,
                "base_table_relationships": base_relationships,
            },
        )
    logical = _resolve_logical_table(catalog, table_name)
    if logical is None:
        return ToolExecutionResult(
            ok=False, content=_unknown_logical_table_content(catalog, table_name)
        )
    source_key = (str(logical.get("source_asset_path")), logical.get("source_table"))
    source_to_table = {
        (str(item.get("source_asset_path")), item.get("source_table")): str(item["table"])
        for item in iter_logical_tables(catalog)
    }
    relationships: list[dict[str, Any]] = []
    for rel in catalog.get("relationships", []):
        source = rel.get("source", {})
        target = rel.get("target", {})
        rel_source_key = (str(source.get("asset_path")), source.get("table"))
        rel_target_key = (str(target.get("asset_path")), target.get("table"))
        if source_key not in {rel_source_key, rel_target_key}:
            continue
        relationships.append(
            {
                "source_table": source_to_table.get(rel_source_key),
                "source_fields": source.get("fields", []),
                "target_table": source_to_table.get(rel_target_key),
                "target_fields": target.get("fields", []),
                "relationship_type": rel.get("relationship_type"),
                "cardinality": rel.get("cardinality"),
                "confidence": rel.get("confidence"),
                "evidence": rel.get("evidence"),
            }
        )
    return ToolExecutionResult(
        ok=True, content={"table": logical["table"], "relationships": relationships}
    )


IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp"})


def _read_context_image(
    runtime_context: ToolRuntimeContext, action_input: dict[str, Any]
) -> ToolExecutionResult:
    image_path = str(action_input["path"])
    normalized_path = normalize_context_relative_path(image_path)
    path = resolve_context_path(runtime_context.task, normalized_path)
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        return ToolExecutionResult(
            ok=False,
            content={
                "error": f"Unsupported image type: {normalized_path}. Supported: jpg, jpeg, png, webp."
            },
        )
    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    image_b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    detail = str(action_input.get("detail", "auto") or "auto")
    runtime_context.inspected_image_paths.add(normalized_path)
    return ToolExecutionResult(
        ok=True,
        content={
            "path": normalized_path,
            "mime_type": mime_type,
            "size": path.stat().st_size,
            "status": "image attached to next model request",
        },
        model_content_parts=[
            {"type": "text", "text": f"Image from `{normalized_path}`:"},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{image_b64}",
                    "detail": detail,
                },
            },
        ],
    )


def _record_visual_evidence(
    runtime_context: ToolRuntimeContext, action_input: dict[str, Any]
) -> ToolExecutionResult:
    image_path = str(action_input["path"])
    normalized_path = normalize_context_relative_path(image_path)
    if normalized_path not in runtime_context.inspected_image_paths:
        return ToolExecutionResult(
            ok=False,
            content={
                "error": (
                    "Visual evidence can only be recorded after a successful "
                    f"read_context_image call for the same path: {normalized_path}."
                )
            },
        )
    observations = str(action_input["observations"]).strip()
    if not observations:
        return ToolExecutionResult(ok=False, content={"error": "observations must not be empty."})
    return ToolExecutionResult(
        ok=True,
        content={
            "path": normalized_path,
            "observations": truncate_content(observations, max_str_tokens=300, max_list_items=5),
            "status": "visual observation recorded after image inspection",
        },
    )


def _list_context(
    runtime_context: ToolRuntimeContext, action_input: dict[str, Any]
) -> ToolExecutionResult:
    max_depth = int(action_input.get("max_depth", 4))
    return ToolExecutionResult(
        ok=True,
        content=list_context_tree(runtime_context.task, max_depth=max_depth),
    )


def _lookup_doc_outline(
    runtime_context: ToolRuntimeContext, action_input: dict[str, Any]
) -> ToolExecutionResult:
    doc_path = str(action_input["path"])

    catalog = _ensure_catalog(runtime_context)
    normalized_path = normalize_context_relative_path(doc_path)

    for schema in catalog["schemas"]:
        if schema.get("kind") != "document":
            continue
        if schema.get("asset_path") == normalized_path:
            headings = schema.get("headings", [])
            return ToolExecutionResult(
                ok=True,
                content={
                    "path": normalized_path,
                    "head_count": len(headings),
                    "headings": headings,
                },
            )

    for asset in catalog.get("assets", []):
        if asset.get("asset_path") == normalized_path:
            return ToolExecutionResult(
                ok=False,
                content={
                    "error": f"'{normalized_path}' is not a text document (kind={asset.get('kind')}). lookup_doc_outline only supports .md, .txt, .rst files.",
                },
            )

    return ToolExecutionResult(
        ok=False,
        content={
            "error": f"No document found matching path '{doc_path}'.",
            "hint": "Use list_context or check the catalog for available documents.",
        },
    )


def _search_doc(
    runtime_context: ToolRuntimeContext, action_input: dict[str, Any]
) -> ToolExecutionResult:
    query = str(action_input["query"])
    context_lines = int(action_input.get("context_lines", 5))
    path = action_input.get("path")
    page = int(action_input.get("page", 1))
    page_size = int(action_input.get("page_size", 20))
    return ToolExecutionResult(
        ok=True,
        content=search_doc_text(
            runtime_context.task,
            query,
            context_lines=context_lines,
            path=path,
            page=page,
            page_size=page_size,
        ),
    )


def _read_doc(
    runtime_context: ToolRuntimeContext, action_input: dict[str, Any]
) -> ToolExecutionResult:
    path = str(action_input["path"])
    heading = action_input.get("heading")
    return ToolExecutionResult(
        ok=True,
        content=read_doc_preview(runtime_context.task, path, heading=heading),
    )



def _execute_python(
    runtime_context: ToolRuntimeContext, action_input: dict[str, Any]
) -> ToolExecutionResult:
    code = str(action_input["code"])
    workspace_root = runtime_context.python_workspace.materialize()
    catalog = _ensure_catalog(runtime_context)
    content = execute_python_code(
        context_root=workspace_root,
        code=code,
        timeout_seconds=EXECUTE_PYTHON_TIMEOUT_SECONDS,
        catalog=catalog,
    )
    return ToolExecutionResult(ok=bool(content.get("success")), content=content)


def _probe_queries_from_args(action_input: dict[str, Any]) -> list[str]:
    if "queries" in action_input:
        return [str(q) for q in action_input["queries"]]
    if "sql" in action_input:
        # Backward compatibility for older traces/tests and for models that
        # still emit the pre-batching argument name.
        return [str(action_input["sql"])]
    raise ValueError("execute_probe_query requires `queries` (list[str]).")


def _execute_probe_query(
    runtime_context: ToolRuntimeContext, action_input: dict[str, Any]
) -> ToolExecutionResult:
    catalog = _ensure_catalog(runtime_context)
    limit = min(int(action_input.get("limit", 200)), 200)
    try:
        queries = _probe_queries_from_args(action_input)
        result = execute_probe_query(
            context_dir=_effective_context_dir(runtime_context),
            catalog=catalog,
            queries=queries,
            limit=limit,
        )
    except ValueError as exc:
        return ToolExecutionResult(ok=False, content={"error": str(exc)})
    return ToolExecutionResult(
        ok=bool(result.get("ok")),
        content=result,
    )


def _normalize_fields_arg(value: Any) -> list[str] | None:
    """Accept fields as a list or a JSON-encoded string of a list."""
    if value is None:
        return None
    if isinstance(value, list):
        return [str(v).strip() for v in value if v is not None]
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                return None
            if isinstance(parsed, list):
                return [str(v).strip() for v in parsed if v is not None]
        return [stripped]
    return None


def _extract_structured_doc(
    runtime_context: ToolRuntimeContext, action_input: dict[str, Any]
) -> ToolExecutionResult:
    catalog = _ensure_catalog(runtime_context)
    target_table = action_input.get("target_table")
    structured_doc_config = runtime_context.registry.tool_config.structured_doc
    raw_max_model_calls = action_input.get(
        "max_model_calls",
        structured_doc_config.default_max_model_calls,
    )
    gate = runtime_context.tool_gate
    gate_acquired = False
    try:
        if gate is not None:
            priority = estimate_structured_doc_chunk_count(
                task=runtime_context.task,
                workspace=runtime_context.python_workspace,
                catalog=catalog,
                path=str(action_input["path"]),
                knowledge_path=str(action_input.get("knowledge_path") or "knowledge.md"),
                target_table=None if target_table in (None, "") else str(target_table),
                fields=_normalize_fields_arg(action_input.get("fields")),
                max_model_calls=int(raw_max_model_calls),
                structured_doc_config=structured_doc_config,
            )
            gate.acquire(
                "extract_structured_doc",
                priority_chunk_count=priority.priority_chunk_count,
                selected_line_count=priority.selected_line_count,
                priority_source=priority.priority_source,
                priority_error=priority.error,
            )
            gate_acquired = True
        extraction = extract_structured_doc(
            task=runtime_context.task,
            workspace=runtime_context.python_workspace,
            catalog=catalog,
            model=runtime_context.model,
            path=str(action_input["path"]),
            knowledge_path=str(action_input.get("knowledge_path") or "knowledge.md"),
            target_table=None if target_table in (None, "") else str(target_table),
            fields=_normalize_fields_arg(action_input.get("fields")),
            max_model_calls=int(raw_max_model_calls),
            structured_doc_config=structured_doc_config,
            log_dir=runtime_context.trace_dir,
        )
    except StructuredDocExtractionError as exc:
        content = {
            "error": str(exc),
            "extraction": {
                "log_summary": exc.log_summary,
            },
        }
        content.update(exc.details)
        return ToolExecutionResult(
            ok=False,
            content=content,
        )
    except Exception as exc:  # noqa: BLE001
        return ToolExecutionResult(ok=False, content={"error": str(exc)})
    finally:
        if gate_acquired and gate is not None:
            gate.release("extract_structured_doc")
    return ToolExecutionResult(
        ok=True,
        content={
            "columns": extraction.columns,
            "rows": extraction.rows,
            "extraction": extraction.metadata,
            "unit_normalized": True,
        },
    )


def _inspect_doc_structure(
    runtime_context: ToolRuntimeContext, action_input: dict[str, Any]
) -> ToolExecutionResult:
    target_table = action_input.get("target_table")
    structured_doc_config = runtime_context.registry.tool_config.structured_doc
    raw_max_model_calls = action_input.get(
        "max_model_calls",
        structured_doc_config.inspect_doc_structure_max_model_calls,
    )
    try:
        structure = inspect_doc_structure(
            task=runtime_context.task,
            workspace=runtime_context.python_workspace,
            model=runtime_context.model,
            path=str(action_input["path"]),
            knowledge_path=str(action_input.get("knowledge_path") or "knowledge.md"),
            target_table=None if target_table in (None, "") else str(target_table),
            fields=None,
            max_model_calls=int(raw_max_model_calls),
            structured_doc_config=structured_doc_config,
            log_dir=runtime_context.trace_dir,
        )
    except Exception as exc:  # noqa: BLE001
        return ToolExecutionResult(ok=False, content={"error": str(exc)})
    return ToolExecutionResult(
        ok=True,
        content={
            "blocks": structure.blocks,
            "structure": structure.metadata,
        },
    )


def _get_column_distinct_values(
    runtime_context: ToolRuntimeContext, action_input: dict[str, Any]
) -> ToolExecutionResult:
    catalog = _ensure_catalog(runtime_context)
    table = str(action_input["table"])
    column = str(action_input["column"])
    top_n = min(int(action_input.get("top_n", 20)), 200)

    result = get_column_distinct_values(
        context_dir=runtime_context.task.context_dir,
        catalog=catalog,
        table=table,
        column=column,
        top_n=top_n,
    )
    return ToolExecutionResult(
        ok=bool(result.get("ok")),
        content=result,
    )


# ---------------------------------------------------------------------------
# submit_tool_result: 通过执行数据工具并提取其输出来提交答案
# ---------------------------------------------------------------------------


def _extract_answer_from_probe_query(content: dict[str, Any]) -> tuple[list[str], list[list[Any]]]:
    """从 execute_probe_query 的批量结果中，选取最后一个成功子查询作为答案。"""
    results = content.get("results", [])
    for result in reversed(results):
        if result.get("ok") and result.get("columns") and result.get("rows") is not None:
            columns = list(result["columns"])
            rows = [list(row) for row in result["rows"]]
            return columns, rows
    raise ValueError(
        "No successful query result found in execute_probe_query output. "
        "Ensure at least one query in the batch succeeds."
    )


def _extract_answer_from_python(content: dict[str, Any]) -> tuple[list[str], list[list[Any]]]:
    """从 execute_python 的 stdout 中解析 JSON 格式的答案。

    约定模型在 Python 代码中 print(json.dumps({"columns": [...], "rows": [...]}))。
    """
    output = content.get("output", "")
    decoder = json.JSONDecoder()
    parsed: Any = None
    last_valid_json: Any = None
    parse_error: json.JSONDecodeError | None = None
    for start, char in enumerate(output):
        if char != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(output[start:])
        except json.JSONDecodeError as exc:
            parse_error = exc
            continue
        last_valid_json = candidate
        if isinstance(candidate, dict) and "columns" in candidate and "rows" in candidate:
            parsed = candidate
    if parsed is None:
        if last_valid_json is not None:
            raise ValueError(
                "Parsed JSON must contain 'columns' (list[str]) and 'rows' (list[list])."
            )
        if parse_error is not None:
            raise ValueError(
                f"Failed to parse JSON from execute_python output: {parse_error}"
            ) from parse_error
        raise ValueError(
            "execute_python output does not contain a valid JSON object. "
            "The Python code must print a JSON object with 'columns' and 'rows' keys, "
            "e.g., print(json.dumps({'columns': [...], 'rows': [...]}))"
        )
    columns = parsed.get("columns")
    rows = parsed.get("rows")
    if not isinstance(columns, list) or not isinstance(rows, list):
        raise ValueError("Parsed JSON must contain 'columns' (list[str]) and 'rows' (list[list]).")
    normalized_rows: list[list[Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, (list, tuple)):
            raise ValueError(
                "Parsed JSON rows must be list[list]. "
                f"Row {index} is {type(row).__name__}; build rows in column order, "
                "e.g. [[record[col] for col in columns] for record in records]."
            )
        normalized_rows.append(list(row))
    return list(columns), normalized_rows


def _extract_answer_from_context_sql(content: dict[str, Any]) -> tuple[list[str], list[list[Any]]]:
    """Extract an answer from a direct columns/rows SQL-style payload."""
    columns = content.get("columns")
    rows = content.get("rows")
    if not isinstance(columns, list) or not isinstance(rows, list):
        raise ValueError("SQL result did not return columns/rows.")
    return list(columns), [list(row) for row in rows]


def _selected_probe_query_index(content: dict[str, Any]) -> int | None:
    results = content.get("results", [])
    if not isinstance(results, list):
        return None
    for index in range(len(results) - 1, -1, -1):
        result = results[index]
        if (
            isinstance(result, dict)
            and result.get("ok")
            and result.get("columns")
            and result.get("rows") is not None
        ):
            return index
    return None


# 注册每种源工具的结果提取器
_ANSWER_EXTRACTORS: dict[
    str,
    Callable[[dict[str, Any]], tuple[list[str], list[list[Any]]]],
] = {
    "execute_probe_query": _extract_answer_from_probe_query,
    "execute_python": _extract_answer_from_python,
}


def _execute_submit_source_tool(
    runtime_context: ToolRuntimeContext,
    tool_name: str,
    tool_args: dict[str, Any],
) -> ToolExecutionResult:
    if tool_name == "execute_probe_query":
        catalog = _ensure_catalog(runtime_context)
        try:
            result = execute_probe_query(
                context_dir=_effective_context_dir(runtime_context),
                catalog=catalog,
                queries=_probe_queries_from_args(tool_args),
                limit=None,
            )
        except ValueError as exc:
            return ToolExecutionResult(ok=False, content={"error": str(exc)})
        return ToolExecutionResult(ok=bool(result.get("ok")), content=result)

    if tool_name == "execute_python":
        return _execute_python(runtime_context, tool_args)

    registry = runtime_context.registry
    if registry is None:
        return ToolExecutionResult(ok=False, content={"error": "Tool registry is not available."})
    try:
        return registry.execute(runtime_context, tool_name, tool_args)
    except Exception as exc:
        return ToolExecutionResult(
            ok=False, content={"error": f"Failed to execute {tool_name}: {exc}"}
        )


def _submit_tool_result(
    runtime_context: ToolRuntimeContext,
    action_input: dict[str, Any],
) -> ToolExecutionResult:
    tool_name = str(action_input.get("tool_name", ""))
    tool_args = action_input.get("tool_args", {})
    requested_columns = action_input.get("columns")

    # 1. 校验源工具是否支持
    if tool_name not in _ANSWER_EXTRACTORS:
        supported = ", ".join(sorted(_ANSWER_EXTRACTORS.keys()))
        return ToolExecutionResult(
            ok=False,
            content={"error": f"Unsupported source tool: {tool_name!r}. Supported: {supported}"},
        )

    # 2. 校验 tool_args 类型
    if not isinstance(tool_args, dict):
        return ToolExecutionResult(
            ok=False,
            content={"error": "tool_args must be a dict."},
        )
    submission_tool_args = deepcopy(tool_args)
    submission_column_override = deepcopy(requested_columns)

    # 3. 执行源工具。最终提交路径不使用探查预览 limit。
    source_result = _execute_submit_source_tool(runtime_context, tool_name, tool_args)

    if not source_result.ok:
        return ToolExecutionResult(
            ok=False,
            content={
                "error": f"{tool_name} execution failed.",
                "details": source_result.content,
            },
        )

    # 4. 提取答案数据
    try:
        extractor = _ANSWER_EXTRACTORS[tool_name]
        columns, rows = extractor(source_result.content)
    except Exception as exc:
        return ToolExecutionResult(
            ok=False,
            content={"error": f"Failed to extract answer from {tool_name} output: {exc}"},
        )

    source_output_columns = list(columns)
    selected_query_index = (
        _selected_probe_query_index(source_result.content)
        if tool_name == "execute_probe_query"
        else None
    )

    # 5. 处理可选的列覆盖
    if requested_columns is not None:
        if not isinstance(requested_columns, list) or not all(
            isinstance(c, str) for c in requested_columns
        ):
            return ToolExecutionResult(
                ok=False,
                content={"error": "columns must be a list of strings."},
            )
        if len(requested_columns) != len(columns):
            return ToolExecutionResult(
                ok=False,
                content={
                    "error": (
                        f"Column count mismatch: specified {len(requested_columns)} columns "
                        f"({requested_columns}), but tool returned {len(columns)} columns ({columns})."
                    ),
                },
            )
        columns = list(requested_columns)

    # 6. 校验行数据
    for i, row in enumerate(rows):
        if len(row) != len(columns):
            return ToolExecutionResult(
                ok=False,
                content={
                    "error": f"Row {i} has {len(row)} cells, expected {len(columns)}.",
                },
            )

    # 7. 构造 AnswerTable
    answer = AnswerTable(columns=columns, rows=rows)
    return ToolExecutionResult(
        ok=True,
        content={
            "status": "submitted",
            "source_tool": tool_name,
            "column_count": len(columns),
            "row_count": len(rows),
        },
        is_terminal=True,
        answer=answer,
        answer_submission={
            "submission_tool": "submit_tool_result",
            "source_tool": tool_name,
            "source_tool_args": submission_tool_args,
            "column_override": submission_column_override,
            "source_output_columns": source_output_columns,
            "final_columns": list(columns),
            "selected_query_index": selected_query_index,
        },
    )


@dataclass(slots=True)
class BoundToolRegistry:
    registry: ToolRegistry
    runtime_context: ToolRuntimeContext
    tools: dict[str, BaseTool]

    def langchain_tools(self) -> list[BaseTool]:
        return [self.tools[name] for name in sorted(self.tools)]

    def execute(self, action: str, action_input: dict[str, Any]) -> ToolExecutionResult:
        return self.registry.execute(self.runtime_context, action, action_input)


@dataclass(slots=True)
class ToolRegistry:
    specs: dict[str, ToolSpec]
    handlers: dict[str, ToolHandler]
    tool_config: ToolConfig = field(default_factory=ToolConfig)

    def bind(self, runtime_context: ToolRuntimeContext) -> BoundToolRegistry:
        runtime_context.registry = self
        tools: dict[str, BaseTool] = {}
        for name, spec in self.specs.items():
            tools[name] = create_structured_tool(
                name=spec.name,
                description=spec.description,
                args_schema=spec.args_schema,
                invoke=self._build_tool_wrapper(runtime_context, name),
            )
        return BoundToolRegistry(registry=self, runtime_context=runtime_context, tools=tools)

    def _build_tool_wrapper(
        self,
        runtime_context: ToolRuntimeContext,
        action: str,
    ) -> Callable[..., dict[str, Any]]:
        def invoke(**kwargs: Any) -> dict[str, Any]:
            result = self.execute(runtime_context, action, kwargs)
            return self.format_result(action, result)

        return invoke

    def format_result(self, action: str, result: ToolExecutionResult) -> dict[str, Any]:
        payload = {
            "ok": result.ok,
            "content": result.content,
        }
        if result.answer is not None:
            payload["answer"] = truncate_answer_content(
                result.answer.to_dict(),
                max_str_tokens=self.tool_config.max_output_tokens,
                max_list_items=self.tool_config.max_list_items,
            )
        if action != "submit_tool_result":
            payload["content"] = truncate_content(
                payload["content"],
                max_str_tokens=self.tool_config.max_output_tokens,
                max_list_items=self.tool_config.max_list_items,
            )
        return payload

    def execute(
        self,
        runtime_context: ToolRuntimeContext,
        action: str,
        action_input: dict[str, Any],
    ) -> ToolExecutionResult:
        if action not in self.handlers:
            raise KeyError(f"Unknown tool: {action}")
        runtime_context.registry = self
        return self.handlers[action](runtime_context, action_input)


def create_default_tool_registry(tool_config: ToolConfig | None = None) -> ToolRegistry:
    specs = {
        "execute_probe_query": ToolSpec(
            name="execute_probe_query",
            description=(
                "Execute batched read-only SQL probes against logical tables using DuckDB. "
                "Use this as the default tool for "
                "understanding structured data: candidate-field checks, COUNTs, "
                "DISTINCT scans, sample rows, filters, joins that DuckDB can express, "
                "and quick aggregations. MANDATORY: pack multiple independent queries "
                "into ONE call whenever possible instead of sending them one by one. "
                "Each query in queries must be SELECT or WITH. "
                "Reference tables by the logical table names shown in the lightweight catalog. "
                "Do NOT wrap table references in single quotes in SQL — "
                "use FROM qualifying, not FROM 'qualifying'. "
                "This exploration tool returns a preview results list with at most "
                "200 rows per query, even if a larger limit is requested. For a final "
                "submission, use a clean query batch in which every query is known to "
                "succeed; submit_tool_result re-executes the batch and uses its last "
                "successful query as the answer."
            ),
            args_schema=ExecuteProbeQueryArgs,
        ),
        "execute_python": ToolSpec(
            name="execute_python",
            description=(
                "Execute Python code inside a per-task temporary copy of the context directory. "
                "The namespace already contains query(sql) and query_rows(sql) as global "
                "helper functions. Call them directly; do not import them. There is no "
                "`query` module, so never write `from query import query_rows`. "
                'Use `rows = query_rows("SELECT ... FROM logical_table")` for logical '
                "table queries. query(sql) returns a dict with `columns` and `rows`; "
                "query_rows(sql) returns only list-of-list rows, not dict rows. These "
                "helpers query the same logical tables exposed to execute_probe_query "
                "and return complete results; use them instead of creating a bare "
                "duckdb.connect(':memory:') when you need logical tables. "
                "Use when SQL tools are not enough: complex multi-file transformations, "
                "custom parsing, iterative logic, exact final row construction, or "
                "machine-readable JSON export of a large/intermediate result. Avoid using "
                "Python just to list files, grep text documents, or run simple COUNT/"
                "DISTINCT/sample probes that execute_probe_query can handle. "
                f"The execution timeout is fixed at {EXECUTE_PYTHON_TIMEOUT_SECONDS} seconds."
            ),
            args_schema=ExecutePythonArgs,
        ),
        "extract_structured_doc": ToolSpec(
            name="extract_structured_doc",
            description=(
                "Extract a structured table from a Markdown/text document using the "
                "field definitions in knowledge.md and LLM-assisted fact extraction. "
                "Use this when a domain table/entity is stored as a .md/.txt document "
                "rather than a SQL-visible logical table. The tool extracts visible "
                "facts from source lines, merges facts by entity key when fields are "
                "spread across sections, writes the merged table into the task "
                "runtime artifacts under structured_doc/ beside trace.json, keeps an "
                "internal query copy, and registers a DuckDB "
                "table named after the source document stem, or "
                "<stem>_extracted if the name conflicts with an existing logical table. "
                "After calling it, use the returned extraction.registered_table with "
                "execute_probe_query or execute_python query(sql) for filtering, joins, "
                "aggregation, and final submission. For sectioned Markdown documents, "
                "call inspect_doc_structure first; this tool then reuses the cached "
                "structure and automatically selects relevant blocks based on the "
                "requested fields. Pass the smallest exact fields list needed for the "
                "task, including join keys; manual block/range selection arguments are "
                "not exposed by this tool. If no structure cache is available, the tool "
                "returns missing_doc_structure instead of guessing from the full document. "
                "If it returns input-too-large, first reduce the requested fields; if "
                "the necessary scope remains too large, use read_doc/search_doc plus "
                "execute_python for regex/programmatic parsing. "
                "Do not use this tool directly as a submit_tool_result source tool; "
                "after extraction, submit the final answer via execute_probe_query or "
                "execute_python against the registered table. "
                "**All numeric values returned by this tool are already normalised to "
                "the base unit (unit=1): currency to yuan (元), percentages to decimal "
                "(1% → 0.01). Never apply additional unit conversions based on "
                "knowledge.md — the data is already in canonical form.**"
            ),
            args_schema=ExtractStructuredDocArgs,
        ),
        "inspect_doc_structure": ToolSpec(
            name="inspect_doc_structure",
            description=(
                "Inspect a Markdown/text document that carries structured data across "
                "natural-language sections. It auto-discovers all candidate fields, "
                "detects section boundaries from headings, no-number narrative lines, "
                "and transition sentences, then classifies each block with a scope "
                "label, line range, candidate fields, confidence, and evidence. "
                "The returned block summary tells you which fields exist in which "
                "blocks. Call this before extract_structured_doc so the cached "
                "structure enables automatic block selection. Do NOT pass a fields "
                "argument; read the returned summary, then pass only the verified "
                "field names (and their exact casing) to extract_structured_doc."
            ),
            args_schema=InspectDocStructureArgs,
        ),
        "get_column_distinct_values": ToolSpec(
            name="get_column_distinct_values",
            description=(
                "Get the most frequent distinct values for a specific column/field, "
                "ranked by frequency. Supports CSV, JSON, and SQLite. "
                "Use this to quickly understand what values a field contains, verify "
                "candidate field mapping, or identify filter values. Use a base logical "
                "table backed by CSV, JSON, or SQLite; use execute_probe_query for "
                "derived views or structured-document extracted tables."
            ),
            args_schema=GetColumnDistinctValuesArgs,
        ),
        "search_semantic_catalog": ToolSpec(
            name="search_semantic_catalog",
            description=(
                "Search the full semantic catalog using a case-insensitive keyword "
                "substring. Use concise table stems, field names, or document tokens "
                "to find candidate logical tables, fields, documents, relationships, "
                "or catalog warnings without loading the entire catalog into the prompt. "
                "This is not semantic retrieval. Use scope='all' when you are not sure "
                "whether a name refers to a table or a document."
            ),
            args_schema=SearchSemanticCatalogArgs,
        ),
        "get_table_profile": ToolSpec(
            name="get_table_profile",
            description=(
                "Return the full semantic profile for one SQL-visible logical table or "
                "derived view listed in query_surfaces, including fields, types, missing "
                "counts, cardinalities, top distinct values, numeric ranges, and source "
                "metadata for derived views. Do not call this tool for an exact document "
                "stem shown in documents: use search_doc/read_doc, then "
                "inspect_doc_structure and extract_structured_doc when that document "
                "contains structured entities or metrics. If the supplied table name is "
                "unknown, this tool returns suggestions only; it cannot profile a document."
            ),
            args_schema=GetTableProfileArgs,
        ),
        "get_field_profile": ToolSpec(
            name="get_field_profile",
            description=(
                "Return the full semantic profile for one field on a logical table, including "
                "type, missing count, cardinality, top distinct values, and numeric range."
            ),
            args_schema=GetFieldProfileArgs,
        ),
        "get_table_relationships": ToolSpec(
            name="get_table_relationships",
            description=(
                "Return inferred and explicit semantic relationships involving one logical table, "
                "including source/target fields, relationship type, confidence, and evidence."
                " Treat inferred relationships as candidate join paths, not established "
                "facts; verify every material join with knowledge evidence or observed rows."
            ),
            args_schema=GetTableRelationshipsArgs,
        ),
        "lookup_doc_outline": ToolSpec(
            name="lookup_doc_outline",
            description=(
                "Look up the table of contents / heading structure of a markdown "
                "or text document from the catalog. Use before read_doc so you can "
                "choose a targeted heading instead of reading a full document. "
                "Returns headings with level (1 = '#', 2 = '##', etc.). "
            ),
            args_schema=LookupDocOutlineArgs,
        ),
        "list_context": ToolSpec(
            name="list_context",
            description=(
                "List files and directories available under context. Use to discover "
                "available assets or resolve an unknown/missing path. It lists path "
                "metadata only; it does not read file contents. If the catalog already names the needed "
                "CSV/JSON/SQLite assets, go directly to execute_probe_query instead."
            ),
            args_schema=ListContextArgs,
        ),
        "read_doc": ToolSpec(
            name="read_doc",
            description=(
                "Read a text-like document inside context, optionally restricted to "
                "one heading. Use after lookup_doc_outline when you know the relevant "
                "section. Prefer search_doc first when you do not know which document "
                "or heading contains the needed fact."
            ),
            args_schema=ReadDocArgs,
        ),
        "read_context_image": ToolSpec(
            name="read_context_image",
            description=(
                "Attach an image from context to the next model request. Use this for stable "
                "video frames or other image evidence after inspecting the timeline or file list. "
                "Supported types are jpg, jpeg, png, and webp."
            ),
            args_schema=ReadContextImageArgs,
        ),
        "record_visual_evidence": ToolSpec(
            name="record_visual_evidence",
            description=(
                "Record a concise textual receipt for a stable frame after it was opened "
                "with read_context_image. Use the exact same path and preserve material "
                "visible values exactly."
            ),
            args_schema=RecordVisualEvidenceArgs,
        ),
        "search_doc": ToolSpec(
            name="search_doc",
            description=(
                "Search text documents (.md, .txt, .rst) inside context for a regex "
                "pattern or plain keyword. Returns matching lines with surrounding "
                "context lines and their locations. "
                "Results are paginated (default 20 matches per page, 1-indexed). "
                "Use 'page' to request different pages; check 'total_pages' and "
                "'total_matches' in the response to know how many pages are available. "
                "Use this before read_doc to locate relevant sections when you "
                "don't know where the information lives, instead of writing Python "
                "code to grep through documents."
            ),
            args_schema=SearchDocArgs,
        ),
        "submit_tool_result": ToolSpec(
            name="submit_tool_result",
            description=(
                "Submit the final answer. IMPORTANT: this tool RE-EXECUTES the "
                "specified source tool from scratch in the current task workspace with the given tool_args and "
                "uses its fresh output as the answer — it does NOT reuse or submit "
                "any previously observed tool output. You must provide the complete "
                "tool_args needed to reproduce the final result in a single fresh "
                "execution. "
                "Workflow: choose which source tool produces the answer "
                "(execute_probe_query for direct SQL, execute_python when SQL cannot "
                "perform the required transformation or formatting), then pass the exact same tool_args you "
                "would use to call that tool directly. "
                "Keep simple SQL filters in execute_probe_query; use execute_python "
                "for transformations or formatting such as converting datetime strings "
                "to ISO 8601, and "
                "include the full transformation code in tool_args. "
                "The system ignores preview limits: execute_probe_query returns all "
                "rows (no 200-row cap) and any limit value in tool_args is ignored. "
                "Supported source tools: execute_probe_query, execute_python. "
                "If the answer comes from extract_structured_doc, first extract and "
                "register the table, then submit an execute_probe_query or execute_python "
                "call against that registered table. "
                "For execute_python, the code MUST print a JSON object to stdout: "
                "print(json.dumps({'columns': [...], 'rows': [...]})). "
                "For execute_probe_query, the last successful query in the batch "
                "becomes the answer, but every query in a submitted batch must succeed "
                "or the submission fails."
            ),
            args_schema=SubmitToolResultArgs,
        ),
    }
    handlers = {
        "execute_probe_query": _execute_probe_query,
        "execute_python": _execute_python,
        "extract_structured_doc": _extract_structured_doc,
        "inspect_doc_structure": _inspect_doc_structure,
        "get_column_distinct_values": _get_column_distinct_values,
        "search_semantic_catalog": _search_semantic_catalog,
        "get_table_profile": _get_table_profile,
        "get_field_profile": _get_field_profile,
        "get_table_relationships": _get_table_relationships,
        "lookup_doc_outline": _lookup_doc_outline,
        "list_context": _list_context,
        "read_doc": _read_doc,
        "read_context_image": _read_context_image,
        "record_visual_evidence": _record_visual_evidence,
        "search_doc": _search_doc,
        "submit_tool_result": _submit_tool_result,
    }
    return ToolRegistry(
        specs=specs,
        handlers=handlers,
        tool_config=tool_config if tool_config is not None else ToolConfig(),
    )
