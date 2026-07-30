from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field


class SubmitToolResultArgs(BaseModel):
    tool_name: str = Field(
        description=(
            "The source tool to RE-EXECUTE from scratch to produce the answer. "
            "The system will run this tool fresh with tool_args; it does NOT reuse "
            "any previous tool output. "
            "Use 'execute_probe_query' when the answer is a direct SQL query result. "
            "Use 'execute_python' when data needs transformation, formatting, or "
            "computation before submission (e.g., converting datetime to ISO 8601). "
            "Use 'extract_structured_doc' only when its extracted table itself is the "
            "final answer; it requires a matching inspect_doc_structure cache from an "
            "earlier call in the same task workspace."
        )
    )
    tool_args: dict[str, Any] = Field(
        description=(
            "The arguments passed to the source tool for a fresh execution. "
            "Must be a dict with the exact same keys and format as calling the "
            "tool directly. "
            "Examples by tool_name: "
            "- 'execute_probe_query': {'queries': ['SELECT col1, col2 FROM table']} "
            "  (limit is ignored for final submission). "
            "- 'execute_python': {'code': 'import json\\n...\\n"
            "print(json.dumps({\"columns\": [...], \"rows\": [...]}))'} "
            "  (the code must print a JSON object with columns and rows to stdout). "
            "- 'extract_structured_doc': {'path': 'doc/table.md', "
            "'target_table': 'table', 'fields': ['key', 'metric']} "
            "  (only after inspect_doc_structure established the matching cache). "
            "For a submitted SQL batch, every query must succeed; a failed supporting "
            "probe makes the fresh submission execution fail."
        )
    )
    columns: list[str] | None = Field(
        default=None,
        description=(
            "Optional: override or rename the answer columns. "
            "MUST be a Python list of strings, e.g. ['col1', 'col2']. "
            "Do NOT pass a JSON string like '[\"col1\", \"col2\"]' — that will cause "
            "a type error. "
            "If omitted, column names are extracted from the tool output automatically."
        ),
    )



class ExecutePythonArgs(BaseModel):
    code: str = Field(
        description=(
            "Python code to execute inside the task's temporary context workspace. "
            "The helpers query(sql) and query_rows(sql) are already injected as global "
            "functions; call them directly, e.g. rows = query_rows('SELECT ...'). "
            "Do not import them: there is no `query` module, so never write "
            "`from query import query_rows`. query(sql) returns {'columns': [...], "
            "'rows': [[...]]}; query_rows(sql) returns only list-of-list rows, not "
            "dict rows. These helpers query the same logical tables exposed to "
            "execute_probe_query. Do not use a bare duckdb.connect(':memory:') for "
            "logical tables. "
            "Read files by paths relative to context. For final results, print full "
            "machine-readable JSON rather than pandas previews."
        )
    )


class ExecuteProbeQueryArgs(BaseModel):
    queries: list[str] = Field(
        description=(
            "A list of read-only SQL queries to execute against task data files. "
            "Each query must be SELECT or WITH. "
            "All queries share the same in-memory DuckDB connection and views, "
            "so you can batch multiple lookups into one efficient call — "
            "e.g., send COUNT + DISTINCT + sample rows together instead of "
            "three separate calls. "
            "CSV/JSON files are accessible by their file-name stem (no extension), "
            "e.g., 'trans' or 'member'. SQLite tables by their table name, "
            "or by 'asset_path_stem__table_name' if the name collides. "
            "Asset paths with directory prefixes can also be used, e.g., "
            "'csv/trans.csv' is resolved to the view 'trans'. "
            "IMPORTANT: do NOT wrap table references in single quotes in SQL. "
            "Write FROM qualifying, not FROM 'qualifying' or FROM 'csv/qualifying.csv'. "
            "Single quotes create string literals, not table references."
        ),
    )
    limit: int = Field(
        default=5,
        description="Preview row limit per query (default 5, max 200). Final submission via submit_tool_result is not limited by this.",
    )


class GetColumnDistinctValuesArgs(BaseModel):
    table: str = Field(
        description=(
            "The table name to inspect. For CSV/JSON this is the file-name stem "
            "(e.g., 'member' for 'csv/member.csv'). For SQLite this is the table "
            "name (e.g., 'users'). This tool supports base CSV/JSON/SQLite-backed "
            "tables only; use execute_probe_query for derived views or structured-"
            "document extracted tables."
        ),
    )
    column: str = Field(description="The column/field name to inspect.")
    top_n: int = Field(
        default=20, description="Maximum number of distinct values to return, ranked by frequency."
    )


class SearchSemanticCatalogArgs(BaseModel):
    query: str = Field(
        description=(
            "Case-insensitive keyword substring to search across logical table names, "
            "columns, documents, and relationships. This is not semantic retrieval; "
            "prefer concise exact stems or field tokens."
        )
    )
    scope: str = Field(
        default="all",
        description="One of all, tables, fields, documents, relationships, uncertainties.",
    )
    limit: int = Field(default=20, description="Maximum number of matches to return.")


class GetTableProfileArgs(BaseModel):
    table: str = Field(
        description=(
            "SQL-visible logical table or derived-view name from query_surfaces. Do not "
            "pass an exact document stem; route documents through document tools instead."
        )
    )


class GetFieldProfileArgs(BaseModel):
    table: str = Field(description="Logical table name from the lightweight catalog.")
    column: str = Field(description="Column name to inspect.")


class GetTableRelationshipsArgs(BaseModel):
    table: str = Field(
        description=(
            "Logical table name from the lightweight catalog. Returned inferred "
            "relationships are candidate join paths and require evidence verification."
        )
    )


class ReadContextImageArgs(BaseModel):
    path: str = Field(
        description=(
            "Relative jpg, jpeg, png, or webp image path under the task context "
            "directory. The image is attached to the next model request."
        )
    )
    detail: str = Field(default="auto", description="Image detail hint: auto, low, or high.")


class RecordVisualEvidenceArgs(BaseModel):
    path: str = Field(
        description=(
            "Relative path of a stable frame already opened with read_context_image in "
            "this run. Use the exact same path."
        )
    )
    observations: str = Field(
        description=(
            "Concise observation made after viewing that frame. Preserve material visible "
            "text, values, punctuation, spacing, and separators exactly."
        )
    )


class LookupDocOutlineArgs(BaseModel):
    path: str = Field(
        description="Relative path to a text document under the task context directory. Use the path exactly as listed by list_context and do not prefix it with `context/`."
    )


class ListContextArgs(BaseModel):
    max_depth: int = Field(
        default=4,
        description="Maximum directory recursion depth to list. Use a small depth first unless paths are still missing.",
    )


class ReadDocArgs(BaseModel):
    path: str = Field(
        description="Relative path to a text-like document under the task context directory. Use the path exactly as listed by list_context and do not prefix it with `context/`."
    )
    heading: str | None = Field(
        default=None,
        description="Optional heading name to read a specific section. Prefer the exact heading text from lookup_doc_outline. If numbering is omitted, read_doc can also match the corresponding numbered heading case-insensitively. Content is extracted from the matched heading until the next heading of the same or higher level.",
    )


class SearchDocArgs(BaseModel):
    query: str = Field(
        description="Regex pattern or plain text keyword to search for. Case-insensitive. Use character classes like \\d, \\w, \\s for flexible matching. Examples: 'patient \\d{6,7}', 'creatinine', 'normal range', 'TR\\d{3}'."
    )
    context_lines: int = Field(
        default=5,
        description="Number of lines before and after each matching line to include for context.",
    )
    path: str | None = Field(
        default=None,
        description="Optional: restrict search to a single document file by its relative path. If omitted, searches all .md, .txt, and .rst files under context.",
    )
    page: int = Field(
        default=1,
        ge=1,
        description="Page number (1-indexed) of results to return. Use this along with page_size to paginate through large result sets.",
    )
    page_size: int = Field(
        default=20,
        ge=0,
        description="Number of matches per page. Set to 0 to return all matches (no pagination). Default is 20.",
    )


class ExtractStructuredDocArgs(BaseModel):
    path: str = Field(
        description=(
            "Relative path to a Markdown/text document under context. Use when a "
            "domain table is stored as a .md/.txt document; the tool extracts visible "
            "facts and merges them by entity key when fields are spread across lines."
        )
    )
    knowledge_path: str = Field(
        default="knowledge.md",
        description="Relative path to the task knowledge document that defines target fields.",
    )
    target_table: str | None = Field(
        default=None,
        description=(
            "Target table/entity name whose fields should be extracted. "
            "Defaults to the document stem."
        ),
    )
    fields: list[str] | None = Field(
        default=None,
        description=(
            "Exact minimal subset of fields to extract, including join keys when needed. "
            "This drives automatic cached-block selection; reuse the same fields with "
            "inspect_doc_structure. Omit only when the task genuinely needs all fields "
            "for target_table. Manual block/range selection parameters are not exposed."
        ),
    )
    max_model_calls: int = Field(
        default=20,
        ge=1,
        le=20,
        description=(
            "Maximum model calls the tool may spend on schema/extraction/repair. "
            "The effective value is capped by the configured structured_doc hard limit."
        ),
    )


class InspectDocStructureArgs(BaseModel):
    path: str = Field(
        description=(
            "Relative path to a Markdown/text document under context. Use this "
            "before extract_structured_doc when a natural-language document carries "
            "structured data across sections. This tool auto-discovers all candidate "
            "fields and blocks in the document; do NOT supply a fields list. Read the "
            "returned block summary to decide which fields to pass to "
            "extract_structured_doc."
        )
    )
    knowledge_path: str = Field(
        default="knowledge.md",
        description="Relative path to the task knowledge document that defines target fields.",
    )
    target_table: str | None = Field(
        default=None,
        description="Target table/entity name. Defaults to the document stem.",
    )
    max_model_calls: int = Field(
        default=3,
        ge=1,
        le=3,
        description=(
            "Maximum model calls for block classification. The effective value is capped "
            "by the configured structured_doc inspect limit."
        ),
    )


def create_structured_tool(
    *,
    name: str,
    description: str,
    args_schema: type[BaseModel],
    invoke: Callable[..., dict[str, Any]],
) -> BaseTool:
    return StructuredTool.from_function(
        func=invoke,
        name=name,
        description=description,
        args_schema=args_schema,
    )
