"""Explorer sub-agent tool registry: discovery tools + report terminal tool."""

from __future__ import annotations

import json
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agents.benchmark.schema import PublicTask
from agents.explorer.etl_sources import (
    ensure_knowledge_doc_etl_sources,
    required_knowledge_doc_etl_sources,
)
from agents.tools.execute_sql import execute_context_sql
from agents.tools.grep_context import grep_context
from agents.tools.inspect_files import inspect_files
from agents.tools.preview import preview_file
from agents.tools.registry import FunctionTool, ToolExecutionResult, ToolRegistry
from agents.tools.schema_normalize import normalize_for_vllm


def coerce_dict_field(val: Any) -> Any:
    """Convert a list-of-objects to a keyed dict if the model sent the wrong shape."""
    if isinstance(val, dict):
        return cast(dict[str, Any], val)
    if not isinstance(val, list):
        return val
    raw_items = cast(list[Any], val)
    if any(not isinstance(item, dict) for item in raw_items):
        return raw_items
    items = cast(list[dict[str, Any]], raw_items)
    out: dict[str, Any] = {}
    for item in items:
        key: str | None = item.get("name") or item.get("table") or item.get("file")
        if key:
            out[str(key)] = {k: v for k, v in item.items() if k not in ("name", "table", "file")}
            continue
        out[f"_item_{len(out)}"] = item
    return out


class ReportInput(BaseModel):
    """Explorer report terminal tool input — structured fields."""

    model_config = ConfigDict(extra="forbid")

    files: list[dict[str, Any]] = Field(  # pyright: ignore[reportUnknownVariableType]
        default_factory=list,
        description="Flat list of file entries, one per file.",
    )
    schema_map: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Per-table/file schema: columns with dtypes, semantics, "
            "and column_conversions if inspect_files provided them."
        ),
    )
    knowledge: dict[str, Any] = Field(
        default_factory=dict,
        description="Extracted mappings, formulas, units, thresholds from docs.",
    )
    etl_sources: list[dict[str, str]] = Field(  # pyright: ignore[reportUnknownVariableType]
        default_factory=list,
        description=(
            "Documents that the main agent must convert with run_etl before "
            "analysis. If knowledge.md names exact table X, no structured "
            "source named X exists, and a same-stem doc/X.pdf exists, include "
            "that document; similar table names are not substitutes. "
            "Example: [{'path': 'doc/source.pdf', 'reason': 'contains the "
            "required transaction table'}]."
        ),
    )
    join_paths: list[dict[str, str]] = Field(  # pyright: ignore[reportUnknownVariableType]
        default_factory=list,
        description='Discovered relationships, e.g. [{"left": "a.col", "right": "b.col"}].',
    )
    value_samples: dict[str, list[Any]] = Field(
        default_factory=dict,
        description="Sampled distinct values for filter-relevant columns.",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Objective data-quality observations (one atomic fact each).",
    )

    @field_validator("schema_map", "knowledge", "value_samples", mode="before")
    @classmethod
    def _coerce_list_to_dict(cls, v: Any) -> Any:
        return coerce_dict_field(v)


REPORT_SCHEMA = normalize_for_vllm(ReportInput.model_json_schema())

REPORT_KEYS = (
    "files",
    "schema_map",
    "knowledge",
    "etl_sources",
    "join_paths",
    "value_samples",
    "warnings",
)


def handle_report(_task: PublicTask, args: Any) -> ToolExecutionResult:
    d: dict[str, Any] = cast(
        dict[str, Any],
        args if isinstance(args, dict) else {k: getattr(args, k, None) for k in REPORT_KEYS},
    )
    report = ReportInput.model_validate(d)
    content: dict[str, Any] = {
        "files": report.files,
        "schema_map": report.schema_map,
        "knowledge": report.knowledge,
        "etl_sources": report.etl_sources,
        "join_paths": report.join_paths,
        "value_samples": report.value_samples,
        "warnings": report.warnings,
    }
    required_etl_sources = required_knowledge_doc_etl_sources(_task, content)
    etl_sources = report.etl_sources
    existing_paths = {str(entry.get("path")) for entry in etl_sources if entry.get("path")}
    missing_etl_sources = [
        source for source in required_etl_sources if source["path"] not in existing_paths
    ]
    if missing_etl_sources:
        missing_paths = ", ".join(source["path"] for source in missing_etl_sources)
        return ToolExecutionResult(
            ok=False,
            content={
                "status": "rejected",
                "error": (
                    "report.etl_sources is missing required same-stem document "
                    f"source(s): {missing_paths}. knowledge.md named an exact "
                    "table that is absent from structured schemas; similar "
                    "CSV/JSON/SQLite table names are not substitutes. Call "
                    "report again with these required_etl_sources."
                ),
                "required_etl_sources": missing_etl_sources,
            },
            is_terminal=False,
        )
    ensure_knowledge_doc_etl_sources(_task, content)
    findings = json.dumps(content, ensure_ascii=False)
    return ToolExecutionResult(
        ok=True,
        content={"status": "reported", "findings": findings},
        is_terminal=True,
    )


def create_explorer_tool_registry(
    *,
    video_tool: FunctionTool | None = None,
) -> ToolRegistry:
    """Discovery tool registry for the Explorer sub-agent.

    Includes inspection/preview tools from top-level tool modules, plus ``report``
    as the only terminal tool. Excludes ``execute_python`` and ``answer``.
    """
    definitions: list[FunctionTool] = [
        inspect_files.with_description(
            "Use FIRST to bootstrap discovery. Returns a one-shot summary of "
            "every file under context: {path, size, format, schema} where schema "
            "contains columns, dtypes, row_count, profile (min/max/null_rate/ "
            "distinct values) for CSV/SQLite, keys/length for JSON, line_count/ "
            "head for docs, and page_count/paragraph_count/entity_groups_sample "
            "for PDF. Includes column_conversions when unit metadata is "
            "available. Does NOT read full data — use preview_file or "
            "execute_context_sql for deeper inspection.",
        ),
        preview_file.with_description(
            "Preview a single context file. Auto-detects format by extension. "
            "CSV → {columns, dtypes, row_count, head (20 rows), tail (5 rows)}. "
            "JSON → {kind, keys/length, head/value_preview}. "
            "SQLite → {tables: [{name, create_sql, row_count}]}. "
            "PDF → {page_count, paragraph_count, entity_groups_sample}; "
            "inspect_files already returns this PDF entity-group shape, so use "
            "preview_file on a PDF only for a targeted single-file re-check. "
            "Text → {total_lines, head (first 100 lines), has_more}. "
            "Sample only, not full data access. Do NOT use for aggregation — "
            "use execute_context_sql for SQL or note in report that full-file "
            "processing is needed. "
            'Example: preview_file({"path": "csv/member.csv"})',
        ),
        execute_context_sql.with_description(
            "Run SQL on a virtual-context SQLite database copy. Supports "
            "SELECT/WITH/PRAGMA, EXPLAIN QUERY PLAN, and local "
            "CREATE INDEX IF NOT EXISTS for faster discovery on large tables. "
            "Returns {columns, rows, row_count, truncated}; default limit 200. "
            "Use for SELECT DISTINCT, MIN/MAX, sample queries, and lightweight "
            "aggregations to discover value distributions or verify join keys. "
            "Do NOT call on .csv/.json/.md/.txt paths. "
            'Example: execute_context_sql({"path": "db/sales.db", '
            '"sql": "SELECT DISTINCT category FROM products"})',
        ),
        grep_context.with_description(
            "Search for a regex pattern across ALL context files at once: CSV "
            "values, JSON fields, Markdown text, and SQLite text columns. Use "
            "to check whether a specific value exists in any file without "
            "reading each one. Do NOT use for counting or aggregation — use "
            "execute_context_sql for that. Returns up to 30 matches with file "
            "path and location; for SQLite reports {table, column, "
            "total_matches, samples}. "
            'Example: grep_context({"pattern": "全国"})',
        ),
    ]
    if video_tool is not None:
        definitions.append(video_tool)
    definitions.append(
        FunctionTool(
            name="report",
            description=(
                "Submit your exploration findings. This is the ONLY way to complete "
                "the exploration — call it when you have mapped the data landscape. "
                "Pass each field as a separate parameter: files (flat list of file "
                "entries), schema_map (per-table/file columns with dtypes and "
                "semantics), knowledge (extracted mappings/formulas/units from docs), "
                "etl_sources (documents the caller must convert with run_etl; "
                "empty is rejected when knowledge.md names a missing exact table "
                "that exists as a same-stem document), "
                "join_paths (discovered relationships), value_samples (sampled "
                "distinct values for filter-relevant columns), warnings (data-quality "
                "observations)."
            ),
            json_schema=REPORT_SCHEMA,
            handler=handle_report,
            is_terminal=True,
            input_model=ReportInput,
        ),
    )
    return ToolRegistry(definitions={d.name: d for d in definitions})
