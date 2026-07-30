"""preview_file tool: unified single-file preview dispatched by extension."""

from __future__ import annotations

import hashlib
import random
import re
from pathlib import Path
from typing import Annotated, Any

from agents.benchmark.schema import PublicTask
from agents.etl._anchors import discover_record_id_patterns, paragraph_record_ids
from agents.etl._pdf import page_to_paragraphs, stitch_page_paragraphs
from agents.tools import constants
from agents.tools._fields import path_field
from agents.tools.context import resolve_context_path
from agents.tools.decorator import function_tool
from agents.tools.registry import ToolExecutionResult

PREVIEW_DOC_HEAD_LINES = 100
PDF_SECTION_BREAK_PREFIXES = (
    "title:",
    "chapter ",
    "section ",
    "part ",
    "i. ",
    "ii. ",
    "iii. ",
    "iv. ",
    "v. ",
    "法证审查报告",
    "审查报告",
    "卷宗",
    "审查对象",
    "审查目的",
    "报告主题",
)
PDF_SENTENCE_END = re.compile(r"[。！？.!?」）)\"']\s*$")


def _preview_doc(path: Path) -> dict[str, Any]:
    """Preview for text documents.  knowledge.md is returned in full."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    total = len(lines)
    if (
        path.name.lower() == "knowledge.md"
        and path.stat().st_size <= constants.KNOWLEDGE_MD_PRELOAD_LIMIT
    ):
        return {"format": "text", "total_lines": total, "lines": lines}
    head = lines[:PREVIEW_DOC_HEAD_LINES]
    return {
        "format": "text",
        "total_lines": total,
        "head": head,
        "has_more": total > PREVIEW_DOC_HEAD_LINES,
    }


def _extract_pdf_paragraphs(path: Path) -> tuple[int, list[dict[str, Any]]]:
    """Extract PDF paragraphs with the same page-boundary stitching used by ETL."""
    import pymupdf  # pyright: ignore[reportMissingImports]

    doc = pymupdf.open(str(path))  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
    try:
        page_count: int = len(doc)  # pyright: ignore[reportUnknownArgumentType]
        pages: list[tuple[int, list[str]]] = []
        for index in range(page_count):
            paragraphs = page_to_paragraphs(doc[index])  # pyright: ignore[reportIndexIssue]
            if paragraphs:
                pages.append((index + 1, paragraphs))

        extracted: list[dict[str, Any]] = []
        for page_no, paragraphs in stitch_page_paragraphs(pages):
            for paragraph in paragraphs:
                text = paragraph.strip()
                if not text:
                    continue
                extracted.append(
                    {
                        "paragraph_index": len(extracted) + 1,
                        "page": page_no,
                        "text": text,
                    }
                )
        return page_count, extracted
    finally:
        doc.close()  # pyright: ignore[reportUnknownMemberType]


def _is_pdf_section_break(text: str) -> bool:
    flat = " ".join(text.split())
    if not flat:
        return True
    lower = flat.lower()
    if any(lower.startswith(prefix) for prefix in PDF_SECTION_BREAK_PREFIXES):
        return True
    if len(flat) <= 80 and not PDF_SENTENCE_END.search(flat):
        return True
    return len(flat) <= 140 and ("：" in flat or ":" in flat) and not PDF_SENTENCE_END.search(flat)


def _group_pdf_paragraphs_by_entity(
    paragraphs: list[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], int, list[str]]:
    patterns = discover_record_id_patterns([str(paragraph["text"]) for paragraph in paragraphs])
    groups: dict[str, list[dict[str, Any]]] = {}
    current_entities: list[str] = []
    ungrouped_count = 0

    for paragraph in paragraphs:
        record_ids = paragraph_record_ids(str(paragraph["text"]), patterns)
        if record_ids:
            current_entities = record_ids

        if not current_entities:
            ungrouped_count += 1
            continue
        if not record_ids and _is_pdf_section_break(str(paragraph["text"])):
            current_entities = []
            ungrouped_count += 1
            continue
        if not record_ids and len(current_entities) > 1:
            current_entities = []
            ungrouped_count += 1
            continue

        for record_id in current_entities:
            groups.setdefault(record_id, []).append(paragraph)

    return groups, ungrouped_count, [name for name, _pattern in patterns]


def _sample_pdf_entity_ids(path: Path, entity_ids: list[str]) -> list[str]:
    sample_size = min(constants.PREVIEW_PDF_ENTITY_SAMPLE_SIZE, len(entity_ids))
    if sample_size == len(entity_ids):
        return entity_ids

    joined_entity_ids = "\n".join(entity_ids)
    seed_material = f"{path.as_posix()}\n{joined_entity_ids}"
    seed = int.from_bytes(hashlib.sha256(seed_material.encode("utf-8")).digest()[:8], "big")
    sampled = set(random.Random(seed).sample(entity_ids, sample_size))
    return [entity_id for entity_id in entity_ids if entity_id in sampled]


def preview_pdf_summary(path: Path) -> dict[str, Any]:
    page_count, paragraphs = _extract_pdf_paragraphs(path)
    groups, ungrouped_count, anchor_patterns = _group_pdf_paragraphs_by_entity(paragraphs)
    entity_ids = list(groups)
    sampled_entity_ids = _sample_pdf_entity_ids(path, entity_ids)

    return {
        "format": "pdf",
        "page_count": page_count,
        "paragraph_count": len(paragraphs),
        "entity_grouping": "record_id_contiguous_paragraphs",
        "entity_anchor_patterns": anchor_patterns,
        "entity_group_count": len(groups),
        "entity_sample_size": len(sampled_entity_ids),
        "entity_groups_sample": [
            {
                "entity_id": entity_id,
                "paragraph_count": len(groups[entity_id]),
                "paragraphs": groups[entity_id],
            }
            for entity_id in sampled_entity_ids
        ],
        "ungrouped_paragraph_count": ungrouped_count,
    }


@function_tool
def preview_file(
    task: PublicTask,
    path: Annotated[
        str,
        path_field(
            file_kind="any context file",
            examples=(
                "csv/member.csv",
                "db/sales.db",
                "knowledge.md",
                "json/Patient.json",
                "doc/profiles.pdf",
            ),
        ),
    ],
) -> ToolExecutionResult:
    """Preview a single context file. Automatically detects the format by
    extension and returns a structured summary. Supported formats:
    CSV → {columns, dtypes, row_count, head (20 rows), tail (5 rows)}.
    JSON → structured summary by kind: array (length + head), object
    (keys + value_preview), scalar (value). Files >100MB streamed.
    SQLite (.db/.sqlite) → {tables: [{name, create_sql, row_count}]}
    showing CREATE TABLE statements and row counts for all tables.
    PDF → {page_count, paragraph_count, entity_groups_sample}, where
    entity_groups_sample contains up to 3 deterministically sampled record-id
    entity groups with all contiguous paragraphs for each sampled entity.
    Text (md/txt) → {total_lines, head (first 100 lines), has_more};
    knowledge.md is returned in full (never truncated).
    For PDFs, inspect_files already returns the same entity-group shape; use
    preview_file on a PDF for a targeted single-file re-check. Use when you
    need to inspect one file's structure or sample values after inspect_files.
    This is a SAMPLE preview, not full data access
    — for filtering, aggregation, joins, or full-file scans, use
    execute_python. For SQL queries on SQLite, use execute_context_sql.
    Example: preview_file({"path": "csv/a.csv"})
    Example: preview_file({"path": "doc/b.pdf"})"""
    resolved = resolve_context_path(task, path)
    suffix = resolved.suffix.lower()

    if suffix in constants.CSV_EXTS:
        from agents.tools.read_csv import read_csv_preview

        return ToolExecutionResult(ok=True, content=read_csv_preview(task, path))

    if suffix in constants.JSON_EXTS:
        from agents.tools.read_json import read_json_preview

        return ToolExecutionResult(ok=True, content=read_json_preview(task, path))

    if suffix in constants.SQLITE_EXTS:
        from agents.tools.inspect_sqlite import inspect_sqlite_database

        return ToolExecutionResult(ok=True, content=inspect_sqlite_database(resolved))

    if suffix in constants.PDF_EXTS:
        return ToolExecutionResult(ok=True, content=preview_pdf_summary(resolved))

    if suffix in constants.DOC_EXTS:
        return ToolExecutionResult(ok=True, content=_preview_doc(resolved))

    raise ValueError(
        f"Unsupported format '{suffix}' for preview_file. "
        "Use execute_python to read this file type."
    )
