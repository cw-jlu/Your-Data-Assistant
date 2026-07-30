"""Deterministic ETL-source completion for explorer findings."""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from agents.benchmark.schema import PublicTask

_IDENT = r"[A-Za-z_][A-Za-z0-9_]*"
_BACKTICK_IDENT_RE = re.compile(rf"`({_IDENT}(?:\.{_IDENT})?)`")
_TABLE_DOT_COLUMN_RE = re.compile(rf"\b({_IDENT})\.{_IDENT}\b")
_PROSE_EXTS = frozenset({".md", ".txt", ".pdf"})


def _flatten_strings(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        mapping = cast(dict[Any, Any], value)
        return "\n".join(_flatten_strings(item) for item in mapping.values())
    if isinstance(value, list | tuple):
        sequence = cast(Sequence[Any], value)
        return "\n".join(_flatten_strings(item) for item in sequence)
    return ""


def _knowledge_question_slice(task: PublicTask) -> str:
    km_path = task.context_dir / "knowledge.md"
    if not km_path.is_file():
        return ""
    try:
        text = km_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    question = task.question.strip()
    if not question:
        return ""
    index = text.find(question)
    if index < 0:
        return ""
    section_start = text.rfind("\n###", 0, index)
    section_end = text.find("\n###", index)
    start = section_start if section_start >= 0 else max(0, index - 1000)
    end = section_end if section_end >= 0 else min(len(text), index + 2000)
    return text[start:end]


def _mentioned_identifiers(text: str) -> set[str]:
    identifiers: set[str] = set()
    for match in _BACKTICK_IDENT_RE.finditer(text):
        identifiers.add(match.group(1).split(".", 1)[0].lower())
    for match in _TABLE_DOT_COLUMN_RE.finditer(text):
        identifiers.add(match.group(1).lower())
    return identifiers


def _prose_sources_by_stem(task: PublicTask) -> dict[str, str]:
    sources: dict[str, str] = {}
    if not task.context_dir.is_dir():
        return sources
    for path in sorted(task.context_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in _PROSE_EXTS:
            continue
        if path.name.lower() == "knowledge.md":
            continue
        try:
            rel_path = path.relative_to(task.context_dir).as_posix()
        except ValueError:
            continue
        sources.setdefault(path.stem.lower(), rel_path)
    return sources


def _add_sqlite_table_names(names: set[str], tables: Any) -> None:
    if isinstance(tables, dict):
        table_map = cast(dict[Any, Any], tables)
        names.update(str(name).lower() for name in table_map)
        return
    if not isinstance(tables, list):
        return
    table_list = cast(list[Any], tables)
    for table in table_list:
        if isinstance(table, str):
            names.add(table.lower())
        elif isinstance(table, dict):
            table_info = cast(dict[str, Any], table)
            name = table_info.get("name")
            if name is not None:
                names.add(str(name).lower())


def _structured_source_names(content: dict[str, Any]) -> set[str]:
    names: set[str] = set()

    raw_schema_map = content.get("schema_map")
    if isinstance(raw_schema_map, dict):
        schema_map = cast(dict[Any, Any], raw_schema_map)
        for key in schema_map:
            key_text = str(key)
            names.add((Path(key_text).stem if "/" in key_text else key_text).lower())

    raw_files = content.get("files")
    if not isinstance(raw_files, list):
        return names

    files = cast(list[Any], raw_files)
    for raw_entry in files:
        if not isinstance(raw_entry, dict):
            continue
        entry = cast(dict[str, Any], raw_entry)
        _add_sqlite_table_names(names, entry.get("tables"))

        schema = entry.get("schema")
        if isinstance(schema, dict):
            schema_info = cast(dict[str, Any], schema)
            _add_sqlite_table_names(names, schema_info.get("tables"))

        path = entry.get("path")
        fmt = str(entry.get("format", "")).lower()
        if path and fmt in {"csv", "json", "sqlite", "sqlite3", "database"}:
            names.add(Path(str(path)).stem.lower())

    return names


def required_knowledge_doc_etl_sources(
    task: PublicTask, content: dict[str, Any]
) -> list[dict[str, str]]:
    """Return required same-stem prose ETL sources implied by knowledge.md."""
    question_slice = _knowledge_question_slice(task)
    evidence_text = question_slice or _flatten_strings(content.get("knowledge", {}))
    if not evidence_text:
        return []

    prose_by_stem = _prose_sources_by_stem(task)
    if not prose_by_stem:
        return []

    structured_names = _structured_source_names(content)
    mentioned = _mentioned_identifiers(evidence_text)
    missing_doc_stems = sorted(
        stem for stem in mentioned if stem in prose_by_stem and stem not in structured_names
    )
    return [
        {
            "path": prose_by_stem[stem],
            "reason": (
                f"knowledge.md references table {stem}, no exact structured "
                f"source named {stem} was reported, and this same-stem "
                "document contains the table records"
            ),
        }
        for stem in missing_doc_stems
    ]


def ensure_knowledge_doc_etl_sources(task: PublicTask, content: dict[str, Any]) -> None:
    """Force same-stem prose ETL when knowledge names a missing exact table."""

    required = required_knowledge_doc_etl_sources(task, content)
    if not required:
        return

    raw_etl_sources = content.get("etl_sources")
    etl_sources: list[dict[str, str]]
    if isinstance(raw_etl_sources, list):
        etl_sources = cast(list[dict[str, str]], raw_etl_sources)
    else:
        etl_sources = []
        content["etl_sources"] = etl_sources

    existing_paths = {str(entry.get("path")) for entry in etl_sources if entry.get("path")}
    for source in required:
        path = source["path"]
        if path in existing_paths:
            continue
        etl_sources.append(source)
        existing_paths.add(path)
