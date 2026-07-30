"""Document routing: select which prose files are relevant to the task question.

Given ≥2 prose files, builds a lightweight profile per file (schema fields from
knowledge.md + sampled paragraphs) and asks the LLM which files contain data
needed to answer the question. Files not selected are skipped from ETL entirely
and excluded from the virtual context.
"""

from __future__ import annotations

import csv
import json
import logging
import re
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, cast

from agents.etl._compress import sample_paragraphs_by_section
from agents.etl._constants import (
    ROUTER_MAX_DESC_CHARS,
    ROUTER_MAX_FIELDS_PER_FILE,
    ROUTER_MAX_SAMPLE_CHARS,
    ROUTER_MAX_STRUCTURED_COLUMNS,
    ROUTER_MAX_STRUCTURED_FILES,
    ROUTER_MAX_STRUCTURED_TABLES_PER_DB,
    ROUTER_SAMPLE_TOKEN_BUDGET,
)
from agents.etl._pdf import pdf_to_markdown
from agents.etl.knowledge import km_table_fields

if TYPE_CHECKING:
    from agents.llm import ModelAdapter

logger = logging.getLogger(__name__)


class FileProfile(TypedDict):
    filename: str
    stem: str
    sample: str
    schema_fields: list[str]
    schema_descriptions: dict[str, str]
    schema_source: str
    char_count: int


class StructuredTableProfile(TypedDict):
    name: str
    columns: list[str]
    row_count: int | None


class StructuredSourceProfile(TypedDict):
    path: str
    format: str
    columns: list[str]
    row_count: int | None
    tables: list[StructuredTableProfile]


STRUCTURED_SOURCE_EXTS = frozenset({".csv", ".db", ".sqlite", ".sqlite3"})


def read_prose_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return pdf_to_markdown(path)
    return path.read_text(encoding="utf-8", errors="replace")


def _infer_schema_lite(adapter: ModelAdapter, sample: str, question: str) -> list[str] | None:
    from agents.llm.types import ModelMessage

    if not sample or len(sample) < 50:
        return None

    prompt = (
        f"Below is a sample from a prose data document. Each paragraph describes one entity.\n\n"
        f"Question for context: {question}\n\n"
        f"Sample:\n{sample}\n\n"
        f"List ALL structured data field names present (snake_case). "
        f"Ignore narrative filler. Output ONLY a comma-separated list on one line."
    )
    messages = [
        ModelMessage(
            role="system", content="Extract field names from prose. One line, comma-separated."
        ),
        ModelMessage(role="user", content=prompt),
    ]
    try:
        response = adapter.complete(messages)
    except Exception as exc:
        logger.warning("Lite schema inference failed: %s", exc)
        return None

    line = response.content.strip().splitlines()[0] if response.content.strip() else ""
    fields = [f.strip().lower().replace(" ", "_") for f in line.split(",") if f.strip()]
    return fields if fields else None


def empty_file_profile(path: Path) -> FileProfile:
    return {
        "filename": path.name,
        "stem": path.stem,
        "sample": "",
        "schema_fields": [],
        "schema_descriptions": {},
        "schema_source": "error",
        "char_count": 0,
    }


def build_file_profile(
    path: Path, km_text: str, question: str, adapter: ModelAdapter | None = None
) -> FileProfile:
    text = read_prose_text(path)
    sample = sample_paragraphs_by_section(text, token_limit=ROUTER_SAMPLE_TOKEN_BUDGET)
    km_fields = km_table_fields(km_text, path.stem) if km_text else {}

    schema_fields: list[str] = list(km_fields.keys())
    schema_descriptions: dict[str, str] = dict(km_fields)
    schema_source = "knowledge.md" if km_fields else "none"

    if not km_fields and adapter is not None:
        fields = _infer_schema_lite(adapter, sample, question)
        if fields:
            schema_fields = fields
            schema_source = "inferred"

    return {
        "filename": path.name,
        "stem": path.stem,
        "sample": sample[:3000],
        "schema_fields": schema_fields,
        "schema_descriptions": schema_descriptions,
        "schema_source": schema_source,
        "char_count": len(text),
    }


def _quote_sqlite_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _relative_path(path: Path, context_dir: Path) -> str:
    try:
        return path.relative_to(context_dir).as_posix()
    except ValueError:
        return path.name


def _context_root_for_prose_files(prose_files: list[Path]) -> Path | None:
    if not prose_files:
        return None
    parent = prose_files[0].parent
    if parent.name == "doc":
        return parent.parent
    return parent


def _csv_source_profile(path: Path, context_dir: Path) -> StructuredSourceProfile | None:
    try:
        with path.open(newline="", encoding="utf-8", errors="replace") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
    except (OSError, csv.Error):
        return None

    if header is None:
        return None
    columns = [col for col in header if col]
    if not columns:
        return None
    return {
        "path": _relative_path(path, context_dir),
        "format": "csv",
        "columns": columns[:ROUTER_MAX_STRUCTURED_COLUMNS],
        "row_count": None,
        "tables": [],
    }


def _sqlite_source_profile(path: Path, context_dir: Path) -> StructuredSourceProfile | None:
    uri = f"file:{path.resolve().as_posix()}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True)
    except sqlite3.DatabaseError:
        return None

    try:
        table_rows = cast(
            list[tuple[object, ...]],
            conn.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
                ORDER BY name
                """
            ).fetchall(),
        )
        tables: list[StructuredTableProfile] = []
        for row in table_rows[:ROUTER_MAX_STRUCTURED_TABLES_PER_DB]:
            if not row or not isinstance(row[0], str):
                continue
            table_name = row[0]
            quoted = _quote_sqlite_identifier(table_name)
            try:
                info_rows = cast(
                    list[tuple[object, ...]],
                    conn.execute(f"PRAGMA table_info({quoted})").fetchall(),
                )
            except sqlite3.DatabaseError:
                continue

            columns = [info[1] for info in info_rows if len(info) > 1 and isinstance(info[1], str)]
            if not columns:
                continue

            row_count: int | None = None
            try:
                count_row = cast(
                    tuple[object, ...] | None,
                    conn.execute(f"SELECT COUNT(*) FROM {quoted}").fetchone(),
                )
            except sqlite3.DatabaseError:
                count_row = None
            if count_row and isinstance(count_row[0], int):
                row_count = count_row[0]

            tables.append(
                {
                    "name": table_name,
                    "columns": columns[:ROUTER_MAX_STRUCTURED_COLUMNS],
                    "row_count": row_count,
                }
            )

        if not tables:
            return None
        return {
            "path": _relative_path(path, context_dir),
            "format": "sqlite",
            "columns": [],
            "row_count": None,
            "tables": tables,
        }
    finally:
        conn.close()


def build_structured_source_profiles(context_dir: Path | None) -> list[StructuredSourceProfile]:
    """Return lightweight CSV/SQLite schemas available beside prose documents."""
    if context_dir is None or not context_dir.is_dir():
        return []

    candidates: list[Path] = []
    for child in context_dir.rglob("*"):
        if not child.is_file():
            continue
        try:
            rel_parts = child.relative_to(context_dir).parts
        except ValueError:
            continue
        if any(part.startswith(".") for part in rel_parts):
            continue
        if child.suffix.lower() in STRUCTURED_SOURCE_EXTS:
            candidates.append(child)

    profiles: list[StructuredSourceProfile] = []
    for path in sorted(candidates, key=lambda item: _relative_path(item, context_dir))[
        :ROUTER_MAX_STRUCTURED_FILES
    ]:
        if path.suffix.lower() == ".csv":
            profile = _csv_source_profile(path, context_dir)
        else:
            profile = _sqlite_source_profile(path, context_dir)
        if profile is not None:
            profiles.append(profile)
    return profiles


def format_structured_sources_for_prompt(sources: list[StructuredSourceProfile]) -> str:
    if not sources:
        return "(none)"

    parts: list[str] = []
    for source in sources:
        if source["format"] == "csv":
            columns = ", ".join(source["columns"])
            parts.append(f"- {source['path']} (CSV): columns: {columns}")
            continue

        table_lines = [f"- {source['path']} (SQLite):"]
        for table in source["tables"]:
            columns = ", ".join(table["columns"])
            row_suffix = f"; rows: {table['row_count']}" if table["row_count"] is not None else ""
            table_lines.append(f"  - table {table['name']}: columns: {columns}{row_suffix}")
        parts.append("\n".join(table_lines))
    return "\n".join(parts)


def format_profiles_for_prompt(profiles: list[FileProfile]) -> str:
    parts: list[str] = []
    for i, p in enumerate(profiles, 1):
        fields = p["schema_fields"][:ROUTER_MAX_FIELDS_PER_FILE]
        fields_str = ", ".join(fields) if fields else "(unknown)"
        descs = p.get("schema_descriptions", {})
        desc_str = "; ".join(f"{k}={v}" for k, v in list(descs.items())[:8]) if descs else ""
        parts.append(
            f"[{i}] {p['filename']}\n"
            f"    Fields: {fields_str}\n"
            f"    Descriptions: {desc_str[:ROUTER_MAX_DESC_CHARS]}\n"
            f"    Sample: {p['sample'][:ROUTER_MAX_SAMPLE_CHARS]}\n"
        )
    return "\n".join(parts)


def route_file_profiles(
    adapter: ModelAdapter,
    question: str,
    profiles: list[FileProfile],
    structured_sources: list[StructuredSourceProfile] | None = None,
) -> list[str]:
    """Ask the LLM which file profiles are relevant. Returns selected stems."""
    from agents.llm.types import ModelMessage

    profiles_text = format_profiles_for_prompt(profiles)
    structured_text = format_structured_sources_for_prompt(structured_sources or [])
    filenames = [p["filename"] for p in profiles]

    prompt = (
        f"Question: {question}\n\n"
        f"Prose document candidates for ETL:\n{profiles_text}\n\n"
        f"Structured sources already available (not output candidates):\n{structured_text}\n\n"
        f"Select prose documents whose fields are needed to answer the question when "
        f"combined with the structured sources above.\n"
        f"Rules:\n"
        f"- Output ONLY prose filenames from Available.\n"
        f"- Include master/archive/dimension prose documents that provide join keys, "
        f"group-by labels, entity categories, or code-to-label mappings.\n"
        f"- If filter or metric fields already exist in CSV/SQLite sources, select prose "
        f"documents for the missing dimensions/labels instead of preferring a prose "
        f"ranking or metric document solely because it contains similar field names.\n"
        f"Output a JSON array of filenames. Available: {json.dumps(filenames)}"
    )
    messages = [
        ModelMessage(role="system", content="Select relevant documents. Output ONLY a JSON array."),
        ModelMessage(role="user", content=prompt),
    ]

    response = adapter.complete(messages)
    content = response.content.strip()
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

    match = re.search(r"\[.*?\]", content, re.DOTALL)
    if not match:
        logger.warning("Router returned unparseable response, selecting all: %s", content[:200])
        return [p["stem"] for p in profiles]

    try:
        selected = cast(list[object], json.loads(match.group()))
    except json.JSONDecodeError:
        logger.warning("Router JSON parse failed, selecting all: %s", content[:200])
        return [p["stem"] for p in profiles]

    logger.info("Router raw selection: %s", selected)

    name_to_stem = {p["filename"]: p["stem"] for p in profiles}
    stem_to_stem = {p["stem"]: p["stem"] for p in profiles}
    name_lower_to_stem = {p["filename"].lower(): p["stem"] for p in profiles}
    stem_lower_to_stem = {p["stem"].lower(): p["stem"] for p in profiles}
    result: list[str] = []
    seen: set[str] = set()
    for item in selected:
        name = str(item).strip()
        matched: str | None = None
        if name in name_to_stem:
            matched = name_to_stem[name]
        elif name in stem_to_stem:
            matched = stem_to_stem[name]
        else:
            stem = Path(name).stem
            nl = name.lower()
            sl = stem.lower()
            if nl in name_lower_to_stem:
                matched = name_lower_to_stem[nl]
            elif sl in stem_lower_to_stem:
                matched = stem_lower_to_stem[sl]
        if matched and matched not in seen:
            seen.add(matched)
            result.append(matched)

    if not result:
        logger.warning("Router selected no valid files (raw: %s), returning all", selected)
        return [p["stem"] for p in profiles]
    return result


def route_prose_files(
    adapter: ModelAdapter,
    question: str,
    prose_files: list[Path],
    km_text: str,
) -> list[Path]:
    """Select prose files relevant to the task question.

    Returns the subset of prose_files that should undergo ETL extraction.
    Conservative: returns all files on parse failure or when fewer than 2 files.
    """
    if len(prose_files) < 2:
        return list(prose_files)

    profiles: list[FileProfile] = []
    for pf in prose_files:
        try:
            profiles.append(build_file_profile(pf, km_text, question, adapter=adapter))
        except Exception as exc:
            logger.warning("Failed to build profile for %s: %s", pf.name, exc)
            profiles.append(empty_file_profile(pf))

    try:
        structured_sources = build_structured_source_profiles(
            _context_root_for_prose_files(prose_files)
        )
        selected_stems = route_file_profiles(adapter, question, profiles, structured_sources)
    except Exception as exc:
        logger.warning("Router LLM call failed, selecting all: %s", exc)
        return list(prose_files)

    stem_to_path = {pf.stem: pf for pf in prose_files}
    result: list[Path] = []
    seen: set[str] = set()
    for stem in selected_stems:
        matched = stem_to_path.get(stem)
        if matched and matched.name not in seen:
            seen.add(matched.name)
            result.append(matched)

    if not result:
        logger.warning("Router selected no valid files (raw: %s), returning all", selected_stems)
        return list(prose_files)

    skipped = [pf.name for pf in prose_files if pf not in result]
    if skipped:
        logger.info("Router: skipping %d files: %s", len(skipped), skipped)
    return result
