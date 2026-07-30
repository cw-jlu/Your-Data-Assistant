"""read_doc tool: paged text preview with keyword search."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from pydantic import Field

from agents.benchmark.schema import PublicTask
from agents.tools import constants
from agents.tools._fields import path_field, positive_int_field
from agents.tools.context import resolve_context_path
from agents.tools.decorator import function_tool
from agents.tools.registry import ToolExecutionResult


def _normalize_line_count(line_count: int) -> int:
    """Keep line windows inside a stable observation range."""
    return min(300, max(1, int(line_count)))


def _split_paragraphs(lines: list[str]) -> list[tuple[int, int, str]]:
    """Split lines into blank-line-delimited paragraphs."""
    paragraphs: list[tuple[int, int, str]] = []
    current: list[str] = []
    current_start = 0
    for one_based_idx, line in enumerate(lines, start=1):
        if line.strip() == "":
            if current:
                paragraphs.append((current_start, one_based_idx - 1, "\n".join(current)))
                current = []
                current_start = 0
        else:
            if not current:
                current_start = one_based_idx
            current.append(line)
    if current:
        paragraphs.append((current_start, len(lines), "\n".join(current)))
    return paragraphs


def _collect_keyword_matches(
    paragraphs: list[tuple[int, int, str]], keyword: str
) -> tuple[list[dict[str, Any]], int, bool]:
    """Collect all paragraphs containing keyword, with result-size caps."""
    needle = keyword.lower()
    match_count = 0
    matches: list[dict[str, Any]] = []
    cumulative_bytes = 0
    capacity_exhausted = False
    for start, end, text in paragraphs:
        if needle not in text.lower():
            continue
        match_count += 1
        if capacity_exhausted:
            continue
        if len(matches) >= constants.READ_DOC_KEYWORD_MATCH_CAP:
            capacity_exhausted = True
            continue
        paragraph_truncated = False
        text_to_emit = text
        if len(text) > constants.READ_DOC_KEYWORD_PARAGRAPH_BYTES:
            text_to_emit = text[: constants.READ_DOC_KEYWORD_PARAGRAPH_BYTES] + "\n..."
            paragraph_truncated = True
        if (
            cumulative_bytes + len(text_to_emit) > constants.READ_DOC_KEYWORD_TOTAL_BYTES
            and matches
        ):
            capacity_exhausted = True
            continue
        matches.append(
            {
                "start_line": start,
                "end_line": end,
                "text": text_to_emit,
                "paragraph_truncated": paragraph_truncated,
            }
        )
        cumulative_bytes += len(text_to_emit)
    matches_truncated = len(matches) < match_count
    return matches, match_count, matches_truncated


def read_doc_preview(
    task: PublicTask,
    relative_path: str,
    *,
    start_line: int = 1,
    line_count: int = 80,
    keyword: str | None = None,
) -> dict[str, Any]:
    """Preview a text document. Paging mode and keyword mode are mutually exclusive."""
    path = resolve_context_path(task, relative_path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    total_lines = len(lines)

    if total_lines == 0:
        return {
            "total_lines": 0,
            "returned_range": None,
            "has_more": False,
            "keyword_match": None,
            "preview": "",
        }

    if keyword:
        paragraphs = _split_paragraphs(lines)
        matches, match_count, matches_truncated = _collect_keyword_matches(paragraphs, keyword)
        return {
            "total_lines": total_lines,
            "returned_range": None,
            "has_more": False,
            "keyword_match": {
                "keyword": keyword,
                "found": match_count > 0,
                "match_count": match_count,
                "matches": matches,
                "matches_truncated": matches_truncated,
            },
            "preview": None,
        }

    normalized_start_line = max(1, int(start_line))
    normalized_line_count = _normalize_line_count(line_count)

    start_index = normalized_start_line - 1
    end_index = min(total_lines, start_index + normalized_line_count)
    selected_lines = lines[start_index:end_index]
    returned_range = None
    if selected_lines:
        returned_range = {
            "start_line": start_index + 1,
            "end_line": end_index,
        }

    return {
        "total_lines": total_lines,
        "returned_range": returned_range,
        "has_more": end_index < total_lines,
        "keyword_match": None,
        "preview": "\n".join(selected_lines),
    }


def summarize_doc(path: Path) -> dict[str, Any]:
    """Summarize a text document for inspect_files."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return {
        "line_count": len(lines),
        "head": lines[: constants.INSPECT_FILES_DOC_HEAD_LINES],
    }


@function_tool
def read_doc(
    task: PublicTask,
    path: Annotated[
        str,
        path_field(
            file_kind="text document (md/txt/...)",
            examples=("knowledge.md", "docs/readme.txt"),
        ),
    ],
    start_line: Annotated[
        int,
        positive_int_field(description="1-based line number to start reading from."),
    ] = 1,
    line_count: Annotated[
        int,
        positive_int_field(description="Number of lines to return in this page."),
    ] = 100,
    keyword: Annotated[
        str | None,
        Field(
            description=(
                "Optional case-insensitive keyword. When set, returns ALL paragraphs "
                "(blank-line-separated) containing the keyword across the whole file "
                "(start_line / line_count are ignored), capped at 20 matches / 8KB total. "
                "Prefer this over manual paging when searching for an ID/name/term — "
                "one call replaces a sequential scan."
            ),
        ),
    ] = None,
) -> ToolExecutionResult:
    """Preview a text document (md/txt). Two modes:
    (1) Paging — without keyword, returns a fixed line window; use
    has_more + next start_line to continue.
    (2) Keyword — with keyword, returns ALL paragraphs containing the
    keyword across the whole file (start_line/line_count ignored),
    capped at 20 matches / 8KB total. ALWAYS prefer keyword mode when
    searching for an ID/name/term — one call replaces a sequential scan.
    For full-document regex or structured parsing, use execute_python.
    Example paging: read_doc({"path": "knowledge.md", "start_line": 1})
    Example keyword: read_doc({"path": "knowledge.md", "keyword": "利率"})"""
    kw = keyword if keyword else None
    return ToolExecutionResult(
        ok=True,
        content=read_doc_preview(
            task,
            path,
            start_line=start_line,
            line_count=line_count,
            keyword=kw,
        ),
    )
