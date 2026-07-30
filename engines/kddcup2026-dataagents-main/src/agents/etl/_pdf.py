"""PDF → markdown conversion for the ETL pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from statistics import median
from typing import Any, cast

from agents.etl._constants import PDF_CJK_PUNCT, PDF_SENTENCE_END, PDF_STRUCTURAL_LINE

logger = logging.getLogger(__name__)

_Line = tuple[float, float, float, float, str]


def _is_cjk(char: str) -> bool:
    return "\u4e00" <= char <= "\u9fff"


def _join_without_space(prev: str, cur: str) -> bool:
    return bool((_is_cjk(prev) or prev in PDF_CJK_PUNCT) and _is_cjk(cur))


def _join_prose_lines(lines: list[str]) -> str:
    text = lines[0]
    for line in lines[1:]:
        if text.endswith("-") and line and line[0].islower():
            text = f"{text[:-1]}{line}"
            continue
        prev = text[-1] if text else ""
        cur = line[0] if line else ""
        sep = "" if _join_without_space(prev, cur) else " "
        text = f"{text}{sep}{line}"
    return text


def _is_structural_line(line: str) -> bool:
    if PDF_STRUCTURAL_LINE.match(line):
        return True
    return bool(line.startswith("|") or line.endswith("|") or "\t" in line)


def _unwrap_soft_line_breaks(text: str) -> str:
    """Join PDF soft-wrapped prose lines within existing text blocks."""
    blocks: list[str] = []
    prose: list[str] = []
    structural: list[str] = []

    def flush_prose() -> None:
        nonlocal prose
        if prose:
            blocks.append(_join_prose_lines(prose))
            prose = []

    def flush_structural() -> None:
        nonlocal structural
        if structural:
            blocks.append("\n".join(structural))
            structural = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            flush_prose()
            flush_structural()
            continue

        if _is_structural_line(line):
            flush_prose()
            structural.append(line)
            continue

        flush_structural()
        prose.append(line)

    flush_prose()
    flush_structural()
    return "\n\n".join(blocks)


def _extract_page_lines(page: Any) -> list[_Line]:
    data = cast(dict[str, Any], page.get_text("dict"))
    lines: list[_Line] = []
    for block in data.get("blocks", []):
        if not isinstance(block, dict):
            continue
        for raw_line in block.get("lines", []):  # pyright: ignore[reportUnknownVariableType,reportUnknownMemberType]
            line_dict = cast(dict[str, Any], raw_line)
            spans = cast(list[dict[str, Any]], line_dict.get("spans", []))
            text = "".join(str(span.get("text", "")) for span in spans).strip()
            if not text:
                continue
            x0, y0, x1, y1 = cast(tuple[float, float, float, float], line_dict["bbox"])
            lines.append((float(x0), float(y0), float(x1), float(y1), text))
    return sorted(lines, key=lambda item: (item[1], item[0]))


def _paragraphs_from_positioned_lines(lines: list[_Line]) -> list[str]:
    if not lines:
        return []

    heights = [max(1.0, y1 - y0) for _x0, y0, _x1, y1, _text in lines]
    gap_threshold = max(8.0, median(heights) * 0.65)

    paragraphs: list[str] = []
    current: list[str] = []
    prev_y1: float | None = None

    def flush() -> None:
        nonlocal current
        if current:
            paragraph = _unwrap_soft_line_breaks("\n".join(current))
            if paragraph:
                paragraphs.append(paragraph)
            current = []

    for _x0, y0, _x1, y1, text in lines:
        if prev_y1 is not None and y0 - prev_y1 > gap_threshold:
            flush()
        current.append(text)
        prev_y1 = y1
    flush()
    return paragraphs


def page_to_paragraphs(page: Any) -> list[str]:
    """Extract one PDF page as paragraphs using layout gaps as boundaries."""
    paragraphs = _paragraphs_from_positioned_lines(_extract_page_lines(page))
    if paragraphs:
        return paragraphs

    text = cast(str, page.get_text())
    return [p for p in _unwrap_soft_line_breaks(text.strip()).split("\n\n") if p]


def _is_complete_sentence(paragraph: str) -> bool:
    return bool(PDF_SENTENCE_END.search(paragraph))


def stitch_page_paragraphs(pages: list[tuple[int, list[str]]]) -> list[tuple[int, list[str]]]:
    """Join paragraphs split across page boundaries using punctuation only."""
    stitched: list[tuple[int, list[str]]] = []
    for page_no, paragraphs in pages:
        remaining = list(paragraphs)
        if (
            stitched
            and stitched[-1][1]
            and remaining
            and not _is_complete_sentence(stitched[-1][1][-1])
            and not _is_structural_line(stitched[-1][1][-1])
            and not _is_structural_line(remaining[0])
        ):
            prev_paragraphs = list(stitched[-1][1])
            prev_paragraphs[-1] = _join_prose_lines([prev_paragraphs[-1], remaining.pop(0)])
            stitched[-1] = (stitched[-1][0], prev_paragraphs)
        if remaining:
            stitched.append((page_no, remaining))
    return stitched


_page_to_paragraphs = page_to_paragraphs
_stitch_page_paragraphs = stitch_page_paragraphs


def pdf_to_markdown(path: Path) -> str:
    """Extract text from a PDF and return it as markdown with page headings."""
    import pymupdf  # pyright: ignore[reportMissingImports]

    doc = pymupdf.open(str(path))  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
    try:
        pages: list[tuple[int, list[str]]] = []
        page_count: int = len(doc)  # pyright: ignore[reportUnknownArgumentType]
        for i in range(page_count):
            paragraphs = page_to_paragraphs(doc[i])  # pyright: ignore[reportIndexIssue]
            if not paragraphs:
                continue
            pages.append((i + 1, paragraphs))
        parts = [
            f"## Page {page_no}\n\n" + "\n\n".join(paragraphs)
            for page_no, paragraphs in stitch_page_paragraphs(pages)
        ]
        return "\n\n".join(parts)
    finally:
        doc.close()  # pyright: ignore[reportUnknownMemberType]
