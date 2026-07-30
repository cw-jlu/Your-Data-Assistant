"""Prose file detection and section parsing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agents.etl._constants import PROSE_EXTS, SECTION_HEADING

if TYPE_CHECKING:
    from agents.benchmark.schema import PublicTask


@dataclass(frozen=True, slots=True)
class ETLResult:
    """Metadata about a single pre-extracted CSV, for prompt injection."""

    source_file: str
    csv_path: str
    columns: list[str]
    row_count: int


def detect_sections(text: str) -> list[dict[str, Any]]:
    """Split text by heading markers into sections with metadata."""
    headings = list(SECTION_HEADING.finditer(text))
    if not headings:
        return []
    top_level = min(len(m.group(1)) for m in headings)
    headings = [m for m in headings if len(m.group(1)) == top_level]

    sections: list[dict[str, Any]] = []
    for i, m in enumerate(headings):
        title = m.group(2).strip()
        start = m.end()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        content = text[start:end]
        paras = [p.strip() for p in content.split("\n\n") if p.strip() and len(p.strip()) > 30]
        sections.append({"title": title, "content": content, "para_count": len(paras)})
    return sections


def detect_prose_files(task: PublicTask) -> list[Path]:
    """Find non-empty UTF-8 .md/.txt files and non-empty PDFs."""
    seen: set[Path] = set()
    results: list[Path] = []

    def _scan_dir(directory: Path) -> None:
        if not directory.is_dir():
            return
        for f in sorted(directory.iterdir()):
            if not f.is_file() or f.suffix.lower() not in PROSE_EXTS:
                continue
            if f.name.lower() == "knowledge.md":
                continue
            resolved = f.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            if f.suffix.lower() == ".pdf":
                if f.stat().st_size == 0:
                    continue
                results.append(f)
                continue
            try:
                with f.open(encoding="utf-8") as handle:
                    line_count = sum(1 for _ in handle)
            except (OSError, UnicodeDecodeError):
                continue
            if line_count == 0:
                continue
            results.append(f)

    _scan_dir(task.context_dir / "doc")
    _scan_dir(task.context_dir)
    return results
