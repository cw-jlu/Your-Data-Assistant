"""PDF text cache helpers for exp_151 prose-doc PoC.

The agent still addresses files by their original context-relative PDF path
(for example, ``doc/mf_mainfinancialindexq.pdf``). This module maps that path
to a deterministic extracted-text cache under artifacts, so preamble and tools
can share the same PDF text without repeatedly running pypdf extraction.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from kobushi_core.benchmark.schema import PublicTask


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def cache_root() -> Path:
    root = Path(os.environ.get("EXP151_PDF_TEXT_CACHE_ROOT", "/tmp/kobushi_exp151_pdf_text_cache"))
    if not root.is_absolute():
        root = repo_root() / root
    return root


def _rel_pdf_path(task: PublicTask, pdf_path: Path) -> str:
    return pdf_path.resolve().relative_to(task.context_dir.resolve()).as_posix()


def text_cache_path(task: PublicTask, pdf_path: Path) -> Path:
    rel = _rel_pdf_path(task, pdf_path)
    return cache_root() / task.task_id / f"{rel}.txt"


def meta_cache_path(task: PublicTask, pdf_path: Path) -> Path:
    rel = _rel_pdf_path(task, pdf_path)
    return cache_root() / task.task_id / f"{rel}.meta.json"


def source_fingerprint(pdf_path: Path) -> dict[str, object]:
    stat = pdf_path.stat()
    digest = hashlib.sha1(pdf_path.read_bytes()).hexdigest()
    return {
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "source_sha1": digest,
    }


def extract_pdf_text(pdf_path: Path) -> tuple[str, int]:
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    pages = [(page.extract_text() or "") for page in reader.pages]
    return "\n".join(pages), len(pages)


@dataclass(frozen=True, slots=True)
class CacheWriteResult:
    source_rel: str
    text_path: Path
    meta_path: Path
    page_count: int
    text_chars: int
    source_size: int
    source_sha1: str


def write_pdf_cache(task: PublicTask, pdf_path: Path, *, force: bool = False) -> CacheWriteResult:
    pdf_path = pdf_path.resolve()
    rel = _rel_pdf_path(task, pdf_path)
    text_path = text_cache_path(task, pdf_path)
    meta_path = meta_cache_path(task, pdf_path)
    fp = source_fingerprint(pdf_path)

    if not force and text_path.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if (
                meta.get("source_size") == fp["source_size"]
                and meta.get("source_sha1") == fp["source_sha1"]
            ):
                return CacheWriteResult(
                    source_rel=rel,
                    text_path=text_path,
                    meta_path=meta_path,
                    page_count=int(meta.get("page_count", 0)),
                    text_chars=int(meta.get("text_chars", text_path.stat().st_size)),
                    source_size=int(fp["source_size"]),
                    source_sha1=str(fp["source_sha1"]),
                )
        except Exception:
            pass

    text, page_count = extract_pdf_text(pdf_path)
    text_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.write_text(text, encoding="utf-8")
    meta = {
        "task_id": task.task_id,
        "source_rel": rel,
        "text_rel": text_path.relative_to(cache_root()).as_posix(),
        "page_count": page_count,
        "text_chars": len(text),
        **fp,
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return CacheWriteResult(
        source_rel=rel,
        text_path=text_path,
        meta_path=meta_path,
        page_count=page_count,
        text_chars=len(text),
        source_size=int(fp["source_size"]),
        source_sha1=str(fp["source_sha1"]),
    )


def ensure_task_pdf_cache(task: PublicTask, *, force: bool = False) -> list[CacheWriteResult]:
    """Extract all PDFs for a task into the text cache if needed."""
    results: list[CacheWriteResult] = []
    for pdf_path in sorted(task.context_dir.rglob("*.pdf")):
        results.append(write_pdf_cache(task, pdf_path, force=force))
    return results


def read_cached_pdf_text(task: PublicTask, pdf_path: Path) -> str | None:
    path = text_cache_path(task, pdf_path)
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8", errors="replace")


def read_cached_pdf_meta(task: PublicTask, pdf_path: Path) -> dict[str, object] | None:
    path = meta_cache_path(task, pdf_path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
