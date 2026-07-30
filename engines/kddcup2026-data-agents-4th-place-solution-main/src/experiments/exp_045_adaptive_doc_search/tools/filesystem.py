from __future__ import annotations

import csv
import json
from pathlib import Path

from kobushi_core.benchmark.schema import PublicTask


def resolve_context_path(task: PublicTask, relative_path: str) -> Path:
    candidate = (task.context_dir / relative_path).resolve()
    context_root = task.context_dir.resolve()
    if context_root not in candidate.parents and candidate != context_root:
        raise ValueError(f"Path escapes context dir: {relative_path}")
    if not candidate.exists():
        raise FileNotFoundError(f"Missing context asset: {relative_path}")
    return candidate


def list_context_tree(task: PublicTask, *, max_depth: int = 4) -> dict[str, object]:
    entries: list[dict[str, object]] = []

    def walk(path: Path, depth: int) -> None:
        if depth > max_depth:
            return
        for child in sorted(path.iterdir(), key=lambda item: (item.is_file(), item.name)):
            rel_path = child.relative_to(task.context_dir).as_posix()
            entries.append(
                {
                    "path": rel_path,
                    "kind": "dir" if child.is_dir() else "file",
                    "size": child.stat().st_size if child.is_file() else None,
                }
            )
            if child.is_dir():
                walk(child, depth + 1)

    walk(task.context_dir, 1)
    return {
        "root": str(task.context_dir),
        "entries": entries,
    }


def read_csv_preview(
    task: PublicTask, relative_path: str, *, max_rows: int = 20
) -> dict[str, object]:
    path = resolve_context_path(task, relative_path)
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)

    if not rows:
        return {
            "path": relative_path,
            "columns": [],
            "rows": [],
            "row_count": 0,
        }

    header = rows[0]
    data_rows = rows[1:]
    return {
        "path": relative_path,
        "columns": header,
        "rows": data_rows[:max_rows],
        "row_count": len(data_rows),
    }


def read_json_preview(
    task: PublicTask, relative_path: str, *, max_chars: int = 4000
) -> dict[str, object]:
    path = resolve_context_path(task, relative_path)
    payload = json.loads(path.read_text())
    preview = json.dumps(payload, ensure_ascii=False, indent=2)
    return {
        "path": relative_path,
        "preview": preview[:max_chars],
        "truncated": len(preview) > max_chars,
    }


def read_doc_preview(
    task: PublicTask, relative_path: str, *, max_chars: int = 4000
) -> dict[str, object]:
    path = resolve_context_path(task, relative_path)
    text = path.read_text(errors="replace")
    return {
        "path": relative_path,
        "preview": text[:max_chars],
        "truncated": len(text) > max_chars,
    }


def search_doc(
    task: PublicTask,
    relative_path: str,
    query: str,
    *,
    max_chunks: int = 5,
    context_lines: int = 5,
) -> dict[str, object]:
    """Keyword-grep a doc file and return matching passages with surrounding context.

    All space-separated keywords must appear (case-insensitive) in a line for it
    to match. Returns up to max_chunks passages; if nothing matches, match_count=0
    with an empty chunks list (not an error).
    """
    path = resolve_context_path(task, relative_path)
    lines = path.read_text(errors="replace").splitlines()
    keywords = query.lower().split()
    matched_indices = [
        i for i, line in enumerate(lines)
        if all(k in line.lower() for k in keywords)
    ]
    chunks = []
    for idx in matched_indices[:max_chunks]:
        start = max(0, idx - context_lines)
        end = min(len(lines), idx + context_lines + 1)
        chunks.append({"line": idx + 1, "context": "\n".join(lines[start:end])})
    return {
        "path": relative_path,
        "query": query,
        "match_count": len(matched_indices),
        "chunks": chunks,
    }
