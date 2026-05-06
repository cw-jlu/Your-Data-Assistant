"""
此模块提供文件系统相关的工具，如列出目录内容、读取 CSV、JSON 和文档预览。
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from data_agent_baseline.benchmark.schema import PublicTask


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


def read_csv_preview(task: PublicTask, relative_path: str, *, max_rows: int = 20) -> dict[str, object]:
    path = resolve_context_path(task, relative_path)
    with path.open(newline="", encoding="utf-8") as handle:
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


def read_json_preview(task: PublicTask, relative_path: str, *, max_chars: int = 4000) -> dict[str, object]:
    path = resolve_context_path(task, relative_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    preview = json.dumps(payload, ensure_ascii=False, indent=2)
    return {
        "path": relative_path,
        "preview": preview[:max_chars],
        "truncated": len(preview) > max_chars,
    }


def read_doc_preview(task: PublicTask, relative_path: str, *, max_chars: int = 4000, page: int = 1) -> dict[str, object]:
    path = resolve_context_path(task, relative_path)
    text = path.read_text(encoding="utf-8", errors="replace")
    
    total_pages = (len(text) + max_chars - 1) // max_chars
    start_idx = (page - 1) * max_chars
    end_idx = start_idx + max_chars
    
    return {
        "path": relative_path,
        "preview": text[start_idx:end_idx],
        "page": page,
        "total_pages": total_pages,
        "truncated": len(text) > end_idx,
    }


def read_doc_lines(task: PublicTask, relative_path: str, start_line: int, end_line: int) -> dict[str, object]:
    path = resolve_context_path(task, relative_path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    
    # Lines are usually 1-indexed in the tree
    total_lines = len(lines)
    s = max(0, start_line - 1)
    e = min(total_lines, end_line)
    
    return {
        "path": relative_path,
        "content": "\n".join(lines[s:e]),
        "start_line": s + 1,
        "end_line": e,
        "total_lines": total_lines
    }

def get_doc_structure(task: PublicTask, relative_path: str) -> dict[str, object]:
    from data_agent_baseline.agents.pageindex_lite import get_md_pageindex_summary
    path = resolve_context_path(task, relative_path)
    if not path.name.lower().endswith(".md"):
        return {"error": "Only markdown files are supported for tree structure."}
    
    structure = get_md_pageindex_summary(path)
    return {
        "path": relative_path,
        "structure": structure
    }
