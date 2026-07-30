from __future__ import annotations

import math
import re
from pathlib import Path, PurePosixPath

from data_agent_baseline.benchmark.context_view import (
    iter_context_file_assets,
    normalize_context_relative_path,
    resolve_context_asset,
    resolve_context_path as _resolve_context_path,
)
from data_agent_baseline.benchmark.schema import PublicTask


def resolve_context_path(task: PublicTask, relative_path: str) -> Path:
    return _resolve_context_path(task, relative_path)


# 递归列出 context/ 目录树，用于让模型先了解有哪些可用资产。
def list_context_tree(task: PublicTask, *, max_depth: int = 4) -> dict[str, object]:
    context_view = task.assets.context_view
    if context_view is not None:
        entries_by_path: dict[str, dict[str, object]] = {}
        for asset in iter_context_file_assets(task):
            parts = PurePosixPath(asset.visible_path).parts
            for depth in range(1, len(parts)):
                if depth > max_depth:
                    continue
                dir_path = "/".join(parts[:depth])
                entries_by_path.setdefault(
                    dir_path,
                    {"path": dir_path, "kind": "dir", "size": None},
                )
            if len(parts) <= max_depth:
                entries_by_path[asset.visible_path] = {
                    "path": asset.visible_path,
                    "kind": "file",
                    "size": asset.physical_path.stat().st_size,
                }
        entries = sorted(
            entries_by_path.values(),
            key=lambda entry: (entry["kind"] == "file", str(entry["path"])),
        )
        return {
            "root": ".",
            "path_convention": "All paths are relative to the context directory. Use them exactly as listed and do not prefix them with `context/`.",
            "entries": entries,
        }

    entries: list[dict[str, object]] = []

    # 递归遍历目录，并记录相对路径、类型和文件大小。
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
        "root": ".",
        "path_convention": "All paths are relative to the context directory. Use them exactly as listed and do not prefix them with `context/`.",
        "entries": entries,
    }


_HEADING_NUMBER_PREFIX_RE = re.compile(
    r"^\s*(?:"
    r"(?:\d+(?:\.\d+)*\.?\s+)"
    r"|(?:第\s*[一二三四五六七八九十百千\d]+\s*[章节篇部分]\s*)"
    r")"
)


def _normalize_heading_for_match(heading: str) -> str:
    normalized = heading.strip().lower()
    normalized = _HEADING_NUMBER_PREFIX_RE.sub("", normalized).strip()
    return re.sub(r"\s+", " ", normalized)


def _extract_section(text: str, heading: str) -> tuple[str, str] | None:
    lines = text.splitlines()
    target = heading.strip().lower()
    normalized_target = _normalize_heading_for_match(heading)

    target_idx = -1
    target_level = 0
    matched_heading = ""

    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            heading_text = stripped.lstrip("#").strip()
            if heading_text.lower() == target:
                target_idx = i
                target_level = level
                matched_heading = heading_text
                break

    if target_idx == -1 and normalized_target:
        for i, line in enumerate(lines):
            stripped = line.lstrip()
            if stripped.startswith("#"):
                level = len(line) - len(line.lstrip("#"))
                heading_text = stripped.lstrip("#").strip()
                if _normalize_heading_for_match(heading_text) == normalized_target:
                    target_idx = i
                    target_level = level
                    matched_heading = heading_text
                    break

    if target_idx == -1:
        return None

    section_lines: list[str] = []
    for line in lines[target_idx + 1:]:
        stripped = line.lstrip()
        if stripped.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            if level <= target_level:
                break
        section_lines.append(line)

    result = lines[target_idx] + "\n" + "\n".join(section_lines)
    return result.strip(), matched_heading


def _list_all_headings(text: str) -> list[dict[str, object]]:
    headings: list[dict[str, object]] = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#") and stripped.lstrip("#").strip():
            level = len(line) - len(line.lstrip("#"))
            headings.append({"level": level, "text": stripped.lstrip("#").strip()})
    return headings


# 读取普通文本文件，支持按章节标题提取指定段落。
# 文本内容不做截断，输出截断由 ToolConfig.max_output_tokens 在 format_result 层统一处理。
def read_doc_preview(task: PublicTask, relative_path: str, *, heading: str | None = None) -> dict[str, object]:
    normalized_path = normalize_context_relative_path(relative_path)
    asset = resolve_context_asset(task, normalized_path)
    path = asset.physical_path
    text = path.read_text(encoding="utf-8", errors="replace")

    if heading is not None:
        extracted = _extract_section(text, heading)
        if extracted is None:
            return {
                "path": normalized_path,
                "error": f"Heading '{heading}' not found in document.",
                "available_headings": _list_all_headings(text),
            }
        section, matched_heading = extracted
        content: dict[str, object] = {
            "path": normalized_path,
            "preview": section,
            "section": heading,
        }
        if matched_heading and matched_heading.lower() != heading.strip().lower():
            content["matched_heading"] = matched_heading
        return content

    return {
        "path": normalized_path,
        "preview": text,
    }


_TEXT_EXTENSIONS = frozenset({".md", ".txt", ".rst"})


def _slice_results_by_page(
    file_results: list[dict[str, object]],
    page: int,
    page_size: int,
) -> list[dict[str, object]]:
    start = (page - 1) * page_size
    end = start + page_size

    flat: list[tuple[str, dict[str, object]]] = []
    for fr in file_results:
        fpath = fr["file"]
        for match in fr["matches"]:
            flat.append((fpath, match))

    if start >= len(flat):
        return []

    page_flat = flat[start:end]

    page_file_map: dict[str, list[dict[str, object]]] = {}
    for fpath, match in page_flat:
        if fpath not in page_file_map:
            page_file_map[fpath] = []
        page_file_map[fpath].append(match)

    return [
        {"file": fpath, "matches": matches}
        for fpath, matches in page_file_map.items()
    ]


# 在 context 目录的文本文件中搜索正则/关键词，返回匹配行及上下文。
# 替换 AI 反复手写 grep 模式：open → read → re.finditer → print context。
def search_doc_text(
    task: PublicTask,
    query: str,
    *,
    context_lines: int = 3,
    path: str | None = None,
    page: int = 1,
    page_size: int = 20,
) -> dict[str, object]:
    compiled = re.compile(query, re.IGNORECASE)

    if path is not None:
        candidate_assets = [resolve_context_asset(task, normalize_context_relative_path(path))]
    else:
        candidate_assets = [
            asset for asset in iter_context_file_assets(task)
            if Path(asset.visible_path).suffix.lower() in _TEXT_EXTENSIONS
        ]

    candidate_assets = sorted(candidate_assets, key=lambda asset: asset.visible_path)

    file_results: list[dict[str, object]] = []
    total_matches = 0

    for asset in candidate_assets:
        file_path = asset.physical_path
        lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
        file_matches: list[dict[str, object]] = []

        for idx, line in enumerate(lines):
            if compiled.search(line):
                before_start = max(0, idx - context_lines)
                after_end = min(len(lines), idx + context_lines + 1)

                file_matches.append({
                    "line_number": idx + 1,
                    "match_text": line,
                    "context_before": lines[before_start:idx],
                    "context_after": lines[idx + 1:after_end],
                })
                total_matches += 1

        if file_matches:
            file_results.append({
                "file": asset.visible_path,
                "matches": file_matches,
            })

    if page_size > 0 and total_matches > 0:
        sliced_results = _slice_results_by_page(file_results, page, page_size)
        total_pages = max(1, math.ceil(total_matches / page_size))
    else:
        sliced_results = file_results
        total_pages = 1

    return {
        "query": query,
        "total_matches": total_matches,
        "page": page if page_size > 0 else 1,
        "page_size": page_size,
        "total_pages": total_pages,
        "results": sliced_results,
    }
