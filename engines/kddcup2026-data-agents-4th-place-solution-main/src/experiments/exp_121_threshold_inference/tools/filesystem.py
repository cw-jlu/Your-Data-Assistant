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
    task: PublicTask, relative_path: str, *,
    max_chars: int = 4000,
    search: str | None = None,
    offset: int = 0,
) -> dict[str, object]:
    """Read a text doc.

    Two modes:
    - search=<term>: return paragraphs containing the term (case-insensitive),
      joined by `\\n---\\n`. Use this for long docs to extract specific facts.
    - offset=<int> (default 0): return text[offset:offset+max_chars]. Use
      next_offset (in the response) to paginate through long docs.

    Both modes respect max_chars to bound the response.
    """
    import re
    path = resolve_context_path(task, relative_path)
    text = path.read_text(errors="replace")
    total_chars = len(text)

    if search:
        needle = str(search).lower()
        # Split by blank-line paragraphs (= the typical MD structure)
        paragraphs = re.split(r'\n\s*\n', text)
        matching = [p for p in paragraphs if needle in p.lower()]
        joined = "\n---\n".join(matching)
        truncated = len(joined) > max_chars
        snippet = joined[:max_chars]
        return {
            "path": relative_path,
            "search": search,
            "n_matches": len(matching),
            "snippets": snippet,
            "truncated": truncated,
            "total_chars": total_chars,
            "hint": (
                "n_matches=0 → try a different search term (= shorter, broader, "
                "or a substring like an ID prefix). n_matches > 0 but truncated "
                "→ the matching paragraphs exceed max_chars; refine the search "
                "or increase max_chars."
            ),
        }

    # Pagination mode
    if offset < 0:
        offset = 0
    end = offset + max_chars
    chunk = text[offset:end]
    has_more = end < total_chars
    return {
        "path": relative_path,
        "offset": offset,
        "preview": chunk,
        "total_chars": total_chars,
        "next_offset": end if has_more else None,
        "truncated": has_more,
    }


def grep_context(
    task: PublicTask, *,
    pattern: str,
    path: str | None = None,
    glob: str | None = None,
    output_mode: str = "content",
    context_lines: int = 0,
    case_insensitive: bool = True,
    max_results: int = 50,
    max_chars: int = 8000,
) -> dict[str, object]:
    """Regex search across text files in the task's context dir.

    Modes (mirrors Claude Code's Grep):
      - "content"            → list of "path:line:text" with optional ±N context
      - "files_with_matches" → just unique paths that contain a match
      - "count"              → match-count per file

    pattern: Python re syntax.
    path: limit to one file (relative to context dir). If None and glob is None,
          searches all text-readable files (= md, txt, csv, json).
    glob: filename pattern under context (= e.g. "doc/*.md", "csv/*.csv").
    """
    import re
    import fnmatch as _fnmatch

    flags = re.IGNORECASE if case_insensitive else 0
    try:
        regex = re.compile(pattern, flags)
    except re.error as exc:
        return {"error": f"invalid regex: {exc}", "pattern": pattern}

    context_root = task.context_dir.resolve()

    # Build candidate file list
    candidates: list[Path] = []
    if path:
        p = resolve_context_path(task, str(path)).resolve()
        if p.is_file():
            candidates.append(p)
    else:
        for p in context_root.rglob("*"):
            if not p.is_file():
                continue
            suf = p.suffix.lower()
            if suf in (".md", ".txt", ".csv", ".json"):
                rel = p.relative_to(context_root).as_posix()
                if glob and not _fnmatch.fnmatch(rel, glob):
                    continue
                candidates.append(p)

    file_matches: dict[str, list[tuple[int, str]]] = {}
    file_count: dict[str, int] = {}
    for p in candidates:
        rel = p.relative_to(context_root).as_posix()
        try:
            text = p.read_text(errors="replace")
        except Exception as exc:
            file_matches[rel] = [(0, f"<read error: {exc}>")]
            continue
        lines = text.splitlines()
        hits = []
        for i, line in enumerate(lines, start=1):
            if regex.search(line):
                hits.append((i, line))
        if hits:
            file_matches[rel] = hits
            file_count[rel] = len(hits)

    if output_mode == "files_with_matches":
        return {
            "pattern": pattern,
            "files_with_matches": list(file_matches.keys()),
            "n_files": len(file_matches),
        }

    if output_mode == "count":
        return {
            "pattern": pattern,
            "counts_per_file": file_count,
            "total_matches": sum(file_count.values()),
        }

    # output_mode == "content"
    out_lines: list[str] = []
    char_used = 0
    n_results = 0
    truncated = False
    for rel, hits in file_matches.items():
        all_lines = (context_root / rel).read_text(errors="replace").splitlines()
        for line_no, _ in hits:
            if context_lines > 0:
                lo = max(1, line_no - context_lines)
                hi = min(len(all_lines), line_no + context_lines)
                for k in range(lo, hi + 1):
                    marker = ":" if k == line_no else "-"
                    line_text = all_lines[k - 1]
                    rendered = f"{rel}{marker}{k}{marker}{line_text}"
                    if char_used + len(rendered) > max_chars or n_results >= max_results:
                        truncated = True
                        break
                    out_lines.append(rendered)
                    char_used += len(rendered) + 1
            else:
                line_text = all_lines[line_no - 1]
                rendered = f"{rel}:{line_no}:{line_text}"
                if char_used + len(rendered) > max_chars or n_results >= max_results:
                    truncated = True
                    break
                out_lines.append(rendered)
                char_used += len(rendered) + 1
            n_results += 1
            if truncated:
                break
        if truncated:
            break

    return {
        "pattern": pattern,
        "n_files_with_matches": len(file_matches),
        "total_matches_in_response": n_results,
        "truncated": truncated,
        "content": "\n".join(out_lines) if out_lines else "(no matches)",
    }
