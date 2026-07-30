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


def _load_doc_text(path: Path, task: PublicTask | None = None) -> str:
    """Load a doc as text. exp_149 lever ② (prose_on): decode .pdf via pypdf
    (pure-python, Docker-portable — the submission image has no pdftotext).
    With the prose flag OFF, fall back to read_text (= exp_143 behavior: a
    .pdf yields binary mojibake, which is the intended baseline)."""
    from experiments.exp_167_domain_pruner import flags
    if path.suffix.lower() == ".pdf" and flags.prose_on():
        if flags.pdf_preprocess_on() and task is not None:
            try:
                from experiments.exp_167_domain_pruner.pdf_text_cache import read_cached_pdf_text
                cached = read_cached_pdf_text(task, path)
                if cached is not None:
                    return cached
            except Exception:
                pass
        try:
            from pypdf import PdfReader
            return "\n".join((pg.extract_text() or "") for pg in PdfReader(str(path)).pages)
        except Exception as exc:
            return f"[PDF read error: {exc!r} — pypdf may be unavailable]"
    return path.read_text(errors="replace")


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
    n_lines: int = 100,
) -> dict[str, object]:
    """Read a text doc — Claude Code Read-tool style (line-based).

    Modes:
    - search=<term>: return paragraphs containing the term (case-insensitive),
      joined by `\\n---\\n`. Use this for long docs to extract specific facts.
    - offset=<int> (default 0 = line 1): start at LINE NUMBER `offset` (1-based).
      Returns up to `n_lines` lines. Output is `cat -n`-style
      (= "<line_no>\\t<content>"), so the agent can directly
      map grep's `path:line:content` output to a follow-up
      `read_doc(path=X, offset=line)` call.

    max_chars is an internal safety valve; callers should control line coverage
    with n_lines and next_offset.
    """
    import re
    path = resolve_context_path(task, relative_path)
    text = _load_doc_text(path, task)
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
                "→ matching paragraphs exceeded the internal response cap; refine "
                "the search term, use grep with context_lines, or read around "
                "specific hit lines with offset/n_lines."
            ),
        }

    # Line-based pagination (Claude Code Read style).
    # offset: 1-based line number to start at (0 → treat as 1).
    # n_lines: how many lines to include. max_chars is only an internal safety cap.
    # Output is `cat -n` style (= "<line_no>\t<content>") so grep's
    # `path:line:content` directly maps to read_doc(path=X, offset=line).
    all_lines = text.splitlines()
    total_lines = len(all_lines)
    start_line = max(1, offset if offset > 0 else 1)
    if start_line > total_lines:
        return {
            "path": relative_path,
            "start_line": start_line,
            "preview": "",
            "total_lines": total_lines,
            "total_chars": total_chars,
            "next_offset": None,
            "truncated": False,
            "hint": f"start_line ({start_line}) > total_lines ({total_lines}); doc is shorter.",
        }
    rendered: list[str] = []
    char_used = 0
    line_idx = start_line - 1
    end_line_inclusive = start_line
    while line_idx < total_lines and len(rendered) < n_lines:
        line_no = line_idx + 1
        content = all_lines[line_idx]
        line_str = f"{line_no}\t{content}"
        if char_used + len(line_str) + 1 > max_chars and rendered:
            break
        rendered.append(line_str)
        char_used += len(line_str) + 1
        end_line_inclusive = line_no
        line_idx += 1
    has_more = line_idx < total_lines
    next_offset = (end_line_inclusive + 1) if has_more else None
    if has_more:
        hint = (
            f"Only lines {start_line}-{end_line_inclusive} were returned. "
            f"This is NOT the full document. Continue with "
            f"read_doc(path='{relative_path}', offset={next_offset}, n_lines={n_lines}) "
            "or jump to grep hit lines."
        )
    else:
        hint = "next_offset is null; this read reached the end of the document."
    return {
        "path": relative_path,
        "start_line": start_line,
        "end_line": end_line_inclusive,
        "preview": "\n".join(rendered),
        "total_lines": total_lines,
        "total_chars": total_chars,
        "next_offset": next_offset,
        "truncated": has_more,
        "hint": hint,
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
        from experiments.exp_167_domain_pruner import flags
        _allowed_suf = (".md", ".txt", ".csv", ".json") + ((".pdf",) if flags.prose_on() else ())
        for p in context_root.rglob("*"):
            if not p.is_file():
                continue
            suf = p.suffix.lower()
            if suf in _allowed_suf:
                rel = p.relative_to(context_root).as_posix()
                if glob and not _fnmatch.fnmatch(rel, glob):
                    continue
                candidates.append(p)

    file_matches: dict[str, list[tuple[int, str]]] = {}
    file_count: dict[str, int] = {}
    for p in candidates:
        rel = p.relative_to(context_root).as_posix()
        try:
            text = _load_doc_text(p, task)
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
        all_lines = _load_doc_text(context_root / rel, task).splitlines()
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
