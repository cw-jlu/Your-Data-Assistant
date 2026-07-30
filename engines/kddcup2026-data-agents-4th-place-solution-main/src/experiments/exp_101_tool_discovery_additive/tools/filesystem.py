from __future__ import annotations

import csv
import json
import re
import sqlite3
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


# ============================================================================
# Tool discovery additive (exp_101, [tool:discovery-additive] axis)
# ============================================================================
# Two narrow lookup tools to complement (NOT replace) the existing
# read_csv_preview / read_doc_preview / read_json_preview tools.
# Mechanism rationale (exp_096 trace evidence): for tasks where the
# question requires a specific value/format that the agent cannot
# rote-copy from the 150K-token preamble (e.g., task_89 Chinese GP
# "+16.445" string format), an explicit small lookup or stat call lets
# the model see the exact value/format and answer correctly. We keep
# the full preamble — only ADD these tools, never replace.
#
# Safety: stat_file deliberately omits null% / cardinality / dtype
# inference to avoid the exp_048 inspect_table failure mode (high null%
# triggered the agent to give up early on task_243 etc.). Only
# mechanical, factual fields are returned.

_GREP_MAX_LINES = 50
_GREP_MAX_CHARS = 500
_STAT_SAMPLE_LINES = 3


def grep_file(file_path: Path, pattern: str, *, max_lines: int = _GREP_MAX_LINES) -> str:
    """Return up to `max_lines` lines from `file_path` that match `pattern`
    (Python regex, case-insensitive). Total output capped at _GREP_MAX_CHARS.

    Use for narrow lookups in large files (>10MB) or when grep is faster
    than full read. For normal-sized files prefer read_csv_preview /
    read_json_preview / read_doc_preview.
    """
    try:
        compiled = re.compile(pattern, re.IGNORECASE)
    except re.error as exc:
        return f"[grep_file: invalid regex: {exc}]"
    try:
        text = file_path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"[grep_file: read error: {exc!r}]"
    out_lines: list[str] = []
    used_chars = 0
    for line in text.splitlines():
        if compiled.search(line):
            seg = line.rstrip()
            if used_chars + len(seg) + 1 > _GREP_MAX_CHARS:
                out_lines.append("[...truncated...]")
                break
            out_lines.append(seg)
            used_chars += len(seg) + 1
            if len(out_lines) >= max_lines:
                out_lines.append(f"[...{max_lines}-line cap reached...]")
                break
    if not out_lines:
        return "(no matches)"
    return "\n".join(out_lines)


def stat_file(file_path: Path) -> dict[str, object]:
    """Return mechanical metadata for `file_path`. NO null% / cardinality /
    dtype inference (exp_048 inspect_table failure-mode avoidance).

    Returns a dict with keys: size_bytes, line_count, file_type, header
    (CSV only) / tables (SQLite only) / json_root_kind (JSON only),
    sample_first_lines.
    """
    info: dict[str, object] = {}
    try:
        info["size_bytes"] = int(file_path.stat().st_size)
    except Exception:
        info["size_bytes"] = -1
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        info["file_type"] = "csv"
        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
            lines = text.splitlines()
            info["line_count"] = len(lines)
            info["header"] = lines[0] if lines else ""
            info["sample_first_lines"] = lines[1:1 + _STAT_SAMPLE_LINES]
        except Exception as exc:
            info["read_error"] = repr(exc)
    elif suffix in {".db", ".sqlite", ".sqlite3"}:
        info["file_type"] = "sqlite"
        try:
            conn = sqlite3.connect(file_path)
            try:
                cur = conn.cursor()
                tables = [
                    r[0] for r in cur.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
                    ).fetchall()
                ]
                info["tables"] = tables
                info["line_count"] = None  # SQLite has no line concept
            finally:
                conn.close()
        except Exception as exc:
            info["read_error"] = repr(exc)
    elif suffix == ".json":
        info["file_type"] = "json"
        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
            info["line_count"] = len(text.splitlines())
            try:
                obj = json.loads(text)
                if isinstance(obj, list):
                    info["json_root_kind"] = "list"
                    info["json_n_items"] = len(obj)
                    if obj:
                        info["sample_first_item_keys"] = (
                            list(obj[0].keys()) if isinstance(obj[0], dict) else None
                        )
                elif isinstance(obj, dict):
                    info["json_root_kind"] = "object"
                    info["sample_top_level_keys"] = list(obj.keys())[:20]
            except Exception:
                info["json_parse_error"] = True
        except Exception as exc:
            info["read_error"] = repr(exc)
    else:
        info["file_type"] = suffix.lstrip(".") or "unknown"
        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
            lines = text.splitlines()
            info["line_count"] = len(lines)
            info["sample_first_lines"] = lines[:_STAT_SAMPLE_LINES]
        except Exception as exc:
            info["read_error"] = repr(exc)
    return info
