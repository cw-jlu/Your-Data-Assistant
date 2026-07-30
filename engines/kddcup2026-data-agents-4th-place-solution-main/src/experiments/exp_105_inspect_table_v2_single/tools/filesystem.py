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


# ============================================================================
# inspect_table v2 (exp_105 [tool:discovery]):
#   v2 spec: name + dtype + 3 sample rows + true row_count.
#   v2 explicitly EXCLUDES null_pct / cardinality / nunique / min / max /
#   value distribution. exp_048 v1 included null_pct → caused early-stop
#   bias on task_243/350. Do NOT add any of these fields back.
# ============================================================================

import sqlite3 as _sqlite3  # local import to keep top-level imports unchanged

_INSPECT_SAMPLE_ROWS = 3


def _inspect_dataframe_v2(df) -> dict[str, object]:
    """Schema-only summary: name + dtype + 3 sample rows + row_count.

    Intentionally omits null_pct / cardinality / min-max — see module note.
    """
    columns = [
        {
            "name": str(col),
            "dtype": str(df.dtypes[col]),
            "samples": [
                ("" if v is None else str(v))
                for v in df[col].head(_INSPECT_SAMPLE_ROWS).tolist()
            ],
        }
        for col in df.columns
    ]
    return {"columns": columns, "row_count": int(len(df))}


def inspect_table_v2(task: PublicTask, relative_path: str) -> dict[str, object]:
    """Schema discovery for CSV / SQLite / JSON / Parquet — single call.

    Returns {"ok": True, "file_type": <type>, ...} on success, or
    {"ok": False, "error": str} on unsupported / unreadable input.
    """
    import pandas as pd  # local import (heavy)

    path = resolve_context_path(task, relative_path)
    ext = path.suffix.lower()
    try:
        if ext == ".csv":
            df = pd.read_csv(path)
            return {"ok": True, "file_type": "csv", "path": relative_path,
                    **_inspect_dataframe_v2(df)}
        if ext in {".db", ".sqlite", ".sqlite3"}:
            uri = f"file:{path.resolve().as_posix()}?mode=ro"
            conn = _sqlite3.connect(uri, uri=True)
            try:
                tables = [r[0] for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' "
                    "AND name NOT LIKE 'sqlite_%' ORDER BY name").fetchall()]
                tables_out = {}
                for tbl in tables:
                    df = pd.read_sql_query(f'SELECT * FROM "{tbl}"', conn)
                    tables_out[tbl] = _inspect_dataframe_v2(df)
            finally:
                conn.close()
            return {"ok": True, "file_type": "sqlite",
                    "path": relative_path, "tables": tables_out}
        if ext == ".json":
            df = pd.read_json(path)
            return {"ok": True, "file_type": "json", "path": relative_path,
                    **_inspect_dataframe_v2(df)}
        if ext == ".parquet":
            df = pd.read_parquet(path)
            return {"ok": True, "file_type": "parquet", "path": relative_path,
                    **_inspect_dataframe_v2(df)}
        return {"ok": False, "error":
                f"Unsupported extension: {ext}. Supported: .csv .db .sqlite .json .parquet"}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc!s}"}
