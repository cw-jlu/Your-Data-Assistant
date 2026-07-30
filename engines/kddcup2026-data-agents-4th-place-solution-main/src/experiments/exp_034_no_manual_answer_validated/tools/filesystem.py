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


def profile_table(
    task: PublicTask, relative_path: str, *, max_output_chars: int = 1500
) -> dict[str, object]:
    """Return compact column statistics for a CSV file or a SQLite table.

    For SQLite, use 'path/to/file.db::TableName' to target a specific table.
    If no table name is given, returns DDL and row counts for all tables only.
    """
    import json as _json

    import pandas as pd

    try:
        if "::" in relative_path:
            file_part, table_name = relative_path.split("::", 1)
            db_path = resolve_context_path(task, file_part)
            import sqlite3

            with sqlite3.connect(str(db_path)) as conn:
                df = pd.read_sql_query(
                    f"SELECT * FROM \"{table_name}\" LIMIT 10000", conn  # noqa: S608
                )
            source = f"{file_part}::{table_name}"
        elif relative_path.endswith((".db", ".sqlite")):
            db_path = resolve_context_path(task, relative_path)
            import sqlite3

            with sqlite3.connect(str(db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                summary: dict[str, object] = {"path": relative_path, "tables": {}}
                for tbl in tables:
                    cursor.execute(f"SELECT COUNT(*) FROM \"{tbl}\"")  # noqa: S608
                    count = cursor.fetchone()[0]
                    cursor.execute(
                        f"SELECT sql FROM sqlite_master WHERE name=?", (tbl,)  # noqa: S608
                    )
                    ddl_row = cursor.fetchone()
                    ddl = ddl_row[0] if ddl_row else ""
                    summary["tables"][tbl] = {"row_count": count, "ddl": ddl}  # type: ignore[index]
            return summary
        else:
            path = resolve_context_path(task, relative_path)
            df = pd.read_csv(path, nrows=10_000)
            source = relative_path

        def _col_stats(col: pd.Series) -> dict[str, object]:
            null_count = int(col.isna().sum())
            n = len(col)
            top_vals = col.dropna().value_counts().head(5).index.tolist()
            return {
                "col": col.name,
                "dtype": str(col.dtype),
                "null_count": null_count,
                "null_pct": round(null_count / n, 3) if n else 0.0,
                "n_unique": int(col.nunique()),
                "min": (None if col.dropna().empty else col.min()),
                "max": (None if col.dropna().empty else col.max()),
                "top_values": [str(v) for v in top_vals],
            }

        columns_stats = [_col_stats(df[c]) for c in df.columns]
        result: dict[str, object] = {
            "path": source,
            "row_count": len(df),
            "columns": columns_stats,
        }

        if len(_json.dumps(result)) > max_output_chars:
            for stat in columns_stats:
                stat.pop("top_values", None)  # type: ignore[union-attr]
            result["columns"] = columns_stats
            if len(_json.dumps(result)) > max_output_chars:
                result["columns"] = columns_stats[:20]
                result["truncated_to_20_cols"] = True

        return result
    except Exception as exc:
        return {"error": str(exc)}
