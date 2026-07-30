#!/usr/bin/env python3
"""Read-only task sandbox viewer with a SQLite console.

Run:
  uv run python scripts/task_sandbox_viewer.py --host 127.0.0.1 --port 8768
"""
from __future__ import annotations

import argparse
import csv
import hmac
import json
import re
import sqlite3
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import duckdb


REPO = Path(__file__).resolve().parents[1]
INPUT_ROOT = REPO / "data" / "phase2_demo" / "demo_samples_phase2" / "input"
OUTPUT_ROOT = REPO / "data" / "phase2_demo" / "demo_samples_phase2" / "output"
TEXT_LIMIT = 1_500_000
PDF_TEXT_LIMIT = 500_000
DEFAULT_QUERY_LIMIT = 500
MAX_QUERY_LIMIT = 5000
SQL_TIMEOUT_MS = 8000
DUCKDB_TIMEOUT_MS = 10_000
MEDIA_CONTENT_TYPES = {
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".webm": "video/webm",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}
_DUCK_LOCK = threading.RLock()
_DUCK_CONN_CACHE: dict[str, duckdb.DuckDBPyConnection] = {}
_DUCK_CATALOG_CACHE: dict[str, list[dict[str, Any]]] = {}


def _json_default(obj: Any) -> str:
    return str(obj)


def _read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_text(path: Path, *, limit: int = TEXT_LIMIT) -> dict[str, Any]:
    if not path.is_file():
        return {"exists": False, "text": "", "truncated": False, "size": 0}
    size = path.stat().st_size
    data = path.read_bytes()[:limit]
    return {
        "exists": True,
        "text": data.decode("utf-8", errors="replace"),
        "truncated": size > limit,
        "size": size,
    }


def _task_sort_key(task_id: str) -> tuple[int, str]:
    m = re.search(r"(\d+)$", task_id)
    return (int(m.group(1)) if m else 10**9, task_id)


def _safe_task_id(task_id: str) -> str:
    name = unquote(task_id)
    if not re.fullmatch(r"task_\d+", name):
        raise ValueError("invalid task id")
    if not (INPUT_ROOT / name).is_dir():
        raise FileNotFoundError(name)
    return name


def _safe_rel_path(task_id: str, rel_path: str) -> Path:
    task_dir = INPUT_ROOT / _safe_task_id(task_id)
    rel = unquote(rel_path).lstrip("/")
    if not rel or "\x00" in rel:
        raise ValueError("invalid path")
    path = (task_dir / rel).resolve()
    root = task_dir.resolve()
    if path != root and root not in path.parents:
        raise ValueError("path escapes task directory")
    if not path.is_file():
        raise FileNotFoundError(rel)
    return path


def _db_path(task_id: str) -> Path:
    path = INPUT_ROOT / _safe_task_id(task_id) / "context" / "db" / "sub_db.sqlite"
    if not path.is_file():
        raise FileNotFoundError(f"no sqlite database for {task_id}")
    return path


def _quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _quote_sql_string(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _connect_ro(task_id: str) -> sqlite3.Connection:
    path = _db_path(task_id)
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA query_only=ON")
    con.execute("PRAGMA busy_timeout=1000")
    start = time.monotonic()

    def progress() -> int:
        return 1 if (time.monotonic() - start) * 1000 > SQL_TIMEOUT_MS else 0

    con.set_progress_handler(progress, 20_000)
    return con


def _context_dir(task_id: str) -> Path:
    return INPUT_ROOT / _safe_task_id(task_id) / "context"


def _safe_duck_alias(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not cleaned:
        cleaned = "db"
    if cleaned[0].isdigit():
        cleaned = "_" + cleaned
    return cleaned


def _unique_view_name(preferred: str, kind: str, used: set[str]) -> str:
    if preferred not in used:
        used.add(preferred)
        return preferred
    stem = f"{preferred}__{kind}"
    candidate = stem
    i = 2
    while candidate in used:
        candidate = f"{stem}_{i}"
        i += 1
    used.add(candidate)
    return candidate


def _duck_to_json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, (list, tuple)):
        return [_duck_to_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _duck_to_json_safe(v) for k, v in value.items()}
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _duck_describe(conn: duckdb.DuckDBPyConnection, view_name: str) -> list[dict[str, Any]]:
    rows = conn.execute(f"DESCRIBE {_quote_ident(view_name)}").fetchall()
    return [
        {
            "cid": i,
            "name": str(row[0]),
            "type": str(row[1]) if len(row) > 1 else "",
            "notnull": None,
            "pk": None,
            "hidden": 0,
        }
        for i, row in enumerate(rows)
    ]


def _duck_count_rows(conn: duckdb.DuckDBPyConnection, view_name: str) -> int | None:
    try:
        return int(conn.execute(f"SELECT COUNT(*) FROM {_quote_ident(view_name)}").fetchone()[0])
    except Exception:
        return None


def _duck_attach_sqlite(
    conn: duckdb.DuckDBPyConnection, db_path: Path, alias: str, used: set[str], context_dir: Path
) -> list[dict[str, Any]]:
    conn.execute(f"ATTACH {_quote_sql_string(db_path.as_posix())} AS {alias} (TYPE SQLITE, READ_ONLY)")
    table_rows = conn.execute(
        f"SELECT table_name FROM information_schema.tables "
        f"WHERE table_catalog = {_quote_sql_string(alias)} AND table_schema = 'main'"
    ).fetchall()
    catalog = []
    for (table_name,) in table_rows:
        view_name = _unique_view_name(str(table_name), "sqlite", used)
        conn.execute(
            f"CREATE OR REPLACE VIEW {_quote_ident(view_name)} AS "
            f"SELECT * FROM {alias}.main.{_quote_ident(str(table_name))}"
        )
        catalog.append(
            {
                "engine": "duckdb",
                "kind": "sqlite",
                "type": "sqlite",
                "name": view_name,
                "source": db_path.relative_to(context_dir).as_posix(),
                "columns": _duck_describe(conn, view_name),
                "row_count": _duck_count_rows(conn, view_name),
                "sql": f"ATTACH {db_path.name} AS {alias}",
            }
        )
    return catalog


def _duck_create_csv_view(
    conn: duckdb.DuckDBPyConnection, csv_path: Path, used: set[str], context_dir: Path
) -> dict[str, Any]:
    view_name = _unique_view_name(csv_path.stem, "csv", used)
    conn.execute(
        f"CREATE OR REPLACE VIEW {_quote_ident(view_name)} AS "
        f"SELECT * FROM read_csv_auto({_quote_sql_string(csv_path.as_posix())}, header=True, sample_size=-1)"
    )
    return {
        "engine": "duckdb",
        "kind": "csv",
        "type": "csv",
        "name": view_name,
        "source": csv_path.relative_to(context_dir).as_posix(),
        "columns": _duck_describe(conn, view_name),
        "row_count": _duck_count_rows(conn, view_name),
        "sql": f"read_csv_auto({csv_path.name})",
    }


def _duck_create_json_view(
    conn: duckdb.DuckDBPyConnection, json_path: Path, used: set[str], context_dir: Path
) -> dict[str, Any]:
    view_name = _unique_view_name(json_path.stem, "json", used)
    path_sql = _quote_sql_string(json_path.as_posix())
    probe = conn.execute(
        f"SELECT * FROM read_json_auto({path_sql}, maximum_object_size=536870912) LIMIT 1"
    ).fetchall()
    cols = [d[0] for d in (conn.description or [])]
    if "records" in cols:
        conn.execute(
            f"""CREATE OR REPLACE VIEW {_quote_ident(view_name)} AS
            SELECT r.* FROM read_json_auto({path_sql}, maximum_object_size=536870912) AS j,
                   UNNEST(j.records) AS t(r)"""
        )
    else:
        conn.execute(
            f"CREATE OR REPLACE VIEW {_quote_ident(view_name)} AS "
            f"SELECT * FROM read_json_auto({path_sql}, maximum_object_size=536870912)"
        )
    return {
        "engine": "duckdb",
        "kind": "json",
        "type": "json",
        "name": view_name,
        "source": json_path.relative_to(context_dir).as_posix(),
        "columns": _duck_describe(conn, view_name),
        "row_count": _duck_count_rows(conn, view_name),
        "sql": f"read_json_auto({json_path.name})",
    }


def _build_duck_connection(context_dir: Path) -> tuple[duckdb.DuckDBPyConnection, list[dict[str, Any]]]:
    conn = duckdb.connect(database=":memory:")
    conn.execute("SET threads TO 1")
    try:
        conn.execute("SET enable_external_access = true")
    except duckdb.Error:
        pass
    try:
        conn.execute("INSTALL sqlite; LOAD sqlite;")
    except duckdb.Error:
        conn.execute("LOAD sqlite;")
    catalog: list[dict[str, Any]] = []
    used: set[str] = set()
    candidates: list[Path] = []
    for sub in ("db", "csv", "json"):
        sub_dir = context_dir / sub
        if sub_dir.is_dir():
            candidates.extend(sorted(p for p in sub_dir.rglob("*") if p.is_file()))
    if not candidates:
        candidates.extend(sorted(p for p in context_dir.glob("*") if p.is_file()))

    for path in candidates:
        suffix = path.suffix.lower()
        try:
            if suffix in {".sqlite", ".db"}:
                alias = _safe_duck_alias(path.stem) + "_db"
                catalog.extend(_duck_attach_sqlite(conn, path, alias, used, context_dir))
            elif suffix == ".csv":
                catalog.append(_duck_create_csv_view(conn, path, used, context_dir))
            elif suffix == ".json":
                catalog.append(_duck_create_json_view(conn, path, used, context_dir))
        except Exception as exc:
            catalog.append(
                {
                    "engine": "duckdb",
                    "kind": f"{suffix.lstrip('.') or 'file'}_error",
                    "type": "error",
                    "name": None,
                    "source": path.relative_to(context_dir).as_posix(),
                    "columns": [],
                    "row_count": None,
                    "error": str(exc),
                }
            )
    return conn, catalog


def _get_duck_connection(task_id: str) -> duckdb.DuckDBPyConnection:
    context_dir = _context_dir(task_id)
    key = str(context_dir.resolve())
    with _DUCK_LOCK:
        if key not in _DUCK_CONN_CACHE:
            conn, catalog = _build_duck_connection(context_dir)
            _DUCK_CONN_CACHE[key] = conn
            _DUCK_CATALOG_CACHE[key] = catalog
        return _DUCK_CONN_CACHE[key]


def _duck_catalog(task_id: str) -> list[dict[str, Any]]:
    _get_duck_connection(task_id)
    key = str(_context_dir(task_id).resolve())
    return _DUCK_CATALOG_CACHE.get(key, [])


def _task_files(task_dir: Path) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for path in sorted(task_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(task_dir).as_posix()
        stat = path.stat()
        suffix = path.suffix.lower()
        files.append(
            {
                "path": rel,
                "name": path.name,
                "dir": path.parent.relative_to(task_dir).as_posix(),
                "suffix": suffix,
                "kind": _file_kind(path),
                "size": stat.st_size,
                "mtime_iso": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)),
            }
        )
    return files


def _file_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt", ".log"}:
        return "text"
    if suffix == ".pdf":
        return "pdf"
    if suffix == ".csv":
        return "csv"
    if suffix == ".json":
        return "json"
    if suffix in {".sqlite", ".db"}:
        return "sqlite"
    if suffix in {".mp4", ".mov", ".webm"}:
        return "video"
    if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        return "image"
    return "binary"


def _gold_preview(task_id: str) -> dict[str, Any]:
    path = OUTPUT_ROOT / task_id / "gold.csv"
    if not path.is_file():
        return {"exists": False, "columns": [], "rows": []}
    text = _read_text(path)["text"]
    rows = list(csv.reader(text.splitlines()))
    return {"exists": True, "columns": rows[0] if rows else [], "rows": rows[1:50], "raw": text}


def _db_overview(task_id: str, *, include_columns: bool = False) -> dict[str, Any]:
    path = _db_path(task_id)
    tables: list[dict[str, Any]] = []
    with _connect_ro(task_id) as con:
        rows = con.execute(
            """
            SELECT type, name, tbl_name, sql
            FROM sqlite_master
            WHERE type IN ('table', 'view')
            ORDER BY type, name
            """
        ).fetchall()
        for row in rows:
            item: dict[str, Any] = {
                "type": row["type"],
                "name": row["name"],
                "tbl_name": row["tbl_name"],
                "sql": row["sql"],
                "columns": [],
                "row_count": None,
            }
            try:
                item["row_count"] = con.execute(
                    f"SELECT COUNT(*) AS n FROM {_quote_ident(row['name'])}"
                ).fetchone()["n"]
            except Exception:
                item["row_count"] = None
            if include_columns:
                try:
                    cols = con.execute(f"PRAGMA table_xinfo({_quote_ident(row['name'])})").fetchall()
                    item["columns"] = [
                        {
                            "cid": col["cid"],
                            "name": col["name"],
                            "type": col["type"],
                            "notnull": col["notnull"],
                            "pk": col["pk"],
                            "hidden": col["hidden"],
                        }
                        for col in cols
                    ]
                except Exception:
                    item["columns"] = []
            tables.append(item)
    return {
        "db_path": str(path),
        "db_size": path.stat().st_size,
        "tables": tables,
        "table_count": sum(1 for t in tables if t["type"] == "table"),
        "view_count": sum(1 for t in tables if t["type"] == "view"),
    }


def _db_table_count(task_id: str) -> int | None:
    try:
        with _connect_ro(task_id) as con:
            return con.execute(
                "SELECT COUNT(*) AS n FROM sqlite_master WHERE type = 'table'"
            ).fetchone()["n"]
    except Exception:
        return None


def list_tasks() -> dict[str, Any]:
    tasks = []
    for task_dir in sorted(INPUT_ROOT.glob("task_*"), key=lambda p: _task_sort_key(p.name)):
        if not task_dir.is_dir():
            continue
        task_json = _read_json(task_dir / "task.json") or {}
        files = _task_files(task_dir)
        db_file = task_dir / "context" / "db" / "sub_db.sqlite"
        tables = _db_table_count(task_dir.name) if db_file.is_file() else None
        tasks.append(
            {
                "task_id": task_dir.name,
                "question": task_json.get("question"),
                "db_exists": db_file.is_file(),
                "db_size": db_file.stat().st_size if db_file.is_file() else None,
                "table_count": tables,
                "file_count": len(files),
                "pdf_count": sum(1 for f in files if f["kind"] == "pdf"),
                "md_count": sum(1 for f in files if f["suffix"] == ".md"),
                "video_count": sum(1 for f in files if f["kind"] == "video"),
            }
        )
    return {"tasks": tasks, "root": str(INPUT_ROOT)}


def get_task(task_id: str) -> dict[str, Any]:
    tid = _safe_task_id(task_id)
    task_dir = INPUT_ROOT / tid
    task_json = _read_json(task_dir / "task.json") or {}
    knowledge = _read_text(task_dir / "context" / "knowledge.md")
    files = _task_files(task_dir)
    db = _db_overview(tid, include_columns=True) if (task_dir / "context" / "db" / "sub_db.sqlite").is_file() else None
    duck_entries = _duck_catalog(tid) if (task_dir / "context").is_dir() else []
    return {
        "task_id": tid,
        "question": task_json.get("question"),
        "task_json": task_json,
        "paths": {"task_dir": str(task_dir), "input_root": str(INPUT_ROOT)},
        "knowledge": knowledge,
        "files": files,
        "db": db,
        "duckdb": {
            "catalog": duck_entries,
            "view_count": sum(1 for entry in duck_entries if entry.get("name")),
            "error_count": sum(1 for entry in duck_entries if entry.get("type") == "error"),
        },
        "gold": _gold_preview(tid),
    }


def get_table(task_id: str, table: str, *, limit: int = 200, offset: int = 0) -> dict[str, Any]:
    tid = _safe_task_id(task_id)
    limit = max(1, min(int(limit), MAX_QUERY_LIMIT))
    offset = max(0, int(offset))
    with _connect_ro(tid) as con:
        exists = con.execute(
            "SELECT 1 FROM sqlite_master WHERE type IN ('table','view') AND name = ?", (table,)
        ).fetchone()
        if not exists:
            raise FileNotFoundError(table)
        cols = con.execute(f"PRAGMA table_xinfo({_quote_ident(table)})").fetchall()
        rows = con.execute(
            f"SELECT * FROM {_quote_ident(table)} LIMIT ? OFFSET ?", (limit + 1, offset)
        ).fetchall()
    columns = [c["name"] for c in cols if c["hidden"] == 0]
    out_rows = [[row[col] for col in row.keys()] for row in rows[:limit]]
    return {
        "table": table,
        "columns": list(rows[0].keys()) if rows else columns,
        "rows": out_rows,
        "limit": limit,
        "offset": offset,
        "truncated": len(rows) > limit,
    }


_DENY_SQL_RE = re.compile(
    r"\b(attach|detach|insert|update|delete|replace|create|drop|alter|vacuum|reindex|"
    r"analyze|begin|commit|rollback|savepoint|release|pragma\s+writable_schema)\b",
    re.IGNORECASE,
)
_READ_ONLY_PRAGMA_RE = re.compile(
    r"^\s*pragma\s+(main\.)?"
    r"(table_info|table_xinfo|table_list|index_list|index_info|index_xinfo|"
    r"foreign_key_list|database_list|schema_version)\b",
    re.IGNORECASE,
)
_DUCK_DENY_SQL_RE = re.compile(
    r"\b(attach|detach|insert|update|delete|replace|create|drop|alter|vacuum|reindex|"
    r"analyze|begin|commit|rollback|savepoint|release|copy|export|install|load|call|"
    r"set|reset|pragma\s+writable_schema)\b",
    re.IGNORECASE,
)
_DUCK_ALLOWED_PREFIX_RE = re.compile(r"^\s*(select|with|describe|show|explain|pragma)\b", re.IGNORECASE)
_DUCK_ALLOWED_PRAGMA_RE = re.compile(
    r"^\s*pragma\s+(table_info|table_xinfo|table_list|show_tables|database_list|version)\b",
    re.IGNORECASE,
)


def _clean_sql(sql: str) -> str:
    text = (sql or "").strip()
    if not text:
        raise ValueError("SQL is empty")
    text = re.sub(r";+\s*$", "", text)
    if ";" in text:
        raise ValueError("Only one SQL statement is allowed")
    if _DENY_SQL_RE.search(text):
        raise ValueError("Only read-only SQL is allowed")
    if not re.match(r"^\s*(select|with|explain|pragma)\b", text, re.IGNORECASE):
        raise ValueError("SQL must start with SELECT, WITH, EXPLAIN, or a read-only PRAGMA")
    if re.match(r"^\s*pragma\b", text, re.IGNORECASE) and not _READ_ONLY_PRAGMA_RE.match(text):
        raise ValueError("Only schema-inspection PRAGMA statements are allowed")
    return text


def _clean_duck_sql(sql: str) -> str:
    text = (sql or "").strip()
    if not text:
        raise ValueError("SQL is empty")
    text = re.sub(r";+\s*$", "", text)
    if ";" in text:
        raise ValueError("Only one SQL statement is allowed")
    if _DUCK_DENY_SQL_RE.search(text):
        raise ValueError("Only read-only DuckDB SQL is allowed")
    if not _DUCK_ALLOWED_PREFIX_RE.match(text):
        raise ValueError("DuckDB SQL must start with SELECT, WITH, DESCRIBE, SHOW, EXPLAIN, or a read-only PRAGMA")
    if re.match(r"^\s*pragma\b", text, re.IGNORECASE) and not _DUCK_ALLOWED_PRAGMA_RE.match(text):
        raise ValueError("Only schema-inspection DuckDB PRAGMA statements are allowed")
    return text


def execute_sqlite_query(task_id: str, sql: str, *, limit: int = DEFAULT_QUERY_LIMIT) -> dict[str, Any]:
    tid = _safe_task_id(task_id)
    sql = _clean_sql(sql)
    limit = max(1, min(int(limit), MAX_QUERY_LIMIT))
    started = time.monotonic()
    with _connect_ro(tid) as con:
        cur = con.execute(sql)
        columns = [desc[0] for desc in (cur.description or [])]
        rows = cur.fetchmany(limit + 1)
    elapsed_ms = round((time.monotonic() - started) * 1000, 1)
    return {
        "columns": columns,
        "rows": [[row[col] for col in row.keys()] for row in rows[:limit]],
        "row_count": min(len(rows), limit),
        "truncated": len(rows) > limit,
        "limit": limit,
        "elapsed_ms": elapsed_ms,
        "engine": "sqlite",
        "sql": sql,
    }


def execute_duckdb_query(task_id: str, sql: str, *, limit: int = DEFAULT_QUERY_LIMIT) -> dict[str, Any]:
    tid = _safe_task_id(task_id)
    sql = _clean_duck_sql(sql)
    limit = max(1, min(int(limit), MAX_QUERY_LIMIT))
    started = time.monotonic()
    with _DUCK_LOCK:
        conn = _get_duck_connection(tid)
        cur = conn.execute(sql)
        columns = [desc[0] for desc in (cur.description or [])]
        rows = cur.fetchmany(limit + 1)
    elapsed_ms = round((time.monotonic() - started) * 1000, 1)
    return {
        "columns": columns,
        "rows": [[_duck_to_json_safe(value) for value in row] for row in rows[:limit]],
        "row_count": min(len(rows), limit),
        "truncated": len(rows) > limit,
        "limit": limit,
        "elapsed_ms": elapsed_ms,
        "engine": "duckdb",
        "sql": sql,
    }


def execute_query(
    task_id: str, sql: str, *, limit: int = DEFAULT_QUERY_LIMIT, engine: str = "duckdb"
) -> dict[str, Any]:
    if engine == "sqlite":
        return execute_sqlite_query(task_id, sql, limit=limit)
    if engine == "duckdb":
        return execute_duckdb_query(task_id, sql, limit=limit)
    raise ValueError("engine must be duckdb or sqlite")


def extract_pdf_text(path: Path) -> dict[str, Any]:
    try:
        from pypdf import PdfReader
    except Exception as exc:
        return {
            "exists": True,
            "kind": "pdf",
            "text": "",
            "error": f"pypdf unavailable: {exc}",
            "size": path.stat().st_size,
        }
    try:
        reader = PdfReader(str(path))
        chunks: list[str] = []
        truncated = False
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            block = f"\n\n--- page {i + 1} ---\n{text}"
            if sum(len(c) for c in chunks) + len(block) > PDF_TEXT_LIMIT:
                truncated = True
                break
            chunks.append(block)
        return {
            "exists": True,
            "kind": "pdf",
            "text": "".join(chunks).strip(),
            "page_count": len(reader.pages),
            "truncated": truncated,
            "size": path.stat().st_size,
        }
    except Exception as exc:
        return {"exists": True, "kind": "pdf", "text": "", "error": str(exc), "size": path.stat().st_size}


def get_file(task_id: str, rel_path: str) -> dict[str, Any]:
    path = _safe_rel_path(task_id, rel_path)
    kind = _file_kind(path)
    base = {
        "path": path.relative_to(INPUT_ROOT / _safe_task_id(task_id)).as_posix(),
        "name": path.name,
        "kind": kind,
        "suffix": path.suffix.lower(),
        "size": path.stat().st_size,
    }
    if kind == "pdf":
        return {**base, **extract_pdf_text(path)}
    if kind in {"text", "csv", "json"}:
        return {**base, **_read_text(path)}
    return {**base, "exists": True, "text": "", "truncated": False}


def get_media_path(task_id: str, rel_path: str) -> tuple[Path, str]:
    path = _safe_rel_path(task_id, rel_path)
    kind = _file_kind(path)
    if kind not in {"video", "image"}:
        raise ValueError("media preview is allowed only for video/image files")
    return path, MEDIA_CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream")


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Kobushi Task Sandbox</title>
  <style>__CSS__</style>
</head>
<body>
  <div id="app">
    <aside class="sidebar">
      <div class="brand">
        <div>
          <div class="brand-title">Task Sandbox</div>
          <div class="brand-sub">SQLite console and source explorer</div>
        </div>
        <button id="refreshBtn" class="button">Refresh</button>
      </div>
      <input id="taskSearch" class="search" placeholder="Filter tasks, question, files">
      <div id="taskList" class="task-list"></div>
    </aside>
    <main class="main">
      <header id="taskHeader" class="task-header"></header>
      <nav class="tabs">
        <button class="tab active" data-tab="sql">SQL</button>
        <button class="tab" data-tab="schema">Schema</button>
        <button class="tab" data-tab="files">Files</button>
        <button class="tab" data-tab="metadata">Metadata</button>
      </nav>
      <section id="content" class="content"></section>
    </main>
  </div>
  <script>__JS__</script>
</body>
</html>
"""


CSS = r"""
:root {
  --bg: #f5f7fa;
  --panel: #ffffff;
  --line: #d7dce5;
  --line-strong: #aeb7c5;
  --text: #182230;
  --muted: #667085;
  --strong: #101828;
  --accent: #126e82;
  --accent-soft: #e5f3f6;
  --good: #067647;
  --good-soft: #e9f8ef;
  --warn: #b54708;
  --warn-soft: #fff6e5;
  --bad: #b42318;
  --bad-soft: #fff1f0;
  --code-bg: #0f172a;
  --code-fg: #e5e7eb;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  color: var(--text);
  background: var(--bg);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  font-size: 14px;
}
button, input, textarea, select { font: inherit; }
#app {
  display: grid;
  grid-template-columns: minmax(300px, 380px) minmax(0, 1fr);
  height: 100vh;
  min-height: 640px;
}
.sidebar {
  min-width: 0;
  display: grid;
  grid-template-rows: auto auto 1fr;
  border-right: 1px solid var(--line);
  background: #fbfcfe;
}
.brand {
  min-height: 64px;
  padding: 12px;
  border-bottom: 1px solid var(--line);
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}
.brand-title { font-weight: 760; color: var(--strong); }
.brand-sub { margin-top: 2px; color: var(--muted); font-size: 12px; }
.button, .tab, .small-button {
  border: 1px solid var(--line);
  background: var(--panel);
  color: var(--text);
  border-radius: 6px;
  padding: 7px 10px;
  cursor: pointer;
}
.button:hover, .tab:hover, .small-button:hover { border-color: var(--line-strong); }
.button.primary {
  color: #fff;
  border-color: var(--accent);
  background: var(--accent);
}
.button.primary:hover { filter: brightness(0.96); }
.search {
  width: calc(100% - 24px);
  margin: 10px 12px;
  border: 1px solid var(--line);
  background: var(--panel);
  border-radius: 6px;
  padding: 9px 10px;
  outline: none;
}
.search:focus, textarea:focus, select:focus {
  border-color: var(--accent);
  box-shadow: 0 0 0 3px var(--accent-soft);
}
.task-list { min-height: 0; overflow: auto; border-top: 1px solid var(--line); }
.task-item {
  width: 100%;
  text-align: left;
  border: 0;
  border-bottom: 1px solid #edf0f4;
  background: transparent;
  padding: 10px 12px;
  cursor: pointer;
}
.task-item:hover { background: #f0f4f8; }
.task-item.active { background: var(--accent-soft); box-shadow: inset 3px 0 0 var(--accent); }
.task-name { font-weight: 720; color: var(--strong); }
.task-question {
  margin-top: 5px;
  color: var(--muted);
  font-size: 12px;
  line-height: 1.35;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}
.pill-row { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 7px; }
.pill {
  display: inline-flex;
  align-items: center;
  min-height: 22px;
  padding: 2px 7px;
  border-radius: 999px;
  color: #344054;
  background: #eef2f6;
  font-size: 12px;
  white-space: nowrap;
}
.pill.good { color: var(--good); background: var(--good-soft); }
.pill.warn { color: var(--warn); background: var(--warn-soft); }
.pill.bad { color: var(--bad); background: var(--bad-soft); }
.main {
  min-width: 0;
  display: grid;
  grid-template-rows: auto auto 1fr;
}
.task-header {
  padding: 13px 18px;
  min-height: 74px;
  border-bottom: 1px solid var(--line);
  background: var(--panel);
}
.header-title { font-size: 18px; font-weight: 780; color: var(--strong); }
.header-question { margin-top: 6px; color: var(--muted); line-height: 1.45; }
.tabs {
  display: flex;
  gap: 8px;
  padding: 10px 18px;
  border-bottom: 1px solid var(--line);
  background: #fbfcfe;
}
.tab.active { color: var(--accent); border-color: var(--accent); background: var(--accent-soft); }
.content { min-height: 0; overflow: auto; padding: 16px 18px 32px; }
.empty {
  color: var(--muted);
  padding: 20px;
  border: 1px dashed var(--line);
  background: var(--panel);
  border-radius: 8px;
}
.layout-sql {
  display: grid;
  grid-template-columns: minmax(260px, 34%) minmax(0, 1fr);
  gap: 14px;
  min-height: calc(100vh - 170px);
}
.block {
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--panel);
  overflow: hidden;
}
.block-title {
  padding: 10px 12px;
  border-bottom: 1px solid var(--line);
  background: #fbfcfe;
  color: var(--strong);
  font-weight: 720;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
}
.block-body { padding: 12px; }
.side-list { max-height: calc(100vh - 250px); overflow: auto; }
.table-button, .file-button {
  width: 100%;
  border: 0;
  border-bottom: 1px solid #edf0f4;
  background: transparent;
  text-align: left;
  padding: 9px 10px;
  cursor: pointer;
}
.table-button:hover, .file-button:hover { background: #f5f7fa; }
.table-button.active, .file-button.active { background: var(--accent-soft); }
.table-name, .file-name { font-weight: 670; color: var(--strong); overflow-wrap: anywhere; }
.table-meta, .file-meta { margin-top: 4px; color: var(--muted); font-size: 12px; }
textarea {
  width: 100%;
  min-height: 210px;
  resize: vertical;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 11px;
  outline: none;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
  line-height: 1.45;
}
.toolbar {
  display: flex;
  gap: 10px;
  align-items: center;
  flex-wrap: wrap;
  margin-top: 10px;
}
select, input.limit-input {
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 7px 8px;
  background: #fff;
}
.result-meta { color: var(--muted); font-size: 12px; margin: 10px 0; }
.table-wrap {
  overflow: auto;
  max-height: 56vh;
  border: 1px solid var(--line);
  border-radius: 7px;
}
table { border-collapse: collapse; width: 100%; background: var(--panel); }
th, td {
  border-bottom: 1px solid #edf0f4;
  padding: 7px 8px;
  text-align: left;
  vertical-align: top;
}
th {
  position: sticky;
  top: 0;
  z-index: 1;
  background: #f8fafc;
  color: #344054;
  font-weight: 700;
}
td {
  max-width: 420px;
  overflow-wrap: anywhere;
  font-size: 13px;
}
pre {
  margin: 0;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  line-height: 1.45;
  border: 1px solid var(--line);
  border-radius: 7px;
  background: #f8fafc;
  padding: 10px;
  max-height: 62vh;
  overflow: auto;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
}
.json-pre { background: var(--code-bg); color: var(--code-fg); }
.media-preview {
  width: 100%;
  max-height: min(68vh, 720px);
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #05070d;
}
.media-preview.image {
  object-fit: contain;
  background: #f8fafc;
}
.schema-grid, .files-grid {
  display: grid;
  grid-template-columns: minmax(250px, 32%) minmax(0, 1fr);
  gap: 14px;
}
.columns-list {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 8px;
}
.column-card {
  border: 1px solid var(--line);
  border-radius: 7px;
  padding: 8px;
  background: #fbfcfe;
}
.column-name { font-weight: 700; color: var(--strong); overflow-wrap: anywhere; }
.column-type { color: var(--muted); font-size: 12px; margin-top: 3px; }
.split {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
  gap: 14px;
}
@media (max-width: 980px) {
  #app { grid-template-columns: 1fr; grid-template-rows: 42vh 58vh; }
  .sidebar { border-right: 0; border-bottom: 1px solid var(--line); }
  .layout-sql, .schema-grid, .files-grid, .split { grid-template-columns: 1fr; }
  .side-list { max-height: 260px; }
}
"""


JS = r"""
const state = {
  tasks: [],
  selectedTask: null,
  detail: null,
  tab: "sql",
  engine: "duckdb",
  selectedTable: null,
  selectedFile: null,
  queryResult: null,
  queryError: null,
  fileDetail: null,
  authToken: null,
};

const $ = (id) => document.getElementById(id);

function esc(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function pill(text, cls = "") {
  return `<span class="pill ${cls}">${esc(text)}</span>`;
}

function fmtBytes(n) {
  if (n === null || n === undefined) return "-";
  const units = ["B", "KB", "MB", "GB"];
  let v = Number(n);
  let i = 0;
  while (v >= 1024 && i < units.length - 1) { v /= 1024; i++; }
  return `${v.toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
}

async function api(path, opts = {}) {
  const headers = Object.assign({"Content-Type": "application/json"}, state.authToken ? {"X-Task-Sandbox-Token": state.authToken} : {});
  const res = await fetch(path, Object.assign({}, opts, {headers: Object.assign(headers, opts.headers || {})}));
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status} ${text}`);
  }
  const ct = res.headers.get("content-type") || "";
  return ct.includes("application/json") ? res.json() : res.text();
}

function filteredTasks() {
  const q = $("taskSearch").value.trim().toLowerCase();
  if (!q) return state.tasks;
  return state.tasks.filter(t => JSON.stringify(t).toLowerCase().includes(q));
}

async function loadTasks() {
  const data = await api("/api/tasks");
  state.tasks = data.tasks || [];
  renderTaskList();
  if (!state.selectedTask && state.tasks.length) {
    await selectTask(state.tasks[0].task_id);
  }
}

async function selectTask(taskId) {
  state.selectedTask = taskId;
  state.detail = await api(`/api/tasks/${encodeURIComponent(taskId)}`);
  const tables = currentTables();
  state.selectedTable = tables.length ? tables[0].name : null;
  const files = state.detail.files || [];
  state.selectedFile = (files.find(f => f.kind === "text") || files.find(f => f.kind === "pdf") || files[0] || {}).path || null;
  state.queryResult = null;
  state.queryError = null;
  state.fileDetail = null;
  renderTaskList();
  renderHeader();
  renderContent();
}

function renderTaskList() {
  $("taskList").innerHTML = filteredTasks().map(t => {
    const active = t.task_id === state.selectedTask ? "active" : "";
    return `<button class="task-item ${active}" data-task="${esc(t.task_id)}">
      <div class="task-name">${esc(t.task_id)}</div>
      <div class="task-question">${esc(t.question || "")}</div>
      <div class="pill-row">
        ${pill(`${t.table_count ?? 0} tables`, t.db_exists ? "good" : "warn")}
        ${pill(`${t.file_count ?? 0} files`)}
        ${t.pdf_count ? pill(`${t.pdf_count} pdf`) : ""}
        ${t.video_count ? pill(`${t.video_count} video`) : ""}
      </div>
    </button>`;
  }).join("") || `<div class="empty">No tasks found.</div>`;
  document.querySelectorAll("[data-task]").forEach(btn => {
    btn.addEventListener("click", () => selectTask(btn.dataset.task));
  });
}

function renderHeader() {
  if (!state.detail) {
    $("taskHeader").innerHTML = `<div class="header-title">Select a task</div>`;
    return;
  }
  const d = state.detail;
  $("taskHeader").innerHTML = `
    <div class="header-title">${esc(d.task_id)}</div>
    <div class="header-question">${esc(d.question || "")}</div>
    <div class="pill-row">
      ${pill(`${(d.db && d.db.table_count) || 0} tables`, d.db ? "good" : "warn")}
      ${d.duckdb ? pill(`duckdb ${d.duckdb.view_count || 0} views`, "good") : ""}
      ${pill(`${(d.files || []).length} files`)}
      ${d.gold && d.gold.exists ? pill("gold.csv", "good") : ""}
      ${d.knowledge && d.knowledge.exists ? pill("knowledge.md", "good") : ""}
    </div>`;
}

function renderContent() {
  if (!state.detail) {
    $("content").innerHTML = `<div class="empty">Select a task.</div>`;
    return;
  }
  if (state.tab === "sql") renderSql();
  if (state.tab === "schema") renderSchema();
  if (state.tab === "files") renderFiles();
  if (state.tab === "metadata") renderMetadata();
}

function currentTables() {
  if (!state.detail) return [];
  if (state.engine === "sqlite") {
    return (state.detail.db && state.detail.db.tables) || [];
  }
  return ((state.detail.duckdb && state.detail.duckdb.catalog) || []).filter(t => t.name);
}

function selectedTableInfo() {
  const tables = currentTables();
  return tables.find(t => t.name === state.selectedTable) || tables[0] || null;
}

function tableListHtml() {
  const tables = currentTables();
  return `<div class="side-list">${tables.map(t => `
    <button class="table-button ${t.name === state.selectedTable ? "active" : ""}" data-table="${esc(t.name)}">
      <div class="table-name">${esc(t.name)}</div>
      <div class="table-meta">${esc(t.kind || t.type)} · ${t.row_count ?? "-"} rows · ${(t.columns || []).length} cols${t.source ? ` · ${esc(t.source)}` : ""}</div>
    </button>`).join("")}</div>`;
}

function defaultSqlForTable(name) {
  return name ? `SELECT *\nFROM "${name.replaceAll('"', '""')}"\nLIMIT 100` : "SELECT 1";
}

function renderSql() {
  const t = selectedTableInfo();
  const initial = defaultSqlForTable(t && t.name);
  $("content").innerHTML = `
    <div class="layout-sql">
      <div class="block">
        <div class="block-title">
          <span>Sources</span>
          <select id="engineSelect" title="SQL engine">
            <option value="duckdb" ${state.engine === "duckdb" ? "selected" : ""}>DuckDB: DB + CSV + JSON</option>
            <option value="sqlite" ${state.engine === "sqlite" ? "selected" : ""}>SQLite DB only</option>
          </select>
        </div>
        ${tableListHtml()}
      </div>
      <div class="block">
        <div class="block-title">
          <span>Query console</span>
          <span>${pill(state.engine)} ${state.selectedTable ? pill(state.selectedTable) : ""}</span>
        </div>
        <div class="block-body">
          <textarea id="sqlText" spellcheck="false">${esc(window.lastSql || initial)}</textarea>
          <div class="toolbar">
            <button id="runSqlBtn" class="button primary">Run</button>
            <button id="sampleSqlBtn" class="button">Sample table</button>
            <label>Limit <input id="limitInput" class="limit-input" type="number" min="1" max="5000" value="500"></label>
            <span class="pill">read-only ${state.engine === "duckdb" ? "DuckDB" : "SQLite"}</span>
          </div>
          <div id="queryOutput">${queryOutputHtml()}</div>
        </div>
      </div>
    </div>`;
  bindTableButtons();
  $("engineSelect").addEventListener("change", () => {
    state.engine = $("engineSelect").value;
    const tables = currentTables();
    state.selectedTable = tables.length ? tables[0].name : null;
    window.lastSql = defaultSqlForTable(state.selectedTable);
    state.queryResult = null;
    state.queryError = null;
    renderContent();
  });
  $("runSqlBtn").addEventListener("click", runSql);
  $("sampleSqlBtn").addEventListener("click", () => {
    window.lastSql = defaultSqlForTable(state.selectedTable);
    $("sqlText").value = window.lastSql;
  });
}

function queryOutputHtml() {
  if (state.queryError) {
    return `<div class="empty" style="border-color: var(--bad); color: var(--bad);">${esc(state.queryError)}</div>`;
  }
  const r = state.queryResult;
  if (!r) return `<div class="empty">Run SQL to see results.</div>`;
  return `
    <div class="result-meta">${r.row_count} rows · ${r.elapsed_ms} ms ${r.truncated ? "· truncated" : ""}</div>
    ${resultTable(r.columns, r.rows)}`;
}

async function runSql() {
  const sql = $("sqlText").value;
  window.lastSql = sql;
  state.queryResult = null;
  state.queryError = null;
  $("queryOutput").innerHTML = `<div class="empty">Running...</div>`;
  try {
    state.queryResult = await api(`/api/tasks/${encodeURIComponent(state.selectedTask)}/query`, {
      method: "POST",
      body: JSON.stringify({sql, limit: Number($("limitInput").value || 500), engine: state.engine}),
    });
  } catch (err) {
    state.queryError = err.message;
  }
  $("queryOutput").innerHTML = queryOutputHtml();
}

function bindTableButtons() {
  document.querySelectorAll("[data-table]").forEach(btn => {
    btn.addEventListener("click", () => {
      state.selectedTable = btn.dataset.table;
      if (state.tab === "sql") {
        window.lastSql = defaultSqlForTable(state.selectedTable);
      }
      state.queryResult = null;
      state.queryError = null;
      renderContent();
    });
  });
}

function resultTable(columns, rows) {
  if (!columns || !columns.length) return `<div class="empty">Statement returned no columns.</div>`;
  return `<div class="table-wrap"><table>
    <thead><tr>${columns.map(c => `<th>${esc(c)}</th>`).join("")}</tr></thead>
    <tbody>${(rows || []).map(row => `<tr>${row.map(v => `<td>${esc(v)}</td>`).join("")}</tr>`).join("")}</tbody>
  </table></div>`;
}

function renderSchema() {
  const t = selectedTableInfo();
  $("content").innerHTML = `
    <div class="schema-grid">
      <div class="block">
        <div class="block-title">
          <span>Tables and views</span>
          <select id="schemaEngineSelect" title="SQL engine">
            <option value="duckdb" ${state.engine === "duckdb" ? "selected" : ""}>DuckDB</option>
            <option value="sqlite" ${state.engine === "sqlite" ? "selected" : ""}>SQLite</option>
          </select>
        </div>
        ${tableListHtml()}
      </div>
      <div class="block">
        <div class="block-title">${esc((t && t.name) || "No table")}</div>
        <div class="block-body">
          ${t ? schemaDetailHtml(t) : `<div class="empty">No source found for this engine.</div>`}
        </div>
      </div>
    </div>`;
  bindTableButtons();
  $("schemaEngineSelect").addEventListener("change", () => {
    state.engine = $("schemaEngineSelect").value;
    const tables = currentTables();
    state.selectedTable = tables.length ? tables[0].name : null;
    window.lastSql = defaultSqlForTable(state.selectedTable);
    state.queryResult = null;
    state.queryError = null;
    renderContent();
  });
}

function schemaDetailHtml(t) {
  return `
    <div class="pill-row">
      ${pill(t.kind || t.type)}
      ${t.source ? pill(t.source) : ""}
      ${pill(`${t.row_count ?? "-"} rows`)}
      ${pill(`${(t.columns || []).length} columns`)}
    </div>
    <div style="height:12px"></div>
    <div class="columns-list">
      ${(t.columns || []).map(c => `
        <div class="column-card">
          <div class="column-name">${esc(c.name)}</div>
          <div class="column-type">${esc(c.type || "-")} ${c.pk ? "· PK" : ""} ${c.notnull ? "· NOT NULL" : ""}</div>
        </div>`).join("")}
    </div>
    <div style="height:14px"></div>
    <div class="block-title" style="border:1px solid var(--line); border-radius:7px 7px 0 0;">Create SQL</div>
    <pre>${esc(t.sql || "")}</pre>`;
}

function fileListHtml() {
  const groups = {};
  for (const f of state.detail.files || []) {
    groups[f.dir] ||= [];
    groups[f.dir].push(f);
  }
  return Object.entries(groups).map(([dir, files]) => `
    <div class="block-title">${esc(dir || ".")}</div>
    ${files.map(f => `
      <button class="file-button ${f.path === state.selectedFile ? "active" : ""}" data-file="${esc(f.path)}">
        <div class="file-name">${esc(f.name)}</div>
        <div class="file-meta">${esc(f.kind)} · ${fmtBytes(f.size)}</div>
      </button>`).join("")}`).join("");
}

function renderFiles() {
  $("content").innerHTML = `
    <div class="files-grid">
      <div class="block">
        <div class="block-title">Files</div>
        <div class="side-list">${fileListHtml()}</div>
      </div>
      <div class="block">
        <div class="block-title">${esc(state.selectedFile || "Preview")}</div>
        <div id="filePreview" class="block-body">${filePreviewHtml()}</div>
      </div>
    </div>`;
  document.querySelectorAll("[data-file]").forEach(btn => {
    btn.addEventListener("click", () => selectFile(btn.dataset.file));
  });
  if (state.selectedFile && (!state.fileDetail || state.fileDetail.path !== state.selectedFile)) {
    selectFile(state.selectedFile);
  }
}

async function selectFile(path) {
  state.selectedFile = path;
  state.fileDetail = null;
  $("filePreview").innerHTML = `<div class="empty">Loading...</div>`;
  try {
    state.fileDetail = await api(`/api/tasks/${encodeURIComponent(state.selectedTask)}/file?path=${encodeURIComponent(path)}`);
  } catch (err) {
    state.fileDetail = {error: err.message, path, text: ""};
  }
  renderFiles();
}

function mediaUrl(path) {
  const params = new URLSearchParams({path});
  if (state.authToken) params.set("token", state.authToken);
  return `/api/tasks/${encodeURIComponent(state.selectedTask)}/media?${params.toString()}`;
}

function filePreviewHtml() {
  const f = state.fileDetail;
  if (!state.selectedFile) return `<div class="empty">Select a file.</div>`;
  if (!f) return `<div class="empty">Loading...</div>`;
  if (f.error) return `<div class="empty" style="border-color: var(--bad); color: var(--bad);">${esc(f.error)}</div>`;
  const meta = `<div class="pill-row">${pill(f.kind || "")}${pill(fmtBytes(f.size))}${f.page_count ? pill(`${f.page_count} pages`) : ""}${f.truncated ? pill("truncated", "warn") : ""}</div><div style="height:10px"></div>`;
  if (["text", "csv", "json", "pdf"].includes(f.kind)) {
    const cls = f.kind === "json" ? "json-pre" : "";
    return `${meta}<pre class="${cls}">${esc(f.text || "")}</pre>`;
  }
  if (f.kind === "video") {
    return `${meta}<video class="media-preview" controls preload="metadata" src="${esc(mediaUrl(f.path))}"></video>`;
  }
  if (f.kind === "image") {
    return `${meta}<img class="media-preview image" src="${esc(mediaUrl(f.path))}" alt="${esc(f.name || "image")}">`;
  }
  return `${meta}<div class="empty">Preview is not available for this file type.</div>`;
}

function renderMetadata() {
  const d = state.detail;
  const db = d.db || {};
  const duck = d.duckdb || {};
  $("content").innerHTML = `
    <div class="split">
      <div class="block">
        <div class="block-title">Task metadata</div>
        <div class="block-body">
          <div class="pill-row">
            ${pill(`db ${fmtBytes(db.db_size)}`)}
            ${pill(`${db.table_count || 0} tables`)}
            ${pill(`${db.view_count || 0} views`)}
            ${pill(`duckdb ${duck.view_count || 0} views`, "good")}
            ${duck.error_count ? pill(`${duck.error_count} duckdb errors`, "warn") : ""}
            ${pill(`${(d.files || []).length} files`)}
          </div>
          <div style="height:10px"></div>
          <pre class="json-pre">${esc(JSON.stringify(d.task_json, null, 2))}</pre>
        </div>
      </div>
      <div class="block">
        <div class="block-title">Knowledge and gold</div>
        <div class="block-body">
          <div class="block-title" style="border:1px solid var(--line); border-radius:7px 7px 0 0;">knowledge.md</div>
          <pre>${esc((d.knowledge && d.knowledge.text) || "")}</pre>
          <div style="height:14px"></div>
          <div class="block-title" style="border:1px solid var(--line); border-radius:7px 7px 0 0;">gold.csv</div>
          ${d.gold && d.gold.exists ? resultTable(d.gold.columns, d.gold.rows) : `<div class="empty">No gold.csv found.</div>`}
        </div>
      </div>
    </div>`;
}

function setup() {
  const params = new URLSearchParams(location.search);
  state.authToken = params.get("token");
  $("refreshBtn").addEventListener("click", loadTasks);
  $("taskSearch").addEventListener("input", renderTaskList);
  document.querySelectorAll(".tab").forEach(btn => {
    btn.addEventListener("click", () => {
      state.tab = btn.dataset.tab;
      document.querySelectorAll(".tab").forEach(b => b.classList.toggle("active", b.dataset.tab === state.tab));
      renderContent();
    });
  });
  loadTasks().catch(err => {
    $("taskList").innerHTML = `<div class="empty">${esc(err.message)}</div>`;
  });
}

setup();
"""


def render_html() -> bytes:
    return HTML.replace("__CSS__", CSS).replace("__JS__", JS).encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    server_version = "KobushiTaskSandbox/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"{self.address_string()} - {fmt % args}")

    def _send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=_json_default).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, text: str, status: HTTPStatus = HTTPStatus.OK, content_type: str = "text/plain") -> None:
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self) -> None:
        body = render_html()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_media(self, path: Path, content_type: str) -> None:
        size = path.stat().st_size
        range_header = self.headers.get("Range")
        start = 0
        end = size - 1
        status = HTTPStatus.OK
        if range_header:
            m = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
            if not m:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            start_s, end_s = m.groups()
            if start_s == "" and end_s == "":
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            if start_s == "":
                length = int(end_s)
                start = max(0, size - length)
            else:
                start = int(start_s)
            if end_s:
                end = min(size - 1, int(end_s))
            if start >= size or start > end:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            status = HTTPStatus.PARTIAL_CONTENT
        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(length))
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with path.open("rb") as fh:
            fh.seek(start)
            remaining = length
            while remaining > 0:
                chunk = fh.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def _access_email(self) -> str | None:
        email = self.headers.get("Cf-Access-Authenticated-User-Email")
        return email.strip().lower() if email else None

    def _check_auth(self, query: dict[str, list[str]]) -> bool:
        token = getattr(self.server, "auth_token", None)
        if token:
            started = getattr(self.server, "started_at", time.time())
            ttl_hours = getattr(self.server, "auth_token_ttl_hours", 0) or 0
            if ttl_hours and time.time() - started > ttl_hours * 3600:
                self._send_json({"error": "task sandbox token expired"}, HTTPStatus.UNAUTHORIZED)
                return False
            supplied = self.headers.get("X-Task-Sandbox-Token") or (query.get("token") or [""])[0]
            if not hmac.compare_digest(str(supplied), str(token)):
                self._send_json({"error": "task sandbox token required"}, HTTPStatus.UNAUTHORIZED)
                return False

        if not getattr(self.server, "require_cf_access", False):
            return True
        email = self._access_email()
        if not email:
            self._send_json({"error": "missing Cloudflare Access user header"}, HTTPStatus.UNAUTHORIZED)
            return False
        allowed_emails = getattr(self.server, "allowed_access_emails", set())
        allowed_domains = getattr(self.server, "allowed_access_domains", set())
        if allowed_emails and email in allowed_emails:
            return True
        if allowed_domains and any(email.endswith("@" + d) for d in allowed_domains):
            return True
        if not allowed_emails and not allowed_domains:
            return True
        self._send_json({"error": f"access denied for {email}"}, HTTPStatus.FORBIDDEN)
        return False

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        path = parsed.path.rstrip("/") or "/"
        if path == "/healthz":
            self._send_text("ok")
            return
        if not self._check_auth(query):
            return
        try:
            if path == "/":
                self._send_html()
            elif path == "/api/tasks":
                self._send_json(list_tasks())
            elif re.fullmatch(r"/api/tasks/task_\d+", path):
                task_id = path.rsplit("/", 1)[1]
                self._send_json(get_task(task_id))
            elif re.fullmatch(r"/api/tasks/task_\d+/file", path):
                task_id = path.split("/")[3]
                rel = (query.get("path") or [""])[0]
                self._send_json(get_file(task_id, rel))
            elif re.fullmatch(r"/api/tasks/task_\d+/media", path):
                task_id = path.split("/")[3]
                rel = (query.get("path") or [""])[0]
                media_path, content_type = get_media_path(task_id, rel)
                self._send_media(media_path, content_type)
            elif re.fullmatch(r"/api/tasks/task_\d+/table/[^/]+", path):
                parts = path.split("/")
                task_id = parts[3]
                table = unquote(parts[5])
                limit = int((query.get("limit") or ["200"])[0])
                offset = int((query.get("offset") or ["0"])[0])
                self._send_json(get_table(task_id, table, limit=limit, offset=offset))
            else:
                self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        except FileNotFoundError as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.NOT_FOUND)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        path = parsed.path.rstrip("/")
        if not self._check_auth(query):
            return
        try:
            if re.fullmatch(r"/api/tasks/task_\d+/query", path):
                task_id = path.split("/")[3]
                length = int(self.headers.get("Content-Length") or "0")
                body = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
                self._send_json(
                    execute_query(
                        task_id,
                        body.get("sql") or "",
                        limit=int(body.get("limit") or DEFAULT_QUERY_LIMIT),
                        engine=str(body.get("engine") or "duckdb"),
                    )
                )
            else:
                self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except sqlite3.OperationalError as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except duckdb.Error as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8768)
    parser.add_argument("--require-cf-access", action="store_true")
    parser.add_argument("--access-email", action="append", default=[])
    parser.add_argument("--access-email-domain", action="append", default=[])
    parser.add_argument("--auth-token")
    parser.add_argument("--auth-token-ttl-hours", type=float, default=0)
    args = parser.parse_args()

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.require_cf_access = args.require_cf_access
    httpd.allowed_access_emails = {e.strip().lower() for e in args.access_email if e.strip()}
    httpd.allowed_access_domains = {
        d.strip().lower().removeprefix("@") for d in args.access_email_domain if d.strip()
    }
    httpd.auth_token = args.auth_token
    httpd.auth_token_ttl_hours = args.auth_token_ttl_hours
    httpd.started_at = time.time()
    print(f"Task sandbox viewer: http://{args.host}:{args.port}")
    print(f"Input root: {INPUT_ROOT}")
    if args.auth_token:
        print("Task sandbox token check: enabled")
        if args.auth_token_ttl_hours:
            print(f"Task sandbox token TTL: {args.auth_token_ttl_hours:g}h")
    if args.require_cf_access:
        print("Cloudflare Access header check: enabled")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
