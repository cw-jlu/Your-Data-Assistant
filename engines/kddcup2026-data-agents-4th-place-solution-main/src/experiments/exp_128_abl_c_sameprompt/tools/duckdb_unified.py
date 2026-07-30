"""DuckDB-backed unified data layer.

Auto-loads every CSV/JSON/sqlite-DB file under the task's context dir
into a single in-memory DuckDB connection, so the agent can write ONE
SQL query across all sources. CSV → view, JSON → view, .db/.sqlite →
ATTACH'ed schema.

Connection cache: keyed by absolute path of context dir, so multiple
calls within one task share state. Tasks have isolated connections
because the cache key differs.

Usage:
    conn = get_connection(task.context_dir)
    rows = execute_sql(task.context_dir, "SELECT ... FROM ...")
"""
from __future__ import annotations

import re
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

import duckdb


# Thread-local cache: keyed by (context_dir, thread_id) so concurrent attempts
# on the SAME task each get an independent DuckDB connection.
# DuckDB Python bindings are NOT thread-safe — sharing a connection across
# threads causes "pure virtual method called" / "terminate called without an
# active exception" C++ aborts (= reproduced in bench v20).
_LOCK = threading.Lock()
_CONN_CACHE: dict[tuple[str, int], duckdb.DuckDBPyConnection] = {}
_CATALOG_CACHE: dict[tuple[str, int], list[dict[str, Any]]] = {}


_VALID_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _safe_identifier(name: str) -> str:
    """Convert a filename stem into a valid SQL identifier.

    DuckDB requires identifiers to start with letter/underscore.  Replace
    invalid chars with `_`, prefix with `_` if it starts with a digit.
    """
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not cleaned:
        cleaned = "tbl"
    if cleaned[0].isdigit():
        cleaned = "_" + cleaned
    return cleaned


def _attach_sqlite_db(conn: duckdb.DuckDBPyConnection, db_path: Path, alias: str) -> list[str]:
    """ATTACH a SQLite DB and copy all of its tables into the main schema as views."""
    conn.execute(f"ATTACH '{db_path.as_posix()}' AS {alias} (TYPE SQLITE, READ_ONLY)")
    table_rows = conn.execute(
        f"SELECT table_name FROM information_schema.tables WHERE table_catalog = '{alias}' AND table_schema = 'main'"
    ).fetchall()
    materialized = []
    for (table_name,) in table_rows:
        view_name = _safe_identifier(table_name)
        # Use the qualified reference and create a top-level view for ergonomics.
        # If the same view name already exists (e.g., another DB has the same table),
        # disambiguate by prefixing alias.
        try:
            conn.execute(
                f'CREATE OR REPLACE VIEW {view_name} AS SELECT * FROM {alias}.main."{table_name}"'
            )
            materialized.append(view_name)
        except duckdb.Error:
            disambiguated = f"{alias}_{view_name}"
            conn.execute(
                f'CREATE OR REPLACE VIEW {disambiguated} AS SELECT * FROM {alias}.main."{table_name}"'
            )
            materialized.append(disambiguated)
    return materialized


def _create_csv_view(conn: duckdb.DuckDBPyConnection, csv_path: Path) -> str | None:
    view_name = _safe_identifier(csv_path.stem)
    try:
        conn.execute(
            f"CREATE OR REPLACE VIEW {view_name} AS SELECT * FROM read_csv_auto('{csv_path.as_posix()}', header=True, sample_size=-1)"
        )
        return view_name
    except duckdb.Error as exc:
        return f"!{view_name}: {exc}"


def _create_json_view(conn: duckdb.DuckDBPyConnection, json_path: Path) -> str | None:
    view_name = _safe_identifier(json_path.stem)
    # DABench JSON shape is {"table": "...", "records": [{...}, {...}]}.
    # Detect shape and UNNEST records into rows when present.
    try:
        # Probe first row to detect shape
        probe = conn.execute(
            f"SELECT * FROM read_json_auto('{json_path.as_posix()}', maximum_object_size=536870912) LIMIT 1"
        ).fetchall()
        cols = [d[0] for d in (conn.description or [])]
        if "records" in cols:
            # DABench wrapper shape — unwrap records array
            conn.execute(
                f"""CREATE OR REPLACE VIEW {view_name} AS
                SELECT r.* FROM read_json_auto('{json_path.as_posix()}', maximum_object_size=536870912) AS j,
                       UNNEST(j.records) AS t(r)"""
            )
        else:
            conn.execute(
                f"CREATE OR REPLACE VIEW {view_name} AS SELECT * FROM read_json_auto('{json_path.as_posix()}', maximum_object_size=536870912)"
            )
        return view_name
    except duckdb.Error as exc:
        return f"!{view_name}: {exc}"


def _build_connection(context_dir: Path) -> tuple[duckdb.DuckDBPyConnection, list[dict[str, Any]]]:
    """Construct a fresh DuckDB conn populated with all files in context_dir."""
    conn = duckdb.connect(database=":memory:")
    conn.execute("INSTALL sqlite; LOAD sqlite;")
    catalog: list[dict[str, Any]] = []

    # Scan db/, csv/, json/ subdirs — DABench standard layout. Only fall
    # back to top-level if the subdirs are empty (= flat-layout tasks).
    candidates: list[Path] = []
    has_subdir = False
    for sub in ("db", "csv", "json"):
        sub_dir = context_dir / sub
        if sub_dir.exists():
            has_subdir = True
            candidates.extend(sorted(p for p in sub_dir.rglob("*") if p.is_file()))
    if not has_subdir:
        candidates.extend(sorted(p for p in context_dir.glob("*") if p.is_file()))

    seen_stems: set[str] = set()
    for p in candidates:
        # Dedup by (suffix, stem) — same data file shouldn't be loaded twice
        # even if it appears in both `db/` and top-level.
        stem_key = f"{p.suffix.lower()}::{_safe_identifier(p.stem)}"
        if stem_key in seen_stems:
            continue
        seen_stems.add(stem_key)
        suffix = p.suffix.lower()
        try:
            if suffix in (".db", ".sqlite"):
                alias = _safe_identifier(p.stem) + "_db"
                names = _attach_sqlite_db(conn, p, alias)
                for n in names:
                    cols = conn.execute(f"DESCRIBE {n}").fetchall()
                    catalog.append({"source": str(p.relative_to(context_dir)), "view": n, "kind": "sqlite", "columns": [c[0] for c in cols]})
            elif suffix == ".csv":
                v = _create_csv_view(conn, p)
                if v and not v.startswith("!"):
                    cols = conn.execute(f"DESCRIBE {v}").fetchall()
                    catalog.append({"source": str(p.relative_to(context_dir)), "view": v, "kind": "csv", "columns": [c[0] for c in cols]})
                elif v:
                    catalog.append({"source": str(p.relative_to(context_dir)), "view": None, "kind": "csv_error", "error": v[1:]})
            elif suffix == ".json":
                v = _create_json_view(conn, p)
                if v and not v.startswith("!"):
                    cols = conn.execute(f"DESCRIBE {v}").fetchall()
                    catalog.append({"source": str(p.relative_to(context_dir)), "view": v, "kind": "json", "columns": [c[0] for c in cols]})
                elif v:
                    catalog.append({"source": str(p.relative_to(context_dir)), "view": None, "kind": "json_error", "error": v[1:]})
        except duckdb.Error as exc:
            catalog.append({"source": str(p.relative_to(context_dir)), "view": None, "kind": f"{suffix}_error", "error": str(exc)})

    return conn, catalog


def get_connection(context_dir: Path) -> duckdb.DuckDBPyConnection:
    # Per-thread connection: same context_dir from different threads gets
    # separate DuckDB connections (= avoids thread-unsafe shared state).
    key = (str(context_dir.resolve()), threading.get_ident())
    with _LOCK:
        if key not in _CONN_CACHE:
            conn, catalog = _build_connection(context_dir)
            _CONN_CACHE[key] = conn
            _CATALOG_CACHE[key] = catalog
        return _CONN_CACHE[key]


def get_catalog(context_dir: Path) -> list[dict[str, Any]]:
    """Return the materialized-view catalog for this task (build connection if needed)."""
    get_connection(context_dir)
    key = (str(context_dir.resolve()), threading.get_ident())
    return _CATALOG_CACHE[key]


def describe_catalog(context_dir: Path) -> str:
    """Return a human-readable description of all views available."""
    cat = get_catalog(context_dir)
    if not cat:
        return "(no data sources discovered)"
    lines = []
    for entry in cat:
        view = entry.get("view")
        kind = entry.get("kind", "?")
        src = entry.get("source", "?")
        if view is None:
            lines.append(f"- ❌ {src} ({kind}): {entry.get('error', '')[:200]}")
        else:
            cols = entry.get("columns", [])
            cols_str = ", ".join(cols[:8]) + (f", … (+{len(cols)-8})" if len(cols) > 8 else "")
            lines.append(f"- `{view}` ← {src} ({kind}): {cols_str}")
    return "\n".join(lines)


_JSON_SAFE_TYPES = (str, int, float, bool, type(None))


def _to_json_safe(value: Any) -> Any:
    """Coerce DuckDB-native types (date/datetime/Decimal/UUID/bytes) to JSON-safe."""
    if isinstance(value, _JSON_SAFE_TYPES):
        return value
    # DuckDB returns datetime.date / datetime.datetime / Decimal / UUID / bytes / list / dict
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="replace")
        except Exception:
            return value.hex()
    return str(value)


_PROSE_FILE_RE = re.compile(
    r"""read_(?:csv|csv_auto|json|json_auto|text|parquet)\s*\(\s*['"][^'"]*?\.(md|txt|markdown)\b""",
    re.IGNORECASE,
)


def execute_sql(context_dir: Path, sql: str, limit: int = 200) -> dict[str, Any]:
    """Run SQL against the unified DuckDB connection.

    Read-only enforced via prefix check (DuckDB doesn't have a per-query
    read-only mode): only allow SELECT/WITH/PRAGMA/DESCRIBE/SHOW/EXPLAIN.
    All result values are coerced to JSON-safe types.

    Prose-file guard: `.md` / `.txt` / `.markdown` files contain free-form
    narrative, not tabular rows. Attempting to parse them via DuckDB's
    file-reading functions (read_csv, read_json, etc.) wastes steps on
    a structurally impossible task. Reject such SQL with an actionable
    error pointing the agent at `read_doc` instead.
    """
    normalized = sql.lstrip().lower()
    if not normalized.startswith(("select", "with", "pragma", "describe", "show", "explain")):
        raise ValueError(
            "Only read-only SQL is allowed (SELECT/WITH/PRAGMA/DESCRIBE/SHOW/EXPLAIN)."
        )
    m = _PROSE_FILE_RE.search(sql)
    if m:
        raise ValueError(
            "Cannot SQL-parse a prose file (`.md` / `.txt`). These are "
            "free-form narrative documents, not tables. "
            "Use the `read_doc` tool to read the file, compute the answer "
            "from its text, then submit via `answer_from_sql` with a "
            "literal-wrap such as `SELECT <value> AS <col>` "
            "(or via the `answer` tool for doc-only tasks)."
        )

    conn = get_connection(context_dir)
    cursor = conn.execute(sql)
    column_names = [d[0] for d in (cursor.description or [])]
    rows = cursor.fetchmany(limit + 1)
    truncated = len(rows) > limit
    safe_rows = [[_to_json_safe(v) for v in row] for row in rows[:limit]]
    return {
        "columns": column_names,
        "rows": safe_rows,
        "row_count": len(safe_rows),
        "truncated": truncated,
    }


def reset_cache() -> None:
    """Drop all cached connections (= for testing)."""
    with _LOCK:
        for c in _CONN_CACHE.values():
            try:
                c.close()
            except Exception:
                pass
        _CONN_CACHE.clear()
        _CATALOG_CACHE.clear()
