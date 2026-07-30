"""Shared DuckDB view registration and schema inspection helpers."""

from __future__ import annotations

import logging
import json
import re
import sqlite3
from pathlib import Path
from typing import Any

import duckdb


DEFAULT_DUCKDB_JSON_MAXIMUM_OBJECT_SIZE = 16 * 1024 * 1024
logger = logging.getLogger(__name__)


def quote_duckdb_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def quote_duckdb_path(path: Path) -> str:
    return str(path).replace("'", "''")


def asset_view_name(asset_path: str) -> str:
    return Path(asset_path).stem


def json_schema_has_records_fields(schema: dict[str, Any]) -> bool:
    if schema.get("json_structure") == "object_with_records":
        return True
    return any(
        str(field.get("name", "")).startswith("records.")
        or str(field.get("json_path", "")).startswith("records.")
        for field in schema.get("fields", [])
    )


def sqlite_view_names(catalog: dict[str, Any]) -> dict[tuple[str, str], str]:
    reserved_names = {
        asset_view_name(str(schema.get("asset_path", "")))
        for schema in catalog.get("schemas", [])
        if schema.get("kind") in {"csv", "json"} and schema.get("asset_path")
    }
    sqlite_tables: list[tuple[str, str]] = []
    table_counts: dict[str, int] = {}
    for schema in catalog.get("schemas", []):
        if schema.get("kind") != "sqlite":
            continue
        asset_path = str(schema.get("asset_path", ""))
        for table in schema.get("tables", []):
            table_name = str(table.get("name", ""))
            if not asset_path or not table_name:
                continue
            sqlite_tables.append((asset_path, table_name))
            table_counts[table_name] = table_counts.get(table_name, 0) + 1

    used_names = set(reserved_names)
    view_names: dict[tuple[str, str], str] = {}
    for asset_path, table_name in sqlite_tables:
        if table_counts.get(table_name, 0) == 1 and table_name not in used_names:
            view_name = table_name
        else:
            view_name = f"{asset_view_name(asset_path)}__{table_name}"
        base_name = view_name
        suffix = 2
        while view_name in used_names:
            view_name = f"{base_name}__{suffix}"
            suffix += 1
        used_names.add(view_name)
        view_names[(asset_path, table_name)] = view_name
    return view_names


def connect_sqlite_read_only(path: Path) -> sqlite3.Connection:
    uri = f"file:{path.resolve().as_posix()}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def _json_maximum_object_size(path: Path) -> int:
    try:
        return max(DEFAULT_DUCKDB_JSON_MAXIMUM_OBJECT_SIZE, path.stat().st_size + 1024 * 1024)
    except OSError:
        return DEFAULT_DUCKDB_JSON_MAXIMUM_OBJECT_SIZE


def _register_sqlite_view(
    conn: duckdb.DuckDBPyConnection,
    file_path: Path,
    sqlite_table: str,
    view_name: str,
) -> None:
    import pandas as pd

    with connect_sqlite_read_only(file_path) as sqlite_conn:
        dataframe = pd.read_sql_query(
            f"SELECT * FROM {quote_duckdb_identifier(sqlite_table)}",
            sqlite_conn,
        )
    temp_name = (
        f"__sqlite_probe_{len(view_name)}_"
        f"{abs(hash((str(file_path), sqlite_table))) & 0xFFFFFFFF}"
    )
    conn.register(temp_name, dataframe)
    conn.execute(
        f"CREATE VIEW {quote_duckdb_identifier(view_name)} AS "
        f"SELECT * FROM {quote_duckdb_identifier(temp_name)}"
    )


def _sql_references_view(sql: str, view_name: str) -> bool:
    if quote_duckdb_identifier(view_name) in sql:
        return True
    pattern = rf"(?<![A-Za-z0-9_]){re.escape(view_name)}(?![A-Za-z0-9_])"
    return re.search(pattern, sql) is not None


def _derived_view_dependencies(catalog: dict[str, Any], view_name: str) -> set[tuple[str, str | None]]:
    dependencies: set[tuple[str, str | None]] = set()
    for view in catalog.get("derived_views", []):
        if str(view.get("name", "")) != view_name:
            continue
        base_source = view.get("base_source", {})
        dependencies.add((str(base_source.get("asset_path", "")), base_source.get("table")))
        for join in view.get("joins", []):
            dimension_source = join.get("dimension_source", {})
            dependencies.add(
                (str(dimension_source.get("asset_path", "")), dimension_source.get("table"))
            )
        break
    return dependencies


def _sql_references_derived_dependency(
    sql: str | None,
    catalog: dict[str, Any],
    dependency: tuple[str, str | None],
) -> bool:
    if sql is None:
        return False
    for view in catalog.get("derived_views", []):
        view_name = str(view.get("name", ""))
        if not view_name or not _sql_references_view(sql, view_name):
            continue
        if dependency in _derived_view_dependencies(catalog, view_name):
            return True
    return False


def _render_derived_view_sql(view: dict[str, Any]) -> str:
    base_table = str(view.get("base_table", ""))
    base_alias = "__base"
    dimension_aliases = {
        str(join.get("dimension_table", "")): f"__dim_{index}"
        for index, join in enumerate(view.get("joins", []), start=1)
    }
    select_parts: list[str] = []
    for column in view.get("columns", []):
        column_name = str(column.get("name", ""))
        source_table = str(column.get("source_table", ""))
        source_field = str(column.get("source_field", ""))
        if not column_name or not source_table or not source_field:
            continue
        alias = base_alias if source_table == base_table else dimension_aliases.get(source_table)
        if not alias:
            continue
        select_parts.append(
            f"{alias}.{quote_duckdb_identifier(source_field)} AS {quote_duckdb_identifier(column_name)}"
        )
    if not select_parts:
        raise ValueError(f"Derived view {view.get('name')!r} has no renderable columns.")

    sql = (
        f"CREATE VIEW {quote_duckdb_identifier(str(view['name']))} AS "
        f"SELECT {', '.join(select_parts)} "
        f"FROM {quote_duckdb_identifier(base_table)} AS {base_alias}"
    )
    for index, join in enumerate(view.get("joins", []), start=1):
        dimension_table = str(join.get("dimension_table", ""))
        source_fields = [str(field) for field in join.get("source_fields", [])]
        target_fields = [str(field) for field in join.get("target_fields", [])]
        if not dimension_table or len(source_fields) != len(target_fields) or not source_fields:
            continue
        alias = f"__dim_{index}"
        conditions = [
            f"{base_alias}.{quote_duckdb_identifier(source)} = "
            f"{alias}.{quote_duckdb_identifier(target)}"
            for source, target in zip(source_fields, target_fields, strict=False)
        ]
        sql += (
            f" LEFT JOIN {quote_duckdb_identifier(dimension_table)} AS {alias} "
            f"ON {' AND '.join(conditions)}"
        )
    # Deterministic tie-breaker: order by the base table's join key columns.
    # These are the columns that link the base table to its dimensions and are
    # typically unique per row. When the agent later queries with
    # ORDER BY <metric> DESC LIMIT N, the view's internal order serves as the
    # implicit tie-breaker for rows with equal metric values, making top-N
    # results stable regardless of the view's JOIN-induced row permutation.
    join_key_fields: list[str] = []
    for join in view.get("joins", []):
        for field in join.get("source_fields", []):
            f = str(field)
            if f not in join_key_fields:
                join_key_fields.append(f)
    if not join_key_fields:
        # Fallback: use all base columns (should not normally be reached —
        # a derived view without joins would not have been created).
        join_key_fields = [
            str(c["source_field"])
            for c in view.get("columns", [])
            if c.get("role") == "base"
        ]
    order_parts = [
        f"{base_alias}.{quote_duckdb_identifier(f)}" for f in join_key_fields
    ]
    sql += f" ORDER BY {', '.join(order_parts)}"
    return sql


def _create_referenced_derived_views(
    conn: duckdb.DuckDBPyConnection,
    catalog: dict[str, Any],
    *,
    sql: str | None,
) -> None:
    if sql is None:
        return
    for view in catalog.get("derived_views", []):
        view_name = str(view.get("name", ""))
        if not view_name or not _sql_references_view(sql, view_name):
            continue
        try:
            conn.execute(_render_derived_view_sql(view))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to create derived view %s: %s", view_name, exc)


def _register_generated_structured_doc_views(
    conn: duckdb.DuckDBPyConnection,
    context_dir: Path,
) -> None:
    manifest_path = context_dir / ".generated" / "structured_doc" / "manifest.json"
    if not manifest_path.exists():
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to read structured doc manifest %s: %s", manifest_path, exc)
        return

    entries = manifest.get("tables", [])
    if not isinstance(entries, list):
        return
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        table_name = str(entry.get("registered_table") or "").strip()
        file_name = str(entry.get("file") or "").strip()
        if not table_name or not file_name:
            continue
        file_path = (manifest_path.parent / file_name).resolve()
        if not file_path.exists():
            continue
        try:
            safe_path = quote_duckdb_path(file_path)
            conn.execute(
                f"CREATE OR REPLACE VIEW {quote_duckdb_identifier(table_name)} AS "
                f"SELECT * FROM read_json_auto('{safe_path}', format='newline_delimited')"
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to register structured doc view %s: %s", table_name, exc)


def create_duckdb_views(
    conn: duckdb.DuckDBPyConnection,
    context_dir: Path,
    catalog: dict[str, Any],
    *,
    logical_tables: list[dict[str, Any]],
    sql: str | None = None,
    register_all_sqlite: bool = False,
) -> None:
    logical_by_source = {
        (str(table.get("source_asset_path")), table.get("source_table")): str(table.get("table"))
        for table in logical_tables
    }
    for schema in catalog.get("schemas", []):
        asset_path: str = schema.get("asset_path", "")
        kind: str = schema.get("kind", "")
        if kind not in ("csv", "json"):
            continue
        file_path = context_dir / asset_path
        table_name = asset_view_name(asset_path)
        quoted_table = quote_duckdb_identifier(table_name)
        safe_path = quote_duckdb_path(file_path)
        try:
            if kind == "csv":
                conn.execute(
                    f"CREATE VIEW {quoted_table} AS "
                    f"SELECT * FROM read_csv_auto('{safe_path}')"
                )
                logical_name = logical_by_source.get((asset_path, None))
                if logical_name and logical_name != table_name:
                    conn.execute(
                        f"CREATE VIEW {quote_duckdb_identifier(logical_name)} "
                        f"AS SELECT * FROM {quoted_table}"
                    )
                continue
            max_object_size = _json_maximum_object_size(file_path)
            if json_schema_has_records_fields(schema):
                conn.execute(
                    f"CREATE VIEW {quoted_table} AS "
                    f"SELECT r.*, r AS records "
                    f"FROM read_json_auto('{safe_path}', maximum_object_size={max_object_size}), "
                    "UNNEST(records) AS t(r)"
                )
                logical_name = logical_by_source.get((asset_path, None))
                if logical_name and logical_name != table_name:
                    conn.execute(
                        f"CREATE VIEW {quote_duckdb_identifier(logical_name)} "
                        f"AS SELECT * FROM {quoted_table}"
                    )
                continue
            conn.execute(
                f"CREATE VIEW {quoted_table} AS "
                f"SELECT * FROM read_json_auto('{safe_path}', maximum_object_size={max_object_size})"
            )
            logical_name = logical_by_source.get((asset_path, None))
            if logical_name and logical_name != table_name:
                conn.execute(
                    f"CREATE VIEW {quote_duckdb_identifier(logical_name)} "
                    f"AS SELECT * FROM {quoted_table}"
                )
        except Exception:
            continue

    view_names = sqlite_view_names(catalog)
    for schema in catalog.get("schemas", []):
        if schema.get("kind") != "sqlite":
            continue
        asset_path = str(schema.get("asset_path", ""))
        file_path = context_dir / asset_path
        for table in schema.get("tables", []):
            table_name = str(table.get("name", ""))
            view_name = view_names.get((asset_path, table_name))
            if not view_name:
                continue
            dependency = (asset_path, table_name)
            referenced_by_derived_view = _sql_references_derived_dependency(
                sql,
                catalog,
                dependency,
            )
            if (
                not register_all_sqlite
                and (
                    sql is None
                    or (
                        not _sql_references_view(sql, view_name)
                        and not referenced_by_derived_view
                    )
                )
            ):
                continue
            try:
                _register_sqlite_view(conn, file_path, table_name, view_name)
                logical_name = logical_by_source.get((asset_path, table_name))
                if logical_name and logical_name != view_name:
                    conn.execute(
                        f"CREATE VIEW {quote_duckdb_identifier(logical_name)} AS "
                        f"SELECT * FROM {quote_duckdb_identifier(view_name)}"
                    )
            except Exception:
                continue
    _create_referenced_derived_views(conn, catalog, sql=sql)
    _register_generated_structured_doc_views(conn, context_dir)


def validate_derived_views(
    context_dir: Path,
    catalog: dict[str, Any],
    *,
    logical_tables: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Return only derived views that DuckDB can create without changing base grain."""
    candidate_views = list(catalog.get("derived_views", []))
    if not candidate_views:
        return [], []

    validation_catalog = {**catalog, "derived_views": candidate_views}
    conn = duckdb.connect(":memory:")
    valid_views: list[dict[str, Any]] = []
    warnings: list[str] = []
    try:
        create_duckdb_views(
            conn,
            context_dir,
            validation_catalog,
            logical_tables=logical_tables,
            register_all_sqlite=True,
        )
        for view in candidate_views:
            view_name = str(view.get("name", ""))
            base_table = str(view.get("base_table", ""))
            if not view_name or not base_table:
                warnings.append(f"derived_view_invalid_metadata:{view_name or '<unnamed>'}")
                continue
            try:
                conn.execute(_render_derived_view_sql(view))
                described = conn.execute(
                    f"DESCRIBE {quote_duckdb_identifier(view_name)}"
                ).fetchall()
                view_count = int(
                    conn.execute(
                        f"SELECT COUNT(*) FROM {quote_duckdb_identifier(view_name)}"
                    ).fetchone()[0]
                )
                base_count = int(
                    conn.execute(
                        f"SELECT COUNT(*) FROM {quote_duckdb_identifier(base_table)}"
                    ).fetchone()[0]
                )
            except Exception as exc:  # noqa: BLE001
                message = f"derived_view_validation_failed:{view_name}:{exc}"
                warnings.append(message)
                logger.warning("Derived view validation failed for %s: %s", view_name, exc)
                continue

            if view_count != base_count:
                message = (
                    f"derived_view_fanout:{view_name}:base_rows={base_count}:"
                    f"view_rows={view_count}"
                )
                warnings.append(message)
                logger.warning(
                    "Derived view %s changes base grain: base_rows=%s view_rows=%s",
                    view_name,
                    base_count,
                    view_count,
                )
                continue

            columns_by_name = {
                str(column.get("name", "")): column
                for column in view.get("columns", [])
                if column.get("name")
            }
            validated_columns: list[dict[str, Any]] = []
            for row in described:
                column_name = str(row[0])
                column_type = str(row[1])
                column = dict(columns_by_name.get(column_name, {"name": column_name}))
                column["name"] = column_name
                column["type"] = column_type
                validated_columns.append(column)
            validated_view = dict(view)
            validated_view["row_count"] = view_count
            validated_view["columns"] = validated_columns
            valid_views.append(validated_view)
    finally:
        conn.close()
    return valid_views, warnings


def inspect_duckdb_logical_schemas(
    context_dir: Path,
    catalog: dict[str, Any],
    *,
    logical_tables: list[dict[str, Any]],
) -> dict[str, dict[str, str]]:
    """Return DuckDB column types for every logical table that can be registered."""
    conn = duckdb.connect(":memory:")
    try:
        create_duckdb_views(
            conn,
            context_dir,
            catalog,
            logical_tables=logical_tables,
            register_all_sqlite=True,
        )
        schemas: dict[str, dict[str, str]] = {}
        for logical in logical_tables:
            table_name = str(logical.get("table", ""))
            if not table_name:
                continue
            try:
                rows = conn.execute(
                    f"DESCRIBE {quote_duckdb_identifier(table_name)}"
                ).fetchall()
            except Exception:
                continue
            schemas[table_name] = {str(row[0]): str(row[1]) for row in rows}
        return schemas
    finally:
        conn.close()
