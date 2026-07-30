"""Build source-field context for final answer validation.

The context is intentionally structural: table names, field names, aliases, and
lineage only. It must not include row samples or source value examples.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from sqlglot import exp, parse
from sqlglot.errors import ParseError

from data_agent_baseline.inspectors.semantic_catalog import iter_logical_tables


SubmissionFieldContext = dict[str, Any]


def build_submission_field_context(
    submission_context: dict[str, Any] | None,
    answer: dict[str, Any] | None,
    *,
    catalog: dict[str, Any] | None,
    context_dir: str | Path | None = None,
) -> SubmissionFieldContext:
    """Create a bounded field whitelist context from final submission source."""
    context: SubmissionFieldContext = {
        "status": "unavailable",
        "source_tool": None,
        "source_tables": [],
        "field_universe": [],
        "output_lineage": [],
        "join_edges": [],
        "filter_fields": [],
        "warnings": [],
    }
    if not isinstance(submission_context, dict):
        context["warnings"].append({"kind": "missing_submission_context"})
        return context

    source_tool = submission_context.get("source_tool")
    source_tool_args = submission_context.get("source_tool_args")
    context["source_tool"] = source_tool
    sql_items: list[dict[str, Any]] = []

    if source_tool == "execute_probe_query":
        queries = _queries_from_args(source_tool_args)
        selected_index = submission_context.get("selected_query_index")
        if isinstance(selected_index, int) and 0 <= selected_index < len(queries):
            sql_items.append({"sql": queries[selected_index], "location": f"queries[{selected_index}]"})
        elif queries:
            index = len(queries) - 1
            sql_items.append({"sql": queries[index], "location": f"queries[{index}]"})
            if len(queries) > 1:
                context["warnings"].append({"kind": "selected_query_index_unavailable"})
    elif source_tool == "execute_python":
        code = ""
        if isinstance(source_tool_args, dict):
            code = str(source_tool_args.get("code") or "")
        sql_items, python_warnings = _extract_python_sql_items(code)
        context["warnings"].extend(python_warnings)
    else:
        context["warnings"].append({"kind": "unsupported_source_tool", "source_tool": source_tool})

    if not sql_items:
        context["status"] = "partial"
        context["warnings"].append({"kind": "no_static_sql_found"})
        _attach_answer_columns(context, answer)
        return context

    table_profiles = _build_table_profiles(catalog, context_dir=context_dir, warnings=context["warnings"])
    for item in sql_items:
        _merge_sql_context(context, item, table_profiles)

    _dedupe_context_lists(context)
    _attach_answer_columns(context, answer)
    context["status"] = "complete" if not _has_partial_warning(context) else "partial"
    return context


def _queries_from_args(source_tool_args: Any) -> list[str]:
    if not isinstance(source_tool_args, dict):
        return []
    if isinstance(source_tool_args.get("queries"), list):
        return [str(query) for query in source_tool_args["queries"]]
    if "sql" in source_tool_args:
        return [str(source_tool_args["sql"])]
    return []


def _extract_python_sql_items(code: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not code.strip():
        return [], [{"kind": "empty_python_source"}]
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [], [{"kind": "python_parse_error", "error": _shorten(str(exc))}]

    constants: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            if not isinstance(node.value.value, str):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    constants[target.id] = node.value.value

    visitor = _PythonSqlVisitor(code=code, constants=constants)
    visitor.visit(tree)
    return visitor.sql_items, visitor.warnings


class _PythonSqlVisitor(ast.NodeVisitor):
    def __init__(self, *, code: str, constants: dict[str, str]) -> None:
        self.code = code
        self.constants = constants
        self.sql_items: list[dict[str, Any]] = []
        self.warnings: list[dict[str, Any]] = []

    def visit_Call(self, node: ast.Call) -> Any:
        name = _call_name(node.func).rsplit(".", 1)[-1]
        if name in {"query", "query_rows"} and node.args:
            sql = _static_string(node.args[0], self.constants)
            location = f"code:line {getattr(node, 'lineno', '?')}"
            if sql is None:
                self.warnings.append({"kind": "dynamic_python_sql", "location": location})
            else:
                self.sql_items.append({"sql": sql, "location": location})
        self.generic_visit(node)


def _call_name(func: ast.AST) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        parent = _call_name(func.value)
        return f"{parent}.{func.attr}" if parent else func.attr
    return ""


def _static_string(node: ast.AST, constants: dict[str, str]) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return constants.get(node.id)
    return None


def _build_table_profiles(
    catalog: dict[str, Any] | None,
    *,
    context_dir: str | Path | None = None,
    warnings: list[dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    profiles: dict[str, dict[str, Any]] = {}
    if isinstance(catalog, dict):
        for table in iter_logical_tables(catalog):
            table_name = str(table.get("table") or "")
            if not table_name:
                continue
            profiles[table_name.lower()] = {
                "table": table_name,
                "kind": table.get("source_kind") or "logical_table",
                "fields": _compact_fields(table.get("columns", [])),
                "joins": [],
                "base_table": None,
            }

        for view in catalog.get("derived_views", []):
            if not isinstance(view, dict):
                continue
            view_name = str(view.get("name") or "")
            if not view_name:
                continue
            profiles[view_name.lower()] = {
                "table": view_name,
                "kind": "derived_view",
                "fields": _compact_fields(view.get("columns", [])),
                "joins": view.get("joins", []),
                "base_table": view.get("base_table"),
            }

    profiles.update(_structured_doc_table_profiles(context_dir, warnings=warnings))
    return profiles


def _structured_doc_table_profiles(
    context_dir: str | Path | None,
    *,
    warnings: list[dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    if context_dir is None:
        return {}
    manifest_path = Path(context_dir) / ".generated" / "structured_doc" / "manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        if warnings is not None:
            warnings.append(
                {
                    "kind": "structured_doc_manifest_read_error",
                    "path": str(manifest_path),
                    "error": _shorten(str(exc)),
                }
            )
        return {}
    entries = manifest.get("tables") if isinstance(manifest, dict) else None
    if not isinstance(entries, list):
        if warnings is not None:
            warnings.append(
                {"kind": "structured_doc_manifest_invalid", "path": str(manifest_path)}
            )
        return {}

    profiles: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        table_name = str(entry.get("registered_table") or "").strip()
        if not table_name:
            continue
        profiles[table_name.lower()] = {
            "table": table_name,
            "kind": "structured_doc_table",
            "fields": _compact_structured_doc_columns(entry.get("columns", [])),
            "joins": [],
            "base_table": None,
            "source_path": entry.get("source_path"),
            "target_table": entry.get("target_table"),
            "registered_table": table_name,
            "primary_key_field": entry.get("primary_key_field"),
        }
    return profiles


def _compact_structured_doc_columns(columns: Any) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    if not isinstance(columns, list):
        return compact
    for column in columns:
        if column is None:
            continue
        if isinstance(column, dict):
            name = column.get("name") or column.get("json_path")
            if not name:
                continue
            compact.append({"name": str(name)})
        else:
            name = str(column).strip()
            if name:
                compact.append({"name": name})
    return compact


def _compact_fields(fields: Any) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    if not isinstance(fields, list):
        return compact
    for field in fields:
        if not isinstance(field, dict):
            continue
        name = field.get("name") or field.get("json_path")
        if not name:
            continue
        item = {"name": str(name)}
        if field.get("type") is not None:
            item["type"] = field.get("type")
        if field.get("primary_key") is not None:
            item["primary_key"] = bool(field.get("primary_key"))
        source_table = field.get("source_table")
        source_field = field.get("source_field")
        if source_table is not None:
            item["source_table"] = source_table
        if source_field is not None:
            item["source_field"] = source_field
        compact.append(item)
    return compact


def _merge_sql_context(
    context: SubmissionFieldContext,
    sql_item: dict[str, Any],
    table_profiles: dict[str, dict[str, Any]],
) -> None:
    sql = str(sql_item.get("sql") or "")
    location = str(sql_item.get("location") or "sql")
    if not sql.strip():
        context["warnings"].append({"kind": "empty_sql", "location": location})
        return
    try:
        expressions = [expression for expression in parse(sql, read="duckdb") if expression is not None]
    except ParseError as exc:
        context["warnings"].append(
            {"kind": "sql_parse_error", "location": location, "error": _shorten(str(exc))}
        )
        return

    for statement_index, expression in enumerate(expressions):
        statement_location = (
            location if len(expressions) == 1 else f"{location}.statements[{statement_index}]"
        )
        for select in list(expression.find_all(exp.Select)) or [expression]:
            if isinstance(select, exp.Select):
                _merge_select_context(context, select, statement_location, table_profiles)


def _merge_select_context(
    context: SubmissionFieldContext,
    select: exp.Select,
    location: str,
    table_profiles: dict[str, dict[str, Any]],
) -> None:
    alias_map: dict[str, str] = {}
    for table in select.find_all(exp.Table):
        table_name = str(table.name)
        alias = str(table.alias_or_name)
        if not table_name or not alias:
            continue
        alias_map[alias] = table_name
        _append_source_table(context, table_name, alias, table_profiles, location)

    for edge in _join_edges(select, alias_map, location):
        context["join_edges"].append(edge)

    if len(alias_map) > 1 and not _has_join_edge_for_aliases(context["join_edges"], set(alias_map)):
        context["warnings"].append({"kind": "possible_cross_join", "location": location})

    for column in _filter_columns(select):
        resolved = _resolve_column(column, alias_map, table_profiles)
        if resolved is None:
            context["warnings"].append(
                {
                    "kind": "unresolved_filter_field",
                    "location": location,
                    "field": column.sql(dialect="duckdb"),
                }
            )
        else:
            context["filter_fields"].append(resolved)

    for projection in select.expressions:
        context["output_lineage"].append(
            _projection_lineage(projection, alias_map, table_profiles, location, context)
        )


def _append_source_table(
    context: SubmissionFieldContext,
    table_name: str,
    alias: str,
    table_profiles: dict[str, dict[str, Any]],
    location: str,
) -> None:
    profile = table_profiles.get(table_name.lower())
    if profile is None:
        profile = {
            "table": table_name,
            "kind": "unknown",
            "fields": [],
            "joins": [],
            "base_table": None,
        }
        context["warnings"].append(
            {"kind": "source_table_not_found_in_catalog", "table": table_name, "location": location}
        )
    source_entry = {
        "table": table_name,
        "alias": alias,
        "kind": profile.get("kind"),
        "base_table": profile.get("base_table"),
        "location": location,
    }
    for key in ("source_path", "target_table", "registered_table"):
        if profile.get(key) is not None:
            source_entry[key] = profile.get(key)
    context["source_tables"].append(source_entry)
    field_entry = {
        "table": table_name,
        "alias": alias,
        "kind": profile.get("kind"),
        "fields": [
            {
                **field,
                "qualified_name": f"{alias}.{field['name']}",
            }
            for field in profile.get("fields", [])
            if isinstance(field, dict) and field.get("name")
        ],
    }
    for key in ("source_path", "target_table", "registered_table", "primary_key_field"):
        if profile.get(key) is not None:
            field_entry[key] = profile.get(key)
    if profile.get("kind") == "derived_view":
        field_entry["joins"] = profile.get("joins", [])
    context["field_universe"].append(field_entry)


def _join_edges(
    select: exp.Select,
    alias_map: dict[str, str],
    location: str,
) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    for equality in select.find_all(exp.EQ):
        left = equality.left
        right = equality.right
        if not isinstance(left, exp.Column) or not isinstance(right, exp.Column):
            continue
        left_table = str(left.table or "")
        right_table = str(right.table or "")
        if not left_table or not right_table or left_table == right_table:
            continue
        if left_table not in alias_map or right_table not in alias_map:
            continue
        edges.append(
            {
                "left": _column_ref(left, alias_map),
                "right": _column_ref(right, alias_map),
                "predicate": equality.sql(dialect="duckdb"),
                "location": location,
            }
        )
    return edges


def _has_join_edge_for_aliases(edges: list[dict[str, Any]], aliases: set[str]) -> bool:
    for edge in edges:
        left_alias = str(edge.get("left", {}).get("alias") or "")
        right_alias = str(edge.get("right", {}).get("alias") or "")
        if left_alias in aliases and right_alias in aliases and left_alias != right_alias:
            return True
    return False


def _filter_columns(select: exp.Select) -> list[exp.Column]:
    columns: list[exp.Column] = []
    for clause_type in (exp.Where, exp.Having, exp.Qualify):
        for clause in select.find_all(clause_type):
            columns.extend(list(clause.find_all(exp.Column)))
    return columns


def _projection_lineage(
    projection: exp.Expression,
    alias_map: dict[str, str],
    table_profiles: dict[str, dict[str, Any]],
    location: str,
    context: SubmissionFieldContext,
) -> dict[str, Any]:
    output_name = projection.alias_or_name
    if isinstance(projection, exp.Star):
        return {"output_column": "*", "sources": [], "expression": "*", "location": location}
    if not output_name:
        output_name = projection.sql(dialect="duckdb")
    sources: list[dict[str, Any]] = []
    for column in projection.find_all(exp.Column):
        resolved = _resolve_column(column, alias_map, table_profiles)
        if resolved is None:
            context["warnings"].append(
                {
                    "kind": "unresolved_output_field",
                    "location": location,
                    "field": column.sql(dialect="duckdb"),
                }
            )
        else:
            sources.append(resolved)
    return {
        "output_column": str(output_name),
        "sources": _dedupe_dicts(sources),
        "expression": projection.sql(dialect="duckdb"),
        "location": location,
    }


def _resolve_column(
    column: exp.Column,
    alias_map: dict[str, str],
    table_profiles: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    column_name = str(column.name)
    column_table = str(column.table or "")
    if column_table:
        table_name = alias_map.get(column_table, column_table)
        return _resolved_column(table_name, column_table, column_name)

    matches: list[tuple[str, str]] = []
    for alias, table_name in alias_map.items():
        profile = table_profiles.get(table_name.lower())
        field_names = [
            str(field.get("name"))
            for field in (profile or {}).get("fields", [])
            if isinstance(field, dict) and field.get("name")
        ]
        if column_name in field_names or column_name.lower() in {name.lower() for name in field_names}:
            matches.append((alias, table_name))
    if len(matches) == 1:
        alias, table_name = matches[0]
        return _resolved_column(table_name, alias, column_name)
    if len(alias_map) == 1:
        alias, table_name = next(iter(alias_map.items()))
        return _resolved_column(table_name, alias, column_name)
    return None


def _resolved_column(table_name: str, alias: str, column_name: str) -> dict[str, Any]:
    return {
        "table": table_name,
        "alias": alias,
        "field": column_name,
        "qualified_name": f"{alias}.{column_name}",
    }


def _column_ref(column: exp.Column, alias_map: dict[str, str]) -> dict[str, Any]:
    alias = str(column.table or "")
    return _resolved_column(alias_map.get(alias, alias), alias, str(column.name))


def _attach_answer_columns(context: SubmissionFieldContext, answer: dict[str, Any] | None) -> None:
    if isinstance(answer, dict) and isinstance(answer.get("columns"), list):
        context["answer_columns"] = list(answer["columns"])


def _has_partial_warning(context: SubmissionFieldContext) -> bool:
    partial_kinds = {
        "dynamic_python_sql",
        "empty_python_source",
        "python_parse_error",
        "no_static_sql_found",
        "sql_parse_error",
        "source_table_not_found_in_catalog",
    }
    return any(warning.get("kind") in partial_kinds for warning in context.get("warnings", []))


def _dedupe_context_lists(context: SubmissionFieldContext) -> None:
    for key in ("source_tables", "field_universe", "output_lineage", "join_edges", "filter_fields", "warnings"):
        context[key] = _dedupe_dicts(context.get(key, []))


def _dedupe_dicts(items: Any) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        marker = repr(sorted(item.items(), key=lambda pair: str(pair[0])))
        if marker in seen:
            continue
        seen.add(marker)
        result.append(item)
    return result


def _shorten(text: str, limit: int = 220) -> str:
    normalized = " ".join(str(text).split())
    return normalized if len(normalized) <= limit else f"{normalized[: limit - 3]}..."
