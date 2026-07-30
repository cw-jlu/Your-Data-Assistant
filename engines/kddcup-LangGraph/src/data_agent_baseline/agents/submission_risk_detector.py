"""Programmatic risk detector for final answer submission source code."""

from __future__ import annotations

import ast
import re
from typing import Any

from sqlglot import exp, parse
from sqlglot.errors import ParseError


RiskReport = dict[str, Any]
RiskDetection = dict[str, Any]

DETECTOR_VERSION = 1
MAX_EVIDENCE_CHARS = 220


def detect_submission_risks(submission_context: dict[str, Any] | None) -> RiskReport:
    """Detect source-level answer risks for the answer validator LLM."""
    report: RiskReport = {
        "detector_version": DETECTOR_VERSION,
        "source_tool": None,
        "detected": [],
        "parse_errors": [],
        "has_high_confidence_risks": False,
    }
    if not isinstance(submission_context, dict):
        return report

    source_tool = submission_context.get("source_tool")
    source_tool_args = submission_context.get("source_tool_args")
    report["source_tool"] = source_tool

    detections: list[RiskDetection] = []
    parse_errors: list[dict[str, Any]] = []
    if source_tool == "execute_probe_query":
        for index, sql in enumerate(_queries_from_args(source_tool_args)):
            location = f"source_tool_args.queries[{index}]"
            sql_detections, sql_errors = _detect_sql_risks(sql, location=location)
            detections.extend(sql_detections)
            parse_errors.extend(sql_errors)
    elif source_tool == "execute_python":
        code = ""
        if isinstance(source_tool_args, dict):
            code = str(source_tool_args.get("code") or "")
        python_detections, python_errors = _detect_python_risks(code, location="source_tool_args.code")
        detections.extend(python_detections)
        parse_errors.extend(python_errors)

    report["detected"] = _dedupe_detections(detections)
    report["parse_errors"] = parse_errors
    report["has_high_confidence_risks"] = any(
        detection.get("confidence") == "high" for detection in report["detected"]
    )
    return report


def summarize_submission_risks(report: RiskReport | None) -> list[str]:
    """Build compact, model-facing risk summary lines."""
    if not isinstance(report, dict):
        return []
    lines: list[str] = []
    for detection in report.get("detected", []):
        if not isinstance(detection, dict):
            continue
        lines.append(
            "- {kind} ({confidence}) at {location}: {evidence}".format(
                kind=detection.get("kind", "unknown"),
                confidence=detection.get("confidence", "unknown"),
                location=detection.get("location", "unknown"),
                evidence=detection.get("evidence", ""),
            )
        )
    return lines


def _queries_from_args(source_tool_args: Any) -> list[str]:
    if not isinstance(source_tool_args, dict):
        return []
    if isinstance(source_tool_args.get("queries"), list):
        return [str(query) for query in source_tool_args["queries"]]
    if "sql" in source_tool_args:
        return [str(source_tool_args["sql"])]
    return []


def _detect_sql_risks(sql: str, *, location: str) -> tuple[list[RiskDetection], list[dict[str, Any]]]:
    detections: list[RiskDetection] = []
    parse_errors: list[dict[str, Any]] = []
    if not sql.strip():
        return detections, parse_errors

    try:
        expressions = [expression for expression in parse(sql, read="duckdb") if expression is not None]
    except ParseError as exc:
        parse_errors.append({"location": location, "error": _shorten(str(exc))})
        return _detect_sql_risks_with_regex(sql, location=location), parse_errors

    for statement_index, expression in enumerate(expressions):
        statement_location = (
            location if len(expressions) == 1 else f"{location}.statements[{statement_index}]"
        )
        detections.extend(_detect_sql_expression_risks(expression, location=statement_location))

    detections.extend(_detect_sql_risks_with_regex(sql, location=location, ast_already_used=True))
    return detections, parse_errors


def _detect_sql_expression_risks(expression: exp.Expression, *, location: str) -> list[RiskDetection]:
    detections: list[RiskDetection] = []

    for condition in _iter_sql_conditions(expression):
        if _condition_contains_not_null_filter(condition):
            detections.append(
                _risk(
                    "null_filter",
                    "high",
                    location,
                    _sql_evidence(condition),
                    "The final source appears to filter out NULL values. Check whether the question explicitly requests non-null/valid records.",
                )
            )
        if _condition_contains_empty_string_filter(condition):
            detections.append(
                _risk(
                    "null_filter",
                    "high",
                    location,
                    _sql_evidence(condition),
                    "The final source appears to filter out empty strings. Check whether the question explicitly requests non-empty/valid records.",
                )
            )

    for node in expression.find_all(exp.Limit, exp.Fetch):
        detections.append(
            _risk(
                "row_limit",
                "high",
                location,
                _sql_evidence(node),
                "The final source limits returned rows. Check whether the question explicitly asks for a limited/top/bottom/first/last result.",
            )
        )

    if expression.args.get("distinct") is not None:
        detections.append(
            _risk(
                "deduplication",
                "high",
                location,
                "SELECT DISTINCT",
                "The final source deduplicates result rows. Check whether the question explicitly asks for unique/distinct values.",
            )
        )
    for node in expression.find_all(exp.Count):
        if any(isinstance(child, exp.Distinct) for child in node.walk()):
            detections.append(
                _risk(
                    "deduplication",
                    "high",
                    location,
                    _sql_evidence(node),
                    "The final source counts distinct values. Check whether a distinct count is explicitly requested.",
                )
            )

    if expression.args.get("group") is not None:
        detections.append(
            _risk(
                "row_collapse",
                "high",
                location,
                _sql_evidence(expression.args["group"]),
                "The final source collapses rows with GROUP BY. Check whether aggregation/grouping is explicitly requested.",
            )
        )

    for cast_node in expression.find_all(exp.Cast):
        detections.append(
            _risk(
                "type_cast",
                "medium",
                location,
                _sql_evidence(cast_node),
                "The final source uses CAST to convert a column type. "
                "Preserve raw source formats by default; reject CAST unless "
                "the question explicitly requests a specific format or the "
                "cast is mathematically required for a calculation.",
            )
        )

    return detections


def _iter_sql_conditions(expression: exp.Expression) -> list[exp.Expression]:
    conditions: list[exp.Expression] = []
    for clause_type in (exp.Where, exp.Having, exp.Qualify):
        for clause in expression.find_all(clause_type):
            condition = clause.this
            if condition is not None:
                conditions.append(condition)
    return conditions


def _condition_contains_not_null_filter(condition: exp.Expression) -> bool:
    for node in condition.walk():
        if isinstance(node, exp.Not):
            child = node.this
            if isinstance(child, exp.Is) and isinstance(child.expression, exp.Null):
                return True
    return False


def _condition_contains_empty_string_filter(condition: exp.Expression) -> bool:
    for node in condition.walk():
        if isinstance(node, (exp.NEQ, exp.GT, exp.GTE)):
            if _is_empty_string_literal(node.left) or _is_empty_string_literal(node.right):
                return True
    return False


def _is_empty_string_literal(node: exp.Expression | None) -> bool:
    return isinstance(node, exp.Literal) and node.is_string and node.this == ""


def _detect_sql_risks_with_regex(
    sql: str,
    *,
    location: str,
    ast_already_used: bool = False,
) -> list[RiskDetection]:
    cleaned = _strip_sql_comments_and_strings(sql)
    detections: list[RiskDetection] = []
    confidence = "medium" if ast_already_used else "high"

    regex_checks = [
        (
            "null_filter",
            r"\bis\s+not\s+null\b",
            "SQL predicate contains IS NOT NULL.",
        ),
        (
            "null_filter",
            r"(?:!=|<>)\s*''|''\s*(?:!=|<>)",
            "SQL predicate appears to filter empty strings.",
        ),
        (
            "row_limit",
            r"\btop\s+\d+\b|\blimit\s+\d+\b|\bfetch\s+first\s+\d+\s+rows\b",
            "SQL source contains an explicit row limit.",
        ),
        (
            "type_cast",
            r"\bcast\s*\(|::\s*(?:date|text|varchar|integer|numeric|float|timestamp)\b",
            "SQL source contains an explicit type cast.",
        ),
    ]
    for kind, pattern, instruction in regex_checks:
        match = re.search(pattern, cleaned, flags=re.IGNORECASE)
        if match:
            detections.append(
                _risk(
                    kind,
                    confidence,
                    location,
                    _shorten(match.group(0)),
                    instruction,
                )
            )
    return detections


def _strip_sql_comments_and_strings(sql: str) -> str:
    result: list[str] = []
    index = 0
    while index < len(sql):
        char = sql[index]
        next_char = sql[index + 1] if index + 1 < len(sql) else ""
        if char == "-" and next_char == "-":
            end = sql.find("\n", index + 2)
            index = len(sql) if end == -1 else end
            result.append(" ")
        elif char == "/" and next_char == "*":
            end = sql.find("*/", index + 2)
            index = len(sql) if end == -1 else end + 2
            result.append(" ")
        elif char in {"'", '"'}:
            quote = char
            result.append("''" if quote == "'" else '""')
            index += 1
            while index < len(sql):
                if sql[index] == quote:
                    if index + 1 < len(sql) and sql[index + 1] == quote:
                        index += 2
                        continue
                    index += 1
                    break
                index += 1
        else:
            result.append(char)
            index += 1
    return "".join(result)


def _detect_python_risks(
    code: str,
    *,
    location: str,
) -> tuple[list[RiskDetection], list[dict[str, Any]]]:
    if not code.strip():
        return [], []
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return _detect_python_risks_with_regex(code, location=location), [
            {"location": location, "error": _shorten(str(exc))}
        ]

    visitor = _PythonRiskVisitor(code=code, location=location)
    visitor.visit(tree)
    return _dedupe_detections(visitor.detections), visitor.parse_errors


class _PythonRiskVisitor(ast.NodeVisitor):
    def __init__(self, *, code: str, location: str) -> None:
        self.code = code
        self.location = location
        self.detections: list[RiskDetection] = []
        self.parse_errors: list[dict[str, Any]] = []

    def visit_Call(self, node: ast.Call) -> Any:
        name = _call_name(node.func)
        attr = name.rsplit(".", 1)[-1]
        call_location = _python_location(self.location, node)
        evidence = _python_evidence(self.code, node)

        if attr in {"dropna", "notna", "notnull"}:
            self.detections.append(
                _risk(
                    "null_filter",
                    "high",
                    call_location,
                    evidence,
                    "Python code appears to filter or drop NULL values. Check whether the question explicitly requests non-null/valid records.",
                )
            )
        elif attr in {"head", "tail", "nlargest", "nsmallest"}:
            self.detections.append(
                _risk(
                    "row_limit",
                    "high",
                    call_location,
                    evidence,
                    "Python code appears to limit returned rows. Check whether the question explicitly asks for a limited/top/bottom/first/last result.",
                )
            )
        elif attr in {"drop_duplicates", "unique"} or name in {"set", "dict.fromkeys"}:
            self.detections.append(
                _risk(
                    "deduplication",
                    "high",
                    call_location,
                    evidence,
                    "Python code appears to deduplicate values or rows. Check whether the question explicitly asks for unique/distinct values.",
                )
            )
        elif attr in {"groupby", "pivot_table"}:
            self.detections.append(
                _risk(
                    "row_collapse",
                    "high",
                    call_location,
                    evidence,
                    "Python code appears to collapse rows through grouping or pivoting. Check whether aggregation/grouping is explicitly requested.",
                )
            )
        elif attr in {"astype", "to_datetime", "to_numeric", "to_timedelta"}:
            self.detections.append(
                _risk(
                    "type_cast",
                    "medium",
                    call_location,
                    evidence,
                    "Python code converts column types. Preserve raw source formats by default; "
                    "reject type conversion unless the question explicitly requires a specific format.",
                )
            )
        elif attr in {"date", "strftime"} and name.rsplit(".", 1)[0].endswith(".dt"):
            # .dt.date, .dt.strftime(...)
            self.detections.append(
                _risk(
                    "type_cast",
                    "medium",
                    call_location,
                    evidence,
                    "Python code strips or reformats datetime values. Preserve raw source formats.",
                )
            )

        if attr in {"query", "query_rows"} and node.args:
            first_arg = node.args[0]
            if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
                sql_detections, sql_errors = _detect_sql_risks(
                    first_arg.value,
                    location=f"{call_location}.sql",
                )
                self.detections.extend(sql_detections)
                self.parse_errors.extend(sql_errors)

        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> Any:
        if _slice_has_upper_bound(node.slice):
            confidence = "high" if _is_dataframe_indexer(node.value) else "medium"
            self.detections.append(
                _risk(
                    "row_limit",
                    confidence,
                    _python_location(self.location, node),
                    _python_evidence(self.code, node),
                    "Python slicing appears to limit returned rows. Check whether the question explicitly asks for a limited result.",
                )
            )
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> Any:
        if _compare_filters_none_or_empty(node):
            self.detections.append(
                _risk(
                    "null_filter",
                    "medium",
                    _python_location(self.location, node),
                    _python_evidence(self.code, node),
                    "Python comparison appears to filter NULL or empty values. Check whether the question explicitly requests non-null/non-empty records.",
                )
            )
        self.generic_visit(node)


def _detect_python_risks_with_regex(code: str, *, location: str) -> list[RiskDetection]:
    checks = [
        ("null_filter", r"\.(dropna|notna|notnull)\s*\(", "Python source references NULL filtering."),
        ("row_limit", r"\.(head|tail|nlargest|nsmallest)\s*\(|\[[^\]]*:\s*\d+\s*\]", "Python source references row limiting."),
        ("deduplication", r"\.(drop_duplicates|unique)\s*\(|\bset\s*\(|\bdict\.fromkeys\s*\(", "Python source references deduplication."),
        ("row_collapse", r"\.(groupby|pivot_table)\s*\(", "Python source references row collapse."),
        ("type_cast", r"\.(astype|to_datetime|to_numeric|to_timedelta)\s*\(|\.dt\.(date|strftime)\s*\(", "Python source references type conversion."),
    ]
    detections: list[RiskDetection] = []
    for kind, pattern, instruction in checks:
        match = re.search(pattern, code)
        if match:
            detections.append(_risk(kind, "medium", location, match.group(0), instruction))
    return detections


def _call_name(func: ast.AST) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        parent = _call_name(func.value)
        return f"{parent}.{func.attr}" if parent else func.attr
    return ""


def _slice_has_upper_bound(node: ast.AST) -> bool:
    if isinstance(node, ast.Slice):
        return _is_int_like(node.upper)
    return False


def _is_int_like(node: ast.AST | None) -> bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return True
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return _is_int_like(node.operand)
    return False


def _is_dataframe_indexer(node: ast.AST) -> bool:
    return isinstance(node, ast.Attribute) and node.attr in {"iloc", "loc"}


def _compare_filters_none_or_empty(node: ast.Compare) -> bool:
    values = [node.left, *node.comparators]
    compares_none_or_empty = any(_is_none_literal(value) or _is_empty_string_constant(value) for value in values)
    if not compares_none_or_empty:
        return False
    return any(isinstance(op, (ast.IsNot, ast.NotEq, ast.NotIn)) for op in node.ops)


def _is_none_literal(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


def _is_empty_string_constant(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value == ""


def _risk(
    kind: str,
    confidence: str,
    location: str,
    evidence: str,
    instruction: str,
) -> RiskDetection:
    return {
        "kind": kind,
        "confidence": confidence,
        "location": location,
        "evidence": _shorten(evidence),
        "instruction": instruction,
    }


def _sql_evidence(node: exp.Expression) -> str:
    try:
        return _shorten(node.sql(dialect="duckdb"))
    except Exception:  # noqa: BLE001
        return _shorten(str(node))


def _python_location(base_location: str, node: ast.AST) -> str:
    line = getattr(node, "lineno", None)
    return f"{base_location}:line {line}" if line is not None else base_location


def _python_evidence(code: str, node: ast.AST) -> str:
    segment = ast.get_source_segment(code, node)
    if segment:
        return _shorten(segment.strip())
    try:
        return _shorten(ast.unparse(node))
    except Exception:  # noqa: BLE001
        return type(node).__name__


def _shorten(text: str, limit: int = MAX_EVIDENCE_CHARS) -> str:
    normalized = " ".join(str(text).split())
    return normalized if len(normalized) <= limit else f"{normalized[: limit - 3]}..."


def _dedupe_detections(detections: list[RiskDetection]) -> list[RiskDetection]:
    seen: set[tuple[Any, ...]] = set()
    unique: list[RiskDetection] = []
    for detection in detections:
        key = (
            detection.get("kind"),
            detection.get("confidence"),
            detection.get("location"),
            detection.get("evidence"),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(detection)
    return unique
