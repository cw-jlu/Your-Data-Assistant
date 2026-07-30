"""SQL-only tool registry for exp_111.

NO Python execution. All computation happens in a single DuckDB
connection that exposes every CSV/JSON/sqlite-DB file in the task
context as views/tables.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from kobushi_core.benchmark.schema import AnswerTable, PublicTask
from kobushi_core.model import ModelAdapter
from experiments.exp_143_dynamic_doc.tools.filesystem import (
    grep_context,
    list_context_tree,
    read_csv_preview,
    read_doc_preview,
    read_json_preview,
)
from experiments.exp_143_dynamic_doc.tools.duckdb_unified import (
    describe_catalog,
    execute_sql,
    get_catalog,
)


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolExecutionResult:
    ok: bool
    content: dict[str, Any]
    is_terminal: bool = False
    answer: AnswerTable | None = None


ToolHandler = Callable[[PublicTask, dict[str, Any]], ToolExecutionResult]


def _list_context(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    max_depth = int(action_input.get("max_depth", 4))
    return ToolExecutionResult(ok=True, content=list_context_tree(task, max_depth=max_depth))


def _read_csv(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    path = str(action_input["path"])
    max_rows = int(action_input.get("max_rows", 20))
    return ToolExecutionResult(ok=True, content=read_csv_preview(task, path, max_rows=max_rows))


def _read_json(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    path = str(action_input["path"])
    max_chars = int(action_input.get("max_chars", 4000))
    return ToolExecutionResult(ok=True, content=read_json_preview(task, path, max_chars=max_chars))


def _read_doc(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    path = str(action_input["path"])
    max_chars = int(action_input.get("max_chars", 4000))
    search = action_input.get("search") or None
    offset = int(action_input.get("offset", 0))
    n_lines = int(action_input.get("n_lines", 100))
    return ToolExecutionResult(
        ok=True,
        content=read_doc_preview(
            task, path, max_chars=max_chars, search=search,
            offset=offset, n_lines=n_lines,
        ),
    )


def _grep(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    pattern = str(action_input["pattern"])
    path = action_input.get("path") or None
    glob = action_input.get("glob") or None
    output_mode = str(action_input.get("output_mode", "content"))
    ctx = int(action_input.get("context_lines", 0))
    ci = bool(action_input.get("case_insensitive", True))
    max_results = int(action_input.get("max_results", 50))
    max_chars = int(action_input.get("max_chars", 8000))
    return ToolExecutionResult(
        ok=True,
        content=grep_context(
            task, pattern=pattern, path=path, glob=glob,
            output_mode=output_mode, context_lines=ctx,
            case_insensitive=ci, max_results=max_results, max_chars=max_chars,
        ),
    )


def _describe_data(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    return ToolExecutionResult(
        ok=True,
        content={
            "catalog": get_catalog(task.context_dir),
            "summary": describe_catalog(task.context_dir),
        },
    )


def _coerce_sql_input(action_input: dict[str, Any] | str) -> dict[str, Any]:
    if isinstance(action_input, str):
        return {"sql": action_input}
    return action_input


def _execute_sql(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    action_input = _coerce_sql_input(action_input)
    sql = str(action_input["sql"])
    limit = int(action_input.get("limit", 200))
    try:
        result = execute_sql(task.context_dir, sql, limit=limit)
    except Exception as exc:
        return ToolExecutionResult(
            ok=False,
            content={"error": str(exc), "sql": sql},
        )
    return ToolExecutionResult(ok=True, content=result)


def _make_answer_from_sql_with_audit(
    auditor_model: ModelAdapter | None,
    question_provider,
    *,
    max_refines: int = 2,
) -> ToolHandler:
    """Self-review answer_from_sql: returns review prompt non-terminal,
    asks the agent itself to verify and call confirm_answer or refine.

    Flow:
      1. Run SQL, audit columns (= existing column auditor still runs).
      2. Return review observation: question + SQL + result preview + checklist.
         Status: "review_required". is_terminal=False.
      3. Agent's next thought re-states the goal, walks the checklist, decides:
         - call confirm_answer to commit, OR
         - call answer_from_sql again with a refined SQL.
      4. After max_refines, ANY further answer_from_sql commits as terminal.
    """
    from experiments.exp_143_dynamic_doc.column_auditor import audit_columns, apply_audit

    state = {"refines": 0, "pending_answer": None}

    def handler(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
        ai = _coerce_sql_input(action_input)
        sql = str(ai["sql"])
        # IMPLICIT-COMPUTATION GUARDS (= pre-execution rule checks based on
        # universal BIRD patterns; precision ≥ 70% on 1534 examples).
        import re as _re
        q_text = (question_provider() if question_provider else task.question)
        rule_violations = []
        # Rule 1: Q has "%" → SQL must have *100 AND /col division (prec 97%/95%)
        if _re.search(r"\b(percent|percentage|%)\b", q_text, _re.IGNORECASE):
            if not _re.search(r"\*\s*100", sql) or not _re.search(r"/\s*[a-zA-Z_]\w*", sql):
                rule_violations.append(
                    "Q has 'percent/%' but SQL is missing the classic percentage formula "
                    "(*100 + division by COUNT). Expected: CAST(SUM(CASE WHEN ...) AS REAL) * 100 / COUNT(*)"
                )
        # Rule 2: Q has "average <period>ly" → SQL needs AVG (not SUM/N)
        if _re.search(r"\baverage\s+(monthly|yearly|daily|hourly|weekly|annual)\b", q_text, _re.IGNORECASE):
            if not _re.search(r"\bAVG\s*\(", sql, _re.IGNORECASE):
                rule_violations.append(
                    "Q has 'average <period>ly' but SQL has no AVG(). Expected AVG(metric) / N, "
                    "NOT SUM(metric) / N — these differ when rows are missing."
                )
        # Rule 3: Q has "per <unit>" → SQL must have /<col> division
        if _re.search(r"\bper\s+(unit|item|day|hour|minute|capita|person|customer)\b", q_text, _re.IGNORECASE):
            if not _re.search(r"/\s*[a-zA-Z_]\w*", sql):
                rule_violations.append(
                    "Q has 'per <unit>' but SQL has no /<col> division. 'paid X per unit' usually means "
                    "Price / Amount > X, with an Amount > 0 guard to avoid divide-by-zero."
                )
        # Note: LIMIT 1 guard removed (= regressed task_80 by stripping the
        # LIMIT 1 mask without fixing the underlying filter). Math advisor's
        # filter-back formula handles superlative ties instead.
        if rule_violations:
            return ToolExecutionResult(
                ok=False,
                content={"error": "[Universal rule violation] " + " | ".join(rule_violations), "sql": sql},
            )
        try:
            # No row cap on the FINAL answer (= Phase 2 gold can exceed 12K
            # rows; the old 10000 cap silently truncated task_30/33). The
            # EXPLORE-phase execute_sql keeps its finite limit for small
            # observation context.
            result = execute_sql(task.context_dir, sql, limit=None)
        except Exception as exc:
            return ToolExecutionResult(ok=False, content={"error": str(exc), "sql": sql})
        columns = [str(c) for c in result.get("columns") or []]
        rows = [
            [("" if v is None else str(v)) for v in row]
            for row in (result.get("rows") or [])
        ]
        if not columns:
            return ToolExecutionResult(ok=False, content={"error": "SQL returned no columns.", **result})
        if not rows:
            return ToolExecutionResult(
                ok=False,
                content={
                    "error": "SQL returned 0 rows. The filter likely excludes everything. "
                             "Re-check WHERE clauses and re-run.",
                    "sql": sql, "columns": columns,
                },
            )

        # NAME-CONCAT GUARD (= Rule 16 enforcement, post-execution)
        # Detect: output column containing "full*name" AND source schema has NO native
        # `full_name` column (= agent invented it by concat). Skip warning if a real
        # `full_name` column exists somewhere in the schema (= e.g. superhero.full_name).
        import re as _re
        full_name_alias = _re.compile(r"full[_]?name", _re.IGNORECASE)
        suspect_cols = [c for c in columns if full_name_alias.search(c)]
        if suspect_cols:
            try:
                tbls = execute_sql(task.context_dir, "SHOW TABLES", limit=1000)
                has_native_fullname = False
                for tbl_row in tbls["rows"]:
                    tbl = tbl_row[0]
                    cols_res = execute_sql(task.context_dir, f'DESCRIBE "{tbl}"', limit=1000)
                    col_lcs = [str(r[0]).lower() for r in cols_res["rows"]]
                    if any(full_name_alias.search(c) for c in col_lcs):
                        has_native_fullname = True
                        break
                # Also check doc files (= prose-only entities)
                if not has_native_fullname:
                    for p in task.context_dir.rglob("*.md"):
                        try:
                            txt = p.read_text()[:5000].lower()
                            if "full_name" in txt or "fullname" in txt:
                                has_native_fullname = True
                                break
                        except Exception:
                            pass
                if not has_native_fullname:
                    return ToolExecutionResult(
                        ok=False,
                        content={
                            "error": (
                                f"[Rule 16 violation] Output column(s) {suspect_cols} contain "
                                "'full_name' but the source schema has NO native `full_name` "
                                "column — you concatenated first_name + last_name. The official "
                                "scorer expects SEPARATE first_name + last_name columns. "
                                "Rewrite SQL to: SELECT ..., first_name, last_name, ..."
                            ),
                            "sql": sql, "columns": columns,
                        },
                    )
            except Exception:
                pass  # schema check failed → don't block

        question = question_provider() if question_provider else task.question

        # Column audit (= still runs to drop irrelevant cols before review)
        col_audit_meta = None
        if auditor_model is not None and len(columns) > 1:
            audit = audit_columns(
                question=question, columns=columns, rows=rows[:50],
                model=auditor_model,
            )
            new_cols, new_rows = apply_audit(columns, rows, audit)
            col_audit_meta = {
                "kept_indices": audit.keep_indices,
                "original_n_cols": len(columns),
                "after_audit_n_cols": len(new_cols),
                "reason": audit.reason,
            }
            columns, rows = new_cols, new_rows

        answer = AnswerTable(columns=columns, rows=rows)
        state["pending_answer"] = answer
        refines_left = max_refines - state["refines"]
        state["refines"] += 1

        # If refines exhausted → forced terminal commit
        if refines_left <= 0:
            content = {
                "status": "submitted_forced",
                "column_count": len(columns),
                "row_count": len(rows),
                "note": "max refines reached; this answer is now committed.",
            }
            if col_audit_meta: content["column_audit"] = col_audit_meta
            return ToolExecutionResult(ok=True, content=content, is_terminal=True, answer=answer)

        # Non-terminal: build self-review observation (= STRICT)
        preview_rows = rows[:5]
        review_text = (
            f"=== STRICT RULE-COMPLIANCE REVIEW (mandatory before confirm) ===\n"
            f"Question: {question}\n\n"
            f"Your SQL:\n{sql}\n\n"
            f"Result: {len(rows)} rows × {len(columns)} columns "
            f"(columns: {columns})\n"
            f"Preview: {preview_rows}\n\n"
            f"In your NEXT thought, you MUST output a verdict line for EACH rule below.\n"
            f"Format: 'RULE <n>: PASS — <one-sentence justification>' OR\n"
            f"        'RULE <n>: VIOLATION — <what is wrong>'.\n"
            f"\n"
            f"  RULE 13: INTERPRET — each ambiguous noun in the question is mapped to\n"
            f"           a SPECIFIC schema column. Quote your interpretation.\n"
            f"  RULE 14: FILTER JUSTIFICATION — every WHERE/HAVING clause traces to a\n"
            f"           verbatim phrase from the question. List each clause + phrase.\n"
            f"  RULE 15: SUPERLATIVE = PER-ITEM — for 'lowest/highest/best X' the SQL\n"
            f"           uses `WHERE col = (SELECT MIN/MAX(col))`. If the SQL has\n"
            f"           GROUP BY + SUM/AVG without explicit 'total/average/sum' in the\n"
            f"           question, it is a VIOLATION. Re-write per-item.\n"
            f"  RULE 16: NAME schema-following. If the question asks for 'name' and the\n"
            f"           schema has split first_name/last_name, you MUST return both.\n"
            f"  RULE 17: ENTITY DESCRIPTOR — for 'what is the X / which X', the result\n"
            f"           is the textual representation of X (= name/text/title/url),\n"
            f"           NOT an opaque numeric ID column. ID is acceptable ONLY if the\n"
            f"           question explicitly says 'ID/id/identifier/number/code'.\n"
            f"\n"
            f"DECISION (= one of two, MUST follow your verdicts):\n"
            f"  - If ALL rules are PASS → call `confirm_answer` (no input needed).\n"
            f"  - If ANY rule is VIOLATION → call `answer_from_sql` again with a\n"
            f"    rewritten SQL that fixes the violation. Do NOT confirm a violating\n"
            f"    answer just to save a step.\n"
            f"\n"
            f"Refines remaining after this turn: {refines_left - 1}."
        )
        content = {
            "status": "review_required",
            "refines_remaining": refines_left - 1,
            "column_count": len(columns),
            "row_count": len(rows),
            "review": review_text,
        }
        if col_audit_meta: content["column_audit"] = col_audit_meta
        return ToolExecutionResult(ok=True, content=content, is_terminal=False)
    return handler


def _make_confirm_answer(answer_handler) -> ToolHandler:
    """Force-finalize the most recently proposed answer (= override audit objections)."""
    def handler(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
        # Pull the pending answer from the answer_handler's closure state via attribute
        # access. `answer_handler` is a closure; access via its __closure__.
        state = None
        for cell in (answer_handler.__closure__ or []):
            v = cell.cell_contents
            if isinstance(v, dict) and "pending_answer" in v:
                state = v; break
        if state is None or state.get("pending_answer") is None:
            return ToolExecutionResult(
                ok=False, content={"error": "No pending answer to confirm. Call answer_from_sql first."},
            )
        answer = state["pending_answer"]
        meta = state.get("pending_meta") or {}
        content = {
            "status": "submitted_via_confirm",
            "column_count": len(answer.columns),
            "row_count": len(answer.rows),
        }
        if meta.get("filter_audit"): content["filter_audit"] = meta["filter_audit"]
        return ToolExecutionResult(ok=True, content=content, is_terminal=True, answer=answer)
    return handler


def _answer(_: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    columns = action_input.get("columns")
    rows = action_input.get("rows")
    if (
        not isinstance(columns, list)
        or not columns
        or not all(isinstance(item, str) for item in columns)
    ):
        raise ValueError("answer.columns must be a non-empty list of strings.")
    if not isinstance(rows, list):
        raise ValueError("answer.rows must be a list.")

    normalized_rows: list[list[Any]] = []
    for row in rows:
        if not isinstance(row, list):
            raise ValueError("Each answer row must be a list.")
        if len(row) != len(columns):
            raise ValueError("Each answer row must match the number of columns.")
        normalized_rows.append(list(row))

    answer = AnswerTable(columns=list(columns), rows=normalized_rows)
    return ToolExecutionResult(
        ok=True,
        content={
            "status": "submitted",
            "column_count": len(columns),
            "row_count": len(normalized_rows),
        },
        is_terminal=True,
        answer=answer,
    )


@dataclass(slots=True)
class ToolRegistry:
    specs: dict[str, ToolSpec]
    handlers: dict[str, ToolHandler]

    def describe_for_prompt(self) -> str:
        lines = []
        for name in sorted(self.specs):
            spec = self.specs[name]
            lines.append(f"- {spec.name}: {spec.description}")
            lines.append(f"  input_schema: {spec.input_schema}")
        return "\n".join(lines)

    def execute(
        self, task: PublicTask, action: str, action_input: dict[str, Any]
    ) -> ToolExecutionResult:
        if action not in self.handlers:
            raise KeyError(f"Unknown tool: {action}")
        return self.handlers[action](task, action_input)


def create_default_tool_registry(
    auditor_model: ModelAdapter | None = None,
    question_provider=None,
    *,
    context_dir=None,
) -> ToolRegistry:
    # `answer` is exposed only for doc-only tasks (= no DB/CSV/JSON in context).
    # SQL-backed tasks must go through answer_from_sql so column auditor runs.
    doc_only = False
    if context_dir is not None:
        from pathlib import Path
        ctx = Path(context_dir)
        try:
            doc_only = not any(
                p.is_file() and p.suffix.lower() in (".db", ".sqlite", ".csv", ".json")
                for p in ctx.rglob("*")
            )
        except Exception:
            doc_only = False

    specs: dict[str, ToolSpec] = {}
    if doc_only:
        specs["answer"] = ToolSpec(
            name="answer",
            description=(
                "Submit the final answer table directly. Terminating action. "
                "Use this for doc-only tasks (= no SQL data sources). For SQL-backed "
                "tasks, use answer_from_sql instead."
            ),
            input_schema={
                "columns": ["column_name"],
                "rows": [["value_1"]],
            },
        )
    specs.update({
        "answer_from_sql": ToolSpec(
            name="answer_from_sql",
            description=(
                "Propose a SQL-based answer. Runs SQL, then auto-runs column + filter audits. "
                "If audits PASS → terminal commit. If filter audit flags issues, returns "
                "feedback (= NON-terminal) so you can refine the SQL or call confirm_answer "
                "to override. Up to 2 refines allowed before forced commit."
            ),
            input_schema={"sql": "SELECT ... FROM ..."},
        ),
        "confirm_answer": ToolSpec(
            name="confirm_answer",
            description=(
                "Commit the most recent answer_from_sql proposal as final, overriding any "
                "filter audit objections. Use only when you've reviewed the audit and decided "
                "the audit is wrong (= rare). Terminating action."
            ),
            input_schema={},
        ),
        "describe_data": ToolSpec(
            name="describe_data",
            description=(
                "Show all data sources in this task — every CSV, JSON, and sqlite-DB is exposed "
                "as a view in DuckDB. Returns view names, source file, and columns. "
                "Call this once at the start to know what is available."
            ),
            input_schema={},
        ),
        "execute_sql": ToolSpec(
            name="execute_sql",
            description=(
                "Run a read-only SQL query (SELECT/WITH/PRAGMA/DESCRIBE/SHOW/EXPLAIN) against "
                "the unified DuckDB connection. Use this for inspection and intermediate queries. "
                "Returns up to `limit` rows."
            ),
            input_schema={"sql": "SELECT ...", "limit": 200},
        ),
        "list_context": ToolSpec(
            name="list_context",
            description="List files and directories under the task context dir.",
            input_schema={"max_depth": 4},
        ),
        "read_csv": ToolSpec(
            name="read_csv",
            description="Preview a CSV file (= inspection only; use SQL for analysis).",
            input_schema={"path": "relative/path/to/file.csv", "max_rows": 20},
        ),
        "read_doc": ToolSpec(
            name="read_doc",
            description=(
                "Read a text-like document, Claude Code Read-tool style "
                "(line-based). Modes:\n"
                "  - search='<term>': returns paragraphs containing the term "
                "(case-insensitive), joined by `---`. Use for fact extraction.\n"
                "  - offset=<line_number>: 1-based START LINE. Returns up to "
                "n_lines lines (default 100, capped by max_chars) in `cat -n` "
                "format (= '<line>\\t<content>'). Pair with grep: grep returns "
                "'path:N:content' → read_doc(path=X, offset=N) to read around "
                "that line. Use next_offset to continue paging.\n"
                "Default (no search, no offset): returns lines 1-100."
            ),
            input_schema={
                "path": "relative/path/to/file.md",
                "max_chars": 4000,
                "search": "optional keyword (= grep paragraphs)",
                "offset": 0,
            },
        ),
        "grep": ToolSpec(
            name="grep",
            description=(
                "Regex search across text files in the task context (= md, txt, "
                "csv, json). Inspired by Claude Code's Grep. Three output modes:\n"
                "  - 'content' (default): line-by-line `path:line:text` with "
                "    optional ±N context_lines.\n"
                "  - 'files_with_matches': just the list of paths.\n"
                "  - 'count': match-count per file.\n"
                "Pattern is Python regex. Case-insensitive by default. Optional "
                "`path` (= one file) or `glob` (= e.g. 'doc/*.md') filter."
            ),
            input_schema={
                "pattern": "regex (Python re syntax)",
                "path": "optional, e.g. 'doc/major.md'",
                "glob": "optional, e.g. 'doc/*.md'",
                "output_mode": "'content' | 'files_with_matches' | 'count'",
                "context_lines": 0,
                "case_insensitive": True,
                "max_results": 50,
                "max_chars": 8000,
            },
        ),
        "read_json": ToolSpec(
            name="read_json",
            description="Preview a JSON file (= inspection only; use SQL on the view for analysis).",
            input_schema={"path": "relative/path/to/file.json", "max_chars": 4000},
        ),
    })
    if auditor_model:
        ans_handler = _make_answer_from_sql_with_audit(auditor_model, question_provider)
        confirm_handler = _make_confirm_answer(ans_handler)
    else:
        ans_handler = _answer_from_sql
        confirm_handler = _answer_from_sql  # no-op fallback when auditor disabled

    handlers: dict[str, ToolHandler] = {
        "answer_from_sql": ans_handler,
        "confirm_answer": confirm_handler,
        "describe_data": _describe_data,
        "execute_sql": _execute_sql,
        "grep": _grep,
        "list_context": _list_context,
        "read_csv": _read_csv,
        "read_doc": _read_doc,
        "read_json": _read_json,
    }
    if doc_only:
        handlers["answer"] = _answer
    return ToolRegistry(specs=specs, handlers=handlers)
