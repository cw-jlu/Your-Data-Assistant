"""SQL-only tool registry + domain-RAG consultation.

NO Python execution. All computation happens in a single DuckDB
connection that exposes every CSV/JSON/sqlite-DB file in the task
context as views/tables. In addition, `consult_domain_db` provides
keyword-retrieval over a per-DB schema reference shipped with the
submission image.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from kobushi_core.benchmark.schema import AnswerTable, PublicTask
from kobushi_core.model import ModelAdapter
from experiments.exp_127_python_doc_tool.tools.filesystem import (
    grep_context,
    list_context_tree,
    read_csv_preview,
    read_doc_preview,
    read_json_preview,
)
from experiments.exp_127_python_doc_tool.tools.duckdb_unified import (
    describe_catalog,
    execute_sql,
    get_catalog,
)


# Domain-DB root: keyword-retrievable schema reference shipped with the image.
# Override with KDD_DOMAIN_DB_DIR for local dev / smoke testing.
_DOMAIN_DB_DIR = Path(os.environ.get(
    "KDD_DOMAIN_DB_DIR",
    str(Path(__file__).resolve().parents[4] / "data/external/domain_db"),
))


def _normalize_db_name(raw: str) -> str:
    """Convert display name → BIRD db_id (= lowercase_with_underscores)."""
    s = raw.strip().rstrip(".:").strip()
    s = s.lower()
    # Drop a trailing "database" word if present.
    s = re.sub(r"\s+database\s*$", "", s)
    # Collapse whitespace runs to a single underscore.
    s = re.sub(r"\s+", "_", s)
    return s


def _detect_db_id_from_knowledge(context_dir: Path) -> str | None:
    """Parse `knowledge.md` header for the DB name.

    Handles two header styles:
      - "# Enterprise Data Governance Knowledge Guide for Database: Debit Card Specializing"
      - "# Knowledge Guide for Thrombosis Prediction Database"
    Returns the BIRD-style db_id (lowercase_with_underscores), or None.
    """
    km = context_dir / "knowledge.md"
    if not km.exists():
        return None
    try:
        for i, line in enumerate(km.open(errors="replace")):
            if i > 10:
                break
            # Style 1: "Database: <Name>"
            m = re.search(r"Database\s*:\s*([^\n#]+)", line)
            if m:
                return _normalize_db_name(m.group(1))
            # Style 2: "Knowledge Guide for <Name> Database"
            m = re.search(r"Knowledge\s+Guide\s+for\s+([^\n#]+?)\s+Database", line, re.IGNORECASE)
            if m:
                return _normalize_db_name(m.group(1))
            if line.startswith("##"):
                break
    except Exception:
        pass
    return None


def _consult_domain_db(task: PublicTask, action_input: dict[str, Any]) -> "ToolExecutionResult":
    """Keyword retrieval over the task's per-DB schema reference.

    Returns top-k field entries whose text contains the most of the query
    tokens (case-insensitive). Returns an empty result if the task's DB is
    not in the shipped reference set.
    """
    query = str(action_input.get("query", "")).strip()
    top_k = int(action_input.get("top_k", 5))
    if not query:
        return ToolExecutionResult(ok=False, content="query is required")

    db_id = _detect_db_id_from_knowledge(task.context_dir)
    if not db_id:
        return ToolExecutionResult(
            ok=True,
            content="(no database identifier found in knowledge.md — domain reference unavailable)",
        )
    ref_path = _DOMAIN_DB_DIR / f"{db_id}.md"
    if not ref_path.exists():
        return ToolExecutionResult(
            ok=True,
            content=f"(no domain reference shipped for db_id={db_id})",
        )

    text = ref_path.read_text(errors="replace")
    # Split into per-field entries by the "### Field:" marker.
    entries = re.split(r"\n(?=### Field:)", text)
    qtokens = [t for t in re.findall(r"\w+", query.lower()) if len(t) >= 2]
    if not qtokens:
        return ToolExecutionResult(ok=False, content="query has no usable tokens")

    scored: list[tuple[int, str]] = []
    for e in entries:
        low = e.lower()
        score = sum(low.count(t) for t in qtokens)
        if score > 0:
            scored.append((score, e.strip()))
    scored.sort(key=lambda x: -x[0])
    top = scored[:top_k]
    if not top:
        return ToolExecutionResult(
            ok=True,
            content=f"(no entries matched query tokens {qtokens} in {db_id})",
        )
    header = f"## Domain Reference: {db_id} (top {len(top)} of {len(entries)} fields)"
    return ToolExecutionResult(
        ok=True,
        content=header + "\n\n" + "\n\n".join(e for _, e in top),
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


_EXTRACT_SUBAGENT_SYSTEM = (
    "You are a structured-data extractor. Your only job is to read a "
    "free-form document and emit a JSON array of records that match the "
    "user's requested schema. Output ONLY the JSON array — no markdown "
    "fences, no commentary, no preamble.\n\n"
    "EVERY record must contain EVERY field listed in the schema. Scan the "
    "ENTIRE document for each field — values for the same entity are often "
    "scattered across distant paragraphs or sections (e.g., id in one "
    "section, measurements in another). Pair them by the identifier.\n\n"
    "An entity's identifier may be phrased in multiple ways throughout the "
    "document (e.g., 'ID X', 'reference code X', 'identifier X', "
    "'cataloged under X', 'registered under X'). Treat ANY consistent "
    "unique identifier referring to the same entity as that entity's id.\n\n"
    "Preserve numeric values LITERALLY: if a field is recorded as 0.0, "
    "0, or another placeholder-looking low number, emit that exact value. "
    "Do NOT replace numeric zeros or recorded placeholders with null. "
    "Use null ONLY when the document does not mention the field at all "
    "for that entity.\n\n"
    "Before emitting null for a field, do one more pass for varied "
    "phrasings (e.g., 'height is 175 cm' / '175 centimeters tall' / "
    "'documented as 175' / '175 cm'). Be exhaustive: if the document "
    "describes N entities, emit N records."
)


def _make_extract_structured(model) -> "ToolHandler":
    """Sub-agent extraction tool: LLM reads a doc and returns JSON records.

    Use case: prose-heavy docs where regex over read_doc / grep cannot
    reliably structure data scattered across paragraphs. The main agent
    calls this once with the target schema; the sub-agent specializes in
    extraction and returns a clean record list the main agent can aggregate.
    """
    def handler(task: PublicTask, action_input: dict[str, Any]) -> "ToolExecutionResult":
        import json as _json
        from kobushi_core.model import ModelMessage

        if model is None:
            return ToolExecutionResult(ok=False, content="sub-agent model is unavailable")
        doc_path = str(action_input.get("doc_path", "")).strip()
        schema = str(action_input.get("schema", "")).strip()
        instruction = str(action_input.get("instruction", "")).strip()
        # Default close to Qwen3.5 64K-token context budget (= ~240K chars at
        # 4 chars/token). Leaves room for system + schema + JSON response.
        max_chars = int(action_input.get("max_chars", 240000))
        if not doc_path or not schema:
            return ToolExecutionResult(
                ok=False, content="`doc_path` and `schema` are both required",
            )
        path = (task.context_dir / doc_path).resolve()
        try:
            path.relative_to(task.context_dir.resolve())
        except ValueError:
            return ToolExecutionResult(ok=False, content="doc_path must stay inside context_dir")
        if not path.exists():
            return ToolExecutionResult(ok=False, content=f"file not found: {doc_path}")
        try:
            text = path.read_text(errors="replace")
        except Exception as exc:
            return ToolExecutionResult(ok=False, content=f"read failed: {exc}")
        if len(text) > max_chars:
            text = text[:max_chars] + "\n…[truncated]"

        user = (
            f"## Schema (= JSON fields per record)\n{schema}\n\n"
            + (f"## Notes\n{instruction}\n\n" if instruction else "")
            + f"## Document ({doc_path}, {len(text)} chars)\n{text}\n\n"
            "Return ONLY a JSON array of records. No markdown, no comments."
        )
        try:
            raw = model.complete(
                [
                    ModelMessage(role="system", content=_EXTRACT_SUBAGENT_SYSTEM),
                    ModelMessage(role="user", content=user),
                ],
                enable_thinking=False,
            )
        except Exception as exc:
            return ToolExecutionResult(ok=False, content=f"sub-agent LLM error: {exc!r}")

        # Permissive parse: take the first JSON array found.
        m = re.search(r"\[\s*(?:\{|\])", raw)
        if m is None:
            return ToolExecutionResult(
                ok=False, content=f"no JSON array in sub-agent response: {raw[:300]}",
            )
        # Try to parse the substring from m.start() to the last "]".
        end = raw.rfind("]")
        candidate = raw[m.start(): end + 1] if end > m.start() else raw[m.start():]
        try:
            records = _json.loads(candidate)
        except Exception as exc:
            return ToolExecutionResult(
                ok=False,
                content=f"JSON parse failed ({exc}); raw head: {raw[:400]}",
            )
        if not isinstance(records, list):
            return ToolExecutionResult(ok=False, content="sub-agent did not return a list")
        return ToolExecutionResult(
            ok=True,
            content={"n": len(records), "records": records[:200]},
        )

    return handler


def _execute_python(task: PublicTask, action_input: dict[str, Any]) -> "ToolExecutionResult":
    """Run a short Python script in an isolated subprocess.

    The script runs in a fresh temporary directory (= NOT context_dir) to
    keep any temp files / outputs out of the task context. Read access to
    context files is exposed via TASK_CONTEXT_DIR env var, which the agent
    must use with absolute paths (or os.path.join). stdout is the agent's
    answer; stderr is surfaced for debugging.
    """
    import subprocess
    import tempfile
    import os

    code = str(action_input.get("code", "")).strip()
    if not code:
        return ToolExecutionResult(ok=False, content="`code` is required")

    timeout_sec = min(max(int(action_input.get("timeout_sec", 30)), 5), 60)

    with tempfile.TemporaryDirectory() as workdir:
        script_path = Path(workdir) / "script.py"
        script_path.write_text(code)
        env = {**os.environ, "TASK_CONTEXT_DIR": str(task.context_dir.resolve())}
        try:
            result = subprocess.run(
                ["python3", str(script_path)],
                capture_output=True, text=True, timeout=timeout_sec,
                cwd=workdir,
                env=env,
            )
            return ToolExecutionResult(
                ok=(result.returncode == 0),
                content={
                    "stdout": (result.stdout or "")[:8000],
                    "stderr": (result.stderr or "")[:2000],
                    "returncode": result.returncode,
                },
            )
        except subprocess.TimeoutExpired:
            return ToolExecutionResult(
                ok=False,
                content=f"Python execution exceeded {timeout_sec}s timeout",
            )
        except Exception as exc:
            return ToolExecutionResult(
                ok=False, content=f"Python subprocess failed: {exc!r}",
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
    from experiments.exp_127_python_doc_tool.column_auditor import audit_columns, apply_audit

    state = {"refines": 0, "pending_answer": None}

    def handler(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
        ai = _coerce_sql_input(action_input)
        sql = str(ai["sql"])
        try:
            result = execute_sql(task.context_dir, sql, limit=10000)
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
    difficulty: str | None = None,
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
        "consult_domain_db": ToolSpec(
            name="consult_domain_db",
            description=(
                "Look up field-level schema notes (descriptions, value ranges, "
                "code mappings, format hints) from the task's domain reference. "
                "Use this when a question term needs grounding — e.g., 'normal "
                "range of X', 'what does code Y mean', 'format of column Z'. "
                "Returns the top-k matching field entries by keyword overlap. "
                "Returns empty if the task's DB has no shipped reference."
            ),
            input_schema={
                "query": "keywords to search (= column names, value terms)",
                "top_k": 5,
            },
        ),
        "read_json": ToolSpec(
            name="read_json",
            description="Preview a JSON file (= inspection only; use SQL on the view for analysis).",
            input_schema={"path": "relative/path/to/file.json", "max_chars": 4000},
        ),
        "execute_python": ToolSpec(
            name="execute_python",
            description=(
                "Run Python for prose extraction + aggregation over `.md`/`.txt` "
                "docs when SQL can't structure them. cwd is the task context_dir. "
                "print() the final aggregated value; the script's role is to "
                "produce the answer, not to export data."
            ),
            input_schema={
                "code": "Python source; print() the final answer",
                "timeout_sec": 30,
            },
        ),
        "extract_structured": ToolSpec(
            name="extract_structured",
            description=(
                "Ask a sub-agent to read a doc and return all matching entities "
                "as JSON records. Use when fields are scattered across paragraphs "
                "in a prose narrative and regex/grep cannot reliably pair them. "
                "The sub-agent specializes in extraction; your job is to use the "
                "returned records to compute the answer."
            ),
            input_schema={
                "doc_path": "relative path under context, e.g. 'doc/superhero.md'",
                "schema": "fields per record, e.g. 'id (int), height_cm (float), publisher_id (int)'",
                "instruction": "optional notes (e.g., 'extract every superhero, including those without all fields')",
                "max_chars": 240000,
            },
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
        "consult_domain_db": _consult_domain_db,
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

    # execute_python is gated to hard/extreme tasks: easy/medium tasks have
    # no doc/ files (= empirically verified across the public 50 set) so the
    # tool's prose-focused use case never applies; keeping it out of their
    # action space avoids unnecessary noise. doc-only extreme tasks always
    # get it regardless of difficulty label.
    python_eligible = doc_only or difficulty in ("hard", "extreme")
    if python_eligible:
        # PoC: remove execute_python to force the agent to rely on
        # extract_structured (= sub-agent) for prose extraction.
        specs.pop("execute_python", None)
        if auditor_model is not None:
            handlers["extract_structured"] = _make_extract_structured(auditor_model)
        else:
            specs.pop("extract_structured", None)
    else:
        specs.pop("execute_python", None)
        specs.pop("extract_structured", None)

    return ToolRegistry(specs=specs, handlers=handlers)
