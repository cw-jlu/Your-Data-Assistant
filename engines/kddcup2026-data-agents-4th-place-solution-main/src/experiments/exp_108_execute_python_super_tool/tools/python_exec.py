from __future__ import annotations

import contextlib
import io
import multiprocessing
import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any


@contextlib.contextmanager
def _capture_process_streams(stdout_path: Path, stderr_path: Path):
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    saved_stdout_fd = os.dup(1)
    saved_stderr_fd = os.dup(2)

    with stdout_path.open("w+b") as stdout_file, stderr_path.open("w+b") as stderr_file:
        try:
            if original_stdout is not None:
                original_stdout.flush()
            if original_stderr is not None:
                original_stderr.flush()

            os.dup2(stdout_file.fileno(), 1)
            os.dup2(stderr_file.fileno(), 2)

            stdout_encoding = getattr(original_stdout, "encoding", None) or "utf-8"
            stderr_encoding = getattr(original_stderr, "encoding", None) or "utf-8"

            sys.stdout = io.TextIOWrapper(
                os.fdopen(os.dup(1), "wb"),
                encoding=stdout_encoding,
                errors="replace",
                line_buffering=True,
                write_through=True,
            )
            sys.stderr = io.TextIOWrapper(
                os.fdopen(os.dup(2), "wb"),
                encoding=stderr_encoding,
                errors="replace",
                line_buffering=True,
                write_through=True,
            )
            yield
        finally:
            if sys.stdout is not None:
                sys.stdout.flush()
            if sys.stderr is not None:
                sys.stderr.flush()

            if sys.stdout is not original_stdout:
                sys.stdout.close()
            if sys.stderr is not original_stderr:
                sys.stderr.close()

            sys.stdout = original_stdout
            sys.stderr = original_stderr
            os.dup2(saved_stdout_fd, 1)
            os.dup2(saved_stderr_fd, 2)
            os.close(saved_stdout_fd)
            os.close(saved_stderr_fd)


def _read_captured_stream(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _run_python_code(
    context_root: str,
    code: str,
    stdout_path: str,
    stderr_path: str,
    queue: multiprocessing.Queue[Any],
) -> None:
    # Pre-bind common data libraries so agent code can use them without imports.
    import json as _json
    import sqlite3 as _sqlite3
    try:
        import pandas as _pd
    except ImportError:
        _pd = None  # type: ignore[assignment]
    try:
        import numpy as _np
    except ImportError:
        _np = None  # type: ignore[assignment]
    try:
        import duckdb as _duckdb
    except ImportError:
        _duckdb = None  # type: ignore[assignment]

    namespace: dict[str, Any] = {
        "__builtins__": __builtins__,
        "__name__": "__main__",
        "context_root": context_root,
        "Path": Path,
        "ctx": Path(context_root),
        "pd": _pd,
        "np": _np,
        "duckdb": _duckdb,
        "sqlite3": _sqlite3,
        "json": _json,
    }
    resolved_stdout_path = Path(stdout_path)
    resolved_stderr_path = Path(stderr_path)

    try:
        os.chdir(context_root)
        with _capture_process_streams(resolved_stdout_path, resolved_stderr_path):
            exec(code, namespace, namespace)
        queue.put({"success": True})
    except BaseException as exc:  # noqa: BLE001
        queue.put(
            {
                "success": False,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )


def _extract_answer_from_namespace(namespace: dict[str, Any]) -> dict[str, Any] | None:
    """Pull `answer_df` (DataFrame) or `answer_table` (dict) from namespace and
    convert to {"columns": [...], "rows": [[str, ...], ...]}. Return None if not found."""
    if "answer_df" in namespace:
        df = namespace["answer_df"]
        try:
            columns = [str(c) for c in df.columns.tolist()]
            rows = [[("" if v is None else str(v)) for v in row] for row in df.values.tolist()]
            return {"columns": columns, "rows": rows}
        except Exception as exc:  # noqa: BLE001
            return {"_extract_error": f"answer_df is not a DataFrame-like object: {exc!r}"}
    if "answer_table" in namespace:
        table = namespace["answer_table"]
        if isinstance(table, dict) and "columns" in table and "rows" in table:
            try:
                columns = [str(c) for c in table["columns"]]
                rows = [[("" if v is None else str(v)) for v in row] for row in table["rows"]]
                return {"columns": columns, "rows": rows}
            except Exception as exc:  # noqa: BLE001
                return {"_extract_error": f"answer_table malformed: {exc!r}"}
        return {"_extract_error": "answer_table must be a dict with keys 'columns' and 'rows'"}
    return None


def _run_python_for_answer(
    context_root: str,
    code: str,
    stdout_path: str,
    stderr_path: str,
    queue: multiprocessing.Queue[Any],
) -> None:
    # Pre-bind common data libraries so agent code can use them without imports.
    import json as _json
    import sqlite3 as _sqlite3
    try:
        import pandas as _pd
    except ImportError:
        _pd = None  # type: ignore[assignment]
    try:
        import numpy as _np
    except ImportError:
        _np = None  # type: ignore[assignment]
    try:
        import duckdb as _duckdb
    except ImportError:
        _duckdb = None  # type: ignore[assignment]

    namespace: dict[str, Any] = {
        "__builtins__": __builtins__,
        "__name__": "__main__",
        "context_root": context_root,
        "Path": Path,
        "ctx": Path(context_root),
        "pd": _pd,
        "np": _np,
        "duckdb": _duckdb,
        "sqlite3": _sqlite3,
        "json": _json,
    }
    resolved_stdout_path = Path(stdout_path)
    resolved_stderr_path = Path(stderr_path)

    try:
        os.chdir(context_root)
        with _capture_process_streams(resolved_stdout_path, resolved_stderr_path):
            exec(code, namespace, namespace)
        extracted = _extract_answer_from_namespace(namespace)
        if extracted is None:
            queue.put(
                {
                    "success": False,
                    "error": (
                        "No answer variable found. The code must set "
                        "`answer_df` (a pandas DataFrame) or `answer_table` "
                        "(a dict with keys 'columns' and 'rows')."
                    ),
                }
            )
        elif "_extract_error" in extracted:
            queue.put({"success": False, "error": extracted["_extract_error"]})
        else:
            queue.put({"success": True, "answer": extracted})
    except BaseException as exc:  # noqa: BLE001
        queue.put(
            {
                "success": False,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )


def _run_with_subprocess(
    target,
    context_root: Path,
    code: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    resolved_context_root = context_root.resolve()
    with tempfile.TemporaryDirectory() as temp_dir:
        stdout_path = Path(temp_dir) / "stdout.txt"
        stderr_path = Path(temp_dir) / "stderr.txt"
        stdout_path.write_text("")
        stderr_path.write_text("")

        queue: multiprocessing.Queue[Any] = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=target,
            args=(
                resolved_context_root.as_posix(),
                code,
                stdout_path.as_posix(),
                stderr_path.as_posix(),
                queue,
            ),
        )
        process.start()
        process.join(timeout_seconds)

        if process.is_alive():
            process.terminate()
            process.join()
            return {
                "success": False,
                "output": _read_captured_stream(stdout_path),
                "stderr": _read_captured_stream(stderr_path),
                "error": f"Python execution timed out after {timeout_seconds} seconds.",
            }

        if queue.empty():
            return {
                "success": False,
                "output": _read_captured_stream(stdout_path),
                "stderr": _read_captured_stream(stderr_path),
                "error": "Python execution exited without returning a result.",
            }

        result = queue.get()
        result["output"] = _read_captured_stream(stdout_path)
        result["stderr"] = _read_captured_stream(stderr_path)
        return result


def execute_python_code(
    context_root: Path, code: str, *, timeout_seconds: int = 30
) -> dict[str, Any]:
    return _run_with_subprocess(_run_python_code, context_root, code, timeout_seconds)


def execute_python_for_answer(
    context_root: Path, code: str, *, timeout_seconds: int = 30
) -> dict[str, Any]:
    return _run_with_subprocess(_run_python_for_answer, context_root, code, timeout_seconds)
