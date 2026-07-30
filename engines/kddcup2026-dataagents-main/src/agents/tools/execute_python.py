"""execute_python tool: arbitrary Python execution in a subprocess."""

from __future__ import annotations

import contextlib
import io
import multiprocessing
import os
import sys
import tempfile
import traceback
from collections.abc import Callable
from pathlib import Path
from queue import Empty
from typing import Annotated, Any

from pydantic import Field

from agents.benchmark.schema import PublicTask
from agents.tools.contracts import EXECUTE_PYTHON_TIMEOUT_SECONDS
from agents.tools.decorator import function_tool
from agents.tools.registry import ToolExecutionResult

# Use spawn so host process threads/locks are not inherited by agent code.
_MP_SPAWN = multiprocessing.get_context("spawn")

_PYTHON_EXEC_STREAM_CAP_BYTES = 64 * 1024
_PYTHON_EXEC_STREAM_HEAD_BYTES = 32 * 1024
_PYTHON_EXEC_STREAM_TAIL_BYTES = 32 * 1024


@contextlib.contextmanager
def _capture_process_streams(stdout_path: Path, stderr_path: Path):
    """Redirect Python and fd-level stdout/stderr to files inside the child process."""
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

            sys.stdout = io.TextIOWrapper(
                os.fdopen(os.dup(1), "wb"),
                encoding="utf-8",
                errors="replace",
                line_buffering=True,
                write_through=True,
            )
            sys.stderr = io.TextIOWrapper(
                os.fdopen(os.dup(2), "wb"),
                encoding="utf-8",
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

            if sys.stdout is not None and sys.stdout is not original_stdout:
                sys.stdout.close()
            if sys.stderr is not None and sys.stderr is not original_stderr:
                sys.stderr.close()

            sys.stdout = original_stdout
            sys.stderr = original_stderr
            os.dup2(saved_stdout_fd, 1)
            os.dup2(saved_stderr_fd, 2)
            os.close(saved_stdout_fd)
            os.close(saved_stderr_fd)


def _read_captured_stream(path: Path) -> str:
    """Read captured stdout/stderr, keeping head and tail when output is too large."""
    size = path.stat().st_size
    if size <= _PYTHON_EXEC_STREAM_CAP_BYTES:
        return path.read_text(encoding="utf-8", errors="replace")

    elided = size - (_PYTHON_EXEC_STREAM_HEAD_BYTES + _PYTHON_EXEC_STREAM_TAIL_BYTES)
    with path.open("rb") as fp:
        head_bytes = fp.read(_PYTHON_EXEC_STREAM_HEAD_BYTES)
        fp.seek(size - _PYTHON_EXEC_STREAM_TAIL_BYTES)
        tail_bytes = fp.read(_PYTHON_EXEC_STREAM_TAIL_BYTES)
    head = head_bytes.decode("utf-8", errors="replace")
    tail = tail_bytes.decode("utf-8", errors="replace")
    return f"{head}\n... [TRUNCATED: {elided} bytes elided] ...\n{tail}"


def _run_python_code(
    context_root: str,
    code: str,
    stdout_path: str,
    stderr_path: str,
    answer_dir: str,
    queue: multiprocessing.Queue[Any],
) -> None:
    """Child-process entry point for executing model-authored Python code."""
    os.environ["DABENCH_ANSWER_DIR"] = answer_dir
    import csv
    import json
    import math
    import re
    import sqlite3

    import numpy as np
    import pandas as pd

    namespace: dict[str, Any] = {
        "__builtins__": __builtins__,
        "__name__": "__main__",
        "context_root": context_root,
        "Path": Path,
        "os": os,
        "json": json,
        "re": re,
        "csv": csv,
        "math": math,
        "sqlite3": sqlite3,
        "pd": pd,
        "np": np,
    }

    # Monkey-patch pandas to use stable sort by default so tied rows
    # preserve their original (insertion-order / primary-key) position.
    _orig_df_sort: Callable[..., Any] = pd.DataFrame.sort_values
    _orig_sr_sort: Callable[..., Any] = pd.Series.sort_values

    def _stable_df_sort(self: Any, *args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("kind", "mergesort")
        return _orig_df_sort(self, *args, **kwargs)

    def _stable_sr_sort(self: Any, *args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("kind", "mergesort")
        return _orig_sr_sort(self, *args, **kwargs)

    # Dynamic patching avoids pyright trying to match pandas' overload set.
    setattr(pd.DataFrame, "sort_values", _stable_df_sort)  # noqa: B010
    setattr(pd.Series, "sort_values", _stable_sr_sort)  # noqa: B010

    resolved_stdout_path = Path(stdout_path)
    resolved_stderr_path = Path(stderr_path)

    try:
        os.chdir(context_root)
        with _capture_process_streams(resolved_stdout_path, resolved_stderr_path):
            exec(code, namespace, namespace)
        queue.put({"success": True})
    except BaseException as exc:
        queue.put(
            {
                "success": False,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )


def execute_python_code(
    context_root: Path,
    code: str,
    *,
    answer_dir: Path,
    timeout_seconds: int = 30,
) -> dict[str, Any]:
    """Execute Python code in a spawned subprocess with timeout and captured streams."""
    resolved_context_root = context_root.resolve()
    answer_dir.mkdir(parents=True, exist_ok=True)
    resolved_answer_dir = answer_dir.resolve()
    with tempfile.TemporaryDirectory() as temp_dir:
        stdout_path = Path(temp_dir) / "stdout.txt"
        stderr_path = Path(temp_dir) / "stderr.txt"
        stdout_path.write_text("")
        stderr_path.write_text("")

        queue: multiprocessing.Queue[Any] = _MP_SPAWN.Queue()
        process = _MP_SPAWN.Process(
            target=_run_python_code,
            args=(
                resolved_context_root.as_posix(),
                code,
                stdout_path.as_posix(),
                stderr_path.as_posix(),
                resolved_answer_dir.as_posix(),
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

        try:
            result = queue.get_nowait()
        except Empty:
            return {
                "success": False,
                "output": _read_captured_stream(stdout_path),
                "stderr": _read_captured_stream(stderr_path),
                "error": "Python execution exited without returning a result.",
            }
        result["output"] = _read_captured_stream(stdout_path)
        result["stderr"] = _read_captured_stream(stderr_path)
        return result


_ANSWER_SCRATCH_ROOT = Path("/tmp/dabench")


def _answer_dir_for(task: PublicTask) -> Path:
    return _ANSWER_SCRATCH_ROOT / task.task_id / "_answer"


@function_tool(
    description=(
        "Execute arbitrary Python in a subprocess. Returns {success, output "
        "(stdout), stderr, error, traceback}. Use when you need computation, "
        "aggregation, cross-file joins, or full-file scans beyond what the "
        "preview tools (preview_file, execute_context_sql) provide. Do NOT "
        "use for simple lookups that a single SQL query or preview tool can "
        "answer — those are faster and cheaper. Working directory: the task "
        f"context dir. Timeout: {EXECUTE_PYTHON_TIMEOUT_SECONDS} s per call. "
        "Default to pandas for tabular work — projections, filters, and "
        "joins are idiomatic and types are preserved automatically. "
        "Available libs: pandas, numpy, pyarrow, polars, openpyxl, sqlalchemy. "
        "The csv module is fine only for trivial reads (a handful of rows). "
        "ALWAYS use os.environ['DABENCH_ANSWER_DIR'] for answer CSV paths "
        "— never hardcode paths. "
        'CSV example: execute_python({"code": "import pandas as pd\\n'
        "df = pd.read_csv('csv/sales.csv')\\n"
        "print(df.groupby('region')['amount'].sum())\"}) "
        'SQLite example: execute_python({"code": "import pandas as pd\\n'
        "df = pd.read_sql_query('SELECT * FROM t LIMIT 5', "
        "'sqlite:///db/sub_db.sqlite')\\nprint(df)\"}) "
        'Answer CSV artifact example: execute_python({"code": "import os, pandas as pd\\n'
        "answer_df = pd.DataFrame({'name': ['Alice'], 'total': [100]})\\n"
        "out_path = os.path.join(os.environ['DABENCH_ANSWER_DIR'], 'answer.csv')\\n"
        "answer_df.to_csv(out_path, index=False)\\n"
        'print(out_path)"})'
    ),
)
def execute_python(
    task: PublicTask,
    code: Annotated[
        str,
        Field(
            description=(
                "Python source to execute with the task context directory as CWD. "
                f"Execution timeout is {EXECUTE_PYTHON_TIMEOUT_SECONDS} seconds."
            ),
        ),
    ],
) -> ToolExecutionResult:
    content = execute_python_code(
        context_root=task.context_dir,
        code=code,
        answer_dir=_answer_dir_for(task),
        timeout_seconds=EXECUTE_PYTHON_TIMEOUT_SECONDS,
    )
    return ToolExecutionResult(ok=bool(content.get("success")), content=content)
