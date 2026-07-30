from __future__ import annotations

import contextlib
import io
import multiprocessing
import os
import shutil
import sys
import tempfile
import time
import traceback
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty
from typing import Any

from data_agent_baseline.benchmark.schema import ContextView
from data_agent_baseline.tools.probe_engine import execute_probe_query


@dataclass(slots=True)
class TaskContextWorkspace:
    source_root: Path
    context_view: ContextView | None = None
    _temporary_dir: tempfile.TemporaryDirectory[str] | None = field(default=None, init=False, repr=False)
    _workspace_root: Path | None = field(default=None, init=False)

    # 延迟创建任务级临时工作区，只有真的需要执行 Python 时才复制 context。
    def materialize(self) -> Path:
        if self._workspace_root is not None:
            return self._workspace_root

        temporary_dir = tempfile.TemporaryDirectory(prefix="dabench-workspace-")
        workspace_root = Path(temporary_dir.name) / "context"
        if self.context_view is None:
            shutil.copytree(self.source_root.resolve(), workspace_root)
        else:
            workspace_root.mkdir(parents=True, exist_ok=True)
            for asset in self.context_view.assets:
                target_path = workspace_root / asset.visible_path
                target_path.parent.mkdir(parents=True, exist_ok=True)
                if asset.generated:
                    shutil.copy2(asset.physical_path, target_path)
                    continue
                try:
                    os.symlink(asset.physical_path.resolve(), target_path)
                except OSError:
                    shutil.copy2(asset.physical_path, target_path)
        self._temporary_dir = temporary_dir
        self._workspace_root = workspace_root
        return workspace_root

    @property
    def path(self) -> Path | None:
        return self._workspace_root

    def cleanup(self) -> None:
        """Best-effort cleanup that must not invalidate a completed task.

        Windows can retain a handle to a copied SQLite file briefly after the
        final query has returned.  Retrying covers that transient case; a
        persistent lock is reported as a warning and left for the OS cleanup
        path instead of turning a successful task into a runner failure.
        """
        temporary_dir = self._temporary_dir
        if temporary_dir is None:
            return

        cleanup_error: OSError | None = None
        for attempt in range(3):
            try:
                temporary_dir.cleanup()
            except OSError as exc:
                cleanup_error = exc
                if attempt < 2:
                    time.sleep(0.1 * (attempt + 1))
            else:
                cleanup_error = None
                break

        self._temporary_dir = None
        self._workspace_root = None
        if cleanup_error is not None:
            warnings.warn(
                f"Could not remove temporary task workspace {temporary_dir.name!r}: {cleanup_error}",
                ResourceWarning,
                stacklevel=2,
            )


# 临时重定向子进程的 stdout / stderr，到文件中进行捕获。
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

            # 强制使用 UTF-8 编码写入，与 _read_captured_stream 的
            # utf-8 读取保持一致，避免 Windows GBK 环境下的中文乱码。
            stdout_encoding = "utf-8"
            stderr_encoding = "utf-8"

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
            # 退出时恢复原始标准流，避免影响宿主进程。
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


# 读取捕获文件中的内容，统一按 utf-8 容错解码。
def _read_captured_stream(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


# 真正执行 Python 代码的子进程入口。
def _run_python_code(
    context_root: str,
    code: str,
    catalog: dict[str, Any] | None,
    stdout_path: str,
    stderr_path: str,
    queue: multiprocessing.Queue[Any],
) -> None:
    def query(sql: str) -> dict[str, Any]:
        if catalog is None:
            raise RuntimeError("Logical table query helper is unavailable: catalog was not provided.")
        result = execute_probe_query(
            context_dir=Path(context_root),
            catalog=catalog,
            queries=[sql],
            limit=None,
        )
        if not result.get("ok"):
            error = result.get("results", [{}])[0].get("error", "query failed")
            raise RuntimeError(str(error))
        query_result = result.get("results", [{}])[0]
        return {
            "columns": query_result.get("columns", []),
            "rows": query_result.get("rows", []),
        }

    def query_rows(sql: str) -> list[list[Any]]:
        return list(query(sql)["rows"])

    # 向执行环境暴露最小上下文：内置对象、context_root 和 Path。
    namespace: dict[str, Any] = {
        "__builtins__": __builtins__,
        "__name__": "__main__",
        "context_root": context_root,
        "Path": Path,
        "query": query,
        "query_rows": query_rows,
    }
    resolved_stdout_path = Path(stdout_path)
    resolved_stderr_path = Path(stderr_path)

    try:
        # 将工作目录切到任务 context，方便模型直接使用相对路径。
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


# 公开的 Python 执行入口：在独立进程中运行代码，并收集输出、错误和超时信息。
def execute_python_code(
    context_root: Path,
    code: str,
    *,
    timeout_seconds: int = 30,
    catalog: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_context_root = context_root.resolve()
    with tempfile.TemporaryDirectory() as temp_dir:
        stdout_path = Path(temp_dir) / "stdout.txt"
        stderr_path = Path(temp_dir) / "stderr.txt"
        stdout_path.write_text("")
        stderr_path.write_text("")

        ctx = multiprocessing.get_context("spawn")
        queue: multiprocessing.Queue[Any] = ctx.Queue()
        process = ctx.Process(
            target=_run_python_code,
            args=(
                resolved_context_root.as_posix(),
                code,
                catalog,
                stdout_path.as_posix(),
                stderr_path.as_posix(),
                queue,
            ),
        )
        process.start()
        try:
            result = queue.get(timeout=timeout_seconds)
        except Empty:
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
                if process.is_alive():
                    process.kill()
                    process.join()
                return {
                    "success": False,
                    "output": _read_captured_stream(stdout_path),
                    "stderr": _read_captured_stream(stderr_path),
                    "error": f"Python execution timed out after {timeout_seconds} seconds.",
                }

            process.join(timeout=1.0)
            return {
                "success": False,
                "output": _read_captured_stream(stdout_path),
                "stderr": _read_captured_stream(stderr_path),
                "error": "Python execution exited without returning a result.",
            }
        finally:
            queue.close()
            queue.join_thread()

        process.join(timeout=1.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=1.0)
            if process.is_alive():
                process.kill()
                process.join()
            return {
                "success": False,
                "output": _read_captured_stream(stdout_path),
                "stderr": _read_captured_stream(stderr_path),
                "error": "Python execution returned a result but did not exit cleanly.",
            }

        # 无论成功失败，都把 stdout / stderr 附回结果，方便 trace 复盘。
        result["output"] = _read_captured_stream(stdout_path)
        result["stderr"] = _read_captured_stream(stderr_path)
        return result
