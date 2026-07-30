"""Persistent Python kernel for the ReAct agent.

A worker subprocess executes user code while preserving its namespace
across calls. This lets the agent load a large CSV once with
`df = pd.read_csv(...)` and analyze it across many subsequent
`execute_python` calls without re-loading — critical for tasks whose
context is hundreds of MB and would otherwise saturate the 30s
per-call subprocess budget in tools/python_exec.py.

Tradeoffs vs the ephemeral subprocess executor:
- Hard timeout or fatal exec error kills the worker; the namespace is
  lost on the next `execute()` (a fresh worker is spawned). The result
  carries `namespace_reset=True` so the agent can react.
- Stdout/stderr are captured in-process via contextlib redirects (not
  fd-level), so spawned subprocesses inside user code do NOT have their
  output captured. This is acceptable for typical pandas/sql/numpy
  analysis but worth noting if user code shells out.

Mode is controlled by AgentConfig.python_kernel_mode ("persistent" or
"ephemeral") so the harness can fall back without a Docker rebuild if
nested multiprocessing causes deadlocks in the eval container.
"""

from __future__ import annotations

import contextlib
import io
import multiprocessing
import os
import traceback
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any


_OUTPUT_TRUNCATION_LIMIT = 100_000  # bytes per stream — keeps Pipe traffic bounded


def _kernel_worker_loop(context_root: str, conn: Connection) -> None:
    """Executed inside the child process. Receives ops, replies via conn."""
    try:
        os.chdir(context_root)
    except OSError:
        # If chdir fails the parent will see it in the very first user code call.
        pass

    namespace: dict[str, Any] = {
        "__builtins__": __builtins__,
        "__name__": "__main__",
        "context_root": context_root,
        "Path": Path,
    }

    while True:
        try:
            message = conn.recv()
        except (EOFError, KeyboardInterrupt):
            break

        if not isinstance(message, tuple) or not message:
            break

        op = message[0]
        if op == "shutdown":
            break
        if op != "exec":
            try:
                conn.send({"success": False, "error": f"Unknown op: {op!r}", "output": "", "stderr": ""})
            except (BrokenPipeError, OSError):
                break
            continue

        code = message[1] if len(message) > 1 else ""
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        result: dict[str, Any]
        try:
            with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
                exec(code, namespace, namespace)
            result = {"success": True}
        except BaseException as exc:  # noqa: BLE001
            result = {
                "success": False,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }

        result["output"] = stdout_buffer.getvalue()[:_OUTPUT_TRUNCATION_LIMIT]
        result["stderr"] = stderr_buffer.getvalue()[:_OUTPUT_TRUNCATION_LIMIT]
        try:
            conn.send(result)
        except (BrokenPipeError, OSError):
            break


class PersistentKernel:
    """Owns a single worker subprocess. Namespace persists between execute() calls.

    Not thread-safe. Use one kernel per task; the ToolRegistry routes by task_id.
    """

    def __init__(self, context_root: Path) -> None:
        self._context_root = str(context_root.resolve())
        self._parent_conn: Connection | None = None
        self._process: multiprocessing.Process | None = None
        self._spawn()

    def _spawn(self) -> None:
        parent_conn, child_conn = multiprocessing.Pipe(duplex=True)
        process = multiprocessing.Process(
            target=_kernel_worker_loop,
            args=(self._context_root, child_conn),
            daemon=True,
        )
        process.start()
        # Close the child end in the parent — keeps EOF semantics correct.
        child_conn.close()
        self._parent_conn = parent_conn
        self._process = process

    def _cleanup(self) -> None:
        process = self._process
        if process is not None:
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=1.0)
        self._process = None
        if self._parent_conn is not None:
            try:
                self._parent_conn.close()
            except OSError:
                pass
            self._parent_conn = None

    def _ensure_alive(self) -> None:
        if self._process is None or not self._process.is_alive():
            self._cleanup()
            self._spawn()

    def execute(self, code: str, *, timeout_seconds: int = 30) -> dict[str, Any]:
        """Run code in the worker. On timeout/crash, respawn and report namespace_reset."""
        self._ensure_alive()
        assert self._parent_conn is not None  # for type checkers

        try:
            self._parent_conn.send(("exec", code))
        except (BrokenPipeError, OSError) as exc:
            self._cleanup()
            self._spawn()
            return {
                "success": False,
                "output": "",
                "stderr": "",
                "error": f"Failed to send code to kernel: {exc}",
                "namespace_reset": True,
            }

        if not self._parent_conn.poll(timeout_seconds):
            self._cleanup()
            self._spawn()
            return {
                "success": False,
                "output": "",
                "stderr": "",
                "error": f"Python execution timed out after {timeout_seconds} seconds.",
                "namespace_reset": True,
            }

        try:
            result = self._parent_conn.recv()
        except (EOFError, OSError) as exc:
            self._cleanup()
            self._spawn()
            return {
                "success": False,
                "output": "",
                "stderr": "",
                "error": f"Kernel died unexpectedly: {exc}",
                "namespace_reset": True,
            }

        if not isinstance(result, dict):
            return {
                "success": False,
                "output": "",
                "stderr": "",
                "error": f"Unexpected kernel response: {result!r}",
            }
        return result

    def close(self) -> None:
        process = self._process
        if process is not None and process.is_alive() and self._parent_conn is not None:
            try:
                self._parent_conn.send(("shutdown", None))
                process.join(timeout=1.0)
            except (BrokenPipeError, OSError):
                pass
        self._cleanup()

    def __enter__(self) -> "PersistentKernel":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
