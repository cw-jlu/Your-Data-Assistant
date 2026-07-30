"""TraceCtxManager — run-level lifecycle wrapper for trace creation."""

from __future__ import annotations

from typing import Any

from agents.tracing.setup import get_trace_provider
from agents.tracing.traces import Trace


class TraceCtxManager:
    def __init__(
        self,
        workflow_name: str,
        trace_id: str | None = None,
        run_id: str = "",
        task_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._workflow_name = workflow_name
        self._trace_id = trace_id
        self._run_id = run_id
        self._task_id = task_id
        self._metadata = metadata
        self._trace: Trace | None = None

    def __enter__(self) -> Trace:
        self._trace = get_trace_provider().create_trace(
            name=self._workflow_name,
            trace_id=self._trace_id,
            run_id=self._run_id,
            task_id=self._task_id,
            metadata=self._metadata,
        )
        self._trace.start(mark_as_current=True)
        return self._trace

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._trace is not None:
            self._trace.finish(reset_current=True)
