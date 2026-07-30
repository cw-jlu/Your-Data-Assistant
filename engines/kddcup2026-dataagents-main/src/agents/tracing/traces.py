"""Trace ABC + TraceImpl + NoOpTrace — run-level lifecycle container."""

from __future__ import annotations

import abc
from contextvars import Token
from typing import Any

from agents.tracing.processor_interface import TracingProcessor
from agents.tracing.util import time_iso


class Trace(abc.ABC):
    @property
    @abc.abstractmethod
    def trace_id(self) -> str: ...

    @property
    @abc.abstractmethod
    def name(self) -> str: ...

    @abc.abstractmethod
    def start(self, mark_as_current: bool = False) -> None: ...

    @abc.abstractmethod
    def finish(self, reset_current: bool = False) -> None: ...

    @abc.abstractmethod
    def export(self) -> dict[str, Any] | None: ...

    def __enter__(self) -> Trace:
        self.start(mark_as_current=True)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.finish(reset_current=True)


class TraceImpl(Trace):
    __slots__ = (
        "_ended_at",
        "_metadata",
        "_name",
        "_prev_trace_token",
        "_processor",
        "_run_id",
        "_started_at",
        "_task_id",
        "_trace_id",
    )

    def __init__(
        self,
        name: str,
        trace_id: str,
        run_id: str,
        task_id: str,
        processor: TracingProcessor,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._name = name
        self._trace_id = trace_id
        self._run_id = run_id
        self._task_id = task_id
        self._processor = processor
        self._metadata = metadata if metadata is not None else {}
        self._started_at: str | None = None
        self._ended_at: str | None = None
        self._prev_trace_token: Token[Trace | None] | None = None

    @property
    def trace_id(self) -> str:
        return self._trace_id

    @property
    def name(self) -> str:
        return self._name

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def task_id(self) -> str:
        return self._task_id

    def start(self, mark_as_current: bool = False) -> None:
        self._started_at = time_iso()
        if mark_as_current:
            from agents.tracing.scope import Scope

            self._prev_trace_token = Scope.set_current_trace(self)
        self._processor.on_trace_start(self)

    def finish(self, reset_current: bool = False) -> None:
        self._ended_at = time_iso()
        if reset_current and self._prev_trace_token is not None:
            from agents.tracing.scope import Scope

            Scope.reset_current_trace(self._prev_trace_token)
            self._prev_trace_token = None
        self._processor.on_trace_end(self)

    def export(self) -> dict[str, Any] | None:
        return {
            "object": "trace",
            "id": self._trace_id,
            "name": self._name,
            "run_id": self._run_id,
            "task_id": self._task_id,
            "started_at": self._started_at,
            "ended_at": self._ended_at,
            "metadata": self._metadata,
        }


class NoOpTrace(Trace):
    __slots__ = ("_prev_trace_token",)

    def __init__(self) -> None:
        self._prev_trace_token: Token[Trace | None] | None = None

    @property
    def trace_id(self) -> str:
        return "no-op"

    @property
    def name(self) -> str:
        return ""

    def start(self, mark_as_current: bool = False) -> None:
        if mark_as_current:
            from agents.tracing.scope import Scope

            self._prev_trace_token = Scope.set_current_trace(self)

    def finish(self, reset_current: bool = False) -> None:
        if reset_current and self._prev_trace_token is not None:
            from agents.tracing.scope import Scope

            Scope.reset_current_trace(self._prev_trace_token)
            self._prev_trace_token = None

    def export(self) -> dict[str, Any] | None:
        return None
