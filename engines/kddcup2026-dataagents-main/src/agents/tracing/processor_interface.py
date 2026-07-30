"""ABC interfaces for tracing processors and exporters."""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agents.tracing.spans import Span
    from agents.tracing.traces import Trace


class TracingProcessor(abc.ABC):
    @abc.abstractmethod
    def on_trace_start(self, trace: Trace) -> None: ...

    @abc.abstractmethod
    def on_trace_end(self, trace: Trace) -> None: ...

    @abc.abstractmethod
    def on_span_start(self, span: Span[Any]) -> None: ...

    @abc.abstractmethod
    def on_span_end(self, span: Span[Any]) -> None: ...

    @abc.abstractmethod
    def shutdown(self) -> None: ...

    @abc.abstractmethod
    def force_flush(self) -> None: ...


class TracingExporter(abc.ABC):
    @abc.abstractmethod
    def export(self, items: list[dict[str, Any] | None]) -> None: ...
