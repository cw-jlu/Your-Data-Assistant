"""TraceProvider — central factory that creates Trace and Span instances."""

from __future__ import annotations

import abc
import contextlib
import logging
from typing import Any, TypeVar

from agents.tracing.processor_interface import TracingProcessor
from agents.tracing.span_data import SpanData
from agents.tracing.spans import NoOpSpan, Span, SpanImpl
from agents.tracing.traces import NoOpTrace, Trace, TraceImpl
from agents.tracing.util import gen_span_id, gen_trace_id

logger = logging.getLogger(__name__)

TSpanData = TypeVar("TSpanData", bound=SpanData)


class SynchronousMultiTracingProcessor(TracingProcessor):
    """Fans out to all registered processors, catching exceptions per-processor."""

    def __init__(self) -> None:
        self._processors: list[TracingProcessor] = []

    def add_processor(self, processor: TracingProcessor) -> None:
        self._processors.append(processor)

    def remove_processor(self, processor: TracingProcessor) -> None:
        with contextlib.suppress(ValueError):
            self._processors.remove(processor)

    def on_trace_start(self, trace: Trace) -> None:
        for p in self._processors:
            try:
                p.on_trace_start(trace)
            except Exception:
                logger.exception("Processor %s failed on_trace_start", type(p).__name__)

    def on_trace_end(self, trace: Trace) -> None:
        for p in self._processors:
            try:
                p.on_trace_end(trace)
            except Exception:
                logger.exception("Processor %s failed on_trace_end", type(p).__name__)

    def on_span_start(self, span: Span[Any]) -> None:
        for p in self._processors:
            try:
                p.on_span_start(span)
            except Exception:
                logger.exception("Processor %s failed on_span_start", type(p).__name__)

    def on_span_end(self, span: Span[Any]) -> None:
        for p in self._processors:
            try:
                p.on_span_end(span)
            except Exception:
                logger.exception("Processor %s failed on_span_end", type(p).__name__)

    def shutdown(self) -> None:
        for p in self._processors:
            try:
                p.shutdown()
            except Exception:
                logger.exception("Processor %s failed shutdown", type(p).__name__)

    def force_flush(self) -> None:
        for p in self._processors:
            try:
                p.force_flush()
            except Exception:
                logger.exception("Processor %s failed force_flush", type(p).__name__)


class TraceProvider(abc.ABC):
    @abc.abstractmethod
    def create_trace(
        self,
        name: str,
        trace_id: str | None = None,
        run_id: str = "",
        task_id: str = "",
        metadata: dict[str, Any] | None = None,
        disabled: bool = False,
    ) -> Trace: ...

    @abc.abstractmethod
    def create_span(
        self,
        span_data: TSpanData,
        span_id: str | None = None,
        parent: Span[Any] | None = None,
        disabled: bool = False,
    ) -> Span[TSpanData]: ...

    @abc.abstractmethod
    def set_disabled(self, disabled: bool) -> None: ...

    def register_processor(self, processor: TracingProcessor) -> None:
        raise NotImplementedError

    def unregister_processor(self, processor: TracingProcessor) -> None:
        raise NotImplementedError

    def shutdown(self) -> None:  # noqa: B027
        pass

    def force_flush(self) -> None:  # noqa: B027
        pass


class DefaultTraceProvider(TraceProvider):
    def __init__(self) -> None:
        self._disabled = False
        self._multi_processor = SynchronousMultiTracingProcessor()

    def register_processor(self, processor: TracingProcessor) -> None:
        self._multi_processor.add_processor(processor)

    def unregister_processor(self, processor: TracingProcessor) -> None:
        self._multi_processor.remove_processor(processor)

    def set_disabled(self, disabled: bool) -> None:
        self._disabled = disabled

    def create_trace(
        self,
        name: str,
        trace_id: str | None = None,
        run_id: str = "",
        task_id: str = "",
        metadata: dict[str, Any] | None = None,
        disabled: bool = False,
    ) -> Trace:
        if self._disabled or disabled:
            return NoOpTrace()
        return TraceImpl(
            name=name,
            trace_id=trace_id or gen_trace_id(),
            run_id=run_id,
            task_id=task_id,
            processor=self._multi_processor,
            metadata=metadata,
        )

    def create_span(
        self,
        span_data: TSpanData,
        span_id: str | None = None,
        parent: Span[Any] | None = None,
        disabled: bool = False,
    ) -> Span[TSpanData]:
        if self._disabled or disabled:
            return NoOpSpan(span_data)

        from agents.tracing.scope import Scope

        if parent is not None:
            parent_id = parent.span_id
            trace_id = parent.trace_id
        else:
            current_span = Scope.get_current_span()
            parent_id = current_span.span_id if current_span else None
            current_trace = Scope.get_current_trace()
            trace_id = current_trace.trace_id if current_trace else ""

        return SpanImpl(
            trace_id=trace_id,
            span_id=span_id or gen_span_id(),
            parent_id=parent_id if parent_id != "no-op" else None,
            processor=self._multi_processor,
            span_data=span_data,
        )

    def shutdown(self) -> None:
        self._multi_processor.shutdown()

    def force_flush(self) -> None:
        self._multi_processor.force_flush()
