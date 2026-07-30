"""Span ABC + SpanImpl + NoOpSpan — the three-layer pattern from OpenAI Agents SDK."""

from __future__ import annotations

import abc
import logging
from contextvars import Token
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from agents.tracing.processor_interface import TracingProcessor
from agents.tracing.span_data import SpanData
from agents.tracing.util import time_iso

logger = logging.getLogger(__name__)

TSpanData = TypeVar("TSpanData", bound=SpanData)


@dataclass(frozen=True, slots=True)
class SpanError:
    message: str
    data: Any = None


class Span(abc.ABC, Generic[TSpanData]):
    @property
    @abc.abstractmethod
    def trace_id(self) -> str: ...

    @property
    @abc.abstractmethod
    def span_id(self) -> str: ...

    @property
    @abc.abstractmethod
    def parent_id(self) -> str | None: ...

    @property
    @abc.abstractmethod
    def span_data(self) -> TSpanData: ...

    @abc.abstractmethod
    def start(self, mark_as_current: bool = False) -> None: ...

    @abc.abstractmethod
    def finish(self, reset_current: bool = False) -> None: ...

    @abc.abstractmethod
    def set_error(self, error: SpanError) -> None: ...

    @abc.abstractmethod
    def export(self) -> dict[str, Any] | None: ...

    def __enter__(self) -> Span[TSpanData]:
        self.start(mark_as_current=True)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if exc_val is not None:
            self.set_error(SpanError(message=str(exc_val)))
        self.finish(reset_current=True)


class SpanImpl(Span[TSpanData]):
    __slots__ = (
        "_ended_at",
        "_error",
        "_parent_id",
        "_prev_span_token",
        "_processor",
        "_span_data",
        "_span_id",
        "_started_at",
        "_trace_id",
    )

    def __init__(
        self,
        trace_id: str,
        span_id: str,
        parent_id: str | None,
        processor: TracingProcessor,
        span_data: TSpanData,
    ) -> None:
        self._trace_id = trace_id
        self._span_id = span_id
        self._parent_id = parent_id
        self._processor = processor
        self._span_data = span_data
        self._started_at: str | None = None
        self._ended_at: str | None = None
        self._error: SpanError | None = None
        self._prev_span_token: Token[Span[Any] | None] | None = None

    @property
    def trace_id(self) -> str:
        return self._trace_id

    @property
    def span_id(self) -> str:
        return self._span_id

    @property
    def parent_id(self) -> str | None:
        return self._parent_id

    @property
    def span_data(self) -> TSpanData:
        return self._span_data

    def start(self, mark_as_current: bool = False) -> None:
        self._started_at = time_iso()
        if mark_as_current:
            from agents.tracing.scope import Scope

            self._prev_span_token = Scope.set_current_span(self)
        self._processor.on_span_start(self)

    def finish(self, reset_current: bool = False) -> None:
        self._ended_at = time_iso()
        if reset_current and self._prev_span_token is not None:
            from agents.tracing.scope import Scope

            Scope.reset_current_span(self._prev_span_token)
            self._prev_span_token = None
        self._processor.on_span_end(self)

    def set_error(self, error: SpanError) -> None:
        self._error = error

    def export(self) -> dict[str, Any] | None:
        return {
            "object": "trace.span",
            "id": self._span_id,
            "trace_id": self._trace_id,
            "parent_id": self._parent_id,
            "started_at": self._started_at,
            "ended_at": self._ended_at,
            "span_data": self._span_data.export(),
            "error": {"message": self._error.message, "data": self._error.data}
            if self._error
            else None,
        }


class NoOpSpan(Span[TSpanData]):
    __slots__ = ("_prev_span_token", "_span_data")

    def __init__(self, span_data: TSpanData) -> None:
        self._span_data = span_data
        self._prev_span_token: Token[Span[Any] | None] | None = None

    @property
    def trace_id(self) -> str:
        return "no-op"

    @property
    def span_id(self) -> str:
        return "no-op"

    @property
    def parent_id(self) -> str | None:
        return None

    @property
    def span_data(self) -> TSpanData:
        return self._span_data

    def start(self, mark_as_current: bool = False) -> None:
        if mark_as_current:
            from agents.tracing.scope import Scope

            self._prev_span_token = Scope.set_current_span(self)

    def finish(self, reset_current: bool = False) -> None:
        if reset_current and self._prev_span_token is not None:
            from agents.tracing.scope import Scope

            Scope.reset_current_span(self._prev_span_token)
            self._prev_span_token = None

    def set_error(self, error: SpanError) -> None:
        pass

    def export(self) -> dict[str, Any] | None:
        return None
