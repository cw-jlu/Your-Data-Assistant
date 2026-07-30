"""Scope — static class wrapping two ContextVars for current trace/span."""

from __future__ import annotations

from contextvars import ContextVar, Token
from typing import Any

from agents.tracing.spans import Span
from agents.tracing.traces import Trace

_current_span: ContextVar[Span[Any] | None] = ContextVar("current_span", default=None)
_current_trace: ContextVar[Trace | None] = ContextVar("current_trace", default=None)


class Scope:
    @classmethod
    def get_current_span(cls) -> Span[Any] | None:
        return _current_span.get()

    @classmethod
    def set_current_span(cls, span: Span[Any]) -> Token[Span[Any] | None]:
        return _current_span.set(span)

    @classmethod
    def reset_current_span(cls, token: Token[Span[Any] | None]) -> None:
        _current_span.reset(token)

    @classmethod
    def get_current_trace(cls) -> Trace | None:
        return _current_trace.get()

    @classmethod
    def set_current_trace(cls, trace: Trace) -> Token[Trace | None]:
        return _current_trace.set(trace)

    @classmethod
    def reset_current_trace(cls, token: Token[Trace | None]) -> None:
        _current_trace.reset(token)
