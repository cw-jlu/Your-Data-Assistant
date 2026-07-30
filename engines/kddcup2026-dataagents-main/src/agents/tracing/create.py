"""Top-level factory functions — the public API callers use."""

from __future__ import annotations

from typing import Any

from agents.tracing.setup import get_trace_provider
from agents.tracing.span_data import (
    AgentSpanData,
    ErrorSpanData,
    FunctionSpanData,
    GenerationSpanData,
    TurnSpanData,
)
from agents.tracing.spans import Span
from agents.tracing.traces import Trace


def trace(
    workflow_name: str,
    trace_id: str | None = None,
    run_id: str = "",
    task_id: str = "",
    metadata: dict[str, Any] | None = None,
    disabled: bool = False,
) -> Trace:
    return get_trace_provider().create_trace(
        name=workflow_name,
        trace_id=trace_id,
        run_id=run_id,
        task_id=task_id,
        metadata=metadata,
        disabled=disabled,
    )


def get_current_trace() -> Trace | None:
    from agents.tracing.scope import Scope

    return Scope.get_current_trace()


def get_current_span() -> Span[Any] | None:
    from agents.tracing.scope import Scope

    return Scope.get_current_span()


def agent_span(
    name: str,
    tools: list[str] | None = None,
    max_steps: int | None = None,
    protocol: str | None = None,
    disabled: bool = False,
) -> Span[AgentSpanData]:
    return get_trace_provider().create_span(
        span_data=AgentSpanData(name=name, tools=tools, max_steps=max_steps, protocol=protocol),
        disabled=disabled,
    )


def turn_span(
    turn: int,
    agent_name: str = "",
    disabled: bool = False,
) -> Span[TurnSpanData]:
    return get_trace_provider().create_span(
        span_data=TurnSpanData(turn=turn, agent_name=agent_name),
        disabled=disabled,
    )


def generation_span(
    model: str | None = None,
    disabled: bool = False,
) -> Span[GenerationSpanData]:
    return get_trace_provider().create_span(
        span_data=GenerationSpanData(model=model),
        disabled=disabled,
    )


def function_span(
    name: str,
    disabled: bool = False,
) -> Span[FunctionSpanData]:
    return get_trace_provider().create_span(
        span_data=FunctionSpanData(name=name),
        disabled=disabled,
    )


def error_span(
    name: str,
    error_type: str = "",
    message: str = "",
    data: Any = None,
    disabled: bool = False,
) -> Span[ErrorSpanData]:
    return get_trace_provider().create_span(
        span_data=ErrorSpanData(name=name, error_type=error_type, message=message, data=data),
        disabled=disabled,
    )
