"""Instrumentation wrappers for ModelAdapter and ToolRegistry."""

from __future__ import annotations

import json
from collections.abc import Callable
from contextvars import ContextVar
from typing import Any

from agents.benchmark.schema import PublicTask
from agents.llm.types import ModelAdapter, ModelMessage, ModelResponse, ToolSchemaSource
from agents.tools import ToolExecutionResult, ToolRegistry
from agents.tracing.create import function_span, generation_span
from agents.tracing.spans import SpanError

LLMExchangeCallback = Callable[[list[ModelMessage], ModelResponse], None]

_llm_exchange_hook: ContextVar[LLMExchangeCallback | None] = ContextVar(
    "_llm_exchange_hook", default=None
)


def set_llm_exchange_hook(hook: LLMExchangeCallback | None) -> Any:
    """Register a per-context callback invoked after every LLM completion."""
    return _llm_exchange_hook.set(hook)


def reset_llm_exchange_hook(token: Any) -> None:
    _llm_exchange_hook.reset(token)


def _serialize_messages(messages: list[ModelMessage]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for msg in messages:
        entry: dict[str, Any] = {"role": msg.role}
        if isinstance(msg.content, str):
            entry["content"] = msg.content
        else:
            entry["content"] = [
                {**b, "video_url": {"url": "<base64-stripped>"}}
                if b.get("type") == "video_url"
                else {**b, "image_url": {"url": "<base64-stripped>"}}
                if b.get("type") == "image_url"
                and isinstance(b.get("image_url", {}).get("url"), str)
                and b["image_url"]["url"].startswith("data:")
                else b
                for b in msg.content
            ]
        if msg.tool_calls:
            entry["tool_calls"] = msg.tool_calls
        if msg.tool_call_id:
            entry["tool_call_id"] = msg.tool_call_id
        result.append(entry)
    return result


class TracedModelAdapter:
    def __init__(self, inner: ModelAdapter) -> None:
        self._inner = inner

    @property
    def inner(self) -> ModelAdapter:
        return self._inner

    def complete(
        self,
        messages: list[ModelMessage],
        *,
        tools: ToolSchemaSource | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        model_name = getattr(self._inner, "model", None)
        with generation_span(model=model_name) as span:
            span.span_data.input = _serialize_messages(messages)
            response = self._inner.complete(messages, tools=tools, **kwargs)
            span.span_data.usage = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "reasoning_tokens": response.usage.reasoning_tokens,
                "cached_tokens": response.usage.cached_tokens,
                "total_tokens": response.usage.total_tokens,
                "latency_ms": response.latency_ms,
            }
            output_msg: dict[str, Any] = {"role": "assistant", "content": response.content}
            if response.reasoning_content:
                output_msg["reasoning_content"] = response.reasoning_content
            if response.raw_tool_calls:
                output_msg["tool_calls"] = response.raw_tool_calls
            span.span_data.output = [output_msg]
            if response.rate_limit_metrics is not None:
                span.span_data.usage["rate_limit"] = response.rate_limit_metrics

            hook = _llm_exchange_hook.get(None)
            if hook is not None:
                hook(messages, response)

            return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class TracedToolRegistry:
    def __init__(self, inner: ToolRegistry) -> None:
        self._inner = inner

    def execute(
        self, task: PublicTask, action: str, action_input: dict[str, Any]
    ) -> ToolExecutionResult:
        with function_span(name=action) as span:
            span.span_data.input = json.dumps(action_input, ensure_ascii=False, default=str)
            result = self._inner.execute(task, action, action_input)
            span.span_data.output = json.dumps(result.content, ensure_ascii=False, default=str)
            if not result.ok:
                span.set_error(SpanError(message=f"Tool {action} returned ok=False"))
            return result

    @property
    def definitions(self) -> Any:
        return self._inner.definitions

    def to_openai_tools(self) -> list[dict[str, Any]]:
        """ToolSchemaSource 显式成员：agent 每轮把 traced registry 下推给 complete()。"""
        return self._inner.to_openai_tools()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)
