"""Provider-neutral model protocol and response types."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class ModelMessage:
    """chat.completions-style message used by all model adapters."""

    role: str
    content: str | list[dict[str, Any]]
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


@dataclass(frozen=True, slots=True)
class ModelToolCall:
    """Normalized tool call emitted by the native model protocol."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Token usage snapshot returned by model providers."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0

    def to_dict(self) -> dict[str, int]:
        """Return a JSON-friendly usage payload with stable field names."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ModelResponse:
    """Normalized response returned by all model adapters."""

    content: str
    tool_calls: list[ModelToolCall] = field(default_factory=lambda: [])  # noqa: PIE807
    raw_response: str = ""
    raw_tool_calls: list[dict[str, Any]] = field(default_factory=lambda: [])  # noqa: PIE807
    reasoning_content: str = ""
    usage: TokenUsage = field(default_factory=TokenUsage)
    latency_ms: int = 0
    rate_limit_metrics: dict[str, Any] | None = None


class ToolSchemaSource(Protocol):
    """Anything that can render an OpenAI ``tools=[...]`` payload (e.g. ToolRegistry)."""

    def to_openai_tools(self) -> list[dict[str, Any]]:
        raise NotImplementedError


class ModelAdapter(Protocol):
    """Anything exposing ``complete(messages) -> ModelResponse`` can act as a model.

    ``tools`` 是 per-call 工具广告（与 chat.completions 请求形态一致）：调用方（ReAct
    循环）每轮把自己手里的 registry 下推，保证请求体广告与执行分发同源。不支持
    native 工具协议的适配器忽略该参数。None = 沿用适配器构造期的默认工具集。
    """

    def complete(
        self,
        messages: list[ModelMessage],
        *,
        tools: ToolSchemaSource | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        raise NotImplementedError
