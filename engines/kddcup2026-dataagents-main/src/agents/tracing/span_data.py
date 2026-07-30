"""Typed SpanData subclasses — each span kind carries its own mutable data bag.

Follows the OpenAI Agents SDK pattern: SpanData is an ABC with `.export()` and
`.type`. Concrete subclasses use `__slots__` for memory efficiency. Callers
mutate attributes between span `__enter__` and `__exit__`; the final `.export()`
captures whatever state was set.
"""

from __future__ import annotations

import abc
from typing import Any


class SpanData(abc.ABC):
    @abc.abstractmethod
    def export(self) -> dict[str, Any]: ...

    @property
    @abc.abstractmethod
    def type(self) -> str: ...


class AgentSpanData(SpanData):
    __slots__ = ("max_steps", "name", "protocol", "question", "tools")

    def __init__(
        self,
        name: str = "",
        tools: list[str] | None = None,
        max_steps: int | None = None,
        protocol: str | None = None,
    ) -> None:
        self.name = name
        self.tools = tools or []
        self.max_steps = max_steps
        self.protocol = protocol
        self.question: str | None = None

    @property
    def type(self) -> str:
        return "agent"

    def export(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "name": self.name,
            "tools": self.tools,
            "max_steps": self.max_steps,
            "protocol": self.protocol,
            "question": self.question,
        }


class TurnSpanData(SpanData):
    __slots__ = ("agent_name", "turn", "usage")

    def __init__(self, turn: int = 0, agent_name: str = "") -> None:
        self.turn = turn
        self.agent_name = agent_name
        self.usage: dict[str, Any] | None = None

    @property
    def type(self) -> str:
        return "turn"

    def export(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "name": f"turn {self.turn}",
            "turn": self.turn,
            "agent_name": self.agent_name,
            "usage": self.usage,
        }


class GenerationSpanData(SpanData):
    __slots__ = ("input", "model", "model_config_data", "output", "usage")

    def __init__(self, model: str | None = None) -> None:
        self.model = model
        self.input: list[dict[str, Any]] | None = None
        self.output: list[dict[str, Any]] | None = None
        self.usage: dict[str, Any] | None = None
        self.model_config_data: dict[str, Any] | None = None

    @property
    def type(self) -> str:
        return "generation"

    def export(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "name": self.model or "generation",
            "model": self.model,
            "input": self.input,
            "output": self.output,
            "usage": self.usage,
            "model_config": self.model_config_data,
        }


class FunctionSpanData(SpanData):
    __slots__ = ("input", "name", "output")

    def __init__(self, name: str = "") -> None:
        self.name = name
        self.input: str | None = None
        self.output: str | None = None

    @property
    def type(self) -> str:
        return "function"

    def export(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "name": self.name,
            "input": self.input,
            "output": self.output,
        }


class ErrorSpanData(SpanData):
    __slots__ = ("data", "error_type", "message", "name")

    def __init__(
        self,
        name: str = "",
        error_type: str = "",
        message: str = "",
        data: Any = None,
    ) -> None:
        self.name = name
        self.error_type = error_type
        self.message = message
        self.data = data

    @property
    def type(self) -> str:
        return "error"

    def export(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "name": self.name,
            "error_type": self.error_type,
            "message": self.message,
            "data": self.data,
        }
