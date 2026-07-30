"""``@function_tool`` decorator.

Supports bare ``@function_tool`` and parameterised ``@function_tool(...)``
forms.  Name defaults to the handler's ``__name__``, description to its
docstring, and the input model is auto-generated from the signature.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, get_type_hints

from pydantic import BaseModel, ConfigDict, create_model

from agents.tools.registry import FunctionTool, ToolHandler
from agents.tools.schema_normalize import normalize_for_vllm


def _model_from_signature(handler: Callable[..., Any]) -> type[BaseModel]:
    hints = get_type_hints(handler, include_extras=True)
    sig = inspect.signature(handler)
    params = list(sig.parameters.values())[1:]  # skip task

    field_definitions: dict[str, Any] = {}
    for param in params:
        ann = hints.get(param.name, Any)
        default = ... if param.default is inspect.Parameter.empty else param.default
        field_definitions[param.name] = (ann, default)

    return create_model(
        handler.__name__,
        __config__=ConfigDict(extra="forbid"),
        **field_definitions,
    )


def _unpack_handler(handler: Callable[..., Any], model: type[BaseModel]) -> ToolHandler:
    fields = list(model.model_fields)
    if not fields:
        return lambda task, _args: handler(task)  # type: ignore[return-value]
    return lambda task, args: handler(task, **{k: getattr(args, k) for k in fields})  # type: ignore[return-value]


def function_tool(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    input_model: type[BaseModel] | None = None,
    is_terminal: bool = False,
) -> Any:
    """Decorator: wraps a handler into a ``FunctionTool``.

    ``@function_tool`` (bare) and ``@function_tool(...)`` (parameterised)
    are both supported.  *name* defaults to ``handler.__name__``,
    *description* to ``inspect.cleandoc(handler.__doc__)``.
    """

    def decorator(handler: Callable[..., Any]) -> FunctionTool:
        tool_name = name or handler.__name__
        tool_desc = description or inspect.cleandoc(handler.__doc__ or "")

        if input_model is not None:
            model = input_model
            actual_handler: ToolHandler = handler  # type: ignore[assignment]
        else:
            model = _model_from_signature(handler)
            actual_handler = _unpack_handler(handler, model)

        schema = normalize_for_vllm(model.model_json_schema())
        return FunctionTool(
            name=tool_name,
            description=tool_desc,
            input_model=model,
            handler=actual_handler,
            is_terminal=is_terminal,
            json_schema=schema,
        )

    if func is not None:
        return decorator(func)
    return decorator
