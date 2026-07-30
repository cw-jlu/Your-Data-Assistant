__all__ = [
    "BoundToolRegistry",
    "ToolExecutionResult",
    "ToolRegistry",
    "ToolRuntimeContext",
    "ToolSpec",
    "create_default_tool_registry",
]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(name)
    from data_agent_baseline.tools import registry

    return getattr(registry, name)
