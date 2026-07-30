from __future__ import annotations

from agents.benchmark.schema import PublicTask
from agents.llm.types import ModelToolCall
from agents.tools.registry import ToolExecutionResult, ToolRegistry


def dispatch_tool_call(
    task: PublicTask,
    call: ModelToolCall,
    registry: ToolRegistry,
) -> tuple[ToolExecutionResult, bool]:
    """执行一次 tool_call 并返回 `(result, should_terminate)`"""
    result = registry.execute(task, call.name, call.arguments)
    definition = registry.definitions[call.name]
    should_terminate = definition.is_terminal and result.ok
    return result, should_terminate
