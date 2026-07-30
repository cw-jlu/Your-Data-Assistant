"""Factory for the Phase 1 Exploration Agent."""
from __future__ import annotations

from kobushi_core.model import ModelAdapter

from experiments.exp_022_cascade_v3.agent import ReActAgent, ReActAgentConfig
from experiments.exp_022_cascade_v3.prompts.exploration import (
    build_exploration_system_prompt,
)
from experiments.exp_022_cascade_v3.tools.exploration_registry import (
    create_exploration_tool_registry,
    ToolRegistry,
)

# Phase 1 has a generous step budget for thorough exploration before commit_spec.
EXPLORATION_MAX_STEPS = 24


def create_exploration_agent(
    *,
    model: ModelAdapter,
    preamble: str | None,
    tools: ToolRegistry | None = None,
) -> ReActAgent:
    effective_tools = tools or create_exploration_tool_registry()
    system_prompt = build_exploration_system_prompt(effective_tools.describe_for_prompt())
    return ReActAgent(
        model=model,
        tools=effective_tools,
        config=ReActAgentConfig(max_steps=EXPLORATION_MAX_STEPS),
        system_prompt=system_prompt,
        preamble=preamble,
    )
