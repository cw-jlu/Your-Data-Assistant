"""Factory for the Phase 2 Answer Agent."""
from __future__ import annotations

from kobushi_core.model import ModelAdapter

from experiments.exp_015_cascade_json_fix.agent import ReActAgent, ReActAgentConfig
from experiments.exp_015_cascade_json_fix.prompts.answer import build_answer_system_prompt
from experiments.exp_015_cascade_json_fix.tools.answer_registry import (
    create_answer_tool_registry,
)

# Phase 2 has a tight step budget: write SQL, fix at most 3 times.
ANSWER_MAX_STEPS = 4


def create_answer_agent(
    *,
    model: ModelAdapter,
    spec_preamble: str,
    spec_data: dict | None = None,
) -> ReActAgent:
    tools = create_answer_tool_registry(spec_data=spec_data or {})
    system_prompt = build_answer_system_prompt(tools.describe_for_prompt())
    return ReActAgent(
        model=model,
        tools=tools,
        config=ReActAgentConfig(max_steps=ANSWER_MAX_STEPS),
        system_prompt=system_prompt,
        preamble=spec_preamble,
    )
