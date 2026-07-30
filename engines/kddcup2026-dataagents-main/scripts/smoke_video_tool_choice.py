"""Deployment smoke test: named tool_choice + thinking coexistence on vLLM.

Usage:
    uv run python scripts/smoke_video_tool_choice.py \
        --api-base http://<vllm-host>/v1 --model qwen3.5-35b-a3b --api-key EMPTY

PASS: the response contains a `report` tool call.
FAIL: HTTP error or prose-only response means use `agent.video_tool_choice: auto`.
"""

from __future__ import annotations

import argparse

from agents.llm.openai import NativeToolsOpenAIAdapter
from agents.llm.types import ModelMessage
from agents.video.prompt import VIDEO_AGENT_SYSTEM_PROMPT
from agents.video.registry import create_video_tool_registry


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default="EMPTY")
    args = parser.parse_args()

    adapter = NativeToolsOpenAIAdapter(
        model=args.model,
        api_base=args.api_base,
        api_key=args.api_key,
        tools=create_video_tool_registry(),
        allow_parallel_tool_calls=False,
        tool_choice={"type": "function", "function": {"name": "report"}},
        enable_thinking=True,
        max_tokens=2048,
    )
    response = adapter.complete(
        [
            ModelMessage(role="system", content=VIDEO_AGENT_SYSTEM_PROMPT),
            ModelMessage(
                role="user",
                content=(
                    "No video is attached in this smoke test. Submit a report with "
                    'summary "smoke test", empty timeline/extracted_data/coverage, '
                    'and one warning "no video attached".'
                ),
            ),
        ]
    )
    names = [call.name for call in response.tool_calls]
    if names == ["report"]:
        print("PASS: forced tool_choice produced a report call; thinking coexists.")
        return 0
    print(f"FAIL: tool_calls={names!r}, content={response.content[:200]!r}")
    print("Set `agent.video_tool_choice: auto` in the run config.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
