"""Run a single task and print the raw model context at every step.

Output goes to stdout AND artifacts/debug/task_XX_calls.txt for later review.

Usage:
    uv run python scripts/run_task_debug.py
"""

import json
import sys
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "easy.yaml"
TASK_ID = "task_86"


class PrintModelWrapper(BaseChatModel):
    """Wraps a BaseChatModel to dump the raw messages before each invocation."""

    def __init__(self, wrapped: BaseChatModel, output_dir: Path):
        super().__init__()
        self._wrapped = wrapped
        self._call_count = 0
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def _llm_type(self) -> str:
        return "print_wrapper"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Any:
        self._call_count += 1
        call_file = self._output_dir / f"call_{self._call_count:02d}.txt"
        lines: list[str] = []
        lines.append("=" * 90)
        lines.append(f"MODEL CALL #{self._call_count}")
        lines.append("=" * 90)
        for i, msg in enumerate(messages):
            msg_type = getattr(msg, "type", type(msg).__name__)
            content = getattr(msg, "content", "")
            if isinstance(content, list):
                content_str = json.dumps(content, ensure_ascii=False, indent=2)
            elif isinstance(content, str):
                content_str = content
            else:
                content_str = str(content)
            lines.append(f"\n--- Message {i} [{msg_type}] (chars={len(content_str)}) ---")
            lines.append(content_str)
        lines.append("")
        text = "\n".join(lines)
        call_file.write_text(text, encoding="utf-8")

        # Also print a brief header to stdout
        msg_types = [getattr(m, "type", "?") for m in messages]
        total_chars = sum(
            len(json.dumps(getattr(m, "content", ""))) if isinstance(getattr(m, "content", ""), list)
            else len(str(getattr(m, "content", "")))
            for m in messages
        )
        print(f"[Call #{self._call_count:02d}] {msg_types} | total_chars={total_chars}")
        sys.stdout.flush()

        return self._wrapped._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    def bind_tools(self, *args: Any, **kwargs: Any) -> Any:
        inner = self._wrapped.bind_tools(*args, **kwargs)
        if inner is not self._wrapped:
            return PrintModelWrapper(inner, self._output_dir)
        return self


def main() -> None:
    from data_agent_baseline.agents.model import create_chat_model
    from data_agent_baseline.config import load_app_config
    from data_agent_baseline.agents.langgraph_runtime import LangGraphAgent, LangGraphAgentConfig
    from data_agent_baseline.tools.registry import create_default_tool_registry
    from data_agent_baseline.benchmark.dataset import DABenchPublicDataset

    config = load_app_config(CONFIG_PATH)

    print(f"Task: {TASK_ID}")
    print(f"Config: {CONFIG_PATH}")
    print(f"Model: {config.agent.model}")
    print(f"API Base: {config.agent.api_base}")
    print()

    dataset = DABenchPublicDataset(config.dataset.root_path)
    task = dataset.get_task(TASK_ID)
    print(f"Question: {task.question}")
    print()

    output_dir = PROJECT_ROOT / "artifacts" / "debug" / TASK_ID
    print(f"Full call dumps will be written to: {output_dir}")
    print()

    real_model = create_chat_model(
        model=config.agent.model,
        api_base=config.agent.api_base,
        api_key=config.agent.api_key,
        api_key_env=config.agent.api_key_env,
        temperature=config.agent.temperature,
    )
    debug_model = PrintModelWrapper(real_model, output_dir)

    tools = create_default_tool_registry(config.tool)
    agent = LangGraphAgent(
        model=debug_model,
        tools=tools,
        config=LangGraphAgentConfig(
            max_steps=config.agent.max_steps,
            enable_answer_validator=config.agent.enable_answer_validator,
            enable_data_inspector=config.agent.enable_data_inspector,
            enable_ambiguity_analysis=config.agent.enable_ambiguity_analysis,
            strip_reasoning_history=config.agent.strip_reasoning_history,
            reasoning_history_limit=config.agent.reasoning_history_limit,
            data_inspector=config.data_inspector,
            prompt_version=config.agent.prompt_version,
        ),
    )

    try:
        result = agent.run(task)
    except Exception as exc:
        import traceback
        print(f"\nERROR: {exc}")
        traceback.print_exc()
        return

    print(f"\nSucceeded: {result.succeeded}")
    print(f"Failure: {result.failure_reason}")
    if result.answer is not None:
        print(f"Answer: columns={result.answer.columns}, rows={len(result.answer.rows)}")
    print(f"\nTotal steps: {len(result.steps)}")
    print(f"Total model calls: {debug_model._call_count}")
    print(f"\nFull context saved to: {output_dir}")


if __name__ == "__main__":
    main()
