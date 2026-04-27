"""
此模块负责构建 Agent 所需的所有提示词（System Prompt, Task Prompt, Observation Prompt）。
"""
from __future__ import annotations

import json
from pathlib import Path

from data_agent_baseline.benchmark.schema import PublicTask

# 获取提示词文件的目录
_PROMPTS_DIR = Path(__file__).parent / "prompts"

# 从文件加载提示词
def _load_prompt(filename: str) -> str:
    """从提示词文件中加载文本内容。"""
    prompt_file = _PROMPTS_DIR / filename
    if not prompt_file.exists():
        raise FileNotFoundError(f"提示词文件不存在: {prompt_file}")
    return prompt_file.read_text(encoding="utf-8").strip()

REACT_SYSTEM_PROMPT = _load_prompt("react_system_prompt.txt")
RESPONSE_EXAMPLES = _load_prompt("response_examples.txt")


def build_system_prompt(tool_descriptions: str, system_prompt: str | None = None) -> str:
    base_prompt = system_prompt or REACT_SYSTEM_PROMPT
    return (
        f"{base_prompt}\n\n"
        "Available tools:\n"
        f"{tool_descriptions}\n\n"
        f"{RESPONSE_EXAMPLES}\n\n"
        "You must always return a single ```json fenced block containing one JSON object "
        "with keys `thought`, `action`, and `action_input`, and no extra text."
    )


def build_task_prompt(task: PublicTask) -> str:
    return (
        f"Question: {task.question}\n"
        f"Task Difficulty: {task.difficulty}\n"
        "All tool file paths are relative to the task context directory. "
        "When you have the final table, call the `answer` tool."
    )


def build_observation_prompt(observation: dict[str, object]) -> str:
    rendered = json.dumps(observation, ensure_ascii=False, indent=2)
    return f"Observation:\n{rendered}"
