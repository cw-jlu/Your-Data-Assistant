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


def build_system_prompt(tool_descriptions: str, system_prompt: str | None = None, data_roadmap: str | None = None) -> str:
    base_prompt = system_prompt or REACT_SYSTEM_PROMPT
    roadmap_section = f"\n\n{data_roadmap}\n" if data_roadmap else ""
    return (
        f"{base_prompt}\n"
        f"{roadmap_section}\n"
        "Available tools:\n"
        f"{tool_descriptions}\n\n"
        f"{RESPONSE_EXAMPLES}\n\n"
        "You must always return a single ```json fenced block containing one JSON object "
        "with keys `thought`, `reflection`, `data_sufficient`, `action`, and `action_input`, and no extra text."
    )


def build_task_prompt(task: PublicTask) -> str:
    return (
        f"Question: {task.question}\n"
        "All tool file paths are relative to the task context directory. "
        "When you have the final table, call the `answer` tool.\n\n"
        "🚨 BEFORE CALLING `answer`, YOU MUST VERIFY:\n"
        "1. Formatting: Does your final output exactly match the requested columns and rows? Do NOT include extra explanatory columns.\n"
        "2. Aggregation: Did the question ask for a SUM, AVG, or Percentage? Ensure your mathematical operations strictly follow the business logic (e.g., percentages usually need to be multiplied by 100).\n"
        "3. Time Reference: If calculating age or duration, use the current year 2026 unless explicitly stated otherwise in the document.\n"
        "4. Knowledge Rules: Did you consult `knowledge.md` to confirm the exact thresholds (e.g., 'abnormal') before filtering the data?"
    )


def build_observation_prompt(observation: dict[str, object]) -> str:
    rendered = json.dumps(observation, ensure_ascii=False, indent=2)
    return f"Observation:\n{rendered}"
