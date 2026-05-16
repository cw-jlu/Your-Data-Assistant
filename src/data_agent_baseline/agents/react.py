from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from data_agent_baseline.agents.model import ModelAdapter, ModelMessage
from data_agent_baseline.agents.prompt import (
    build_observation_prompt,
    build_system_prompt,
    build_task_prompt,
    REACT_SYSTEM_PROMPT,
)
from data_agent_baseline.benchmark.schema import AnswerTable, PublicTask
from data_agent_baseline.tools.registry import ToolRegistry


# 定义 ReAct Agent 的配置
@dataclass(frozen=True, slots=True)
class ReActAgentConfig:
    # 最大步数，防止 Agent 进入无限循环
    max_steps: int = 35
    # 连续报错达到多少次时进行反思提醒
    error_reflection_threshold: int = 3
    # 连续报错达到多少次时停止任务（熔断）
    max_consecutive_errors: int = 6
    # RAG 检索返回的最大分块数
    rag_top_k: int = 5
    # 连续执行相同操作多少次时判定为死循环
    max_repeated_actions: int = 3


# 清理模型输出中的 Markdown 代码块（JSON 围栏）
def _strip_markdown_code_blocks(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) > 2 and lines[0].startswith("```") and lines[-1].startswith("```"):
            return "\n".join(lines[1:-1])
    return text


def _extract_json_block(text: str) -> str | None:
    """Extract the last JSON object from text, even if preceded by other content."""
    import re
    # Find all ```...``` fenced blocks
    blocks = re.findall(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
    for block in reversed(blocks):
        stripped = block.strip()
        if stripped.startswith("{"):
            return stripped
    # Try to find a raw JSON object in the text
    matches = re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    for match in reversed(matches):
        try:
            json.loads(match)
            return match
        except json.JSONDecodeError:
            continue
    return None


# 解析模型的单步响应为 JSON 结构
def parse_model_step(raw_response: str) -> dict[str, Any]:
    # Try direct parse first (fast path)
    cleaned_text = _strip_markdown_code_blocks(raw_response)
    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError:
        pass
    # Try extracting JSON from fenced blocks or raw text
    extracted = _extract_json_block(raw_response)
    if extracted:
        try:
            return json.loads(extracted)
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Failed to parse model response as JSON: {raw_response[:500]}")


@dataclass
class AgentStep:
    thought: str
    action: str
    action_input: dict[str, Any]
    observation: Any
    raw_response: str


@dataclass
class AgentRuntimeState:
    steps: list[AgentStep] = field(default_factory=list)
    succeeded: bool = False
    failure_reason: str | None = None
    answer: AnswerTable | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "steps": [
                {
                    "thought": s.thought,
                    "action": s.action,
                    "action_input": s.action_input,
                    "observation": s.observation,
                }
                for s in self.steps
            ],
            "succeeded": self.succeeded,
            "failure_reason": self.failure_reason,
        }
        if self.answer is not None:
            result["answer"] = self.answer.to_dict()
        return result


# ReAct Agent 核心类
class ReActAgent:
    def __init__(
        self,
        *,
        model: ModelAdapter,
        tools: ToolRegistry,
        config: ReActAgentConfig | None = None,
        system_prompt: str | None = None,
        wiki_context: str | None = None,
    ) -> None:
        self.model = model
        self.tools = tools
        self.config = config or ReActAgentConfig()
        self.system_prompt = system_prompt or REACT_SYSTEM_PROMPT
        self.wiki_context = wiki_context

    # 构建发送给模型的对话消息列表（包含系统提示词、任务描述和历史步骤）
    def _build_messages(self, task: PublicTask, state: AgentRuntimeState) -> list[ModelMessage]:
        system_content = build_system_prompt(
            self.tools.describe_for_prompt(),
            system_prompt=self.system_prompt,
            wiki_context=self.wiki_context,
        )
        messages = [ModelMessage(role="system", content=system_content)]
        messages.append(ModelMessage(role="user", content=build_task_prompt(task)))
        for step in state.steps:
            messages.append(ModelMessage(role="assistant", content=step.raw_response))
            messages.append(
                ModelMessage(role="user", content=build_observation_prompt(step.observation))
            )
        return messages

    def run(self, task: PublicTask, task_output_dir: Path | None = None) -> AgentRuntimeState:
        state = AgentRuntimeState()
        
        def _log(msg: str):
            if task_output_dir:
                log_file = task_output_dir / "agent.log"
                with log_file.open("a", encoding="utf-8") as f:
                    f.write(msg + "\n")

        _log(f"=== Starting Task {task.task_id} ===")
        
        # 开始 ReAct 循环：思考 -> 行动 -> 观察
        consecutive_errors = 0
        action_history: Counter[tuple[str, str]] = Counter()

        for step_index in range(1, self.config.max_steps + 1):
            _log(f"\n--- Step {step_index} ---")

            # 准备基础报错信息
            reflection_hint = ""
            if consecutive_errors >= self.config.error_reflection_threshold:
                reflection_hint = (
                    "\n\n[SYSTEM WARNING] You have encountered multiple consecutive errors. "
                    "Please carefully analyze the error messages above and your previous steps. "
                    "Ensure your output strictly follows the JSON format and tool specifications. "
                    "Rethink your current approach and correct any recurring mistakes before proceeding."
                )

            thought = ""
            action = ""
            action_input: dict[str, Any] = {}
            observation: Any = None
            raw_response = ""

            try:
                raw_response = self.model.complete(self._build_messages(task, state))
                _log(f"Model Response:\n{raw_response}")

                model_step = parse_model_step(raw_response)
                thought = model_step.get("thought", "")
                action = model_step.get("action", "")
                action_input = model_step.get("action_input", {})

                if not action or action.lower() == "finish":
                    state.succeeded = True
                    break

                # 检查死循环（重复操作）
                current_action_key = (action, json.dumps(action_input, sort_keys=True))
                action_history[current_action_key] += 1
                if action_history[current_action_key] >= self.config.max_repeated_actions:
                    observation = (
                        f"Error: Detected repeated action '{action}' with same inputs. "
                        f"Please try a different approach.{reflection_hint}"
                    )
                    consecutive_errors += 1
                else:
                    # 执行工具
                    result = self.tools.execute(task, action, action_input)
                    observation = result.content
                    if result.answer is not None:
                        state.answer = result.answer
                    if result.is_terminal:
                        state.succeeded = True
                        break
                    consecutive_errors = 0

            except Exception as e:
                observation = f"Error: {str(e)}{reflection_hint}"
                consecutive_errors += 1
                if not thought:
                    thought = "Error occurred during parsing."
                if not action:
                    action = "error"

            if consecutive_errors >= self.config.max_consecutive_errors:
                state.failure_reason = f"Max consecutive errors ({self.config.max_consecutive_errors}) reached."
                _log("Max consecutive errors reached. Aborting.")
                break

            state.steps.append(AgentStep(
                thought=thought,
                action=action,
                action_input=action_input,
                observation=observation,
                raw_response=raw_response
            ))
            _log(f"Observation: {observation}")

        if not state.succeeded and not state.failure_reason:
            state.failure_reason = f"Max steps ({self.config.max_steps}) reached without answer."

        return state
