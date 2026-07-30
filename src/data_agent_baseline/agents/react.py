from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from data_agent_baseline.agents.model import ModelAdapter, ModelMessage
from data_agent_baseline.agents.prompt import (
    build_observation_prompt,
    build_system_prompt,
    build_task_prompt,
    REACT_SYSTEM_PROMPT,
)
from data_agent_baseline.benchmark.dataset import PublicTask
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


# 解析模型的单步响应为 JSON 结构
def parse_model_step(raw_response: str) -> dict[str, Any]:
    cleaned_text = _strip_markdown_code_blocks(raw_response)
    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse model response as JSON: {cleaned_text}") from e


@dataclass
class AgentStep:
    thought: str
    action: str
    action_input: dict[str, Any]
    observation: str
    raw_response: str


@dataclass
class AgentRuntimeState:
    steps: list[AgentStep] = field(default_factory=list)


from dataclasses import field


# ReAct Agent 核心类
class ReActAgent:
    def __init__(
        self,
        *,
        model: ModelAdapter,
        tools: ToolRegistry,
        config: ReActAgentConfig | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self.model = model
        self.tools = tools
        self.config = config or ReActAgentConfig()
        self.system_prompt = system_prompt or REACT_SYSTEM_PROMPT

    # 构建发送给模型的对话消息列表（包含系统提示词、任务描述和历史步骤）
    def _build_messages(self, task: PublicTask, state: AgentRuntimeState, data_roadmap: str | None = None) -> list[ModelMessage]:
        system_content = build_system_prompt(
            self.tools.describe_for_prompt(),
            system_prompt=self.system_prompt,
            data_roadmap=data_roadmap,
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
        action_history: list[tuple[str, str]] = []
        data_roadmap = None  # Baseline 默认无路线图

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

            raw_response = self.model.complete(self._build_messages(task, state, data_roadmap=data_roadmap))
            _log(f"Model Response:\n{raw_response}")
            try:
                model_step = parse_model_step(raw_response)
                thought = model_step.get("thought", "")
                action = model_step.get("action", "")
                action_input = model_step.get("action_input", {})

                if not action or action.lower() == "finish":
                    break

                # 检查死循环（重复操作）
                current_action_key = (action, json.dumps(action_input, sort_keys=True))
                action_history.append(current_action_key)
                if action_history.count(current_action_key) >= self.config.max_repeated_actions:
                    observation = f"Error: Detected repeated action '{action}' with same inputs. Please try a different approach."
                    consecutive_errors += 1
                else:
                    # 执行工具
                    observation = self.tools.call(action, action_input, task_context_dir=task.context_dir)
                    consecutive_errors = 0

            except Exception as e:
                observation = f"Error: {str(e)}{reflection_hint}"
                consecutive_errors += 1
                thought = "Error occurred during parsing or execution."
                action = "error"
                action_input = {}

            if consecutive_errors >= self.config.max_consecutive_errors:
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

        return state
