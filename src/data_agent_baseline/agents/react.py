"""
此模块实现了 ReAct (Reasoning and Acting) 代理逻辑，负责指导模型进行思考、行动和观察的循环。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from data_agent_baseline.agents.model import ModelAdapter, ModelMessage, ModelStep
from data_agent_baseline.agents.prompt import (
    REACT_SYSTEM_PROMPT,
    build_observation_prompt,
    build_system_prompt,
    build_task_prompt,
)
from data_agent_baseline.agents.runtime import AgentRunResult, AgentRuntimeState, StepRecord
from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.tools.registry import ToolRegistry

# 定义 ReAct Agent 的配置
@dataclass(frozen=True, slots=True)
class ReActAgentConfig:
    # 最大步数，防止 Agent 进入无限循环
    max_steps: int = 50


# 清理模型输出中的 Markdown 代码块（JSON 围栏）
def _strip_json_fence(raw_response: str) -> str:
    text = raw_response.strip()
    fence_match = re.search(r"```json\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fence_match is not None:
        return fence_match.group(1).strip()
    generic_fence_match = re.search(r"```\s*(.*?)\s*```", text, flags=re.DOTALL)
    if generic_fence_match is not None:
        return generic_fence_match.group(1).strip()
    return text


# 将文本解析为单个 JSON 对象
def _load_single_json_object(text: str) -> dict[str, object]:
    payload, end = json.JSONDecoder().raw_decode(text)
    remainder = text[end:].strip()
    if remainder:
        cleaned_remainder = re.sub(r"(?:\\[nrt])+", "", remainder).strip()
        if cleaned_remainder:
            raise ValueError("Model response must contain only one JSON object.")
    if not isinstance(payload, dict):
        raise ValueError("Model response must be a JSON object.")
    return payload


# 解析模型输出的一步，提取 Thought、Action 和 Action Input
def parse_model_step(raw_response: str) -> ModelStep:
    normalized = _strip_json_fence(raw_response)
    payload = _load_single_json_object(normalized)

    thought = payload.get("thought", "")
    action = payload.get("action")
    action_input = payload.get("action_input", {})
    if not isinstance(thought, str):
        raise ValueError("thought must be a string.")
    if not isinstance(action, str) or not action:
        raise ValueError("action must be a non-empty string.")
    if not isinstance(action_input, dict):
        raise ValueError("action_input must be a JSON object.")

    return ModelStep(
        thought=thought,
        action=action,
        action_input=action_input,
        raw_response=raw_response,
    )


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
    def _build_messages(self, task: PublicTask, state: AgentRuntimeState) -> list[ModelMessage]:
        system_content = build_system_prompt(
            self.tools.describe_for_prompt(),
            system_prompt=self.system_prompt,
        )
        messages = [ModelMessage(role="system", content=system_content)]
        messages.append(ModelMessage(role="user", content=build_task_prompt(task)))
        for step in state.steps:
            messages.append(ModelMessage(role="assistant", content=step.raw_response))
            messages.append(
                ModelMessage(role="user", content=build_observation_prompt(step.observation))
            )
        return messages

    # 启动 Agent 解决指定的任务
    def run(self, task: PublicTask, task_output_dir: Path | None = None) -> AgentRunResult:
        state = AgentRuntimeState()
        
        def _log(msg: str):
            if task_output_dir:
                log_file = task_output_dir / "agent.log"
                with log_file.open("a", encoding="utf-8") as f:
                    f.write(msg + "\n")

        _log(f"=== Starting Task {task.task_id} ===")
        
        # 开始 ReAct 循环：思考 -> 行动 -> 观察
        for step_index in range(1, self.config.max_steps + 1):
            _log(f"\n--- Step {step_index} ---")
            raw_response = self.model.complete(self._build_messages(task, state))
            _log(f"Model Response:\n{raw_response}")
            try:
                model_step = parse_model_step(raw_response)
            except Exception as exc:
                observation = {
                    "ok": False,
                    "error": f"Failed to parse response: {exc}. Please check your JSON format, ensure you output exactly one complete JSON block without being truncated, and try again.",
                }
                _log(f"Parse Error:\n{observation['error']}")
                state.steps.append(
                    StepRecord(
                        step_index=step_index,
                        thought="",
                        action="__error__",
                        action_input={},
                        raw_response=raw_response,
                        observation=observation,
                        ok=False,
                    )
                )
                continue

            try:
                tool_result = self.tools.execute(task, model_step.action, model_step.action_input)
                observation = {
                    "ok": tool_result.ok,
                    "tool": model_step.action,
                    "content": tool_result.content,
                }
                _log(f"Tool Result ({model_step.action}):\n{json.dumps(observation, ensure_ascii=False, indent=2)}")
                step_record = StepRecord(
                    step_index=step_index,
                    thought=model_step.thought,
                    action=model_step.action,
                    action_input=model_step.action_input,
                    raw_response=raw_response,
                    observation=observation,
                    ok=tool_result.ok,
                )
                state.steps.append(step_record)
                if tool_result.is_terminal:
                    _log("Terminal tool called. Ending loop.")
                    state.answer = tool_result.answer
                    break
            except Exception as exc:
                observation = {
                    "ok": False,
                    "error": f"Tool execution failed: {exc}. Please check your action_input and try again.",
                }
                _log(f"Tool Error ({model_step.action}):\n{observation['error']}")
                state.steps.append(
                    StepRecord(
                        step_index=step_index,
                        thought=model_step.thought,
                        action=model_step.action,
                        action_input=model_step.action_input,
                        raw_response=raw_response,
                        observation=observation,
                        ok=False,
                    )
                )

        if state.answer is None and state.failure_reason is None:
            state.failure_reason = "Agent did not submit an answer within max_steps."
            _log(f"\n=== Task Failed: {state.failure_reason} ===")
        else:
            _log(f"\n=== Task Finished ===")

        return AgentRunResult(
            task_id=task.task_id,
            answer=state.answer,
            steps=list(state.steps),
            failure_reason=state.failure_reason,
        )
