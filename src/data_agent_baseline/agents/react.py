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
    max_steps: int = 35
    # 连续报错达到多少次时进行反思提醒
    error_reflection_threshold: int = 3
    # 连续报错达到多少次时停止任务（熔断）
    max_consecutive_errors: int = 6
    # 连续执行相同操作多少次时判定为死循环
    max_repeated_actions: int = 3


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


# 将文本解析为 JSON 对象
def _load_json_object(text: str) -> dict[str, object]:
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("Model response must be a JSON object.")
    return payload


# 解析模型输出的一步，提取 Thought、Action 和 Action Input
def parse_model_step(raw_response: str) -> ModelStep:
    normalized = _strip_json_fence(raw_response)
    payload = _load_json_object(normalized)

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

    # 启动 Agent 解决指定的任务
    def run(self, task: PublicTask, task_output_dir: Path | None = None) -> AgentRunResult:
        state = AgentRuntimeState()
        
        def _log(msg: str):
            if task_output_dir:
                log_file = task_output_dir / "agent.log"
                with log_file.open("a", encoding="utf-8") as f:
                    f.write(msg + "\n")

        _log(f"=== Starting Task {task.task_id} ===")

        # --- 组合导航器 A: KG + PageRAG ---
        from data_agent_baseline.agents.db_navigator import get_data_roadmap
        from data_agent_baseline.agents.pagerag import PageRAGNavigator
        
        # 1. 获取全局结构图谱 (KG)
        kg_roadmap = get_data_roadmap(task.context_dir)
        
        # 2. 获取文档目录与相关页 (PageRAG)
        pagerag = PageRAGNavigator(task.context_dir)
        doc_catalog = pagerag.get_catalog()
        retrieved_pages = pagerag.retrieve(task.question, top_k=3)
        
        # 合并路线图
        data_roadmap = kg_roadmap
        if doc_catalog:
            data_roadmap += "\n" + doc_catalog
        if retrieved_pages:
            data_roadmap += "\n" + retrieved_pages
            
        _log("Hybrid Navigator (KG + PageRAG) initialized and injected.")
        
        # 开始 ReAct 循环：思考 -> 行动 -> 观察
        consecutive_errors = 0
        action_history: list[tuple[str, str]] = []

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
            except Exception as exc:
                observation = {
                    "ok": False,
                    "error": f"Failed to parse response: {exc}. Please check your JSON format, ensure you output exactly one complete JSON block without being truncated, and try again.{reflection_hint}",
                }
                _log(f"Parse Error (Consecutive: {consecutive_errors + 1}):\n{observation['error']}")
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
                consecutive_errors += 1
                if consecutive_errors >= self.config.max_consecutive_errors:
                    state.failure_reason = f"Agent failed with {consecutive_errors} consecutive parse errors despite reflection warnings."
                    _log(f"Circuit breaker triggered: {state.failure_reason}")
                    break
                continue

            # 检测重复行为
            current_action = (model_step.action, json.dumps(model_step.action_input, sort_keys=True))
            action_history.append(current_action)
            if len(action_history) >= self.config.max_repeated_actions:
                last_n = action_history[-self.config.max_repeated_actions:]
                if all(a == last_n[0] for a in last_n):
                    state.failure_reason = f"Agent is stuck in a loop (repeated action: {model_step.action} 3 times)."
                    _log(f"Loop detection triggered: {state.failure_reason}")
                    break

            try:
                tool_result = self.tools.execute(task, model_step.action, model_step.action_input)
                
                # 如果报错，加入反思提醒
                obs_error = ""
                if not tool_result.ok:
                    consecutive_errors += 1
                    obs_error = reflection_hint
                else:
                    consecutive_errors = 0 # 只要有一次成功，就重置连续错误计数
                
                observation = {
                    "ok": tool_result.ok,
                    "tool": model_step.action,
                    "content": tool_result.content,
                }
                if not tool_result.ok:
                    observation["error_hint"] = obs_error # 注入反思消息

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

                if consecutive_errors >= self.config.max_consecutive_errors:
                    state.failure_reason = f"Agent failed with {consecutive_errors} consecutive tool/parse errors."
                    _log(f"Circuit breaker triggered: {state.failure_reason}")
                    break

                if tool_result.is_terminal:
                    _log("Terminal tool called. Ending loop.")
                    state.answer = tool_result.answer
                    break
            except Exception as exc:
                consecutive_errors += 1
                observation = {
                    "ok": False,
                    "error": f"Tool execution failed: {exc}. Please check your action_input and try again.{reflection_hint}",
                }
                _log(f"Tool Error ({model_step.action}, Consecutive: {consecutive_errors}):\n{observation['error']}")
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
                if consecutive_errors >= self.config.max_consecutive_errors:
                    state.failure_reason = f"Agent failed with {consecutive_errors} consecutive tool execution errors."
                    _log(f"Circuit breaker triggered: {state.failure_reason}")
                    break

        if state.answer is None and state.failure_reason is None:
            state.failure_reason = "Agent did not submit an answer within max_steps."
            _log(f"\n=== Task Failed: {state.failure_reason} ===")
        elif state.failure_reason:
            _log(f"\n=== Task Failed: {state.failure_reason} ===")
        else:
            _log(f"\n=== Task Finished ===")

        return AgentRunResult(
            task_id=task.task_id,
            answer=state.answer,
            steps=list(state.steps),
            failure_reason=state.failure_reason,
        )
