"""ReAct Agent 主循环驱动器。

本模块只剩两件事：循环预算配置（`ReActAgentConfig`）和驱动主循环本身
（调模型、把响应转成事件、记账、判终止）。
具体的协议解析、工具派发、事件序列化分别下沉到：
- `agents.llm.protocol.adapt_model_response`：规整 ModelResponse → 统一 `ProtocolTurn`
- `agents.tools.dispatcher.dispatch_tool_call`：执行单个 tool_call，返回 `(result, should_terminate)`
- `agents.runtime.recorder.record_step_event`：把事件序列化成 `StepRecord` 并通知 step_callback
- `agents.runtime.messages` / `agents.runtime.media`：构造 system/user/replay messages
- `agents.runtime.budget`：预算压力提示与 final-step guardrail
- `agents.verification.terminal_policy`：answer verifier 拒绝与 fallback answer 策略

协议解析 helper 由各子模块自行导出，测试直接从子模块 import。
"""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

from agents.benchmark.schema import PublicTask
from agents.llm import (
    ModelAdapter,
    ModelMessage,
    ModelToolCall,
    TokenUsage,
    ToolCallParseError,
    adapt_model_response,
)
from agents.prompts import build_task_prompt
from agents.runtime import (
    AgentRunResult,
    AgentRuntimeState,
    EmptyToolCallsEvent,
    ModelErrorEvent,
    ObservationEvent,
    ParseErrorEvent,
    StepCallback,
    ToolErrorEvent,
    build_budget_prompt,
    build_messages,
    build_system_message,
    compute_budget_status,
    final_step_block_error,
    is_final_step_blocking_active,
    record_step_event,
    serialize_message,
    serialize_response,
    should_block_non_answer_final_turn,
    synthesize_raw_tool_calls,
)
from agents.tools import ToolRegistry, dispatch_tool_call
from agents.verification import AnswerVerifier, TerminalAnswerPolicy


def _has_mixed_terminal_calls(calls: Sequence[ModelToolCall], tools: ToolRegistry) -> bool:
    """同一 turn 是否混合了终止与非终止调用。

    混合轮里的终止调用（answer/report）参数是在看到本轮 observation 之前盲写的；
    vLLM 不尊重 `parallel_tool_calls` 请求字段，无法在请求侧阻止，只能在派发层拒绝。
    未知工具名按非终止处理（随后会走 KeyError → tool_error 路径）。
    """
    flags = [
        (defn := tools.definitions.get(call.name)) is not None and defn.is_terminal
        for call in calls
    ]
    return any(flags) and not all(flags)


__all__ = [
    "ReActAgent",
    "ReActAgentConfig",
    "StepCallback",
]


@dataclass(frozen=True, slots=True)
class ReActAgentConfig:
    """ReAct 循环运行参数。

    仅暴露循环预算，其余行为（prompt / 工具集）通过构造参数注入，
    让本类对评测脚本/单元测试都易替换。

    `max_steps` 是**有效模型轮数**上限。每轮非空 tool_calls 可能并行发出多个
    工具调用，因此实际工具调用次数可能高于 max_steps。

    `max_empty_tool_call_retries` 用于 response.tool_calls 为空的情况。
    这类响应只产生纠偏 observation，不应消耗业务 max_steps；单独设上限避免无限重试。
    """

    max_steps: int = 16
    max_empty_tool_call_retries: int = 8
    enable_preact: bool = False
    stop_after_final_step_retries: bool = False


class ReActAgent:
    """ReAct 范式的 Agent 实例。

    构造时注入 model 和 tools 允许在不同场景复用同一套循环逻辑：
    - 线上：NativeToolsOpenAIAdapter + 默认 ToolRegistry
    - 离线测试：Scripted[Native]ModelAdapter + mock ToolRegistry
    """

    def __init__(
        self,
        *,
        model: ModelAdapter,
        tools: ToolRegistry,
        config: ReActAgentConfig | None = None,
        system_prompt: str | None = None,
        step_callback: StepCallback | None = None,
        answer_verifier: AnswerVerifier | None = None,
    ) -> None:
        self.model = model
        self.tools = tools
        self.config = config or ReActAgentConfig()
        self.system_prompt = system_prompt
        self.step_callback = step_callback
        self.answer_verifier = answer_verifier

    # ----------------------------- tracing 辅助 -----------------------------

    @staticmethod
    def _open_turn_span(turn_index: int) -> Any:
        try:
            from agents.tracing.create import get_current_trace, turn_span

            if get_current_trace() is None:
                return None
            span = turn_span(turn=turn_index, agent_name="ReActAgent")
            span.start(mark_as_current=True)
            return span
        except Exception:
            return None

    @staticmethod
    def _close_turn_span(span: Any) -> None:
        if span is None:
            return
        with contextlib.suppress(Exception):
            span.finish(reset_current=True)

    # ----------------------------- 消息构造 -----------------------------

    def _system_message(self) -> ModelMessage:
        """构造 system prompt。"""
        return build_system_message(
            tools=self.tools,
            system_prompt=self.system_prompt,
        )

    def _build_messages(
        self,
        state: AgentRuntimeState,
        initial_content: str | list[dict[str, Any]],
    ) -> list[ModelMessage]:
        return build_messages(
            state=state,
            initial_content=initial_content,
            tools=self.tools,
            system_prompt=self.system_prompt,
        )

    # ----------------------------- 主循环 -----------------------------

    def run(
        self,
        task: PublicTask,
        *,
        initial_user_content: str | list[dict[str, Any]] | None = None,
    ) -> AgentRunResult:
        """驱动 ReAct 循环解决单个任务。

        循环不变式：
        - 每轮至少 append 一条 StepRecord（成功 / 工具异常 / 解析 __error__），用于 trace 可观测
        - 调用 answer 工具会设置 `state.answer` 并立刻 break
        - native 空 tool_calls 只消耗 empty retry budget，不消耗业务 max_steps
        - 达到 max_steps 未 answer 时，写入 failure_reason 并正常返回

        `initial_user_content`: 子代理（video）可注入自定义首条消息；
        None 走默认任务提示词路径。
        """
        state = AgentRuntimeState()
        initial_content = (
            initial_user_content
            if initial_user_content is not None
            else build_task_prompt(task, preact=self.config.enable_preact)
        )
        step_counter = 0
        effective_turns = 0
        empty_tool_call_retries = 0
        turn_index = 0
        terminated = False
        budget_warning_fired = False
        last_step_retries = 0
        terminal_policy = TerminalAnswerPolicy(self.answer_verifier)
        terminal_tool_names = tuple(
            name for name, defn in self.tools.definitions.items() if defn.is_terminal
        )

        _turn_span = None
        while effective_turns < self.config.max_steps:
            turn_index += 1
            self._close_turn_span(_turn_span)
            _turn_span = self._open_turn_span(turn_index)

            # ---- 1. 调用模型 ----
            messages = self._build_messages(state, initial_content)

            budget_status = compute_budget_status(
                effective_turns=effective_turns,
                max_steps=self.config.max_steps,
            )
            budget_prompt, budget_warning_fired = build_budget_prompt(
                status=budget_status,
                effective_turns=effective_turns,
                max_steps=self.config.max_steps,
                warning_fired=budget_warning_fired,
                terminal_tool_names=terminal_tool_names,
            )
            if budget_prompt is not None:
                messages.append(budget_prompt)

            # 提前把 messages 序列化好：即便 model.complete 抛错，也能把当时发出的
            # prompt_messages 落进 ModelErrorEvent，方便复盘"哪条请求把 LLM 炸了"
            serialized_prompt = [serialize_message(message) for message in messages]
            try:
                # 每轮下推自己手里的 registry：请求体工具广告与执行分发同源
                response = self.model.complete(messages, tools=self.tools)
            except ToolCallParseError as exc:
                # tool_call arguments JSON 解析失败：携带部分响应，保留
                # raw_tool_calls / usage / reasoning 供 trace 和回放
                partial = exc.partial_response
                effective_turns += 1
                step_counter += 1
                record_step_event(
                    state,
                    ParseErrorEvent(
                        error=f"model.complete failed: {exc}",
                        raw_response=partial.raw_response,
                        reasoning=partial.reasoning_content,
                        usage=partial.usage,
                        prompt_messages=serialized_prompt,
                        response_message=serialize_response(partial),
                    ),
                    step_index=step_counter,
                    turn_index=turn_index,
                    callback=self.step_callback,
                )
                continue
            except Exception as exc:
                # 网络错误 / API 失败：记一条 model_error，让循环在下一轮重试
                effective_turns += 1
                step_counter += 1
                record_step_event(
                    state,
                    ModelErrorEvent(
                        error=f"model.complete failed: {exc}",
                        prompt_messages=serialized_prompt,
                    ),
                    step_index=step_counter,
                    turn_index=turn_index,
                    callback=self.step_callback,
                )
                continue

            # ---- 2. 协议规整 ----
            serialized_response = serialize_response(response)
            turn = adapt_model_response(
                response,
                turn_index=turn_index,
                allowed_tool_names=self.tools.definitions,
            )

            if not turn.calls:
                # native 协议下模型回了空 tool_calls：记纠偏 observation，独立 retry budget
                empty_tool_call_retries += 1
                step_counter += 1
                record_step_event(
                    state,
                    EmptyToolCallsEvent(
                        thought=turn.thought,
                        raw_response=response.raw_response,
                        reasoning=response.reasoning_content,
                        usage=response.usage,
                        drafted_tool_call=turn.reasoning_draft,
                        prompt_messages=serialized_prompt,
                        response_message=serialized_response,
                    ),
                    step_index=step_counter,
                    turn_index=turn_index,
                    callback=self.step_callback,
                )
                if empty_tool_call_retries >= self.config.max_empty_tool_call_retries:
                    state.failure_reason = (
                        "Agent exceeded max_empty_tool_call_retries "
                        f"({self.config.max_empty_tool_call_retries}) without an "
                        "executable tool call."
                    )
                    terminated = True
                    break
                continue

            # ---- 3. 依次派发 tool_calls ----

            reject_terminal_this_turn = _has_mixed_terminal_calls(turn.calls, self.tools)

            # Last-step guardrail: on the final step of a long-running task
            # (past the 90% CRITICAL threshold), block non-answer tools and
            # give the model one free retry to call `answer`. The blocked
            # turn does NOT consume effective_turns budget.
            # 末步混合轮走免费重试而非逐调用拒绝，否则模型在预算最后一步
            # "带工具的 answer" 会被拒掉并烧光预算。
            if should_block_non_answer_final_turn(
                status=budget_status,
                last_step_retries=last_step_retries,
                calls=turn.calls,
                terminal_tool_names=terminal_tool_names,
            ) or (
                reject_terminal_this_turn
                and is_final_step_blocking_active(
                    status=budget_status, last_step_retries=last_step_retries
                )
            ):
                last_step_retries += 1
                step_counter += 1
                record_step_event(
                    state,
                    ToolErrorEvent(
                        thought=turn.thought,
                        arguments={},
                        error=final_step_block_error(terminal_tool_names=terminal_tool_names),
                        raw_response=response.raw_response,
                        tool_call_id=turn.calls[0].id if turn.calls else None,
                        raw_tool_calls=response.raw_tool_calls,
                        reasoning=response.reasoning_content,
                        usage=response.usage,
                        prompt_messages=serialized_prompt,
                        response_message=serialized_response,
                    ),
                    step_index=step_counter,
                    turn_index=turn_index,
                    callback=self.step_callback,
                )
                if self.config.stop_after_final_step_retries and last_step_retries >= 2:
                    state.failure_reason = final_step_block_error(
                        terminal_tool_names=terminal_tool_names
                    )
                    terminated = True
                    break
                continue

            effective_turns += 1
            # recovered tool call（从 content/reasoning 解析）没有原始 raw_tool_calls，
            # 需要合成以便 message 回放时 assistant.tool_calls 与 tool.tool_call_id 匹配
            effective_raw_tool_calls: list[dict[str, object]] = response.raw_tool_calls or []
            if not effective_raw_tool_calls and turn.calls:
                effective_raw_tool_calls = synthesize_raw_tool_calls(turn.calls)
            replay_raw_tool_calls = effective_raw_tool_calls
            verifier_rejected_this_turn = False
            turn_first_state_index = len(state.steps)
            for call_idx, tool_call in enumerate(turn.calls):
                step_counter += 1
                is_lead = call_idx == 0
                step_tool_call_id: str | None = tool_call.id or None
                step_raw_tool_calls: list[dict[str, object]] | None = (
                    replay_raw_tool_calls if is_lead else None
                )
                # usage / reasoning / prompt_messages / response_message 与 raw_tool_calls
                # 同约定：仅 lead 承载，避免双计 + 节省体积
                step_reasoning = response.reasoning_content if is_lead else ""
                step_usage = response.usage if is_lead else TokenUsage()
                step_prompt_messages = serialized_prompt if is_lead else None
                step_response_message = serialized_response if is_lead else None

                # 终止工具预验证：在 handler 执行（会删 CSV artifact）之前跑 verifier
                defn = self.tools.definitions.get(tool_call.name)

                if reject_terminal_this_turn and defn is not None and defn.is_terminal:
                    record_step_event(
                        state,
                        ToolErrorEvent(
                            thought=turn.thought,
                            arguments=tool_call.arguments,
                            error=(
                                f"{tool_call.name} rejected: it was issued in the same "
                                "turn as non-terminal tool calls, so its arguments were "
                                "written before seeing this turn's observations. Review "
                                f"the observations, then call {tool_call.name} again in "
                                "its own turn."
                            ),
                            raw_response=response.raw_response,
                            tool_call_id=step_tool_call_id,
                            raw_tool_calls=step_raw_tool_calls,
                            reasoning=step_reasoning,
                            usage=step_usage,
                            prompt_messages=step_prompt_messages,
                            response_message=step_response_message,
                        ),
                        step_index=step_counter,
                        turn_index=turn_index,
                        callback=self.step_callback,
                    )
                    continue

                rejection = terminal_policy.maybe_reject(
                    task=task,
                    arguments=tool_call.arguments,
                    is_terminal=defn is not None and defn.is_terminal,
                    remaining_steps=budget_status.remaining,
                    state=state,
                )
                if rejection is not None:
                    # Replay requires one tool message for every assistant
                    # tool_call. If a rejection cuts the turn short, omit later
                    # unexecuted tool calls from replay metadata.
                    replay_raw_tool_calls = effective_raw_tool_calls[: call_idx + 1]
                    if is_lead:
                        step_raw_tool_calls = replay_raw_tool_calls
                    elif turn_first_state_index < len(state.steps):
                        lead_step = state.steps[turn_first_state_index]
                        state.steps[turn_first_state_index] = replace(
                            lead_step,
                            raw_tool_calls=replay_raw_tool_calls,
                        )
                    record_step_event(
                        state,
                        ObservationEvent(
                            thought=turn.thought,
                            tool_name=tool_call.name,
                            arguments=tool_call.arguments,
                            content=rejection.content,
                            ok=True,
                            raw_response=response.raw_response,
                            tool_call_id=step_tool_call_id,
                            raw_tool_calls=step_raw_tool_calls,
                            reasoning=step_reasoning,
                            usage=step_usage,
                            prompt_messages=step_prompt_messages,
                            response_message=step_response_message,
                        ),
                        step_index=step_counter,
                        turn_index=turn_index,
                        callback=self.step_callback,
                    )
                    verifier_rejected_this_turn = True
                    break

                try:
                    result, should_terminate = dispatch_tool_call(task, tool_call, self.tools)
                except Exception as exc:
                    record_step_event(
                        state,
                        ToolErrorEvent(
                            thought=turn.thought,
                            arguments=tool_call.arguments,
                            error=str(exc),
                            raw_response=response.raw_response,
                            tool_call_id=step_tool_call_id,
                            raw_tool_calls=step_raw_tool_calls,
                            reasoning=step_reasoning,
                            usage=step_usage,
                            prompt_messages=step_prompt_messages,
                            response_message=step_response_message,
                        ),
                        step_index=step_counter,
                        turn_index=turn_index,
                        callback=self.step_callback,
                    )
                    continue

                if should_terminate:
                    state.answer = result.answer
                record_step_event(
                    state,
                    ObservationEvent(
                        thought=turn.thought,
                        tool_name=tool_call.name,
                        arguments=tool_call.arguments,
                        content=result.content,
                        ok=result.ok,
                        raw_response=response.raw_response,
                        tool_call_id=step_tool_call_id,
                        raw_tool_calls=step_raw_tool_calls,
                        reasoning=step_reasoning,
                        usage=step_usage,
                        prompt_messages=step_prompt_messages,
                        response_message=step_response_message,
                    ),
                    step_index=step_counter,
                    turn_index=turn_index,
                    callback=self.step_callback,
                )
                if should_terminate:
                    terminated = True
                    break

            if terminated:
                break
            if verifier_rejected_this_turn:
                continue

        self._close_turn_span(_turn_span)

        if not terminated and effective_turns >= self.config.max_steps:
            state.failure_reason = "Agent did not submit an answer within max_steps."

        # 循环正常结束但没拿到 answer：判为 max_steps 耗尽
        if state.answer is None and state.failure_reason is None:
            state.failure_reason = "Agent did not submit an answer within max_steps."

        # Fallback: use the latest valid verifier-rejected answer rather than scoring 0.
        terminal_policy.promote_fallback_answer(task=task, state=state)

        return AgentRunResult(
            task_id=task.task_id,
            answer=state.answer,
            # list(...) 拷贝切断 state.steps 与返回值之间的引用，防止后续修改串扰
            steps=list(state.steps),
            failure_reason=state.failure_reason,
        )
