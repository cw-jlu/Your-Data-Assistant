"""PhasedReActAgent — Fujitsu-style 4-phase agent loop.

Single conversation thread with phase tracking. Each phase exposes a
focused system prompt and a whitelisted tool set; calls to non-whitelisted
tools return [ERROR] observations rather than executing.

Phases (auto-transition):
  PLAN     →  EXPLORE  on first non-error turn (= after structured plan)
  EXPLORE  →  ANSWER   on `answer_from_sql` call (post 3+ explore queries)
  ANSWER   →  VERIFY   on `answer_from_sql` returning review_required
  VERIFY   →  done     on `confirm_answer`
                       (or rewrite via `answer_from_sql` keeps it in VERIFY)
"""
from __future__ import annotations

from dataclasses import dataclass

from kobushi_core.model import ModelAdapter, ModelMessage
from kobushi_core.benchmark.schema import PublicTask

from experiments.exp_122_column_auditor.agent import (
    parse_model_step,
)
from experiments.exp_122_column_auditor.prompt import (
    PHASE_NAME_TO_TOOLS,
    build_observation_prompt,
    build_phased_system_prompt,
    build_task_prompt,
)
from experiments.exp_122_column_auditor.runtime import (
    AgentRunResult,
    AgentRuntimeState,
    StepRecord,
)
from experiments.exp_122_column_auditor.tools.registry import ToolRegistry


PHASES_ORDER = ("plan", "explore", "answer", "verify")
PHASE_TO_INDEX = {p: i for i, p in enumerate(PHASES_ORDER)}
TRANSITION_ACTION = "complete_phase"


@dataclass(frozen=True, slots=True)
class PhasedAgentConfig:
    max_steps: int = 64
    min_explore_queries: int = 3  # min successful EXPLORE queries before complete_phase to answer


def _validate_action(
    current: str, action: str, action_input: dict,
    n_explore_ok: int, cfg: PhasedAgentConfig
) -> tuple[bool, str, str | None, str]:
    """Pre-execute validation. Returns (allow, new_phase_if_transition, error_msg, virtual_action).

    `virtual_action` is "transition" if action == complete_phase (= no tool dispatch),
    otherwise "tool" (= execute via registry as usual).
    """
    allowed = PHASE_NAME_TO_TOOLS[current]
    if action not in allowed:
        return False, current, (
            f"Tool '{action}' is not allowed in phase {current.upper()}. "
            f"Allowed: {list(allowed)}. Continue with one of the allowed tools."
        ), "tool"

    # Handle complete_phase as a virtual transition action.
    if action == TRANSITION_ACTION:
        next_phase = action_input.get("next_phase") if isinstance(action_input, dict) else None
        # Default next-phase if missing: advance one step in PHASES_ORDER.
        if next_phase is None or next_phase not in PHASES_ORDER:
            cur_idx = PHASE_TO_INDEX[current]
            if cur_idx + 1 >= len(PHASES_ORDER):
                return False, current, (
                    "complete_phase: no next phase from VERIFY. "
                    "Use confirm_answer to commit, or answer_from_sql to rewrite."
                ), "transition"
            next_phase = PHASES_ORDER[cur_idx + 1]

        # Check phase progression direction (= forbid going backward).
        if PHASE_TO_INDEX[next_phase] <= PHASE_TO_INDEX[current]:
            return False, current, (
                f"complete_phase: cannot go from {current.upper()} → "
                f"{next_phase.upper()} (backwards or same). "
                f"Allowed targets from {current.upper()}: "
                f"{[p for p in PHASES_ORDER if PHASE_TO_INDEX[p] > PHASE_TO_INDEX[current]]}."
            ), "transition"

        # Phase-specific gates on transition.
        if current == "explore" and next_phase == "answer":
            if n_explore_ok < cfg.min_explore_queries:
                return False, current, (
                    f"complete_phase to ANSWER is blocked: only {n_explore_ok} "
                    f"successful explore queries (need {cfg.min_explore_queries}). "
                    "Issue more describe_data / execute_sql / read_doc calls."
                ), "transition"

        return True, next_phase, None, "transition"

    # Regular tool — allowed by whitelist.
    return True, current, None, "tool"


class PhasedReActAgent:
    def __init__(
        self,
        *,
        model: ModelAdapter,
        tools: ToolRegistry,
        config: PhasedAgentConfig | None = None,
        preamble: str | None = None,
        collect_logprobs: bool = False,
    ) -> None:
        self.model = model
        self.tools = tools
        self.config = config or PhasedAgentConfig()
        self.preamble = preamble
        self.collect_logprobs = collect_logprobs
        # Optional per-instance trace log path (= overrides env var).
        # Useful for parallel benches where each attempt needs its own trace.
        self.trace_log_path: str | None = None

    def _build_messages(
        self, task: PublicTask, state: AgentRuntimeState, phase: str
    ) -> list[ModelMessage]:
        # Phase-specific system prompt rebuilt every turn (= prompt changes
        # with phase). Prior conversation history is preserved.
        system_content = build_phased_system_prompt(
            phase, self.tools.describe_for_prompt()
        )
        messages = [ModelMessage(role="system", content=system_content)]
        task_prompt = build_task_prompt(task)
        if self.preamble:
            task_prompt = f"{self.preamble}\n\n---\n\n{task_prompt}"
        messages.append(ModelMessage(role="user", content=task_prompt))
        for step in state.steps:
            messages.append(ModelMessage(role="assistant", content=step.raw_response))
            messages.append(
                ModelMessage(role="user", content=build_observation_prompt(step.observation))
            )
        return messages

    def run(self, task: PublicTask) -> AgentRunResult:
        import os
        # Per-instance path takes precedence over the env var (= for parallel
        # benches where each attempt writes to its own file).
        trace_log = self.trace_log_path or os.environ.get("KOBUSHI_TRACE_LOG")

        def _log(msg: str) -> None:
            if trace_log:
                with open(trace_log, "a") as f:
                    f.write(msg + "\n")

        state = AgentRuntimeState()
        phase = "plan"
        n_explore_ok = 0  # successful EXPLORE-phase queries (for min gate)

        use_logprobs = bool(self.collect_logprobs) and hasattr(
            self.model, "complete_with_logprobs"
        )

        for step_index in range(1, self.config.max_steps + 1):
            messages = self._build_messages(task, state, phase)
            if use_logprobs:
                raw_response, top_buf = self.model.complete_with_logprobs(messages)
                if top_buf:
                    state.logprobs_buffer.extend(top_buf)
            else:
                raw_response = self.model.complete(messages)

            try:
                model_step = parse_model_step(raw_response)
                _log(
                    f"[{task.task_id}] phase={phase} step {step_index} "
                    f"action={model_step.action} "
                    f"thought={(model_step.thought or '')[:160].replace(chr(10),' | ')}"
                )

                # Validate action (whitelist + phase-specific gates + transition).
                allow, new_phase, reject_msg, kind = _validate_action(
                    current=phase,
                    action=model_step.action,
                    action_input=model_step.action_input,
                    n_explore_ok=n_explore_ok,
                    cfg=self.config,
                )
                if not allow:
                    _log(f"[{task.task_id}] phase={phase} REJECT {model_step.action}: {reject_msg}")
                    state.steps.append(
                        StepRecord(
                            step_index=step_index,
                            thought=model_step.thought,
                            action=model_step.action,
                            action_input=model_step.action_input,
                            raw_response=raw_response,
                            observation={
                                "ok": False,
                                "phase": phase,
                                "error": f"[PHASE-GATE] {reject_msg}",
                            },
                            ok=False,
                        )
                    )
                    continue

                if kind == "transition":
                    # Virtual action: just transition phase, no tool dispatch.
                    _log(f"[{task.task_id}] phase {phase} → {new_phase} (via complete_phase)")
                    state.steps.append(
                        StepRecord(
                            step_index=step_index,
                            thought=model_step.thought,
                            action=model_step.action,
                            action_input=model_step.action_input,
                            raw_response=raw_response,
                            observation={
                                "ok": True,
                                "phase": phase,
                                "transition": {"from": phase, "to": new_phase},
                                "content": f"Phase advanced: {phase} → {new_phase}.",
                            },
                            ok=True,
                        )
                    )
                    phase = new_phase
                    continue

                # Execute the tool (regular dispatch).
                tool_result = self.tools.execute(
                    task, model_step.action, model_step.action_input
                )
                observation = {
                    "ok": tool_result.ok,
                    "phase": phase,
                    "tool": model_step.action,
                    "content": tool_result.content,
                }
                _content_str = str(tool_result.content)[:300].replace("\n", " | ")
                _log(
                    f"[{task.task_id}] phase={phase} step {step_index} "
                    f"result ok={tool_result.ok} terminal={tool_result.is_terminal} "
                    f"content={_content_str}"
                )

                state.steps.append(
                    StepRecord(
                        step_index=step_index,
                        thought=model_step.thought,
                        action=model_step.action,
                        action_input=model_step.action_input,
                        raw_response=raw_response,
                        observation=observation,
                        ok=tool_result.ok,
                    )
                )

                # Update explore counter.
                if phase == "explore" and tool_result.ok and model_step.action in (
                    "describe_data", "execute_sql", "read_doc", "list_context"
                ):
                    n_explore_ok += 1

                # AUTO-TRANSITION: ANSWER → VERIFY after answer_from_sql returns
                # review_required. Submission-then-review is a forced protocol
                # flow (= the agent shouldn't need to declare it manually).
                if (
                    phase == "answer"
                    and model_step.action == "answer_from_sql"
                    and tool_result.ok
                    and not tool_result.is_terminal
                ):
                    _log(f"[{task.task_id}] phase answer → verify (auto on review_required)")
                    phase = "verify"

                if tool_result.is_terminal:
                    state.answer = tool_result.answer
                    break

            except Exception as exc:
                state.steps.append(
                    StepRecord(
                        step_index=step_index,
                        thought="",
                        action="__error__",
                        action_input={},
                        raw_response=raw_response,
                        observation={"ok": False, "phase": phase, "error": str(exc)},
                        ok=False,
                    )
                )

        if state.answer is None and state.failure_reason is None:
            state.failure_reason = (
                f"Agent did not submit an answer within max_steps "
                f"(stopped in phase {phase})."
            )

        # Mean entropy across collected logprobs (AIMO3 style).
        mean_h = float("inf")
        n_tok = len(state.logprobs_buffer)
        if n_tok:
            import math as _m
            total = 0.0
            for top in state.logprobs_buffer:
                h = 0.0
                for lp in top.values():
                    p = _m.exp(lp)
                    if p > 0:
                        h -= p * _m.log2(p)
                total += h
            mean_h = total / n_tok

        return AgentRunResult(
            task_id=task.task_id,
            answer=state.answer,
            steps=list(state.steps),
            failure_reason=state.failure_reason,
            preamble=self.preamble,
            mean_entropy=mean_h,
            n_logprob_tokens=n_tok,
        )
