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

from experiments.exp_164_exp149_replay.agent import (
    parse_model_step,
)
from experiments.exp_164_exp149_replay.prompt import (
    allowed_tools_for_phase,
    build_observation_prompt,
    build_phased_system_prompt,
    build_task_prompt,
)
from experiments.exp_164_exp149_replay.runtime import (
    AgentRunResult,
    AgentRuntimeState,
    StepRecord,
)
from experiments.exp_164_exp149_replay.tools.registry import ToolRegistry
from experiments.exp_164_exp149_replay import flags
from experiments.exp_164_exp149_replay.source_router import route_answer_source

import re as _re

PHASES_ORDER = ("plan", "explore", "answer", "verify")
PHASE_TO_INDEX = {p: i for i, p in enumerate(PHASES_ORDER)}
TRANSITION_ACTION = "complete_phase"

# exp_164 lever ② (prose_on): detect a DuckDB catalog-404 so we can redirect
# the agent to the same-stem prose doc instead of letting it substitute a
# wrong table or loop on sqlite_master.
_CATALOG_404_RE = _re.compile(r"Table with name\s+([A-Za-z0-9_]+)\s+does not exist", _re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class PhasedAgentConfig:
    max_steps: int = 64
    min_explore_queries: int = 3  # min successful EXPLORE queries before complete_phase to answer


def _validate_action(
    current: str, action: str, action_input: dict,
    n_explore_ok: int, cfg: PhasedAgentConfig,
    *, watch_video_required: bool = False, watched_video_ok: bool = False,
) -> tuple[bool, str, str | None, str]:
    """Pre-execute validation. Returns (allow, new_phase_if_transition, error_msg, virtual_action).

    `virtual_action` is "transition" if action == complete_phase (= no tool dispatch),
    otherwise "tool" (= execute via registry as usual).
    """
    allowed = allowed_tools_for_phase(current)
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
            if watch_video_required and not watched_video_ok:
                return False, current, (
                    "complete_phase to ANSWER is blocked: this task has a video and "
                    "EXP164_WATCH_VIDEO=1, but no successful watch_video call has "
                    "verified the raw video evidence. Call watch_video once with a "
                    "short focus on the relevant date window, threshold, filter, "
                    "ranking, or displayed answer."
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
        video_keyframe_note: str | None = None,
        collect_logprobs: bool = False,
    ) -> None:
        self.model = model
        self.tools = tools
        self.config = config or PhasedAgentConfig()
        self.preamble = preamble
        self.video_keyframe_note = video_keyframe_note
        self.collect_logprobs = collect_logprobs
        # Optional per-instance trace log path (= overrides env var).
        # Useful for parallel benches where each attempt needs its own trace.
        self.trace_log_path: str | None = None

    @staticmethod
    def _find_video_path(context_dir):
        """Locate the first video file under context_dir (= Phase 2 modality)."""
        from pathlib import Path as _P
        if context_dir is None:
            return None
        cd = _P(context_dir)
        if not cd.is_dir():
            return None
        for ext in ("*.mp4", "*.webm", "*.mov", "*.avi", "*.mkv"):
            for p in sorted(cd.glob(f"**/{ext}")):
                return p
        return None

    @staticmethod
    def _build_user_content_with_video(text: str, video_path):
        """Build multimodal content list for the first PLAN user message.

        Returns either a plain str (= no video) or OpenAI Chat API content
        list with text + video_url part.
        """
        if video_path is None:
            return text
        import base64
        b64 = base64.b64encode(video_path.read_bytes()).decode()
        note = (
            "\n\n# [VIDEO] briefing — attached raw video\n"
            "Inspect the attached video directly for criteria/displayed answers.\n"
        )
        return [
            {"type": "text", "text": text + note},
            {"type": "video_url",
             "video_url": {"url": f"data:video/mp4;base64,{b64}"}},
        ]

    @staticmethod
    def _find_keyframe_paths(task: PublicTask) -> list:
        """Locate pre-extracted unique video keyframes for this task."""
        import os
        from pathlib import Path as _P

        root = _P(os.environ.get("EXP164_KEYFRAME_ROOT", "/tmp/kobushi_exp164_keyframe_cache"))
        if not root.is_absolute():
            root = _P.cwd() / root
        task_id = getattr(task, "task_id", "")
        frames = sorted((root / task_id / "keyframes").glob("*.jpg"))
        if not frames:
            return []

        try:
            max_images = max(1, int(os.environ.get("EXP164_KEYFRAME_MAX_IMAGES", "18")))
        except ValueError:
            max_images = 18
        if len(frames) <= max_images:
            return frames
        if max_images == 1:
            return [frames[0]]
        idxs = [round(i * (len(frames) - 1) / (max_images - 1)) for i in range(max_images)]
        return [frames[i] for i in sorted(set(idxs))]

    @staticmethod
    def _build_user_content_with_keyframes(text: str, keyframe_paths: list):
        """Build multimodal content with extracted dashboard keyframes."""
        if not keyframe_paths:
            return text
        import base64

        note = (
            "\n\n# [VIDEO] keyframes — extracted unique dashboard screens\n"
            "Inspect attached keyframes for criteria/displayed answers. Prefer explicit "
            "configuration screens when screens conflict.\n"
        )
        content = [{"type": "text", "text": text + note}]
        for p in keyframe_paths:
            b64 = base64.b64encode(p.read_bytes()).decode()
            content.append({"type": "text", "text": f"Keyframe: {p.name}"})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                }
            )
        return content

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
        # Phase 2: attach modality content to FIRST user message only.
        # Subsequent turns just see text observations.
        if flags.keyframes_on():
            keyframes = self._find_keyframe_paths(task)
            if keyframes:
                first_user_content = self._build_user_content_with_keyframes(
                    task_prompt, keyframes
                )
            else:
                video_path = None if flags.watch_video_on() else self._find_video_path(
                    getattr(task, "context_dir", None)
                )
                first_user_content = self._build_user_content_with_video(task_prompt, video_path)
        else:
            # exp_164 lever ③ (video_on): the raw video is NOT attached — a
            # pre-extracted summary is already in the preamble. Attaching raw
            # video only when the video lever is OFF (= exp_143 behavior).
            video_path = None
            if not flags.video_on() and not flags.watch_video_on():
                video_path = self._find_video_path(getattr(task, "context_dir", None))
            first_user_content = self._build_user_content_with_video(task_prompt, video_path)
        messages.append(ModelMessage(role="user", content=first_user_content))
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
        watch_video_required = flags.watch_video_on() and (
            self._find_video_path(getattr(task, "context_dir", None)) is not None
        )
        watched_video_ok = False

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
                    watch_video_required=watch_video_required,
                    watched_video_ok=watched_video_ok,
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

                # Execute the tool (regular dispatch). Source-router PoC:
                # immediately before a proposed final SQL, classify whether SQL
                # should be primary, support-only, or avoided based on the
                # exploration trace so far.
                tool_action_input = model_step.action_input
                if flags.source_router_on() and model_step.action == "answer_from_sql":
                    tool_action_input = dict(model_step.action_input)
                    proposed_sql = str(tool_action_input.get("sql", ""))
                    tool_action_input["_source_route"] = route_answer_source(
                        question=task.question,
                        steps=state.steps,
                        sql=proposed_sql,
                        model=self.model,
                        video_keyframe_note=self.video_keyframe_note,
                    )
                tool_result = self.tools.execute(
                    task, model_step.action, tool_action_input
                )
                observation = {
                    "ok": tool_result.ok,
                    "phase": phase,
                    "tool": model_step.action,
                    "content": tool_result.content,
                }
                # exp_164 lever ② (prose_on): a catalog-404 on a knowledge-named
                # table means the data lives in a same-stem prose doc. Rewrite the
                # observation to redirect to read_doc instead of letting the agent
                # substitute a wrong table or loop on sqlite_master.
                if flags.prose_on() and not tool_result.ok and isinstance(tool_result.content, dict):
                    _m404 = _CATALOG_404_RE.search(str(tool_result.content.get("error", "")))
                    if _m404:
                        _stem = _m404.group(1)
                        _cd = getattr(task, "context_dir", None)
                        _hit = None
                        if _cd is not None:
                            from pathlib import Path as _P
                            for _ext in (".md", ".pdf"):
                                _cands = list(_P(_cd).rglob(f"{_stem}{_ext}"))
                                if _cands:
                                    _hit = _cands[0].relative_to(_P(_cd)).as_posix()
                                    break
                        if _hit:
                            observation["content"] = {
                                "error": tool_result.content.get("error", ""),
                                "PROSE_DOC_REDIRECT": (
                                    f"Table '{_stem}' is NOT a SQL view — its rows live in the "
                                    f"PROSE document '{_hit}'. You CANNOT SELECT from it. Switch "
                                    f"strategy: read_doc(path='{_hit}') (grep it for the record "
                                    f"anchors / IDs), extract the records per the EXPLORE prose-doc "
                                    f"protocol, then build the answer with `SELECT ... UNION ALL ...` "
                                    f"or a literal VALUES clause."
                                ),
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
                        action_input=tool_action_input,
                        raw_response=raw_response,
                        observation=observation,
                        ok=tool_result.ok,
                    )
                )

                # Update explore counter.
                if (
                    phase == "explore"
                    and model_step.action == "watch_video"
                    and tool_result.ok
                ):
                    watched_video_ok = True

                if phase == "explore" and tool_result.ok and model_step.action in (
                    "describe_data", "execute_sql", "read_doc", "list_context", "watch_video"
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
