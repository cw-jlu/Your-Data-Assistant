"""3-phase orchestration for exp_110: Plan → Solve → Verify (+ Repair).

Used inside runner.py:_run_single_task_in_subprocess to wrap a single
attempt's full PSV pipeline. Returns the standard run_result shape so the
multi-attempt union and trace JSON keep working unchanged.
"""
from __future__ import annotations

import json
import time
from typing import Any

from kobushi_core.model import ModelAdapter

from experiments.exp_110_plan_solve_verify.agent import ReActAgent, ReActAgentConfig
from experiments.exp_110_plan_solve_verify.phase1_prompt import generate_plan
from experiments.exp_110_plan_solve_verify.phase3_prompt import verify_answer
from experiments.exp_110_plan_solve_verify.preamble import build_preamble
from experiments.exp_110_plan_solve_verify.prompt import (
    REACT_SYSTEM_PROMPT,
    PHASE2_PLAN_BINDING_TEMPLATE,
)


def _format_plan_binding(plan: dict[str, Any]) -> str:
    """Render the approved plan as a human-readable section appended to the
    Phase 2 system prompt (= so the agent sees the plan as binding).
    """
    return PHASE2_PLAN_BINDING_TEMPLATE.format(plan_json=json.dumps(plan, indent=2, ensure_ascii=False))


_SUPERLATIVE_WORDS = {
    "lowest", "highest", "least", "most", "minimum", "maximum",
    "cheapest", "largest", "smallest", "fastest", "slowest",
    "oldest", "newest", "best", "worst", "top", "bottom",
    "first", "last",
}


def _question_is_superlative(question: str) -> bool:
    q = (question or "").lower()
    return any(f" {w} " in f" {q} " or q.startswith(w + " ") or q.endswith(" " + w) for w in _SUPERLATIVE_WORDS)


def _drop_extra_columns(answer: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    """Mechanical column-drop fix for FIX_DROP_COLUMNS verdict.

    Drops columns from the answer when there are more than plan.column_count.
    Keeps the FIRST `column_count` columns (= preserves agent's primary
    column choice). If column_count is 0 or unset, returns answer unchanged.
    """
    target = int(plan.get("column_count") or 0)
    if target <= 0:
        return answer
    cols = answer.get("columns") or []
    rows = answer.get("rows") or []
    if len(cols) <= target:
        return answer
    new_cols = cols[:target]
    new_rows = [row[:target] for row in rows]
    return {"columns": new_cols, "rows": new_rows}


def _is_recompute_safe(
    *, original_answer: dict[str, Any], repaired_answer: dict[str, Any] | None,
    question: str, plan: dict[str, Any], verdict: dict[str, Any],
) -> bool:
    """Decide whether to keep the repaired answer or fall back to the original.

    Veto recompute when the heuristics suggest the verifier was wrong:
    - Question is superlative and original had multi-row, repair shrank to 1.
      → likely tied results clobbered by a buggy plan. Keep original.
    - Repair produced 0 rows or empty answer. Keep original.
    - Repair changed column-set entirely (= different concept, not a fix).
    """
    if repaired_answer is None:
        return False
    orig_rows = len(original_answer.get("rows") or [])
    rep_rows = len(repaired_answer.get("rows") or [])
    if rep_rows == 0:
        return False
    # Superlative + row shrink veto
    if _question_is_superlative(question) and orig_rows > 1 and rep_rows == 1:
        return False
    # Column-set drift veto: if repair changed all column names, it's likely
    # a different answer, not a fix.
    orig_cols = set(str(c).lower() for c in (original_answer.get("columns") or []))
    rep_cols = set(str(c).lower() for c in (repaired_answer.get("columns") or []))
    if orig_cols and rep_cols and orig_cols.isdisjoint(rep_cols):
        return False
    return True


def run_psv_pipeline(
    *,
    task,
    config,
    model: ModelAdapter,
    tools,
) -> dict[str, Any]:
    """Run Plan → Solve → Verify (→ Repair) for one attempt.

    Returns a run_result dict compatible with runner.py:_run_single_task_core,
    augmented with `psv_metadata` capturing per-phase observations.
    """
    pipeline_t0 = time.time()
    psv_meta: dict[str, Any] = {}

    # ---- Build preamble ONCE; reused across phases.
    preamble_result = build_preamble(task)
    preamble_text = preamble_result.text

    # ---- Phase 1: Plan
    plan, plan_meta = generate_plan(
        question=task.question,
        preamble=preamble_text,
        model=model,
    )
    psv_meta["phase1"] = plan_meta
    psv_meta["plan"] = plan

    # ---- Phase 2: ReAct execution with plan injected into system prompt
    plan_section = _format_plan_binding(plan)
    phase2_system_prompt = f"{REACT_SYSTEM_PROMPT}\n\n{plan_section}"

    agent = ReActAgent(
        model=model,
        tools=tools,
        config=ReActAgentConfig(
            max_steps=config.agent.max_steps,
            min_steps=config.agent.min_steps,
        ),
        system_prompt=phase2_system_prompt,
        preamble=preamble_text,
    )
    p2_t0 = time.time()
    react_result = agent.run(task)
    p2_dt = time.time() - p2_t0

    psv_meta["phase2"] = {
        "phase": "react",
        "latency_s": round(p2_dt, 2),
        "n_steps": len(react_result.steps),
        "succeeded": react_result.succeeded,
    }

    # If ReAct failed entirely (no answer), skip Phase 3/4.
    if not react_result.succeeded or not react_result.answer:
        psv_meta["pipeline_total_s"] = round(time.time() - pipeline_t0, 2)
        return {
            "task_id": task.task_id,
            "answer": react_result.answer.to_dict() if react_result.answer is not None else None,
            "steps": [s.to_dict() for s in react_result.steps],
            "succeeded": react_result.succeeded,
            "failure_reason": react_result.failure_reason,
            "preamble": preamble_text,
            "preamble_metadata": preamble_result.metadata(),
            "psv_metadata": psv_meta,
        }

    # ---- Phase 3: Verify
    answer = react_result.answer.to_dict()  # dict shape for downstream
    verdict, verify_meta = verify_answer(
        plan=plan,
        answer=answer,
        steps=[s.to_dict() for s in react_result.steps],
        model=model,
    )
    psv_meta["phase3"] = verify_meta
    psv_meta["verdict"] = verdict

    # ---- Phase 4: Conditional repair
    final_answer = answer
    p4_meta: dict[str, Any] = {"phase": "repair", "applied": False, "kind": None}
    verdict_kind = verdict.get("verdict", "OK")

    if verdict_kind == "FIX_DROP_COLUMNS":
        # Mechanical fix, no LLM call.
        repaired = _drop_extra_columns(answer, plan)
        if repaired != answer:
            final_answer = repaired
            p4_meta["applied"] = True
            p4_meta["kind"] = "drop_columns"
            p4_meta["before_cols"] = answer.get("columns")
            p4_meta["after_cols"] = repaired.get("columns")

    elif verdict_kind in ("FIX_RECOMPUTE", "FIX_TARGETED"):
        # Anti-false-alarm: skip recompute for superlative questions when
        # the agent's multi-row answer suggests ties (= classic plan-was-wrong
        # case from task_25 smoke).
        is_superlative = _question_is_superlative(task.question)
        n_rows_orig = len(answer.get("rows") or [])
        if is_superlative and n_rows_orig > 1 and plan.get("row_count_category") == "single":
            p4_meta["applied"] = False
            p4_meta["kind"] = "skipped_superlative_protection"
            p4_meta["note"] = (
                "Question is superlative and answer has multiple rows. "
                "Agent's tied multi-row answer kept; verifier overruled."
            )
        elif verdict.get("confidence") == "high":
            fix_instruction = verdict.get("fix_instruction") or "Re-examine the answer based on the verifier's issues."
            issues_text = "; ".join(verdict.get("issues") or [])
            # Run a small extra ReAct loop with the fix as the seed thought
            repair_seed = (
                f"Verifier flagged: {issues_text}. Fix instruction: {fix_instruction}. "
                f"Re-run only the steps needed to correct this, then re-submit the answer."
            )
            repair_agent = ReActAgent(
                model=model,
                tools=tools,
                config=ReActAgentConfig(max_steps=6, min_steps=0),
                system_prompt=phase2_system_prompt + f"\n\n## Repair seed\n{repair_seed}",
                preamble=preamble_text,
            )
            p4_t0 = time.time()
            repair_result = repair_agent.run(task)
            p4_meta["latency_s"] = round(time.time() - p4_t0, 2)
            p4_meta["applied"] = True
            p4_meta["kind"] = verdict_kind.lower()
            repaired = repair_result.answer.to_dict() if (repair_result.succeeded and repair_result.answer) else None
            # Sanity-check the repair before accepting it. If the repair
            # produced an obviously worse answer (zero rows / superlative
            # row-shrink / total column-set drift), keep the original.
            if _is_recompute_safe(
                original_answer=answer,
                repaired_answer=repaired,
                question=task.question,
                plan=plan,
                verdict=verdict,
            ):
                final_answer = repaired
                p4_meta["repaired_succeeded"] = True
            else:
                p4_meta["repaired_succeeded"] = False
                p4_meta["repair_rejected_reason"] = "sanity_check_failed"
                if repaired is None:
                    p4_meta["failure_reason"] = repair_result.failure_reason
        else:
            p4_meta["applied"] = False
            p4_meta["kind"] = "skipped_low_confidence"
    else:
        p4_meta["kind"] = "ok"

    psv_meta["phase4"] = p4_meta
    psv_meta["pipeline_total_s"] = round(time.time() - pipeline_t0, 2)

    return {
        "task_id": task.task_id,
        "answer": final_answer,
        "steps": [s.to_dict() for s in react_result.steps],
        "succeeded": react_result.succeeded,
        "failure_reason": react_result.failure_reason,
        "preamble": preamble_text,
        "preamble_metadata": preamble_result.metadata(),
        "psv_metadata": psv_meta,
    }
