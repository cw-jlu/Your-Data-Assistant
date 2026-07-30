"""PoC: share PLAN/EXPLORE once, then sample multiple ANSWER branches.

This script deliberately does not change the main exp155/159 bench paths. It
uses the exp155 prompt/tool stack, stops immediately after EXPLORE completes,
then resumes from the same runtime state for N answer branches. Voting is not
applied here; the goal is to inspect branch variance.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import re
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from dotenv import load_dotenv

load_dotenv(REPO / ".env")

from kobushi_core.benchmark.dataset import DABenchPublicDataset
from kobushi_core.benchmark.schema import AnswerTable, PublicTask
from kobushi_core.model import OpenAIModelAdapter

from experiments.exp_155_phase_tool_visibility import flags
from experiments.exp_155_phase_tool_visibility.agent import parse_model_step
from experiments.exp_155_phase_tool_visibility.anti_aggregation import (
    no_agg_note,
    review_aggregation,
)
from experiments.exp_155_phase_tool_visibility.audio_asr import (
    audio_preamble_block,
    find_video,
    load_existing_asr,
    prepare_task_audio,
)
from experiments.exp_155_phase_tool_visibility.math_advisor import generate_formula
from experiments.exp_155_phase_tool_visibility.phased_agent import (
    PHASES_ORDER,
    PHASE_TO_INDEX,
    TRANSITION_ACTION,
    PhasedAgentConfig,
    PhasedReActAgent,
    _validate_action,
)
from experiments.exp_155_phase_tool_visibility.preamble import build_preamble
from experiments.exp_155_phase_tool_visibility.prefix_cache import (
    prefix_cache_enabled,
    with_prefix_cache_header,
)
from experiments.exp_155_phase_tool_visibility.prompt import allowed_tools_for_phase
from experiments.exp_155_phase_tool_visibility.runtime import (
    AgentRunResult,
    AgentRuntimeState,
    StepRecord,
)
from experiments.exp_155_phase_tool_visibility.source_router import route_answer_source
from experiments.exp_155_phase_tool_visibility.tools.registry import create_default_tool_registry
from experiments.exp_155_phase_tool_visibility.video_keyframe_note import (
    extract_keyframe_note,
    preamble_block as video_keyframe_preamble_block,
)
from experiments.exp_155_phase_tool_visibility.video_summarizer import summarize_video


DATA_ROOT = REPO / "data" / "phase2_demo" / "demo_samples_phase2"
INPUT_ROOT = DATA_ROOT / "input"
GOLD_ROOT = DATA_ROOT / "output"
MAX_STEPS = 64

_CATALOG_404_RE = re.compile(
    r"Table with name\s+([A-Za-z0-9_]+)\s+does not exist", re.IGNORECASE
)


def make_model(temp: float = 0.6, *, task_id: str | None = None) -> OpenAIModelAdapter:
    headers = with_prefix_cache_header(
        {
            "CF-Access-Client-Id": os.environ.get("CF_ACCESS_CLIENT_ID", ""),
            "CF-Access-Client-Secret": os.environ.get("CF_ACCESS_CLIENT_SECRET", ""),
        },
        task_id,
    )
    return OpenAIModelAdapter(
        model="qwen3.5-35b-a3b",
        api_base=os.environ.get("AGENT_API_BASE", ""),
        api_key=os.environ.get("AGENT_API_KEY", ""),
        temperature=temp,
        extra_headers=headers,
    )


def _official_score(pred_csv: Path, gold_csv: Path) -> float:
    from kobushi_core.eval.csv_compare import EvaluationOptions, _evaluate_task

    if not gold_csv.is_file():
        return 0.0
    try:
        ev = _evaluate_task(
            task_id=str(pred_csv.parent.name),
            prediction_path=pred_csv,
            gold_path=gold_csv,
            options=EvaluationOptions(),
        )
        return float(ev.official_score_lambda_0_5)
    except Exception as exc:
        print(f"  [score err] {exc}", flush=True)
        return 0.0


def _write_answer_csv(path: Path, answer: AnswerTable) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(answer.columns)
        for row in answer.rows:
            writer.writerow(row)


def _run_advisors(question: str, tid: str) -> tuple[str, object]:
    if not flags.anti_agg_on():
        formula = generate_formula(question, task_id=tid)
        anti = SimpleNamespace(
            raw="AGG: disabled", label="AGG", prompt_lang="en", no_agg=False
        )
        return formula, anti

    with ThreadPoolExecutor(max_workers=2) as ex:
        f_formula = ex.submit(generate_formula, question, task_id=tid)
        f_anti = ex.submit(review_aggregation, question, task_id=tid)
        return f_formula.result(), f_anti.result()


def _log(path: Path | None, msg: str) -> None:
    if path is not None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(msg + "\n")


def _catalog_redirect_observation(task: PublicTask, tool_result_content: Any) -> Any:
    if not flags.prose_on() or not isinstance(tool_result_content, dict):
        return tool_result_content
    match = _CATALOG_404_RE.search(str(tool_result_content.get("error", "")))
    if not match:
        return tool_result_content
    stem = match.group(1)
    hit = None
    context_dir = getattr(task, "context_dir", None)
    if context_dir is not None:
        base = Path(context_dir)
        for ext in (".md", ".pdf"):
            cands = list(base.rglob(f"{stem}{ext}"))
            if cands:
                hit = cands[0].relative_to(base).as_posix()
                break
    if not hit:
        return tool_result_content
    return {
        "error": tool_result_content.get("error", ""),
        "PROSE_DOC_REDIRECT": (
            f"Table '{stem}' is NOT a SQL view — its rows live in the PROSE "
            f"document '{hit}'. You CANNOT SELECT from it. Switch strategy: "
            f"read_doc(path='{hit}'), extract the records, then answer with "
            "`SELECT ... UNION ALL ...` or a literal VALUES clause."
        ),
    }


def _mean_entropy(state: AgentRuntimeState) -> tuple[float, int]:
    n_tok = len(state.logprobs_buffer)
    if not n_tok:
        return float("inf"), 0
    import math

    total = 0.0
    for top in state.logprobs_buffer:
        h = 0.0
        for lp in top.values():
            p = math.exp(lp)
            if p > 0:
                h -= p * math.log2(p)
        total += h
    return total / n_tok, n_tok


def _run_phase_loop(
    *,
    agent: PhasedReActAgent,
    task: PublicTask,
    state: AgentRuntimeState,
    phase: str,
    start_step_index: int,
    stop_after_explore: bool,
    trace_path: Path,
) -> tuple[AgentRunResult, str, AgentRuntimeState]:
    n_explore_ok = sum(
        1
        for step in state.steps
        if step.ok
        and step.observation.get("phase") == "explore"
        and step.action in {"describe_data", "execute_sql", "read_doc", "list_context", "watch_video"}
    )
    watch_video_required = flags.watch_video_on() and (
        agent._find_video_path(getattr(task, "context_dir", None)) is not None
    )
    watched_video_ok = any(
        step.ok and step.action == "watch_video" for step in state.steps
    )
    use_logprobs = bool(agent.collect_logprobs) and hasattr(agent.model, "complete_with_logprobs")

    for step_index in range(start_step_index, agent.config.max_steps + 1):
        messages = agent._build_messages(task, state, phase)
        if use_logprobs:
            raw_response, top_buf = agent.model.complete_with_logprobs(messages)
            if top_buf:
                state.logprobs_buffer.extend(top_buf)
        else:
            raw_response = agent.model.complete(messages)

        try:
            model_step = parse_model_step(raw_response)
            _log(
                trace_path,
                f"[{task.task_id}] phase={phase} step {step_index} "
                f"action={model_step.action} "
                f"thought={(model_step.thought or '')[:180].replace(chr(10), ' | ')}",
            )
            allow, new_phase, reject_msg, kind = _validate_action(
                current=phase,
                action=model_step.action,
                action_input=model_step.action_input,
                n_explore_ok=n_explore_ok,
                cfg=agent.config,
                watch_video_required=watch_video_required,
                watched_video_ok=watched_video_ok,
            )
            if not allow:
                _log(
                    trace_path,
                    f"[{task.task_id}] phase={phase} REJECT {model_step.action}: {reject_msg}",
                )
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
                _log(trace_path, f"[{task.task_id}] phase {phase} -> {new_phase} (via complete_phase)")
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
                            "content": f"Phase advanced: {phase} -> {new_phase}.",
                        },
                        ok=True,
                    )
                )
                phase = new_phase
                if stop_after_explore and phase == "answer":
                    break
                continue

            tool_action_input = model_step.action_input
            if flags.source_router_on() and model_step.action == "answer_from_sql":
                tool_action_input = dict(model_step.action_input)
                proposed_sql = str(tool_action_input.get("sql", ""))
                tool_action_input["_source_route"] = route_answer_source(
                    question=task.question,
                    steps=state.steps,
                    sql=proposed_sql,
                    model=agent.model,
                    video_keyframe_note=agent.video_keyframe_note,
                )

            tool_result = agent.tools.execute(task, model_step.action, tool_action_input)
            content = (
                _catalog_redirect_observation(task, tool_result.content)
                if not tool_result.ok
                else tool_result.content
            )
            observation = {
                "ok": tool_result.ok,
                "phase": phase,
                "tool": model_step.action,
                "content": content,
            }
            preview = str(content)[:320].replace("\n", " | ")
            _log(
                trace_path,
                f"[{task.task_id}] phase={phase} step {step_index} "
                f"result ok={tool_result.ok} terminal={tool_result.is_terminal} content={preview}",
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

            if phase == "explore" and model_step.action == "watch_video" and tool_result.ok:
                watched_video_ok = True
            if phase == "explore" and tool_result.ok and model_step.action in (
                "describe_data",
                "execute_sql",
                "read_doc",
                "list_context",
                "watch_video",
            ):
                n_explore_ok += 1

            if (
                phase == "answer"
                and model_step.action == "answer_from_sql"
                and tool_result.ok
                and not tool_result.is_terminal
            ):
                _log(trace_path, f"[{task.task_id}] phase answer -> verify (auto on review_required)")
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
            _log(trace_path, f"[{task.task_id}] phase={phase} step {step_index} ERROR {exc!r}")

    if state.answer is None and state.failure_reason is None and not (stop_after_explore and phase == "answer"):
        state.failure_reason = (
            f"Agent did not submit an answer within max_steps (stopped in phase {phase})."
        )

    mean_h, n_tok = _mean_entropy(state)
    result = AgentRunResult(
        task_id=task.task_id,
        answer=state.answer,
        steps=list(state.steps),
        failure_reason=state.failure_reason,
        preamble=agent.preamble,
        mean_entropy=mean_h,
        n_logprob_tokens=n_tok,
    )
    return result, phase, state


def _precompute_task_context(
    task: PublicTask,
    tid: str,
    task_dir: Path,
    existing_asr: dict[str, dict],
    whisper_model,
    whisper_lock,
    args,
) -> tuple[str, str | None, str, object, dict[str, Any]]:
    if flags.pdf_preprocess_on():
        try:
            from experiments.exp_155_phase_tool_visibility.pdf_text_cache import ensure_task_pdf_cache

            pdf_cache_results = ensure_task_pdf_cache(task)
            (task_dir / "pdf_preprocess.json").write_text(
                json.dumps(
                    [
                        {
                            "source_rel": r.source_rel,
                            "text_path": str(r.text_path),
                            "meta_path": str(r.meta_path),
                            "page_count": r.page_count,
                            "text_chars": r.text_chars,
                            "source_sha1": r.source_sha1,
                        }
                        for r in pdf_cache_results
                    ],
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception as exc:
            (task_dir / "pdf_preprocess_error.txt").write_text(str(exc), encoding="utf-8")

    if flags.keyframes_on():
        try:
            from experiments.exp_155_phase_tool_visibility.keyframe_cache import ensure_task_keyframes

            keyframe_result = ensure_task_keyframes(task)
            if keyframe_result is not None:
                (task_dir / "keyframes.json").write_text(
                    json.dumps(keyframe_result.__dict__, ensure_ascii=False, indent=2, default=str),
                    encoding="utf-8",
                )
        except Exception as exc:
            (task_dir / "keyframes_error.txt").write_text(str(exc), encoding="utf-8")

    video_keyframe_note = None
    if flags.video_keyframe_note_on():
        try:
            video_keyframe_note = extract_keyframe_note(
                task, make_model(temp=0.2, task_id=tid)
            )
            if video_keyframe_note is not None:
                (task_dir / "video_keyframe_note.md").write_text(
                    video_keyframe_note.text, encoding="utf-8"
                )
        except Exception as exc:
            (task_dir / "video_keyframe_note_error.txt").write_text(str(exc), encoding="utf-8")

    asr_text = None
    asr_meta: dict[str, Any] = {"status": "disabled"}
    if flags.audio_asr_on():
        try:
            asr_text, asr_meta = prepare_task_audio(
                task_id=tid,
                task=task,
                task_dir=task_dir,
                whisper_model=whisper_model,
                whisper_lock=whisper_lock,
                language=args.asr_language,
                quality_filter=args.asr_quality_filter,
                existing_asr=existing_asr,
                existing_asr_path=args.asr_existing_json,
                reuse_cache=args.reuse_asr_cache,
            )
        except Exception as exc:
            asr_meta = {"status": "error", "error": repr(exc)}
            (task_dir / "asr_error.txt").write_text(str(exc), encoding="utf-8")

    formula = "NO_CALC"
    try:
        formula, anti_agg = _run_advisors(task.question, tid)
    except Exception as exc:
        formula = f"NO_CALC ({exc})"
        anti_agg = SimpleNamespace(
            raw=f"AGG: advisor_error={exc!r}",
            label="AGG",
            prompt_lang="en",
            no_agg=False,
        )

    (task_dir / "formula.txt").write_text(formula, encoding="utf-8")
    (task_dir / "anti_aggregation.txt").write_text(
        f"label={anti_agg.label}\nlang={anti_agg.prompt_lang}\nraw={anti_agg.raw}\n",
        encoding="utf-8",
    )

    video_summary = None
    if flags.video_on():
        try:
            video_summary = summarize_video(task, make_model(temp=0.6, task_id=tid))
        except Exception as exc:
            video_summary = f"[video summary error: {exc!r}]"
        if video_summary:
            (task_dir / "video_summary.txt").write_text(video_summary, encoding="utf-8")

    preamble = build_preamble(task, video_summary=video_summary)
    preamble_text = preamble.text
    router_video_note = video_keyframe_note.text if video_keyframe_note else None
    if asr_text:
        preamble_text = audio_preamble_block(
            task.question,
            asr_text,
            max_chars=args.asr_max_chars,
        ) + preamble_text
        if args.audio_to_router:
            router_video_note = (
                (router_video_note or "")
                + "\n\n# VIDEO AUDIO TRANSCRIPT (ASR; may contain errors)\n"
                + asr_text[: args.router_asr_max_chars]
            ).strip()
    if video_keyframe_note is not None:
        preamble_text = video_keyframe_preamble_block(video_keyframe_note) + preamble_text

    skip_advisor = formula.strip().upper().startswith("NO_CALC") or anti_agg.no_agg
    if anti_agg.no_agg:
        injected = no_agg_note(anti_agg.prompt_lang) + preamble_text
    elif skip_advisor:
        injected = preamble_text
    else:
        injected = (
            "# MATH FORMULA HINT (= expert calculation guide)\n"
            f"  {formula}\n\n"
            "Follow this aggregation/division/filter structure EXACTLY.\n\n"
        ) + preamble_text

    return injected, router_video_note, formula, anti_agg, asr_meta


def run_task(
    tid: str,
    out_dir: Path,
    existing_asr: dict[str, dict],
    whisper_model,
    whisper_lock,
    args,
) -> dict[str, Any]:
    t0 = time.time()
    ds = DABenchPublicDataset(root_dir=INPUT_ROOT)
    task = ds.get_task(tid)
    task_dir = out_dir / tid
    task_dir.mkdir(parents=True, exist_ok=True)

    injected, router_video_note, formula, anti_agg, asr_meta = _precompute_task_context(
        task, tid, task_dir, existing_asr, whisper_model, whisper_lock, args
    )

    base_model = make_model(temp=args.temperature, task_id=tid)
    tools = create_default_tool_registry(
        auditor_model=base_model,
        question_provider=lambda: task.question,
        context_dir=task.context_dir,
        anti_agg_label_provider=lambda: anti_agg.label,
    )
    base_agent = PhasedReActAgent(
        model=base_model,
        tools=tools,
        config=PhasedAgentConfig(max_steps=MAX_STEPS, min_explore_queries=3),
        preamble=injected,
        video_keyframe_note=router_video_note,
    )

    shared_result, phase, shared_state = _run_phase_loop(
        agent=base_agent,
        task=task,
        state=AgentRuntimeState(),
        phase="plan",
        start_step_index=1,
        stop_after_explore=True,
        trace_path=task_dir / "shared_explore.steps.log",
    )
    shared_steps = len(shared_state.steps)
    (task_dir / "shared_explore.json").write_text(
        json.dumps(shared_result.to_dict(), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    branches: list[dict[str, Any]] = []
    for branch_id in range(args.branches):
        branch_model = make_model(temp=args.temperature, task_id=tid)
        branch_tools = create_default_tool_registry(
            auditor_model=branch_model,
            question_provider=lambda: task.question,
            context_dir=task.context_dir,
            anti_agg_label_provider=lambda: anti_agg.label,
        )
        branch_agent = PhasedReActAgent(
            model=branch_model,
            tools=branch_tools,
            config=PhasedAgentConfig(max_steps=MAX_STEPS, min_explore_queries=3),
            preamble=injected,
            video_keyframe_note=router_video_note,
        )
        branch_state = copy.deepcopy(shared_state)
        branch_result, branch_phase, _ = _run_phase_loop(
            agent=branch_agent,
            task=task,
            state=branch_state,
            phase=phase,
            start_step_index=shared_steps + 1,
            stop_after_explore=False,
            trace_path=task_dir / f"branch_{branch_id}.steps.log",
        )
        branch_payload: dict[str, Any] = {
            "branch": branch_id,
            "phase": branch_phase,
            "succeeded": branch_result.succeeded,
            "failure_reason": branch_result.failure_reason,
            "n_steps_total": len(branch_result.steps),
            "n_steps_branch": max(0, len(branch_result.steps) - shared_steps),
            "answer": (
                {
                    "columns": list(branch_result.answer.columns),
                    "rows": [list(row) for row in branch_result.answer.rows],
                }
                if branch_result.answer
                else None
            ),
        }
        if branch_result.answer and branch_result.answer.rows:
            pred = task_dir / f"prediction_branch_{branch_id}.csv"
            _write_answer_csv(pred, branch_result.answer)
            branch_payload["score"] = _official_score(pred, GOLD_ROOT / tid / "gold.csv")
        else:
            branch_payload["score"] = 0.0
        (task_dir / f"branch_{branch_id}.json").write_text(
            json.dumps(branch_result.to_dict(), ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        branches.append(branch_payload)

    signatures = [
        json.dumps(b.get("answer"), ensure_ascii=False, sort_keys=True)
        for b in branches
    ]
    unique_answers = len(set(signatures))
    result = {
        "tid": tid,
        "score_by_branch": [b.get("score", 0.0) for b in branches],
        "unique_answers": unique_answers,
        "shared_steps": shared_steps,
        "shared_phase": phase,
        "formula": formula,
        "anti_aggregation": getattr(anti_agg, "raw", ""),
        "anti_agg_label": getattr(anti_agg, "label", ""),
        "asr_status": asr_meta.get("status"),
        "elapsed": time.time() - t0,
        "branches": branches,
    }
    (task_dir / "branch_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="task_1", help="comma-separated task ids")
    ap.add_argument("--branches", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--asr-model", default="small")
    ap.add_argument("--asr-cpu-threads", type=int, default=4)
    ap.add_argument("--asr-language", default="auto", choices=["auto", "zh", "en"])
    ap.add_argument("--asr-existing-json", type=Path, default=None)
    ap.add_argument("--reuse-asr-cache", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--asr-quality-filter", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--asr-max-chars", type=int, default=2600)
    ap.add_argument("--audio-to-router", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--router-asr-max-chars", type=int, default=1600)
    args = ap.parse_args()

    task_ids = [t.strip() for t in args.tasks.split(",") if t.strip()]

    tag = f"answer_branch_poc_{args.branches}branch"
    runs = REPO / "artifacts" / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    i = 1
    while (runs / f"{tag}_{i:03d}").exists():
        i += 1
    out_dir = runs / f"{tag}_{i:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if flags.pdf_preprocess_on() and not os.environ.get("EXP155_PDF_TEXT_CACHE_ROOT"):
        os.environ["EXP155_PDF_TEXT_CACHE_ROOT"] = str(
            Path("/tmp") / "kobushi_exp155_pdf_text_cache" / out_dir.name
        )
    if (flags.keyframes_on() or flags.video_keyframe_note_on()) and not os.environ.get("EXP155_KEYFRAME_ROOT"):
        os.environ["EXP155_KEYFRAME_ROOT"] = str(
            Path("/tmp") / "kobushi_exp155_keyframe_cache" / out_dir.name
        )
    if flags.audio_asr_on() and not os.environ.get("EXP155_AUDIO_ASR_CACHE_ROOT"):
        os.environ["EXP155_AUDIO_ASR_CACHE_ROOT"] = str(
            Path("/tmp") / "kobushi_exp155_audio_asr_cache" / out_dir.name
        )

    ds = DABenchPublicDataset(root_dir=INPUT_ROOT)
    existing_asr = load_existing_asr(args.asr_existing_json) if flags.audio_asr_on() else {}
    whisper_model = None
    whisper_lock = threading.Lock()
    if flags.audio_asr_on():
        needs_whisper = False
        for tid in task_ids:
            if tid in existing_asr:
                continue
            try:
                task = ds.get_task(tid)
            except Exception:
                continue
            if find_video(task.context_dir) is not None:
                needs_whisper = True
                break
        if needs_whisper:
            from faster_whisper import WhisperModel

            print(f"  loading ASR model={args.asr_model} on CPU int8", flush=True)
            whisper_model = WhisperModel(
                args.asr_model,
                device="cpu",
                compute_type="int8",
                cpu_threads=args.asr_cpu_threads,
            )

    print(
        f"=== answer-branch PoC tasks={task_ids} branches={args.branches} "
        f"out={out_dir.name} prefix_cache={prefix_cache_enabled()} ===",
        flush=True,
    )
    print(f"  pdf_text_cache_root={os.environ.get('EXP155_PDF_TEXT_CACHE_ROOT')}", flush=True)
    print(f"  keyframe_root={os.environ.get('EXP155_KEYFRAME_ROOT')}", flush=True)
    print(f"  audio_asr_cache_root={os.environ.get('EXP155_AUDIO_ASR_CACHE_ROOT')}", flush=True)

    results = []
    for tid in task_ids:
        row = run_task(tid, out_dir, existing_asr, whisper_model, whisper_lock, args)
        results.append(row)
        print(
            f"{tid}: scores={row['score_by_branch']} unique_answers={row['unique_answers']} "
            f"shared_steps={row['shared_steps']} elapsed={row['elapsed']:.0f}s",
            flush=True,
        )

    summary = {
        "tag": tag,
        "n_tasks": len(results),
        "branches": args.branches,
        "temperature": args.temperature,
        "prefix_cache": prefix_cache_enabled(),
        "config": {
            "base": "exp155_phase_tool_visibility",
            "max_steps": MAX_STEPS,
            "branching": "shared PLAN/EXPLORE, sequential ANSWER/VERIFY branches",
            "vote_applied": False,
            "active_levers": flags.active_levers(),
        },
        "results": results,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"saved {out_dir}/summary.json")


if __name__ == "__main__":
    main()
