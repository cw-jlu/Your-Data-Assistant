from __future__ import annotations

import csv
import json
import multiprocessing
import signal
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from data_agent_baseline.agents.model import OpenAIModelAdapter
from data_agent_baseline.agents.react import ReActAgent, ReActAgentConfig
from data_agent_baseline.agents.self_consistency import SelfConsistencyAgent
from data_agent_baseline.benchmark.dataset import DABenchPublicDataset
from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.config import AppConfig
from data_agent_baseline.memory import (
    ShapePolicy,
    classify_task,
    load_error_patterns,
    load_learnings,
    render_advisories_for_prompt,
    resolve_advisories,
    resolve_policy,
)
from data_agent_baseline.memory.question_shape import infer as infer_question_shape
from data_agent_baseline.memory.task_brief import build_task_brief
from data_agent_baseline.tools.registry import ToolRegistry, create_default_tool_registry


@dataclass(frozen=True, slots=True)
class TaskRunArtifacts:
    task_id: str
    task_output_dir: Path
    prediction_csv_path: Path | None
    trace_path: Path
    succeeded: bool
    failure_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_output_dir": str(self.task_output_dir),
            "prediction_csv_path": str(self.prediction_csv_path) if self.prediction_csv_path else None,
            "trace_path": str(self.trace_path),
            "succeeded": self.succeeded,
            "failure_reason": self.failure_reason,
        }


def create_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def resolve_run_id(run_id: str | None = None) -> str:
    if run_id is None:
        return create_run_id()

    normalized = run_id.strip()
    if not normalized:
        raise ValueError("run_id must not be empty.")
    if normalized in {".", ".."} or "/" in normalized or "\\" in normalized:
        raise ValueError("run_id must be a single directory name, not a path.")
    return normalized


def create_run_output_dir(
    output_root: Path,
    *,
    run_id: str | None = None,
    flat: bool = False,
) -> tuple[str, Path]:
    """Resolve the directory the runner writes per-task subdirs into.

    flat=False (default): wrap with a fresh `<run_id>/` subdirectory. The
    `mkdir(exist_ok=False)` guard rejects collisions, so re-running with
    a custom run_id picks up an explicit error rather than silently
    overwriting prior artifacts.

    flat=True: write directly under `output_root/<task_id>/...` with no
    run_id wrapper. This matches the eval container which mounts
    `/output` and expects `/output/task_<id>/prediction.csv`. We allow
    the directory to pre-exist (the judges may mount an empty dir) and
    we synthesize a run_id internally for trace metadata.
    """
    if flat:
        effective_run_id = resolve_run_id(run_id) if run_id else create_run_id()
        output_root.mkdir(parents=True, exist_ok=True)
        return effective_run_id, output_root
    effective_run_id = resolve_run_id(run_id)
    run_output_dir = output_root / effective_run_id
    run_output_dir.mkdir(parents=True, exist_ok=False)
    return effective_run_id, run_output_dir


def _resolve_seed(config: AppConfig, task_id: str, sample_idx: int = 0) -> int | None:
    """v5 S-1: Per-task deterministic seed.

    Returns ``None`` when seeding is disabled via env ``DABENCH_SEED_BASE=disabled``
    so the endpoint picks its own sampling path (legacy behavior, useful when
    deterministic seed surfaces a bug we want to reproduce stochastically).

    Otherwise: combine ``DABENCH_SEED_BASE`` (default ``42``) with a stable hash
    of ``task_id`` and the sample index — so the same task always gets the same
    seed across runs, AND SelfConsistencyAgent's k samples get k distinct seeds.
    """
    import os
    base_raw = os.environ.get("DABENCH_SEED_BASE")
    if base_raw == "disabled":
        return None
    try:
        base = int(base_raw) if base_raw else 42
    except ValueError:
        base = 42
    # Stable, deterministic across Python invocations — PYTHONHASHSEED affects
    # the builtin hash() but blake2b is stable.
    import hashlib
    digest = hashlib.blake2b(task_id.encode("utf-8"), digest_size=4).digest()
    task_component = int.from_bytes(digest, "big")
    # Stay within int32 range so the endpoint never overflows.
    return (base + task_component + sample_idx) % (2**31 - 1)


def build_model_adapter(config: AppConfig, *, seed: int | None = None):
    return OpenAIModelAdapter(
        model=config.agent.model,
        api_base=config.agent.api_base,
        api_key=config.agent.api_key,
        temperature=config.agent.temperature,
        seed=seed,
    )


def _build_high_temp_model_adapter(config: AppConfig, *, seed: int | None = None):
    """Adapter at config.agent.self_consistency_temperature for sample-mode runs."""
    return OpenAIModelAdapter(
        model=config.agent.model,
        api_base=config.agent.api_base,
        api_key=config.agent.api_key,
        temperature=config.agent.self_consistency_temperature,
        seed=seed,
    )


def _should_use_self_consistency(
    config: AppConfig,
    *,
    difficulty: str,
    overrides_supplied: bool,
) -> bool:
    """Self-consistency only kicks in when:

    - the task has a configured high-variance difficulty (hard / extreme),
    - k > 1 (DABENCH_DISABLE_SELF_CONSISTENCY=1 collapses k to 1 already),
    - no `model=` override was passed in (tests use scripted models with
      ReActAgent directly; SelfConsistencyAgent factory pattern is the
      production path).
    """
    if overrides_supplied:
        return False
    if config.agent.self_consistency_k <= 1:
        return False
    if not difficulty:
        return False
    return difficulty in config.agent.self_consistency_difficulties


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _write_csv(path: Path, columns: list[str], rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            writer.writerow(row)


def _failure_run_result_payload(task_id: str, failure_reason: str) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "answer": None,
        "steps": [],
        "failure_reason": failure_reason,
        "succeeded": False,
    }


def _resolve_task_policy(task: PublicTask) -> ShapePolicy:
    """Classify the task and resolve its ShapePolicy from bundled learnings.

    Best-effort — any failure (missing learnings.json, malformed entry,
    classifier exception) returns the default no-op policy so the runtime
    behaves identically to v6.
    """
    try:
        shape = classify_task(task)
        learnings = load_learnings()
        return resolve_policy(shape, learnings)
    except Exception:  # noqa: BLE001 — policy is advisory; never block a task
        return ShapePolicy()


def _build_task_advisories(task: PublicTask) -> tuple[str, ...]:
    """v7.1 N-1 + N-3: combine the pre-flight task brief with cross-run error
    advisories, returning a tuple of strings to inject as policy hints.

    Both layers are fully best-effort — any exception falls through to an
    empty tuple so the agent path stays identical to v7 baseline.
    """
    advisories: list[str] = []
    try:
        shape = classify_task(task)
    except Exception:  # noqa: BLE001
        return tuple(advisories)

    try:
        brief = build_task_brief(task, shape=shape)
        rendered = brief.render()
        if rendered:
            advisories.append(rendered)
    except Exception:  # noqa: BLE001
        pass

    try:
        patterns = load_error_patterns()
        adv_list = resolve_advisories(shape, patterns)
        block = render_advisories_for_prompt(adv_list)
        if block:
            advisories.append(block)
    except Exception:  # noqa: BLE001
        pass

    # v5 S-3: question-pattern → answer-shape hint. Rule-based; appends
    # only when the question matches a known BIRD pattern (percentage /
    # count / average / list / single-value). Best-effort — any exception
    # leaves the legacy advisory stack untouched.
    #
    # v8 T-G: when the question is a list with "multiple attributes"
    # rationale (task_199-style: "names and funding types"), force the hint
    # to the FRONT of the advisory stack with a ★ marker so the agent reads
    # the multi-column requirement before the brief / error advisories.
    try:
        qhint = infer_question_shape(task.question)
        if qhint is not None:
            rendered_q = qhint.to_prompt_line()
            if rendered_q:
                rationale = (qhint.rationale or "").lower()
                is_multi_attr = (
                    qhint.aggregation_kind == "list"
                    and ("multiple attributes" in rationale or "separate columns" in rationale)
                )
                if is_multi_attr:
                    advisories.insert(0, "★ MULTI-ATTRIBUTE QUESTION — " + rendered_q)
                else:
                    advisories.append(rendered_q)
    except Exception:  # noqa: BLE001
        pass

    return tuple(advisories)


def _run_single_task_core(
    *,
    task_id: str,
    config: AppConfig,
    model=None,
    tools: ToolRegistry | None = None,
) -> dict[str, Any]:
    public_dataset = DABenchPublicDataset(config.dataset.root_path)
    task = public_dataset.get_task(task_id)

    policy = _resolve_task_policy(task)
    # v7.1 N-1 + N-3: enrich the policy's prompt_hints with the per-task brief
    # AND any cross-run error advisories. Both layers are advisory: the agent
    # can still ignore them. We prepend them so the most-actionable shape-
    # specific guidance from learnings.json still appears at the bottom of
    # the hint stack (i.e. closest to the task question).
    extra_hints = _build_task_advisories(task)
    if extra_hints:
        merged_hints = (*extra_hints, *policy.prompt_hints)
        policy = ShapePolicy(
            timeout_multiplier=policy.timeout_multiplier,
            max_steps_multiplier=policy.max_steps_multiplier,
            prompt_hints=merged_hints,
            preferred_tools=policy.preferred_tools,
            avoid_tools=policy.avoid_tools,
            enable_dual_path=policy.enable_dual_path,
            notes=policy.notes,
        )

    react_config = ReActAgentConfig(
        max_steps=config.agent.max_steps,
        max_steps_by_difficulty=config.agent.max_steps_by_difficulty,
    )
    overrides_supplied = (model is not None) or (tools is not None)
    effective_tools = tools or create_default_tool_registry(
        python_kernel_mode=config.agent.python_kernel_mode,
    )

    if _should_use_self_consistency(
        config, difficulty=task.difficulty, overrides_supplied=overrides_supplied
    ):
        # v4 fix #1: budget the k samples within the subprocess timeout so
        # k=3 doesn't blow past the 900s wall-clock and SIGKILL with no
        # trace. We compute the same per-task timeout the outer wrapper
        # will enforce, multiply by the policy multiplier, and reserve
        # 15% headroom for cleanup / queue serialization.
        sc_budget: float | None = None
        base_timeout = _resolve_task_timeout(config, task.difficulty)
        if base_timeout > 0:
            multiplier = float(policy.timeout_multiplier or 1.0)
            full_timeout = base_timeout * multiplier
            sc_budget = full_timeout * 0.85
        # v5 S-1: deterministic per-sample seed so the k samples become
        # reproducible across runs (same task → same k answers → same vote).
        sc_seed_for_sample = lambda idx: _resolve_seed(config, task.task_id, sample_idx=idx)  # noqa: E731
        agent: ReActAgent | SelfConsistencyAgent = SelfConsistencyAgent(
            model_factory_with_index=lambda idx: _build_high_temp_model_adapter(
                config, seed=sc_seed_for_sample(idx)
            ),
            tools=effective_tools,
            config=react_config,
            k=config.agent.self_consistency_k,
            policy_hints=policy.prompt_hints,
            preferred_tools=policy.preferred_tools,
            avoid_tools=policy.avoid_tools,
            max_steps_multiplier=policy.max_steps_multiplier,
            total_budget_seconds=sc_budget,
        )
    else:
        agent = ReActAgent(
            model=model or build_model_adapter(config, seed=_resolve_seed(config, task.task_id)),
            tools=effective_tools,
            config=react_config,
            policy_hints=policy.prompt_hints,
            preferred_tools=policy.preferred_tools,
            avoid_tools=policy.avoid_tools,
            max_steps_multiplier=policy.max_steps_multiplier,
        )
    run_result = agent.run(task)
    return run_result.to_dict()


def _run_single_task_in_subprocess(task_id: str, config: AppConfig, queue: multiprocessing.Queue[Any]) -> None:
    try:
        queue.put(
            {
                "ok": True,
                "run_result": _run_single_task_core(task_id=task_id, config=config),
            }
        )
    except BaseException as exc:  # noqa: BLE001
        queue.put(
            {
                "ok": False,
                "error": str(exc),
            }
        )


def _resolve_task_timeout(config: AppConfig, difficulty: str) -> int:
    """Pick the per-difficulty subprocess wall-clock budget.

    Falls back to ``config.run.task_timeout_seconds`` when the mapping is
    missing the difficulty key (or the mapping itself is None — e.g. when
    DABENCH_TASK_TIMEOUT env was set, signaling explicit single-value
    override). Used to give hard / extreme tier more headroom for
    doc-grounded reasoning that v4 hit the 600 s cap on.
    """
    by_diff = config.run.task_timeout_by_difficulty
    if by_diff and difficulty in by_diff:
        return int(by_diff[difficulty])
    return int(config.run.task_timeout_seconds)


def _peek_task_difficulty(config: AppConfig, task_id: str) -> str:
    """Best-effort difficulty lookup; empty string if anything goes wrong.

    The dataset is the source of truth — re-instantiating it costs only a
    directory scan and keeps the rest of the runner unaware of metadata.
    """
    try:
        dataset = DABenchPublicDataset(config.dataset.root_path)
        return dataset.get_task(task_id).record.difficulty
    except Exception:  # noqa: BLE001 — best-effort, fall through to default timeout
        return ""


def _peek_task_timeout_multiplier(config: AppConfig, task_id: str) -> float:
    """Best-effort policy lookup for timeout scaling. Returns 1.0 on any error."""
    try:
        dataset = DABenchPublicDataset(config.dataset.root_path)
        task = dataset.get_task(task_id)
        policy = _resolve_task_policy(task)
        return float(policy.timeout_multiplier or 1.0)
    except Exception:  # noqa: BLE001
        return 1.0


def _run_single_task_subprocess_once(
    *, task_id: str, config: AppConfig, timeout_seconds: int
) -> dict[str, Any]:
    """One subprocess attempt. Used by _run_single_task_with_timeout below.
    Returns a run_result dict — either the agent's actual result or a
    `_failure_run_result_payload` describing why the subprocess didn't return one.
    """
    queue: multiprocessing.Queue[Any] = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=_run_single_task_in_subprocess,
        args=(task_id, config, queue),
    )
    process.start()
    process.join(timeout_seconds)

    if process.is_alive():
        process.terminate()
        process.join(timeout=1.0)
        if process.is_alive():
            process.kill()
            process.join()
        return _failure_run_result_payload(task_id, f"Task timed out after {timeout_seconds} seconds.")

    if queue.empty():
        exit_code = process.exitcode
        if exit_code not in (None, 0):
            return _failure_run_result_payload(
                task_id,
                f"Task exited unexpectedly with exit code {exit_code}.",
            )
        return _failure_run_result_payload(task_id, "Task exited without returning a result.")

    result = queue.get()
    if result.get("ok"):
        return dict(result["run_result"])
    return _failure_run_result_payload(task_id, f"Task failed with uncaught error: {result['error']}")


# F-5: When the FIRST subprocess attempt produced no progress (0 steps) and
# the failure looks transient (Connection error / first-step timeout), do
# one more subprocess attempt. This recovers tasks where the first
# `model.complete()` got blocked by a vLLM queue spike or network blip,
# without affecting tasks that fail for genuine reasons (max_steps,
# parse errors, tool errors — all of which produce >= 1 step).
_FIRST_STEP_TRANSIENT_HINTS: tuple[str, ...] = (
    "Connection error",
    "Task timed out after",
    "Request timed out",
)

# L-1 (v7): Subprocess-timeout retry only makes sense for easy / medium tier.
# For hard / extreme, hitting the per-task timeout almost always means "the
# task is genuinely too heavy for one attempt, retry will burn the same
# budget for the same outcome". v5/v6 forensics: task_352 / 396 / 418
# burned 1800s every run due to the doubled retry without ever recovering.
# Endpoint blips ("Connection error" / "Request timed out") still retry
# regardless of tier — those are real transients, not heavy-task ceilings.
_TIER_INELIGIBLE_FOR_TIMEOUT_RETRY: frozenset[str] = frozenset({"hard", "extreme"})


def _looks_like_first_step_transient(
    run_result: dict[str, Any], *, difficulty: str = ""
) -> bool:
    if run_result.get("succeeded"):
        return False
    steps = run_result.get("steps") or []
    if len(steps) > 0:
        return False  # got at least one step → not a first-step issue
    reason = str(run_result.get("failure_reason") or "")
    matched_hints = [h for h in _FIRST_STEP_TRANSIENT_HINTS if h in reason]
    if not matched_hints:
        return False
    # L-1: only "Task timed out after" gets the tier filter.
    only_subprocess_timeout = matched_hints == ["Task timed out after"]
    if only_subprocess_timeout and difficulty in _TIER_INELIGIBLE_FOR_TIMEOUT_RETRY:
        return False
    return True


def _run_single_task_with_timeout(*, task_id: str, config: AppConfig) -> dict[str, Any]:
    difficulty = _peek_task_difficulty(config, task_id)
    base_timeout = _resolve_task_timeout(config, difficulty)
    if base_timeout <= 0:
        return _run_single_task_core(task_id=task_id, config=config)

    # M-4 (v7): scale per-task timeout by the matching ShapePolicy. heavy
    # tasks get +50% headroom; default policy returns 1.0 (no change).
    multiplier = _peek_task_timeout_multiplier(config, task_id)
    timeout_seconds = max(base_timeout, int(round(base_timeout * multiplier)))

    result = _run_single_task_subprocess_once(
        task_id=task_id, config=config, timeout_seconds=timeout_seconds
    )
    # L-1 (v7): bounded F-5 retry — see _looks_like_first_step_transient
    # for the tier-aware filter. Hard/extreme subprocess-timeouts no longer
    # consume the retry slot.
    if _looks_like_first_step_transient(result, difficulty=difficulty):
        # One retry of the whole subprocess. No exponential backoff — the
        # first attempt already burned its full timeout slot, so a fresh
        # subprocess is what we need (model adapter rebuilt, queue
        # position reset).
        result = _run_single_task_subprocess_once(
            task_id=task_id, config=config, timeout_seconds=timeout_seconds
        )
    return result


def _write_task_outputs(
    task_id: str,
    run_output_dir: Path,
    run_result: dict[str, Any],
    *,
    prefer_normalized: bool = True,
) -> TaskRunArtifacts:
    task_output_dir = run_output_dir / task_id
    task_output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = task_output_dir / "trace.json"
    _write_json(trace_path, run_result)

    prediction_csv_path: Path | None = None
    raw_answer = run_result.get("answer")
    normalized_answer = run_result.get("normalized_answer") if prefer_normalized else None
    chosen_answer = normalized_answer if isinstance(normalized_answer, dict) else raw_answer
    if isinstance(chosen_answer, dict):
        prediction_csv_path = task_output_dir / "prediction.csv"
        _write_csv(
            prediction_csv_path,
            list(chosen_answer.get("columns", [])),
            [list(row) for row in chosen_answer.get("rows", [])],
        )

    return TaskRunArtifacts(
        task_id=task_id,
        task_output_dir=task_output_dir,
        prediction_csv_path=prediction_csv_path,
        trace_path=trace_path,
        succeeded=bool(run_result.get("succeeded")),
        failure_reason=run_result.get("failure_reason"),
    )


def run_single_task(
    *,
    task_id: str,
    config: AppConfig,
    run_output_dir: Path,
    model=None,
    tools: ToolRegistry | None = None,
    runtime_logger: "RuntimeLogger | None" = None,
) -> TaskRunArtifacts:
    started_at = perf_counter()
    if model is None and tools is None:
        run_result = _run_single_task_with_timeout(task_id=task_id, config=config)
    else:
        run_result = _run_single_task_core(task_id=task_id, config=config, model=model, tools=tools)
    elapsed = round(perf_counter() - started_at, 3)
    run_result["e2e_elapsed_seconds"] = elapsed
    artifact = _write_task_outputs(task_id, run_output_dir, run_result)
    if runtime_logger is not None:
        runtime_logger.log_task(artifact, elapsed_seconds=elapsed)
    return artifact


_GOVERNOR_DOWNGRADE_FACTOR = 0.5
_GOVERNOR_MIN_TIMEOUT_SECONDS = 60
_GOVERNOR_MIN_MAX_STEPS = 1
# v6: cascading governor — number of times we allow downgrade.
# Each downgrade halves timeout + max_steps. With 3 cascades the
# original hard=900s becomes 900 → 450 → 225 → 112 (floor 60), giving
# the budget headroom needed on the hidden ~400-task set where
# v4 (=v8) hit SIGTERM at 12h.
_GOVERNOR_MAX_CASCADES = 3
# Re-evaluation gate: don't re-engage immediately after a downgrade.
# Wait this many additional tasks before checking if the new average
# is still over-budget.
_GOVERNOR_RECHECK_TASK_INTERVAL = 5
# v_next: pre-emptive trigger margin. Engage when projected runtime
# exceeds 80% of remaining budget (not 100%) — v6 reached the 100%
# trigger only after the budget was already lost. 0.8 = 20% headroom.
_GOVERNOR_SAFETY_MARGIN = 0.8
# v_next: probe window. After this many tasks complete, emit a
# `hidden_set_probe` runtime event with elapsed statistics so a
# slow-endpoint scenario can be diagnosed from /logs.
_HIDDEN_PROBE_AFTER_N_TASKS = 10


def _governor_should_engage(
    *,
    budget_seconds: int | None,
    elapsed_seconds: float,
    tasks_remaining: int,
    elapsed_per_task: list[float],
) -> bool:
    """Decide if the governor should kick in this iteration.

    Engagement rule (v_next): trigger when remaining_budget * SAFETY_MARGIN
    < remaining_tasks * average_observed_elapsed. SAFETY_MARGIN = 0.8 leaves
    a 20% buffer so the governor cascades earlier — v6 was killed at 12h
    with the previous "exactly when projected to fail" trigger.

    Returns False if governor disabled (budget None) or no observations yet.
    """
    if budget_seconds is None or tasks_remaining <= 0 or not elapsed_per_task:
        return False
    avg = sum(elapsed_per_task) / len(elapsed_per_task)
    remaining_budget = budget_seconds - elapsed_seconds
    return remaining_budget * _GOVERNOR_SAFETY_MARGIN < tasks_remaining * avg


def _downgrade_config(config: AppConfig) -> AppConfig:
    """Halve max_steps + task_timeout to claw back wall-clock time."""
    halved_mapping: dict[str, int] | None = None
    if config.agent.max_steps_by_difficulty:
        halved_mapping = {
            k: max(_GOVERNOR_MIN_MAX_STEPS, int(v * _GOVERNOR_DOWNGRADE_FACTOR))
            for k, v in config.agent.max_steps_by_difficulty.items()
        }
    # v_next: once cascading starts, force SC k=1 — voting is no longer
    # affordable when we're racing the budget. Saves up to 4× LLM calls
    # per hard/extreme task.
    new_agent = replace(
        config.agent,
        max_steps=max(_GOVERNOR_MIN_MAX_STEPS, int(config.agent.max_steps * _GOVERNOR_DOWNGRADE_FACTOR)),
        max_steps_by_difficulty=halved_mapping,
        self_consistency_k=1,
    )
    halved_timeout_mapping: dict[str, int] | None = None
    if config.run.task_timeout_by_difficulty:
        halved_timeout_mapping = {
            k: max(_GOVERNOR_MIN_TIMEOUT_SECONDS, int(v * _GOVERNOR_DOWNGRADE_FACTOR))
            for k, v in config.run.task_timeout_by_difficulty.items()
        }
    new_run = replace(
        config.run,
        task_timeout_seconds=max(
            _GOVERNOR_MIN_TIMEOUT_SECONDS,
            int(config.run.task_timeout_seconds * _GOVERNOR_DOWNGRADE_FACTOR),
        ),
        task_timeout_by_difficulty=halved_timeout_mapping,
    )
    return replace(config, agent=new_agent, run=new_run)


def _install_sigterm_trap(runtime_logger: "RuntimeLogger | None") -> Callable[[], None]:
    """Best-effort: log a benchmark_end event on SIGTERM before being killed.

    Only installs in the main thread (signal API constraint). Returns a
    callable that restores the previous handler. If installation fails
    (e.g. when called from a non-main thread by tests), returns a no-op.
    """
    if threading.current_thread() is not threading.main_thread():
        return lambda: None

    def handler(signum: int, _frame: Any) -> None:  # noqa: ANN401
        if runtime_logger is not None:
            try:
                runtime_logger.log_event("sigterm_received", signum=int(signum))
                runtime_logger.close()
            except Exception:  # noqa: BLE001
                pass
        # Re-raise SystemExit so any cleanup in finally clauses still runs
        # before we exit. The container has 30s before SIGKILL.
        raise SystemExit(143)  # 128 + 15 = standard SIGTERM exit code

    try:
        previous = signal.signal(signal.SIGTERM, handler)
    except (ValueError, OSError):
        return lambda: None

    def restore() -> None:
        try:
            signal.signal(signal.SIGTERM, previous)
        except (ValueError, OSError):
            pass

    return restore


def run_benchmark_with_passes(
    *,
    config: AppConfig,
    model=None,
    tools: ToolRegistry | None = None,
    limit: int | None = None,
    progress_callback: Callable[[TaskRunArtifacts], None] | None = None,
    task_filter: list[str] | None = None,
) -> tuple[Path, list[TaskRunArtifacts]]:
    """Orchestrate cross-run voting (H-1) — wraps ``run_benchmark``.

    When ``config.run.repeat_max > 1`` and no model/tools override is
    supplied, this runs the benchmark up to ``repeat_max`` times into
    ``output_dir/_runs/run_<i>/`` and votes the predictions into the
    flat layout ``output_dir/task_<id>/prediction.csv`` at the end.

    Adaptive budget guard: between passes we compare the remaining wall-
    clock budget to ``last_pass_duration × pass_safety_margin``. If the
    next pass cannot safely fit, we stop early and vote across whatever
    passes completed. The first pass therefore always survives — worst-
    case (e.g. unexpectedly large hidden set) the orchestrator behaves
    indistinguishably from single-pass.

    ``repeat_max == 1`` or any override → delegates to ``run_benchmark``
    directly to keep the legacy path untouched (existing tests rely on
    its exact behaviour).
    """
    overrides_supplied = (model is not None) or (tools is not None)
    if config.run.repeat_max <= 1 or overrides_supplied:
        return run_benchmark(
            config=config,
            model=model,
            tools=tools,
            limit=limit,
            progress_callback=progress_callback,
            task_filter=task_filter,
        )

    # 1. Resolve the master output dir (with run_id wrapper if non-flat).
    master_run_id, master_output_dir = create_run_output_dir(
        config.run.output_dir,
        run_id=config.run.run_id,
        flat=config.run.flat_output_dir,
    )

    # 2. Top-level runtime logger event for the multi-pass envelope.
    master_logger = RuntimeLogger.maybe_open(config.run.log_file, run_id=master_run_id)
    if master_logger is not None:
        master_logger.log_event(
            "multi_pass_start",
            run_id=master_run_id,
            repeat_max=config.run.repeat_max,
            pass_safety_margin=config.run.pass_safety_margin,
            wall_clock_budget_seconds=config.run.wall_clock_budget_seconds,
        )

    pass_outputs: list[Path] = []
    last_pass_artifacts: list[TaskRunArtifacts] = []
    multi_pass_started_at = perf_counter()

    try:
        for pass_idx in range(config.run.repeat_max):
            pass_dir = master_output_dir / "_runs" / f"run_{pass_idx}"
            # Each pass writes to its own flat subdir; we don't override
            # the log_file so all passes append to the master log.
            pass_config = replace(
                config,
                run=replace(
                    config.run,
                    output_dir=pass_dir,
                    run_id=None,                   # let pass auto-generate
                    flat_output_dir=True,          # flat per-pass tree
                    repeat_max=1,                  # prevent recursion
                ),
            )

            pass_started = perf_counter()
            _, last_pass_artifacts = run_benchmark(
                config=pass_config,
                limit=limit,
                progress_callback=progress_callback,
                task_filter=task_filter,
            )
            last_pass_duration = perf_counter() - pass_started
            pass_outputs.append(pass_dir)

            if master_logger is not None:
                master_logger.log_event(
                    "multi_pass_iteration_done",
                    run_id=master_run_id,
                    pass_index=pass_idx,
                    pass_duration_seconds=round(last_pass_duration, 3),
                    elapsed_total_seconds=round(perf_counter() - multi_pass_started_at, 3),
                )

            # Safety guard: only consider another pass if there is
            # demonstrably enough remaining budget. Always respect
            # repeat_max as a hard cap.
            if pass_idx + 1 >= config.run.repeat_max:
                break
            budget = config.run.wall_clock_budget_seconds
            if budget is None or budget <= 0:
                continue  # no budget configured → run all configured passes
            elapsed_total = perf_counter() - multi_pass_started_at
            margin = float(config.run.pass_safety_margin)
            if (budget - elapsed_total) < last_pass_duration * margin:
                if master_logger is not None:
                    master_logger.log_event(
                        "multi_pass_early_stop",
                        run_id=master_run_id,
                        completed_passes=pass_idx + 1,
                        configured_passes=config.run.repeat_max,
                        last_pass_duration_seconds=round(last_pass_duration, 3),
                        elapsed_total_seconds=round(elapsed_total, 3),
                        remaining_budget_seconds=round(budget - elapsed_total, 3),
                        safety_margin=margin,
                    )
                break

        # 3. Cross-run vote → flat output under master_output_dir.
        try:
            from data_agent_baseline.scoring.cross_run_vote import vote_across_runs
            vote_results = vote_across_runs(
                prediction_roots=pass_outputs,
                output_dir=master_output_dir,
            )
        except Exception as exc:  # noqa: BLE001
            # Fallback: copy first pass's predictions into master_output_dir.
            # This ensures we never end up with FEWER predictions than the
            # single-pass baseline, even on voter failure.
            vote_results = []
            if pass_outputs:
                _fallback_copy_pass(pass_outputs[0], master_output_dir)
            if master_logger is not None:
                master_logger.log_event(
                    "multi_pass_vote_failed",
                    run_id=master_run_id,
                    error=str(exc),
                    fallback_pass=str(pass_outputs[0]) if pass_outputs else None,
                )

        if master_logger is not None:
            master_logger.log_event(
                "multi_pass_end",
                run_id=master_run_id,
                completed_passes=len(pass_outputs),
                voted_tasks=len(vote_results),
                unanimous_tasks=sum(
                    1 for r in vote_results
                    if r.bucket_size > 0 and r.bucket_size == r.total_present_runs
                ),
                split_tasks=sum(
                    1 for r in vote_results
                    if r.total_present_runs >= 2 and r.bucket_size < r.total_present_runs
                ),
                missing_in_all=sum(1 for r in vote_results if r.total_present_runs == 0),
                wall_clock_elapsed_seconds=round(perf_counter() - multi_pass_started_at, 3),
            )
    finally:
        if master_logger is not None:
            master_logger.close()

    return master_output_dir, last_pass_artifacts


def _fallback_copy_pass(source_dir: Path, dest_dir: Path) -> None:
    """Copy ``source_dir/task_*/prediction.csv`` flat into ``dest_dir``.

    Used as a graceful degradation when ``vote_across_runs`` itself
    raises mid-orchestration. Preserves whatever predictions the first
    pass produced so we never ship FEWER tasks than single-pass.
    """
    import shutil

    if not source_dir.is_dir():
        return
    for child in source_dir.iterdir():
        if not child.is_dir() or not child.name.startswith("task_"):
            continue
        pred = child / "prediction.csv"
        if not pred.is_file():
            continue
        out_task = dest_dir / child.name
        out_task.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pred, out_task / "prediction.csv")


def run_benchmark(
    *,
    config: AppConfig,
    model=None,
    tools: ToolRegistry | None = None,
    limit: int | None = None,
    progress_callback: Callable[[TaskRunArtifacts], None] | None = None,
    task_filter: list[str] | None = None,
) -> tuple[Path, list[TaskRunArtifacts]]:
    effective_run_id, run_output_dir = create_run_output_dir(
        config.run.output_dir,
        run_id=config.run.run_id,
        flat=config.run.flat_output_dir,
    )

    runtime_logger = RuntimeLogger.maybe_open(config.run.log_file, run_id=effective_run_id)
    if runtime_logger is not None:
        runtime_logger.log_event(
            "benchmark_start",
            run_id=effective_run_id,
            wall_clock_budget_seconds=config.run.wall_clock_budget_seconds,
        )
    restore_sigterm = _install_sigterm_trap(runtime_logger)

    dataset = DABenchPublicDataset(config.dataset.root_path)
    tasks = dataset.iter_tasks()
    if task_filter is not None:
        wanted = set(task_filter)
        tasks = [t for t in tasks if t.task_id in wanted]
    if limit is not None:
        tasks = tasks[:limit]

    effective_workers = config.run.max_workers
    if effective_workers < 1:
        raise ValueError("max_workers must be at least 1.")
    if model is not None or tools is not None:
        effective_workers = 1

    task_ids = [task.task_id for task in tasks]
    indexed_artifacts: list[TaskRunArtifacts | None] = [None] * len(task_ids)
    elapsed_per_task: list[float] = []
    governor_level = 0
    last_governor_engage_at_task = -1
    effective_config = config
    run_started_at = perf_counter()
    hidden_probe_emitted = False

    def _maybe_emit_hidden_probe() -> None:
        """v_next I-1: after N completed tasks, log average elapsed so the
        operator can diagnose slow endpoint vs slow tasks vs many tasks
        from runtime.log alone — without needing the full eval to finish.
        Emits exactly once per benchmark run."""
        nonlocal hidden_probe_emitted
        if hidden_probe_emitted or runtime_logger is None:
            return
        if len(elapsed_per_task) < _HIDDEN_PROBE_AFTER_N_TASKS:
            return
        first_n = elapsed_per_task[:_HIDDEN_PROBE_AFTER_N_TASKS]
        avg = sum(first_n) / len(first_n)
        sorted_n = sorted(first_n)
        p50 = sorted_n[len(sorted_n) // 2]
        p95 = sorted_n[max(0, int(len(sorted_n) * 0.95) - 1)]
        runtime_logger.log_event(
            "hidden_set_probe",
            run_id=effective_run_id,
            first_n_tasks=_HIDDEN_PROBE_AFTER_N_TASKS,
            avg_elapsed_seconds=round(avg, 2),
            p50_elapsed_seconds=round(p50, 2),
            p95_elapsed_seconds=round(p95, 2),
            total_remaining_tasks=len(task_ids) - len(elapsed_per_task),
            budget_remaining_seconds=(
                round(config.run.wall_clock_budget_seconds - (perf_counter() - run_started_at), 1)
                if config.run.wall_clock_budget_seconds
                else None
            ),
        )
        hidden_probe_emitted = True

    def _maybe_engage_governor(remaining: int) -> None:
        """v6: cascading. Re-evaluate after each batch / every N tasks; downgrade
        again if the new average is still over-budget."""
        nonlocal governor_level, last_governor_engage_at_task, effective_config
        if governor_level >= _GOVERNOR_MAX_CASCADES:
            return
        # Avoid thrashing: require at least N task observations between cascades.
        tasks_seen = len(elapsed_per_task)
        if (
            governor_level > 0
            and tasks_seen - last_governor_engage_at_task < _GOVERNOR_RECHECK_TASK_INTERVAL
        ):
            return
        if not _governor_should_engage(
            budget_seconds=config.run.wall_clock_budget_seconds,
            elapsed_seconds=perf_counter() - run_started_at,
            tasks_remaining=remaining,
            elapsed_per_task=elapsed_per_task,
        ):
            return
        governor_level += 1
        last_governor_engage_at_task = tasks_seen
        effective_config = _downgrade_config(effective_config)
        if runtime_logger is not None:
            runtime_logger.log_event(
                "governor_engaged",
                run_id=effective_run_id,
                budget_seconds=config.run.wall_clock_budget_seconds,
                elapsed_seconds=int(perf_counter() - run_started_at),
                tasks_remaining=remaining,
                avg_seen_seconds=(sum(elapsed_per_task) / len(elapsed_per_task)) if elapsed_per_task else None,
                governor_level=governor_level,
            )

    try:
        if effective_workers == 1:
            shared_model = model or build_model_adapter(config)
            shared_tools = tools or create_default_tool_registry(
                python_kernel_mode=config.agent.python_kernel_mode,
            )
            for index, task_id in enumerate(task_ids):
                _maybe_engage_governor(remaining=len(task_ids) - index)
                task_started = perf_counter()
                artifact = run_single_task(
                    task_id=task_id,
                    config=effective_config,
                    run_output_dir=run_output_dir,
                    model=shared_model,
                    tools=shared_tools,
                    runtime_logger=runtime_logger,
                )
                elapsed_per_task.append(perf_counter() - task_started)
                _maybe_emit_hidden_probe()
                indexed_artifacts[index] = artifact
                if progress_callback is not None:
                    progress_callback(artifact)
        else:
            # Batched parallel: governor checked between batches so the
            # downgraded config takes effect on the next batch's submissions.
            batch_start = 0
            while batch_start < len(task_ids):
                _maybe_engage_governor(remaining=len(task_ids) - batch_start)
                batch_end = min(batch_start + effective_workers, len(task_ids))
                batch_indices = list(range(batch_start, batch_end))
                submission_clock: dict[int, float] = {}
                with ThreadPoolExecutor(max_workers=effective_workers) as executor:
                    future_to_index = {}
                    for i in batch_indices:
                        submission_clock[i] = perf_counter()
                        future_to_index[
                            executor.submit(
                                run_single_task,
                                task_id=task_ids[i],
                                config=effective_config,
                                run_output_dir=run_output_dir,
                                runtime_logger=runtime_logger,
                            )
                        ] = i
                    for future in as_completed(future_to_index):
                        artifact = future.result()
                        idx = future_to_index[future]
                        indexed_artifacts[idx] = artifact
                        elapsed_per_task.append(perf_counter() - submission_clock[idx])
                        _maybe_emit_hidden_probe()
                        if progress_callback is not None:
                            progress_callback(artifact)
                batch_start = batch_end

        task_artifacts = [a for a in indexed_artifacts if a is not None]
    finally:
        restore_sigterm()

    summary_path = run_output_dir / "summary.json"
    _write_json(
        summary_path,
        {
            "run_id": effective_run_id,
            "task_count": len(task_artifacts),
            "succeeded_task_count": sum(1 for artifact in task_artifacts if artifact.succeeded),
            "max_workers": effective_workers,
            "flat_output_dir": config.run.flat_output_dir,
            "wall_clock_elapsed_seconds": round(perf_counter() - run_started_at, 3),
            "wall_clock_budget_seconds": config.run.wall_clock_budget_seconds,
            "governor_engaged": governor_level > 0,
            "governor_level": governor_level,
            "tasks": [artifact.to_dict() for artifact in task_artifacts],
        },
    )
    if runtime_logger is not None:
        runtime_logger.log_event(
            "benchmark_end",
            run_id=effective_run_id,
            task_count=len(task_artifacts),
            succeeded_task_count=sum(1 for a in task_artifacts if a.succeeded),
            wall_clock_elapsed_seconds=round(perf_counter() - run_started_at, 3),
            governor_engaged=governor_level > 0,
            governor_level=governor_level,
        )
        runtime_logger.close()
    return run_output_dir, task_artifacts


class RuntimeLogger:
    """Append-only JSON Lines logger for /logs/runtime.log.

    Cheap, thread-safe, no external deps. We do not use the stdlib
    logging module because the eval runtime expects a single structured
    file and we want guaranteed flush per record.
    """

    def __init__(self, path: Path, *, run_id: str | None = None) -> None:
        self._path = path
        self._run_id = run_id
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("a", buffering=1, encoding="utf-8")
        self._lock = threading.Lock()

    @classmethod
    def maybe_open(cls, path: Path | None, *, run_id: str | None) -> "RuntimeLogger | None":
        if path is None:
            return None
        return cls(path, run_id=run_id)

    def log_event(self, event: str, **fields: Any) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "run_id": self._run_id,
            "event": event,
            **fields,
        }
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            self._handle.write(line + "\n")
            self._handle.flush()

    def log_task(self, artifact: TaskRunArtifacts, *, elapsed_seconds: float) -> None:
        self.log_event(
            "task_done",
            task_id=artifact.task_id,
            succeeded=artifact.succeeded,
            elapsed_seconds=elapsed_seconds,
            failure_reason=artifact.failure_reason,
            wrote_prediction=artifact.prediction_csv_path is not None,
        )

    def close(self) -> None:
        with self._lock:
            if not self._handle.closed:
                self._handle.close()
