from __future__ import annotations

import csv
import dataclasses
import json
import multiprocessing
import re
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from kobushi_core.model import OpenAIModelAdapter
from kobushi_core.benchmark.dataset import DABenchPublicDataset

from experiments.exp_129_rule_checkers.agent import ReActAgent, ReActAgentConfig
from experiments.exp_129_rule_checkers.phased_agent import (
    PhasedReActAgent,
    PhasedAgentConfig,
)
from experiments.exp_129_rule_checkers.adaptive_vote import adaptive_vote
from experiments.exp_129_rule_checkers.config import AppConfig
from experiments.exp_129_rule_checkers.preamble import build_preamble
from experiments.exp_129_rule_checkers.tools.registry import (
    ToolRegistry,
    create_default_tool_registry,
)


# ============================================================================
# Majority-3x merge: signature-based column voting across N parallel attempts
# ============================================================================

# Each task is run N=3 times in parallel with diversified temperatures.
# After all attempts return, columns are merged by *value-vector signature*:
# the eval scorer matches columns by VALUES (column names ignored), so two
# columns with identical sorted normalized values are interchangeable. We
# keep any signature that appears in K=2 of the 3 attempts (consensus), so
# noise-only columns (errors that differ across attempts) get dropped while
# columns the model agrees on (the gold answer) survive. This trades raw
# recall (some single-attempt-only correct columns vanish) for lower
# extras_ratio, which is the right side of the λ=0.5 score.

_MAJORITY_K = 1  # union (k=1): keep any signature appearing in 1+ attempt
_ATTEMPT_TEMPS: tuple[float, ...] = (0.6, 0.6, 0.6)  # 3-attempt vote, matches bench 0.7950 mean

# Early-stop K-of-N (AIMO3-inspired): once any answer signature appears
# in K attempts, terminate the rest. Disabled by default (= 0) to match
# the bench config that produced the 0.7950 baseline (= no early stop, all
# attempts always complete before adaptive_vote runs).
import os as _os
_EARLY_STOP_K = int(_os.environ.get("KOBUSHI_EARLY_STOP_K", "0"))


def _normalize_value(v: Any) -> str:
    """Normalize a single cell for signature matching.
    Mirrors the eval scorer: numeric → 2dp string, null/NaN → '', strings lowered.
    Type-first numeric check: int/float/numpy scalars bypass str() for consistency,
    avoiding repr differences such as int(163109) vs numpy.int64(163109).
    """
    if v is None:
        return ""
    # Fast path: native int/float and numpy scalar types (avoids str(numpy.int64) surprises).
    # bool is excluded (subclass of int; "True"→"1.00" would be wrong).
    if not isinstance(v, (bool, str, bytes)):
        try:
            return f"{float(v):.2f}"
        except (TypeError, ValueError, OverflowError):
            pass
    s = str(v).strip()
    if s.lower() in ("none", "null", "nan", "nat", "<na>", ""):
        return ""
    try:
        f = float(s.replace(",", ""))
        return f"{f:.2f}"
    except (TypeError, ValueError):
        pass
    return s.lower()


def _column_signature(values: list[Any]) -> tuple[str, ...]:
    return tuple(sorted(_normalize_value(v) for v in values))


def _norm_cell(v):
    if v is None: return (0, "")
    s = str(v).strip()
    try: return (1, round(float(s), 3))
    except: return (2, s.lower())


def _answer_signature(ans: dict) -> tuple:
    """Order-insensitive signature for an answer dict."""
    if not ans or not ans.get("columns"): return ("__empty__",)
    cols = ans.get("columns", [])
    rows = ans.get("rows", []) or []
    n = len(cols)
    norm = []
    for r in rows:
        if len(r) != n: continue
        norm.append(tuple(_norm_cell(v) for v in r))
    norm.sort()
    return (n, len(norm), tuple(norm))


def _majority_pick(answers: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Pick the answer whose signature is most common (= simple majority vote)."""
    if not answers: return None
    sig_to_idx: dict[tuple, list[int]] = {}
    for i, a in enumerate(answers):
        sig_to_idx.setdefault(_answer_signature(a), []).append(i)
    best = max(sig_to_idx, key=lambda k: len(sig_to_idx[k]))
    return answers[sig_to_idx[best][0]]


def _signature_majority_merge(
    answers: list[dict[str, Any]], *, k: int = _MAJORITY_K
) -> dict[str, Any] | None:
    """Merge agent answers by column-signature majority vote.

    Row-padding fix (exp_093, 2026-05-06): never pad short cols with None to
    reach max_len. Instead group kept columns by intrinsic length, pick the
    length contributed by the most distinct attempts (mode by attempt count;
    on tie, the smaller length wins — DABench gold skews toward 1-row
    answers, median=1, 72% of tasks). Cols from non-chosen lengths are
    dropped entirely. The chosen group's columns all share one length, so
    no padding is needed and the multiset signature round-trips through CSV
    intact (= fixes task_11 / task_25 / task_259 / task_379-style failures).
    """
    if not answers:
        return None
    sig_count: Counter[tuple] = Counter()
    sig_data: dict[tuple, tuple[str, list]] = {}
    sig_attempt: dict[tuple, int] = {}
    for attempt_idx, ans in enumerate(answers):
        cols = (ans or {}).get("columns") or []
        rows = (ans or {}).get("rows") or []
        if not cols or not rows:
            continue
        for col_idx, col_name in enumerate(cols):
            col_values = [
                (row[col_idx] if col_idx < len(row) else None) for row in rows
            ]
            sig = _column_signature(col_values)
            if sig not in sig_data:
                sig_data[sig] = (col_name, col_values)
                sig_attempt[sig] = attempt_idx
            sig_count[sig] += 1
    kept_sigs = [sig for sig, c in sig_count.items() if c >= k]
    if not kept_sigs:
        for ans in answers:
            if ans and (ans.get("columns") or ans.get("rows")):
                return ans
        return None

    # Group kept columns by intrinsic length.
    by_length: dict[int, list[tuple[str, list, int]]] = {}
    for sig in kept_sigs:
        name, values = sig_data[sig]
        by_length.setdefault(len(values), []).append((name, values, sig_attempt[sig]))

    # Score each length: (num_distinct_attempts, -length). max() picks the
    # length with most distinct attempts; on tie the smaller length wins.
    def _length_score(length: int) -> tuple[int, int]:
        cols = by_length[length]
        n_distinct = len({a for _, _, a in cols})
        return (n_distinct, -length)

    chosen_length = max(by_length.keys(), key=_length_score)
    chosen = by_length[chosen_length]
    columns = [name for name, _, _ in chosen]
    cols_data = [data for _, data, _ in chosen]
    # All cols share chosen_length — no padding, no signature pollution.
    merged_rows = [[c[i] for c in cols_data] for i in range(chosen_length)]
    return {"columns": columns, "rows": merged_rows}


def _config_with_temperature(config: AppConfig, temperature: float) -> AppConfig:
    new_agent = dataclasses.replace(config.agent, temperature=temperature)
    return dataclasses.replace(config, agent=new_agent)


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
            "prediction_csv_path": str(self.prediction_csv_path)
            if self.prediction_csv_path
            else None,
            "trace_path": str(self.trace_path),
            "succeeded": self.succeeded,
            "failure_reason": self.failure_reason,
        }


# Derive experiment prefix from the parent directory name (e.g. "exp_061_r16_only")
_EXP_PREFIX = Path(__file__).resolve().parent.name


def create_run_id(output_root: Path) -> str:
    """Generate ``{exp_dir_name}_{NNN}`` by scanning existing run directories."""
    pattern = re.compile(rf"^{re.escape(_EXP_PREFIX)}_(\d{{3}})$")
    max_seq = 0
    if output_root.is_dir():
        for child in output_root.iterdir():
            m = pattern.match(child.name)
            if m:
                max_seq = max(max_seq, int(m.group(1)))
    return f"{_EXP_PREFIX}_{max_seq + 1:03d}"


def resolve_run_id(run_id: str | None = None, output_root: Path | None = None) -> str:
    if run_id is None:
        if output_root is None:
            raise ValueError("output_root is required when run_id is not specified.")
        return create_run_id(output_root)

    normalized = run_id.strip()
    if not normalized:
        raise ValueError("run_id must not be empty.")
    if normalized in {".", ".."} or "/" in normalized or "\\" in normalized:
        raise ValueError("run_id must be a single directory name, not a path.")
    return normalized


def create_run_output_dir(output_root: Path, *, run_id: str | None = None) -> tuple[str, Path]:
    effective_run_id = resolve_run_id(run_id, output_root=output_root)
    run_output_dir = output_root / effective_run_id
    run_output_dir.mkdir(parents=True, exist_ok=False)
    return effective_run_id, run_output_dir


def build_model_adapter(config: AppConfig):
    return OpenAIModelAdapter(
        model=config.agent.model,
        api_base=config.agent.api_base,
        api_key=config.agent.api_key,
        extra_headers=config.agent.extra_headers,
        temperature=config.agent.temperature,
        presence_penalty=1.0,  # PRIORITIES.md R1 spec override (kobushi_core default=1.5)
    )


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


def _run_single_task_core(
    *,
    task_id: str,
    config: AppConfig,
    model=None,
    tools: ToolRegistry | None = None,
) -> dict[str, Any]:
    public_dataset = DABenchPublicDataset(config.dataset.root_path)
    task = public_dataset.get_task(task_id)
    preamble_result = build_preamble(task)

    agent_model = model or build_model_adapter(config)
    if tools is None:
        # Wire the column auditor (= deterministic temp=0 LLM) into the registry.
        from kobushi_core.model import OpenAIModelAdapter
        try:
            from kobushi_core.model_factory import build_model_adapter as _bm  # noqa
        except Exception:
            _bm = build_model_adapter
        # Reuse the same model adapter; auditor uses the same endpoint.
        tools = create_default_tool_registry(
            auditor_model=agent_model,
            question_provider=lambda: task.question,
            context_dir=task.context_dir,
        )

    agent = PhasedReActAgent(
        model=agent_model,
        tools=tools,
        config=PhasedAgentConfig(
            max_steps=config.agent.max_steps,
            min_explore_queries=3,
        ),
        preamble=preamble_result.text,
    )
    run_result = agent.run(task)
    payload = run_result.to_dict()
    payload["preamble_metadata"] = preamble_result.metadata()
    return payload


def _run_single_task_in_subprocess(
    task_id: str, config: AppConfig, queue: multiprocessing.Queue[Any]
) -> None:
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


def _run_single_task_with_timeout(*, task_id: str, config: AppConfig) -> dict[str, Any]:
    """Run 3 parallel attempts at temperatures (0.0, 0.3, 0.7) and merge by
    signature majority vote (k=2 of 3). All attempts share the per-task timeout.
    """
    timeout_seconds = config.run.task_timeout_seconds
    if timeout_seconds <= 0:
        # No timeout = no parallelism (would risk hanging forever).
        return _run_single_task_core(task_id=task_id, config=config)

    n_attempts = len(_ATTEMPT_TEMPS)
    ctx = multiprocessing.get_context("spawn")

    queues: list[multiprocessing.Queue[Any]] = []
    processes: list[multiprocessing.Process] = []
    for temp in _ATTEMPT_TEMPS:
        q = ctx.Queue()
        attempt_config = _config_with_temperature(config, temp)
        p = ctx.Process(
            target=_run_single_task_in_subprocess,
            args=(task_id, attempt_config, q),
        )
        p.start()
        queues.append(q)
        processes.append(p)

    deadline = perf_counter() + timeout_seconds
    results: list[dict[str, Any] | None] = [None] * n_attempts
    pending = list(range(n_attempts))
    # Track answer signatures of completed attempts for early-stop.
    sig_counter: Counter = Counter()
    early_stopped = False
    while pending:
        progressed = False
        for idx in list(pending):
            q = queues[idx]
            p = processes[idx]
            remaining = deadline - perf_counter()
            if remaining <= 0:
                break
            try:
                results[idx] = q.get(timeout=0.2)
                pending.remove(idx)
                progressed = True
                # Update signature counter; check K-of-N early-stop threshold.
                if _EARLY_STOP_K > 0:
                    r = results[idx]
                    if r and r.get("ok"):
                        ans = (r.get("run_result") or {}).get("answer")
                        if ans:
                            sig = _answer_signature(ans)
                            sig_counter[sig] += 1
                            if sig_counter[sig] >= _EARLY_STOP_K:
                                early_stopped = True
            except Exception:
                if not p.is_alive():
                    pending.remove(idx)
                    progressed = True
        if early_stopped:
            break
        if perf_counter() > deadline:
            break
        if not progressed:
            time.sleep(0.2)

    # Cleanup any still-alive processes (timed out or crashed).
    for p in processes:
        if p.is_alive():
            p.terminate()
            p.join(timeout=1.0)
            if p.is_alive():
                p.kill()
                p.join()
        else:
            p.join(timeout=2.0)

    # Collect successful run_results + answers.
    attempt_run_results: list[dict[str, Any]] = []
    attempt_answers: list[dict[str, Any]] = []
    failure_reasons: list[str] = []
    for r in results:
        if r is None:
            failure_reasons.append("attempt timed out")
            continue
        if r.get("ok"):
            run_result = r.get("run_result", {})
            attempt_run_results.append(run_result)
            ans = run_result.get("answer")
            if ans:
                attempt_answers.append(ans)
            else:
                failure_reasons.append("attempt returned no answer")
        else:
            failure_reasons.append(f"attempt error: {r.get('error', '?')}")

    if not attempt_answers:
        return _failure_run_result_payload(
            task_id,
            f"all {n_attempts} parallel attempts failed: {' | '.join(failure_reasons)}",
        )

    # Adaptive vote (subset-pick > numeric-convergent majority > union).
    # Matches the bench config that produced the 0.7950 baseline mean.
    from kobushi_core.benchmark.schema import AnswerTable
    answer_tables = [
        AnswerTable(columns=list(a.get("columns") or []),
                    rows=[list(r) for r in (a.get("rows") or [])])
        for a in attempt_answers if a
    ]
    voted = adaptive_vote(answer_tables) if answer_tables else None
    if voted is not None:
        merged_answer = {
            "columns": list(voted.columns),
            "rows": [list(r) for r in voted.rows],
        }
    else:
        merged_answer = attempt_answers[0]

    # Use the first successful attempt's run_result as the backbone (steps,
    # preamble metadata, etc.), but replace the answer with the merged one.
    # Save ALL attempts' run_results so per-attempt SQL/result are recoverable
    # offline (= for vote-strategy comparison without re-running).
    backbone = dict(attempt_run_results[0])
    backbone["answer"] = merged_answer
    n_completed = sum(1 for r in results if r is not None)
    dominant_sig_count = max(sig_counter.values()) if sig_counter else 0
    backbone["majority_3x_metadata"] = {
        "n_attempts": n_attempts,
        "n_completed": n_completed,
        "n_succeeded": len(attempt_answers),
        "n_failed": n_attempts - len(attempt_answers),
        "vote_mode": "majority_pick",
        "temps": list(_ATTEMPT_TEMPS),
        "failure_reasons": failure_reasons,
        "early_stop_k": _EARLY_STOP_K,
        "early_stopped": early_stopped,
        "dominant_sig_count": dominant_sig_count,
    }
    # Per-attempt records: capture each attempt's answer + terminal SQL +
    # last-3 steps so we can reconstruct the vote inputs for offline analysis.
    per_attempt = []
    for i, rr in enumerate(attempt_run_results):
        ans = rr.get("answer") if isinstance(rr, dict) else None
        steps = rr.get("steps", []) if isinstance(rr, dict) else []
        # find terminal SQL
        terminal_sql = None
        for s in reversed(steps):
            if isinstance(s, dict) and s.get("action") == "answer_from_sql":
                ai = s.get("action_input", {})
                if isinstance(ai, dict) and "sql" in ai:
                    terminal_sql = ai["sql"]; break
        per_attempt.append({
            "attempt_index": i,
            "answer": ans,
            "terminal_sql": terminal_sql,
            "n_steps": len(steps),
            "succeeded": rr.get("succeeded") if isinstance(rr, dict) else None,
        })
    backbone["per_attempt"] = per_attempt
    return backbone


def _write_task_outputs(
    task_id: str, run_output_dir: Path, run_result: dict[str, Any]
) -> TaskRunArtifacts:
    task_output_dir = run_output_dir / task_id
    task_output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = task_output_dir / "trace.json"
    _write_json(trace_path, run_result)

    prediction_csv_path: Path | None = None
    answer = run_result.get("answer")
    if isinstance(answer, dict):
        prediction_csv_path = task_output_dir / "prediction.csv"
        _write_csv(
            prediction_csv_path,
            list(answer.get("columns", [])),
            [list(row) for row in answer.get("rows", [])],
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
) -> TaskRunArtifacts:
    started_at = perf_counter()
    if model is None and tools is None:
        run_result = _run_single_task_with_timeout(task_id=task_id, config=config)
    else:
        run_result = _run_single_task_core(task_id=task_id, config=config, model=model, tools=tools)
    run_result["e2e_elapsed_seconds"] = round(perf_counter() - started_at, 3)
    return _write_task_outputs(task_id, run_output_dir, run_result)


def run_benchmark(
    *,
    config: AppConfig,
    model=None,
    tools: ToolRegistry | None = None,
    limit: int | None = None,
    progress_callback: Callable[[TaskRunArtifacts], None] | None = None,
    exclude_task_ids: frozenset[str] | None = None,
) -> tuple[Path, list[TaskRunArtifacts]]:
    effective_run_id, run_output_dir = create_run_output_dir(
        config.run.output_dir, run_id=config.run.run_id
    )

    dataset = DABenchPublicDataset(config.dataset.root_path)
    tasks = dataset.iter_tasks()
    if exclude_task_ids:
        tasks = [t for t in tasks if t.task_id not in exclude_task_ids]
    if limit is not None:
        tasks = tasks[:limit]

    effective_workers = config.run.max_workers
    if effective_workers < 1:
        raise ValueError("max_workers must be at least 1.")
    if model is not None or tools is not None:
        effective_workers = 1

    task_ids = [task.task_id for task in tasks]

    task_artifacts: list[TaskRunArtifacts]
    if effective_workers == 1:
        shared_model = model or build_model_adapter(config)
        # Don't pre-build shared_tools when None — let run_single_task build
        # per-task tools so the column auditor's question_provider closure
        # captures the right task. Only honour explicit `tools` override.
        shared_tools = tools
        task_artifacts = []
        for task_id in task_ids:
            artifact = run_single_task(
                task_id=task_id,
                config=config,
                run_output_dir=run_output_dir,
                model=shared_model,
                tools=shared_tools,
            )
            task_artifacts.append(artifact)
            if progress_callback is not None:
                progress_callback(artifact)
    else:
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            future_to_index = {
                executor.submit(
                    run_single_task,
                    task_id=task_id,
                    config=config,
                    run_output_dir=run_output_dir,
                ): index
                for index, task_id in enumerate(task_ids)
            }
            indexed_artifacts: list[TaskRunArtifacts | None] = [None] * len(task_ids)
            for future in as_completed(future_to_index):
                artifact = future.result()
                indexed_artifacts[future_to_index[future]] = artifact
                if progress_callback is not None:
                    progress_callback(artifact)
            task_artifacts = [artifact for artifact in indexed_artifacts if artifact is not None]

    summary_path = run_output_dir / "summary.json"
    _write_json(
        summary_path,
        {
            "run_id": effective_run_id,
            "task_count": len(task_artifacts),
            "succeeded_task_count": sum(1 for artifact in task_artifacts if artifact.succeeded),
            "max_workers": effective_workers,
            "tasks": [artifact.to_dict() for artifact in task_artifacts],
        },
    )
    return run_output_dir, task_artifacts
