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

from experiments.exp_100_aggr_shape_filter.agent import ReActAgent, ReActAgentConfig
from experiments.exp_100_aggr_shape_filter.config import AppConfig
from experiments.exp_100_aggr_shape_filter.preamble import build_preamble
from experiments.exp_100_aggr_shape_filter.tools.registry import (
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
#
# (exp_100 [aggr:shape-filter] axis, 2026-05-06): After the union merge,
# a deterministic shape filter examines the question text. When the
# question phrasing strongly implies a smaller column count than the
# union produced, columns that appear in only 1 of N attempts are
# dropped (columns in 2/3+ attempts are protected). Skip-on-uncertain
# guarantees recall preservation: low confidence / matching count /
# merge-None paths leave the answer byte-identical to exp_086 base.

_MAJORITY_K = 1  # union (k=1): keep any signature appearing in 1+ attempt
_ATTEMPT_TEMPS: tuple[float, ...] = (0.6, 0.6, 0.7)  # Qwen3.5 precise mode


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


def _signature_majority_merge(
    answers: list[dict[str, Any]], *, k: int = _MAJORITY_K
) -> dict[str, Any] | None:
    """Merge agent answers by column-signature majority vote.

    answers: list of {"columns": [...], "rows": [[...], ...]}.
    k: minimum count of attempts a column signature must appear in to be kept.

    Returns a merged answer dict, or None when no answers contained data.
    Falls back to the first non-empty answer if no signature reaches k.
    """
    if not answers:
        return None
    sig_count: Counter[tuple] = Counter()
    sig_data: dict[tuple, tuple[str, list]] = {}
    for ans in answers:
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
            sig_count[sig] += 1
    kept = [(sig_data[sig][0], sig_data[sig][1])
            for sig, c in sig_count.items() if c >= k]
    if not kept:
        for ans in answers:
            if ans and (ans.get("columns") or ans.get("rows")):
                return ans
        return None
    columns = [name for name, _ in kept]
    cols_data = [data for _, data in kept]
    max_len = max(len(c) for c in cols_data) if cols_data else 0
    merged_rows = []
    for i in range(max_len):
        merged_rows.append([(c[i] if i < len(c) else None) for c in cols_data])
    return {"columns": columns, "rows": merged_rows}


# ============================================================================
# Post-union shape filter (exp_100, [aggr:shape-filter] axis)
# ============================================================================
# Heuristic shape estimator + post-union filter. Goals:
#   1) Estimate the question's expected output column count from question
#      text using stdlib regex (5 type + multi-ask modifier).
#   2) When the union output exceeds the expected count, drop columns that
#      appear in only 1 of the 3 attempts (= appears-in-1/3 only).
#      Columns appearing in 2/3+ attempts are PROTECTED to preserve the
#      union-only correct answers (the lesson from exp_087 k=2 majority
#      regression: high diversity makes single-attempt-correct columns
#      common, so any majority-style filter must spare them).
#   3) Skip the filter entirely when:
#      - confidence is "low" (= avoid mis-estimation)
#      - union_cols <= expected_cols (= preserve recall, exp_019 lesson)
#      - merge returned None (= empty answer fallback path unchanged)
# Stdlib regex only, no LLM, no agent feedback loop. Operates strictly
# AFTER the merge — the agent's plan/tool-call/answer phases are unchanged
# from exp_086, structurally avoiding the exp_092 multi-ask blind trap.
# NO public-task vocabulary in any pattern or comment (leak-safe).

_SHAPE_COUNTING_RE = re.compile(r"\b(how many|count of|number of|count\b)", re.IGNORECASE)
_SHAPE_SINGLE_RE = re.compile(r"\b(what is the|who is|which (?!.*\b(?:are|list|all)\b))", re.IGNORECASE)
_SHAPE_LIST_RE = re.compile(r"\b(list (?:all|the)|show me|what are|all the )", re.IGNORECASE)
_SHAPE_AGG_RE = re.compile(r"\b(average|mean|total|sum|max(?:imum)?|min(?:imum)?|highest|lowest)\b", re.IGNORECASE)
_SHAPE_AGG_BY_RE = re.compile(r"\b(?:average|mean|total|sum|max|min)\b.*\bby\b", re.IGNORECASE)
_SHAPE_MULTI_ASK_RE = re.compile(r"\b(\w{3,})\s+and\s+(\w{3,})\b", re.IGNORECASE)
_SHAPE_FIRST_WORDS = 5  # boost to "high" when match falls in the first N tokens


def _estimate_expected_columns(question: str) -> tuple[int | None, str]:
    """Heuristic, deterministic shape estimator. No LLM. Returns
    (expected_cols, confidence) where confidence is "high"/"med"/"low".
    On low confidence, the caller MUST skip the filter to preserve recall.
    """
    if not question or len(question.strip()) < 5:
        return (None, "low")
    q = question.strip()
    q_first = " ".join(q.split()[:_SHAPE_FIRST_WORDS])

    base_cols: int | None = None
    base_conf: str = "low"
    # Specific patterns first (more specific signals win on ties).
    if _SHAPE_COUNTING_RE.search(q):
        base_cols = 1
        base_conf = "high" if _SHAPE_COUNTING_RE.search(q_first) else "med"
    elif _SHAPE_AGG_RE.search(q):
        base_cols = 2 if _SHAPE_AGG_BY_RE.search(q) else 1
        base_conf = "high" if _SHAPE_AGG_RE.search(q_first) else "med"
    elif _SHAPE_SINGLE_RE.search(q):
        base_cols = 1
        base_conf = "high" if _SHAPE_SINGLE_RE.search(q_first) else "med"
    elif _SHAPE_LIST_RE.search(q):
        base_cols = 1
        base_conf = "high" if _SHAPE_LIST_RE.search(q_first) else "med"

    if base_cols is None:
        return (None, "low")

    # Multi-ask modifier: "X and Y" suggests one extra column on top of base.
    if _SHAPE_MULTI_ASK_RE.search(q):
        base_cols = base_cols + 1
        # Multi-ask detection is noisy; cap confidence at "med".
        if base_conf == "high":
            base_conf = "med"

    return (base_cols, base_conf)


def _apply_shape_filter(
    union_output: dict[str, Any],
    expected_cols: int,
    attempt_answers: list[dict[str, Any]],
) -> dict[str, Any]:
    """Drop columns that appear in exactly 1 attempt (out of 3) when the
    union output has more columns than the question implies.

    Conservative-by-design:
      - Returns union_output unchanged if union has <= expected_cols.
      - Returns union_output unchanged if there are no "appears-in-1" cols.
      - Otherwise drops only enough appears-in-1 cols to reach expected_cols
        (or as many as available; never more).
      - Columns appearing in 2/3+ attempts are NEVER dropped (= union-only
        correct answers protected, exp_087 lesson).
    """
    cols: list[str] = list(union_output.get("columns", []))
    rows: list[list[Any]] = [list(r) for r in union_output.get("rows", [])]
    n_cols = len(cols)
    if n_cols <= expected_cols or n_cols == 0:
        return union_output

    # Compute appears-in-N-attempts for each column by signature matching.
    col_sigs = [_column_signature([row[i] if i < len(row) else None for row in rows])
                for i in range(n_cols)]
    sig_attempt_count: Counter[tuple] = Counter()
    for ans in attempt_answers:
        if not ans:
            continue
        a_cols = ans.get("columns") or []
        a_rows = ans.get("rows") or []
        if not a_cols or not a_rows:
            continue
        seen_in_this_attempt: set[tuple] = set()
        for ci in range(len(a_cols)):
            col_values = [(row[ci] if ci < len(row) else None) for row in a_rows]
            sig = _column_signature(col_values)
            seen_in_this_attempt.add(sig)
        for sig in seen_in_this_attempt:
            sig_attempt_count[sig] += 1

    # Pair each column with its attempt-count; identify droppable singletons.
    indexed = [(i, col_sigs[i], sig_attempt_count.get(col_sigs[i], 0)) for i in range(n_cols)]
    droppable = [i for i, _, c in indexed if c <= 1]
    if not droppable:
        return union_output

    # Drop only enough singletons to reach expected_cols (never more).
    n_to_drop = min(len(droppable), n_cols - expected_cols)
    drop_set = set(droppable[:n_to_drop])
    kept_indices = [i for i in range(n_cols) if i not in drop_set]
    new_cols = [cols[i] for i in kept_indices]
    new_rows = [[row[i] for i in kept_indices] for row in rows]
    return {"columns": new_cols, "rows": new_rows}


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


# Derive experiment prefix from the parent directory name (e.g. "exp_100_aggr_shape_filter")
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

    agent = ReActAgent(
        model=model or build_model_adapter(config),
        tools=tools or create_default_tool_registry(),
        config=ReActAgentConfig(
            max_steps=config.agent.max_steps,
            min_steps=config.agent.min_steps,
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
            except Exception:
                if not p.is_alive():
                    pending.remove(idx)
                    progressed = True
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

    # Majority-vote merge across attempts (signature on column values).
    merged_answer = _signature_majority_merge(attempt_answers, k=_MAJORITY_K)

    # exp_100 [aggr:shape-filter]: post-union shape filter.
    # Skip-on-uncertain (low confidence / merged_answer None / question
    # missing) — guarantees no regression below exp_086 baseline behavior.
    shape_metadata: dict[str, Any] = {"applied": False}
    if merged_answer is not None:
        try:
            public_dataset = DABenchPublicDataset(config.dataset.root_path)
            task = public_dataset.get_task(task_id)
            expected_cols, conf = _estimate_expected_columns(task.question)
            if conf == "high" and expected_cols is not None:
                pre_n_cols = len(merged_answer.get("columns", []))
                merged_answer = _apply_shape_filter(merged_answer, expected_cols, attempt_answers)
                post_n_cols = len(merged_answer.get("columns", []))
                shape_metadata = {
                    "applied": True,
                    "expected_cols": expected_cols,
                    "confidence": conf,
                    "pre_cols": pre_n_cols,
                    "post_cols": post_n_cols,
                    "dropped": pre_n_cols - post_n_cols,
                }
        except Exception as exc:
            shape_metadata = {"applied": False, "error": str(exc)}

    # Use the first successful attempt's run_result as the backbone (steps,
    # preamble metadata, etc.), but replace the answer with the merged one.
    backbone = dict(attempt_run_results[0])
    backbone["answer"] = merged_answer
    backbone["shape_filter_metadata"] = shape_metadata
    backbone["majority_3x_metadata"] = {
        "n_attempts": n_attempts,
        "n_succeeded": len(attempt_answers),
        "n_failed": n_attempts - len(attempt_answers),
        "k": _MAJORITY_K,
        "temps": list(_ATTEMPT_TEMPS),
        "failure_reasons": failure_reasons,
    }
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
) -> tuple[Path, list[TaskRunArtifacts]]:
    effective_run_id, run_output_dir = create_run_output_dir(
        config.run.output_dir, run_id=config.run.run_id
    )

    dataset = DABenchPublicDataset(config.dataset.root_path)
    tasks = dataset.iter_tasks()
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
        shared_tools = tools or create_default_tool_registry()
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
