"""Summarize a benchmark run by walking trace.json files.

Reads a `predictions_root` (the same directory shape that
``runner.py:_write_task_outputs`` produces) and groups the run into
human-friendly buckets:

- succeeded vs. missing-prediction
- failure category (max_steps / timeout / connection / other)
- soft-rejected-then-committed (validation_blocking codes)
- step-count distribution
- tool-call frequency
- per-difficulty average elapsed
- (optional) score breakdown when gold_root is supplied

Pure logic + a `format_text(summary)` renderer; CLI surface is in cli.py.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


# Failure-reason fragment → bucket name. First-match wins; case-sensitive.
_FAILURE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("Connection error", "model_connection_error"),
    ("Model request failed", "model_other_error"),
    ("did not submit an answer within max_steps", "max_steps_exhausted"),
    ("timed out after", "subprocess_timeout"),
    ("Task exited", "subprocess_exit"),
    ("Task failed with uncaught error", "uncaught_error"),
)


def _classify_failure(reason: str | None) -> str:
    if not reason:
        return "unknown"
    for fragment, bucket in _FAILURE_PATTERNS:
        if fragment in reason:
            return bucket
    return "other"


@dataclass
class TaskSummary:
    task_id: str
    difficulty: str
    succeeded: bool
    wrote_prediction: bool
    failure_reason: str | None
    failure_bucket: str | None
    step_count: int
    elapsed_seconds: float | None
    tool_calls: dict[str, int]
    soft_reject_codes: list[str]
    answer_columns: int
    answer_rows: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunSummary:
    predictions_root: str
    task_count: int
    succeeded_count: int
    missing_prediction_count: int
    soft_rejected_count: int
    failure_buckets: dict[str, list[str]] = field(default_factory=dict)
    soft_reject_codes: dict[str, int] = field(default_factory=dict)
    tool_calls: dict[str, int] = field(default_factory=dict)
    step_distribution: dict[str, float] = field(default_factory=dict)
    avg_elapsed_by_difficulty: dict[str, float] = field(default_factory=dict)
    counts_by_difficulty: dict[str, int] = field(default_factory=dict)
    score: dict[str, Any] | None = None  # populated when gold_root supplied
    tasks: list[TaskSummary] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "predictions_root": self.predictions_root,
            "task_count": self.task_count,
            "succeeded_count": self.succeeded_count,
            "missing_prediction_count": self.missing_prediction_count,
            "soft_rejected_count": self.soft_rejected_count,
            "failure_buckets": self.failure_buckets,
            "soft_reject_codes": self.soft_reject_codes,
            "tool_calls": self.tool_calls,
            "step_distribution": self.step_distribution,
            "avg_elapsed_by_difficulty": self.avg_elapsed_by_difficulty,
            "counts_by_difficulty": self.counts_by_difficulty,
            "score": self.score,
            "tasks": [t.to_dict() for t in self.tasks],
        }


def _extract_soft_rejects(steps: list[dict[str, Any]]) -> list[str]:
    """Codes from observation.content.warnings on validation_blocking steps."""
    codes: list[str] = []
    for step in steps:
        obs = step.get("observation") or {}
        if not isinstance(obs, dict):
            continue
        content = obs.get("content")
        if not isinstance(content, dict):
            continue
        if content.get("status") != "validation_blocking":
            continue
        for w in content.get("warnings") or []:
            if isinstance(w, dict):
                code = w.get("code")
                sev = w.get("severity")
                if code and sev in ("warning", "error"):
                    codes.append(str(code))
    return codes


def _read_difficulty(input_root: Path | None, task_id: str, fallback: str) -> str:
    if input_root is None:
        return fallback
    path = input_root / task_id / "task.json"
    if not path.is_file():
        return fallback
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return fallback
    return str(meta.get("difficulty") or fallback)


def _summarize_one_trace(
    trace_path: Path,
    *,
    input_root: Path | None,
) -> TaskSummary:
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    task_id = trace.get("task_id") or trace_path.parent.name
    succeeded = bool(trace.get("succeeded"))
    failure_reason = trace.get("failure_reason")
    steps = trace.get("steps") or []
    elapsed = trace.get("e2e_elapsed_seconds")
    answer = trace.get("answer") or {}
    cols = answer.get("columns") if isinstance(answer, dict) else None
    rows = answer.get("rows") if isinstance(answer, dict) else None
    pred_path = trace_path.parent / "prediction.csv"
    wrote_prediction = pred_path.is_file()
    tool_calls = Counter(s.get("action") or "?" for s in steps)
    soft_codes = _extract_soft_rejects(steps)
    difficulty = _read_difficulty(input_root, task_id, fallback="?")
    return TaskSummary(
        task_id=task_id,
        difficulty=difficulty,
        succeeded=succeeded,
        wrote_prediction=wrote_prediction,
        failure_reason=failure_reason,
        failure_bucket=_classify_failure(failure_reason) if failure_reason else None,
        step_count=len(steps),
        elapsed_seconds=float(elapsed) if isinstance(elapsed, (int, float)) else None,
        tool_calls=dict(tool_calls),
        soft_reject_codes=soft_codes,
        answer_columns=len(cols) if isinstance(cols, list) else 0,
        answer_rows=len(rows) if isinstance(rows, list) else 0,
    )


def summarize_run(
    predictions_root: Path,
    *,
    input_root: Path | None = None,
    gold_root: Path | None = None,
) -> RunSummary:
    """Walk predictions_root, summarize. Score is filled when gold_root provided."""
    task_summaries: list[TaskSummary] = []
    for task_dir in sorted(predictions_root.glob("task_*")):
        if not task_dir.is_dir():
            continue
        trace_path = task_dir / "trace.json"
        if not trace_path.is_file():
            continue
        task_summaries.append(_summarize_one_trace(trace_path, input_root=input_root))

    summary = RunSummary(
        predictions_root=str(predictions_root),
        task_count=len(task_summaries),
        succeeded_count=sum(1 for t in task_summaries if t.wrote_prediction),
        missing_prediction_count=sum(1 for t in task_summaries if not t.wrote_prediction),
        soft_rejected_count=sum(1 for t in task_summaries if t.soft_reject_codes),
        tasks=task_summaries,
    )

    # failure buckets — task_id list per bucket, only for missing-pred
    failure_buckets: dict[str, list[str]] = {}
    for t in task_summaries:
        if t.wrote_prediction:
            continue
        bucket = t.failure_bucket or "unknown"
        failure_buckets.setdefault(bucket, []).append(t.task_id)
    summary.failure_buckets = failure_buckets

    # soft-reject code frequency
    soft_codes: Counter[str] = Counter()
    for t in task_summaries:
        soft_codes.update(t.soft_reject_codes)
    summary.soft_reject_codes = dict(soft_codes)

    # tool-call frequency (across all tasks)
    tool_calls: Counter[str] = Counter()
    for t in task_summaries:
        for action, count in t.tool_calls.items():
            tool_calls[action] += count
    summary.tool_calls = dict(tool_calls)

    # step distribution
    step_counts = [t.step_count for t in task_summaries if t.step_count > 0]
    if step_counts:
        sorted_counts = sorted(step_counts)
        summary.step_distribution = {
            "mean": round(statistics.mean(step_counts), 2),
            "p50": int(statistics.median(step_counts)),
            "p90": int(sorted_counts[max(0, int(0.9 * len(sorted_counts)) - 1)]),
            "max": max(step_counts),
        }

    # avg elapsed by difficulty
    by_diff: dict[str, list[float]] = {}
    counts_by_diff: Counter[str] = Counter()
    for t in task_summaries:
        counts_by_diff[t.difficulty] += 1
        if t.elapsed_seconds is not None:
            by_diff.setdefault(t.difficulty, []).append(t.elapsed_seconds)
    summary.avg_elapsed_by_difficulty = {
        diff: round(statistics.mean(values), 1) for diff, values in by_diff.items()
    }
    summary.counts_by_difficulty = dict(counts_by_diff)

    # optional score breakdown
    if gold_root is not None:
        summary.score = _score_breakdown(predictions_root, gold_root, input_root)

    return summary


def _score_breakdown(
    predictions_root: Path,
    gold_root: Path,
    input_root: Path | None,
) -> dict[str, Any]:
    """Score predictions in-process via mock_scorer.score_benchmark."""
    from data_agent_baseline.scoring.mock_scorer import score_benchmark

    try:
        result = score_benchmark(
            predictions_root=predictions_root,
            gold_root=gold_root,
            input_root=input_root,
            lambda_value=0.10,
        )
    except Exception as exc:  # noqa: BLE001 — score is optional; never break summary
        return {"error": f"score_benchmark failed: {exc}"}

    perfect: list[str] = []
    zero_with_pred: list[str] = []
    by_diff: dict[str, dict[str, int]] = {}
    for ts in result.task_scores:
        diff = ts.difficulty or "?"
        bucket = by_diff.setdefault(diff, {"total": 0, "perfect": 0, "zero_with_pred": 0})
        bucket["total"] += 1
        if ts.score >= 0.999:
            perfect.append(ts.task_id)
            bucket["perfect"] += 1
        elif ts.score == 0:
            zero_with_pred.append(ts.task_id)
            bucket["zero_with_pred"] += 1
    return {
        "lambda": 0.10,
        "tasks_scored": len(result.task_scores),
        "mean_score": round(result.mean_score, 4),
        "mean_recall": round(result.mean_recall, 4),
        "missing_predictions": result.missing_predictions,
        "perfect_count": len(perfect),
        "zero_with_pred_count": len(zero_with_pred),
        "perfect_task_ids": sorted(perfect),
        "zero_with_pred_task_ids": sorted(zero_with_pred),
        "by_difficulty": by_diff,
    }


# ----- formatting -----------------------------------------------------------


def format_text(summary: RunSummary) -> str:
    """Plain-text rendering — friendly to terminals + CI logs alike."""
    out: list[str] = []
    out.append(f"=== {summary.predictions_root} ({summary.task_count} tasks) ===\n")

    succ_pct = (
        100.0 * summary.succeeded_count / summary.task_count if summary.task_count else 0.0
    )
    out.append(f"succeeded with prediction:  {summary.succeeded_count:>3}    ({succ_pct:.1f}%)")
    miss_pct = (
        100.0 * summary.missing_prediction_count / summary.task_count if summary.task_count else 0.0
    )
    out.append(f"missing prediction:         {summary.missing_prediction_count:>3}    ({miss_pct:.1f}%)")
    for bucket, ids in sorted(summary.failure_buckets.items()):
        head = ", ".join(ids[:6]) + ("…" if len(ids) > 6 else "")
        out.append(f"  └─ {bucket:30s} {len(ids):>3}    [{head}]")
    out.append("")

    if summary.soft_reject_codes:
        sr_pct = (
            100.0 * summary.soft_rejected_count / summary.task_count if summary.task_count else 0.0
        )
        out.append(f"soft-rejected then committed: {summary.soft_rejected_count:>3}  ({sr_pct:.1f}%)")
        for code, count in sorted(summary.soft_reject_codes.items(), key=lambda x: -x[1]):
            out.append(f"  └─ {code:30s} {count:>3}")
        out.append("")

    if summary.step_distribution:
        sd = summary.step_distribution
        out.append(
            f"step distribution: mean={sd['mean']}  p50={sd['p50']}  p90={sd['p90']}  max={sd['max']}"
        )
        out.append("")

    if summary.tool_calls:
        out.append("tool call frequency:")
        for action, count in sorted(summary.tool_calls.items(), key=lambda x: -x[1]):
            out.append(f"  {action:25s} {count:>5}")
        out.append("")

    if summary.avg_elapsed_by_difficulty:
        out.append("avg elapsed:")
        for diff in ("easy", "medium", "hard", "extreme"):
            if diff in summary.avg_elapsed_by_difficulty:
                avg = summary.avg_elapsed_by_difficulty[diff]
                count = summary.counts_by_difficulty.get(diff, 0)
                out.append(f"  {diff:8s} n={count:>3}  avg={avg:>6.1f}s")
        out.append("")

    if summary.score:
        s = summary.score
        if "error" in s:
            out.append(f"score (λ=0.10): error — {s['error']}")
        else:
            out.append(
                f"score (λ=0.10): mean={s.get('mean_score')}  recall={s.get('mean_recall')}"
            )
            out.append(
                f"  perfect={s.get('perfect_count')}  zero-with-pred={s.get('zero_with_pred_count')}  "
                f"scored_of={s.get('tasks_scored')}"
            )
            for diff, b in sorted((s.get("by_difficulty") or {}).items()):
                out.append(
                    f"  {diff:8s} total={b.get('total'):>3}  perfect={b.get('perfect'):>3}  "
                    f"zero-with-pred={b.get('zero_with_pred'):>3}"
                )

    return "\n".join(out)


def format_diff(left: RunSummary, right: RunSummary) -> str:
    """Side-by-side diff between two runs — useful for v4 vs v4-patched."""
    out: list[str] = []
    out.append(f"--- {left.predictions_root}")
    out.append(f"+++ {right.predictions_root}")
    out.append("")
    out.append(f"task_count:                {left.task_count} → {right.task_count}")
    out.append(f"succeeded with prediction: {left.succeeded_count} → {right.succeeded_count}")
    out.append(f"missing prediction:        {left.missing_prediction_count} → {right.missing_prediction_count}")
    out.append(f"soft-rejected:             {left.soft_rejected_count} → {right.soft_rejected_count}")
    out.append("")
    if left.score and right.score and "error" not in left.score and "error" not in right.score:
        out.append(
            f"perfect (λ=0.10):          {left.score.get('perfect_count')} → "
            f"{right.score.get('perfect_count')}"
        )
        out.append(
            f"zero-with-pred (λ=0.10):   {left.score.get('zero_with_pred_count')} → "
            f"{right.score.get('zero_with_pred_count')}"
        )
        l_tasks = set(left.score.get("perfect_task_ids") or [])
        r_tasks = set(right.score.get("perfect_task_ids") or [])
        regressed = sorted(l_tasks - r_tasks)
        recovered = sorted(r_tasks - l_tasks)
        if regressed:
            out.append(f"regressed (perfect → not):  [{', '.join(regressed[:10])}]")
        if recovered:
            out.append(f"recovered (not → perfect):  [{', '.join(recovered[:10])}]")
    return "\n".join(out)
