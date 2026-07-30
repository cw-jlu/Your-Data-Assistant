"""Host-side recorder — turn benchmark results into ``learnings.json`` deltas.

This module is **never** imported at eval time inside the container. It is
the host-side learning loop: after every public-set benchmark run we ingest
the per-task `trace.json` + `prediction.csv` + gold tables, classify each
task by shape, and group failures by shape to propose updates to
``learnings.json``.

Two outputs:

1. ``RecorderReport`` — structured data: per-shape success/failure counts,
   suggested policy adjustments, evidence task ids.
2. A human-readable diff (markdown) printed by the CLI command, plus an
   optional ``--auto-apply-low-risk`` mode that writes back the safe
   adjustments in place.

Risk classification
-------------------

- **Low-risk (auto-applicable)**: timeout multiplier increase, max_steps
  multiplier increase, adding a hint that says "tool X did not work for
  this shape" (purely informational).
- **High-risk (human review)**: any new prompt cue with semantic content,
  changes to ``preferred_tools`` / ``avoid_tools`` (the agent might over-
  rely on a tool that doesn't generalise).

The risk gate exists because v6 forensics taught us that prompt-cue
patches can flip task scores in unpredictable ways; numeric scaling
patches behave more predictably.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from datetime import datetime, timezone

from data_agent_baseline.benchmark.dataset import DABenchPublicDataset
from data_agent_baseline.memory.error_patterns import (
    DEFAULT_ERROR_PATTERNS_PATH,
    error_signature_for,
    load_error_patterns,
    save_error_patterns,
)
from data_agent_baseline.memory.learnings import (
    DEFAULT_LEARNINGS_PATH,
    load_learnings,
    save_learnings,
)
from data_agent_baseline.memory.task_shape import TaskShape, classify_task


# Score threshold below which a task is considered "failing" for shape
# clustering. Anything < 0.5 means the predicted column-multiset is more
# different than similar — clear signal that the strategy didn't work.
_FAIL_THRESHOLD = 0.5

# Shapes need at least this many failing tasks before we propose a policy
# update — below that the signal is too noisy.
_MIN_FAILURES_FOR_POLICY = 2


@dataclass(frozen=True, slots=True)
class StepError:
    """One observed tool error inside a task's trace."""
    action: str
    error_signature: str
    raw_error: str
    step_index: int


@dataclass(frozen=True, slots=True)
class TaskOutcome:
    task_id: str
    shape: TaskShape
    succeeded: bool                    # prediction.csv present
    failure_reason: str | None         # from trace.json
    score: float | None                # mock_scorer score, if available
    elapsed_seconds: float | None      # if available
    step_errors: tuple[StepError, ...] = ()  # accumulated tool failures


@dataclass(slots=True)
class ShapeBucket:
    """Aggregated outcomes for a shape-cluster (group key)."""
    group_key: dict[str, Any]
    total: int = 0
    succeeded: int = 0
    failed: int = 0
    timeouts: int = 0
    max_steps_failures: int = 0
    failing_tasks: list[str] = field(default_factory=list)
    succeeding_tasks: list[str] = field(default_factory=list)
    elapsed_avg: float = 0.0


@dataclass(frozen=True, slots=True)
class PolicySuggestion:
    match: dict[str, Any]
    delta: dict[str, Any]                  # what to update on the policy
    reason: str
    risk: str                              # "low" | "high"
    evidence_tasks: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RecorderReport:
    runs_analyzed: int
    outcomes: tuple[TaskOutcome, ...]
    buckets: tuple[ShapeBucket, ...]
    suggestions: tuple[PolicySuggestion, ...]


# ---- ingest -----------------------------------------------------------


def _read_trace(trace_path: Path) -> dict[str, Any]:
    if not trace_path.is_file():
        return {}
    try:
        return json.loads(trace_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _extract_step_errors(trace: dict[str, Any]) -> list[StepError]:
    """Walk a trace.json and pull out (action, error) pairs from failed steps.

    A step is considered a tool failure when ``ok`` is false and we have a
    non-trivial error message. ``__error__`` parse failures are skipped —
    they are agent-format errors, not tool / data errors, and we don't
    want them in the cross-task pattern memory.
    """
    out: list[StepError] = []
    for step in trace.get("steps") or []:
        if not isinstance(step, dict):
            continue
        if step.get("ok") is True:
            continue
        action = str(step.get("action") or "")
        if not action or action == "__error__":
            continue
        observation = step.get("observation") or {}
        if not isinstance(observation, dict):
            continue
        # Tool failure messages can live in observation.error or
        # observation.content.error / observation.content.stderr.
        raw_error = ""
        if isinstance(observation.get("error"), str):
            raw_error = observation["error"]
        else:
            content = observation.get("content")
            if isinstance(content, dict):
                if isinstance(content.get("error"), str):
                    raw_error = content["error"]
                elif isinstance(content.get("stderr"), str) and content["stderr"]:
                    raw_error = content["stderr"]
        if not raw_error:
            continue
        signature = error_signature_for(raw_error)
        out.append(
            StepError(
                action=action,
                error_signature=signature,
                raw_error=raw_error[:240],
                step_index=int(step.get("step_index") or 0),
            )
        )
    return out


def _read_prediction_score(
    pred_path: Path, gold_root: Path, task_id: str
) -> float | None:
    """Best-effort score lookup. Returns None if unavailable."""
    if not pred_path.is_file():
        return None
    gold_path = gold_root / task_id / "gold.csv"
    if not gold_path.is_file():
        return None
    try:
        from data_agent_baseline.scoring.mock_scorer import score_one
    except ImportError:
        return None
    try:
        with pred_path.open(newline="") as f:
            pred_reader = list(csv.reader(f))
        with gold_path.open(newline="") as f:
            gold_reader = list(csv.reader(f))
        if not pred_reader or not gold_reader:
            return None
        from data_agent_baseline.benchmark.schema import AnswerTable

        pred_table = AnswerTable(columns=pred_reader[0], rows=[list(r) for r in pred_reader[1:]])
        gold_table = AnswerTable(columns=gold_reader[0], rows=[list(r) for r in gold_reader[1:]])
        score_value, _recall, _matched, _gold_cols, _pred_cols = score_one(
            prediction=pred_table, gold=gold_table, lambda_value=0.10
        )
        return float(score_value)
    except Exception:  # noqa: BLE001
        return None


def collect_outcomes(
    *,
    predictions_root: Path,
    gold_root: Path,
    dataset_root: Path,
) -> list[TaskOutcome]:
    """Walk a benchmark output tree and produce one TaskOutcome per task."""
    dataset = DABenchPublicDataset(dataset_root)
    outcomes: list[TaskOutcome] = []
    for task in dataset.iter_tasks():
        task_dir = predictions_root / task.task_id
        trace = _read_trace(task_dir / "trace.json")
        succeeded = (task_dir / "prediction.csv").is_file()
        failure_reason = trace.get("failure_reason") if isinstance(trace, dict) else None
        elapsed = trace.get("e2e_elapsed_seconds") if isinstance(trace, dict) else None
        score = _read_prediction_score(
            task_dir / "prediction.csv", gold_root, task.task_id
        ) if succeeded else None
        step_errors = tuple(_extract_step_errors(trace) if isinstance(trace, dict) else ())
        outcomes.append(
            TaskOutcome(
                task_id=task.task_id,
                shape=classify_task(task),
                succeeded=succeeded,
                failure_reason=str(failure_reason) if failure_reason else None,
                score=score,
                elapsed_seconds=float(elapsed) if isinstance(elapsed, (int, float)) else None,
                step_errors=step_errors,
            )
        )
    return outcomes


# ---- bucketing --------------------------------------------------------


def _group_keys() -> tuple[dict[str, Any], ...]:
    """Shape grouping templates — one bucket per (key set) pattern.

    Each template is a dict whose keys are TaskShape attributes; the values
    encode "what to read off the shape" — the actual group_key copied from
    the shape becomes the bucket's identity.
    """
    return (
        {"is_heavy": True},
        {"has_large_json": True},
        {"has_large_db": True},
        {"has_large_csv": True},
        {"is_aggregate_question": True},
        {"is_plural_question": True},
        {"is_singular_question": True, "is_aggregate_question": False},
        {"difficulty": "extreme"},
        {"difficulty": "hard"},
    )


def _shape_matches_template(shape: TaskShape, template: dict[str, Any]) -> bool:
    shape_dict = shape.to_dict()
    for key, expected in template.items():
        if shape_dict.get(key) != expected:
            return False
    return True


def cluster_by_shape(outcomes: list[TaskOutcome]) -> list[ShapeBucket]:
    buckets: dict[str, ShapeBucket] = {}
    for outcome in outcomes:
        for template in _group_keys():
            if not _shape_matches_template(outcome.shape, template):
                continue
            key_str = json.dumps(template, sort_keys=True)
            bucket = buckets.setdefault(key_str, ShapeBucket(group_key=dict(template)))
            bucket.total += 1
            if outcome.succeeded and (outcome.score is None or outcome.score >= _FAIL_THRESHOLD):
                bucket.succeeded += 1
                bucket.succeeding_tasks.append(outcome.task_id)
            else:
                bucket.failed += 1
                bucket.failing_tasks.append(outcome.task_id)
                reason = (outcome.failure_reason or "").lower()
                if "timed out" in reason or "timeout" in reason:
                    bucket.timeouts += 1
                if "max_steps" in reason or "max steps" in reason:
                    bucket.max_steps_failures += 1
            if outcome.elapsed_seconds is not None:
                # Running-average update.
                cur = bucket.elapsed_avg
                bucket.elapsed_avg = cur + (outcome.elapsed_seconds - cur) / max(1, bucket.total)
    return sorted(buckets.values(), key=lambda b: b.failed, reverse=True)


# ---- suggestion synthesis ---------------------------------------------


def synthesize_suggestions(buckets: list[ShapeBucket]) -> list[PolicySuggestion]:
    suggestions: list[PolicySuggestion] = []
    for bucket in buckets:
        if bucket.failed < _MIN_FAILURES_FOR_POLICY:
            continue
        evidence = tuple(bucket.failing_tasks[:6])

        # Low-risk: timeout pressure -> bump timeout_multiplier.
        if bucket.timeouts >= _MIN_FAILURES_FOR_POLICY:
            suggestions.append(
                PolicySuggestion(
                    match=dict(bucket.group_key),
                    delta={"timeout_multiplier_bump": 0.2},
                    reason=(
                        f"{bucket.timeouts}/{bucket.total} tasks in this shape timed out — "
                        "raising timeout_multiplier by +0.2 may grant partial recovery without "
                        "burning extra budget on shapes that don't time out."
                    ),
                    risk="low",
                    evidence_tasks=evidence,
                )
            )
        # Low-risk: max_steps pressure -> bump max_steps_multiplier.
        if bucket.max_steps_failures >= _MIN_FAILURES_FOR_POLICY:
            suggestions.append(
                PolicySuggestion(
                    match=dict(bucket.group_key),
                    delta={"max_steps_multiplier_bump": 0.25},
                    reason=(
                        f"{bucket.max_steps_failures}/{bucket.total} tasks hit max_steps — "
                        "raising max_steps_multiplier by +0.25 gives the loop more room."
                    ),
                    risk="low",
                    evidence_tasks=evidence,
                )
            )
        # High-risk: prompt hint suggestion (we surface it but never auto-apply).
        if bucket.failed >= 3 and bucket.timeouts == 0 and bucket.max_steps_failures == 0:
            suggestions.append(
                PolicySuggestion(
                    match=dict(bucket.group_key),
                    delta={
                        "prompt_hint_suggestion": (
                            "(human-review) shape is failing for non-timeout reasons; "
                            "consider authoring a targeted hint for this shape."
                        )
                    },
                    reason=(
                        f"{bucket.failed}/{bucket.total} failed without timing out / hitting "
                        "max_steps — failure mode is semantic. Worth manual trace review."
                    ),
                    risk="high",
                    evidence_tasks=evidence,
                )
            )
    return suggestions


def build_report(
    *,
    predictions_root: Path,
    gold_root: Path,
    dataset_root: Path,
) -> RecorderReport:
    outcomes = collect_outcomes(
        predictions_root=predictions_root,
        gold_root=gold_root,
        dataset_root=dataset_root,
    )
    buckets = cluster_by_shape(outcomes)
    suggestions = synthesize_suggestions(buckets)
    return RecorderReport(
        runs_analyzed=1,
        outcomes=tuple(outcomes),
        buckets=tuple(buckets),
        suggestions=tuple(suggestions),
    )


# ---- application ------------------------------------------------------


def apply_low_risk_suggestions(
    suggestions: list[PolicySuggestion] | tuple[PolicySuggestion, ...],
    learnings: dict[str, Any] | None = None,
    *,
    learnings_path: Path | None = None,
) -> dict[str, Any]:
    """Apply low-risk suggestions in place. Returns the new learnings dict.

    Idempotent: applying the same suggestion twice will compound the bump.
    Caller is expected to review + commit the diff before re-running.
    """
    payload = dict(learnings or load_learnings(learnings_path))
    entries = list(payload.get("shape_policies") or [])

    def _find_or_create(match: dict[str, Any]) -> dict[str, Any]:
        for existing in entries:
            if existing.get("match") == match:
                return existing
        new = {"match": dict(match), "policy": {}, "priority": 0, "evidence": []}
        entries.append(new)
        return new

    for suggestion in suggestions:
        if suggestion.risk != "low":
            continue
        target = _find_or_create(suggestion.match)
        policy = dict(target.get("policy") or {})
        if "timeout_multiplier_bump" in suggestion.delta:
            cur = float(policy.get("timeout_multiplier", 1.0))
            policy["timeout_multiplier"] = round(cur + float(suggestion.delta["timeout_multiplier_bump"]), 2)
        if "max_steps_multiplier_bump" in suggestion.delta:
            cur = float(policy.get("max_steps_multiplier", 1.0))
            policy["max_steps_multiplier"] = round(cur + float(suggestion.delta["max_steps_multiplier_bump"]), 2)
        target["policy"] = policy
        # Record evidence for traceability (deduped).
        existing_evidence = list(target.get("evidence") or [])
        for tid in suggestion.evidence_tasks:
            if tid not in existing_evidence:
                existing_evidence.append(tid)
        target["evidence"] = existing_evidence

    payload["shape_policies"] = entries
    return payload


_ERROR_HINT_TEMPLATES: dict[str, str] = {
    "sqlite_no_such_column": (
        "Call inspect_sqlite_schema BEFORE composing SELECT — column casing or table prefix"
        " often differs from what the question implies."
    ),
    "sqlite_no_such_table": (
        "Verify the SQLite file path and run inspect_sqlite_schema; the table name in the"
        " question is not always the actual table name."
    ),
    "sqlite_syntax": (
        "SQLite has stricter syntax than postgres / mysql — quote identifiers with double"
        " quotes, use || for string concatenation, and avoid window functions on older builds."
    ),
    "python_keyerror": (
        "Inspect the dict / DataFrame keys before indexing (use .keys() / df.columns) to"
        " avoid KeyError."
    ),
    "python_jsondecodeerror": (
        "JSONDecodeError typically means the file is JSON-Lines or truncated — call"
        " streaming_json_keys first to discover the actual record path."
    ),
    "python_memoryerror": (
        "Never load the full file into memory for heavy tasks — use chunked iteration"
        " (pd.read_csv(..., chunksize=...)) or streaming_json_aggregate."
    ),
    "python_timeout": (
        "execute_python is capped at 30s — delegate I/O to streaming tools or SQL"
        " aggregations and use Python only for the final small post-processing."
    ),
    "python_filenotfound": (
        "Resolve the path under context/ first via list_context — do not hardcode absolute"
        " paths."
    ),
    "python_valueerror": (
        "Inspect the dtype / shape of the value (df.dtypes, len(value)) before the"
        " operation that raised — most ValueErrors come from wrong-type / wrong-shape inputs."
    ),
    "execute_python_capped": (
        "execute_python output is truncated — print only the final aggregated result, not"
        " full DataFrame dumps."
    ),
    "read_json_capped": (
        "read_json is size-capped — switch to streaming_json_keys / streaming_json_count /"
        " streaming_json_aggregate which stream via ijson."
    ),
    "pandas_columnmissing": (
        "Use df.columns.tolist() before df[<list>] / df[<col>] — column names often differ"
        " from the question's wording."
    ),
}


def _hint_for_signature(signature: str) -> str:
    """Pick a human-readable hint for an error signature class.

    Falls back to a generic instruction when the signature is unknown so
    the agent still gets some useful guidance.
    """
    return _ERROR_HINT_TEMPLATES.get(
        signature,
        "Recurring failure observed on similar tasks — inspect the data structure with"
        " list_context / dataframe_describe / inspect_sqlite_schema before retrying.",
    )


def _shape_match_for_pattern(shape: TaskShape) -> dict[str, Any]:
    """Pick the most-discriminating shape facet for an error-pattern match.

    Priority: is_heavy > has_large_json > has_large_db > has_large_csv >
    difficulty. Default to {"difficulty": <difficulty>} when nothing else fits.
    """
    if shape.is_heavy:
        return {"is_heavy": True}
    if shape.has_large_json:
        return {"has_large_json": True}
    if shape.has_large_db:
        return {"has_large_db": True}
    if shape.has_large_csv:
        return {"has_large_csv": True}
    return {"difficulty": shape.difficulty}


def aggregate_error_patterns(outcomes: list[TaskOutcome]) -> list[dict[str, Any]]:
    """Cluster step errors by (shape facet, action, error_signature) and emit
    pattern entries ready to merge into ``error_patterns.json``.

    Only patterns with at least 2 distinct evidence tasks are emitted —
    one-off errors are noise.
    """
    grouped: dict[tuple[tuple[tuple[str, Any], ...], str, str], dict[str, Any]] = {}
    for outcome in outcomes:
        match = _shape_match_for_pattern(outcome.shape)
        match_key = tuple(sorted(match.items()))
        for err in outcome.step_errors:
            key = (match_key, err.action, err.error_signature)
            entry = grouped.setdefault(
                key,
                {
                    "match": dict(match),
                    "action": err.action,
                    "error_signature": err.error_signature,
                    "evidence_tasks": [],
                    "frequency": 0,
                },
            )
            if outcome.task_id not in entry["evidence_tasks"]:
                entry["evidence_tasks"].append(outcome.task_id)
            entry["frequency"] += 1
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out: list[dict[str, Any]] = []
    for entry in grouped.values():
        if len(entry["evidence_tasks"]) < 2:
            continue
        entry["hint"] = _hint_for_signature(entry["error_signature"])
        entry["first_seen"] = today
        entry["last_seen"] = today
        out.append(entry)
    return sorted(out, key=lambda e: e["frequency"], reverse=True)


def merge_error_patterns(
    new_patterns: list[dict[str, Any]],
    existing: dict[str, Any] | None = None,
    *,
    error_patterns_path: Path | None = None,
) -> dict[str, Any]:
    """Merge freshly observed patterns into the bundled ``error_patterns.json``.

    Existing entries with the same (match, action, error_signature) get
    their frequency / evidence updated; new keys get appended. The
    dedup key is JSON-serialized so range expressions like
    ``{"file_types": {"in": [...]}}`` (which contain nested dicts/lists)
    survive the lookup — pre-v7.1 we used a tuple-of-items which raised
    TypeError on unhashable inner values.
    """
    payload = dict(existing or load_error_patterns(error_patterns_path))
    entries = list(payload.get("patterns") or [])

    def _key(entry: dict[str, Any]) -> tuple[str, str, str]:
        match = entry.get("match") or {}
        # json.dumps with sort_keys gives a stable string-keyed identity
        # without requiring nested values to be hashable.
        return (
            json.dumps(match, sort_keys=True, default=str),
            str(entry.get("action") or ""),
            str(entry.get("error_signature") or ""),
        )

    by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for entry in entries:
        if isinstance(entry, dict):
            by_key[_key(entry)] = entry

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    for new in new_patterns:
        key = _key(new)
        existing_entry = by_key.get(key)
        if existing_entry is None:
            by_key[key] = dict(new)
            continue
        # Update in place — bump frequency, dedupe evidence, refresh last_seen.
        existing_freq = int(existing_entry.get("frequency") or 0)
        existing_entry["frequency"] = existing_freq + int(new.get("frequency") or 0)
        existing_evidence = list(existing_entry.get("evidence_tasks") or [])
        for tid in new.get("evidence_tasks") or []:
            if tid not in existing_evidence:
                existing_evidence.append(tid)
        existing_entry["evidence_tasks"] = existing_evidence
        existing_entry["last_seen"] = today
        existing_entry.setdefault("first_seen", new.get("first_seen") or today)
        # Keep the existing hint if there is one; otherwise inherit.
        if not existing_entry.get("hint"):
            existing_entry["hint"] = new.get("hint") or ""

    payload["patterns"] = sorted(
        by_key.values(),
        key=lambda e: int(e.get("frequency") or 0),
        reverse=True,
    )
    payload["last_updated"] = today
    return payload


def render_markdown(report: RecorderReport) -> str:
    """Pretty-print a markdown summary of the report."""
    lines: list[str] = []
    lines.append("# Recorder report\n")
    lines.append(f"- Outcomes ingested: **{len(report.outcomes)}**")
    failed = [o for o in report.outcomes if not o.succeeded]
    lines.append(f"- Failed (no prediction.csv): **{len(failed)}**\n")

    lines.append("## Shape buckets (by failure count)\n")
    lines.append("| match | total | succeeded | failed | timeouts | max_steps |")
    lines.append("|---|---|---|---|---|---|")
    for bucket in report.buckets:
        match_str = ", ".join(f"{k}={v}" for k, v in bucket.group_key.items())
        lines.append(
            f"| {match_str} | {bucket.total} | {bucket.succeeded} | {bucket.failed} | "
            f"{bucket.timeouts} | {bucket.max_steps_failures} |"
        )

    lines.append("\n## Suggestions\n")
    if not report.suggestions:
        lines.append("_No actionable suggestions — every shape bucket is healthy._\n")
    for suggestion in report.suggestions:
        match_str = ", ".join(f"{k}={v}" for k, v in suggestion.match.items())
        lines.append(f"### [{suggestion.risk.upper()}] {match_str}")
        lines.append(f"- **Delta:** `{json.dumps(suggestion.delta)}`")
        lines.append(f"- **Reason:** {suggestion.reason}")
        lines.append(f"- **Evidence:** {', '.join(suggestion.evidence_tasks) or '—'}\n")
    return "\n".join(lines) + "\n"


__all__ = [
    "TaskOutcome",
    "StepError",
    "ShapeBucket",
    "PolicySuggestion",
    "RecorderReport",
    "build_report",
    "collect_outcomes",
    "cluster_by_shape",
    "synthesize_suggestions",
    "apply_low_risk_suggestions",
    "aggregate_error_patterns",
    "merge_error_patterns",
    "render_markdown",
    "DEFAULT_LEARNINGS_PATH",
    "DEFAULT_ERROR_PATTERNS_PATH",
    "save_learnings",
    "save_error_patterns",
]
