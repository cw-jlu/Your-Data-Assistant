from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from statistics import mean, median
from typing import Any

DEFAULT_LAMBDA_GRID = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5)
DEFAULT_PRIMARY_LAMBDA = 0.1
REVIEW_SCORE_THRESHOLD = 0.5
RULES_URL = "https://dataagent.top/rules"
SUMMARY_TASK_SOURCE = "summary.json.tasks"
NULL_SYNONYMS = frozenset({"", "null", "none", "nan", "nat", "<na>"})
NUMERIC_QUANTIZER = Decimal("0.01")
NUMERIC_PATTERN = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
DATE_PATTERN = re.compile(r"^\d{4}-\d{1,2}-\d{1,2}$")
DATETIME_PATTERN = re.compile(r"^\d{4}-\d{1,2}-\d{1,2}[T ].+$")


@dataclass(frozen=True, slots=True)
class ColumnVector:
    index: int
    name: str
    values: tuple[str, ...]
    signature: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MatchCandidate:
    gold_mask: int
    prediction_mask: int
    covered_gold_columns: int
    matched_prediction_columns: int
    mode: str


@dataclass(frozen=True, slots=True)
class TaskDiagnostics:
    difficulty: str | None
    succeeded: bool | None
    failure_reason: str | None
    e2e_elapsed_seconds: float | None
    model_step_count: int | None
    trace_step_count: int | None
    tool_call_counts: dict[str, int] | None
    node_elapsed: dict[str, list[float]] | None = None
    total_input_tokens: int | None = None
    total_output_tokens: int | None = None
    total_reasoning_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class TaskScore:
    task_id: str
    difficulty: str | None
    gold_csv_path: Path
    prediction_csv_path: Path | None
    full_cover: bool
    covered_gold_columns: int
    matched_prediction_columns: int
    gold_column_count: int
    prediction_column_count: int
    extra_prediction_columns: int
    recall: float
    redundancy_rate: float
    primary_lambda: float
    primary_proxy_score: float
    proxy_scores: dict[str, float]
    reason: str | None
    succeeded: bool | None
    failure_reason: str | None
    e2e_elapsed_seconds: float | None
    model_step_count: int | None
    trace_step_count: int | None
    tool_call_counts: dict[str, int] | None
    node_elapsed: dict[str, list[float]] | None = None
    total_input_tokens: int | None = None
    total_output_tokens: int | None = None
    total_reasoning_tokens: int | None = None

    def to_dict(self) -> dict[str, object]:
        result = {
            "task_id": self.task_id,
            "difficulty": self.difficulty,
            "gold_csv_path": str(self.gold_csv_path),
            "prediction_csv_path": str(self.prediction_csv_path) if self.prediction_csv_path else None,
            "full_cover": self.full_cover,
            "covered_gold_columns": self.covered_gold_columns,
            "matched_prediction_columns": self.matched_prediction_columns,
            "gold_column_count": self.gold_column_count,
            "prediction_column_count": self.prediction_column_count,
            "extra_prediction_columns": self.extra_prediction_columns,
            "recall": _round_metric(self.recall),
            "redundancy_rate": _round_metric(self.redundancy_rate),
            "primary_lambda": self.primary_lambda,
            "primary_proxy_score": _round_metric(self.primary_proxy_score),
            "proxy_scores": {key: _round_metric(value) for key, value in self.proxy_scores.items()},
            "reason": self.reason,
            "succeeded": self.succeeded,
            "failure_reason": self.failure_reason,
            "e2e_elapsed_seconds": _round_metric(self.e2e_elapsed_seconds),
            "step_count": self.trace_step_count,
            "model_step_count": self.model_step_count,
            "trace_step_count": self.trace_step_count,
            "tool_call_counts": self.tool_call_counts,
        }
        if self.total_input_tokens is not None:
            result["total_input_tokens"] = self.total_input_tokens
        if self.total_output_tokens is not None:
            result["total_output_tokens"] = self.total_output_tokens
        if self.total_reasoning_tokens is not None:
            result["total_reasoning_tokens"] = self.total_reasoning_tokens
        return result


@dataclass(frozen=True, slots=True)
class RunScoreSummary:
    run_id: str
    run_output_dir: Path
    score_path: Path
    score_report_path: Path
    rules_url: str
    task_source: str
    lambda_grid: list[float]
    task_count: int
    prediction_task_count: int
    primary_lambda: float
    primary_proxy_score: float
    mean_recall: float
    mean_redundancy_rate: float
    proxy_scores: dict[str, float]
    difficulty_breakdown: dict[str, dict[str, object]]
    failure_breakdown: dict[str, int]
    runtime_summary: dict[str, object]
    tasks: list[TaskScore]

    def to_dict(self) -> dict[str, object]:
        metadata = {
            "run_id": self.run_id,
            "run_output_dir": str(self.run_output_dir),
            "score_path": str(self.score_path),
            "score_report_path": str(self.score_report_path),
            "rules_url": self.rules_url,
            "task_source": self.task_source,
            "scored_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "lambda_grid": list(self.lambda_grid),
        }
        token_summary = self.runtime_summary.get("token_summary", {}) if isinstance(self.runtime_summary, dict) else {}
        total_tokens = token_summary.get("total_tokens") if isinstance(token_summary, dict) else None
        overview = {
            "task_count": self.task_count,
            "prediction_task_count": self.prediction_task_count,
            "primary_lambda": self.primary_lambda,
            "primary_proxy_score": _round_metric(self.primary_proxy_score),
            "mean_recall": _round_metric(self.mean_recall),
            "mean_redundancy_rate": _round_metric(self.mean_redundancy_rate),
            "proxy_scores": {key: _round_metric(value) for key, value in self.proxy_scores.items()},
        }
        if total_tokens is not None:
            overview["total_tokens"] = total_tokens
        result = {
            "metadata": metadata,
            "overview": overview,
            "difficulty_breakdown": self.difficulty_breakdown,
            "failure_breakdown": self.failure_breakdown,
            "runtime_summary": self.runtime_summary,
            "tasks": [task.to_dict() for task in self.tasks],
            "run_id": self.run_id,
            "run_output_dir": str(self.run_output_dir),
            "score_path": str(self.score_path),
            "score_report_path": str(self.score_report_path),
            "scored_at_utc": metadata["scored_at_utc"],
            "rules_url": self.rules_url,
            "task_source": self.task_source,
            "lambda_grid": list(self.lambda_grid),
            "task_count": self.task_count,
            "prediction_task_count": self.prediction_task_count,
            "primary_lambda": self.primary_lambda,
            "primary_proxy_score": _round_metric(self.primary_proxy_score),
            "mean_recall": _round_metric(self.mean_recall),
            "mean_redundancy_rate": _round_metric(self.mean_redundancy_rate),
            "proxy_scores": {key: _round_metric(value) for key, value in self.proxy_scores.items()},
        }
        if total_tokens is not None:
            result["total_tokens"] = total_tokens
        return result


@dataclass(frozen=True, slots=True)
class RunComparisonRow:
    input_ref: str
    run_id: str
    score_json_path: Path
    task_count: int
    prediction_task_count: int
    primary_proxy_score: float
    mean_recall: float
    mean_redundancy_rate: float
    mean_model_step_count: float
    max_model_step_count: int
    p95_e2e_elapsed_seconds: float
    top_failure: str | None


def normalize_lambda_grid(lambda_values: list[float] | tuple[float, ...] | None = None) -> list[float]:
    raw_values = list(DEFAULT_LAMBDA_GRID if not lambda_values else lambda_values)
    normalized: list[float] = []
    seen: set[float] = set()
    for value in raw_values:
        candidate = float(value)
        if not math.isfinite(candidate):
            raise ValueError(f"lambda must be finite, got {value!r}.")
        if candidate < 0:
            raise ValueError(f"lambda must be non-negative, got {value!r}.")
        if candidate in seen:
            continue
        normalized.append(candidate)
        seen.add(candidate)
    if not any(math.isclose(candidate, DEFAULT_PRIMARY_LAMBDA, rel_tol=0.0, abs_tol=1e-12) for candidate in normalized):
        normalized.append(DEFAULT_PRIMARY_LAMBDA)
    if not normalized:
        raise ValueError("lambda grid must not be empty.")
    return sorted(normalized)


def _round_metric(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)


def _score_json_path(candidate: Path) -> Path:
    if candidate.name == "score.json":
        return candidate
    return candidate / "score.json"


def resolve_comparison_score_json(
    *,
    runs_root: Path,
    run_ref: str,
    base_dir: Path | None = None,
) -> Path:
    normalized_ref = run_ref.strip()
    if not normalized_ref:
        raise ValueError("Run reference must not be empty.")

    base_dir = base_dir or Path.cwd()
    candidate = Path(normalized_ref)
    candidates: list[Path] = []
    if candidate.is_absolute():
        candidates.append(candidate)
    else:
        if len(candidate.parts) == 1:
            candidates.append(runs_root / candidate)
        candidates.append(base_dir / candidate)

    score_paths: list[Path] = []
    seen_paths: set[Path] = set()
    for candidate_path in candidates:
        score_path = _score_json_path(candidate_path)
        if score_path in seen_paths:
            continue
        seen_paths.add(score_path)
        score_paths.append(score_path)
        if score_path.is_file():
            return score_path

    rendered_paths = ", ".join(str(path) for path in score_paths)
    raise FileNotFoundError(f"Could not find score.json for {normalized_ref!r}. Tried: {rendered_paths}")


def _coerce_numeric_metric(value: Any, *, field_name: str, score_path: Path) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid numeric field `{field_name}` in {score_path}: {value!r}") from exc


def _coerce_int_metric(value: Any, *, field_name: str, score_path: Path) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid integer field `{field_name}` in {score_path}: {value!r}") from exc


def _top_failure_label(failure_breakdown: Any) -> str | None:
    if not isinstance(failure_breakdown, dict) or not failure_breakdown:
        return None

    normalized_items: list[tuple[str, int]] = []
    for reason, count in failure_breakdown.items():
        try:
            normalized_count = int(count)
        except (TypeError, ValueError):
            normalized_count = 0
        normalized_items.append((str(reason), normalized_count))

    if not normalized_items:
        return None

    reason, count = sorted(normalized_items, key=lambda item: (-item[1], item[0]))[0]
    return f"{reason} ({count})"


def _load_run_comparison_row(*, input_ref: str, score_path: Path) -> RunComparisonRow:
    try:
        payload = json.loads(score_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {score_path}: {exc.msg}.") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid score payload in {score_path}: expected a JSON object.")

    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    overview = payload.get("overview")
    if not isinstance(overview, dict):
        overview = payload
    runtime_summary = payload.get("runtime_summary")
    if not isinstance(runtime_summary, dict):
        runtime_summary = {}

    run_id = str(metadata.get("run_id") or payload.get("run_id") or score_path.parent.name)
    return RunComparisonRow(
        input_ref=input_ref,
        run_id=run_id,
        score_json_path=score_path,
        task_count=_coerce_int_metric(overview.get("task_count"), field_name="task_count", score_path=score_path),
        prediction_task_count=_coerce_int_metric(
            overview.get("prediction_task_count"),
            field_name="prediction_task_count",
            score_path=score_path,
        ),
        primary_proxy_score=_coerce_numeric_metric(
            overview.get("primary_proxy_score"),
            field_name="primary_proxy_score",
            score_path=score_path,
        ),
        mean_recall=_coerce_numeric_metric(
            overview.get("mean_recall"),
            field_name="mean_recall",
            score_path=score_path,
        ),
        mean_redundancy_rate=_coerce_numeric_metric(
            overview.get("mean_redundancy_rate"),
            field_name="mean_redundancy_rate",
            score_path=score_path,
        ),
        mean_model_step_count=_coerce_numeric_metric(
            runtime_summary.get("mean_model_step_count", 0.0),
            field_name="runtime_summary.mean_model_step_count",
            score_path=score_path,
        ),
        max_model_step_count=_coerce_int_metric(
            runtime_summary.get("max_model_step_count", 0),
            field_name="runtime_summary.max_model_step_count",
            score_path=score_path,
        ),
        p95_e2e_elapsed_seconds=_coerce_numeric_metric(
            runtime_summary.get("p95_e2e_elapsed_seconds", 0.0),
            field_name="runtime_summary.p95_e2e_elapsed_seconds",
            score_path=score_path,
        ),
        top_failure=_top_failure_label(payload.get("failure_breakdown")),
    )


def compare_run_scores(
    *,
    runs_root: Path,
    run_refs: list[str] | tuple[str, ...],
    base_dir: Path | None = None,
) -> list[RunComparisonRow]:
    if not run_refs:
        raise ValueError("At least one run reference is required.")
    return [
        _load_run_comparison_row(
            input_ref=run_ref,
            score_path=resolve_comparison_score_json(
                runs_root=runs_root,
                run_ref=run_ref,
                base_dir=base_dir,
            ),
        )
        for run_ref in run_refs
    ]


def _lambda_label(value: float) -> str:
    text = format(value, ".6f").rstrip("0").rstrip(".")
    return text or "0"


def _proxy_scores(*, recall: float, redundancy_rate: float, lambda_grid: list[float]) -> dict[str, float]:
    return {
        _lambda_label(lambda_value): max(recall - (lambda_value * redundancy_rate), 0.0)
        for lambda_value in lambda_grid
    }


def _primary_proxy_score(proxy_scores: dict[str, float], primary_lambda: float = DEFAULT_PRIMARY_LAMBDA) -> float:
    return proxy_scores[_lambda_label(primary_lambda)]


def normalize_cell(raw_value: str) -> str:
    value = raw_value.strip()
    if value.lower() in NULL_SYNONYMS:
        return ""

    normalized_datetime = _normalize_datetime(value)
    if normalized_datetime is not None:
        return normalized_datetime

    normalized_date = _normalize_date(value)
    if normalized_date is not None:
        return normalized_date

    normalized_numeric = _normalize_numeric(value)
    if normalized_numeric is not None:
        return normalized_numeric

    return value


def _normalize_numeric(value: str) -> str | None:
    if not NUMERIC_PATTERN.fullmatch(value):
        return None
    try:
        quantized = Decimal(value).quantize(NUMERIC_QUANTIZER, rounding=ROUND_HALF_UP)
    except InvalidOperation:
        return None
    return format(quantized, "f")


def _normalize_date(value: str) -> str | None:
    if not DATE_PATTERN.fullmatch(value):
        return None
    try:
        year_str, month_str, day_str = value.split("-")
        normalized = date(int(year_str), int(month_str), int(day_str))
    except ValueError:
        return None
    return normalized.isoformat()


def _normalize_datetime(value: str) -> str | None:
    if not DATETIME_PATTERN.fullmatch(value):
        return None

    parse_value = f"{value[:-1]}+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(parse_value)
    except ValueError:
        return None

    if parsed.tzinfo is not None and parsed.utcoffset() is not None:
        normalized = parsed.astimezone(timezone.utc).isoformat()
        return normalized.removesuffix("+00:00") + "Z"
    return parsed.isoformat()


def _column_signature(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(sorted(values))


def _combine_name_columns(left: ColumnVector, right: ColumnVector) -> tuple[str, ...]:
    return tuple(" ".join(part for part in (first, last) if part) for first, last in zip(left.values, right.values))


def _is_name_column(column: ColumnVector) -> bool:
    tokens = [token for token in re.split(r"[^a-z0-9]+", column.name.lower()) if token]
    return "name" in tokens


def _load_csv_columns(path: Path) -> list[ColumnVector]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))

    if not rows:
        return []

    expected_width = len(rows[0])
    columns = [[] for _ in range(expected_width)]
    for row_index, row in enumerate(rows[1:], start=2):
        if len(row) != expected_width:
            raise ValueError(
                f"CSV row width mismatch in {path} at line {row_index}: "
                f"expected {expected_width}, got {len(row)}."
            )
        for column_index, value in enumerate(row):
            columns[column_index].append(normalize_cell(value))

    return [
        ColumnVector(
            index=index,
            name=rows[0][index].strip(),
            values=tuple(values),
            signature=_column_signature(tuple(values)),
        )
        for index, values in enumerate(columns)
    ]


def _build_match_candidates(
    gold_columns: list[ColumnVector],
    prediction_columns: list[ColumnVector],
) -> list[MatchCandidate]:
    candidates: list[MatchCandidate] = []

    for gold_column in gold_columns:
        for prediction_column in prediction_columns:
            if gold_column.signature == prediction_column.signature:
                candidates.append(
                    MatchCandidate(
                        gold_mask=1 << gold_column.index,
                        prediction_mask=1 << prediction_column.index,
                        covered_gold_columns=1,
                        matched_prediction_columns=1,
                        mode="1to1",
                    )
                )

    for left_gold, right_gold in combinations(gold_columns, 2):
        if not (_is_name_column(left_gold) and _is_name_column(right_gold)):
            continue
        combined_signature = _column_signature(_combine_name_columns(left_gold, right_gold))
        for prediction_column in prediction_columns:
            if not _is_name_column(prediction_column):
                continue
            if combined_signature == prediction_column.signature:
                candidates.append(
                    MatchCandidate(
                        gold_mask=(1 << left_gold.index) | (1 << right_gold.index),
                        prediction_mask=1 << prediction_column.index,
                        covered_gold_columns=2,
                        matched_prediction_columns=1,
                        mode="2to1_name",
                    )
                )

    for gold_column in gold_columns:
        if not _is_name_column(gold_column):
            continue
        for left_prediction, right_prediction in combinations(prediction_columns, 2):
            if not (_is_name_column(left_prediction) and _is_name_column(right_prediction)):
                continue
            combined_signature = _column_signature(_combine_name_columns(left_prediction, right_prediction))
            if gold_column.signature == combined_signature:
                candidates.append(
                    MatchCandidate(
                        gold_mask=1 << gold_column.index,
                        prediction_mask=(1 << left_prediction.index) | (1 << right_prediction.index),
                        covered_gold_columns=1,
                        matched_prediction_columns=2,
                        mode="1to2_name",
                    )
                )

    candidates.sort(
        key=lambda item: (
            -item.covered_gold_columns,
            -item.matched_prediction_columns,
            item.gold_mask,
            item.prediction_mask,
            item.mode,
        )
    )
    return candidates


def _select_best_matches(
    gold_columns: list[ColumnVector],
    prediction_columns: list[ColumnVector],
) -> tuple[int, int]:
    candidates = _build_match_candidates(gold_columns, prediction_columns)

    @lru_cache(maxsize=None)
    def search(used_gold_mask: int, used_prediction_mask: int) -> tuple[int, int, tuple[int, ...]]:
        best_covered = 0
        best_prediction_columns = 0
        best_path: tuple[int, ...] = ()

        for index, candidate in enumerate(candidates):
            if candidate.gold_mask & used_gold_mask:
                continue
            if candidate.prediction_mask & used_prediction_mask:
                continue

            covered, matched_prediction_columns, path = search(
                used_gold_mask | candidate.gold_mask,
                used_prediction_mask | candidate.prediction_mask,
            )
            covered += candidate.covered_gold_columns
            matched_prediction_columns += candidate.matched_prediction_columns
            candidate_path = (index,) + path

            if covered > best_covered:
                best_covered = covered
                best_prediction_columns = matched_prediction_columns
                best_path = candidate_path
                continue
            if covered == best_covered and matched_prediction_columns > best_prediction_columns:
                best_prediction_columns = matched_prediction_columns
                best_path = candidate_path
                continue
            if (
                covered == best_covered
                and matched_prediction_columns == best_prediction_columns
                and candidate_path < best_path
            ):
                best_path = candidate_path

        return best_covered, best_prediction_columns, best_path

    covered_gold_columns, matched_prediction_columns, _ = search(0, 0)
    return covered_gold_columns, matched_prediction_columns


def _load_task_difficulty(input_root: Path, task_id: str) -> str | None:
    task_json_path = input_root / task_id / "task.json"
    if not task_json_path.exists():
        return None
    try:
        payload = json.loads(task_json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    difficulty = payload.get("difficulty")
    return str(difficulty) if difficulty is not None else None


def _load_required_summary_payload(run_output_dir: Path) -> dict[str, Any]:
    summary_path = run_output_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"score-run requires {summary_path} and only evaluates task ids recorded there."
        )
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {summary_path}: {exc.msg}.") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid summary payload in {summary_path}: expected a JSON object.")
    return payload


def _load_summary_task_selection(run_output_dir: Path) -> tuple[list[str], dict[str, dict[str, Any]]]:
    summary_path = run_output_dir / "summary.json"
    payload = _load_required_summary_payload(run_output_dir)
    tasks = payload.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError(f"Invalid summary payload in {summary_path}: `tasks` must be a list.")

    task_ids: list[str] = []
    task_map: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    for item in tasks:
        if not isinstance(item, dict):
            continue
        task_id = item.get("task_id")
        if isinstance(task_id, str):
            task_id = task_id.strip()
            if not task_id or task_id in seen:
                continue
            seen.add(task_id)
            task_ids.append(task_id)
            task_map[task_id] = item
    if not task_ids:
        raise ValueError(f"No valid task ids found in {summary_path}.")
    return task_ids, task_map


def _load_trace_payload(task_output_dir: Path) -> dict[str, Any]:
    trace_path = task_output_dir / "trace.json"
    if not trace_path.exists():
        return {}
    try:
        payload = json.loads(trace_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _build_task_diagnostics(
    *,
    task_id: str,
    run_output_dir: Path,
    input_root: Path,
    summary_task_map: dict[str, dict[str, Any]],
) -> TaskDiagnostics:
    summary_item = summary_task_map.get(task_id, {})
    task_output_dir = run_output_dir / task_id
    trace_payload = _load_trace_payload(task_output_dir)
    trace_steps = trace_payload.get("steps")
    runtime = trace_payload.get("e2e_elapsed_seconds")
    trace_step_count = len(trace_steps) if isinstance(trace_steps, list) else None
    model_step_count: int | None = None
    if isinstance(trace_steps, list):
        model_nodes = [
            step for step in trace_steps if isinstance(step, dict) and step.get("node") == "model"
        ]
        if model_nodes:
            model_step_count = len(model_nodes)
        else:
            model_step_count = len(trace_steps)

    tool_call_counts: dict[str, int] | None = None
    node_elapsed: dict[str, list[float]] | None = None
    task_input_tokens = 0
    task_output_tokens = 0
    task_reasoning_tokens = 0
    has_token_data = False
    if isinstance(trace_steps, list):
        counts: dict[str, int] = {}
        for step in trace_steps:
            if not isinstance(step, dict):
                continue
            calls = step.get("tool_calls")
            if isinstance(calls, list):
                for call in calls:
                    if isinstance(call, dict):
                        name = call.get("name")
                        if isinstance(name, str) and name:
                            counts[name] = counts.get(name, 0) + 1
            node = step.get("node")
            elapsed = step.get("elapsed_seconds")
            if isinstance(node, str) and isinstance(elapsed, (int, float)):
                if node_elapsed is None:
                    node_elapsed = {}
                node_elapsed.setdefault(node, []).append(float(elapsed))
            # Extract token usage from model_response
            model_response = step.get("model_response")
            if isinstance(model_response, dict):
                # Direct keys (model steps)
                for key, attr in (("input_tokens", "input"), ("output_tokens", "output"), ("reasoning_tokens", "reasoning")):
                    val = model_response.get(key)
                    if isinstance(val, (int, float)) and val:
                        if attr == "input":
                            task_input_tokens += int(val)
                        elif attr == "output":
                            task_output_tokens += int(val)
                        else:
                            task_reasoning_tokens += int(val)
                        has_token_data = True
                # Nested in attempts (perceive_task and other non-model nodes)
                attempts = model_response.get("attempts")
                if isinstance(attempts, list):
                    for attempt in attempts:
                        if not isinstance(attempt, dict):
                            continue
                        resp = attempt.get("response")
                        if not isinstance(resp, dict):
                            continue
                        for key, attr in (("input_tokens", "input"), ("output_tokens", "output"), ("reasoning_tokens", "reasoning")):
                            val = resp.get(key)
                            if isinstance(val, (int, float)) and val:
                                if attr == "input":
                                    task_input_tokens += int(val)
                                elif attr == "output":
                                    task_output_tokens += int(val)
                                else:
                                    task_reasoning_tokens += int(val)
                                has_token_data = True
        tool_call_counts = counts

    succeeded = summary_item.get("succeeded")
    if not isinstance(succeeded, bool):
        trace_succeeded = trace_payload.get("succeeded")
        succeeded = trace_succeeded if isinstance(trace_succeeded, bool) else None

    failure_reason = summary_item.get("failure_reason")
    if failure_reason is None:
        trace_failure_reason = trace_payload.get("failure_reason")
        failure_reason = trace_failure_reason if isinstance(trace_failure_reason, str) else None
    elif not isinstance(failure_reason, str):
        failure_reason = None

    return TaskDiagnostics(
        difficulty=_load_task_difficulty(input_root, task_id),
        succeeded=succeeded,
        failure_reason=failure_reason,
        e2e_elapsed_seconds=float(runtime) if isinstance(runtime, (int, float)) else None,
        model_step_count=model_step_count,
        trace_step_count=trace_step_count,
        tool_call_counts=tool_call_counts,
        node_elapsed=node_elapsed,
        total_input_tokens=task_input_tokens if has_token_data else None,
        total_output_tokens=task_output_tokens if has_token_data else None,
        total_reasoning_tokens=task_reasoning_tokens if has_token_data else None,
    )


def _build_task_reason(
    *,
    covered_gold_columns: int,
    gold_column_count: int,
    extra_prediction_columns: int,
    prediction_column_count: int,
) -> str | None:
    if gold_column_count == 0 and prediction_column_count == 0:
        return "Both gold and prediction contain no columns."
    if covered_gold_columns == gold_column_count and extra_prediction_columns == 0:
        return None
    if covered_gold_columns == gold_column_count:
        return f"All gold columns covered, with {extra_prediction_columns} extra prediction column(s)."
    if covered_gold_columns == 0:
        return (
            f"No gold columns matched; {prediction_column_count} prediction column(s) were produced and "
            f"{extra_prediction_columns} are redundant."
        )
    return (
        f"Covered {covered_gold_columns}/{gold_column_count} gold column(s), with "
        f"{extra_prediction_columns} extra prediction column(s)."
    )


def _score_task(
    *,
    task_id: str,
    gold_csv_path: Path,
    prediction_csv_path: Path | None,
    lambda_grid: list[float],
    diagnostics: TaskDiagnostics,
) -> TaskScore:
    gold_columns = _load_csv_columns(gold_csv_path)
    gold_column_count = len(gold_columns)

    if prediction_csv_path is None or not prediction_csv_path.exists():
        recall = 0.0
        redundancy_rate = 0.0
        proxy_scores = _proxy_scores(recall=recall, redundancy_rate=redundancy_rate, lambda_grid=lambda_grid)
        return TaskScore(
            task_id=task_id,
            difficulty=diagnostics.difficulty,
            gold_csv_path=gold_csv_path,
            prediction_csv_path=prediction_csv_path,
            full_cover=False,
            covered_gold_columns=0,
            matched_prediction_columns=0,
            gold_column_count=gold_column_count,
            prediction_column_count=0,
            extra_prediction_columns=0,
            recall=recall,
            redundancy_rate=redundancy_rate,
            primary_lambda=DEFAULT_PRIMARY_LAMBDA,
            primary_proxy_score=_primary_proxy_score(proxy_scores),
            proxy_scores=proxy_scores,
            reason="prediction.csv is missing.",
            succeeded=diagnostics.succeeded,
            failure_reason=diagnostics.failure_reason,
            e2e_elapsed_seconds=diagnostics.e2e_elapsed_seconds,
            model_step_count=diagnostics.model_step_count,
            trace_step_count=diagnostics.trace_step_count,
            tool_call_counts=diagnostics.tool_call_counts,
            node_elapsed=diagnostics.node_elapsed,
            total_input_tokens=diagnostics.total_input_tokens,
            total_output_tokens=diagnostics.total_output_tokens,
            total_reasoning_tokens=diagnostics.total_reasoning_tokens,
        )

    try:
        prediction_columns = _load_csv_columns(prediction_csv_path)
    except ValueError as exc:
        recall = 0.0
        redundancy_rate = 0.0
        proxy_scores = _proxy_scores(recall=recall, redundancy_rate=redundancy_rate, lambda_grid=lambda_grid)
        return TaskScore(
            task_id=task_id,
            difficulty=diagnostics.difficulty,
            gold_csv_path=gold_csv_path,
            prediction_csv_path=prediction_csv_path,
            full_cover=False,
            covered_gold_columns=0,
            matched_prediction_columns=0,
            gold_column_count=gold_column_count,
            prediction_column_count=0,
            extra_prediction_columns=0,
            recall=recall,
            redundancy_rate=redundancy_rate,
            primary_lambda=DEFAULT_PRIMARY_LAMBDA,
            primary_proxy_score=_primary_proxy_score(proxy_scores),
            proxy_scores=proxy_scores,
            reason=str(exc),
            succeeded=diagnostics.succeeded,
            failure_reason=diagnostics.failure_reason,
            e2e_elapsed_seconds=diagnostics.e2e_elapsed_seconds,
            model_step_count=diagnostics.model_step_count,
            trace_step_count=diagnostics.trace_step_count,
            tool_call_counts=diagnostics.tool_call_counts,
            node_elapsed=diagnostics.node_elapsed,
            total_input_tokens=diagnostics.total_input_tokens,
            total_output_tokens=diagnostics.total_output_tokens,
            total_reasoning_tokens=diagnostics.total_reasoning_tokens,
        )

    prediction_column_count = len(prediction_columns)
    covered_gold_columns, matched_prediction_columns = _select_best_matches(gold_columns, prediction_columns)
    extra_prediction_columns = max(prediction_column_count - matched_prediction_columns, 0)
    recall = covered_gold_columns / gold_column_count if gold_column_count > 0 else 0.0
    redundancy_rate = (
        extra_prediction_columns / prediction_column_count if prediction_column_count > 0 else 0.0
    )
    full_cover = gold_column_count > 0 and covered_gold_columns == gold_column_count
    proxy_scores = _proxy_scores(recall=recall, redundancy_rate=redundancy_rate, lambda_grid=lambda_grid)

    return TaskScore(
        task_id=task_id,
        difficulty=diagnostics.difficulty,
        gold_csv_path=gold_csv_path,
        prediction_csv_path=prediction_csv_path,
        full_cover=full_cover,
        covered_gold_columns=covered_gold_columns,
        matched_prediction_columns=matched_prediction_columns,
        gold_column_count=gold_column_count,
        prediction_column_count=prediction_column_count,
        extra_prediction_columns=extra_prediction_columns,
        recall=recall,
        redundancy_rate=redundancy_rate,
        primary_lambda=DEFAULT_PRIMARY_LAMBDA,
        primary_proxy_score=_primary_proxy_score(proxy_scores),
        proxy_scores=proxy_scores,
        reason=_build_task_reason(
            covered_gold_columns=covered_gold_columns,
            gold_column_count=gold_column_count,
            extra_prediction_columns=extra_prediction_columns,
            prediction_column_count=prediction_column_count,
        ),
        succeeded=diagnostics.succeeded,
        failure_reason=diagnostics.failure_reason,
        e2e_elapsed_seconds=diagnostics.e2e_elapsed_seconds,
        model_step_count=diagnostics.model_step_count,
        trace_step_count=diagnostics.trace_step_count,
        tool_call_counts=diagnostics.tool_call_counts,
        node_elapsed=diagnostics.node_elapsed,
        total_input_tokens=diagnostics.total_input_tokens,
        total_output_tokens=diagnostics.total_output_tokens,
        total_reasoning_tokens=diagnostics.total_reasoning_tokens,
    )


def resolve_score_run_dir(runs_root: Path, run_id: str | None = None) -> tuple[str, Path]:
    if run_id is not None:
        normalized = run_id.strip()
        if not normalized:
            raise ValueError("run_id must not be empty.")
        if normalized in {".", ".."} or "/" in normalized or "\\" in normalized:
            raise ValueError("run_id must be a single directory name, not a path.")
        run_output_dir = runs_root / normalized
        if not run_output_dir.is_dir():
            raise FileNotFoundError(f"Run directory not found: {run_output_dir}")
        return normalized, run_output_dir

    candidate_dirs = [path for path in runs_root.iterdir() if path.is_dir()]
    if not candidate_dirs:
        raise FileNotFoundError(f"No run directories found under {runs_root}")

    latest_run_dir = max(candidate_dirs, key=lambda path: path.name)
    return latest_run_dir.name, latest_run_dir


def _aggregate_proxy_scores(tasks: list[TaskScore], lambda_grid: list[float]) -> dict[str, float]:
    if not tasks:
        return {_lambda_label(lambda_value): 0.0 for lambda_value in lambda_grid}
    return {
        _lambda_label(lambda_value): mean(
            max(task.recall - (lambda_value * task.redundancy_rate), 0.0) for task in tasks
        )
        for lambda_value in lambda_grid
    }


def _build_difficulty_breakdown(tasks: list[TaskScore], lambda_grid: list[float]) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[TaskScore]] = {}
    for task in tasks:
        difficulty = task.difficulty or "unknown"
        grouped.setdefault(difficulty, []).append(task)

    breakdown: dict[str, dict[str, object]] = {}
    for difficulty in sorted(grouped):
        group_tasks = grouped[difficulty]
        task_count = len(group_tasks)
        prediction_task_count = sum(1 for task in group_tasks if task.prediction_csv_path is not None)
        full_cover_count = sum(1 for task in group_tasks if task.full_cover)
        proxy_scores = _aggregate_proxy_scores(group_tasks, lambda_grid)
        breakdown[difficulty] = {
            "task_count": task_count,
            "prediction_task_count": prediction_task_count,
            "full_cover_count": full_cover_count,
            "primary_proxy_score": _round_metric(_primary_proxy_score(proxy_scores)),
            "mean_recall": _round_metric(mean(task.recall for task in group_tasks) if group_tasks else 0.0),
            "mean_redundancy_rate": _round_metric(
                mean(task.redundancy_rate for task in group_tasks) if group_tasks else 0.0
            ),
            "proxy_scores": {key: _round_metric(value) for key, value in proxy_scores.items()},
        }
    return breakdown


def _build_failure_breakdown(tasks: list[TaskScore]) -> dict[str, int]:
    breakdown: dict[str, int] = {}
    for task in tasks:
        if task.failure_reason is None:
            continue
        breakdown[task.failure_reason] = breakdown.get(task.failure_reason, 0) + 1
    return dict(sorted(breakdown.items(), key=lambda item: (-item[1], item[0])))


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] + ((ordered[upper] - ordered[lower]) * weight)


def _build_runtime_summary(tasks: list[TaskScore], total_elapsed_seconds: float | None = None) -> dict[str, object]:
    runtimes = [task.e2e_elapsed_seconds for task in tasks if task.e2e_elapsed_seconds is not None]
    model_step_counts = [float(task.model_step_count) for task in tasks if task.model_step_count is not None]
    trace_step_counts = [float(task.trace_step_count) for task in tasks if task.trace_step_count is not None]
    
    fallback_total = sum(runtimes) if runtimes else 0.0
    effective_total = total_elapsed_seconds if total_elapsed_seconds is not None else fallback_total

    all_tool_counts: dict[str, list[float]] = {}
    task_tool_usage: dict[str, int] = {}
    tasks_with_tool_traces = sum(1 for task in tasks if task.tool_call_counts is not None)
    for task in tasks:
        if task.tool_call_counts is None:
            continue
        for tool_name, count in task.tool_call_counts.items():
            all_tool_counts.setdefault(tool_name, []).append(float(count))
            task_tool_usage[tool_name] = task_tool_usage.get(tool_name, 0) + 1

    tool_call_stats: dict[str, object] = {}
    if tasks_with_tool_traces > 0:
        tool_call_stats["available_trace_count"] = tasks_with_tool_traces
        tool_call_stats["tools"] = {
            tool_name: {
                "total_calls": int(sum(counts)),
                "mean_per_task": _round_metric(mean(counts)),
                "max_calls": int(max(counts)),
                "tasks_used": task_tool_usage[tool_name],
            }
            for tool_name, counts in sorted(all_tool_counts.items())
        }

    all_node_times: dict[str, list[float]] = {}
    for task in tasks:
        if task.node_elapsed:
            for node, times in task.node_elapsed.items():
                all_node_times.setdefault(node, []).extend(times)

    node_timing_summary: dict[str, dict[str, object]] = {}
    for node in sorted(all_node_times.keys()):
        times = all_node_times[node]
        node_timing_summary[node] = {
            "count": len(times),
            "min": _round_metric(min(times)),
            "max": _round_metric(max(times)),
            "mean": _round_metric(mean(times)),
            "median": _round_metric(median(times)),
            "total": _round_metric(sum(times)),
        }

    # Aggregate token usage across all tasks
    tasks_with_tokens = [t for t in tasks if t.total_input_tokens is not None or t.total_output_tokens is not None]
    total_input_tokens = sum(t.total_input_tokens or 0 for t in tasks_with_tokens)
    total_output_tokens = sum(t.total_output_tokens or 0 for t in tasks_with_tokens)
    total_reasoning_tokens = sum(t.total_reasoning_tokens or 0 for t in tasks_with_tokens)
    total_tokens = total_input_tokens + total_output_tokens

    token_summary: dict[str, object] = {}
    if tasks_with_tokens:
        input_per_task = [t.total_input_tokens or 0 for t in tasks_with_tokens]
        output_per_task = [t.total_output_tokens or 0 for t in tasks_with_tokens]
        token_summary = {
            "available_token_trace_count": len(tasks_with_tokens),
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_reasoning_tokens": total_reasoning_tokens,
            "total_tokens": total_tokens,
            "mean_input_tokens_per_task": _round_metric(mean(input_per_task)),
            "mean_output_tokens_per_task": _round_metric(mean(output_per_task)),
            "max_input_tokens_per_task": int(max(input_per_task)) if input_per_task else 0,
            "max_output_tokens_per_task": int(max(output_per_task)) if output_per_task else 0,
        }

    return {
        "available_runtime_count": len(runtimes),
        "total_e2e_elapsed_seconds": _round_metric(effective_total),
        "mean_e2e_elapsed_seconds": _round_metric(mean(runtimes) if runtimes else 0.0),
        "median_e2e_elapsed_seconds": _round_metric(median(runtimes) if runtimes else 0.0),
        "p95_e2e_elapsed_seconds": _round_metric(_percentile(runtimes, 0.95) if runtimes else 0.0),
        "max_e2e_elapsed_seconds": _round_metric(max(runtimes) if runtimes else 0.0),
        "available_model_step_count": len(model_step_counts),
        "mean_model_step_count": _round_metric(mean(model_step_counts) if model_step_counts else 0.0),
        "max_model_step_count": int(max(model_step_counts)) if model_step_counts else 0,
        "available_trace_step_count": len(trace_step_counts),
        "mean_trace_step_count": _round_metric(mean(trace_step_counts) if trace_step_counts else 0.0),
        "max_trace_step_count": int(max(trace_step_counts)) if trace_step_counts else 0,
        "available_step_count": len(trace_step_counts),
        "mean_step_count": _round_metric(mean(trace_step_counts) if trace_step_counts else 0.0),
        "max_step_count": int(max(trace_step_counts)) if trace_step_counts else 0,
        "tool_call_stats": tool_call_stats,
        "node_timing_summary": node_timing_summary,
        "token_summary": token_summary,
    }


def _task_id_sort_key(task_id: str) -> tuple[int, int | str, str]:
    match = re.search(r"\d+", task_id)
    if match is None:
        return (1, task_id, task_id)
    return (0, int(match.group()), task_id)


def _task_sort_key(task: TaskScore) -> tuple[int, int | str, str]:
    return _task_id_sort_key(task.task_id)


def _difficulty_sort_key(difficulty: str) -> tuple[int, str]:
    difficulty_order = {"easy": 0, "medium": 1, "hard": 2, "extreme": 3, "unknown": 4}
    normalized = difficulty.lower()
    return (difficulty_order.get(normalized, 5), normalized)


def _select_review_tasks(tasks: list[TaskScore]) -> list[TaskScore]:
    candidates = [task for task in tasks if task.primary_proxy_score < REVIEW_SCORE_THRESHOLD]
    return sorted(candidates, key=_task_sort_key)


def _render_markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _build_score_report(summary: RunScoreSummary) -> str:
    proxy_rows = [[label, f"{score:.4f}"] for label, score in summary.proxy_scores.items()]
    primary_lambda_label = _lambda_label(summary.primary_lambda)
    overview_rows = [
        ["任务总数", str(summary.task_count)],
        ["生成 prediction.csv 的任务数", str(summary.prediction_task_count)],
        [f"Primary Score (λ={primary_lambda_label})", f"{summary.primary_proxy_score:.4f}"],
        ["Mean Recall", f"{summary.mean_recall:.4f}"],
        ["Mean Redundancy Rate", f"{summary.mean_redundancy_rate:.4f}"],
    ]
    # Add total token consumption to overview if available
    _runtime = summary.runtime_summary
    _token_summary = _runtime.get("token_summary", {}) if isinstance(_runtime, dict) else {}
    _total_tokens = _token_summary.get("total_tokens") if isinstance(_token_summary, dict) else None
    if _total_tokens is not None:
        overview_rows.append(["总 Token 消耗", f"{_total_tokens:,}"])

    difficulty_rows = []
    for difficulty, payload in sorted(summary.difficulty_breakdown.items(), key=lambda item: _difficulty_sort_key(item[0])):
        difficulty_rows.append(
            [
                difficulty,
                str(payload["task_count"]),
                str(payload["prediction_task_count"]),
                str(payload["full_cover_count"]),
                f"{float(payload['primary_proxy_score']):.4f}",
                f"{float(payload['mean_recall']):.4f}",
                f"{float(payload['mean_redundancy_rate']):.4f}",
            ]
        )

    failure_rows = [
        [reason, str(count)] for reason, count in summary.failure_breakdown.items()
    ] or [["无", "0"]]

    runtime = summary.runtime_summary
    total_minutes = float(runtime.get('total_e2e_elapsed_seconds', 0.0)) / 60.0
    runtime_rows = [
        ["可用耗时任务数", str(runtime["available_runtime_count"])],
        ["总时长（分钟）", f"{total_minutes:.2f}"],
        ["平均耗时（秒）", f"{float(runtime['mean_e2e_elapsed_seconds']):.3f}"],
        ["中位耗时（秒）", f"{float(runtime['median_e2e_elapsed_seconds']):.3f}"],
        ["P95 耗时（秒）", f"{float(runtime['p95_e2e_elapsed_seconds']):.3f}"],
        ["最长耗时（秒）", f"{float(runtime['max_e2e_elapsed_seconds']):.3f}"],
        ["可用模型轮数任务数", str(runtime["available_model_step_count"])],
        ["平均模型轮数", f"{float(runtime['mean_model_step_count']):.2f}"],
        ["最大模型轮数", str(runtime["max_model_step_count"])],
    ]

    review_rows = []
    review_tasks = _select_review_tasks(summary.tasks)
    for task in review_tasks:
        review_rows.append(
            [
                task.task_id,
                task.difficulty or "unknown",
                f"{task.primary_proxy_score:.4f}",
                f"{task.recall:.4f}",
                f"{task.redundancy_rate:.4f}",
                "yes" if task.full_cover else "no",
                task.failure_reason or (task.reason or "-"),
                "-" if task.model_step_count is None else str(task.model_step_count),
                "-" if task.e2e_elapsed_seconds is None else f"{task.e2e_elapsed_seconds:.3f}",
            ]
        )
    if not review_rows:
        review_rows = [["无", "-", "-", "-", "-", "-", "-", "-", "-"]]

    appendix_rows = [
        [
            task.task_id,
            task.difficulty or "unknown",
            str(task.gold_column_count),
            str(task.prediction_column_count),
            str(task.covered_gold_columns),
            str(task.extra_prediction_columns),
            f"{task.primary_proxy_score:.4f}",
            f"{task.recall:.4f}",
            f"{task.redundancy_rate:.4f}",
            "yes" if task.full_cover else "no",
            "-" if task.model_step_count is None else str(task.model_step_count),
            "-" if task.e2e_elapsed_seconds is None else f"{task.e2e_elapsed_seconds:.3f}",
            task.failure_reason or (task.reason or "-"),
        ]
        for task in sorted(summary.tasks, key=_task_sort_key)
    ]

    sections = [
        f"# {summary.run_id} 评测报告",
        "",
        "## 执行摘要",
        "",
        f"- 评分规则来源：[{summary.rules_url}]({summary.rules_url})",
        f"- 评分范围来源：当前 run 目录下的 `{summary.task_source}`。",
        f"- 本地结果采用“默认主分 + 多 λ 分析”体系，默认主分固定为 `λ={primary_lambda_label}`。",
        f"- 当前评分网格为 `{', '.join(_lambda_label(item) for item in summary.lambda_grid)}`。",
        "- 默认主分用于稳定比较默认冗余惩罚下的表现，多 λ 结果用于观察敏感度；不代表官方未公开 λ 下的唯一得分。",
        "- `Recall`：单题覆盖率，计算方式为 `覆盖Gold列数 / Gold列总数`。",
        "- `Mean Recall`：所有任务 `Recall` 的平均值，表示平均每题覆盖了多少 gold 列。",
        "- `Mean Redundancy` / `Mean Redundancy Rate`：所有任务冗余率的平均值，表示平均每题预测列中有多少比例是多余列。",
        "",
        "## 主分与基础指标",
        "",
        _render_markdown_table(["指标", "值"], overview_rows),
        "",
        "## 多 λ 代理分数",
        "",
        _render_markdown_table(["λ", "代理分数"], proxy_rows),
        "",
        "## 按难度拆分表现",
        "",
        _render_markdown_table(
            ["难度", "任务数", "有预测", "完全正确题数", f"Primary(λ={primary_lambda_label})", "Mean Recall", "Mean Redundancy"],
            difficulty_rows or [["无", "0", "0", "0", "0.0000", "0.0000", "0.0000"]],
        ),
        "",
        "## 失败原因与耗时分析",
        "",
        "### 失败原因",
        "",
        _render_markdown_table(["失败原因", "任务数"], failure_rows),
        "",
        "### 运行时摘要",
        "",
        _render_markdown_table(["指标", "值"], runtime_rows),
        "",
    ]

    node_timing_summary = runtime.get("node_timing_summary", {})
    if node_timing_summary:
        node_rows = [
            [
                node,
                str(stats["count"]),
                f"{float(stats['min']):.3f}",
                f"{float(stats['mean']):.3f}",
                f"{float(stats['median']):.3f}",
                f"{float(stats['max']):.3f}",
                f"{float(stats['total']):.3f}",
            ]
            for node, stats in sorted(node_timing_summary.items())
        ]
        sections.extend([
            "### 节点耗时分析",
            "",
            _render_markdown_table(
                ["节点", "次数", "最小(秒)", "平均(秒)", "中位(秒)", "最大(秒)", "总耗时(秒)"],
                node_rows,
            ),
            "",
        ])

    tool_call_stats = runtime.get("tool_call_stats", {})
    if tool_call_stats.get("tools"):
        tool_rows = [
            [
                tool_name,
                str(tool_data["total_calls"]),
                f"{tool_data['mean_per_task']:.2f}",
                str(tool_data["tasks_used"]),
            ]
            for tool_name, tool_data in sorted(tool_call_stats["tools"].items())
        ]
        sections.extend([
            "### 工具调用统计",
            "",
            _render_markdown_table(["工具名称", "调用总次数", "平均每任务", "使用任务数"], tool_rows),
            "",
        ])

    token_summary = runtime.get("token_summary", {})
    if token_summary:
        token_rows = [
            ["有 token 数据的任务数", str(token_summary.get("available_token_trace_count", 0))],
            ["总 Input Tokens", f"{token_summary.get('total_input_tokens', 0):,}"],
            ["总 Output Tokens", f"{token_summary.get('total_output_tokens', 0):,}"],
            ["总 Reasoning Tokens", f"{token_summary.get('total_reasoning_tokens', 0):,}"],
            ["总 Token 消耗（Input + Output）", f"{token_summary.get('total_tokens', 0):,}"],
            ["平均 Input Tokens / 任务", f"{float(token_summary.get('mean_input_tokens_per_task', 0)):,.1f}"],
            ["平均 Output Tokens / 任务", f"{float(token_summary.get('mean_output_tokens_per_task', 0)):,.1f}"],
            ["最大 Input Tokens（单任务）", f"{token_summary.get('max_input_tokens_per_task', 0):,}"],
            ["最大 Output Tokens（单任务）", f"{token_summary.get('max_output_tokens_per_task', 0):,}"],
        ]
        sections.extend([
            "### Token 消耗统计",
            "",
            _render_markdown_table(["指标", "值"], token_rows),
            "",
        ])

    sections.extend([
        f"## 最值得复盘的任务（Primary < {REVIEW_SCORE_THRESHOLD:g}，共 {len(review_tasks)} 题）",
        "",
        _render_markdown_table(
            ["任务", "难度", f"Primary(λ={primary_lambda_label})", "Recall", "Redundancy", "Full Cover", "失败/备注", "模型轮数", "耗时(秒)"],
            review_rows,
        ),
        "",
        "## 全量任务附录",
        "",
        _render_markdown_table(
            [
                "任务",
                "难度",
                "Gold列数",
                "预测列数",
                "覆盖Gold列数",
                "冗余列数",
                f"Primary(λ={primary_lambda_label})",
                "Recall",
                "Redundancy",
                "Full Cover",
                "模型轮数",
                "耗时(秒)",
                "失败/备注",
            ],
            appendix_rows,
        ),
        "",
    ])
    return "\n".join(sections)


def score_run_outputs(
    *,
    run_output_dir: Path,
    gold_root: Path,
    lambda_values: list[float] | tuple[float, ...] | None = None,
    rules_url: str = RULES_URL,
) -> RunScoreSummary:
    if not gold_root.is_dir():
        raise FileNotFoundError(f"Gold output directory not found: {gold_root}")

    input_root = gold_root.parent / "input"
    lambda_grid = normalize_lambda_grid(lambda_values)
    scored_task_ids, summary_task_map = _load_summary_task_selection(run_output_dir)
    try:
        summary_payload = _load_required_summary_payload(run_output_dir)
        total_elapsed_seconds = summary_payload.get("total_elapsed_seconds")
    except Exception:
        total_elapsed_seconds = None

    tasks: list[TaskScore] = []
    for task_id in scored_task_ids:
        gold_csv_path = gold_root / task_id / "gold.csv"
        if not gold_csv_path.exists():
            raise FileNotFoundError(f"Gold file not found for task {task_id}: {gold_csv_path}")

        prediction_csv_path = run_output_dir / task_id / "prediction.csv"
        if not prediction_csv_path.exists():
            prediction_csv_path = None

        diagnostics = _build_task_diagnostics(
            task_id=task_id,
            run_output_dir=run_output_dir,
            input_root=input_root,
            summary_task_map=summary_task_map,
        )
        tasks.append(
            _score_task(
                task_id=task_id,
                gold_csv_path=gold_csv_path,
                prediction_csv_path=prediction_csv_path,
                lambda_grid=lambda_grid,
                diagnostics=diagnostics,
            )
        )

    task_count = len(tasks)
    prediction_task_count = sum(1 for task in tasks if task.prediction_csv_path is not None)
    mean_recall = mean(task.recall for task in tasks) if tasks else 0.0
    mean_redundancy_rate = mean(task.redundancy_rate for task in tasks) if tasks else 0.0
    proxy_scores = _aggregate_proxy_scores(tasks, lambda_grid)
    primary_proxy_score = _primary_proxy_score(proxy_scores)
    score_path = run_output_dir / "score.json"
    score_report_path = run_output_dir / "score_report.md"

    summary = RunScoreSummary(
        run_id=run_output_dir.name,
        run_output_dir=run_output_dir,
        score_path=score_path,
        score_report_path=score_report_path,
        rules_url=rules_url,
        task_source=SUMMARY_TASK_SOURCE,
        lambda_grid=lambda_grid,
        task_count=task_count,
        prediction_task_count=prediction_task_count,
        primary_lambda=DEFAULT_PRIMARY_LAMBDA,
        primary_proxy_score=primary_proxy_score,
        mean_recall=mean_recall,
        mean_redundancy_rate=mean_redundancy_rate,
        proxy_scores=proxy_scores,
        difficulty_breakdown=_build_difficulty_breakdown(tasks, lambda_grid),
        failure_breakdown=_build_failure_breakdown(tasks),
        runtime_summary=_build_runtime_summary(tasks, total_elapsed_seconds),
        tasks=tasks,
    )

    score_path.write_text(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    score_report_path.write_text(_build_score_report(summary), encoding="utf-8")
    return summary
