from __future__ import annotations

import csv
import json
import re
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Any

# Predictions can occasionally contain very large cell values (e.g. an agent
# accidentally dumping a JSON blob into one column). Lift csv's default 128 KB
# field cap so the evaluator can still load the file instead of crashing.
csv.field_size_limit(sys.maxsize)

PREDICTION_FILENAME = "prediction.csv"
GOLD_FILENAME = "gold.csv"
SUMMARY_FILENAME = "summary.json"
EVALUATION_JSON_FILENAME = "evaluation.json"
EVALUATION_CSV_FILENAME = "evaluation.csv"

CSV_COLUMNS = [
    "task_id",
    "missing_prediction",
    "missing_gold",
    "rows_pred",
    "cols_pred",
    "rows_gold",
    "cols_gold",
    "matched_cols",
    "gold_cols",
    "pred_cols",
    "extra_cols",
    "recall",
    "extras_ratio",
    "official_score_lambda_0_0",
    "official_score_lambda_0_1",
    "official_score_lambda_0_3",
    "official_score_lambda_0_5",
    "official_score_lambda_1_0",
    "exact_match",
    "value_match",
    "value_match_unordered",
    "numeric_tolerant_match",
    "numeric_tolerant_unordered_match",
    "diagnosis",
]

DIAGNOSIS_MISSING_GOLD = "missing_gold"
DIAGNOSIS_MISSING_PREDICTION = "missing_prediction"
DIAGNOSIS_EXACT_MATCH = "exact_match"
DIAGNOSIS_COLUMN_NAMES_DIFFER = "column_names_differ"
DIAGNOSIS_SHAPE_MISMATCH = "shape_mismatch"
DIAGNOSIS_ROW_ORDER_DIFFERS = "row_order_differs"
DIAGNOSIS_NUMERIC_PRECISION_DIFFERS = "numeric_precision_differs"
DIAGNOSIS_VALUE_MISMATCH = "value_mismatch"
DIAGNOSIS_PERFECT_RECALL_NO_EXTRAS = "perfect_recall_no_extras"
DIAGNOSIS_PERFECT_RECALL_WITH_EXTRAS = "perfect_recall_with_extras"
DIAGNOSIS_PARTIAL_RECALL = "partial_recall"
DIAGNOSIS_ZERO_RECALL = "zero_recall"

_DATE_RE = re.compile(r"^\d{4}-\d{1,2}-\d{1,2}$")
_NULL_STRINGS = {"", "null", "nan", "none", "nat", "<na>"}
_DECIMAL_QUANT = Decimal("0.01")
_OFFICIAL_SCORE_LAMBDAS = {
    "official_score_lambda_0_0": Decimal("0.0"),
    "official_score_lambda_0_1": Decimal("0.1"),
    "official_score_lambda_0_3": Decimal("0.3"),
    "official_score_lambda_0_5": Decimal("0.5"),
    "official_score_lambda_1_0": Decimal("1.0"),
}


@dataclass(frozen=True, slots=True)
class EvaluationOptions:
    rtol: float = 1e-6
    atol: float = 1e-6
    preview_rows: int = 5


@dataclass(frozen=True, slots=True)
class EvaluationOutputs:
    json_path: Path
    csv_path: Path


@dataclass(frozen=True, slots=True)
class CsvTable:
    path: Path
    columns: list[str]
    rows: list[list[str]]
    raw_rows: list[list[str]]

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def column_count(self) -> int:
        return len(self.columns)


@dataclass(frozen=True, slots=True)
class TaskEvaluation:
    task_id: str
    missing_prediction: bool
    missing_gold: bool
    rows_pred: int
    cols_pred: int
    rows_gold: int
    cols_gold: int
    matched_cols: int
    gold_cols: int
    pred_cols: int
    extra_cols: int
    recall: float
    extras_ratio: float
    official_score_lambda_0_0: float
    official_score_lambda_0_1: float
    official_score_lambda_0_3: float
    official_score_lambda_0_5: float
    official_score_lambda_1_0: float
    exact_match: bool
    value_match: bool
    value_match_unordered: bool
    numeric_tolerant_match: bool
    numeric_tolerant_unordered_match: bool
    shape_match: bool
    diagnosis: str
    preview: dict[str, Any]

    def to_csv_row(self) -> dict[str, Any]:
        return {column: getattr(self, column) for column in CSV_COLUMNS}

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_run(
    run_dir: Path,
    gold_root: Path,
    options: EvaluationOptions,
) -> list[TaskEvaluation]:
    task_ids = _list_task_ids(run_dir)
    return [
        _evaluate_task(
            task_id=task_id,
            prediction_path=run_dir / task_id / PREDICTION_FILENAME,
            gold_path=gold_root / task_id / GOLD_FILENAME,
            options=options,
        )
        for task_id in task_ids
    ]


def write_evaluation_outputs(
    run_dir: Path,
    evaluations: list[TaskEvaluation],
) -> EvaluationOutputs:
    json_path = run_dir / EVALUATION_JSON_FILENAME
    csv_path = run_dir / EVALUATION_CSV_FILENAME
    _write_json(json_path, _build_json_payload(evaluations))
    _write_csv(csv_path, evaluations)
    return EvaluationOutputs(json_path=json_path, csv_path=csv_path)


def _list_task_ids(run_dir: Path) -> list[str]:
    summary_path = run_dir / SUMMARY_FILENAME
    if summary_path.exists():
        return _list_task_ids_from_summary(summary_path)
    return sorted(
        path.name for path in run_dir.iterdir() if path.is_dir() and path.name.startswith("task_")
    )


def _list_task_ids_from_summary(summary_path: Path) -> list[str]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    tasks = payload.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError(f"{summary_path} must contain a tasks list.")

    task_ids: list[str] = []
    for index, item in enumerate(tasks):
        if not isinstance(item, dict) or not isinstance(item.get("task_id"), str):
            raise ValueError(f"{summary_path} tasks[{index}] must contain a string task_id.")
        task_ids.append(item["task_id"])
    return task_ids


def _evaluate_task(
    *,
    task_id: str,
    prediction_path: Path,
    gold_path: Path,
    options: EvaluationOptions,
) -> TaskEvaluation:
    missing_prediction = not prediction_path.exists()
    missing_gold = not gold_path.exists()
    prediction = _read_csv_table(prediction_path) if not missing_prediction else None
    gold = _read_csv_table(gold_path) if not missing_gold else None

    metrics = _compare_tables(prediction, gold, options)
    diagnosis = _diagnose(
        missing_prediction=missing_prediction,
        missing_gold=missing_gold,
        **metrics,
    )

    return TaskEvaluation(
        task_id=task_id,
        missing_prediction=missing_prediction,
        missing_gold=missing_gold,
        rows_pred=prediction.row_count if prediction is not None else 0,
        cols_pred=prediction.column_count if prediction is not None else 0,
        rows_gold=gold.row_count if gold is not None else 0,
        cols_gold=gold.column_count if gold is not None else 0,
        diagnosis=diagnosis,
        preview=_build_preview(prediction, gold, options.preview_rows),
        **metrics,
    )


def _read_csv_table(path: Path) -> CsvTable:
    with path.open(newline="") as handle:
        raw_rows = [list(row) for row in csv.reader(handle)]
    if not raw_rows:
        return CsvTable(path=path, columns=[], rows=[], raw_rows=[])
    return CsvTable(path=path, columns=raw_rows[0], rows=raw_rows[1:], raw_rows=raw_rows)


def _compare_tables(
    prediction: CsvTable | None,
    gold: CsvTable | None,
    options: EvaluationOptions,
) -> dict[str, bool | float | int]:
    if prediction is None or gold is None:
        return {
            "matched_cols": 0,
            "gold_cols": gold.column_count if gold is not None else 0,
            "pred_cols": prediction.column_count if prediction is not None else 0,
            "extra_cols": 0,
            "recall": 0.0,
            "extras_ratio": 0.0,
            **{name: 0.0 for name in _OFFICIAL_SCORE_LAMBDAS},
            "exact_match": False,
            "value_match": False,
            "value_match_unordered": False,
            "numeric_tolerant_match": False,
            "numeric_tolerant_unordered_match": False,
            "shape_match": False,
        }

    shape_match = _shape_matches(prediction, gold)
    normalized_prediction = _normalized_rows(prediction)
    normalized_gold = _normalized_rows(gold)
    official_metrics = _official_column_metrics(prediction, gold)
    official_perfect_match = (
        official_metrics["matched_cols"] == official_metrics["gold_cols"]
        and official_metrics["extra_cols"] == 0
    )
    return {
        **official_metrics,
        "exact_match": prediction.raw_rows == gold.raw_rows,
        "value_match": shape_match and normalized_prediction == normalized_gold,
        "value_match_unordered": official_perfect_match,
        "numeric_tolerant_match": shape_match
        and _rows_match_official_normalized(prediction.rows, gold.rows),
        "numeric_tolerant_unordered_match": official_perfect_match,
        "shape_match": shape_match,
    }


def _shape_matches(prediction: CsvTable, gold: CsvTable) -> bool:
    return prediction.row_count == gold.row_count and prediction.column_count == gold.column_count


def _normalized_rows(table: CsvTable) -> list[list[str]]:
    return [[_normalize_cell(cell) for cell in row] for row in table.rows]


def _normalize_cell(value: str) -> str:
    normalized = str(value).strip().replace("\r\n", "").replace("\r", "").replace("\n", "")
    if normalized.lower() in _NULL_STRINGS:
        return ""
    return _normalize_decimal(normalized) or _normalize_datetime(normalized) or normalized


def _normalize_decimal(value: str) -> str | None:
    try:
        parsed = Decimal(value)
    except InvalidOperation:
        return None
    if not parsed.is_finite():
        return ""
    rounded = parsed.quantize(_DECIMAL_QUANT, rounding=ROUND_HALF_UP)
    return format(rounded, "f")


def _normalize_datetime(value: str) -> str | None:
    if _DATE_RE.match(value):
        year, month, day = (int(part) for part in value.split("-"))
        try:
            return date(year, month, day).isoformat()
        except ValueError:
            return None

    if "T" not in value and " " not in value:
        return None

    candidate = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return None

    if parsed.tzinfo is None:
        return parsed.isoformat()
    return parsed.astimezone(UTC).replace(tzinfo=None).isoformat().replace("+00:00", "") + "Z"


def _rows_match_official_normalized(
    prediction_rows: list[list[str]],
    gold_rows: list[list[str]],
) -> bool:
    return all(
        _row_matches_official_normalized(prediction_row, gold_row)
        for prediction_row, gold_row in zip(prediction_rows, gold_rows, strict=True)
    )


def _row_matches_official_normalized(
    prediction_row: list[str],
    gold_row: list[str],
) -> bool:
    return all(
        _normalize_cell(prediction_cell) == _normalize_cell(gold_cell)
        for prediction_cell, gold_cell in zip(prediction_row, gold_row, strict=True)
    )


def _official_column_metrics(prediction: CsvTable, gold: CsvTable) -> dict[str, int | float]:
    pred_vectors = _normalized_column_vectors(prediction)
    gold_vectors = _normalized_column_vectors(gold)
    matched_cols, matched_pred_cols = _count_matched_gold_columns(pred_vectors, gold_vectors)
    pred_cols = len(pred_vectors)
    gold_cols = len(gold_vectors)
    extra_cols = max(pred_cols - matched_pred_cols, 0)
    recall = Decimal(matched_cols) / Decimal(gold_cols) if gold_cols else Decimal("1.0")
    extras_ratio = Decimal(extra_cols) / Decimal(pred_cols) if pred_cols else Decimal("0.0")
    return {
        "matched_cols": matched_cols,
        "gold_cols": gold_cols,
        "pred_cols": pred_cols,
        "extra_cols": extra_cols,
        "recall": float(recall),
        "extras_ratio": float(extras_ratio),
        **{
            name: float(max(recall - penalty * extras_ratio, Decimal("0")))
            for name, penalty in _OFFICIAL_SCORE_LAMBDAS.items()
        },
    }


def _normalized_column_vectors(table: CsvTable) -> list[list[str]]:
    return [
        [
            _normalize_cell(row[column_index]) if column_index < len(row) else ""
            for row in table.rows
        ]
        for column_index in range(table.column_count)
    ]


def _count_matched_gold_columns(
    pred_vectors: list[list[str]],
    gold_vectors: list[list[str]],
) -> tuple[int, int]:
    matched_pred = [False] * len(pred_vectors)
    matched_gold = [False] * len(gold_vectors)

    for gold_index, gold_vector in enumerate(gold_vectors):
        pred_index = _find_unmatched_signature(gold_vector, pred_vectors, matched_pred)
        if pred_index is not None:
            matched_gold[gold_index] = True
            matched_pred[pred_index] = True

    _match_split_name_columns(pred_vectors, gold_vectors, matched_pred, matched_gold)

    return sum(matched_gold), sum(matched_pred)


def _find_unmatched_signature(
    target_vector: list[str],
    vectors: list[list[str]],
    matched: list[bool],
) -> int | None:
    target_signature = _column_signature(target_vector)
    for index, vector in enumerate(vectors):
        if not matched[index] and _column_signature(vector) == target_signature:
            return index
    return None


def _match_split_name_columns(
    pred_vectors: list[list[str]],
    gold_vectors: list[list[str]],
    matched_pred: list[bool],
    matched_gold: list[bool],
) -> None:
    for gold_index, gold_vector in enumerate(gold_vectors):
        if matched_gold[gold_index]:
            continue
        pred_pair = _find_unmatched_combined_pair(gold_vector, pred_vectors, matched_pred)
        if pred_pair is not None:
            matched_gold[gold_index] = True
            matched_pred[pred_pair[0]] = True
            matched_pred[pred_pair[1]] = True

    for pred_index, pred_vector in enumerate(pred_vectors):
        if matched_pred[pred_index]:
            continue
        gold_pair = _find_unmatched_combined_pair(pred_vector, gold_vectors, matched_gold)
        if gold_pair is not None:
            matched_pred[pred_index] = True
            matched_gold[gold_pair[0]] = True
            matched_gold[gold_pair[1]] = True


def _find_unmatched_combined_pair(
    target_vector: list[str],
    vectors: list[list[str]],
    matched: list[bool],
) -> tuple[int, int] | None:
    if not vectors:
        return None
    target_signature = _column_signature(target_vector)
    for left_index in range(len(vectors) - 1):
        right_index = left_index + 1
        if matched[left_index] or matched[right_index]:
            continue
        combined = [
            _normalize_cell(f"{left} {right}")
            for left, right in zip(vectors[left_index], vectors[right_index], strict=True)
        ]
        if _column_signature(combined) == target_signature:
            return left_index, right_index
    return None


def _column_signature(vector: list[str]) -> tuple[str, ...]:
    return tuple(sorted(vector))


def _diagnose(
    *,
    missing_prediction: bool,
    missing_gold: bool,
    exact_match: bool,
    value_match: bool,
    value_match_unordered: bool,
    numeric_tolerant_match: bool,
    numeric_tolerant_unordered_match: bool,
    shape_match: bool,
    matched_cols: int,
    gold_cols: int,
    pred_cols: int,
    extra_cols: int,
    recall: float,
    extras_ratio: float,
    official_score_lambda_0_0: float,
    official_score_lambda_0_1: float,
    official_score_lambda_0_3: float,
    official_score_lambda_0_5: float,
    official_score_lambda_1_0: float,
) -> str:
    if missing_gold:
        return DIAGNOSIS_MISSING_GOLD
    if missing_prediction:
        return DIAGNOSIS_MISSING_PREDICTION
    if gold_cols > 0 and matched_cols == gold_cols and extra_cols == 0:
        return DIAGNOSIS_PERFECT_RECALL_NO_EXTRAS
    if gold_cols > 0 and matched_cols == gold_cols:
        return DIAGNOSIS_PERFECT_RECALL_WITH_EXTRAS
    if matched_cols > 0:
        return DIAGNOSIS_PARTIAL_RECALL
    return DIAGNOSIS_ZERO_RECALL


def _build_preview(
    prediction: CsvTable | None,
    gold: CsvTable | None,
    preview_rows: int,
) -> dict[str, Any]:
    row_limit = max(preview_rows, 0)
    return {
        "prediction": _preview_table(prediction, row_limit),
        "gold": _preview_table(gold, row_limit),
    }


def _preview_table(table: CsvTable | None, row_limit: int) -> dict[str, Any] | None:
    if table is None:
        return None
    return {
        "path": str(table.path),
        "columns": table.columns,
        "rows": table.rows[:row_limit],
    }


def _build_json_payload(evaluations: list[TaskEvaluation]) -> dict[str, Any]:
    return {
        "task_count": len(evaluations),
        "perfect_recall_no_extras_count": sum(
            1 for item in evaluations if item.diagnosis == DIAGNOSIS_PERFECT_RECALL_NO_EXTRAS
        ),
        "perfect_recall_with_extras_count": sum(
            1 for item in evaluations if item.diagnosis == DIAGNOSIS_PERFECT_RECALL_WITH_EXTRAS
        ),
        "partial_recall_count": sum(
            1 for item in evaluations if item.diagnosis == DIAGNOSIS_PARTIAL_RECALL
        ),
        "zero_recall_count": sum(
            1 for item in evaluations if item.diagnosis == DIAGNOSIS_ZERO_RECALL
        ),
        "official_score_lambda_0_0_mean": _mean_score(evaluations, "official_score_lambda_0_0"),
        "official_score_lambda_0_1_mean": _mean_score(evaluations, "official_score_lambda_0_1"),
        "official_score_lambda_0_3_mean": _mean_score(evaluations, "official_score_lambda_0_3"),
        "official_score_lambda_0_5_mean": _mean_score(evaluations, "official_score_lambda_0_5"),
        "official_score_lambda_1_0_mean": _mean_score(evaluations, "official_score_lambda_1_0"),
        "exact_match_count": sum(1 for item in evaluations if item.exact_match),
        "value_match_count": sum(1 for item in evaluations if item.value_match),
        "value_match_unordered_count": sum(1 for item in evaluations if item.value_match_unordered),
        "numeric_tolerant_match_count": sum(
            1 for item in evaluations if item.numeric_tolerant_match
        ),
        "numeric_tolerant_unordered_match_count": sum(
            1 for item in evaluations if item.numeric_tolerant_unordered_match
        ),
        "missing_prediction_count": sum(1 for item in evaluations if item.missing_prediction),
        "missing_gold_count": sum(1 for item in evaluations if item.missing_gold),
        "tasks": [item.to_json_dict() for item in evaluations],
    }


def _mean_score(evaluations: list[TaskEvaluation], field_name: str) -> float:
    if not evaluations:
        return 0.0
    return sum(float(getattr(item, field_name)) for item in evaluations) / len(evaluations)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, evaluations: list[TaskEvaluation]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for evaluation in evaluations:
            writer.writerow(evaluation.to_csv_row())
