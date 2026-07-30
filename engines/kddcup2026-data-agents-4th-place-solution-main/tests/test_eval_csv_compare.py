from __future__ import annotations

import csv
import json
from pathlib import Path

from kobushi_core.eval import EvaluationOptions, evaluate_run, write_evaluation_outputs


def write_csv(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


def write_summary(run_dir: Path, task_ids: list[str]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": "test-run",
        "task_count": len(task_ids),
        "succeeded_task_count": len(task_ids),
        "max_workers": 1,
        "tasks": [{"task_id": task_id} for task_id in task_ids],
    }
    (run_dir / "summary.json").write_text(json.dumps(payload), encoding="utf-8")


def evaluate_single(tmp_path: Path, task_id: str) -> object:
    evaluations = evaluate_run(
        tmp_path / "run",
        tmp_path / "gold",
        EvaluationOptions(rtol=1e-6, atol=1e-6, preview_rows=5),
    )
    return {item.task_id: item for item in evaluations}[task_id]


def test_should_report_exact_match_when_prediction_and_gold_csv_are_identical(
    tmp_path: Path,
) -> None:
    # Given
    task_id = "task_1"
    write_summary(tmp_path / "run", [task_id])
    rows = [["name", "score"], ["alice", "1"], ["bob", "2"]]
    write_csv(tmp_path / "run" / task_id / "prediction.csv", rows)
    write_csv(tmp_path / "gold" / task_id / "gold.csv", rows)

    # When
    result = evaluate_single(tmp_path, task_id)

    # Then
    assert result.exact_match is True
    assert result.value_match is True
    assert result.value_match_unordered is True
    assert result.numeric_tolerant_match is True
    assert result.numeric_tolerant_unordered_match is True
    assert result.shape_match is True
    assert result.matched_cols == 2
    assert result.recall == 1.0
    assert result.extras_ratio == 0.0
    assert result.official_score_lambda_0_5 == 1.0
    assert result.diagnosis == "perfect_recall_no_extras"


def test_should_ignore_column_names_for_value_match_when_values_are_equal(
    tmp_path: Path,
) -> None:
    # Given
    task_id = "task_2"
    write_summary(tmp_path / "run", [task_id])
    write_csv(tmp_path / "run" / task_id / "prediction.csv", [["answer"], ["alpha"]])
    write_csv(tmp_path / "gold" / task_id / "gold.csv", [["expected"], ["alpha"]])

    # When
    result = evaluate_single(tmp_path, task_id)

    # Then
    assert result.exact_match is False
    assert result.value_match is True
    assert result.numeric_tolerant_match is True
    assert result.diagnosis == "perfect_recall_no_extras"


def test_should_match_column_signature_when_row_order_differs(
    tmp_path: Path,
) -> None:
    # Given
    task_id = "task_3"
    write_summary(tmp_path / "run", [task_id])
    write_csv(tmp_path / "run" / task_id / "prediction.csv", [["value"], ["b"], ["a"]])
    write_csv(tmp_path / "gold" / task_id / "gold.csv", [["value"], ["a"], ["b"]])

    # When
    result = evaluate_single(tmp_path, task_id)

    # Then
    assert result.value_match is False
    assert result.value_match_unordered is True
    assert result.numeric_tolerant_unordered_match is True
    assert result.diagnosis == "perfect_recall_no_extras"


def test_should_round_decimal_half_up_before_matching(
    tmp_path: Path,
) -> None:
    # Given
    task_id = "task_4"
    write_summary(tmp_path / "run", [task_id])
    write_csv(tmp_path / "run" / task_id / "prediction.csv", [["value"], ["1.004"]])
    write_csv(tmp_path / "gold" / task_id / "gold.csv", [["value"], ["1.00"]])

    # When
    result = evaluate_single(tmp_path, task_id)

    # Then
    assert result.value_match is True
    assert result.value_match_unordered is True
    assert result.numeric_tolerant_match is True
    assert result.numeric_tolerant_unordered_match is True
    assert result.diagnosis == "perfect_recall_no_extras"


def test_should_not_use_math_isclose_for_decimal_matching(tmp_path: Path) -> None:
    # Given
    task_id = "task_4b"
    write_summary(tmp_path / "run", [task_id])
    write_csv(tmp_path / "run" / task_id / "prediction.csv", [["value"], ["1.004"]])
    write_csv(tmp_path / "gold" / task_id / "gold.csv", [["value"], ["1.005"]])

    # When
    result = evaluate_single(tmp_path, task_id)

    # Then
    assert result.numeric_tolerant_match is False
    assert result.numeric_tolerant_unordered_match is False
    assert result.diagnosis == "zero_recall"


def test_should_treat_zero_and_zero_point_zero_as_numeric_tolerant_match(
    tmp_path: Path,
) -> None:
    # Given
    task_id = "task_5"
    write_summary(tmp_path / "run", [task_id])
    write_csv(tmp_path / "run" / task_id / "prediction.csv", [["value"], ["0"]])
    write_csv(tmp_path / "gold" / task_id / "gold.csv", [["value"], ["0.0"]])

    # When
    result = evaluate_single(tmp_path, task_id)

    # Then
    assert result.value_match is True
    assert result.numeric_tolerant_match is True
    assert result.numeric_tolerant_unordered_match is True
    assert result.diagnosis == "perfect_recall_no_extras"


def test_should_report_shape_mismatch_when_row_or_column_counts_differ(
    tmp_path: Path,
) -> None:
    # Given
    task_id = "task_6"
    write_summary(tmp_path / "run", [task_id])
    write_csv(tmp_path / "run" / task_id / "prediction.csv", [["value"], ["a"], ["b"]])
    write_csv(tmp_path / "gold" / task_id / "gold.csv", [["value"], ["a"]])

    # When
    result = evaluate_single(tmp_path, task_id)

    # Then
    assert result.rows_pred == 2
    assert result.rows_gold == 1
    assert result.shape_match is False
    assert result.diagnosis == "zero_recall"


def test_should_report_missing_prediction_without_failing_evaluation(tmp_path: Path) -> None:
    # Given
    task_id = "task_7"
    write_summary(tmp_path / "run", [task_id])
    write_csv(tmp_path / "gold" / task_id / "gold.csv", [["value"], ["a"]])

    # When
    result = evaluate_single(tmp_path, task_id)

    # Then
    assert result.missing_prediction is True
    assert result.missing_gold is False
    assert result.diagnosis == "missing_prediction"


def test_should_report_missing_gold_without_failing_evaluation(tmp_path: Path) -> None:
    # Given
    task_id = "task_8"
    write_summary(tmp_path / "run", [task_id])
    write_csv(tmp_path / "run" / task_id / "prediction.csv", [["value"], ["a"]])

    # When
    result = evaluate_single(tmp_path, task_id)

    # Then
    assert result.missing_gold is True
    assert result.missing_prediction is False
    assert result.diagnosis == "missing_gold"


def test_should_preserve_duplicate_rows_when_comparing_unordered_values(
    tmp_path: Path,
) -> None:
    # Given
    task_id = "task_9"
    write_summary(tmp_path / "run", [task_id])
    write_csv(tmp_path / "run" / task_id / "prediction.csv", [["value"], ["a"], ["a"]])
    write_csv(tmp_path / "gold" / task_id / "gold.csv", [["value"], ["a"], ["b"]])

    # When
    result = evaluate_single(tmp_path, task_id)

    # Then
    assert result.shape_match is True
    assert result.value_match_unordered is False
    assert result.numeric_tolerant_unordered_match is False
    assert result.diagnosis == "zero_recall"


def test_should_normalize_blank_nulls_and_line_endings_for_value_match(
    tmp_path: Path,
) -> None:
    # Given
    task_id = "task_10"
    write_summary(tmp_path / "run", [task_id])
    write_csv(
        tmp_path / "run" / task_id / "prediction.csv",
        [["a", "b", "c", "d", "e"], [" NaN ", "x\r\ny", "None", "nat", "<NA>"]],
    )
    write_csv(
        tmp_path / "gold" / task_id / "gold.csv",
        [["v", "w", "x", "y", "z"], ["", "xy", "", "", ""]],
    )

    # When
    result = evaluate_single(tmp_path, task_id)

    # Then
    assert result.exact_match is False
    assert result.value_match is True
    assert result.numeric_tolerant_match is True
    assert result.diagnosis == "perfect_recall_no_extras"


def test_should_normalize_dates_to_iso_strings(tmp_path: Path) -> None:
    # Given
    task_id = "task_10b"
    write_summary(tmp_path / "run", [task_id])
    write_csv(tmp_path / "run" / task_id / "prediction.csv", [["date"], ["2024-3-1"]])
    write_csv(tmp_path / "gold" / task_id / "gold.csv", [["date"], ["2024-03-01"]])

    # When
    result = evaluate_single(tmp_path, task_id)

    # Then
    assert result.value_match is True
    assert result.numeric_tolerant_unordered_match is True


def test_should_normalize_timezone_datetimes_to_utc_z(tmp_path: Path) -> None:
    # Given
    task_id = "task_10bb"
    write_summary(tmp_path / "run", [task_id])
    write_csv(
        tmp_path / "run" / task_id / "prediction.csv",
        [["datetime"], ["2024-03-01T09:30:00+09:00"]],
    )
    write_csv(
        tmp_path / "gold" / task_id / "gold.csv",
        [["datetime"], ["2024-03-01T00:30:00Z"]],
    )

    # When
    result = evaluate_single(tmp_path, task_id)

    # Then
    assert result.value_match is True
    assert result.numeric_tolerant_unordered_match is True


def test_should_keep_string_matching_case_sensitive(tmp_path: Path) -> None:
    # Given
    task_id = "task_10c"
    write_summary(tmp_path / "run", [task_id])
    write_csv(tmp_path / "run" / task_id / "prediction.csv", [["name"], ["Alice"]])
    write_csv(tmp_path / "gold" / task_id / "gold.csv", [["name"], ["alice"]])

    # When
    result = evaluate_single(tmp_path, task_id)

    # Then
    assert result.value_match is False
    assert result.numeric_tolerant_unordered_match is False
    assert result.diagnosis == "zero_recall"


def test_should_score_column_signature_multiset_instead_of_row_multiset(
    tmp_path: Path,
) -> None:
    # Given
    task_id = "task_10d"
    write_summary(tmp_path / "run", [task_id])
    write_csv(
        tmp_path / "run" / task_id / "prediction.csv",
        [["a", "b"], ["1", "x"], ["2", "y"]],
    )
    write_csv(
        tmp_path / "gold" / task_id / "gold.csv",
        [["c", "d"], ["2", "y"], ["1", "x"]],
    )

    # When
    result = evaluate_single(tmp_path, task_id)

    # Then
    assert result.value_match is False
    assert result.value_match_unordered is True
    assert result.matched_cols == 2
    assert result.diagnosis == "perfect_recall_no_extras"


def test_should_report_recall_extra_columns_and_lambda_scores(tmp_path: Path) -> None:
    # Given
    task_id = "task_10e"
    write_summary(tmp_path / "run", [task_id])
    write_csv(
        tmp_path / "run" / task_id / "prediction.csv",
        [["a", "extra"], ["1", "x"]],
    )
    write_csv(tmp_path / "gold" / task_id / "gold.csv", [["a"], ["1"]])

    # When
    result = evaluate_single(tmp_path, task_id)

    # Then
    assert result.matched_cols == 1
    assert result.gold_cols == 1
    assert result.pred_cols == 2
    assert result.extra_cols == 1
    assert result.recall == 1.0
    assert result.extras_ratio == 0.5
    assert result.official_score_lambda_0_5 == 0.75
    assert result.diagnosis == "perfect_recall_with_extras"


def test_should_allow_split_and_joined_name_columns(tmp_path: Path) -> None:
    # Given
    task_id = "task_10f"
    write_summary(tmp_path / "run", [task_id])
    write_csv(
        tmp_path / "run" / task_id / "prediction.csv",
        [["first", "last"], ["Ada", "Lovelace"], ["Grace", "Hopper"]],
    )
    write_csv(
        tmp_path / "gold" / task_id / "gold.csv",
        [["name"], ["Grace Hopper"], ["Ada Lovelace"]],
    )

    # When
    result = evaluate_single(tmp_path, task_id)

    # Then
    assert result.matched_cols == 1
    assert result.extra_cols == 0
    assert result.diagnosis == "perfect_recall_no_extras"


def test_should_write_evaluation_json_and_flat_csv_outputs(tmp_path: Path) -> None:
    # Given
    task_id = "task_11"
    run_dir = tmp_path / "run"
    write_summary(run_dir, [task_id])
    write_csv(run_dir / task_id / "prediction.csv", [["answer"], ["1"]])
    write_csv(tmp_path / "gold" / task_id / "gold.csv", [["expected"], ["1"]])
    evaluations = evaluate_run(
        run_dir,
        tmp_path / "gold",
        EvaluationOptions(rtol=1e-6, atol=1e-6, preview_rows=5),
    )

    # When
    outputs = write_evaluation_outputs(run_dir, evaluations)

    # Then
    assert outputs.json_path == run_dir / "evaluation.json"
    assert outputs.csv_path == run_dir / "evaluation.csv"
    json_payload = json.loads(outputs.json_path.read_text(encoding="utf-8"))
    assert json_payload["task_count"] == 1
    assert json_payload["tasks"][0]["task_id"] == task_id

    with outputs.csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0].keys()) == [
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
    assert rows[0]["task_id"] == task_id
    assert rows[0]["diagnosis"] == "perfect_recall_no_extras"


def test_should_keep_internal_result_containers_out_of_public_eval_api() -> None:
    # Given / When
    import kobushi_core.eval as eval_api

    # Then
    assert eval_api.__all__ == [
        "EvaluationOptions",
        "evaluate_run",
        "write_evaluation_outputs",
    ]
    assert not hasattr(eval_api, "EvaluationOutputs")
    assert not hasattr(eval_api, "TaskEvaluation")


def test_should_treat_lowercase_null_string_as_empty(tmp_path: Path) -> None:
    # Given
    task_id = "task_null"
    write_summary(tmp_path / "run", [task_id])
    write_csv(tmp_path / "run" / task_id / "prediction.csv", [["v"], ["null"]])
    write_csv(tmp_path / "gold" / task_id / "gold.csv", [["v"], [""]])

    # When
    result = evaluate_single(tmp_path, task_id)

    # Then
    assert result.value_match is True
    assert result.diagnosis == "perfect_recall_no_extras"


def test_should_clamp_official_score_at_zero_when_penalty_exceeds_recall(
    tmp_path: Path,
) -> None:
    # Given: recall=0, three extra columns -> raw score would be -1.0 at lambda=1.0
    task_id = "task_floor"
    write_summary(tmp_path / "run", [task_id])
    write_csv(
        tmp_path / "run" / task_id / "prediction.csv",
        [["a", "b", "c"], ["1", "2", "3"]],
    )
    write_csv(tmp_path / "gold" / task_id / "gold.csv", [["x"], ["9"]])

    # When
    result = evaluate_single(tmp_path, task_id)

    # Then
    assert result.recall == 0.0
    assert result.official_score_lambda_0_5 >= 0.0
    assert result.official_score_lambda_1_0 == 0.0
