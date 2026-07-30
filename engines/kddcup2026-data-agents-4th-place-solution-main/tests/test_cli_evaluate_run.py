from __future__ import annotations

import csv
import json
from pathlib import Path

from typer.testing import CliRunner

from kobushi_core.cli import app

import experiments.exp_001_react_baseline.run as exp_run_module
from experiments.exp_001_react_baseline.run import app as exp_app
from experiments.exp_001_react_baseline.runner import TaskRunArtifacts


def write_csv(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


def write_summary(run_dir: Path, task_ids: list[str]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": "cli-test-run",
        "task_count": len(task_ids),
        "succeeded_task_count": len(task_ids),
        "max_workers": 1,
        "tasks": [{"task_id": task_id} for task_id in task_ids],
    }
    (run_dir / "summary.json").write_text(json.dumps(payload), encoding="utf-8")


def write_task(dataset_root: Path, task_id: str) -> None:
    task_dir = dataset_root / task_id
    (task_dir / "context").mkdir(parents=True, exist_ok=True)
    payload = {"task_id": task_id, "difficulty": "easy", "question": "Return the value."}
    (task_dir / "task.json").write_text(json.dumps(payload), encoding="utf-8")


def test_should_write_evaluation_outputs_from_evaluate_run_command(tmp_path: Path) -> None:
    # Given
    task_id = "task_1"
    run_dir = tmp_path / "run"
    gold_root = tmp_path / "gold"
    write_summary(run_dir, [task_id])
    write_csv(run_dir / task_id / "prediction.csv", [["value"], ["1.004"]])
    write_csv(gold_root / task_id / "gold.csv", [["value"], ["1.00"]])

    # When
    result = CliRunner().invoke(
        app,
        [
            "evaluate-run",
            str(run_dir),
            "--gold-root",
            str(gold_root),
            "--rtol",
            "0.0",
            "--atol",
            "0.2",
            "--preview-rows",
            "2",
        ],
    )

    # Then
    assert result.exit_code == 0, result.output
    evaluation_json_path = run_dir / "evaluation.json"
    evaluation_csv_path = run_dir / "evaluation.csv"
    assert evaluation_json_path.exists()
    assert evaluation_csv_path.exists()

    payload = json.loads(evaluation_json_path.read_text(encoding="utf-8"))
    assert payload["task_count"] == 1
    assert payload["tasks"][0]["task_id"] == task_id
    assert payload["tasks"][0]["numeric_tolerant_match"] is True
    assert payload["tasks"][0]["matched_cols"] == 1
    assert payload["tasks"][0]["official_score_lambda_0_5"] == 1.0
    assert payload["tasks"][0]["diagnosis"] == "perfect_recall_no_extras"

    with evaluation_csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["task_id"] == task_id
    assert rows[0]["numeric_tolerant_match"] == "True"
    assert rows[0]["diagnosis"] == "perfect_recall_no_extras"


def test_should_report_ordered_metric_when_unordered_option_is_false(tmp_path: Path) -> None:
    # Given
    task_id = "task_2"
    run_dir = tmp_path / "run"
    gold_root = tmp_path / "gold"
    write_summary(run_dir, [task_id])
    write_csv(run_dir / task_id / "prediction.csv", [["value"], ["b"], ["a"]])
    write_csv(gold_root / task_id / "gold.csv", [["value"], ["a"], ["b"]])

    # When
    result = CliRunner().invoke(
        app,
        [
            "evaluate-run",
            str(run_dir),
            "--gold-root",
            str(gold_root),
            "--no-unordered",
        ],
    )

    # Then
    assert result.exit_code == 0, result.output
    assert "numeric_tolerant_match: 0" in result.output
    assert "official_score_lambda_0_5_mean: 1.000000" in result.output
    assert "numeric_tolerant_unordered_match" not in result.output

    payload = json.loads((run_dir / "evaluation.json").read_text(encoding="utf-8"))
    assert payload["tasks"][0]["numeric_tolerant_unordered_match"] is True
    assert payload["tasks"][0]["diagnosis"] == "perfect_recall_no_extras"


def test_should_evaluate_outputs_after_run_benchmark_when_requested(
    tmp_path: Path,
    monkeypatch,
) -> None:
    # Given
    task_id = "task_1"
    dataset_root = tmp_path / "input"
    output_root = tmp_path / "runs"
    gold_root = tmp_path / "gold"
    write_task(dataset_root, task_id)
    write_csv(gold_root / task_id / "gold.csv", [["value"], ["1"]])
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "dataset:",
                f"  root_path: {dataset_root}",
                "run:",
                f"  output_dir: {output_root}",
                "  run_id: cli-e2e",
                "  max_workers: 1",
            ]
        ),
        encoding="utf-8",
    )

    def fake_run_benchmark(*, config, limit, progress_callback, model=None, tools=None):
        run_dir = config.run.output_dir / "cli-e2e"
        task_dir = run_dir / task_id
        write_csv(task_dir / "prediction.csv", [["value"], ["1.0"]])
        write_summary(run_dir, [task_id])
        artifact = TaskRunArtifacts(
            task_id=task_id,
            task_output_dir=task_dir,
            prediction_csv_path=task_dir / "prediction.csv",
            trace_path=task_dir / "trace.json",
            succeeded=True,
            failure_reason=None,
        )
        if progress_callback is not None:
            progress_callback(artifact)
        return run_dir, [artifact]

    monkeypatch.setattr(exp_run_module, "run_benchmark", fake_run_benchmark)

    # When
    result = CliRunner().invoke(
        exp_app,
        [
            "run-benchmark",
            "--config",
            str(config_path),
            "--limit",
            "1",
            "--evaluate",
            "--gold-root",
            str(gold_root),
        ],
    )

    # Then
    assert result.exit_code == 0, result.output
    assert "Run output:" in result.output
    assert "Run evaluation:" in result.output
    assert "official_score_lambda_0_5_mean: 1.000000" in result.output
    evaluation_json_path = output_root / "cli-e2e" / "evaluation.json"
    evaluation_csv_path = output_root / "cli-e2e" / "evaluation.csv"
    assert evaluation_json_path.exists()
    assert evaluation_csv_path.exists()
