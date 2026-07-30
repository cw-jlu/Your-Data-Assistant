from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from data_agent_baseline import cli as cli_module
from data_agent_baseline.scoring import (
    DEFAULT_PRIMARY_LAMBDA,
    SUMMARY_TASK_SOURCE,
    normalize_cell,
    score_run_outputs,
)

runner = CliRunner()


def _write_csv(path: Path, columns: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_summary(run_output_dir: Path, task_ids: list[str]) -> None:
    _write_json(
        run_output_dir / "summary.json",
        {
            "tasks": [
                {"task_id": task_id, "succeeded": True, "failure_reason": None}
                for task_id in task_ids
            ]
        },
    )


def _create_task(input_root: Path, task_id: str, difficulty: str) -> None:
    task_dir = input_root / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        task_dir / "task.json",
        {
            "task_id": task_id,
            "difficulty": difficulty,
            "question": f"Question for {task_id}",
        },
    )


def _write_prediction(
    run_output_dir: Path, task_id: str, columns: list[str], rows: list[list[object]]
) -> None:
    _write_csv(run_output_dir / task_id / "prediction.csv", columns, rows)


def _write_trace(
    run_output_dir: Path,
    task_id: str,
    *,
    succeeded: bool,
    failure_reason: str | None,
    elapsed_seconds: float,
    step_count: int,
    model_step_count: int | None = None,
    tool_calls_per_step: list[list[dict[str, object]]] | None = None,
) -> None:
    if model_step_count is None:
        steps: list[dict[str, object]] = [{"step_index": index + 1} for index in range(step_count)]
    else:
        if model_step_count > step_count:
            raise ValueError("model_step_count must not exceed step_count")
        steps = [{"step_index": index + 1, "node": "model"} for index in range(model_step_count)]
        steps.extend(
            {"step_index": model_step_count + index + 1, "node": "tool"}
            for index in range(step_count - model_step_count)
        )
    if tool_calls_per_step is not None:
        for step, calls in zip(steps, tool_calls_per_step):
            step["tool_calls"] = calls
    _write_json(
        run_output_dir / task_id / "trace.json",
        {
            "task_id": task_id,
            "answer": None,
            "steps": steps,
            "failure_reason": failure_reason,
            "succeeded": succeeded,
            "e2e_elapsed_seconds": elapsed_seconds,
        },
    )


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("NULL", ""),
        ("  nan  ", ""),
        ("4200000", "4200000.00"),
        ("0.005", "0.01"),
        ("2024-3-1", "2024-03-01"),
        ("2024-03-01T10:30:00+08:00", "2024-03-01T02:30:00Z"),
        ("2024-03-01 10:30:00", "2024-03-01T10:30:00"),
        ("  East Asia\r\n", "East Asia"),
    ],
)
def test_normalize_cell_rules(raw_value: str, expected: str) -> None:
    assert normalize_cell(raw_value) == expected


def test_score_run_outputs_supports_name_field_equivalence_and_prefers_less_redundancy(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "data" / "public" / "input"
    gold_root = tmp_path / "data" / "public" / "output"
    run_output_dir = tmp_path / "artifacts" / "runs" / "sample-run"

    _create_task(input_root, "task_1", "easy")
    _write_csv(gold_root / "task_1" / "gold.csv", ["full_name"], [["John Smith"], ["Jane Doe"]])
    _write_prediction(
        run_output_dir,
        "task_1",
        ["first_name", "last_name", "full_name"],
        [["John", "Smith", "John Smith"], ["Jane", "Doe", "Jane Doe"]],
    )

    _create_task(input_root, "task_2", "medium")
    _write_csv(
        gold_root / "task_2" / "gold.csv",
        ["first_name", "last_name"],
        [["Jane", "Doe"], ["John", "Smith"]],
    )
    _write_prediction(run_output_dir, "task_2", ["full_name"], [["John Smith"], ["Jane Doe"]])

    _create_task(input_root, "task_3", "hard")
    _write_csv(gold_root / "task_3" / "gold.csv", ["value"], [[1]])
    _write_summary(run_output_dir, ["task_1", "task_2"])

    summary = score_run_outputs(run_output_dir=run_output_dir, gold_root=gold_root)
    assert [task.task_id for task in summary.tasks] == ["task_1", "task_2"]

    task_1 = next(task for task in summary.tasks if task.task_id == "task_1")
    assert task_1.full_cover is True
    assert task_1.covered_gold_columns == 1
    assert task_1.matched_prediction_columns == 2
    assert task_1.extra_prediction_columns == 1
    assert task_1.redundancy_rate == pytest.approx(1 / 3)
    assert task_1.primary_lambda == DEFAULT_PRIMARY_LAMBDA
    assert task_1.primary_proxy_score == pytest.approx(task_1.proxy_scores["0.1"])

    task_2 = next(task for task in summary.tasks if task.task_id == "task_2")
    assert task_2.full_cover is True
    assert task_2.covered_gold_columns == 2
    assert task_2.matched_prediction_columns == 1
    assert task_2.extra_prediction_columns == 0
    assert task_2.recall == pytest.approx(1.0)
    assert task_2.primary_lambda == DEFAULT_PRIMARY_LAMBDA
    assert task_2.primary_proxy_score == pytest.approx(task_2.proxy_scores["0.1"])


def test_score_run_outputs_only_combines_name_fields(tmp_path: Path) -> None:
    input_root = tmp_path / "data" / "public" / "input"
    gold_root = tmp_path / "data" / "public" / "output"
    run_output_dir = tmp_path / "artifacts" / "runs" / "sample-run"

    _create_task(input_root, "task_1", "easy")
    _write_csv(
        gold_root / "task_1" / "gold.csv",
        ["city", "state"],
        [["New", "York"], ["Los", "Angeles"]],
    )
    _write_prediction(run_output_dir, "task_1", ["location"], [["New York"], ["Los Angeles"]])
    _write_summary(run_output_dir, ["task_1"])

    summary = score_run_outputs(run_output_dir=run_output_dir, gold_root=gold_root)

    task = summary.tasks[0]
    assert task.full_cover is False
    assert task.covered_gold_columns == 0
    assert task.matched_prediction_columns == 0
    assert task.recall == pytest.approx(0.0)


def test_score_run_outputs_aggregates_metrics_and_generates_report(tmp_path: Path) -> None:
    input_root = tmp_path / "data" / "public" / "input"
    gold_root = tmp_path / "data" / "public" / "output"
    run_output_dir = tmp_path / "artifacts" / "runs" / "sample-run"

    _create_task(input_root, "task_1", "easy")
    _write_csv(gold_root / "task_1" / "gold.csv", ["value"], [[1]])
    _write_prediction(run_output_dir, "task_1", ["value"], [[1.004]])
    _write_trace(
        run_output_dir,
        "task_1",
        succeeded=True,
        failure_reason=None,
        elapsed_seconds=10.0,
        step_count=3,
    )

    _create_task(input_root, "task_2", "hard")
    _write_csv(gold_root / "task_2" / "gold.csv", ["full_name"], [["John Smith"], ["Jane Doe"]])
    _write_prediction(
        run_output_dir,
        "task_2",
        ["first_name", "last_name", "full_name"],
        [["John", "Smith", "John Smith"], ["Jane", "Doe", "Jane Doe"]],
    )
    _write_trace(
        run_output_dir,
        "task_2",
        succeeded=True,
        failure_reason=None,
        elapsed_seconds=20.0,
        step_count=5,
    )

    _create_task(input_root, "task_3", "medium")
    _write_csv(gold_root / "task_3" / "gold.csv", ["answer"], [["42"]])
    _write_trace(
        run_output_dir,
        "task_3",
        succeeded=False,
        failure_reason="Agent did not submit an answer within max_steps.",
        elapsed_seconds=90.0,
        step_count=32,
    )

    _write_json(
        run_output_dir / "summary.json",
        {
            "tasks": [
                {"task_id": "task_1", "succeeded": True, "failure_reason": None},
                {"task_id": "task_2", "succeeded": True, "failure_reason": None},
                {
                    "task_id": "task_3",
                    "succeeded": False,
                    "failure_reason": "Agent did not submit an answer within max_steps.",
                },
            ]
        },
    )

    summary = score_run_outputs(run_output_dir=run_output_dir, gold_root=gold_root)

    assert summary.task_count == 3
    assert summary.prediction_task_count == 2
    assert summary.primary_lambda == DEFAULT_PRIMARY_LAMBDA
    assert summary.primary_proxy_score == pytest.approx((1 + (1 - (0.1 / 3)) + 0) / 3)
    assert summary.mean_recall == pytest.approx(2 / 3)
    assert summary.mean_redundancy_rate == pytest.approx((0 + (1 / 3) + 0) / 3)
    assert summary.proxy_scores["0.1"] == pytest.approx(summary.primary_proxy_score)
    assert summary.proxy_scores["0.5"] == pytest.approx((1 + (1 - (0.5 / 3)) + 0) / 3)
    assert summary.failure_breakdown == {"Agent did not submit an answer within max_steps.": 1}
    assert summary.difficulty_breakdown["easy"]["full_cover_count"] == 1
    assert summary.difficulty_breakdown["easy"]["primary_proxy_score"] == pytest.approx(1.0)
    assert summary.difficulty_breakdown["hard"]["full_cover_count"] == 1
    assert summary.difficulty_breakdown["hard"]["primary_proxy_score"] == pytest.approx(
        1 - (0.1 / 3)
    )
    assert summary.runtime_summary["available_runtime_count"] == 3
    assert summary.runtime_summary["max_model_step_count"] == 32
    assert summary.runtime_summary["max_trace_step_count"] == 32

    task_3 = next(task for task in summary.tasks if task.task_id == "task_3")
    assert task_3.tool_call_counts == {}

    assert summary.task_source == SUMMARY_TASK_SOURCE
    assert summary.score_path.exists()
    assert summary.score_report_path.exists()
    report_text = summary.score_report_path.read_text(encoding="utf-8")
    assert "主分与基础指标" in report_text
    assert "多 λ 代理分数" in report_text
    assert "最大模型轮数" in report_text
    assert "评分范围来源：当前 run 目录下的 `summary.json.tasks`。" in report_text
    assert "`Recall`：单题覆盖率，计算方式为 `覆盖Gold列数 / Gold列总数`。" in report_text
    assert "`Mean Recall`：所有任务 `Recall` 的平均值" in report_text
    assert "`Mean Redundancy` / `Mean Redundancy Rate`：所有任务冗余率的平均值" in report_text
    assert "可用 Trace 节点任务数" not in report_text
    assert "平均 Trace 节点数" not in report_text
    assert "最大 Trace 节点数" not in report_text
    assert "Trace节点数" not in report_text
    assert "Full Cover Rate" not in report_text
    assert (
        "| 难度 | 任务数 | 有预测 | 完全正确题数 | Primary(λ=0.1) | Mean Recall | Mean Redundancy |"
        in report_text
    )
    assert "| easy | 1 | 1 | 1 | 1.0000 | 1.0000 | 0.0000 |" in report_text
    assert "| medium | 1 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 |" in report_text
    difficulty_section = report_text.split("## 按难度拆分表现", 1)[1].split(
        "## 失败原因与耗时分析", 1
    )[0]
    assert difficulty_section.index("| easy |") < difficulty_section.index("| medium |")
    assert difficulty_section.index("| medium |") < difficulty_section.index("| hard |")
    assert "## 最值得复盘的任务（Primary < 0.5，共 1 题）" in report_text
    assert (
        "| 任务 | 难度 | Primary(λ=0.1) | Recall | Redundancy | Full Cover | 失败/备注 | 模型轮数 | 耗时(秒) |"
        in report_text
    )
    review_section = report_text.split("## 最值得复盘的任务", 1)[1].split("## 全量任务附录", 1)[0]
    assert "| task_2 |" not in review_section
    assert (
        "| task_3 | medium | 0.0000 | 0.0000 | 0.0000 | no | Agent did not submit an answer within max_steps. | 32 | 90.000 |"
        in review_section
    )
    assert (
        "| 任务 | 难度 | Gold列数 | 预测列数 | 覆盖Gold列数 | 冗余列数 | Primary(λ=0.1) | Recall | Redundancy | Full Cover | 模型轮数 | 耗时(秒) | 失败/备注 |"
        in report_text
    )
    assert (
        "| task_1 | easy | 1 | 1 | 1 | 0 | 1.0000 | 1.0000 | 0.0000 | yes | 3 | 10.000 | - |"
        in report_text
    )
    assert (
        "| task_2 | hard | 1 | 3 | 1 | 1 | 0.9667 | 1.0000 | 0.3333 | yes | 5 | 20.000 | All gold columns covered, with 1 extra prediction column(s). |"
        in report_text
    )
    assert (
        "| task_3 | medium | 1 | 0 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | no | 32 | 90.000 | Agent did not submit an answer within max_steps. |"
        in report_text
    )

    score_payload = json.loads(summary.score_path.read_text(encoding="utf-8"))
    assert score_payload["metadata"]["run_id"] == "sample-run"
    assert score_payload["metadata"]["task_source"] == SUMMARY_TASK_SOURCE
    assert score_payload["overview"]["primary_lambda"] == DEFAULT_PRIMARY_LAMBDA
    assert score_payload["overview"]["primary_proxy_score"] == pytest.approx(
        summary.primary_proxy_score
    )
    assert score_payload["primary_lambda"] == DEFAULT_PRIMARY_LAMBDA
    assert score_payload["primary_proxy_score"] == pytest.approx(summary.primary_proxy_score)
    assert "total_score" not in score_payload
    assert "accuracy" not in score_payload
    assert "full_cover_rate" not in score_payload["overview"]
    assert "score" not in score_payload["tasks"][0]


def test_score_report_review_section_lists_all_wrong_tasks(tmp_path: Path) -> None:
    input_root = tmp_path / "data" / "public" / "input"
    gold_root = tmp_path / "data" / "public" / "output"
    run_output_dir = tmp_path / "artifacts" / "runs" / "sample-run"

    wrong_task_ids = [
        "task_10",
        "task_2",
        "task_1",
        "task_9",
        "task_3",
        "task_8",
        "task_4",
        "task_7",
        "task_5",
        "task_6",
    ]
    sorted_wrong_task_ids = [f"task_{index}" for index in range(1, 11)]
    for task_id in wrong_task_ids:
        _create_task(input_root, task_id, "medium")
        _write_csv(gold_root / task_id / "gold.csv", ["value"], [["gold"]])
        _write_prediction(run_output_dir, task_id, ["value"], [["wrong"]])

    _create_task(input_root, "task_11", "easy")
    _write_csv(gold_root / "task_11" / "gold.csv", ["value"], [["gold"]])
    _write_prediction(run_output_dir, "task_11", ["value", "extra"], [["gold", "noise"]])

    _create_task(input_root, "task_12", "easy")
    _write_csv(gold_root / "task_12" / "gold.csv", ["value"], [["gold"]])
    _write_prediction(run_output_dir, "task_12", ["value"], [["gold"]])
    _write_summary(run_output_dir, [*wrong_task_ids, "task_11", "task_12"])

    summary = score_run_outputs(run_output_dir=run_output_dir, gold_root=gold_root)

    report_text = summary.score_report_path.read_text(encoding="utf-8")
    assert "## 最值得复盘的任务（Primary < 0.5，共 10 题）" in report_text
    review_section = report_text.split("## 最值得复盘的任务", 1)[1].split("## 全量任务附录", 1)[0]
    review_task_ids = [
        line.split("|")[1].strip()
        for line in review_section.splitlines()
        if line.startswith("| task_")
    ]
    assert review_task_ids == sorted_wrong_task_ids
    assert "| task_11 |" not in review_section
    assert "| task_12 |" not in review_section

    appendix_section = report_text.split("## 全量任务附录", 1)[1]
    appendix_task_ids = [
        line.split("|")[1].strip()
        for line in appendix_section.splitlines()
        if line.startswith("| task_")
    ]
    assert appendix_task_ids == [*sorted_wrong_task_ids, "task_11", "task_12"]


def test_score_run_outputs_reports_invalid_csv_width(tmp_path: Path) -> None:
    input_root = tmp_path / "data" / "public" / "input"
    gold_root = tmp_path / "data" / "public" / "output"
    run_output_dir = tmp_path / "artifacts" / "runs" / "sample-run"

    _create_task(input_root, "task_1", "easy")
    _write_csv(gold_root / "task_1" / "gold.csv", ["a", "b"], [[1, 2]])
    prediction_path = run_output_dir / "task_1" / "prediction.csv"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_path.write_text("a,b\n1\n", encoding="utf-8")
    _write_summary(run_output_dir, ["task_1"])

    summary = score_run_outputs(run_output_dir=run_output_dir, gold_root=gold_root)

    assert summary.tasks[0].full_cover is False
    assert "CSV row width mismatch" in (summary.tasks[0].reason or "")


def test_score_run_outputs_ignores_invalid_trace_encoding(tmp_path: Path) -> None:
    input_root = tmp_path / "data" / "public" / "input"
    gold_root = tmp_path / "data" / "public" / "output"
    run_output_dir = tmp_path / "artifacts" / "runs" / "sample-run"

    _create_task(input_root, "task_1", "easy")
    _write_csv(gold_root / "task_1" / "gold.csv", ["value"], [[1]])
    _write_prediction(run_output_dir, "task_1", ["value"], [[1]])
    _write_summary(run_output_dir, ["task_1"])

    trace_path = run_output_dir / "task_1" / "trace.json"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_bytes(b'{"task_id":"task_1","steps":"\xba"}')

    summary = score_run_outputs(run_output_dir=run_output_dir, gold_root=gold_root)

    task = summary.tasks[0]
    assert task.task_id == "task_1"
    assert task.model_step_count is None
    assert task.trace_step_count is None
    assert task.e2e_elapsed_seconds is None


def test_cli_score_run_supports_custom_lambda_and_writes_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs_root = tmp_path / "artifacts" / "runs"
    run_output_dir = runs_root / "sample-run"
    gold_root = tmp_path / "data" / "public" / "output"
    input_root = tmp_path / "data" / "public" / "input"

    _create_task(input_root, "task_1", "easy")
    _write_csv(gold_root / "task_1" / "gold.csv", ["value"], [[1]])
    _write_prediction(run_output_dir, "task_1", ["value", "extra"], [[1, "noise"]])
    _write_summary(run_output_dir, ["task_1"])

    monkeypatch.setattr(cli_module, "ARTIFACT_RUNS_DIR", runs_root)
    monkeypatch.setattr(cli_module, "PUBLIC_GOLD_DIR", gold_root)

    result = runner.invoke(
        cli_module.app,
        ["score-run", "sample-run", "--lambda", "0.25", "--lambda", "0.5"],
    )

    assert result.exit_code == 0, result.output
    assert "primary_lambda" in result.output
    assert "primary_proxy_score" in result.output
    assert "mean_recall" in result.output
    assert "score_report" in result.output
    assert "0.1" in result.output
    assert "0.25" in result.output
    assert "summary.json.tasks" in result.output
    assert "full_cover_rate" not in result.output
    assert (run_output_dir / "score_report.md").exists()

    score_payload = json.loads((run_output_dir / "score.json").read_text(encoding="utf-8"))
    assert score_payload["metadata"]["lambda_grid"] == [0.1, 0.25, 0.5]
    assert score_payload["primary_lambda"] == DEFAULT_PRIMARY_LAMBDA


def test_score_run_outputs_separates_model_step_count_and_trace_step_count(tmp_path: Path) -> None:
    input_root = tmp_path / "data" / "public" / "input"
    gold_root = tmp_path / "data" / "public" / "output"
    run_output_dir = tmp_path / "artifacts" / "runs" / "sample-run"

    _create_task(input_root, "task_1", "hard")
    _write_csv(gold_root / "task_1" / "gold.csv", ["value"], [[1]])
    _write_prediction(run_output_dir, "task_1", ["value"], [[1]])
    _write_trace(
        run_output_dir,
        "task_1",
        succeeded=False,
        failure_reason="Agent did not submit an answer within max_steps.",
        elapsed_seconds=12.5,
        step_count=64,
        model_step_count=32,
    )

    _write_json(
        run_output_dir / "summary.json",
        {
            "tasks": [
                {
                    "task_id": "task_1",
                    "succeeded": False,
                    "failure_reason": "Agent did not submit an answer within max_steps.",
                }
            ]
        },
    )

    summary = score_run_outputs(run_output_dir=run_output_dir, gold_root=gold_root)

    task = summary.tasks[0]
    assert task.model_step_count == 32
    assert task.trace_step_count == 64
    assert summary.runtime_summary["max_model_step_count"] == 32
    assert summary.runtime_summary["max_trace_step_count"] == 64
    report_text = summary.score_report_path.read_text(encoding="utf-8")
    assert "Trace节点数" not in report_text
    assert (
        "| task_1 | hard | 1 | 1 | 1 | 0 | 1.0000 | 1.0000 | 0.0000 | yes | 32 | 12.500 | Agent did not submit an answer within max_steps. |"
        in report_text
    )


def test_score_run_outputs_requires_summary_json(tmp_path: Path) -> None:
    gold_root = tmp_path / "data" / "public" / "output"
    run_output_dir = tmp_path / "artifacts" / "runs" / "sample-run"
    gold_root.mkdir(parents=True, exist_ok=True)
    run_output_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(FileNotFoundError, match="summary.json"):
        score_run_outputs(run_output_dir=run_output_dir, gold_root=gold_root)


def test_score_run_outputs_rejects_invalid_or_empty_summary_tasks(tmp_path: Path) -> None:
    gold_root = tmp_path / "data" / "public" / "output"
    run_output_dir = tmp_path / "artifacts" / "runs" / "sample-run"
    gold_root.mkdir(parents=True, exist_ok=True)

    _write_json(run_output_dir / "summary.json", {"tasks": []})
    with pytest.raises(ValueError, match="No valid task ids"):
        score_run_outputs(run_output_dir=run_output_dir, gold_root=gold_root)

    _write_json(run_output_dir / "summary.json", {"tasks": {}})
    with pytest.raises(ValueError, match="`tasks` must be a list"):
        score_run_outputs(run_output_dir=run_output_dir, gold_root=gold_root)


def test_score_run_outputs_fails_fast_when_summary_task_has_no_gold_csv(tmp_path: Path) -> None:
    input_root = tmp_path / "data" / "public" / "input"
    gold_root = tmp_path / "data" / "public" / "output"
    run_output_dir = tmp_path / "artifacts" / "runs" / "sample-run"

    _create_task(input_root, "task_1", "easy")
    gold_root.mkdir(parents=True, exist_ok=True)
    _write_summary(run_output_dir, ["task_1"])

    with pytest.raises(FileNotFoundError, match="Gold file not found for task task_1"):
        score_run_outputs(run_output_dir=run_output_dir, gold_root=gold_root)


def test_cli_score_run_reports_missing_summary_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs_root = tmp_path / "artifacts" / "runs"
    run_output_dir = runs_root / "sample-run"
    gold_root = tmp_path / "data" / "public" / "output"
    run_output_dir.mkdir(parents=True, exist_ok=True)
    gold_root.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(cli_module, "ARTIFACT_RUNS_DIR", runs_root)
    monkeypatch.setattr(cli_module, "PUBLIC_GOLD_DIR", gold_root)

    result = runner.invoke(cli_module.app, ["score-run", "sample-run"])

    assert result.exit_code != 0
    assert "summary.json" in result.output


def test_score_run_outputs_counts_tool_calls(tmp_path: Path) -> None:
    input_root = tmp_path / "data" / "public" / "input"
    gold_root = tmp_path / "data" / "public" / "output"
    run_output_dir = tmp_path / "artifacts" / "runs" / "sample-run"

    _create_task(input_root, "task_1", "easy")
    _write_csv(gold_root / "task_1" / "gold.csv", ["value"], [[1]])
    _write_prediction(run_output_dir, "task_1", ["value"], [[1]])
    _write_trace(
        run_output_dir,
        "task_1",
        succeeded=True,
        failure_reason=None,
        elapsed_seconds=10.0,
        step_count=3,
        tool_calls_per_step=[
            [{"id": "c1", "name": "load_csv", "args": {}}],
            [{"id": "c2", "name": "load_csv", "args": {}}],
            [{"id": "c3", "name": "submit_tool_result", "args": {}}],
        ],
    )

    _create_task(input_root, "task_2", "hard")
    _write_csv(gold_root / "task_2" / "gold.csv", ["full_name"], [["A"]])
    _write_prediction(run_output_dir, "task_2", ["full_name"], [["A"]])
    _write_trace(
        run_output_dir,
        "task_2",
        succeeded=True,
        failure_reason=None,
        elapsed_seconds=5.0,
        step_count=1,
        tool_calls_per_step=[
            [{"id": "c4", "name": "load_csv", "args": {}}],
        ],
    )

    _create_task(input_root, "task_3", "medium")
    _write_csv(gold_root / "task_3" / "gold.csv", ["answer"], [["42"]])
    _write_prediction(run_output_dir, "task_3", ["answer"], [["42"]])

    _write_json(
        run_output_dir / "summary.json",
        {
            "tasks": [
                {"task_id": "task_1", "succeeded": True, "failure_reason": None},
                {"task_id": "task_2", "succeeded": True, "failure_reason": None},
                {"task_id": "task_3", "succeeded": True, "failure_reason": None},
            ]
        },
    )

    summary = score_run_outputs(run_output_dir=run_output_dir, gold_root=gold_root)

    task_1 = next(t for t in summary.tasks if t.task_id == "task_1")
    assert task_1.tool_call_counts == {"load_csv": 2, "submit_tool_result": 1}

    task_2 = next(t for t in summary.tasks if t.task_id == "task_2")
    assert task_2.tool_call_counts == {"load_csv": 1}

    task_3 = next(t for t in summary.tasks if t.task_id == "task_3")
    assert task_3.tool_call_counts is None

    rt = summary.runtime_summary
    tool_stats = rt.get("tool_call_stats", {})
    assert tool_stats["available_trace_count"] == 2
    assert tool_stats["tools"]["load_csv"]["total_calls"] == 3
    assert tool_stats["tools"]["load_csv"]["mean_per_task"] == pytest.approx(1.5)
    assert tool_stats["tools"]["load_csv"]["max_calls"] == 2
    assert tool_stats["tools"]["load_csv"]["tasks_used"] == 2
    assert tool_stats["tools"]["submit_tool_result"]["total_calls"] == 1
    assert tool_stats["tools"]["submit_tool_result"]["mean_per_task"] == pytest.approx(1.0)
    assert tool_stats["tools"]["submit_tool_result"]["max_calls"] == 1
    assert tool_stats["tools"]["submit_tool_result"]["tasks_used"] == 1

    score_payload = json.loads(summary.score_path.read_text(encoding="utf-8"))
    task_payloads = {t["task_id"]: t for t in score_payload["tasks"]}
    assert task_payloads["task_1"]["tool_call_counts"] == {
        "load_csv": 2,
        "submit_tool_result": 1,
    }
    assert task_payloads["task_2"]["tool_call_counts"] == {"load_csv": 1}
    assert task_payloads["task_3"]["tool_call_counts"] is None
    assert score_payload["runtime_summary"]["tool_call_stats"] is not None

    report_text = summary.score_report_path.read_text(encoding="utf-8")
    assert "工具调用统计" in report_text
    assert "load_csv" in report_text
    assert "submit_tool_result" in report_text
