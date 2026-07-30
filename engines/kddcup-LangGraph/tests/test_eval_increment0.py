from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from data_agent_baseline import cli as cli_module
from data_agent_baseline.config import PROJECT_ROOT, load_app_config
from data_agent_baseline.scoring import compare_run_scores

runner = CliRunner()


def _write_score(
    score_path: Path,
    *,
    run_id: str,
    primary_proxy_score: float,
    failure_breakdown: dict[str, int] | None = None,
) -> None:
    score_path.parent.mkdir(parents=True, exist_ok=True)
    score_path.write_text(
        json.dumps(
            {
                "metadata": {"run_id": run_id},
                "overview": {
                    "task_count": 3,
                    "prediction_task_count": 2,
                    "primary_proxy_score": primary_proxy_score,
                    "mean_recall": 0.5,
                    "mean_redundancy_rate": 0.25,
                },
                "runtime_summary": {
                    "mean_model_step_count": 12.5,
                    "max_model_step_count": 32,
                    "p95_e2e_elapsed_seconds": 123.456,
                },
                "failure_breakdown": failure_breakdown or {},
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def test_docker_config_reads_platform_model_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_API_URL", "http://model-service/v1")
    monkeypatch.setenv("MODEL_API_KEY", "secret")
    monkeypatch.setenv("MODEL_NAME", "qwen3.5-35b-a3b")

    config = load_app_config(PROJECT_ROOT / "configs" / "docker.yaml")

    assert str(config.dataset.root_path).replace("\\", "/") == "/input"
    assert config.agent.model == "qwen3.5-35b-a3b"
    assert config.agent.model_env == "MODEL_NAME"
    assert config.agent.api_base == "http://model-service/v1"
    assert config.agent.api_base_env == "MODEL_API_URL"
    assert config.agent.api_key == "secret"
    assert config.agent.api_key_env == "MODEL_API_KEY"
    assert str(config.run.output_dir).replace("\\", "/") == "/output"
    assert str(config.run.log_dir).replace("\\", "/") == "/logs"
    assert config.run.output_layout == "flat"


def test_compare_run_scores_accepts_run_ids_and_relative_paths(tmp_path: Path) -> None:
    runs_root = tmp_path / "artifacts" / "runs"
    _write_score(
        runs_root / "run-a" / "score.json",
        run_id="run-a",
        primary_proxy_score=0.7,
        failure_breakdown={"Agent did not submit an answer within max_steps.": 2},
    )
    _write_score(
        tmp_path / "artifacts" / "standard" / "baseline" / "score.json",
        run_id="baseline",
        primary_proxy_score=0.5,
    )

    rows = compare_run_scores(
        runs_root=runs_root,
        run_refs=["run-a", "artifacts/standard/baseline"],
        base_dir=tmp_path,
    )

    assert [row.run_id for row in rows] == ["run-a", "baseline"]
    assert rows[0].primary_proxy_score == pytest.approx(0.7)
    assert rows[0].top_failure == "Agent did not submit an answer within max_steps. (2)"
    assert rows[1].score_json_path == tmp_path / "artifacts" / "standard" / "baseline" / "score.json"


def test_compare_run_scores_reports_missing_score_json(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="score.json"):
        compare_run_scores(
            runs_root=tmp_path / "artifacts" / "runs",
            run_refs=["missing-run"],
            base_dir=tmp_path,
        )


def test_cli_compare_runs_prints_score_table(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runs_root = tmp_path / "artifacts" / "runs"
    _write_score(runs_root / "run-a" / "score.json", run_id="run-a", primary_proxy_score=0.7)
    _write_score(
        tmp_path / "artifacts" / "standard" / "baseline" / "score.json",
        run_id="baseline",
        primary_proxy_score=0.5,
    )
    monkeypatch.setattr(cli_module, "ARTIFACT_RUNS_DIR", runs_root)
    monkeypatch.setattr(cli_module, "PROJECT_ROOT", tmp_path)

    result = runner.invoke(cli_module.app, ["compare-runs", "run-a", "artifacts/standard/baseline"])

    assert result.exit_code == 0, result.output
    assert "Run Comparison" in result.output
    assert "run-a" in result.output
    assert "baseline" in result.output
    assert "0.7000" in result.output
    assert "0.5000" in result.output
