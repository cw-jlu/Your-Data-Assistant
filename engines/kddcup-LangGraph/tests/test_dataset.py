from __future__ import annotations

import json
from pathlib import Path

import pytest

from data_agent_baseline.benchmark.dataset import DABenchPublicDataset


def _write_task(
    root: Path,
    task_id: str,
    payload: dict[str, object],
) -> None:
    task_dir = root / task_id
    (task_dir / "context").mkdir(parents=True, exist_ok=True)
    (task_dir / "task.json").write_text(json.dumps(payload), encoding="utf-8")


def test_dataset_accepts_phase2_task_json_without_difficulty(tmp_path: Path) -> None:
    root = tmp_path / "input"
    _write_task(
        root,
        "task_1",
        {
            "task_id": "task_1",
            "question": "What happens in the attached video?",
            "extra_phase2_field": {"source": "hidden"},
        },
    )

    task = DABenchPublicDataset(root).get_task("task_1")

    assert task.task_id == "task_1"
    assert task.question == "What happens in the attached video?"
    assert task.difficulty == "unknown"


def test_dataset_rejects_task_json_missing_required_phase2_keys(tmp_path: Path) -> None:
    root = tmp_path / "input"
    _write_task(root, "task_1", {"task_id": "task_1", "difficulty": "easy"})

    with pytest.raises(ValueError, match="missing required keys \\['question'\\]"):
        DABenchPublicDataset(root).get_task("task_1")
