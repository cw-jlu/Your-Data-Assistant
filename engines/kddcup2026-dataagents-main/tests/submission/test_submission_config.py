"""Behavior of the container path's forced config overrides (§3.4 / §3.5)."""

from __future__ import annotations

from pathlib import Path

import pytest

from agents.submission import _build_config

CONFIG_WITH_BLOCKLIST = """\
run:
  task_timeout_seconds: 777
  blocklist:
    - task_1
    - task_2
"""


def test_build_config_clears_blocklist(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(CONFIG_WITH_BLOCKLIST)
    monkeypatch.setenv("CONFIG_PATH", str(config_path))
    monkeypatch.setenv("LOGS_ROOT", str(tmp_path / "logs"))
    monkeypatch.setenv("MODEL_API_URL", "http://model.internal/v1")
    monkeypatch.setenv("MODEL_API_KEY", "container-key")
    monkeypatch.setenv("MODEL_NAME", "qwen3.5-35b-a3b")
    monkeypatch.delenv("MAX_STEPS", raising=False)

    config = _build_config(tmp_path / "input")

    # A baked-in local debug skip-list must never reach the judged run:
    # a skipped task is a guaranteed zero.
    assert config.run.blocklist == ()
    # Only blocklist is cleared; sibling run-section keys survive.
    assert config.run.task_timeout_seconds == 777
