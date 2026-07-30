"""ETL must be on-demand rather than an eager task-start hook."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from agents.benchmark.schema import PublicTask, TaskAssets, TaskRecord
from agents.config import AgentConfig, AppConfig, RunConfig
from agents.runs import subprocess as runs_subprocess


def test_task_startup_does_not_run_eager_etl(tmp_path: Path, monkeypatch: Any) -> None:
    context_dir = tmp_path / "task_1" / "context"
    context_dir.mkdir(parents=True)
    task = PublicTask(
        record=TaskRecord(task_id="task_1", difficulty="easy", question="q"),
        assets=TaskAssets(task_dir=tmp_path / "task_1", context_dir=context_dir),
    )

    class _Dataset:
        def get_task(self, task_id: str) -> PublicTask:
            return task

    class _Agent:
        def run(self, task: PublicTask) -> Any:
            return SimpleNamespace(
                to_dict=lambda: {
                    "task_id": "task_1",
                    "answer": "42",
                    "steps": [],
                    "failure_reason": None,
                    "succeeded": True,
                }
            )

    @dataclass(frozen=True)
    class _App:
        dataset: Any
        registry: Any
        model_adapter: Any
        answer_verifier: Any
        agent_config: Any

        def new_agent(self) -> _Agent:
            return _Agent()

    fake_app = _App(
        dataset=_Dataset(),
        registry=SimpleNamespace(definitions={}),
        model_adapter=object(),
        answer_verifier=None,
        agent_config=SimpleNamespace(max_steps=1),
    )
    monkeypatch.setattr(runs_subprocess, "build_application", lambda *_, **__: fake_app)

    import agents.etl.extractor as extractor
    import agents.tools.context as context_mod

    def _boom(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("run_etl_for_task must only run through the run_etl tool")

    monkeypatch.setattr(extractor, "run_etl_for_task", _boom)
    monkeypatch.setattr(context_mod, "build_virtual_context", lambda task: task.context_dir)

    config = AppConfig(
        agent=AgentConfig(model="m", api_base="http://x", api_key="real-key"),
        run=RunConfig(run_id="run-1", task_timeout_seconds=0),
    )

    payload = runs_subprocess.run_single_task_core(task_id="task_1", config=config)

    assert payload["succeeded"] is True
    assert payload["answer"] == "42"
