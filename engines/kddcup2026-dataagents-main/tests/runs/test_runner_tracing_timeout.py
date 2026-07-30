"""Parent-process tracing cleanup for subprocess timeout paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from queue import Empty
from types import SimpleNamespace
from typing import Any

from agents.benchmark.schema import PublicTask, TaskAssets, TaskRecord
from agents.config import AgentConfig, AppConfig, RunConfig, TracingConfig
from agents.runs import runner as runs_runner, subprocess as runs_subprocess
from agents.tracing.store import SQLiteTraceStore


class _EmptyQueue:
    def get(self, timeout: float) -> dict[str, Any]:
        raise Empty

    def get_nowait(self) -> dict[str, Any]:
        raise Empty


class _TimeoutProcess:
    exitcode: int | None = None

    def __init__(self, target: Any, args: tuple[Any, ...]) -> None:
        self._alive = False

    def start(self) -> None:
        self._alive = True

    def is_alive(self) -> bool:
        return self._alive

    def terminate(self) -> None:
        self._alive = False
        self.exitcode = -15

    def kill(self) -> None:
        self._alive = False
        self.exitcode = -9

    def join(self, timeout: float | None = None) -> None:
        pass


def _make_running_trace(db_path: Path) -> None:
    store = SQLiteTraceStore(db_path)
    store.upsert_trace(
        {
            "object": "trace",
            "id": "trace-timeout",
            "run_id": "run-1",
            "task_id": "task_1",
            "name": "ReActAgent",
            "started_at": "2026-01-01T00:00:00+00:00",
            "ended_at": None,
            "metadata": {},
        }
    )
    store.close()


def test_timeout_marks_running_sqlite_trace_timeout(tmp_path: Path, monkeypatch: Any) -> None:
    db_path = tmp_path / "traces.db"
    _make_running_trace(db_path)

    monkeypatch.setattr(runs_subprocess._MP_SPAWN, "Queue", lambda: _EmptyQueue())
    monkeypatch.setattr(runs_subprocess._MP_SPAWN, "Process", _TimeoutProcess)
    ticks = iter([0.0, 2.0])
    monkeypatch.setattr(runs_subprocess, "perf_counter", lambda: next(ticks))

    config = AppConfig(
        agent=AgentConfig(model="m", api_base="http://x", api_key="real-key"),
        run=RunConfig(run_id="run-1", task_timeout_seconds=1),
        tracing=TracingConfig(enabled=True, db_path=db_path),
    )

    payload = runs_subprocess.run_single_task_with_timeout(task_id="task_1", config=config)

    assert payload["failure_reason"] == "Task timed out after 1 seconds."
    store = SQLiteTraceStore(db_path)
    result = store.get_trace_with_spans("trace-timeout")
    store.close()
    assert result is not None
    trace = result["trace"]
    assert trace["status"] == "timeout"
    assert trace["ended_at"] is not None
    assert trace["metadata"]["status"] == "timeout"
    assert trace["metadata"]["failure_reason"] == "Task timed out after 1 seconds."


def test_unsuccessful_agent_result_marks_sqlite_trace_error(
    tmp_path: Path, monkeypatch: Any
) -> None:
    db_path = tmp_path / "traces.db"
    context_dir = tmp_path / "task_1" / "context"
    context_dir.mkdir(parents=True)
    task = PublicTask(
        record=TaskRecord(task_id="task_1", difficulty="easy", question="q"),
        assets=TaskAssets(task_dir=tmp_path / "task_1", context_dir=context_dir),
    )

    class _Dataset:
        def get_task(self, task_id: str) -> PublicTask:
            assert task_id == "task_1"
            return task

    class _Agent:
        def run(self, task: PublicTask) -> Any:
            assert task.task_id == "task_1"
            return SimpleNamespace(
                to_dict=lambda: {
                    "task_id": "task_1",
                    "answer": None,
                    "steps": [],
                    "failure_reason": "Agent did not submit an answer within max_steps.",
                    "succeeded": False,
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

    monkeypatch.setattr(extractor, "run_etl_for_task", lambda *_, **__: None)
    monkeypatch.setattr(context_mod, "build_virtual_context", lambda task: task.context_dir)

    config = AppConfig(
        agent=AgentConfig(model="m", api_base="http://x", api_key="real-key"),
        run=RunConfig(run_id="run-1", task_timeout_seconds=0),
        tracing=TracingConfig(enabled=True, db_path=db_path),
    )

    payload = runs_subprocess.run_single_task_core(task_id="task_1", config=config)

    assert payload["succeeded"] is False
    store = SQLiteTraceStore(db_path)
    traces = store.list_traces(run_id="run-1")
    store.close()
    assert traces["total"] == 1
    trace = traces["items"][0]
    assert trace["status"] == "error"
    assert trace["metadata"]["failure_reason"] == (
        "Agent did not submit an answer within max_steps."
    )


def test_keyboard_interrupt_marks_only_running_traces_error(tmp_path: Path) -> None:
    db_path = tmp_path / "traces.db"
    store = SQLiteTraceStore(db_path)
    store.upsert_trace(
        {
            "object": "trace",
            "id": "trace-running",
            "run_id": "run-1",
            "task_id": "task_1",
            "name": "ReActAgent",
            "started_at": "2026-01-01T00:00:00+00:00",
            "ended_at": None,
            "metadata": {},
        }
    )
    store.upsert_trace(
        {
            "object": "trace",
            "id": "trace-completed",
            "run_id": "run-1",
            "task_id": "task_2",
            "name": "ReActAgent",
            "started_at": "2026-01-01T00:00:00+00:00",
            "ended_at": "2026-01-01T00:01:00+00:00",
            "metadata": {},
        }
    )
    store.close()

    config = AppConfig(
        agent=AgentConfig(model="m", api_base="http://x", api_key="real-key"),
        run=RunConfig(run_id="run-1", task_timeout_seconds=1),
        tracing=TracingConfig(enabled=True, db_path=db_path),
    )

    runs_runner._mark_interrupted_traces(config, run_id="run-1", task_ids=["task_1", "task_2"])

    store = SQLiteTraceStore(db_path)
    running = store.get_trace_with_spans("trace-running")
    completed = store.get_trace_with_spans("trace-completed")
    store.close()
    assert running is not None
    assert completed is not None
    assert running["trace"]["status"] == "error"
    assert running["trace"]["metadata"]["failure_reason"] == "Run interrupted by KeyboardInterrupt."
    assert completed["trace"]["status"] == "completed"
    assert completed["trace"]["metadata"] == {}
