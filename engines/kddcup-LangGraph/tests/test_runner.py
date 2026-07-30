from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from data_agent_baseline.benchmark.schema import PublicTask, TaskAssets, TaskRecord
from data_agent_baseline.config import AgentConfig, AppConfig, DatasetConfig, RunConfig
from data_agent_baseline.run import runner as runner_module
from data_agent_baseline.run.runner import TaskRunArtifacts, run_benchmark


def _create_task(input_root: Path, task_id: str, difficulty: str = "easy") -> None:
    task_dir = input_root / task_id
    (task_dir / "context").mkdir(parents=True, exist_ok=True)
    (task_dir / "task.json").write_text(
        json.dumps(
            {
                "task_id": task_id,
                "difficulty": difficulty,
                "question": f"Question for {task_id}",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


class ScriptedToolCallingModel:
    def __init__(self, responses: list[AIMessage]) -> None:
        self._responses = list(responses)

    def bind_tools(self, tools, tool_choice="auto", parallel_tool_calls=False):  # noqa: ANN001
        del tools, tool_choice, parallel_tool_calls
        return self

    def invoke(self, messages):  # noqa: ANN001
        del messages
        if not self._responses:
            raise RuntimeError("No scripted responses remaining.")
        return self._responses.pop(0)


def test_run_benchmark_summary_includes_runtime_and_agent_config(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset_root = tmp_path / "data" / "public" / "input"
    output_root = tmp_path / "artifacts" / "runs"
    _create_task(dataset_root, "task_1")
    _create_task(dataset_root, "task_2", difficulty="hard")

    config = AppConfig(
        dataset=DatasetConfig(root_path=dataset_root),
        agent=AgentConfig(
            max_steps=48,
            temperature=0.3,
            validation_retry_limit=4,
            strip_reasoning_history=True,
            reasoning_history_limit=2,
        ),
        run=RunConfig(
            output_dir=output_root,
            run_id="summary-test-run",
            max_workers=7,
            task_timeout_seconds=321,
            extract_structured_doc_timeout_bonus_seconds=45,
        ),
    )

    def fake_run_single_task(
        *,
        task_id: str,
        config: AppConfig,
        run_output_dir: Path,
        prediction_output_root: Path | None = None,
        model=None,
        tools=None,
    ) -> TaskRunArtifacts:
        del prediction_output_root
        task_output_dir = run_output_dir / task_id
        task_output_dir.mkdir(parents=True, exist_ok=True)
        trace_path = task_output_dir / "trace.json"
        trace_path.write_text("{}", encoding="utf-8")
        return TaskRunArtifacts(
            task_id=task_id,
            task_output_dir=task_output_dir,
            prediction_csv_path=None,
            trace_path=trace_path,
            succeeded=(task_id == "task_1"),
            failure_reason=None if task_id == "task_1" else "failed",
        )

    monkeypatch.setattr(runner_module, "run_single_task", fake_run_single_task)

    run_output_dir, artifacts = run_benchmark(config=config, model=object())

    assert run_output_dir.name == "summary-test-run"
    assert len(artifacts) == 2

    summary_payload = json.loads((run_output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary_payload["max_workers"] == 1
    assert summary_payload["extract_structured_doc_max_workers"] == 2
    assert summary_payload["task_timeout_seconds"] == 321
    assert summary_payload["extract_structured_doc_timeout_bonus_seconds"] == 45
    assert summary_payload["max_steps"] == 48
    assert summary_payload["temperature"] == 0.3
    assert summary_payload["validation_retry_limit"] == 4
    assert summary_payload["strip_reasoning_history"] is True
    assert summary_payload["reasoning_history_limit"] == 2
    assert summary_payload["succeeded_task_count"] == 1
    assert summary_payload["output_layout"] == "run_dir"
    assert (run_output_dir / "task_status.jsonl").exists()


def test_load_app_config_accepts_empty_reasoning_history_limit(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
agent:
  reasoning_history_limit:
""",
        encoding="utf-8",
    )

    from data_agent_baseline.config import load_app_config

    config = load_app_config(config_path)

    assert config.agent.reasoning_history_limit is None


def test_load_app_config_rejects_negative_reasoning_history_limit(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
agent:
  reasoning_history_limit: -1
""",
        encoding="utf-8",
    )

    from data_agent_baseline.config import load_app_config

    with pytest.raises(ValueError, match="agent.reasoning_history_limit"):
        load_app_config(config_path)


def test_load_app_config_supports_answer_validation_retry_limit(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
agent:
  validation_retry_limit: 5
""",
        encoding="utf-8",
    )

    from data_agent_baseline.config import load_app_config

    config = load_app_config(config_path)

    assert config.agent.validation_retry_limit == 5


def test_load_app_config_supports_structured_doc_tool_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
tool:
  max_output_tokens: 1234
  max_list_items: 56
  structured_doc:
    min_chunk_lines: 10
    max_chunk_lines: 20
    max_selected_lines_for_llm_extraction: 300
    default_max_model_calls: 12
    hard_max_model_calls: 15
    inspect_doc_structure_max_model_calls: 2
    llm:
      temperature: 0.0
      top_p: 1.0
      repetition_penalty: 1.0
""",
        encoding="utf-8",
    )

    from data_agent_baseline.config import load_app_config

    config = load_app_config(config_path)

    assert config.tool.max_output_tokens == 1234
    assert config.tool.max_list_items == 56
    assert config.tool.structured_doc.min_chunk_lines == 10
    assert config.tool.structured_doc.max_chunk_lines == 20
    assert config.tool.structured_doc.max_selected_lines_for_llm_extraction == 300
    assert config.tool.structured_doc.default_max_model_calls == 12
    assert config.tool.structured_doc.hard_max_model_calls == 15
    assert config.tool.structured_doc.inspect_doc_structure_max_model_calls == 2
    assert config.tool.structured_doc.llm.temperature == 0.0
    assert config.tool.structured_doc.llm.top_p == 1.0
    assert config.tool.structured_doc.llm.repetition_penalty == 1.0


def test_load_app_config_supports_extract_structured_doc_worker_limit(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
run:
  extract_structured_doc_max_workers: 3
""",
        encoding="utf-8",
    )

    from data_agent_baseline.config import load_app_config

    config = load_app_config(config_path)

    assert config.run.extract_structured_doc_max_workers == 3


def test_load_app_config_supports_extract_structured_doc_timeout_bonus(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
run:
  extract_structured_doc_timeout_bonus_seconds: 90
""",
        encoding="utf-8",
    )

    from data_agent_baseline.config import load_app_config

    config = load_app_config(config_path)

    assert config.run.extract_structured_doc_timeout_bonus_seconds == 90


def test_load_app_config_uses_default_extract_structured_doc_worker_limit(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("run:\n  max_workers: 5\n", encoding="utf-8")

    from data_agent_baseline.config import load_app_config

    config = load_app_config(config_path)

    assert config.run.extract_structured_doc_max_workers == 2


def test_load_app_config_rejects_invalid_extract_structured_doc_worker_limit(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
run:
  extract_structured_doc_max_workers: 0
""",
        encoding="utf-8",
    )

    from data_agent_baseline.config import load_app_config

    with pytest.raises(ValueError, match="run.extract_structured_doc_max_workers"):
        load_app_config(config_path)


def test_load_app_config_uses_default_structured_doc_tool_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("tool:\n  max_output_tokens: 1234\n", encoding="utf-8")

    from data_agent_baseline.config import load_app_config

    config = load_app_config(config_path)

    assert config.tool.structured_doc.min_chunk_lines == 25
    assert config.tool.structured_doc.max_chunk_lines == 40
    assert config.tool.structured_doc.max_selected_lines_for_llm_extraction == 400
    assert config.tool.structured_doc.default_max_model_calls == 20
    assert config.tool.structured_doc.hard_max_model_calls == 20
    assert config.tool.structured_doc.inspect_doc_structure_max_model_calls == 3
    assert config.tool.structured_doc.llm.temperature == 0.0
    assert config.tool.structured_doc.llm.top_p == 1.0
    assert config.tool.structured_doc.llm.repetition_penalty == 1.0


def test_load_app_config_rejects_invalid_structured_doc_tool_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
tool:
  structured_doc:
    min_chunk_lines: 41
    max_chunk_lines: 40
""",
        encoding="utf-8",
    )

    from data_agent_baseline.config import load_app_config

    with pytest.raises(ValueError, match="min_chunk_lines"):
        load_app_config(config_path)


def test_load_app_config_rejects_negative_answer_validation_retry_limit(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
agent:
  validation_retry_limit: -1
""",
        encoding="utf-8",
    )

    from data_agent_baseline.config import load_app_config

    with pytest.raises(ValueError, match="agent.validation_retry_limit"):
        load_app_config(config_path)


def test_execute_task_passes_answer_validation_retry_limit_to_agent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context_dir = tmp_path / "context"
    context_dir.mkdir()
    task = PublicTask(
        record=TaskRecord(task_id="task_1", difficulty="easy", question="test?"),
        assets=TaskAssets(task_dir=tmp_path, context_dir=context_dir),
    )
    captured = {}

    class FakeRunResult:
        def to_dict(self) -> dict[str, object]:
            return {"task_id": "task_1", "answer": None, "failure_reason": None}

    class FakeAgent:
        def __init__(self, *, model, tools, config, trace_callback=None):  # noqa: ANN001
            del model, tools, trace_callback
            captured["validation_retry_limit"] = config.validation_retry_limit

        def run(self, task):  # noqa: ANN001
            del task
            return FakeRunResult()

    monkeypatch.setattr(runner_module, "LangGraphAgent", FakeAgent)

    runner_module.execute_task(
        task_id="task_1",
        task=task,
        model=object(),
        tools=object(),
        config=AppConfig(agent=AgentConfig(validation_retry_limit=7)),
    )

    assert captured["validation_retry_limit"] == 7


def test_load_app_config_supports_process_validator_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("agent:\n  model: test-model\n", encoding="utf-8")

    from data_agent_baseline.config import load_app_config

    config = load_app_config(config_path)

    assert config.agent.enable_process_validator is False
    assert config.process_validator.checkpoint_model_interval == 10
    assert config.process_validator.retry_limit == 5
    assert config.process_validator.recent_step_limit == 8


def test_load_app_config_supports_process_validator_overrides(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
agent:
  enable_process_validator: true
process_validator:
  checkpoint_model_interval: 5
  retry_limit: 2
  recent_step_limit: 12
""",
        encoding="utf-8",
    )

    from data_agent_baseline.config import load_app_config

    config = load_app_config(config_path)

    assert config.agent.enable_process_validator is True
    assert config.process_validator.checkpoint_model_interval == 5
    assert config.process_validator.retry_limit == 2
    assert config.process_validator.recent_step_limit == 12


def test_task_timeout_wrapper_keeps_result_when_subprocess_cleanup_lags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root = tmp_path / "data" / "public" / "input"
    output_root = tmp_path / "artifacts" / "runs"
    config = AppConfig(
        dataset=DatasetConfig(root_path=dataset_root),
        run=RunConfig(output_dir=output_root, task_timeout_seconds=60),
    )

    class FakeQueue:
        def get(self, timeout):  # noqa: ANN001
            assert timeout == 60
            return {
                "ok": True,
                "run_result": {
                    "task_id": "task_1",
                    "answer": {"columns": ["status"], "rows": [["ok"]]},
                    "steps": [{"node": "validate_answer"}],
                    "failure_reason": None,
                    "succeeded": True,
                },
            }

        def close(self) -> None:
            pass

        def join_thread(self) -> None:
            pass

    class FakeProcess:
        def __init__(self, target, args):  # noqa: ANN001
            self.target = target
            self.args = args
            self.terminated = False
            self.killed = False

        def start(self) -> None:
            pass

        def join(self, timeout=None) -> None:  # noqa: ANN001
            pass

        def is_alive(self) -> bool:
            return not self.terminated

        def terminate(self) -> None:
            self.terminated = True

        def kill(self) -> None:
            self.killed = True
            self.terminated = True

    class FakeContext:
        def Queue(self):  # noqa: N802
            return FakeQueue()

        def Process(self, target, args):  # noqa: N802, ANN001
            return FakeProcess(target, args)

    monkeypatch.setattr(runner_module.multiprocessing, "get_context", lambda _: FakeContext())

    result = runner_module._run_single_task_with_timeout(task_id="task_1", config=config)

    assert result["succeeded"] is True
    assert result["answer"] == {"columns": ["status"], "rows": [["ok"]]}
    assert "cleanup_warning" in result


def test_run_benchmark_uses_configured_task_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root = tmp_path / "data" / "public" / "input"
    output_root = tmp_path / "artifacts" / "runs"
    _create_task(dataset_root, "task_1")
    _create_task(dataset_root, "task_2")
    _create_task(dataset_root, "task_3")
    config = AppConfig(
        dataset=DatasetConfig(root_path=dataset_root),
        agent=AgentConfig(max_steps=16, temperature=0.0),
        run=RunConfig(
            output_dir=output_root,
            run_id="selected-task-run",
            max_workers=1,
            task_ids=("task_2", "task_3"),
        ),
    )
    attempted_task_ids: list[str] = []

    def fake_run_single_task(
        *,
        task_id: str,
        config: AppConfig,
        run_output_dir: Path,
        prediction_output_root: Path | None = None,
        model=None,
        tools=None,
    ) -> TaskRunArtifacts:
        del config, prediction_output_root, model, tools
        attempted_task_ids.append(task_id)
        task_output_dir = run_output_dir / task_id
        trace_path = task_output_dir / "trace.json"
        task_output_dir.mkdir(parents=True, exist_ok=True)
        trace_path.write_text("{}", encoding="utf-8")
        return TaskRunArtifacts(
            task_id=task_id,
            task_output_dir=task_output_dir,
            prediction_csv_path=None,
            trace_path=trace_path,
            succeeded=True,
            failure_reason=None,
        )

    monkeypatch.setattr(runner_module, "run_single_task", fake_run_single_task)

    _, artifacts = run_benchmark(config=config, model=object())

    assert attempted_task_ids == ["task_2", "task_3"]
    assert [artifact.task_id for artifact in artifacts] == ["task_2", "task_3"]


def test_run_benchmark_writes_summary_when_sequential_run_is_interrupted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root = tmp_path / "data" / "public" / "input"
    output_root = tmp_path / "artifacts" / "runs"
    _create_task(dataset_root, "task_1")
    _create_task(dataset_root, "task_2")
    _create_task(dataset_root, "task_3")
    config = AppConfig(
        dataset=DatasetConfig(root_path=dataset_root),
        run=RunConfig(output_dir=output_root, run_id="interrupted-run", max_workers=1),
    )

    def fake_run_single_task(
        *,
        task_id: str,
        config: AppConfig,
        run_output_dir: Path,
        prediction_output_root: Path | None = None,
        model=None,
        tools=None,
    ) -> TaskRunArtifacts:
        del config, prediction_output_root, model, tools
        if task_id == "task_2":
            raise KeyboardInterrupt
        task_output_dir = run_output_dir / task_id
        task_output_dir.mkdir(parents=True, exist_ok=True)
        trace_path = task_output_dir / "trace.json"
        trace_path.write_text("{}", encoding="utf-8")
        return TaskRunArtifacts(
            task_id=task_id,
            task_output_dir=task_output_dir,
            prediction_csv_path=None,
            trace_path=trace_path,
            succeeded=True,
            failure_reason=None,
        )

    monkeypatch.setattr(runner_module, "run_single_task", fake_run_single_task)

    run_output_dir, artifacts = run_benchmark(config=config, model=object())

    assert [artifact.task_id for artifact in artifacts] == ["task_1", "task_2", "task_3"]
    assert artifacts[0].succeeded is True
    assert [artifact.failure_reason for artifact in artifacts[1:]] == [
        runner_module.INTERRUPTED_FAILURE_REASON,
        runner_module.INTERRUPTED_FAILURE_REASON,
    ]
    summary_payload = json.loads((run_output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary_payload["interrupted"] is True
    assert summary_payload["succeeded_task_count"] == 1
    assert [task["failure_reason"] for task in summary_payload["tasks"][1:]] == [
        runner_module.INTERRUPTED_FAILURE_REASON,
        runner_module.INTERRUPTED_FAILURE_REASON,
    ]
    assert (run_output_dir / "task_status.jsonl").exists()


def test_run_benchmark_writes_summary_when_parallel_run_is_interrupted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root = tmp_path / "data" / "public" / "input"
    output_root = tmp_path / "artifacts" / "runs"
    _create_task(dataset_root, "task_1")
    _create_task(dataset_root, "task_2")
    config = AppConfig(
        dataset=DatasetConfig(root_path=dataset_root),
        run=RunConfig(output_dir=output_root, run_id="parallel-interrupted-run", max_workers=2),
    )

    class FakeFuture:
        def __init__(self, task_id: str) -> None:
            self.task_id = task_id
            self.cancelled = False

        def cancel(self) -> bool:
            self.cancelled = True
            return True

    class FakeExecutor:
        futures: list[FakeFuture] = []
        shutdown_calls: list[dict[str, object]] = []

        def __init__(self, max_workers: int) -> None:
            assert max_workers == 2
            self.futures = []
            FakeExecutor.futures = self.futures
            FakeExecutor.shutdown_calls = []

        def submit(self, fn, **kwargs):  # noqa: ANN001
            del fn
            future = FakeFuture(kwargs["task_id"])
            self.futures.append(future)
            return future

        def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
            FakeExecutor.shutdown_calls.append({"wait": wait, "cancel_futures": cancel_futures})

    def interrupted_as_completed(_futures):  # noqa: ANN001
        raise KeyboardInterrupt
        yield  # pragma: no cover

    monkeypatch.setattr(runner_module, "ThreadPoolExecutor", FakeExecutor)
    monkeypatch.setattr(runner_module, "as_completed", interrupted_as_completed)

    run_output_dir, artifacts = run_benchmark(config=config)

    assert [artifact.task_id for artifact in artifacts] == ["task_1", "task_2"]
    assert all(artifact.failure_reason == runner_module.INTERRUPTED_FAILURE_REASON for artifact in artifacts)
    assert all(future.cancelled for future in FakeExecutor.futures)
    assert {"wait": False, "cancel_futures": True} in FakeExecutor.shutdown_calls
    summary_payload = json.loads((run_output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary_payload["interrupted"] is True
    assert summary_payload["succeeded_task_count"] == 0


def test_extract_structured_doc_gate_releases_worker_slot_for_simple_tasks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root = tmp_path / "data" / "public" / "input"
    output_root = tmp_path / "artifacts" / "runs"
    for task_id in ("task_1", "task_2", "task_3", "task_4"):
        _create_task(dataset_root, task_id)
    config = AppConfig(
        dataset=DatasetConfig(root_path=dataset_root),
        run=RunConfig(
            output_dir=output_root,
            run_id="extract-gate-run",
            max_workers=3,
            extract_structured_doc_max_workers=1,
            task_timeout_seconds=60,
        ),
    )
    extract_tasks = {"task_1", "task_2", "task_3"}

    def task_result(task_id: str) -> dict[str, object]:
        return {
            "type": runner_module._TOOL_GATE_TASK_RESULT,
            "task_id": task_id,
            "ok": True,
            "run_result": {
                "task_id": task_id,
                "answer": {"columns": ["task_id"], "rows": [[task_id]]},
                "steps": [],
                "failure_reason": None,
                "succeeded": True,
            },
        }

    class FakeQueue:
        def __init__(self) -> None:
            self.events: list[dict[str, object]] = []

        def put(self, event: dict[str, object]) -> None:
            self.events.append(event)

        def get(self, timeout=None):  # noqa: ANN001
            del timeout
            if not self.events:
                raise runner_module.Empty
            return self.events.pop(0)

        def get_nowait(self):
            return self.get()

        def close(self) -> None:
            pass

        def join_thread(self) -> None:
            pass

    class FakeEvent:
        def __init__(self) -> None:
            self.task_id: str | None = None
            self.queue: FakeQueue | None = None

        def clear(self) -> None:
            pass

        def set(self) -> None:
            assert self.task_id is not None
            assert self.queue is not None
            self.queue.put(
                {
                    "type": runner_module._TOOL_GATE_RELEASE,
                    "task_id": self.task_id,
                    "tool_name": "extract_structured_doc",
                }
            )
            self.queue.put(task_result(self.task_id))

        def wait(self) -> None:
            pass

    class FakeProcess:
        def __init__(self, target, args) -> None:  # noqa: ANN001
            del target
            self.task_id = args[0]
            self.queue: FakeQueue = args[2]
            self.grant_event: FakeEvent = args[3]
            self.grant_event.task_id = self.task_id
            self.grant_event.queue = self.queue
            self.alive = False
            self.exitcode = None

        def start(self) -> None:
            self.alive = True
            if self.task_id in extract_tasks:
                self.queue.put(
                    {
                        "type": runner_module._TOOL_GATE_REQUEST,
                        "task_id": self.task_id,
                        "tool_name": "extract_structured_doc",
                    }
                )
            else:
                self.queue.put(task_result(self.task_id))

        def join(self, timeout=None) -> None:  # noqa: ANN001
            del timeout
            self.alive = False
            self.exitcode = 0

        def is_alive(self) -> bool:
            return self.alive

        def terminate(self) -> None:
            self.alive = False
            self.exitcode = -15

        def kill(self) -> None:
            self.alive = False
            self.exitcode = -9

    class FakeContext:
        def __init__(self) -> None:
            self.queue = FakeQueue()

        def Queue(self):  # noqa: N802
            return self.queue

        def Event(self):  # noqa: N802
            return FakeEvent()

        def Process(self, target, args):  # noqa: N802, ANN001
            return FakeProcess(target, args)

    monkeypatch.setattr(runner_module.multiprocessing, "get_context", lambda _: FakeContext())

    run_output_dir, artifacts = run_benchmark(config=config)

    assert [artifact.task_id for artifact in artifacts] == [
        "task_1",
        "task_2",
        "task_3",
        "task_4",
    ]
    assert all(artifact.succeeded for artifact in artifacts)
    events = [
        json.loads(line)
        for line in (run_output_dir / "tool_gate_events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    running_extracts = 0
    max_running_extracts = 0
    for event in events:
        if event["event"] == "grant":
            running_extracts += 1
            max_running_extracts = max(max_running_extracts, running_extracts)
        elif event["event"] == "release":
            running_extracts -= 1
    assert max_running_extracts == 1
    assert any(event["event"] == "wait" and event["task_id"] == "task_2" for event in events)
    task_4_started_at = next(
        index
        for index, event in enumerate(events)
        if event["event"] == "task_started" and event["task_id"] == "task_4"
    )
    task_3_granted_at = next(
        index
        for index, event in enumerate(events)
        if event["event"] == "grant" and event["task_id"] == "task_3"
    )
    assert task_4_started_at < task_3_granted_at


def _run_fake_extract_priority_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    priorities: dict[str, int | None],
) -> list[dict[str, object]]:
    dataset_root = tmp_path / "data" / "public" / "input"
    output_root = tmp_path / "artifacts" / "runs"
    for task_id in priorities:
        _create_task(dataset_root, task_id)
    worker_count = max(1, len(priorities) - 1)
    config = AppConfig(
        dataset=DatasetConfig(root_path=dataset_root),
        run=RunConfig(
            output_dir=output_root,
            run_id="extract-priority-gate-run",
            max_workers=worker_count,
            extract_structured_doc_max_workers=1,
            task_timeout_seconds=60,
        ),
    )
    first_task_id = next(iter(priorities))
    initial_last_task_id = list(priorities)[worker_count - 1]

    def task_result(task_id: str) -> dict[str, object]:
        return {
            "type": runner_module._TOOL_GATE_TASK_RESULT,
            "task_id": task_id,
            "ok": True,
            "run_result": {
                "task_id": task_id,
                "answer": {"columns": ["task_id"], "rows": [[task_id]]},
                "steps": [],
                "failure_reason": None,
                "succeeded": True,
            },
        }

    class FakeQueue:
        def __init__(self) -> None:
            self.events: list[dict[str, object]] = []

        def put(self, event: dict[str, object]) -> None:
            self.events.append(event)

        def get(self, timeout=None):  # noqa: ANN001
            del timeout
            if not self.events:
                raise runner_module.Empty
            return self.events.pop(0)

        def get_nowait(self):
            return self.get()

        def close(self) -> None:
            pass

        def join_thread(self) -> None:
            pass

    class FakeEvent:
        def __init__(self) -> None:
            self.task_id: str | None = None
            self.queue: FakeQueue | None = None

        def clear(self) -> None:
            pass

        def set(self) -> None:
            assert self.task_id is not None
            assert self.queue is not None
            if self.task_id == first_task_id:
                return
            self.queue.put(
                {
                    "type": runner_module._TOOL_GATE_RELEASE,
                    "task_id": self.task_id,
                    "tool_name": "extract_structured_doc",
                }
            )
            self.queue.put(task_result(self.task_id))

        def wait(self) -> None:
            pass

    class FakeProcess:
        def __init__(self, target, args) -> None:  # noqa: ANN001
            del target
            self.task_id = args[0]
            self.queue: FakeQueue = args[2]
            self.grant_event: FakeEvent = args[3]
            self.grant_event.task_id = self.task_id
            self.grant_event.queue = self.queue
            self.alive = False
            self.exitcode = None

        def start(self) -> None:
            self.alive = True
            priority = priorities[self.task_id]
            event: dict[str, object] = {
                "type": runner_module._TOOL_GATE_REQUEST,
                "task_id": self.task_id,
                "tool_name": "extract_structured_doc",
                "priority_source": (
                    "estimated_chunk_count" if priority is not None else "unknown"
                ),
            }
            if priority is not None:
                event["priority_chunk_count"] = priority
                event["selected_line_count"] = priority * 10
            self.queue.put(event)
            if self.task_id == initial_last_task_id:
                self.queue.put(
                    {
                        "type": runner_module._TOOL_GATE_RELEASE,
                        "task_id": first_task_id,
                        "tool_name": "extract_structured_doc",
                    }
                )
                self.queue.put(task_result(first_task_id))

        def join(self, timeout=None) -> None:  # noqa: ANN001
            del timeout
            self.alive = False
            self.exitcode = 0

        def is_alive(self) -> bool:
            return self.alive

        def terminate(self) -> None:
            self.alive = False
            self.exitcode = -15

        def kill(self) -> None:
            self.alive = False
            self.exitcode = -9

    class FakeContext:
        def __init__(self) -> None:
            self.queue = FakeQueue()

        def Queue(self):  # noqa: N802
            return self.queue

        def Event(self):  # noqa: N802
            return FakeEvent()

        def Process(self, target, args):  # noqa: N802, ANN001
            return FakeProcess(target, args)

    monkeypatch.setattr(runner_module.multiprocessing, "get_context", lambda _: FakeContext())

    run_output_dir, artifacts = run_benchmark(config=config)

    assert all(artifact.succeeded for artifact in artifacts)
    return [
        json.loads(line)
        for line in (run_output_dir / "tool_gate_events.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]


def test_extract_structured_doc_gate_prioritizes_smaller_chunk_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = _run_fake_extract_priority_gate(
        tmp_path,
        monkeypatch,
        {
            "task_1": 9,
            "task_2": 5,
            "task_3": 1,
            "task_4": 3,
            "task_5": 99,
        },
    )

    grant_order = [event["task_id"] for event in events if event["event"] == "grant"]
    assert grant_order == ["task_1", "task_3", "task_4", "task_2", "task_5"], events
    task_3_grant = next(
        event for event in events if event["event"] == "grant" and event["task_id"] == "task_3"
    )
    assert task_3_grant["priority_chunk_count"] == 1
    assert task_3_grant["selected_line_count"] == 10


def test_extract_structured_doc_gate_keeps_fifo_for_ties_and_unknowns_last(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = _run_fake_extract_priority_gate(
        tmp_path,
        monkeypatch,
        {
            "task_1": 9,
            "task_2": None,
            "task_3": 2,
            "task_4": 2,
            "task_5": None,
            "task_6": 99,
        },
    )

    grant_order = [event["task_id"] for event in events if event["event"] == "grant"]
    assert grant_order == ["task_1", "task_3", "task_4", "task_6", "task_2", "task_5"], events
    unknown_grants = [
        event for event in events
        if event["event"] == "grant" and event["priority_chunk_count"] is None
    ]
    assert [event["task_id"] for event in unknown_grants] == ["task_2", "task_5"]


def test_extract_structured_doc_gate_uses_idle_workers_when_no_tasks_remain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root = tmp_path / "data" / "public" / "input"
    output_root = tmp_path / "artifacts" / "runs"
    for task_id in ("task_1", "task_2", "task_3"):
        _create_task(dataset_root, task_id)
    config = AppConfig(
        dataset=DatasetConfig(root_path=dataset_root),
        run=RunConfig(
            output_dir=output_root,
            run_id="extract-idle-worker-run",
            max_workers=3,
            extract_structured_doc_max_workers=1,
            task_timeout_seconds=60,
        ),
    )

    def task_result(task_id: str) -> dict[str, object]:
        return {
            "type": runner_module._TOOL_GATE_TASK_RESULT,
            "task_id": task_id,
            "ok": True,
            "run_result": {
                "task_id": task_id,
                "answer": {"columns": ["task_id"], "rows": [[task_id]]},
                "steps": [],
                "failure_reason": None,
                "succeeded": True,
            },
        }

    class FakeQueue:
        def __init__(self) -> None:
            self.events: list[dict[str, object]] = []

        def put(self, event: dict[str, object]) -> None:
            self.events.append(event)

        def get(self, timeout=None):  # noqa: ANN001
            del timeout
            if not self.events:
                raise runner_module.Empty
            return self.events.pop(0)

        def get_nowait(self):
            return self.get()

        def close(self) -> None:
            pass

        def join_thread(self) -> None:
            pass

    class FakeEvent:
        granted: list[str] = []

        def __init__(self) -> None:
            self.task_id: str | None = None
            self.queue: FakeQueue | None = None

        def clear(self) -> None:
            pass

        def set(self) -> None:
            assert self.task_id is not None
            assert self.queue is not None
            FakeEvent.granted.append(self.task_id)
            if len(FakeEvent.granted) == 3:
                for task_id in list(FakeEvent.granted):
                    self.queue.put(
                        {
                            "type": runner_module._TOOL_GATE_RELEASE,
                            "task_id": task_id,
                            "tool_name": "extract_structured_doc",
                        }
                    )
                    self.queue.put(task_result(task_id))

        def wait(self) -> None:
            pass

    class FakeProcess:
        def __init__(self, target, args) -> None:  # noqa: ANN001
            del target
            self.task_id = args[0]
            self.queue: FakeQueue = args[2]
            self.grant_event: FakeEvent = args[3]
            self.grant_event.task_id = self.task_id
            self.grant_event.queue = self.queue
            self.alive = False
            self.exitcode = None

        def start(self) -> None:
            self.alive = True
            self.queue.put(
                {
                    "type": runner_module._TOOL_GATE_REQUEST,
                    "task_id": self.task_id,
                    "tool_name": "extract_structured_doc",
                }
            )

        def join(self, timeout=None) -> None:  # noqa: ANN001
            del timeout
            self.alive = False
            self.exitcode = 0

        def is_alive(self) -> bool:
            return self.alive

        def terminate(self) -> None:
            self.alive = False
            self.exitcode = -15

        def kill(self) -> None:
            self.alive = False
            self.exitcode = -9

    class FakeContext:
        def __init__(self) -> None:
            self.queue = FakeQueue()

        def Queue(self):  # noqa: N802
            return self.queue

        def Event(self):  # noqa: N802
            return FakeEvent()

        def Process(self, target, args):  # noqa: N802, ANN001
            return FakeProcess(target, args)

    monkeypatch.setattr(runner_module.multiprocessing, "get_context", lambda _: FakeContext())

    run_output_dir, artifacts = run_benchmark(config=config)

    assert all(artifact.succeeded for artifact in artifacts)
    events = [
        json.loads(line)
        for line in (run_output_dir / "tool_gate_events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    running_extracts = 0
    max_running_extracts = 0
    grant_reasons: list[str] = []
    for event in events:
        if event["event"] == "grant":
            running_extracts += 1
            max_running_extracts = max(max_running_extracts, running_extracts)
            grant_reasons.append(event["reason"])
        elif event["event"] == "release":
            running_extracts -= 1

    assert max_running_extracts == 3
    assert "idle_worker" in grant_reasons


def test_run_benchmark_flat_layout_writes_predictions_to_output_and_logs_to_log_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root = tmp_path / "input"
    output_root = tmp_path / "output"
    log_root = tmp_path / "logs"
    _create_task(dataset_root, "task_1")
    config = AppConfig(
        dataset=DatasetConfig(root_path=dataset_root),
        agent=AgentConfig(max_steps=16, temperature=0.0),
        run=RunConfig(
            output_dir=output_root,
            log_dir=log_root,
            output_layout="flat",
            run_id="docker-style-run",
            max_workers=1,
        ),
    )

    def fake_execute_task(
        *,
        task_id: str,
        config: AppConfig,
        task=None,
        model=None,
        tools=None,
        trace_path=None,
    ) -> dict[str, object]:
        del config, task, model, tools, trace_path
        return {
            "task_id": task_id,
            "answer": {"columns": ["value"], "rows": [["ok"]]},
            "steps": [{"node": "model"}],
            "failure_reason": None,
            "succeeded": True,
        }

    monkeypatch.setattr(runner_module, "execute_task", fake_execute_task)

    run_output_dir, artifacts = run_benchmark(config=config, model=object(), tools=object())

    assert run_output_dir == log_root / "docker-style-run"
    assert artifacts[0].prediction_csv_path == output_root / "task_1" / "prediction.csv"
    assert (output_root / "task_1" / "prediction.csv").exists()
    assert not (output_root / "summary.json").exists()
    assert (output_root / "task_1" / "trace.json").exists()
    assert (log_root / "docker-style-run" / "summary.json").exists()
    assert (log_root / "docker-style-run" / "task_status.jsonl").exists()
    assert not (log_root / "docker-style-run" / "task_1" / "trace.json").exists()


def test_run_single_task_exposes_preprocessed_markdown_context_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root = tmp_path / "input"
    output_root = tmp_path / "output"
    _create_task(dataset_root, "task_1")
    doc_dir = dataset_root / "task_1" / "context" / "doc"
    doc_dir.mkdir(parents=True)
    (doc_dir / "example.pdf").write_bytes(b"%PDF synthetic placeholder")

    from data_agent_baseline.inspectors.semantic_catalog import build_semantic_catalog
    from data_agent_baseline.run import context_preprocessor
    from data_agent_baseline.tools.filesystem import (
        list_context_tree,
        read_doc_preview,
        search_doc_text,
    )

    monkeypatch.setattr(
        context_preprocessor,
        "pdf_to_markdown",
        lambda _: "# Converted Title\n\nneedle value\n",
    )

    config = AppConfig(
        dataset=DatasetConfig(root_path=dataset_root),
        run=RunConfig(output_dir=output_root, run_id="preprocess-run", max_workers=1),
    )

    def fake_execute_task(
        *,
        task_id: str,
        config: AppConfig,
        task=None,
        model=None,
        tools=None,
        trace_path=None,
    ) -> dict[str, object]:
        del model, tools, trace_path
        assert task is not None
        listing = list_context_tree(task)
        listed_paths = {entry["path"] for entry in listing["entries"]}
        assert "doc/example.md" in listed_paths
        assert "doc/example.pdf" not in listed_paths
        preview = read_doc_preview(task, "doc/example.md", heading="Converted Title")
        assert str(preview["preview"]).startswith("# Converted Title")
        assert search_doc_text(task, "needle")["total_matches"] == 1
        catalog = build_semantic_catalog(task, budget=config.data_inspector.sample_budget)
        assert all(not asset["asset_path"].endswith(".pdf") for asset in catalog["assets"])
        doc_schema = next(
            schema for schema in catalog["schemas"] if schema["asset_path"] == "doc/example.md"
        )
        assert doc_schema["headings"] == [{"level": 1, "text": "Converted Title"}]
        assert task.assets.context_view is not None
        assert not any(asset.visible_path.endswith(".pdf") for asset in task.assets.context_view.assets)
        return {
            "task_id": task_id,
            "answer": {"columns": ["value"], "rows": [["ok"]]},
            "steps": [{"node": "model"}],
            "failure_reason": None,
            "succeeded": True,
        }

    monkeypatch.setattr(runner_module, "execute_task", fake_execute_task)

    _, artifacts = run_benchmark(config=config, model=object(), tools=object())

    assert artifacts[0].prediction_csv_path == output_root / "preprocess-run" / "task_1" / "prediction.csv"
    assert (output_root / "preprocess-run" / "task_1" / "generated_context" / "doc" / "example.md").exists()
    assert not (output_root / "preprocess-run" / "task_1" / "context").exists()


def test_run_benchmark_flat_layout_writes_preprocessed_context_to_prediction_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root = tmp_path / "input"
    output_root = tmp_path / "output"
    log_root = tmp_path / "logs"
    _create_task(dataset_root, "task_1")
    doc_dir = dataset_root / "task_1" / "context" / "doc"
    doc_dir.mkdir(parents=True)
    (doc_dir / "example.pdf").write_bytes(b"%PDF synthetic placeholder")

    from data_agent_baseline.run import context_preprocessor

    monkeypatch.setattr(context_preprocessor, "pdf_to_markdown", lambda _: "converted text\n")

    config = AppConfig(
        dataset=DatasetConfig(root_path=dataset_root),
        run=RunConfig(
            output_dir=output_root,
            log_dir=log_root,
            output_layout="flat",
            run_id="flat-preprocess-run",
            max_workers=1,
        ),
    )

    def fake_execute_task(
        *,
        task_id: str,
        config: AppConfig,
        task=None,
        model=None,
        tools=None,
        trace_path=None,
    ) -> dict[str, object]:
        del config, task, model, tools, trace_path
        return {
            "task_id": task_id,
            "answer": {"columns": ["value"], "rows": [["ok"]]},
            "steps": [{"node": "model"}],
            "failure_reason": None,
            "succeeded": True,
        }

    monkeypatch.setattr(runner_module, "execute_task", fake_execute_task)

    run_benchmark(config=config, model=object(), tools=object())

    assert (output_root / "task_1" / "generated_context" / "doc" / "example.md").read_text(
        encoding="utf-8"
    ) == "converted text\n"
    assert not (output_root / "task_1" / "context").exists()
    assert (log_root / "flat-preprocess-run" / "summary.json").exists()


def test_run_benchmark_skip_completed_uses_flat_prediction_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root = tmp_path / "input"
    output_root = tmp_path / "output"
    log_root = tmp_path / "logs"
    _create_task(dataset_root, "task_1")
    _create_task(dataset_root, "task_2")
    completed_prediction_path = output_root / "task_1" / "prediction.csv"
    completed_prediction_path.parent.mkdir(parents=True)
    completed_prediction_path.write_text("value\nalready-done\n", encoding="utf-8")
    config = AppConfig(
        dataset=DatasetConfig(root_path=dataset_root),
        agent=AgentConfig(max_steps=16, temperature=0.0),
        run=RunConfig(
            output_dir=output_root,
            log_dir=log_root,
            output_layout="flat",
            run_id="skip-completed-run",
            max_workers=1,
        ),
    )
    attempted_task_ids: list[str] = []

    def fake_run_single_task(
        *,
        task_id: str,
        config: AppConfig,
        run_output_dir: Path,
        prediction_output_root: Path | None = None,
        model=None,
        tools=None,
    ) -> TaskRunArtifacts:
        del config, prediction_output_root, model, tools
        attempted_task_ids.append(task_id)
        task_output_dir = run_output_dir / task_id
        trace_path = task_output_dir / "trace.json"
        task_output_dir.mkdir(parents=True, exist_ok=True)
        trace_path.write_text("{}", encoding="utf-8")
        return TaskRunArtifacts(
            task_id=task_id,
            task_output_dir=task_output_dir,
            prediction_csv_path=None,
            trace_path=trace_path,
            succeeded=True,
            failure_reason=None,
        )

    monkeypatch.setattr(runner_module, "run_single_task", fake_run_single_task)

    run_output_dir, artifacts = run_benchmark(config=config, model=object(), skip_completed=True)

    assert run_output_dir == log_root / "skip-completed-run"
    assert attempted_task_ids == ["task_2"]
    assert [artifact.task_id for artifact in artifacts] == ["task_2"]
    summary_payload = json.loads((run_output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary_payload["skipped_task_count"] == 1
    assert summary_payload["skipped_task_ids"] == ["task_1"]


def test_run_benchmark_writes_trace_and_summary_with_explicit_utf8(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root = tmp_path / "data" / "public" / "input"
    output_root = tmp_path / "artifacts" / "runs"
    _create_task(dataset_root, "task_1")

    config = AppConfig(
        dataset=DatasetConfig(root_path=dataset_root),
        agent=AgentConfig(max_steps=16, temperature=0.0),
        run=RunConfig(
            output_dir=output_root,
            run_id="utf8-trace-run",
            max_workers=1,
            task_timeout_seconds=60,
        ),
    )

    def fake_execute_task(
        *,
        task_id: str,
            config: AppConfig,
            task=None,
            model=None,
            tools=None,
            trace_path=None,
        ) -> dict[str, object]:
        del config, task, model, tools, trace_path
        return {
            "task_id": task_id,
            "answer": None,
            "steps": [{"content": "emoji 🙂 and symbol \u2260"}],
            "failure_reason": None,
            "succeeded": True,
        }

    write_encodings: list[str | None] = []
    real_open = Path.open

    def tracking_open(self: Path, mode: str = "r", *args, **kwargs):
        if "w" in mode and self.is_relative_to(output_root):
            write_encodings.append(kwargs.get("encoding"))
        return real_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(runner_module, "execute_task", fake_execute_task)
    monkeypatch.setattr(Path, "open", tracking_open)

    run_output_dir, artifacts = run_benchmark(config=config, model=object(), tools=object())

    assert len(artifacts) == 1
    trace_payload = json.loads((run_output_dir / "task_1" / "trace.json").read_text(encoding="utf-8"))
    summary_payload = json.loads((run_output_dir / "summary.json").read_text(encoding="utf-8"))
    assert trace_payload["steps"][0]["content"] == "emoji 🙂 and symbol \u2260"
    assert summary_payload["tasks"][0]["task_id"] == "task_1"
    assert write_encodings
    assert set(write_encodings) == {"utf-8"}


def test_live_trace_writer_rewrites_valid_json_each_update(tmp_path: Path) -> None:
    trace_path = tmp_path / "task_1" / "trace.json"
    writer = runner_module.LiveTraceWriter(trace_path=trace_path, task_id="task_1")

    writer.start()
    first_payload = json.loads(trace_path.read_text(encoding="utf-8"))
    writer.update({"steps": [{"node": "model"}], "failure_reason": "still running"})
    second_payload = json.loads(trace_path.read_text(encoding="utf-8"))

    assert first_payload["task_id"] == "task_1"
    assert first_payload["partial"] is True
    assert first_payload["steps"] == []
    assert second_payload["steps"] == [{"node": "model"}]
    assert second_payload["failure_reason"] == "still running"
    assert second_payload["partial"] is True


def test_write_task_outputs_preserves_partial_trace_steps_for_empty_failure(tmp_path: Path) -> None:
    run_output_dir = tmp_path / "run"
    trace_path = run_output_dir / "task_1" / "trace.json"
    runner_module._write_json(
        trace_path,
        {
            "task_id": "task_1",
            "answer": None,
            "steps": [{"step_index": 1, "node": "model"}],
            "failure_reason": None,
            "succeeded": False,
            "inspector": {"perception": {"ok": True}},
            "partial": True,
        },
    )

    artifact = runner_module._write_task_outputs(
        "task_1",
        run_output_dir,
        {
            "task_id": "task_1",
            "answer": None,
            "steps": [],
            "failure_reason": "Task timed out after 60 seconds.",
            "succeeded": False,
        },
    )

    trace_payload = json.loads(trace_path.read_text(encoding="utf-8"))
    assert artifact.succeeded is False
    assert trace_payload["steps"] == [{"step_index": 1, "node": "model"}]
    assert trace_payload["failure_reason"] == "Task timed out after 60 seconds."
    assert trace_payload["succeeded"] is False
    assert trace_payload["finalized_from_partial_trace"] is True
    assert "partial" not in trace_payload
    assert not (run_output_dir / "task_1" / "prediction.csv").exists()


def test_write_task_outputs_success_overwrites_partial_trace(tmp_path: Path) -> None:
    run_output_dir = tmp_path / "run"
    trace_path = run_output_dir / "task_1" / "trace.json"
    runner_module._write_json(
        trace_path,
        {
            "task_id": "task_1",
            "answer": None,
            "steps": [{"step_index": 1, "node": "old"}],
            "failure_reason": None,
            "succeeded": False,
            "partial": True,
        },
    )

    runner_module._write_task_outputs(
        "task_1",
        run_output_dir,
        {
            "task_id": "task_1",
            "answer": {"columns": ["status"], "rows": [["ok"]]},
            "steps": [{"step_index": 1, "node": "model"}],
            "failure_reason": None,
            "succeeded": True,
        },
    )

    trace_payload = json.loads(trace_path.read_text(encoding="utf-8"))
    assert trace_payload["steps"] == [{"step_index": 1, "node": "model"}]
    assert trace_payload["succeeded"] is True
    assert "partial" not in trace_payload
    assert "finalized_from_partial_trace" not in trace_payload
    assert (run_output_dir / "task_1" / "prediction.csv").exists()


def test_run_benchmark_preserves_trace_for_max_steps_failure(tmp_path: Path) -> None:
    dataset_root = tmp_path / "data" / "public" / "input"
    output_root = tmp_path / "artifacts" / "runs"
    _create_task(dataset_root, "task_1")
    config = AppConfig(
        dataset=DatasetConfig(root_path=dataset_root),
        agent=AgentConfig(max_steps=1, temperature=0.0),
        run=RunConfig(
            output_dir=output_root,
            run_id="max-steps-trace-run",
            max_workers=1,
            task_timeout_seconds=60,
        ),
    )
    model = ScriptedToolCallingModel(
        responses=[
            AIMessage(content="I should think more before using a tool.", tool_calls=[]),
            AIMessage(content="I still cannot answer.", tool_calls=[]),
        ]
    )

    run_output_dir, artifacts = run_benchmark(config=config, model=model)

    trace_payload = json.loads((run_output_dir / "task_1" / "trace.json").read_text(encoding="utf-8"))
    assert len(artifacts) == 1
    assert artifacts[0].succeeded is False
    assert trace_payload["failure_reason"] == "Agent did not submit an answer within max_steps."
    assert [step["node"] for step in trace_payload["steps"]] == ["model", "force_answer"]
    assert "partial" not in trace_payload


def test_write_json_is_atomic_when_replace_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target_path = tmp_path / "trace.json"
    target_path.write_text('{"status":"old"}\n', encoding="utf-8")

    def fail_replace(self: Path, target: Path) -> Path:
        raise OSError(f"synthetic replace failure for {target.name}")

    monkeypatch.setattr(Path, "replace", fail_replace)

    with pytest.raises(OSError, match="synthetic replace failure"):
        runner_module._write_json(target_path, {"status": "new"})

    assert target_path.read_text(encoding="utf-8") == '{"status":"old"}\n'
    assert list(tmp_path.iterdir()) == [target_path]


def test_write_task_outputs_recovers_answer_on_failure(tmp_path: Path) -> None:
    run_output_dir = tmp_path / "run"
    trace_path = run_output_dir / "task_1" / "trace.json"
    
    # 模拟在超时被杀死之前，已经在 trace.json 中写好了 answer
    runner_module._write_json(
        trace_path,
        {
            "task_id": "task_1",
            "answer": {"columns": ["col_test"], "rows": [["ans_val"]]},
            "steps": [{"step_index": 1, "node": "model"}],
            "failure_reason": None,
            "succeeded": False,
            "partial": True,
        },
    )

    # 模拟由于超时或异常导致的失败结果 payload，answer 为 None
    runner_module._write_task_outputs(
        "task_1",
        run_output_dir,
        {
            "task_id": "task_1",
            "answer": None,
            "steps": [],
            "failure_reason": "Task timed out after 60 seconds.",
            "succeeded": False,
        },
    )

    trace_payload = json.loads(trace_path.read_text(encoding="utf-8"))
    
    # 确认最终 trace 中的 answer 被成功恢复
    assert trace_payload["answer"] == {"columns": ["col_test"], "rows": [["ans_val"]]}
    assert trace_payload["failure_reason"] == "Task timed out after 60 seconds."
    assert trace_payload["succeeded"] is False
    assert trace_payload["finalized_from_partial_trace"] is True
    assert "partial" not in trace_payload
    
    # 确认 prediction.csv 成功写盘，达成了保底要求！
    assert (run_output_dir / "task_1" / "prediction.csv").exists()
    
    csv_content = (run_output_dir / "task_1" / "prediction.csv").read_text(encoding="utf-8")
    assert csv_content.replace("\r\n", "\n").strip().split("\n") == ["col_test", "ans_val"]

