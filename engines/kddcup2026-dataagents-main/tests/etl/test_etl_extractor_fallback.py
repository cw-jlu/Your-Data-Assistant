from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, ClassVar

import pytest

from agents.benchmark.schema import PublicTask, TaskAssets, TaskRecord
from agents.etl import _router, extractor
from agents.llm import ModelMessage, ModelResponse


class NoFallbackAdapter:
    def complete(
        self, messages: list[ModelMessage], *, tools: object | None = None, **kwargs: Any
    ) -> ModelResponse:
        raise AssertionError("ETL should not make any post-compression LLM CSV calls")


def _make_task(tmp_path: Path) -> PublicTask:
    task_dir = tmp_path / "task_no_generated_code"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    return PublicTask(
        record=TaskRecord(task_id="task_no_generated_code", difficulty="easy", question="?"),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )


def _stub_run_etl_dependencies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(extractor, "ETL_SCRATCH_ROOT", tmp_path / "etl")
    monkeypatch.setattr(extractor, "_make_text_adapter", lambda _adapter: NoFallbackAdapter())
    monkeypatch.setattr(extractor, "parse_knowledge_schema", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(extractor, "infer_proportion_conventions", lambda *_args: {})


def test_extract_prose_file_stops_when_deterministic_csv_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = _make_task(tmp_path)
    monkeypatch.setattr(extractor, "ETL_SCRATCH_ROOT", tmp_path / "etl")
    monkeypatch.setattr(
        extractor,
        "compress_prose",
        lambda *_args, **_kwargs: ("id: 1 | value: source", None, None),
    )
    monkeypatch.setattr(extractor, "records_to_rows", lambda _table: (["id", "value"], []))

    prose_path = task.context_dir / "patient.md"
    prose_path.write_text("ID 1 has value source.", encoding="utf-8")

    result = extractor.extract_prose_file(
        NoFallbackAdapter(),
        prose_path,
        task,
        prose_path.read_text(encoding="utf-8"),
        schema_columns=["id", "value"],
        anchor_keys=["id"],
    )

    assert result is None
    assert not (tmp_path / "etl" / task.task_id / "_etl" / "patient.csv").exists()


def test_run_etl_for_task_skips_single_file_without_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = _make_task(tmp_path)
    _stub_run_etl_dependencies(tmp_path, monkeypatch)
    (task.context_dir / "note.md").write_text("incidental note\n", encoding="utf-8")
    monkeypatch.setattr(extractor, "extract_prose_file", lambda *_args, **_kwargs: None)

    result = extractor.run_etl_for_task(task, object())

    assert result is None


def test_run_etl_for_task_keeps_success_when_other_files_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = _make_task(tmp_path)
    _stub_run_etl_dependencies(tmp_path, monkeypatch)
    monkeypatch.setattr(
        _router,
        "route_prose_files",
        lambda _adapter, _question, prose_files, _km_text: list(prose_files),
    )
    for name in ("good.md", "empty_result.md", "broken.md"):
        (task.context_dir / name).write_text(f"{name} prose\n", encoding="utf-8")

    def fake_extract(_adapter: object, path: Path, *_args: object, **_kwargs: object) -> object:
        if path.name == "good.md":
            return extractor.ETLResult(
                source_file=path.name,
                csv_path=str(tmp_path / "etl" / task.task_id / "_etl" / "good.csv"),
                columns=["id"],
                row_count=1,
            )
        if path.name == "broken.md":
            raise RuntimeError("broken prose")
        return None

    monkeypatch.setattr(extractor, "extract_prose_file", fake_extract)

    result = extractor.run_etl_for_task(task, object())

    assert result is not None
    assert [item.source_file for item in result] == ["good.md"]


def test_run_etl_for_task_processes_selected_files_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = _make_task(tmp_path)
    _stub_run_etl_dependencies(tmp_path, monkeypatch)
    wanted = task.context_dir / "wanted.md"
    ignored = task.context_dir / "ignored.md"
    wanted.write_text("wanted prose\n", encoding="utf-8")
    ignored.write_text("ignored prose\n", encoding="utf-8")
    processed: list[str] = []

    def fake_extract(_adapter: object, path: Path, *_args: object, **_kwargs: object) -> object:
        processed.append(path.name)
        return extractor.ETLResult(
            source_file=path.name,
            csv_path=str(tmp_path / "etl" / task.task_id / "_etl" / f"{path.stem}.csv"),
            columns=["id"],
            row_count=1,
        )

    monkeypatch.setattr(extractor, "extract_prose_file", fake_extract)

    result = extractor.run_etl_for_task(task, object(), selected_prose_files=[wanted])

    assert result is not None
    assert [item.source_file for item in result] == ["wanted.md"]
    assert processed == ["wanted.md"]


def test_run_etl_for_task_timeout_skips_unfinished_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = _make_task(tmp_path)
    _stub_run_etl_dependencies(tmp_path, monkeypatch)
    monkeypatch.setattr(
        _router,
        "route_prose_files",
        lambda _adapter, _question, prose_files, _km_text: list(prose_files),
    )
    for name in ("good.md", "stuck.md"):
        (task.context_dir / name).write_text(f"{name} prose\n", encoding="utf-8")

    monkeypatch.setattr(
        extractor,
        "extract_prose_file",
        lambda _adapter, path, *_args, **_kwargs: extractor.ETLResult(
            source_file=path.name,
            csv_path=str(tmp_path / "etl" / task.task_id / "_etl" / f"{path.stem}.csv"),
            columns=["id"],
            row_count=1,
        ),
    )

    class FakeFuture:
        def __init__(self, result: object | None = None, *, done: bool = False) -> None:
            self._result = result
            self._done = done
            self.cancelled = False

        def result(self) -> object | None:
            return self._result

        def done(self) -> bool:
            return self._done

        def cancel(self) -> None:
            self.cancelled = True

    class FakeExecutor:
        instances: ClassVar[list[FakeExecutor]] = []

        def __init__(self, *, max_workers: int) -> None:
            self.max_workers = max_workers
            self.futures: list[FakeFuture] = []
            self.shutdown_args: tuple[bool, bool] | None = None
            FakeExecutor.instances.append(self)

        def submit(self, fn: object, *args: object, **kwargs: object) -> FakeFuture:
            path = args[-1] if args else kwargs.get("path")
            if hasattr(path, "name") and path.name == "good.md":  # type: ignore[union-attr]
                future = FakeFuture(fn(*args, **kwargs), done=True)  # type: ignore[operator]
            else:
                future = FakeFuture(done=False)
            self.futures.append(future)
            return future

        def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
            self.shutdown_args = (wait, cancel_futures)

    def fake_as_completed(futures: Iterable[FakeFuture], *, timeout: int) -> Iterator[FakeFuture]:
        assert timeout == extractor.ETL_ORCHESTRATOR_LLM_CALL_TIMEOUT * 4
        completed = [future for future in futures if future.done()]
        yield completed[0]
        raise extractor.FuturesTimeout()

    monkeypatch.setattr(extractor, "ThreadPoolExecutor", FakeExecutor)
    monkeypatch.setattr(extractor, "as_completed", fake_as_completed)

    result = extractor.run_etl_for_task(task, object())

    executor = FakeExecutor.instances[0]
    assert result is not None
    assert [item.source_file for item in result] == ["good.md"]
    assert executor.futures[1].cancelled
    assert executor.shutdown_args == (False, True)


def test_run_etl_for_task_caps_orchestrator_workers_at_configured_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = _make_task(tmp_path)
    _stub_run_etl_dependencies(tmp_path, monkeypatch)
    monkeypatch.setattr(
        _router,
        "route_prose_files",
        lambda _adapter, _question, prose_files, _km_text: list(prose_files),
    )
    for idx in range(9):
        (task.context_dir / f"doc_{idx}.md").write_text(f"doc_{idx} prose\n", encoding="utf-8")

    monkeypatch.setattr(
        extractor,
        "extract_prose_file",
        lambda _adapter, path, *_args, **_kwargs: extractor.ETLResult(
            source_file=path.name,
            csv_path=str(tmp_path / "etl" / task.task_id / "_etl" / f"{path.stem}.csv"),
            columns=["id"],
            row_count=1,
        ),
    )

    class FakeFuture:
        def __init__(self, result: object | None = None) -> None:
            self._result = result

        def result(self) -> object | None:
            return self._result

        def done(self) -> bool:
            return True

    class FakeExecutor:
        instances: ClassVar[list[FakeExecutor]] = []

        def __init__(self, *, max_workers: int) -> None:
            self.max_workers = max_workers
            FakeExecutor.instances.append(self)

        def submit(self, fn: object, *args: object, **kwargs: object) -> FakeFuture:
            return FakeFuture(fn(*args, **kwargs))  # type: ignore[operator]

        def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
            return None

    def fake_as_completed(futures: Iterable[FakeFuture], *, timeout: int) -> Iterator[FakeFuture]:
        yield from futures

    monkeypatch.setattr(extractor, "ThreadPoolExecutor", FakeExecutor)
    monkeypatch.setattr(extractor, "as_completed", fake_as_completed)

    result = extractor.run_etl_for_task(task, object())

    executor = FakeExecutor.instances[0]
    assert result is not None
    assert len(result) == 9
    assert executor.max_workers == extractor.ETL_ORCHESTRATOR_MAX_WORKERS
