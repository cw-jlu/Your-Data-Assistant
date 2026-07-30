from __future__ import annotations

import csv
import heapq
import json
import multiprocessing
import statistics
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty
from time import perf_counter
from typing import Any
from uuid import uuid4

from data_agent_baseline.agents.langgraph_runtime import LangGraphAgent, LangGraphAgentConfig
from data_agent_baseline.agents.model import create_chat_model
from data_agent_baseline.benchmark.dataset import DABenchPublicDataset
from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.config import AppConfig
from data_agent_baseline.run.context_preprocessor import prepare_task_context
from data_agent_baseline.run.video_understanding_agent import add_video_understanding_summaries
from data_agent_baseline.tools.registry import ToolRegistry, create_default_tool_registry


# 每个任务运行结束后需要落盘保存的产物元信息。
@dataclass(frozen=True, slots=True)
class TaskRunArtifacts:
    task_id: str
    task_output_dir: Path
    prediction_csv_path: Path | None
    trace_path: Path
    succeeded: bool
    failure_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_output_dir": str(self.task_output_dir),
            "prediction_csv_path": str(self.prediction_csv_path) if self.prediction_csv_path else None,
            "trace_path": str(self.trace_path),
            "succeeded": self.succeeded,
            "failure_reason": self.failure_reason,
        }


@dataclass(frozen=True, slots=True)
class BenchmarkOutputDirs:
    run_id: str
    run_output_dir: Path
    prediction_output_root: Path


# 使用 UTC 时间戳作为 run_id，便于排序且不受本地时区影响。
def create_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# 接受显式传入的 run_id；如果没有提供则自动生成，同时禁止路径形式的输入。
def resolve_run_id(run_id: str | None = None) -> str:
    if run_id is None:
        return create_run_id()

    normalized = run_id.strip()
    if not normalized:
        raise ValueError("run_id must not be empty.")
    if normalized in {".", ".."} or "/" in normalized or "\\" in normalized:
        raise ValueError("run_id must be a single directory name, not a path.")
    return normalized


# 为一次 benchmark 运行创建独立的输出目录，用来存放所有产物。
def create_run_output_dir(output_root: Path, *, run_id: str | None = None) -> tuple[str, Path]:
    effective_run_id = resolve_run_id(run_id)
    run_output_dir = output_root / effective_run_id
    run_output_dir.mkdir(parents=True, exist_ok=False)
    return effective_run_id, run_output_dir


def create_benchmark_output_dirs(config: AppConfig) -> BenchmarkOutputDirs:
    if config.run.output_layout == "run_dir":
        effective_run_id, run_output_dir = create_run_output_dir(
            config.run.output_dir,
            run_id=config.run.run_id,
        )
        return BenchmarkOutputDirs(
            run_id=effective_run_id,
            run_output_dir=run_output_dir,
            prediction_output_root=run_output_dir,
        )

    if config.run.output_layout == "flat":
        if config.run.log_dir is None:
            raise ValueError("run.log_dir is required when run.output_layout is `flat`.")
        effective_run_id, run_output_dir = create_run_output_dir(
            config.run.log_dir,
            run_id=config.run.run_id,
        )
        config.run.output_dir.mkdir(parents=True, exist_ok=True)
        return BenchmarkOutputDirs(
            run_id=effective_run_id,
            run_output_dir=run_output_dir,
            prediction_output_root=config.run.output_dir,
        )

    raise ValueError("run.output_layout must be either `run_dir` or `flat`.")


# 根据配置构造聊天模型实例。
def build_chat_model(config: AppConfig):
    return create_chat_model(
        model=config.agent.model,
        api_base=config.agent.api_base,
        api_key=config.agent.api_key,
        api_key_env=config.agent.api_key_env,
        temperature=config.agent.temperature,
        timeout_seconds=config.agent.model_request_timeout_seconds,
        max_tokens=config.agent.max_tokens,
    )


def build_asr_prompt_model(config: AppConfig):
    return create_chat_model(
        model=config.agent.model,
        api_base=config.agent.api_base,
        api_key=config.agent.api_key,
        api_key_env=config.agent.api_key_env,
        temperature=0.0,
        timeout_seconds=config.agent.model_request_timeout_seconds,
        max_tokens=512,
        top_p=1.0,
        repetition_penalty=1.0,
        enable_thinking=False,
    )


# 供任务产物落盘复用的简单文件写入辅助函数。
def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _write_text_atomic(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class LiveTraceWriter:
    def __init__(self, *, trace_path: Path, task_id: str) -> None:
        self.trace_path = trace_path
        self.task_id = task_id
        self.payload: dict[str, Any] = {
            "task_id": task_id,
            "answer": None,
            "steps": [],
            "failure_reason": None,
            "succeeded": False,
            "inspector": None,
            "partial": True,
            "updated_at": _utc_timestamp(),
        }

    def start(self) -> None:
        self._write()

    def update(self, payload: dict[str, Any]) -> None:
        self.payload = {
            **self.payload,
            **payload,
            "task_id": self.task_id,
            "updated_at": _utc_timestamp(),
        }
        self._write()

    def _write(self) -> None:
        _write_json(self.trace_path, self.payload)


_ATOMIC_REPLACE_RETRIES = 3
_ATOMIC_REPLACE_BACKOFF_BASE = 0.1


def _atomic_replace(temp_path: Path, target_path: Path) -> None:
    last_error: Exception | None = None
    for attempt in range(_ATOMIC_REPLACE_RETRIES):
        try:
            temp_path.replace(target_path)
            return
        except PermissionError as exc:
            last_error = exc
            if attempt < _ATOMIC_REPLACE_RETRIES - 1:
                time.sleep(_ATOMIC_REPLACE_BACKOFF_BASE * (2 ** attempt))

    try:
        target_path.write_text(temp_path.read_text(encoding="utf-8"), encoding="utf-8")
    except Exception as fallback_exc:
        raise last_error from fallback_exc


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            handle.write(content)
        _atomic_replace(temp_path, path)
    except BaseException:  # noqa: BLE001
        temp_path.unlink(missing_ok=True)
        raise


def _write_csv(path: Path, columns: list[str], rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temp_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            for row in rows:
                writer.writerow(row)
        _atomic_replace(temp_path, path)
    except BaseException:  # noqa: BLE001
        temp_path.unlink(missing_ok=True)
        raise


def _write_jsonl(path: Path, payloads: list[dict[str, Any]]) -> None:
    _write_text_atomic(
        path,
        "".join(json.dumps(payload, ensure_ascii=False) + "\n" for payload in payloads),
    )


def _prediction_csv_path(prediction_output_root: Path, task_id: str) -> Path:
    return prediction_output_root / task_id / "prediction.csv"


def _has_completed_prediction(prediction_output_root: Path, task_id: str) -> bool:
    return _prediction_csv_path(prediction_output_root, task_id).is_file()


def _prepare_task_for_run(
    *,
    task_id: str,
    config: AppConfig,
    task_output_dir: Path,
    model=None,
) -> PublicTask:
    original_task = DABenchPublicDataset(config.dataset.root_path).get_task(task_id)
    prompt_model = model
    if (
        prompt_model is None
        and config.video_preprocessing.asr_stage
        in {"dynamic-prompt", "dynamic-plus-ui"}
    ):
        try:
            prompt_model = build_asr_prompt_model(config)
        except Exception:  # noqa: BLE001
            prompt_model = None
    preprocessed_context = prepare_task_context(
        original_task,
        task_output_dir,
        video_config=config.video_preprocessing,
        prompt_model=prompt_model,
        prompt_model_name=config.agent.model,
    )
    has_video_timeline = (
        preprocessed_context.context_view is not None
        and any(
            asset.action in {"video_timeline", "video_preprocessing_failed"}
            for asset in preprocessed_context.context_view.assets
        )
    )
    if config.video_preprocessing.enabled and has_video_timeline:
        video_model = model or prompt_model or build_chat_model(config)
        preprocessed_context = add_video_understanding_summaries(
            preprocessed_context=preprocessed_context,
            task_output_dir=task_output_dir,
            model=video_model,
            timeout_seconds=config.agent.model_request_timeout_seconds,
        )
    return preprocessed_context.task


# 统一失败结果的结构，便于后续按相同流程写出任务产物。
def _failure_run_result_payload(task_id: str, failure_reason: str) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "answer": None,
        "steps": [],
        "failure_reason": failure_reason,
        "succeeded": False,
    }


def _load_existing_trace_payload(trace_path: Path) -> dict[str, Any] | None:
    if not trace_path.exists():
        return None
    try:
        payload = json.loads(trace_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _load_existing_answer(trace_path: Path) -> dict[str, Any] | None:
    payload = _load_existing_trace_payload(trace_path)
    if isinstance(payload, dict):
        answer = payload.get("answer")
        if isinstance(answer, dict) and answer.get("columns") and answer.get("rows"):
            return answer
    return None


def _has_nonempty_steps(payload: dict[str, Any] | None) -> bool:
    if payload is None:
        return False
    steps = payload.get("steps")
    return isinstance(steps, list) and bool(steps)


def _final_trace_payload(trace_path: Path, run_result: dict[str, Any]) -> dict[str, Any]:
    payload = dict(run_result)
    existing_payload = _load_existing_trace_payload(trace_path)
    new_steps = payload.get("steps")
    should_preserve_steps = (
        not payload.get("succeeded")
        and (not isinstance(new_steps, list) or not new_steps)
        and _has_nonempty_steps(existing_payload)
    )

    if should_preserve_steps and existing_payload is not None:
        payload["steps"] = existing_payload["steps"]
        if payload.get("inspector") is None and isinstance(existing_payload.get("inspector"), dict):
            payload["inspector"] = existing_payload["inspector"]
        if payload.get("global_data_profile") is None and isinstance(existing_payload.get("global_data_profile"), str):
            payload["global_data_profile"] = existing_payload["global_data_profile"]
        if payload.get("answer") is None and existing_payload.get("answer") is not None:
            payload["answer"] = existing_payload["answer"]
        payload["finalized_from_partial_trace"] = True

    payload.pop("partial", None)
    payload["updated_at"] = _utc_timestamp()
    return payload


# 单任务执行的核心路径：加载任务、构造 agent，并返回完整运行结果。
def _run_single_task_core(
    *,
    task_id: str,
    config: AppConfig,
    task: PublicTask | None = None,
    model=None,
    tools: ToolRegistry | None = None,
    trace_callback: Callable[[dict[str, Any]], None] | None = None,
    tool_gate: Any | None = None,
) -> dict[str, Any]:
    if task is None:
        public_dataset = DABenchPublicDataset(config.dataset.root_path)
        task = public_dataset.get_task(task_id)

    agent_kwargs: dict[str, Any] = {
        "model": model or build_chat_model(config),
        "tools": tools or create_default_tool_registry(config.tool),
        "config": LangGraphAgentConfig(
            max_steps=config.agent.max_steps,
            model_request_timeout_seconds=config.agent.model_request_timeout_seconds,
            validation_retry_limit=config.agent.validation_retry_limit,
            enable_answer_validator=config.agent.enable_answer_validator,
            enable_process_validator=config.agent.enable_process_validator,
            enable_data_inspector=config.agent.enable_data_inspector,
            enable_ambiguity_analysis=config.agent.enable_ambiguity_analysis,
            strip_reasoning_history=config.agent.strip_reasoning_history,
            reasoning_history_limit=config.agent.reasoning_history_limit,
            data_inspector=config.data_inspector,
            process_validator=config.process_validator,
            prompt_version=config.agent.prompt_version,
            max_attached_video_frames=config.video_preprocessing.max_attached_frames,
            compress_used_image_messages=config.agent.compress_used_image_messages,
            compressed_image_note_chars=config.agent.compressed_image_note_chars,
        ),
        "trace_callback": trace_callback,
    }
    if tool_gate is not None:
        agent_kwargs["tool_gate"] = tool_gate
    agent = LangGraphAgent(**agent_kwargs)
    run_result = agent.run(task)
    return run_result.to_dict()


# 在子进程中执行单任务核心逻辑，便于父进程施加硬超时控制。
def _run_single_task_in_subprocess(
    task_id: str,
    config: AppConfig,
    task: PublicTask | None,
    queue: multiprocessing.Queue[Any],
    trace_path: Path | None,
    structured_doc_used_event: Any | None = None,
) -> None:
    live_trace = LiveTraceWriter(trace_path=trace_path, task_id=task_id) if trace_path is not None else None
    try:
        queue.put(
            {
                "ok": True,
                "run_result": _run_single_task_core(
                    task_id=task_id,
                    config=config,
                task=task,
                trace_callback=live_trace.update if live_trace is not None else None,
                tool_gate=_StructuredDocUsageNotifier(structured_doc_used_event)
                if structured_doc_used_event is not None
                else None,
            ),
            }
        )
    except BaseException as exc:  # noqa: BLE001
        import traceback
        queue.put(
            {
                "ok": False,
                "error": f"{repr(exc)}\n{traceback.format_exc()}",
            }
        )


# 为单任务执行增加进程级超时控制和异常退出处理。
def _run_single_task_with_timeout(
    *,
    task_id: str,
    config: AppConfig,
    task: PublicTask | None = None,
    trace_path: Path | None = None,
    trace_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    timeout_seconds = config.run.task_timeout_seconds
    if timeout_seconds <= 0:
        return _run_single_task_core(
            task_id=task_id,
            config=config,
            task=task,
            trace_callback=trace_callback,
        )

    ctx = multiprocessing.get_context("spawn")
    queue: multiprocessing.Queue[Any] = ctx.Queue()
    timeout_bonus_seconds = config.run.extract_structured_doc_timeout_bonus_seconds
    structured_doc_used_event = ctx.Event() if timeout_bonus_seconds > 0 else None
    # 子进程隔离了模型和工具执行，任务卡住时父进程可以直接终止它。
    process = ctx.Process(
        target=_run_single_task_in_subprocess,
        args=(task_id, config, task, queue, trace_path, structured_doc_used_event),
    )
    process.start()
    try:
        # 先等待子进程把结果放进队列，再回收子进程；否则在某些平台上会因为
        # 大对象仍滞留在 Queue 管道中，导致 join() 误判为超时。
        if structured_doc_used_event is None:
            result = queue.get(timeout=timeout_seconds)
        else:
            started_at = perf_counter()
            while True:
                effective_timeout_seconds = timeout_seconds + (
                    timeout_bonus_seconds if structured_doc_used_event.is_set() else 0
                )
                remaining_seconds = effective_timeout_seconds - (perf_counter() - started_at)
                if remaining_seconds <= 0:
                    raise Empty
                try:
                    result = queue.get(timeout=min(remaining_seconds, 0.1))
                    break
                except Empty:
                    continue
    except Empty:
        recovered_answer = None
        if trace_path is not None:
            recovered_answer = _load_existing_answer(trace_path)

        if process.is_alive():
            process.terminate()
            process.join(timeout=1.0)
            if process.is_alive():
                process.kill()
                process.join()
            effective_timeout_seconds = timeout_seconds + (
                timeout_bonus_seconds
                if structured_doc_used_event is not None and structured_doc_used_event.is_set()
                else 0
            )
            failure_payload = _failure_run_result_payload(task_id, f"Task timed out after {effective_timeout_seconds} seconds.")
            if recovered_answer is not None:
                failure_payload["answer"] = recovered_answer
            return failure_payload

        process.join(timeout=1.0)
        exit_code = process.exitcode
        if exit_code not in (None, 0):
            failure_payload = _failure_run_result_payload(
                task_id,
                f"Task exited unexpectedly with exit code {exit_code}.",
            )
        else:
            failure_payload = _failure_run_result_payload(task_id, "Task exited without returning a result.")
        
        if recovered_answer is not None:
            failure_payload["answer"] = recovered_answer
        return failure_payload
    finally:
        # 父进程负责关闭自身持有的队列句柄，避免后台 feeder 线程悬挂。
        queue.close()
        queue.join_thread()

    process.join(timeout=1.0)
    if process.is_alive():
        process.terminate()
        process.join(timeout=1.0)
        if process.is_alive():
            process.kill()
            process.join()
        logger_payload = (
            "Task subprocess did not exit cleanly after returning a result; "
            "using the returned result and treating cleanup as non-fatal."
        )
        if result.get("ok"):
            run_result = dict(result["run_result"])
            run_result["cleanup_warning"] = logger_payload
            return run_result
        
        recovered_answer = None
        if trace_path is not None:
            recovered_answer = _load_existing_answer(trace_path)
        failure_payload = _failure_run_result_payload(task_id, f"Task failed with uncaught error: {result['error']}")
        if recovered_answer is not None:
            failure_payload["answer"] = recovered_answer
        return failure_payload

    if result.get("ok"):
        return dict(result["run_result"])
    
    recovered_answer = None
    if trace_path is not None:
        recovered_answer = _load_existing_answer(trace_path)
    failure_payload = _failure_run_result_payload(task_id, f"Task failed with uncaught error: {result['error']}")
    if recovered_answer is not None:
        failure_payload["answer"] = recovered_answer
    return failure_payload


_TOOL_GATE_REQUEST = "tool_gate_request"
_TOOL_GATE_RELEASE = "tool_gate_release"
_TOOL_GATE_TASK_RESULT = "task_result"
_EXTRACT_STRUCTURED_DOC_TOOL = "extract_structured_doc"


class _StructuredDocUsageNotifier:
    """Marks a subprocess task as soon as it enters structured-doc extraction."""

    def __init__(self, used_event: Any) -> None:
        self.used_event = used_event

    def acquire(self, tool_name: str, **_metadata: Any) -> None:
        if tool_name == _EXTRACT_STRUCTURED_DOC_TOOL:
            self.used_event.set()

    def release(self, _tool_name: str) -> None:
        pass


class _SubprocessToolGateClient:
    def __init__(self, *, task_id: str, event_queue: Any, grant_event: Any) -> None:
        self.task_id = task_id
        self.event_queue = event_queue
        self.grant_event = grant_event

    def acquire(self, tool_name: str, **metadata: Any) -> None:
        if tool_name != _EXTRACT_STRUCTURED_DOC_TOOL:
            return
        self.grant_event.clear()
        event = {
            "type": _TOOL_GATE_REQUEST,
            "task_id": self.task_id,
            "tool_name": tool_name,
            "timestamp": _utc_timestamp(),
        }
        for key in (
            "priority_chunk_count",
            "selected_line_count",
            "priority_source",
            "priority_error",
        ):
            if key in metadata:
                event[key] = metadata[key]
        self.event_queue.put(event)
        self.grant_event.wait()

    def release(self, tool_name: str) -> None:
        if tool_name != _EXTRACT_STRUCTURED_DOC_TOOL:
            return
        self.event_queue.put(
            {
                "type": _TOOL_GATE_RELEASE,
                "task_id": self.task_id,
                "tool_name": tool_name,
                "timestamp": _utc_timestamp(),
            }
        )


@dataclass(slots=True)
class _ParallelTaskState:
    index: int
    task_id: str
    process: Any
    grant_event: Any
    trace_path: Path
    started_at: float
    status: str = "active"
    extract_granted: bool = False
    exit_observed_at: float | None = None
    suspended_at: float | None = None
    total_suspended_seconds: float = 0.0
    extract_structured_doc_timeout_bonus_applied: bool = False
    priority_chunk_count: int | None = None
    priority_selected_line_count: int | None = None
    priority_source: str = "unknown"
    wait_sequence: int | None = None


def _run_parallel_task_process(
    task_id: str,
    config: AppConfig,
    event_queue: Any,
    grant_event: Any,
    trace_path: Path,
) -> None:
    live_trace = LiveTraceWriter(trace_path=trace_path, task_id=task_id)
    try:
        live_trace.start()
        task_output_dir = trace_path.parent
        preprocessed_task = _prepare_task_for_run(
            task_id=task_id,
            config=config,
            task_output_dir=task_output_dir,
        )
        started_at = perf_counter()
        gate_client = _SubprocessToolGateClient(
            task_id=task_id,
            event_queue=event_queue,
            grant_event=grant_event,
        )
        run_result = _run_single_task_core(
            task_id=task_id,
            config=config,
            task=preprocessed_task,
            trace_callback=live_trace.update,
            tool_gate=gate_client,
        )
        run_result["e2e_elapsed_seconds"] = round(perf_counter() - started_at, 3)
        event_queue.put(
            {
                "type": _TOOL_GATE_TASK_RESULT,
                "task_id": task_id,
                "ok": True,
                "run_result": run_result,
                "timestamp": _utc_timestamp(),
            }
        )
    except BaseException as exc:  # noqa: BLE001
        import traceback

        event_queue.put(
            {
                "type": _TOOL_GATE_TASK_RESULT,
                "task_id": task_id,
                "ok": False,
                "error": f"{repr(exc)}\n{traceback.format_exc()}",
                "timestamp": _utc_timestamp(),
            }
        )


def _append_tool_gate_event(log_path: Path, payload: dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    event = {"timestamp": _utc_timestamp(), **payload}
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def _active_parallel_task_count(states: dict[str, _ParallelTaskState]) -> int:
    return sum(1 for state in states.values() if state.status in {"active", "extracting"})


def _extracting_parallel_task_count(states: dict[str, _ParallelTaskState]) -> int:
    return sum(1 for state in states.values() if state.extract_granted)


def _run_parallel_benchmark_with_extract_gate(
    *,
    config: AppConfig,
    output_dirs: BenchmarkOutputDirs,
    task_ids: list[str],
    effective_workers: int,
    progress_callback: Callable[[TaskRunArtifacts], None] | None = None,
) -> tuple[list[TaskRunArtifacts], bool]:
    ctx = multiprocessing.get_context("spawn")
    event_queue: Any = ctx.Queue()
    indexed_artifacts: list[TaskRunArtifacts | None] = [None] * len(task_ids)
    states: dict[str, _ParallelTaskState] = {}
    waiting_queue: list[tuple[int, int, int, str]] = []
    waiting_task_ids: set[str] = set()
    request_sequence = 0
    next_task_index = 0
    interrupted = False
    extract_limit = config.run.extract_structured_doc_max_workers
    gate_log_path = output_dirs.run_output_dir / "tool_gate_events.jsonl"

    def log_event(payload: dict[str, Any]) -> None:
        _append_tool_gate_event(gate_log_path, payload)

    def remove_from_waiting(task_id: str) -> None:
        waiting_task_ids.discard(task_id)

    def waiting_count() -> int:
        return len(waiting_task_ids)

    def coerce_non_negative_int(value: Any) -> int | None:
        if isinstance(value, bool) or value in (None, ""):
            return None
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed >= 0 else None

    def priority_log_fields(state: _ParallelTaskState) -> dict[str, Any]:
        return {
            "priority_chunk_count": state.priority_chunk_count,
            "selected_line_count": state.priority_selected_line_count,
            "priority_source": state.priority_source,
        }

    def update_state_priority(state: _ParallelTaskState, event: dict[str, Any]) -> None:
        state.priority_chunk_count = coerce_non_negative_int(
            event.get("priority_chunk_count")
        )
        state.priority_selected_line_count = coerce_non_negative_int(
            event.get("selected_line_count")
        )
        raw_source = event.get("priority_source")
        state.priority_source = (
            str(raw_source).strip() if raw_source not in (None, "") else "unknown"
        )

    def waiting_priority_key(
        state: _ParallelTaskState,
        sequence: int,
    ) -> tuple[int, int, int, str]:
        unknown_flag = 1 if state.priority_chunk_count is None else 0
        chunk_count = 0 if state.priority_chunk_count is None else state.priority_chunk_count
        return (unknown_flag, chunk_count, sequence, state.task_id)

    def release_grant_if_needed(state: _ParallelTaskState, *, reason: str) -> None:
        if not state.extract_granted:
            return
        state.extract_granted = False
        log_event(
            {
                "event": "release",
                "task_id": state.task_id,
                "tool_name": _EXTRACT_STRUCTURED_DOC_TOOL,
                "reason": reason,
                "extracting_count": _extracting_parallel_task_count(states),
            }
        )

    def write_artifact(index: int, task_id: str, run_result: dict[str, Any]) -> None:
        artifact = _write_task_outputs(
            task_id,
            output_dirs.run_output_dir,
            run_result,
            prediction_output_root=output_dirs.prediction_output_root,
        )
        indexed_artifacts[index] = artifact
        if progress_callback is not None:
            progress_callback(artifact)

    def finish_state(state: _ParallelTaskState, run_result: dict[str, Any]) -> None:
        release_grant_if_needed(state, reason="task_finished")
        if state.status == "waiting":
            remove_from_waiting(state.task_id)
        state.process.join(timeout=1.0)
        if state.process.is_alive():
            state.process.terminate()
            state.process.join(timeout=1.0)
            if state.process.is_alive():
                state.process.kill()
                state.process.join()
            if run_result.get("succeeded"):
                run_result = dict(run_result)
                run_result["cleanup_warning"] = (
                    "Task subprocess did not exit cleanly after returning a result; "
                    "using the returned result and treating cleanup as non-fatal."
                )
        write_artifact(state.index, state.task_id, run_result)
        states.pop(state.task_id, None)

    def fail_state(state: _ParallelTaskState, failure_reason: str) -> None:
        recovered_answer = _load_existing_answer(state.trace_path)
        failure_payload = _failure_run_result_payload(state.task_id, failure_reason)
        if recovered_answer is not None:
            failure_payload["answer"] = recovered_answer
        if state.process.is_alive():
            state.process.terminate()
            state.process.join(timeout=1.0)
            if state.process.is_alive():
                state.process.kill()
                state.process.join()
        finish_state(state, failure_payload)

    def grant_state(state: _ParallelTaskState, *, reason: str) -> None:
        if state.suspended_at is not None:
            state.total_suspended_seconds += perf_counter() - state.suspended_at
            state.suspended_at = None
        state.status = "extracting"
        state.wait_sequence = None
        state.extract_granted = True
        state.grant_event.set()
        log_event(
            {
                "event": "grant",
                "task_id": state.task_id,
                "tool_name": _EXTRACT_STRUCTURED_DOC_TOOL,
                "reason": reason,
                "active_count": _active_parallel_task_count(states),
                "extracting_count": _extracting_parallel_task_count(states),
                **priority_log_fields(state),
            }
        )

    def current_extract_limit() -> int:
        if next_task_index >= len(task_ids):
            return effective_workers
        return extract_limit

    def try_grant_waiting() -> None:
        while (
            waiting_task_ids
            and _extracting_parallel_task_count(states) < current_extract_limit()
            and _active_parallel_task_count(states) < effective_workers
        ):
            if not waiting_queue:
                break
            _unknown_flag, _chunk_count, sequence, task_id = heapq.heappop(waiting_queue)
            if task_id not in waiting_task_ids:
                continue
            state = states.get(task_id)
            if state is None or state.status != "waiting" or state.wait_sequence != sequence:
                waiting_task_ids.discard(task_id)
                continue
            waiting_task_ids.discard(task_id)
            reason = (
                "idle_worker"
                if _extracting_parallel_task_count(states) >= extract_limit
                else "queue"
            )
            grant_state(state, reason=reason)

    def start_available_tasks() -> None:
        nonlocal next_task_index
        while (
            next_task_index < len(task_ids)
            and _active_parallel_task_count(states) < effective_workers
        ):
            task_id = task_ids[next_task_index]
            task_output_dir = output_dirs.prediction_output_root / task_id
            task_output_dir.mkdir(parents=True, exist_ok=True)
            trace_path = task_output_dir / "trace.json"
            grant_event = ctx.Event()
            process = ctx.Process(
                target=_run_parallel_task_process,
                args=(task_id, config, event_queue, grant_event, trace_path),
            )
            state = _ParallelTaskState(
                index=next_task_index,
                task_id=task_id,
                process=process,
                grant_event=grant_event,
                trace_path=trace_path,
                started_at=perf_counter(),
            )
            states[task_id] = state
            process.start()
            log_event(
                {
                    "event": "task_started",
                    "task_id": task_id,
                    "active_count": _active_parallel_task_count(states),
                    "extracting_count": _extracting_parallel_task_count(states),
                }
            )
            next_task_index += 1

    def handle_gate_request(event: dict[str, Any]) -> None:
        nonlocal request_sequence
        task_id = str(event.get("task_id"))
        state = states.get(task_id)
        if state is None:
            return
        state.extract_structured_doc_timeout_bonus_applied = True
        update_state_priority(state, event)
        log_event(
            {
                "event": "request",
                "task_id": task_id,
                "tool_name": event.get("tool_name"),
                "active_count": _active_parallel_task_count(states),
                "extracting_count": _extracting_parallel_task_count(states),
                "waiting_count": waiting_count(),
                **priority_log_fields(state),
            }
        )
        if (
            waiting_count() == 0
            and _extracting_parallel_task_count(states) < current_extract_limit()
            and state.status == "active"
        ):
            reason = (
                "idle_worker"
                if _extracting_parallel_task_count(states) >= extract_limit
                else "immediate"
            )
            grant_state(state, reason=reason)
            return
        if state.status != "waiting":
            request_sequence += 1
            state.suspended_at = perf_counter()
            state.status = "waiting"
            state.wait_sequence = request_sequence
            waiting_task_ids.add(task_id)
            heapq.heappush(
                waiting_queue,
                waiting_priority_key(state, request_sequence),
            )
            log_event(
                {
                    "event": "wait",
                    "task_id": task_id,
                    "tool_name": _EXTRACT_STRUCTURED_DOC_TOOL,
                    "active_count": _active_parallel_task_count(states),
                    "extracting_count": _extracting_parallel_task_count(states),
                    "waiting_count": waiting_count(),
                    **priority_log_fields(state),
                }
            )

    def handle_gate_release(event: dict[str, Any]) -> None:
        task_id = str(event.get("task_id"))
        state = states.get(task_id)
        if state is None:
            return
        if state.extract_granted:
            state.extract_granted = False
            state.status = "active"
            log_event(
                {
                    "event": "release",
                    "task_id": task_id,
                    "tool_name": event.get("tool_name"),
                    "reason": "tool_finished",
                    "active_count": _active_parallel_task_count(states),
                    "extracting_count": _extracting_parallel_task_count(states),
                }
            )

    def handle_task_result(event: dict[str, Any]) -> None:
        task_id = str(event.get("task_id"))
        state = states.get(task_id)
        if state is None:
            return
        if event.get("ok"):
            run_result = dict(event["run_result"])
        else:
            recovered_answer = _load_existing_answer(state.trace_path)
            run_result = _failure_run_result_payload(
                task_id,
                f"Task failed with uncaught error: {event.get('error')}",
            )
            if recovered_answer is not None:
                run_result["answer"] = recovered_answer
        finish_state(state, run_result)

    def handle_event(event: dict[str, Any]) -> None:
        event_type = event.get("type")
        if event_type == _TOOL_GATE_REQUEST:
            handle_gate_request(event)
        elif event_type == _TOOL_GATE_RELEASE:
            handle_gate_release(event)
        elif event_type == _TOOL_GATE_TASK_RESULT:
            handle_task_result(event)

    def drain_events(*, block: bool = False) -> None:
        if block:
            try:
                handle_event(event_queue.get(timeout=0.1))
            except Empty:
                return
        while True:
            try:
                handle_event(event_queue.get_nowait())
            except Empty:
                break

    def check_timeouts_and_exits() -> None:
        timeout_seconds = config.run.task_timeout_seconds
        timeout_bonus_seconds = config.run.extract_structured_doc_timeout_bonus_seconds
        now = perf_counter()
        for state in list(states.values()):
            effective_timeout_seconds = timeout_seconds + (
                timeout_bonus_seconds
                if timeout_seconds > 0 and state.extract_structured_doc_timeout_bonus_applied
                else 0
            )
            if timeout_seconds > 0:
                effective_elapsed = (
                    now - state.started_at - state.total_suspended_seconds
                )
                if state.suspended_at is not None:
                    effective_elapsed -= now - state.suspended_at
                if effective_elapsed > effective_timeout_seconds:
                    log_event(
                        {
                            "event": "timeout",
                            "task_id": state.task_id,
                            "status": state.status,
                            "timeout_seconds": effective_timeout_seconds,
                            "base_timeout_seconds": timeout_seconds,
                            "extract_structured_doc_timeout_bonus_applied": state.extract_structured_doc_timeout_bonus_applied,
                            "effective_elapsed": round(effective_elapsed, 3),
                            "total_suspended_seconds": round(
                                state.total_suspended_seconds, 3
                            ),
                        }
                    )
                    fail_state(
                        state,
                        f"Task timed out after {effective_timeout_seconds} seconds "
                        f"(effective: {effective_elapsed:.1f}s).",
                    )
                    continue
            if not state.process.is_alive() and state.process.exitcode is not None:
                if state.exit_observed_at is None:
                    state.exit_observed_at = now
                    continue
                if now - state.exit_observed_at < 2.0:
                    continue
                log_event(
                    {
                        "event": "unexpected_exit",
                        "task_id": state.task_id,
                        "status": state.status,
                        "exit_code": state.process.exitcode,
                    }
                )
                if state.process.exitcode == 0:
                    fail_state(state, "Task exited without returning a result.")
                else:
                    fail_state(
                        state,
                        f"Task exited unexpectedly with exit code {state.process.exitcode}.",
                    )

    try:
        start_available_tasks()
        while next_task_index < len(task_ids) or states:
            drain_events(block=True)
            check_timeouts_and_exits()
            try_grant_waiting()
            start_available_tasks()
    except KeyboardInterrupt:
        interrupted = True
        for state in list(states.values()):
            if state.process.is_alive():
                state.process.terminate()
                state.process.join(timeout=1.0)
                if state.process.is_alive():
                    state.process.kill()
                    state.process.join()
        for index, task_id in enumerate(task_ids):
            if indexed_artifacts[index] is None:
                artifact = _interrupted_task_artifact(
                    task_id=task_id,
                    run_output_dir=output_dirs.run_output_dir,
                    prediction_output_root=output_dirs.prediction_output_root,
                )
                indexed_artifacts[index] = artifact
                if progress_callback is not None:
                    progress_callback(artifact)
    finally:
        event_queue.close()
        event_queue.join_thread()

    return [artifact for artifact in indexed_artifacts if artifact is not None], interrupted


# 为每个任务写出结构化 trace；只有产生有效答案时才写 prediction.csv。
def _write_task_outputs(
    task_id: str,
    run_output_dir: Path,
    run_result: dict[str, Any],
    *,
    prediction_output_root: Path | None = None,
) -> TaskRunArtifacts:
    task_output_dir = (prediction_output_root or run_output_dir) / task_id
    task_output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = task_output_dir / "trace.json"
    final_run_result = _final_trace_payload(trace_path, run_result)
    _write_json(trace_path, final_run_result)

    inspector = final_run_result.get("inspector")
    if isinstance(inspector, dict):
        inspector_outputs = {
            "semantic_catalog.json": inspector.get("semantic_catalog"),
            "semantic_index.json": inspector.get("semantic_index"),
            "data_understanding_handoff.json": inspector.get("data_understanding_handoff"),
        }
        for filename, payload in inspector_outputs.items():
            if isinstance(payload, dict):
                _write_json(task_output_dir / filename, payload)
    profile_text = None
    if isinstance(inspector, dict):
        profile_text = inspector.get("global_data_profile")
    if not (isinstance(profile_text, str) and profile_text.strip()):
        profile_text = final_run_result.get("global_data_profile")
    if isinstance(profile_text, str) and profile_text.strip():
        _write_text_atomic(task_output_dir / "global_data_profile.json", profile_text)

    ambiguity_analysis = final_run_result.get("ambiguity_analysis")
    if isinstance(ambiguity_analysis, dict):
        _write_json(task_output_dir / "ambiguity_analysis.json", ambiguity_analysis)

    prediction_csv_path: Path | None = None
    answer = final_run_result.get("answer")
    if isinstance(answer, dict):
        prediction_csv_path = task_output_dir / "prediction.csv"
        _write_csv(
            prediction_csv_path,
            list(answer.get("columns", [])),
            [list(row) for row in answer.get("rows", [])],
        )

    return TaskRunArtifacts(
        task_id=task_id,
        task_output_dir=task_output_dir,
        prediction_csv_path=prediction_csv_path,
        trace_path=trace_path,
        succeeded=bool(run_result.get("succeeded")),
        failure_reason=run_result.get("failure_reason"),
    )


INTERRUPTED_FAILURE_REASON = "Interrupted by user."


def _interrupted_task_artifact(
    *,
    task_id: str,
    run_output_dir: Path,
    prediction_output_root: Path,
) -> TaskRunArtifacts:
    return _write_task_outputs(
        task_id,
        run_output_dir,
        _failure_run_result_payload(task_id, INTERRUPTED_FAILURE_REASON),
        prediction_output_root=prediction_output_root,
    )


def _write_benchmark_summary(
    *,
    run_output_dir: Path,
    output_dirs: BenchmarkOutputDirs,
    config: AppConfig,
    effective_run_id: str,
    effective_workers: int,
    task_artifacts: list[TaskRunArtifacts],
    skipped_task_ids: list[str],
    total_elapsed_seconds: float,
    interrupted: bool = False,
) -> None:
    task_status_path = run_output_dir / "task_status.jsonl"
    _write_jsonl(task_status_path, [artifact.to_dict() for artifact in task_artifacts])

    summary_path = run_output_dir / "summary.json"
    _write_json(
        summary_path,
        {
            "run_id": effective_run_id,
            "output_layout": config.run.output_layout,
            "run_output_dir": str(run_output_dir),
            "prediction_output_root": str(output_dirs.prediction_output_root),
            "log_dir": str(config.run.log_dir) if config.run.log_dir is not None else None,
            "task_count": len(task_artifacts),
            "skipped_task_count": len(skipped_task_ids),
            "skipped_task_ids": skipped_task_ids,
            "succeeded_task_count": sum(1 for artifact in task_artifacts if artifact.succeeded),
            "total_elapsed_seconds": round(total_elapsed_seconds, 3),
            "max_workers": effective_workers,
            "extract_structured_doc_max_workers": config.run.extract_structured_doc_max_workers,
            "task_timeout_seconds": config.run.task_timeout_seconds,
            "extract_structured_doc_timeout_bonus_seconds": config.run.extract_structured_doc_timeout_bonus_seconds,
            "max_steps": config.agent.max_steps,
            "temperature": config.agent.temperature,
            "model_request_timeout_seconds": config.agent.model_request_timeout_seconds,
            "max_tokens": config.agent.max_tokens,
            "enable_data_inspector": config.agent.enable_data_inspector,
            "enable_answer_validator": config.agent.enable_answer_validator,
            "validation_retry_limit": config.agent.validation_retry_limit,
            "enable_ambiguity_analysis": config.agent.enable_ambiguity_analysis,
            "strip_reasoning_history": config.agent.strip_reasoning_history,
            "reasoning_history_limit": config.agent.reasoning_history_limit,
            "compress_used_image_messages": config.agent.compress_used_image_messages,
            "compressed_image_note_chars": config.agent.compressed_image_note_chars,
            "prompt_version": config.agent.prompt_version,
            "interrupted": interrupted,
            "data_inspector": {
                "catalog_top_distinct_values": config.data_inspector.sample_budget.catalog_top_distinct_values,
                "max_doc_tokens": config.data_inspector.sample_budget.max_doc_tokens,
            },
            "video_preprocessing": {
                "enabled": config.video_preprocessing.enabled,
                "sample_fps": config.video_preprocessing.sample_fps,
                "diff_threshold": config.video_preprocessing.diff_threshold,
                "pixel_delta": config.video_preprocessing.pixel_delta,
                "min_stable_duration": config.video_preprocessing.min_stable_duration,
                "resize_width": config.video_preprocessing.resize_width,
                "jpg_quality": config.video_preprocessing.jpg_quality,
                "max_attached_frames": config.video_preprocessing.max_attached_frames,
                "asr_model": config.video_preprocessing.asr_model,
                "asr_device": config.video_preprocessing.asr_device,
                "asr_compute_type": config.video_preprocessing.asr_compute_type,
                "asr_cpu_threads": config.video_preprocessing.asr_cpu_threads,
                "asr_num_workers": config.video_preprocessing.asr_num_workers,
                "asr_stage": config.video_preprocessing.asr_stage,
                "asr_language_detector_model": (
                    config.video_preprocessing.asr_language_detector_model
                ),
                "asr_language_threshold": (
                    config.video_preprocessing.asr_language_threshold
                ),
                "asr_prompt_min_terms": (
                    config.video_preprocessing.asr_prompt_min_terms
                ),
                "asr_prompt_max_terms": (
                    config.video_preprocessing.asr_prompt_max_terms
                ),
                "asr_ui_terms": list(config.video_preprocessing.asr_ui_terms),
            },
            "tool": {
                "max_output_tokens": config.tool.max_output_tokens,
                "max_list_items": config.tool.max_list_items,
            },
            "tasks": [artifact.to_dict() for artifact in task_artifacts],
        },
    )


# 公开的任务执行入口：负责运行单个任务并补齐端到端耗时，但不直接写盘。
def execute_task(
    *,
    task_id: str,
    config: AppConfig,
    task: PublicTask | None = None,
    model=None,
    tools: ToolRegistry | None = None,
    trace_path: Path | None = None,
) -> dict[str, Any]:
    started_at = perf_counter()
    live_trace = LiveTraceWriter(trace_path=trace_path, task_id=task_id) if trace_path is not None else None
    if live_trace is not None:
        live_trace.start()
    if model is None and tools is None:
        run_result = _run_single_task_with_timeout(
            task_id=task_id,
            config=config,
            task=task,
            trace_path=trace_path,
            trace_callback=live_trace.update if live_trace is not None else None,
        )
    else:
        run_result = _run_single_task_core(
            task_id=task_id,
            config=config,
            task=task,
            model=model,
            tools=tools,
            trace_callback=live_trace.update if live_trace is not None else None,
        )
    run_result["e2e_elapsed_seconds"] = round(perf_counter() - started_at, 3)
    return run_result


# 公开的单任务运行入口，负责端到端执行并记录总耗时。
def run_single_task(
    *,
    task_id: str,
    config: AppConfig,
    run_output_dir: Path,
    prediction_output_root: Path | None = None,
    model=None,
    tools: ToolRegistry | None = None,
) -> TaskRunArtifacts:
    task_output_dir = (prediction_output_root or run_output_dir) / task_id
    task_output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = task_output_dir / "trace.json"
    preprocessed_task = _prepare_task_for_run(
        task_id=task_id,
        config=config,
        task_output_dir=task_output_dir,
        model=model,
    )
    run_result = execute_task(
        task_id=task_id,
        config=config,
        task=preprocessed_task,
        model=model,
        tools=tools,
        trace_path=trace_path,
    )
    return _write_task_outputs(
        task_id,
        run_output_dir,
        run_result,
        prediction_output_root=prediction_output_root,
    )


# 运行选中的任务集合，可按需并行执行，并在最后写出本次运行的汇总信息。
def run_benchmark(
    *,
    config: AppConfig,
    model=None,
    tools: ToolRegistry | None = None,
    task_ids: list[str] | None = None,
    limit: int | None = None,
    skip_completed: bool = False,
    progress_callback: Callable[[TaskRunArtifacts], None] | None = None,
) -> tuple[Path, list[TaskRunArtifacts]]:
    start_time = perf_counter()
    output_dirs = create_benchmark_output_dirs(config)
    effective_run_id = output_dirs.run_id
    run_output_dir = output_dirs.run_output_dir

    dataset = DABenchPublicDataset(config.dataset.root_path)
    selected_task_ids = task_ids if task_ids is not None else list(config.run.task_ids or ())
    tasks = dataset.iter_tasks(task_ids=selected_task_ids or None)
    skipped_task_ids: list[str] = []
    if skip_completed:
        remaining_tasks = []
        for task in tasks:
            if _has_completed_prediction(output_dirs.prediction_output_root, task.task_id):
                skipped_task_ids.append(task.task_id)
            else:
                remaining_tasks.append(task)
        tasks = remaining_tasks
    if limit is not None:
        tasks = tasks[:limit]

    effective_workers = config.run.max_workers
    if effective_workers < 1:
        raise ValueError("max_workers must be at least 1.")
    # 如果外部直接传入了 model 或 tools，会复用这些实例，因此强制退回单 worker 执行。
    if model is not None or tools is not None:
        effective_workers = 1

    task_ids = [task.task_id for task in tasks]

    task_artifacts: list[TaskRunArtifacts]
    interrupted = False
    if effective_workers == 1:
        # 顺序执行时复用共享实例，避免每个任务重复构造模型和工具注册表。
        task_artifacts = []
        try:
            shared_model = model or build_chat_model(config)
            shared_tools = tools or create_default_tool_registry(config.tool)
            for task_id in task_ids:
                artifact = run_single_task(
                    task_id=task_id,
                    config=config,
                    run_output_dir=run_output_dir,
                    prediction_output_root=output_dirs.prediction_output_root,
                    model=shared_model,
                    tools=shared_tools,
                )
                task_artifacts.append(artifact)
                if progress_callback is not None:
                    progress_callback(artifact)
        except KeyboardInterrupt:
            interrupted = True
            completed_task_ids = {artifact.task_id for artifact in task_artifacts}
            for interrupted_task_id in task_ids:
                if interrupted_task_id in completed_task_ids:
                    continue
                artifact = _interrupted_task_artifact(
                    task_id=interrupted_task_id,
                    run_output_dir=run_output_dir,
                    prediction_output_root=output_dirs.prediction_output_root,
                )
                task_artifacts.append(artifact)
                if progress_callback is not None:
                    progress_callback(artifact)
    elif config.run.extract_structured_doc_max_workers < effective_workers:
        task_artifacts, interrupted = _run_parallel_benchmark_with_extract_gate(
            config=config,
            output_dirs=output_dirs,
            task_ids=task_ids,
            effective_workers=effective_workers,
            progress_callback=progress_callback,
        )
    else:
        # 并行执行时，每个任务各自处理超时控制与结果落盘。
        executor = ThreadPoolExecutor(max_workers=effective_workers)
        try:
            future_to_index = {
                executor.submit(
                    run_single_task,
                    task_id=task_id,
                    config=config,
                    run_output_dir=run_output_dir,
                    prediction_output_root=output_dirs.prediction_output_root,
                ): index
                for index, task_id in enumerate(task_ids)
            }
            indexed_artifacts: list[TaskRunArtifacts | None] = [None] * len(task_ids)
            try:
                for future in as_completed(future_to_index):
                    artifact = future.result()
                    indexed_artifacts[future_to_index[future]] = artifact
                    if progress_callback is not None:
                        progress_callback(artifact)
            except KeyboardInterrupt:
                interrupted = True
                executor.shutdown(wait=False, cancel_futures=True)
                for future in future_to_index:
                    future.cancel()
                for index, task_id in enumerate(task_ids):
                    if indexed_artifacts[index] is not None:
                        continue
                    artifact = _interrupted_task_artifact(
                        task_id=task_id,
                        run_output_dir=run_output_dir,
                        prediction_output_root=output_dirs.prediction_output_root,
                    )
                    indexed_artifacts[index] = artifact
                    if progress_callback is not None:
                        progress_callback(artifact)
            else:
                executor.shutdown()
            task_artifacts = [artifact for artifact in indexed_artifacts if artifact is not None]
        finally:
            if not interrupted:
                executor.shutdown(wait=False)

    total_elapsed_seconds = perf_counter() - start_time
    _write_benchmark_summary(
        run_output_dir=run_output_dir,
        output_dirs=output_dirs,
        config=config,
        effective_run_id=effective_run_id,
        effective_workers=effective_workers,
        task_artifacts=task_artifacts,
        skipped_task_ids=skipped_task_ids,
        total_elapsed_seconds=total_elapsed_seconds,
        interrupted=interrupted,
    )
    return run_output_dir, task_artifacts


@dataclass(frozen=True, slots=True)
class RepeatRunArtifacts:
    run_id: str
    run_output_dir: Path
    task_id: str
    runs: int
    workers: int
    artifacts: list[TaskRunArtifacts]
    task_scores: list[Any]
    success_count: int
    full_cover_count: int
    mean_recall: float
    mean_redundancy: float
    primary_proxy_score: float
    total_elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_output_dir": str(self.run_output_dir),
            "task_id": self.task_id,
            "runs": self.runs,
            "workers": self.workers,
            "success_count": self.success_count,
            "full_cover_count": self.full_cover_count,
            "mean_recall": round(self.mean_recall, 6),
            "mean_redundancy": round(self.mean_redundancy, 6),
            "primary_proxy_score": round(self.primary_proxy_score, 6),
            "total_elapsed_seconds": round(self.total_elapsed_seconds, 3),
            "per_run": [artifact.to_dict() for artifact in self.artifacts],
            "task_scores": [
                {
                    "recall": round(ts.recall, 6),
                    "redundancy_rate": round(ts.redundancy_rate, 6),
                    "full_cover": ts.full_cover,
                    "primary_proxy_score": round(ts.primary_proxy_score, 6),
                }
                for ts in self.task_scores
            ],
        }


def run_task_repeatedly(
    *,
    task_id: str,
    config: AppConfig,
    runs: int = 10,
    workers: int = 8,
    progress_callback: Callable[[TaskRunArtifacts], None] | None = None,
) -> RepeatRunArtifacts:
    start_time = perf_counter()
    output_dirs = create_benchmark_output_dirs(config)

    if workers < 1:
        workers = 1
    effective_workers = min(workers, runs)

    run_ids = [f"run_{i:02d}" for i in range(1, runs + 1)]

    def run_one(run_label: str) -> TaskRunArtifacts:
        run_dir = output_dirs.run_output_dir / run_label
        return run_single_task(
            task_id=task_id,
            config=config,
            run_output_dir=run_dir,
            prediction_output_root=run_dir,
        )

    indexed_artifacts: list[TaskRunArtifacts | None] = [None] * runs
    if effective_workers == 1:
        for index, run_label in enumerate(run_ids):
            indexed_artifacts[index] = run_one(run_label)
            if progress_callback is not None:
                progress_callback(indexed_artifacts[index])
    else:
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            future_to_index = {
                executor.submit(run_one, run_label): index
                for index, run_label in enumerate(run_ids)
            }
            for future in as_completed(future_to_index):
                artifact = future.result()
                indexed_artifacts[future_to_index[future]] = artifact
                if progress_callback is not None:
                    progress_callback(artifact)

    artifacts = [a for a in indexed_artifacts if a is not None]
    success_count = sum(1 for a in artifacts if a.succeeded)

    from data_agent_baseline.scoring import (
        DEFAULT_LAMBDA_GRID,
        TaskDiagnostics,
        _lambda_label,
        _primary_proxy_score,
        _score_task,
    )

    PUBLIC_GOLD_DIR = Path(__file__).resolve().parents[3] / "data" / "output"
    gold_csv_path = PUBLIC_GOLD_DIR / task_id / "gold.csv"

    task_scores: list[Any] = []
    for artifact in artifacts:
        lambda_grid = list(DEFAULT_LAMBDA_GRID)
        diagnostics = TaskDiagnostics(
            difficulty=None,
            succeeded=artifact.succeeded,
            failure_reason=artifact.failure_reason,
            e2e_elapsed_seconds=None,
            model_step_count=None,
            trace_step_count=None,
            tool_call_counts=None,
        )
        task_score = _score_task(
            task_id=task_id,
            gold_csv_path=gold_csv_path,
            prediction_csv_path=artifact.prediction_csv_path,
            lambda_grid=lambda_grid,
            diagnostics=diagnostics,
        )
        task_scores.append(task_score)

    full_cover_count = sum(1 for ts in task_scores if ts.full_cover)
    mean_recall_val = statistics.mean(ts.recall for ts in task_scores) if task_scores else 0.0
    mean_redundancy_val = statistics.mean(ts.redundancy_rate for ts in task_scores) if task_scores else 0.0
    proxy_scores = {
        _lambda_label(lv): statistics.mean(
            max(ts.recall - (lv * ts.redundancy_rate), 0.0) for ts in task_scores
        )
        for lv in list(DEFAULT_LAMBDA_GRID)
    }
    primary_score = _primary_proxy_score(proxy_scores)

    total_elapsed = perf_counter() - start_time

    result = RepeatRunArtifacts(
        run_id=output_dirs.run_id,
        run_output_dir=output_dirs.run_output_dir,
        task_id=task_id,
        runs=runs,
        workers=workers,
        artifacts=artifacts,
        task_scores=task_scores,
        success_count=success_count,
        full_cover_count=full_cover_count,
        mean_recall=mean_recall_val,
        mean_redundancy=mean_redundancy_val,
        primary_proxy_score=primary_score,
        total_elapsed_seconds=total_elapsed,
    )

    _write_json(output_dirs.run_output_dir / "summary.json", result.to_dict())

    return result
