"""
此模块负责任务的运行调度，包括单任务执行、批量基准测试、超时控制以及结果输出。
"""
from __future__ import annotations

import csv
import json
import multiprocessing
import traceback
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from data_agent_baseline.agents.model import OpenAIModelAdapter
from data_agent_baseline.agents.react import ReActAgent, ReActAgentConfig
from data_agent_baseline.benchmark.dataset import DABenchPublicDataset
from data_agent_baseline.config import AppConfig
from data_agent_baseline.tools.registry import ToolRegistry, create_default_tool_registry


# 记录单个任务运行结果的产物数据结构
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


# 生成基于当前时间的唯一运行 ID
def create_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def resolve_run_id(run_id: str | None = None) -> str:
    if run_id is None:
        return create_run_id()

    normalized = run_id.strip()
    if not normalized:
        raise ValueError("run_id must not be empty.")
    if normalized in {".", ".."} or "/" in normalized or "\\" in normalized:
        raise ValueError("run_id must be a single directory name, not a path.")
    return normalized


# 在 artifacts 目录下创建本次运行的输出文件夹
def create_run_output_dir(output_root: Path, *, run_id: str | None = None) -> tuple[str, Path]:
    effective_run_id = resolve_run_id(run_id)
    run_output_dir = output_root / effective_run_id
    run_output_dir.mkdir(parents=True, exist_ok=False)
    return effective_run_id, run_output_dir


def build_model_adapter(config: AppConfig):
    return OpenAIModelAdapter(
        model=config.agent.model,
        api_base=config.agent.api_base,
        api_key=config.agent.api_key,
        temperature=config.agent.temperature,
        max_tokens=config.agent.max_tokens,
        embedding_model=config.agent.embedding_model,
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, columns: list[str], rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            writer.writerow(row)


def _failure_run_result_payload(task_id: str, failure_reason: str) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "answer": None,
        "steps": [],
        "failure_reason": failure_reason,
        "succeeded": False,
    }


def _run_single_task_core(
    *,
    task_id: str,
    config: AppConfig,
    model=None,
    tools: ToolRegistry | None = None,
    task_output_dir: Path | None = None,
) -> dict[str, Any]:
    public_dataset = DABenchPublicDataset(config.dataset.root_path)
    task = public_dataset.get_task(task_id)

    agent = ReActAgent(
        model=model or build_model_adapter(config),
        tools=tools or create_default_tool_registry(python_timeout=config.agent.python_timeout),
        config=ReActAgentConfig(
            max_steps=config.agent.max_steps,
            max_repeated_actions=config.agent.max_repeated_actions,
            error_reflection_threshold=config.agent.error_reflection_threshold,
            max_consecutive_errors=config.agent.max_consecutive_errors,
        ),
    )
    run_result = agent.run(task, task_output_dir=task_output_dir)
    return run_result.to_dict()


def _run_single_task_in_subprocess(task_id: str, config: AppConfig, result_file: str, task_output_dir: Path | None = None) -> None:
    import sys
    print(f"[SUBPROCESS] Starting task {task_id}", flush=True, file=sys.stderr)
    try:
        result = _run_single_task_core(task_id=task_id, config=config, task_output_dir=task_output_dir)
        print(f"[SUBPROCESS] Task completed", flush=True, file=sys.stderr)
        payload = {"ok": True, "run_result": result}
    except BaseException as exc:  # noqa: BLE001
        print(f"[SUBPROCESS] Error: {exc}", flush=True, file=sys.stderr)
        payload = {"ok": False, "error": traceback.format_exc() or str(exc)}
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


# 运行单个任务，并根据配置支持超时自动终止
def _run_single_task_with_timeout(*, task_id: str, config: AppConfig, task_output_dir: Path | None = None) -> dict[str, Any]:
    timeout_seconds = config.run.task_timeout_seconds
    if timeout_seconds <= 0:
        return _run_single_task_core(task_id=task_id, config=config, task_output_dir=task_output_dir)

    import tempfile
    import os

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as tmp:
        result_file = tmp.name

    try:
        process = multiprocessing.Process(
            target=_run_single_task_in_subprocess,
            args=(task_id, config, result_file, task_output_dir),
        )
        process.start()
        process.join(timeout_seconds)

        if process.is_alive():
            process.terminate()
            process.join(timeout=1.0)
            if process.is_alive():
                process.kill()
                process.join()
            return _failure_run_result_payload(task_id, f"Task timed out after {timeout_seconds} seconds.")

        # Read result from temp file
        try:
            with open(result_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            exit_code = process.exitcode
            if exit_code not in (None, 0):
                return _failure_run_result_payload(
                    task_id,
                    f"Task exited unexpectedly with exit code {exit_code}.",
                )
            return _failure_run_result_payload(task_id, "Task exited without returning a result.")

        if payload.get("ok"):
            return dict(payload["run_result"])
        return _failure_run_result_payload(task_id, f"Task failed with uncaught error: {payload['error']}")
    finally:
        try:
            os.unlink(result_file)
        except OSError:
            pass


# 将任务的运行轨迹 (trace) 和预测结果 (prediction) 写入磁盘
def _write_task_outputs(task_id: str, run_output_dir: Path, run_result: dict[str, Any]) -> TaskRunArtifacts:
    task_output_dir = run_output_dir / task_id
    task_output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = task_output_dir / "trace.json"
    _write_json(trace_path, run_result)

    prediction_csv_path: Path | None = None
    answer = run_result.get("answer")
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


# 执行单个任务的完整流程：运行 Agent + 写入产物
def run_single_task(
    *,
    task_id: str,
    config: AppConfig,
    run_output_dir: Path,
    model=None,
    tools: ToolRegistry | None = None,
) -> TaskRunArtifacts:
    started_at = perf_counter()
    task_output_dir = run_output_dir / task_id
    task_output_dir.mkdir(parents=True, exist_ok=True)
    
    if model is None and tools is None:
        run_result = _run_single_task_with_timeout(task_id=task_id, config=config, task_output_dir=task_output_dir)
    else:
        run_result = _run_single_task_core(task_id=task_id, config=config, model=model, tools=tools, task_output_dir=task_output_dir)
    run_result["e2e_elapsed_seconds"] = round(perf_counter() - started_at, 3)
    return _write_task_outputs(task_id, run_output_dir, run_result)


# 批量运行多个任务（Benchmark），支持并行处理
def run_benchmark(
    *,
    config: AppConfig,
    model=None,
    tools: ToolRegistry | None = None,
    limit: int | None = None,
    progress_callback: Callable[[TaskRunArtifacts], None] | None = None,
) -> tuple[Path, list[TaskRunArtifacts]]:
    effective_run_id, run_output_dir = create_run_output_dir(config.run.output_dir, run_id=config.run.run_id)

    dataset = DABenchPublicDataset(config.dataset.root_path)
    tasks = dataset.iter_tasks()
    if limit is not None:
        tasks = tasks[:limit]

    effective_workers = config.run.max_workers
    if effective_workers < 1:
        raise ValueError("max_workers must be at least 1.")
    if model is not None or tools is not None:
        effective_workers = 1

    task_ids = [task.task_id for task in tasks]

    task_artifacts: list[TaskRunArtifacts]
    if effective_workers == 1:
        task_artifacts = []
        for task_id in task_ids:
            artifact = run_single_task(
                task_id=task_id,
                config=config,
                run_output_dir=run_output_dir,
                model=model,
                tools=tools,
            )
            task_artifacts.append(artifact)
            if progress_callback is not None:
                progress_callback(artifact)
    else:
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            future_to_index = {
                executor.submit(
                    run_single_task,
                    task_id=task_id,
                    config=config,
                    run_output_dir=run_output_dir,
                ): index
                for index, task_id in enumerate(task_ids)
            }
            indexed_artifacts: list[TaskRunArtifacts | None] = [None] * len(task_ids)
            for future in as_completed(future_to_index):
                artifact = future.result()
                indexed_artifacts[future_to_index[future]] = artifact
                if progress_callback is not None:
                    progress_callback(artifact)
            task_artifacts = [artifact for artifact in indexed_artifacts if artifact is not None]

    summary_path = run_output_dir / "summary.json"
    _write_json(
        summary_path,
        {
            "run_id": effective_run_id,
            "task_count": len(task_artifacts),
            "succeeded_task_count": sum(1 for artifact in task_artifacts if artifact.succeeded),
            "max_workers": effective_workers,
            "config": {
                "agent": {
                    "model": config.agent.model,
                    "embedding_model": config.agent.embedding_model,
                    "max_steps": config.agent.max_steps,
                    "temperature": config.agent.temperature,
                    "max_tokens": config.agent.max_tokens,
                }
            },
            "tasks": [artifact.to_dict() for artifact in task_artifacts],
        },
    )
    return run_output_dir, task_artifacts
