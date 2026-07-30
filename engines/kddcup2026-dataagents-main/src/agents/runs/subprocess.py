"""单任务的子进程编排：核心循环 + 硬超时 + crash 兜底。

三层抽象：
- `run_single_task_core`:         在当前进程跑核心 ReAct 循环并返回 dict 结果
- `_run_single_task_in_subprocess`: 子进程入口，把结果/异常塞回 multiprocessing.Queue
- `run_single_task_with_timeout`: 父进程：spawn 子进程 + 限时收 Queue + crash 兜底

`run_single_task_core` 同时供 in-process 路径（测试注入 model/tools 时）和子进程路径
共用，避免两份逻辑分叉。
"""

from __future__ import annotations

import multiprocessing
from queue import Empty
from time import perf_counter
from typing import TYPE_CHECKING, Any

from agents.application import build_application
from agents.benchmark.schema import PublicTask, TaskAssets
from agents.config import ETL_SCRATCH_ROOT, AppConfig
from agents.runs.artifacts import (
    crash_payload,
    timeout_payload,
)

if TYPE_CHECKING:
    from agents.llm.types import ModelAdapter
    from agents.tools.registry import ToolRegistry
    from agents.verification import AnswerVerifier

# 统一 spawn 起子进程：父进程在 run_benchmark / agents.submission 里带着
# ThreadPoolExecutor 线程，Linux ≤3.13 默认的 fork 会让子进程继承 fork 瞬间
# 其他线程持有的锁（import 锁等）而死锁；spawn 从干净解释器启动，且不依赖
# 平台/版本各异的默认 start method（darwin=spawn，Linux 3.14 起=forkserver）。
_MP_SPAWN = multiprocessing.get_context("spawn")


def run_single_task_core(
    *,
    task_id: str,
    config: AppConfig,
    model: ModelAdapter | None = None,
    tools: ToolRegistry | None = None,
    answer_verifier: AnswerVerifier | None = None,
) -> dict[str, Any]:
    """纯业务逻辑：构造 AgentApp → 拿任务 → 跑循环 → 返回 dict。

    `model` / `tools` / `answer_verifier` 不为 None 时透传给 `build_application` 作为 override，
    让测试无需真的初始化 OpenAI 客户端；线上 None 时走默认 wiring。
    """
    _trace_ctx = None
    _processor = None
    _trace_metadata: dict[str, Any] | None = None

    def _record_trace_failure(failure_reason: str | None) -> None:
        if _trace_metadata is None or not failure_reason:
            return
        _trace_metadata["status"] = "error"
        _trace_metadata["failure_reason"] = failure_reason

    if config.tracing is not None and config.tracing.enabled:
        from agents.tracing.context import TraceCtxManager
        from agents.tracing.processors import SimpleProcessor, SQLiteExporter
        from agents.tracing.setup import get_trace_provider
        from agents.tracing.store import SQLiteTraceStore

        _trace_metadata = {}
        _store = SQLiteTraceStore(config.tracing.db_path)
        _exporter = SQLiteExporter(_store)
        _processor = SimpleProcessor(_exporter)
        get_trace_provider().register_processor(_processor)

        _trace_ctx = TraceCtxManager(
            workflow_name="ReActAgent",
            run_id=config.run.run_id or "unknown",
            task_id=task_id,
            metadata=_trace_metadata,
        )
        _trace_ctx.__enter__()

    try:
        app = build_application(
            config,
            model=model,
            tools=tools,
            answer_verifier=answer_verifier,
        )

        if _trace_ctx is not None:
            from dataclasses import replace as _replace

            from agents.tracing.instrument import (
                TracedModelAdapter,
                TracedToolRegistry,
            )

            traced_verifier = app.answer_verifier
            if traced_verifier is not None:
                traced_verifier = _replace(
                    traced_verifier,
                    model=TracedModelAdapter(traced_verifier.model),
                )

            app = _replace(
                app,
                model_adapter=TracedModelAdapter(app.model_adapter),
                registry=TracedToolRegistry(app.registry),
                answer_verifier=traced_verifier,
            )

        task = app.dataset.get_task(task_id)

        from agents.tools.context import build_virtual_context

        virtual_ctx = build_virtual_context(task)
        task = PublicTask(
            record=task.record,
            assets=TaskAssets(task_dir=task.task_dir, context_dir=virtual_ctx),
        )

        _agent_span = None
        if _trace_ctx is not None:
            from agents.tracing.create import agent_span as _create_agent_span

            tool_names = list(app.registry.definitions.keys())
            _agent_span = _create_agent_span(
                name="ReActAgent",
                tools=tool_names,
                max_steps=app.agent_config.max_steps,
                protocol="native",
            )
            _agent_span.span_data.question = task.question
            _agent_span.start(mark_as_current=True)

        try:
            agent = app.new_agent()
            result = agent.run(task).to_dict()
            if not result.get("succeeded"):
                failure_reason = (
                    result.get("failure_reason") or "Agent did not complete successfully."
                )
                _record_trace_failure(str(failure_reason))
                if _agent_span is not None:
                    from agents.tracing.spans import SpanError

                    _agent_span.set_error(SpanError(message=str(failure_reason)))
        finally:
            if _agent_span is not None:
                _agent_span.finish(reset_current=True)
            import shutil

            shutil.rmtree(ETL_SCRATCH_ROOT / task_id / "context", ignore_errors=True)

        return result
    except BaseException as exc:
        _record_trace_failure(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if _trace_ctx is not None:
            _trace_ctx.__exit__(None, None, None)
        if _processor is not None:
            from agents.tracing.setup import get_trace_provider as _gtp

            _gtp().unregister_processor(_processor)
            _processor.shutdown()


def _mark_running_trace_error(
    task_id: str, config: AppConfig, failure_reason: str, *, status: str = "error"
) -> None:
    """Best-effort parent-process cleanup for traced children that cannot exit cleanly."""
    if config.tracing is None or not config.tracing.enabled:
        return
    try:
        from agents.tracing.store import SQLiteTraceStore

        store = SQLiteTraceStore(config.tracing.db_path)
        try:
            store.mark_running_trace_error(
                run_id=config.run.run_id or "unknown",
                task_id=task_id,
                failure_reason=failure_reason,
                status=status,
            )
        finally:
            store.close()
    except Exception:
        return


def _run_single_task_in_subprocess(
    task_id: str,
    config: AppConfig,
    queue: multiprocessing.Queue[Any],
) -> None:
    """子进程入口：跑核心逻辑并把结果/异常扔回主进程 Queue。

    注意 BaseException 而非 Exception——要兜住 KeyboardInterrupt / SystemExit，
    否则子进程退出码非 0 但 Queue 为空，主进程很难诊断。
    """
    try:
        queue.put(
            {
                "ok": True,
                "run_result": run_single_task_core(
                    task_id=task_id,
                    config=config,
                ),
            }
        )
    except BaseException as exc:
        queue.put(
            {
                "ok": False,
                "error": str(exc),
            }
        )


def run_single_task_with_timeout(
    *,
    task_id: str,
    config: AppConfig,
) -> dict[str, Any]:
    """跨进程执行带硬超时的单任务。

    三类退出路径：
    1. 超时还活着 → terminate + 兜底 kill
    2. 退出但 Queue 为空 → 进程死于 OS kill / segfault
    3. Queue 有数据 → 按 ok/error 决定是 pass-through 还是再包一层失败

    `timeout_seconds <= 0` 时绕过子进程机制，直接在当前进程跑——
    用于显式不需要超时的单测场景。
    """
    timeout_seconds = config.run.task_timeout_seconds
    if timeout_seconds <= 0:
        return run_single_task_core(task_id=task_id, config=config)

    queue: multiprocessing.Queue[Any] = _MP_SPAWN.Queue()
    process = _MP_SPAWN.Process(
        target=_run_single_task_in_subprocess,
        args=(task_id, config, queue),
    )
    process.start()

    # 不能先 join 再 get：子进程返回的 run_result 可能很大，multiprocessing.Queue 的
    # feeder 线程会在父进程不读 pipe 时卡住，导致"结果已经 put 进队列，
    # 但子进程迟迟不退出，最终被父进程按 timeout kill"。
    deadline = perf_counter() + timeout_seconds
    result: dict[str, Any] | None = None
    while True:
        remaining = deadline - perf_counter()
        if remaining <= 0:
            break
        try:
            result = queue.get(timeout=min(0.2, remaining))
            break
        except Empty:
            if not process.is_alive():
                break

    if result is None:
        try:
            result = queue.get_nowait()
        except Empty:
            result = None

    if result is not None:
        process.join(timeout=1.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=1.0)
            if process.is_alive():
                process.kill()
                process.join()
        if result.get("ok"):
            # dict(...) 拷贝一次，脱离 Queue 的底层 pickle 壳
            return dict(result["run_result"])
        payload = crash_payload(
            task_id,
            f"Task failed with uncaught error: {result['error']}",
        )
        _mark_running_trace_error(task_id, config, payload["failure_reason"])
        return payload

    # Case 1: 超时 —— 先 SIGTERM 给清理机会，1 秒仍不退就 SIGKILL
    if process.is_alive():
        process.terminate()
        process.join(timeout=1.0)
        if process.is_alive():
            process.kill()
            process.join()
        payload = timeout_payload(task_id, timeout_seconds)
        _mark_running_trace_error(task_id, config, payload["failure_reason"], status="timeout")
        return payload

    # Case 2: 进程已退出但没 put 任何结果
    exit_code = process.exitcode
    if exit_code not in (None, 0):
        payload = crash_payload(
            task_id,
            f"Task exited unexpectedly with exit code {exit_code}.",
        )
        _mark_running_trace_error(task_id, config, payload["failure_reason"])
        return payload
    # 退出码 0 但 Queue 空：逻辑异常（理论上不会发生，留个明确错误信息）
    payload = crash_payload(task_id, "Task exited without returning a result.")
    _mark_running_trace_error(task_id, config, payload["failure_reason"])
    return payload
