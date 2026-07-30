"""任务运行的对外 API：`run_single_task` / `run_benchmark` + summary 落盘。

实现细节按职责拆到两个邻居模块：
- `runs/artifacts.py`  ← TaskRunArtifacts、run-id 工厂、I/O 写盘、
                          payload formatters、`write_task_outputs` 等
- `runs/subprocess.py` ← `run_single_task_core`、`run_single_task_with_timeout`
                          等子进程编排

输出目录结构：
    <output_dir>/<run_id>/
    ├── summary.json                  # 整次 run 的汇总
    └── <task_id>/
        └── prediction.csv            # 答案表格（仅 Agent 提交了 answer 时）

注意：`summary.json` 里的 `succeeded_task_count` 仅统计 Agent 是否调用了 answer，
**不代表答案正确**。真正的评分需要另外比对 `prediction.csv` 与 gold.csv。
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

from agents.application import build_application
from agents.benchmark.dataset import DABenchPublicDataset, task_id_number
from agents.config import AppConfig
from agents.runs.artifacts import (
    TaskRunArtifacts,
    blocklist_payload,
    create_run_id,
    create_run_output_dir,
    pin_key_on_config,
    write_summary_json,
    write_task_outputs,
)
from agents.runs.subprocess import (
    run_single_task_core,
    run_single_task_with_timeout,
)

if TYPE_CHECKING:
    from agents.llm.types import ModelAdapter
    from agents.tools.registry import ToolRegistry
    from agents.verification import AnswerVerifier

__all__ = [
    "TaskRunArtifacts",
    "create_run_id",
    "create_run_output_dir",
    "run_benchmark",
    "run_single_task",
]


def run_single_task(
    *,
    task_id: str,
    config: AppConfig,
    run_output_dir: Path,
    model: ModelAdapter | None = None,
    tools: ToolRegistry | None = None,
    answer_verifier: AnswerVerifier | None = None,
) -> TaskRunArtifacts:
    """对外：跑单个任务并落盘产物。

    model/tools/answer_verifier 的注入决定了隔离强度：
    - 全部为 None（默认）：走 `run_single_task_with_timeout`（子进程 + 硬超时）
    - 任一不为 None：走 `run_single_task_core`（当前进程，复用共享资源；
      用于 workers=1 或测试场景）

    PR-2：CLI `dabench run-task <id>` 不经 run_benchmark，没人替它按 task_index 固定 Key。
    这里若检测到 "配了 api_keys 但 rate_limit.state_dir 仍是 None"（= 未被上游 pin 过），
    就按 `api_keys[0]` 决定性选 Key。上游 benchmark 路径已经做过 pin 的 config 不受影响。

    blocklist：命中名单时直接产出失败 artifact，不分配 Key 也不起子进程；
    `dabench run-task <id>` 单点调用同样会短路，避免名单内任务在调试中被误跑。
    """
    if task_id in config.run.blocklist:
        return write_task_outputs(task_id, run_output_dir, blocklist_payload(task_id))

    if (
        config.agent.api_keys
        and config.agent.rate_limit is not None
        and config.agent.rate_limit.state_dir is None
    ):
        # run_output_dir 已由 caller（CLI run-task / run_benchmark）通过
        # create_run_output_dir 建好，dir name 即权威 run_id；旧实现这里调
        # resolve_run_id(config.run.run_id) 会再生成一个新 id（与 output dir 不一致），
        # 进而导致 ratelimit 状态被独立目录持有——这里直接读 dir name 修复之。
        effective_run_id = run_output_dir.name
        config = pin_key_on_config(config, task_index=0, effective_run_id=effective_run_id)

    started_at = perf_counter()
    if model is None and tools is None and answer_verifier is None:
        run_result = run_single_task_with_timeout(
            task_id=task_id,
            config=config,
        )
    else:
        run_result = run_single_task_core(
            task_id=task_id,
            config=config,
            model=model,
            tools=tools,
            answer_verifier=answer_verifier,
        )
    # end-to-end 耗时记一笔，便于后续做性能分析
    run_result["e2e_elapsed_seconds"] = round(perf_counter() - started_at, 3)
    return write_task_outputs(task_id, run_output_dir, run_result)


def _mark_interrupted_traces(config: AppConfig, *, run_id: str, task_ids: list[str]) -> None:
    if config.tracing is None or not config.tracing.enabled:
        return
    try:
        from agents.tracing.store import SQLiteTraceStore

        store = SQLiteTraceStore(config.tracing.db_path)
        try:
            for task_id in task_ids:
                store.mark_running_trace_error(
                    run_id=run_id,
                    task_id=task_id,
                    failure_reason="Run interrupted by KeyboardInterrupt.",
                    include_completed=False,
                )
        finally:
            store.close()
    except Exception:
        return


def run_benchmark(
    *,
    config: AppConfig,
    model: ModelAdapter | None = None,
    tools: ToolRegistry | None = None,
    limit: int | None = None,
    task_range: tuple[int, int] | None = None,
    progress_callback: Callable[[TaskRunArtifacts], None] | None = None,
) -> tuple[Path, list[TaskRunArtifacts]]:
    """批量跑一批任务并生成 summary.json。

    `model` / `tools` 不为 None 时：
    - 强制 workers=1（资源被共享，无法安全并发）
    - 主要用于测试中注入 Scripted 适配器 + mock tools 的端到端用例

    并发模型：
    - 当 workers == 1：串行 for，复用同一份 model/tools
    - 当 workers > 1：ThreadPoolExecutor + as_completed，每个 worker 通过子进程跑任务
      （线程池 + 子进程叠加是因为 OpenAI 调用主要是 IO 等待，线程足够释放并发；
       子进程用来硬隔离超时/崩溃）

    `task_range=(start, end)`（含两端，按 task_<n> 中的 n 过滤）和 `limit` 正交且
    顺序固定：先 range 过滤，再用 limit 截断。例如 `task_range=(5, 20), limit=3`
    跑 task_5/6/7。`progress_callback` 在每个任务完成时触发，主要供 CLI 渲染进度条。
    """
    # 先建目录再做其他事：显式 run_id 撞名立即暴露；自动 run_id 由
    # create_run_output_dir 内部 mkdir 重试解决并发冲突。
    effective_run_id, run_output_dir = create_run_output_dir(
        config.run.output_dir, run_id=config.run.run_id
    )

    dataset = DABenchPublicDataset(config.dataset.root_path)
    tasks = dataset.iter_tasks()
    if task_range is not None:
        start, end = task_range
        if start > end:
            raise ValueError(f"task_range start ({start}) must be <= end ({end}).")
        tasks = [task for task in tasks if start <= task_id_number(task.task_id) <= end]
    # limit 起 "抽样调试" 作用；切前 N 个（任务已按编号排序）
    if limit is not None:
        tasks = tasks[:limit]

    effective_workers = config.run.max_workers
    if effective_workers < 1:
        raise ValueError("max_workers must be at least 1.")
    # 共享 model/tools 时不能并行（OpenAI client 线程安全但工具注册表状态不可并发修改）
    if model is not None or tools is not None:
        effective_workers = 1

    task_ids = [task.task_id for task in tasks]

    # PR-2：配了 api_keys 就走 "按 task_index 轮询 Key" 的路径；否则保持 PR-1 单 Key 行为
    use_key_pool = bool(config.agent.api_keys) and model is None

    try:
        task_artifacts: list[TaskRunArtifacts]
        if effective_workers == 1:
            task_artifacts = []
            if use_key_pool:
                # Key 池路径：上游 config.rate_limit.state_dir 尚未解析，无法在主进程
                # 构造 RateLimitedAdapter；因此每轮先 pin_key_on_config 把 state_dir / Key
                # 落实，再交由 run_single_task 走子进程或 run_single_task_core 内的
                # build_application 完整构造。
                for index, task_id in enumerate(task_ids):
                    iter_config = pin_key_on_config(
                        config, task_index=index, effective_run_id=effective_run_id
                    )
                    artifact = run_single_task(
                        task_id=task_id,
                        config=iter_config,
                        run_output_dir=run_output_dir,
                        model=model,
                        tools=tools,
                    )
                    task_artifacts.append(artifact)
                    if progress_callback is not None:
                        progress_callback(artifact)
            else:
                # 单 Key 路径：build_application 一次构造 registry + model，全程复用，
                # 减少重复的 OpenAI 客户端 / 注册表初始化开销。
                base_app = build_application(config, model=model, tools=tools)
                for task_id in task_ids:
                    artifact = run_single_task(
                        task_id=task_id,
                        config=config,
                        run_output_dir=run_output_dir,
                        model=base_app.model_adapter,
                        tools=base_app.registry,
                        answer_verifier=base_app.answer_verifier,
                    )
                    task_artifacts.append(artifact)
                    if progress_callback is not None:
                        progress_callback(artifact)
        else:
            # 并行路径：每个线程自己调 run_single_task，内部会 spawn 子进程做超时隔离
            executor = ThreadPoolExecutor(max_workers=effective_workers)
            try:
                future_to_index = {
                    executor.submit(
                        run_single_task,
                        task_id=task_id,
                        # PR-2：按 task_index 固定 Key；池未启用时 pin_key_on_config 原样返回
                        config=pin_key_on_config(
                            config, task_index=index, effective_run_id=effective_run_id
                        ),
                        run_output_dir=run_output_dir,
                    ): index
                    for index, task_id in enumerate(task_ids)
                }
                # 按任务原始顺序占位，最后按顺序收集；保证 summary.json 的 tasks 数组稳定
                indexed_artifacts: list[TaskRunArtifacts | None] = [None] * len(task_ids)
                for future in as_completed(future_to_index):
                    artifact = future.result()
                    indexed_artifacts[future_to_index[future]] = artifact
                    # 回调在 as_completed 的主线程里串行调用，UI 侧不用自己加锁
                    if progress_callback is not None:
                        progress_callback(artifact)
                # 过滤 None（理论上所有格子都被填满；留一手防御 future.result() 意外）
                task_artifacts = [
                    artifact for artifact in indexed_artifacts if artifact is not None
                ]
            except KeyboardInterrupt:
                executor.shutdown(wait=False, cancel_futures=True)
                raise
            finally:
                executor.shutdown(wait=True)
    except KeyboardInterrupt:
        _mark_interrupted_traces(config, run_id=effective_run_id, task_ids=task_ids)
        raise

    # 整次 run 的汇总文件：run_id、任务数、成功数、并发度、各任务摘要
    summary_path = run_output_dir / "summary.json"
    write_summary_json(
        summary_path,
        {
            "run_id": effective_run_id,
            "task_count": len(task_artifacts),
            "succeeded_task_count": sum(1 for artifact in task_artifacts if artifact.succeeded),
            "max_workers": effective_workers,
            "tasks": [artifact.to_dict() for artifact in task_artifacts],
        },
    )
    return run_output_dir, task_artifacts
