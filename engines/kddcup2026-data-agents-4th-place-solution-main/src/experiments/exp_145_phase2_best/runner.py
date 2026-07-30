from __future__ import annotations

import csv
import dataclasses
import json
import multiprocessing
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from kobushi_core.benchmark.dataset import DABenchPublicDataset
from kobushi_core.model import OpenAIModelAdapter

from experiments.exp_144_modality.adaptive_vote import adaptive_vote
from experiments.exp_144_modality.config import AppConfig
from experiments.exp_144_modality.math_advisor import generate_formula
from experiments.exp_144_modality.pdf_text_cache import ensure_task_pdf_cache
from experiments.exp_144_modality.phased_agent import PhasedAgentConfig, PhasedReActAgent
from experiments.exp_144_modality.preamble import build_preamble
from experiments.exp_144_modality.prefix_cache import (
    prefix_cache_enabled,
    with_prefix_cache_header,
)
from experiments.exp_144_modality.tools.registry import create_default_tool_registry


EXPERIMENT_NAME = "exp_145_phase2_best"
MAX_STEPS = 64
ATTEMPT_TEMPERATURE = 0.6

_FIXED_PHASE2_FLAGS = {
    # Promoted from exp_144_answer_shape_prose_pdf_preprocess_source_router_008.
    "EXP144_ANSWER_SHAPE": "1",
    "EXP144_PROSE": "1",
    "EXP144_PDF_PREPROCESS": "1",
    "EXP144_SOURCE_ROUTER": "1",
    "EXP144_SOURCE_ROUTER_MODEL": "1",
    "EXP144_PREFIX_CACHE": "1",
    # Explicitly disabled ablations.
    "EXP144_VIDEO": "0",
    "EXP144_KEYFRAMES": "0",
    "EXP144_WATCH_VIDEO": "0",
    "EXP144_PROSE_INDEX": "0",
}


def _apply_fixed_profile() -> None:
    for key, value in _FIXED_PHASE2_FLAGS.items():
        os.environ[key] = value
    os.environ.setdefault("EXP144_PREFIX_CACHE_NAMESPACE", "kobushi-phase2-exp145")
    os.environ.setdefault("EXP144_PDF_TEXT_CACHE_ROOT", "/tmp/kobushi_exp145_pdf_text_cache")


_apply_fixed_profile()


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
            "prediction_csv_path": str(self.prediction_csv_path)
            if self.prediction_csv_path
            else None,
            "trace_path": str(self.trace_path),
            "succeeded": self.succeeded,
            "failure_reason": self.failure_reason,
        }


def _config_with_temperature(config: AppConfig, temperature: float) -> AppConfig:
    new_agent = dataclasses.replace(config.agent, temperature=temperature)
    return dataclasses.replace(config, agent=new_agent)


def build_model_adapter(config: AppConfig, *, task_id: str | None = None) -> OpenAIModelAdapter:
    _apply_fixed_profile()
    return OpenAIModelAdapter(
        model=config.agent.model,
        api_base=config.agent.api_base,
        api_key=config.agent.api_key,
        extra_headers=with_prefix_cache_header(config.agent.extra_headers, task_id),
        temperature=config.agent.temperature,
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n")


def _write_csv(path: Path, columns: list[str], rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
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


def _stage_log(task_id: str, message: str) -> None:
    line = f"[exp145 {task_id}] {message}"
    print(line, flush=True)
    trace_root = os.environ.get("EXP145_TRACE_ROOT")
    if trace_root:
        trace_dir = Path(trace_root) / task_id
        trace_dir.mkdir(parents=True, exist_ok=True)
        with (trace_dir / "stage.log").open("a") as handle:
            handle.write(line + "\n")


def _run_single_task_core(*, task_id: str, config: AppConfig) -> dict[str, Any]:
    _apply_fixed_profile()
    _stage_log(task_id, "load task")
    public_dataset = DABenchPublicDataset(config.dataset.root_path)
    task = public_dataset.get_task(task_id)

    pdf_cache_results = []
    try:
        _stage_log(task_id, "pdf preprocess start")
        pdf_cache_results = ensure_task_pdf_cache(task)
    except Exception as exc:
        pdf_cache_error = repr(exc)
        _stage_log(task_id, f"pdf preprocess error: {pdf_cache_error}")
    else:
        pdf_cache_error = None
        _stage_log(task_id, f"pdf preprocess done: {len(pdf_cache_results)} pdfs")

    formula = "NO_CALC"
    try:
        _stage_log(task_id, "math advisor start")
        formula = generate_formula(task.question, task_id=task_id)
    except Exception as exc:
        formula = f"NO_CALC ({exc})"
        _stage_log(task_id, f"math advisor error: {exc!r}")
    else:
        _stage_log(task_id, "math advisor done")
    skip_advisor = formula.strip().upper().startswith("NO_CALC")

    _stage_log(task_id, "build model")
    model = build_model_adapter(
        _config_with_temperature(config, ATTEMPT_TEMPERATURE),
        task_id=task_id,
    )
    _stage_log(task_id, "build preamble start")
    preamble_result = build_preamble(task)
    _stage_log(task_id, f"build preamble done: {preamble_result.char_count} chars")
    if skip_advisor:
        injected_preamble = preamble_result.text
    else:
        injected_preamble = (
            "# MATH FORMULA HINT (= expert calculation guide)\n"
            f"  {formula}\n\n"
            "Follow this aggregation/division/filter structure EXACTLY.\n\n"
        ) + preamble_result.text

    tools = create_default_tool_registry(
        auditor_model=model,
        question_provider=lambda: task.question,
        context_dir=task.context_dir,
    )
    agent = PhasedReActAgent(
        model=model,
        tools=tools,
        config=PhasedAgentConfig(max_steps=MAX_STEPS, min_explore_queries=3),
        preamble=injected_preamble,
    )
    trace_root = os.environ.get("EXP145_TRACE_ROOT")
    if trace_root:
        trace_dir = Path(trace_root) / task_id
        trace_dir.mkdir(parents=True, exist_ok=True)
        agent.trace_log_path = str(trace_dir / "attempt_0.steps.log")
    _stage_log(task_id, "agent run start")
    run_result = agent.run(task)
    _stage_log(task_id, "agent run done")
    payload = run_result.to_dict()
    payload["preamble_metadata"] = preamble_result.metadata()
    payload["formula"] = formula
    payload["advisor_used"] = not skip_advisor
    payload["phase2_profile"] = {
        "experiment": EXPERIMENT_NAME,
        "source_run": "exp_144_answer_shape_prose_pdf_preprocess_source_router_008",
        "fixed_flags": dict(_FIXED_PHASE2_FLAGS),
        "prefix_cache": prefix_cache_enabled(),
        "pdf_text_cache_root": os.environ.get("EXP144_PDF_TEXT_CACHE_ROOT"),
        "pdf_preprocess_error": pdf_cache_error,
        "pdf_preprocess": [
            {
                "source_rel": r.source_rel,
                "text_path": str(r.text_path),
                "meta_path": str(r.meta_path),
                "page_count": r.page_count,
                "text_chars": r.text_chars,
                "source_sha1": r.source_sha1,
            }
            for r in pdf_cache_results
        ],
    }
    return payload


def _run_single_task_in_subprocess(
    task_id: str, config: AppConfig, queue: multiprocessing.Queue[Any]
) -> None:
    try:
        queue.put({"ok": True, "run_result": _run_single_task_core(task_id=task_id, config=config)})
    except BaseException as exc:  # noqa: BLE001
        queue.put({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


def _run_single_task_with_timeout(*, task_id: str, config: AppConfig) -> dict[str, Any]:
    timeout_seconds = config.run.task_timeout_seconds
    if timeout_seconds <= 0:
        return _run_single_task_core(task_id=task_id, config=config)

    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_run_single_task_in_subprocess, args=(task_id, config, q))
    p.start()
    try:
        result = q.get(timeout=timeout_seconds)
    except Exception:
        result = None
    finally:
        if p.is_alive():
            p.terminate()
            p.join(timeout=1.0)
            if p.is_alive():
                p.kill()
        p.join(timeout=2.0)

    if result is None:
        return _failure_run_result_payload(task_id, f"attempt timed out after {timeout_seconds}s")
    if not result.get("ok"):
        return _failure_run_result_payload(task_id, str(result.get("error", "unknown error")))
    return result.get("run_result") or _failure_run_result_payload(task_id, "missing run_result")


def _write_task_outputs(
    task_id: str, run_output_dir: Path, run_result: dict[str, Any]
) -> TaskRunArtifacts:
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


def _create_run_output_dir(
    output_root: Path, run_id: str | None, *, allow_existing: bool = False
) -> tuple[str, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    if run_id:
        effective_run_id = run_id
    else:
        i = 1
        while (output_root / f"{EXPERIMENT_NAME}_{i:03d}").exists():
            i += 1
        effective_run_id = f"{EXPERIMENT_NAME}_{i:03d}"
    run_output_dir = output_root / effective_run_id
    run_output_dir.mkdir(parents=True, exist_ok=allow_existing)
    return effective_run_id, run_output_dir


def run_single_task(
    *,
    task_id: str,
    config: AppConfig,
    run_output_dir: Path,
) -> TaskRunArtifacts:
    started_at = perf_counter()
    os.environ["EXP145_TRACE_ROOT"] = str(run_output_dir)
    run_result = _run_single_task_with_timeout(task_id=task_id, config=config)
    run_result["e2e_elapsed_seconds"] = round(perf_counter() - started_at, 3)
    return _write_task_outputs(task_id, run_output_dir, run_result)


def run_benchmark(
    *,
    config: AppConfig,
    limit: int | None = None,
    progress_callback: Callable[[TaskRunArtifacts], None] | None = None,
    exclude_task_ids: frozenset[str] | None = None,
    include_task_ids: frozenset[str] | None = None,
    allow_existing_run_dir: bool = False,
) -> tuple[Path, list[TaskRunArtifacts]]:
    effective_run_id, run_output_dir = _create_run_output_dir(
        config.run.output_dir,
        config.run.run_id,
        allow_existing=allow_existing_run_dir,
    )
    dataset = DABenchPublicDataset(config.dataset.root_path)
    tasks = list(dataset.iter_tasks())
    if include_task_ids:
        tasks = [t for t in tasks if t.task_id in include_task_ids]
    if exclude_task_ids:
        tasks = [t for t in tasks if t.task_id not in exclude_task_ids]
    if limit is not None:
        tasks = tasks[:limit]

    task_ids = [task.task_id for task in tasks]
    workers = max(1, config.run.max_workers)
    artifacts: list[TaskRunArtifacts] = []
    if workers == 1:
        for task_id in task_ids:
            artifact = run_single_task(
                task_id=task_id,
                config=config,
                run_output_dir=run_output_dir,
            )
            artifacts.append(artifact)
            if progress_callback:
                progress_callback(artifact)
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    run_single_task,
                    task_id=task_id,
                    config=config,
                    run_output_dir=run_output_dir,
                ): task_id
                for task_id in task_ids
            }
            for fut in as_completed(futures):
                artifact = fut.result()
                artifacts.append(artifact)
                if progress_callback:
                    progress_callback(artifact)

    _write_json(
        run_output_dir / "summary.json",
        {
            "run_id": effective_run_id,
            "experiment": EXPERIMENT_NAME,
            "task_count": len(artifacts),
            "succeeded_task_count": sum(1 for a in artifacts if a.succeeded),
            "max_workers": workers,
            "profile": dict(_FIXED_PHASE2_FLAGS),
            "tasks": [artifact.to_dict() for artifact in artifacts],
        },
    )
    return run_output_dir, artifacts
