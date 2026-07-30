"""Submission entrypoint per §3.6: iterate /input task_<id>, write /output/<task_id>/prediction.csv.

Each task goes through `agents.runs.runner.run_single_task` — the exact same
path the on-laptop CLI (`dabench run-task` / `run-benchmark`) uses. That path
owns the full pre-agent sequence (ETL extraction → virtual context),
subprocess hard timeout, and prediction.csv writing; routing
the container through it guarantees the judged run behaves like the local
rehearsal instead of silently skipping the preprocessing stages.

Concurrency: tasks run on a `ThreadPoolExecutor`; each `run_single_task` call
spawns a subprocess for hard-timeout isolation (mirrors `run_benchmark`'s
parallel path). Worker count: `MAX_WORKERS` env overrides → falls back to
`config.run.max_workers`.

§3.7 trade-off: only minimal status lines go to /logs/main.log. ReAct traces
contain raw rows from /input CSVs (read_csv / inspect_files outputs); the
official rules treat that as leaked test-set data — best case loss of debug
support, worst case disqualification. For full traces, run the rehearsal
locally via scripts/eval_local.sh; the artifacts/eval_logs from that path
are unrestricted because they live on your machine.

Exit codes: 0 ok; 78 (EX_CONFIG) missing env / mounts; 73 (EX_CANTCREAT)
/output not writable.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import threading
import time
import traceback
from concurrent.futures import (
    ThreadPoolExecutor,
    TimeoutError as FuturesTimeout,
    as_completed,
)
from dataclasses import replace
from pathlib import Path
from typing import Any

from agents.benchmark.dataset import DABenchPublicDataset
from agents.config import AppConfig, RateLimitConfig, load_app_config
from agents.runs.runner import run_single_task

REQUIRED_ENV = ("MODEL_API_URL", "MODEL_API_KEY", "MODEL_NAME")
DEFAULT_INPUT = Path("/input")
DEFAULT_OUTPUT = Path("/output")
DEFAULT_LOGS = Path("/logs")
DEFAULT_CONFIG = Path("/app/configs/react.example.yaml")
# A榜是 2h；B榜前改成 43_200.0（12h）再提交。
DEFAULT_GLOBAL_BUDGET_SECONDS = 43_200.0

_LOG_LOCK = threading.Lock()


def _log(logs_dir: Path, line: str) -> None:
    """Append a status line to /logs/main.log and stderr.

    Per §3.7 we only emit metadata here — never tool outputs or raw rows.
    Lock-guarded so concurrent worker threads don't interleave bytes inside
    a single line on the file or stderr.
    """
    logs_dir.mkdir(parents=True, exist_ok=True)
    with _LOG_LOCK:
        with (logs_dir / "main.log").open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        sys.stderr.write(line + "\n")


def _build_config(input_root: Path) -> AppConfig:
    """Load the baked-in YAML and apply §3.4 / §3.5 overrides.

    Reading from `configs/react.example.yaml` (baked into the image
    at /app/configs/) preserves on-laptop tuning —
    `enable_thinking`, `max_tokens`, `max_empty_tool_call_retries`, prompt
    knobs — that the local CLI uses. Override `CONFIG_PATH` to point at a
    different config inside the image.

    Forced overrides:
    - `dataset.root_path` → `/input` (§3.4 mount)
    - `agent.{model, api_base, api_key}` → §3.5 injected env vars
    - `agent.{api_keys}` → single-element tuple with the env key
    - `agent.rate_limit` → config value or RateLimitConfig defaults;
      state_dir resolved to /logs/ratelimit/. Prevents 500 concurrency
      errors from consuming agent steps.
    - `agent.max_steps` ← `MAX_STEPS` env when set (smoke-test escape hatch)
    - `run.blocklist` → (): local debug skip-lists must never reach the
      judged run — a skipped task is a guaranteed zero.
    """
    config_path = Path(os.environ.get("CONFIG_PATH", DEFAULT_CONFIG))
    # Fail loud rather than silently fall back to `AppConfig()` defaults: a
    # missing config means the image was built without the COPY (or CONFIG_PATH
    # points at a typo), and the silent path produced a "successful" run with
    # all locally-tuned knobs (max_steps, sampling, prompts) reverted to
    # dataclass defaults — undetectable from the judge's side.
    if not config_path.is_file():
        sys.stderr.write(
            f"FATAL: config not found at {config_path}; "
            "verify Dockerfile COPY configs/ and CONFIG_PATH env\n"
        )
        raise SystemExit(78)
    base = load_app_config(config_path)

    api_key = os.environ["MODEL_API_KEY"]
    rl = base.agent.rate_limit or RateLimitConfig()
    state_dir = Path(os.environ.get("LOGS_ROOT", DEFAULT_LOGS)) / "ratelimit"
    state_dir.mkdir(parents=True, exist_ok=True)
    rl = replace(rl, state_dir=state_dir)

    agent_overrides: dict[str, Any] = {
        "model": os.environ["MODEL_NAME"],
        "api_base": os.environ["MODEL_API_URL"],
        "api_key": api_key,
        "api_keys": (api_key,),
        "rate_limit": rl,
    }
    if max_steps := os.environ.get("MAX_STEPS"):
        agent_overrides["max_steps"] = int(max_steps)

    return replace(
        base,
        dataset=replace(base.dataset, root_path=input_root),
        agent=replace(base.agent, **agent_overrides),
        run=replace(base.run, blocklist=()),
    )


def _run_one_task(config: AppConfig, task_id: str, output_root: Path, logs_root: Path) -> bool:
    """Run a single task through the shared runner; return True if prediction.csv landed.

    `run_single_task` owns ETL → virtual context →
    subprocess hard timeout → prediction.csv, identical to the CLI path.
    """
    task_started = time.monotonic()
    _log(logs_root, f"task={task_id} started")

    artifact = None
    failure_reason: str | None = None
    try:
        artifact = run_single_task(
            task_id=task_id,
            config=config,
            run_output_dir=output_root,
        )
        failure_reason = artifact.failure_reason
    except Exception as exc:
        # Tracebacks to stderr only — they may quote question strings, but
        # not raw row payloads. /logs/main.log records the redacted summary.
        traceback.print_exc(file=sys.stderr)
        failure_reason = f"{type(exc).__name__}: {exc}"

    wrote = artifact is not None and artifact.prediction_csv_path is not None

    rows = 0
    if artifact is not None and artifact.prediction_csv_path is not None:
        with contextlib.suppress(OSError):
            rows = artifact.prediction_csv_path.read_text().count("\n")

    elapsed = time.monotonic() - task_started
    status = "ok" if wrote else f"no_answer ({failure_reason or 'unknown'})"
    rows_info = f" rows={rows}" if wrote else ""
    _log(logs_root, f"task={task_id} status={status}{rows_info} elapsed={elapsed:.1f}s")
    return wrote


def main() -> int:
    input_root = Path(os.environ.get("INPUT_ROOT", DEFAULT_INPUT))
    output_root = Path(os.environ.get("OUTPUT_ROOT", DEFAULT_OUTPUT))
    logs_root = Path(os.environ.get("LOGS_ROOT", DEFAULT_LOGS))

    missing = [k for k in REQUIRED_ENV if not os.environ.get(k)]
    if missing:
        sys.stderr.write(f"FATAL: missing env vars: {missing}\n")
        return 78

    for path in (input_root, output_root, logs_root):
        if not path.is_dir():
            sys.stderr.write(f"FATAL: {path} not mounted as directory\n")
            return 78

    try:
        probe = output_root / ".writable_probe"
        probe.touch()
        probe.unlink()
    except OSError as exc:
        sys.stderr.write(f"FATAL: {output_root} not writable: {exc}\n")
        return 73

    global_budget = float(os.environ.get("GLOBAL_WALL_SECONDS", DEFAULT_GLOBAL_BUDGET_SECONDS))

    config = _build_config(input_root)
    a = config.agent
    _log(
        logs_root,
        f"config model={a.model} api_base={a.api_base} max_steps={a.max_steps} "
        f"max_tokens={a.max_tokens} enable_thinking={a.enable_thinking} "
        f"task_timeout={config.run.task_timeout_seconds}s",
    )
    tasks = DABenchPublicDataset(config.dataset.root_path).iter_tasks()
    workers = max(1, int(os.environ.get("MAX_WORKERS", config.run.max_workers)))
    _log(
        logs_root,
        f"start input={input_root} output={output_root} budget={global_budget:.0f}s "
        f"tasks={len(tasks)} workers={workers} "
        f"task_timeout={config.run.task_timeout_seconds}s",
    )

    started = time.monotonic()
    completed = 0
    written = 0

    # Submit every task up front; the executor caps concurrency at `workers`
    # and the global wall budget is enforced by `as_completed(timeout=...)`.
    # Each worker thread calls `run_single_task`, which spawns its own
    # subprocess — task state is process-isolated, nothing is shared beyond
    # the immutable config.
    executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="submission")
    try:
        futures = {
            executor.submit(_run_one_task, config, task.task_id, output_root, logs_root): task
            for task in tasks
        }
        try:
            # Re-anchor the budget to "now" so submission overhead doesn't
            # eat into the deadline observed by `as_completed`.
            timeout = max(0.0, global_budget - (time.monotonic() - started))
            for future in as_completed(futures, timeout=timeout):
                wrote = False
                with contextlib.suppress(Exception):
                    wrote = future.result()
                if wrote:
                    written += 1
                completed += 1
                remaining_s = global_budget - (time.monotonic() - started)
                pending = len(futures) - completed
                _log(
                    logs_root,
                    f"progress completed={completed}/{len(futures)} "
                    f"written={written} pending={pending} "
                    f"remaining={remaining_s:.0f}s",
                )
                if pending > 0 and remaining_s < global_budget * 0.2:
                    _log(
                        logs_root,
                        f"WARNING budget<20% remaining={remaining_s:.0f}s pending={pending}",
                    )
        except FuturesTimeout:
            pending = sum(1 for future in futures if not future.done())
            _log(
                logs_root,
                f"budget exhausted; pending={pending} elapsed={time.monotonic() - started:.0f}s",
            )
    finally:
        # `wait=False` so we never block past the wall budget; in-flight
        # workers finish on their own and any prediction.csv they manage to
        # write before process exit still counts on disk.
        executor.shutdown(wait=False, cancel_futures=True)

    total_elapsed = time.monotonic() - started
    _log(
        logs_root,
        f"finished completed={completed}/{len(tasks)} predictions={written} "
        f"elapsed={total_elapsed:.1f}s workers={workers}",
    )
    (logs_root / "summary.json").write_text(
        json.dumps(
            {
                "task_total": len(tasks),
                "task_completed": completed,
                "predictions_written": written,
                "elapsed_seconds": round(total_elapsed, 1),
                "workers": workers,
            },
            indent=2,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
