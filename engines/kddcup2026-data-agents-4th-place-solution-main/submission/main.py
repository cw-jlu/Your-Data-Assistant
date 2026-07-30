"""Container entrypoint for KDD Cup 2026 DataAgent-Bench submission.

The evaluation system mounts:
  /input  (read-only)  — contains task_<id>/{task.json, context/}
  /output (read-write) — write predictions to task_<id>/prediction.csv
  /logs   (read-write) — write per-run logs

It provides:
  MODEL_API_URL, MODEL_API_KEY, MODEL_NAME — only allowed network target
  (External internet is fully blocked.)

This entrypoint runs the experiment selected by EXPERIMENT_NAME (default
`exp_154_v1_audio_asr`) on every task, writing predictions directly
into /output. We pre-write stub predictions so that a crash mid-run cannot
zero out the rest of the tasks.

Usage (local smoke):

  EXPERIMENT_NAME=exp_154_v1_audio_asr \\
  MODEL_API_URL=https://… MODEL_API_KEY=… MODEL_NAME=qwen3.5-35b-a3b \\
  python -m submission.main
"""
from __future__ import annotations

import csv
import importlib
import json
import os
import re
import signal
import sys
import threading
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from pathlib import Path

INPUT_ROOT = Path(os.environ.get("KDD_INPUT", "/input"))
OUTPUT_ROOT = Path(os.environ.get("KDD_OUTPUT", "/output"))
LOG_ROOT = Path(os.environ.get("KDD_LOGS", "/logs"))
EXPERIMENT_NAME = os.environ.get("EXPERIMENT_NAME", "exp_154_v1_audio_asr")
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "8"))
TASK_TIMEOUT_SECONDS = int(os.environ.get("TASK_TIMEOUT_SECONDS", "6000"))
DEFAULT_MODEL_NAME = os.environ.get("MODEL_NAME", "qwen3.5-35b-a3b")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


SUBMISSION_BOARD = os.environ.get("SUBMISSION_BOARD", "B").strip().upper()
_BOARD_IS_B = SUBMISSION_BOARD.startswith("B")
SUBMISSION_WALL_BUDGET_SECONDS = _env_int(
    "SUBMISSION_WALL_BUDGET_SECONDS", 43200 if _BOARD_IS_B else 7200
)


def _resolve_board(task_count: int) -> tuple[str, int, int, int, int]:
    """Resolve (board, wall_budget, retry_max_tasks, retry_max_timeout, retry_safety).

    SUBMISSION_BOARD=AUTO detects the board from the mounted task count — the one
    signal the container reliably has: the A-board subset is ~60 tasks (2h wall),
    the B-board subset is ~324 (12h wall). A fixed SUBMISSION_BOARD=B image env
    made the A-board run assume a 12h budget, so the empty-prediction retry was
    planned past the real 2h kill. Explicit A/B env values and explicit
    SUBMISSION_* knob overrides still win over AUTO.
    """
    board = SUBMISSION_BOARD
    if board.startswith("AUTO"):
        board = "B" if task_count >= 100 else "A"
    is_b = board.startswith("B")
    wall = _env_int("SUBMISSION_WALL_BUDGET_SECONDS", 43200 if is_b else 7200)
    max_tasks = _env_int("SUBMISSION_RETRY_MAX_TASKS", 12 if is_b else 6)
    max_timeout = _env_int(
        "SUBMISSION_RETRY_MAX_TIMEOUT_SECONDS", 3600 if is_b else 1800
    )
    safety = _env_int("SUBMISSION_RETRY_SAFETY_SECONDS", 600 if is_b else 300)
    return board, wall, max_tasks, max_timeout, safety
SUBMISSION_PIPELINED_PREPROCESS = _env_flag("SUBMISSION_PIPELINED_PREPROCESS", False)
SUBMISSION_PREPROCESS_WORKERS = _env_int("SUBMISSION_PREPROCESS_WORKERS", 3)
SUBMISSION_ASR_WORKERS = _env_int("SUBMISSION_ASR_WORKERS", 3)
SUBMISSION_RETRY_EMPTY = _env_flag("SUBMISSION_RETRY_EMPTY", False)
SUBMISSION_RETRY_WORKERS = _env_int("SUBMISSION_RETRY_WORKERS", 2)
SUBMISSION_RETRY_MAX_TASKS = _env_int("SUBMISSION_RETRY_MAX_TASKS", 12 if _BOARD_IS_B else 6)
SUBMISSION_RETRY_MIN_SECONDS = _env_int("SUBMISSION_RETRY_MIN_SECONDS", 300)
SUBMISSION_RETRY_MAX_TIMEOUT_SECONDS = _env_int(
    "SUBMISSION_RETRY_MAX_TIMEOUT_SECONDS", 3600 if _BOARD_IS_B else 1800
)
SUBMISSION_RETRY_SAFETY_SECONDS = _env_int(
    "SUBMISSION_RETRY_SAFETY_SECONDS", 600 if _BOARD_IS_B else 300
)

STUB_HEADER = ["answer"]
_PREPROCESS_LOCAL = threading.local()


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _stub_path(task_id: str) -> Path:
    return OUTPUT_ROOT / task_id / "prediction.csv"


def _write_stub(task_id: str) -> None:
    path = _stub_path(task_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(STUB_HEADER)


def _prediction_is_empty(task_id: str) -> bool:
    path = _stub_path(task_id)
    if not path.exists():
        return True
    try:
        if path.stat().st_size <= 0:
            return True
        with path.open(newline="") as handle:
            rows = list(csv.reader(handle))
    except Exception:
        return True
    return len(rows) <= 1


import re as _re_norm

# A cell composed ONLY of digits + thousands-commas (+ optional decimal), with
# at least one comma group. Nothing else is touched: text, %, units (元/股/…),
# and 亿/万 scale-multipliers are all left exactly as-is, so this can only fix a
# thousands-separated number and can never alter a gold-matching value.
# (A BULL column stores "0.5%" as a string, so stripping %/units would regress;
#  no demo or BULL gold stores comma-grouped numbers, so comma-strip is safe.)
_NUM_COMMA_RE = _re_norm.compile(r"^(-?\d{1,3}(?:,\d{3})+)(\.\d+)?$")


def _normalize_numeric_cell(value):
    """Strip thousands-commas from cells made up only of digits+commas so a
    correct-but-comma-formatted number matches gold's bare-number form. The
    organizer scorer sees the normalized output, so this transfers to the
    A-board (unlike a scorer-side change)."""
    t = str(value).strip()
    m = _NUM_COMMA_RE.match(t)
    if not m:
        return value
    return m.group(1).replace(",", "") + (m.group(2) or "")


def _write_prediction(task_id: str, columns: list[str], rows: list[list]) -> None:
    path = _stub_path(task_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns or STUB_HEADER)
        for row in rows:
            writer.writerow([_normalize_numeric_cell(c) for c in row])
    tmp_path.replace(path)


def _import_experiment(name: str):
    runner_mod = importlib.import_module(f"experiments.{name}.runner")
    config_mod = importlib.import_module(f"experiments.{name}.config")
    return runner_mod, config_mod


def _experiment_env_prefix(experiment_name: str) -> str | None:
    match = re.match(r"^exp_(\d+)_", experiment_name)
    if not match:
        return None
    return f"EXP{int(match.group(1))}"


def _experiment_env_value(
    suffix: str,
    *,
    experiment_name: str = EXPERIMENT_NAME,
    default: str | None = None,
) -> str | None:
    prefix = _experiment_env_prefix(experiment_name)
    names: list[str] = []
    if prefix:
        names.append(f"{prefix}_{suffix}")
    # v3 used EXP153_* ASR knobs. Keep them as compatibility fallbacks so old
    # local commands still work, but prefer the active experiment prefix.
    if "EXP153" not in {prefix}:
        names.append(f"EXP153_{suffix}")
    for name in names:
        value = os.environ.get(name)
        if value is not None and value.strip():
            return value
    return default


def _experiment_env_flag(suffix: str, default: bool, *, experiment_name: str) -> bool:
    raw = _experiment_env_value(suffix, experiment_name=experiment_name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _experiment_env_int(suffix: str, default: int, *, experiment_name: str) -> int:
    raw = _experiment_env_value(suffix, experiment_name=experiment_name)
    if raw is None:
        return default
    return int(raw)


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    tmp_path.replace(path)


def _configure_preprocess_roots(experiment_name: str) -> Path | None:
    prefix = _experiment_env_prefix(experiment_name)
    if not prefix:
        return None
    root = Path(
        os.environ.get(
            f"{prefix}_PREPROCESS_ROOT",
            f"/tmp/kobushi_{prefix.lower()}_preprocess/submission",
        )
    )
    os.environ.setdefault(f"{prefix}_PREPROCESS_ROOT", str(root))
    os.environ.setdefault(f"{prefix}_USE_PRECOMPUTED_PREPROCESS", "1")
    os.environ.setdefault(
        f"{prefix}_KEYFRAME_ROOT",
        f"/tmp/kobushi_{prefix.lower()}_keyframe_cache/submission",
    )
    os.environ.setdefault(
        f"{prefix}_AUDIO_ASR_CACHE_ROOT",
        f"/tmp/kobushi_{prefix.lower()}_audio_asr_cache/submission",
    )
    return root


def _build_app_config(
    config_mod,
    task_root: Path,
    *,
    max_workers: int = MAX_WORKERS,
    task_timeout_seconds: int = TASK_TIMEOUT_SECONDS,
):
    """Construct an AppConfig pointing at the live /input + a temp run dir.

    We build the dataclass directly rather than going through YAML so the
    container does not depend on a writable config file. Credentials come
    from the eval system's MODEL_API_URL / MODEL_API_KEY env vars (or our
    local AGENT_API_BASE / AGENT_API_KEY in test mode).
    """
    AppConfig = config_mod.AppConfig
    DatasetConfig = config_mod.DatasetConfig
    AgentConfig = config_mod.AgentConfig
    RunConfig = config_mod.RunConfig

    api_base = os.environ.get("MODEL_API_URL") or os.environ.get("AGENT_API_BASE", "")
    api_key = os.environ.get("MODEL_API_KEY") or os.environ.get("AGENT_API_KEY", "")

    extra_headers: dict[str, str] = {}
    cf_id = os.environ.get("CF_ACCESS_CLIENT_ID", "")
    cf_secret = os.environ.get("CF_ACCESS_CLIENT_SECRET", "")
    if cf_id:
        extra_headers["CF-Access-Client-Id"] = cf_id
    if cf_secret:
        extra_headers["CF-Access-Client-Secret"] = cf_secret

    return AppConfig(
        dataset=DatasetConfig(root_path=task_root),
        agent=AgentConfig(
            model=DEFAULT_MODEL_NAME,
            api_base=api_base,
            api_key=api_key,
            extra_headers=extra_headers,
            max_steps=int(os.environ.get("MAX_STEPS", "64")),
            temperature=float(os.environ.get("TEMPERATURE", "0.6")),
        ),
        run=RunConfig(
            output_dir=Path("/tmp/kobushi_run"),
            run_id="submission",
            max_workers=max_workers,
            task_timeout_seconds=task_timeout_seconds,
        ),
    )


def _run_one_task(task_id: str, app_config, runner_mod) -> tuple[str, bool, str | None]:
    try:
        # Use the experiment's per-task entry directly so we can write to /output
        # without going through the runner's <run_id>/<task_id>/ layout.
        result = runner_mod._run_single_task_with_timeout(
            task_id=task_id, config=app_config
        )
        answer = result.get("answer")
        if isinstance(answer, dict):
            columns = list(answer.get("columns") or [])
            rows = [list(r) for r in (answer.get("rows") or [])]
            _write_prediction(task_id, columns, rows)
            return (task_id, True, None)
        return (task_id, False, str(result.get("failure_reason") or "no answer"))
    except BaseException as exc:  # noqa: BLE001
        return (task_id, False, f"{type(exc).__name__}: {exc}")


def _run_task_batch(
    *,
    task_ids: list[str],
    app_config,
    runner_mod,
    workers: int,
    label: str,
) -> dict[str, tuple[bool, str | None]]:
    results: dict[str, tuple[bool, str | None]] = {}
    if not task_ids:
        return results
    effective_workers = max(1, min(workers, len(task_ids)))
    _log(f"{label}: tasks={len(task_ids)} workers={effective_workers}")
    with ThreadPoolExecutor(max_workers=effective_workers) as pool:
        futures = {
            pool.submit(_run_one_task, task_id, app_config, runner_mod): task_id
            for task_id in task_ids
        }
        for fut in as_completed(futures):
            task_id, ok, reason = fut.result()
            results[task_id] = (ok, reason)
            if ok:
                _log(f"  {task_id}: OK")
            else:
                _log(f"  {task_id}: FAIL — {reason}")
    return results


def _empty_prediction_tasks(task_ids: list[str]) -> list[str]:
    return [task_id for task_id in task_ids if _prediction_is_empty(task_id)]


def _trace_env_name(experiment_name: str) -> str | None:
    match = re.match(r"^exp_(\d+)_", experiment_name)
    if not match:
        return None
    return f"EXP{int(match.group(1))}_TRACE_ROOT"


def _video_task_ids(
    *,
    task_ids: list[str],
    app_config,
    experiment_name: str,
) -> list[str]:
    try:
        from kobushi_core.benchmark.dataset import DABenchPublicDataset
        audio_mod = importlib.import_module(f"experiments.{experiment_name}.audio_asr")
        find_video = audio_mod.find_video
    except Exception as exc:
        _log(f"video preprocess disabled: no audio_asr module for {experiment_name} — {exc!r}")
        return []

    dataset = DABenchPublicDataset(app_config.dataset.root_path)
    out: list[str] = []
    for task_id in task_ids:
        try:
            task = dataset.get_task(task_id)
            if find_video(task.context_dir) is not None:
                out.append(task_id)
        except Exception as exc:
            _log(f"  {task_id}: video detect failed — {exc!r}")
    return out


def _get_thread_whisper_model():
    model = getattr(_PREPROCESS_LOCAL, "whisper_model", None)
    if model is not None:
        return model
    from faster_whisper import WhisperModel

    model_name = (
        _experiment_env_value("ASR_MODEL_PATH")
        or _experiment_env_value("ASR_MODEL")
        or "small"
    )
    compute_type = _experiment_env_value("ASR_COMPUTE_TYPE", default="int8")
    cpu_threads = _experiment_env_int("ASR_CPU_THREADS", 2, experiment_name=EXPERIMENT_NAME)
    _log(
        f"loading ASR model for preprocess thread: model={model_name} "
        f"compute={compute_type} cpu_threads={cpu_threads}"
    )
    model = WhisperModel(
        model_name,
        device="cpu",
        compute_type=compute_type,
        cpu_threads=cpu_threads,
    )
    _PREPROCESS_LOCAL.whisper_model = model
    return model


def _preprocess_one_video_task(
    *,
    task_id: str,
    app_config,
    runner_mod,
    experiment_name: str,
    asr_semaphore: threading.Semaphore,
) -> dict:
    preprocess_root = _configure_preprocess_roots(experiment_name)
    if preprocess_root is None:
        return {"task_id": task_id, "status": "skipped", "reason": "no experiment prefix"}

    task_dir = preprocess_root / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    done_path = task_dir / "preprocess_done.json"
    payload = {
        "task_id": task_id,
        "status": "started",
        "started_at": time.time(),
        "experiment": experiment_name,
    }
    try:
        from kobushi_core.benchmark.dataset import DABenchPublicDataset

        flags_mod = importlib.import_module(f"experiments.{experiment_name}.flags")
        dataset = DABenchPublicDataset(app_config.dataset.root_path)
        task = dataset.get_task(task_id)

        keyframe_status = "disabled"
        if getattr(flags_mod, "video_keyframe_note_on", lambda: False)():
            try:
                vk_mod = importlib.import_module(
                    f"experiments.{experiment_name}.video_keyframe_note"
                )
                note_config = runner_mod._config_with_temperature(app_config, 0.2)
                note_model = runner_mod.build_model_adapter(note_config, task_id=task_id)
                note = vk_mod.extract_keyframe_note(task, note_model)
                if note is not None:
                    (task_dir / "video_keyframe_note.md").write_text(
                        note.text,
                        encoding="utf-8",
                    )
                    _write_json_atomic(
                        task_dir / "video_keyframe_note_meta.json",
                        {
                            "prompt_lang": note.prompt_lang,
                            "n_keyframes": note.n_keyframes,
                            "frames": note.frames,
                            "keyframes": note.meta,
                        },
                    )
                    keyframe_status = "ok"
                else:
                    keyframe_status = "none"
            except Exception as exc:
                keyframe_status = "error"
                (task_dir / "video_keyframe_note_error.txt").write_text(
                    repr(exc),
                    encoding="utf-8",
                )

        asr_status = "disabled"
        if getattr(flags_mod, "audio_asr_on", lambda: False)():
            try:
                prepare_task_audio = importlib.import_module(
                    f"experiments.{experiment_name}.audio_asr"
                ).prepare_task_audio

                language = _experiment_env_value(
                    "ASR_LANGUAGE",
                    experiment_name=experiment_name,
                    default="auto",
                )
                quality_filter = _experiment_env_flag(
                    "ASR_QUALITY_FILTER",
                    True,
                    experiment_name=experiment_name,
                )
                reuse_cache = _experiment_env_flag(
                    "REUSE_ASR_CACHE",
                    True,
                    experiment_name=experiment_name,
                )
                with asr_semaphore:
                    asr_text, asr_meta = prepare_task_audio(
                        task_id=task_id,
                        task=task,
                        task_dir=task_dir,
                        whisper_model=_get_thread_whisper_model(),
                        whisper_lock=None,
                        language=language,
                        quality_filter=quality_filter,
                        existing_asr=None,
                        existing_asr_path=None,
                        reuse_cache=reuse_cache,
                    )
                asr_status = str(asr_meta.get("status") or "ok")
                payload["asr_included"] = bool(asr_text)
                payload["asr_meta"] = asr_meta
            except Exception as exc:
                asr_status = "error"
                (task_dir / "asr_error.txt").write_text(repr(exc), encoding="utf-8")

        payload.update(
            {
                "status": "done",
                "finished_at": time.time(),
                "keyframe_note_status": keyframe_status,
                "asr_status": asr_status,
            }
        )
    except Exception as exc:
        payload.update(
            {
                "status": "error",
                "finished_at": time.time(),
                "error": repr(exc),
            }
        )
    _write_json_atomic(done_path, payload)
    return payload


def _run_ordered_preprocessed_tasks(
    *,
    task_ids: list[str],
    app_config,
    runner_mod,
    workers: int,
    label: str,
    experiment_name: str,
) -> dict[str, tuple[bool, str | None]]:
    video_ids = set(
        _video_task_ids(
            task_ids=task_ids,
            app_config=app_config,
            experiment_name=experiment_name,
        )
    )
    if not video_ids:
        return _run_task_batch(
            task_ids=task_ids,
            app_config=app_config,
            runner_mod=runner_mod,
            workers=workers,
            label=label,
        )

    preprocess_root = _configure_preprocess_roots(experiment_name)
    effective_workers = max(1, min(workers, len(task_ids)))
    preprocess_workers = max(1, min(SUBMISSION_PREPROCESS_WORKERS, len(video_ids)))
    asr_workers = max(1, SUBMISSION_ASR_WORKERS)
    asr_semaphore = threading.Semaphore(asr_workers)
    _log(
        f"{label}: ordered pipelined preprocess enabled; tasks={len(task_ids)} "
        f"workers={effective_workers} video_tasks={len(video_ids)} "
        f"preprocess_workers={preprocess_workers} asr_workers={asr_workers} "
        f"preprocess_root={preprocess_root}"
    )

    results: dict[str, tuple[bool, str | None]] = {}
    in_flight = {}
    next_idx = 0
    last_block_log = 0.0
    with ThreadPoolExecutor(max_workers=preprocess_workers) as preprocess_pool, ThreadPoolExecutor(
        max_workers=effective_workers
    ) as task_pool:
        preprocess_futures = {
            task_id: preprocess_pool.submit(
                _preprocess_one_video_task,
                task_id=task_id,
                app_config=app_config,
                runner_mod=runner_mod,
                experiment_name=experiment_name,
                asr_semaphore=asr_semaphore,
            )
            for task_id in task_ids
            if task_id in video_ids
        }

        while next_idx < len(task_ids) or in_flight:
            while len(in_flight) < effective_workers and next_idx < len(task_ids):
                task_id = task_ids[next_idx]
                preprocess_future = preprocess_futures.get(task_id)
                if preprocess_future is not None:
                    if not preprocess_future.done():
                        now = time.time()
                        if now - last_block_log >= 30:
                            _log(
                                f"  waiting for preprocess before enqueuing {task_id} "
                                f"(ordered gate)"
                            )
                            last_block_log = now
                        break
                    try:
                        pre = preprocess_future.result()
                        _log(
                            f"  {task_id}: preprocess {pre.get('status')} "
                            f"keyframe={pre.get('keyframe_note_status')} "
                            f"asr={pre.get('asr_status')}"
                        )
                    except BaseException as exc:  # noqa: BLE001
                        _log(f"  {task_id}: preprocess future failed — {exc!r}")

                fut = task_pool.submit(_run_one_task, task_id, app_config, runner_mod)
                in_flight[fut] = task_id
                next_idx += 1

            if in_flight:
                done, _ = wait(in_flight.keys(), timeout=1.0, return_when=FIRST_COMPLETED)
                for fut in done:
                    task_id, ok, reason = fut.result()
                    in_flight.pop(fut, None)
                    results[task_id] = (ok, reason)
                    if ok:
                        _log(f"  {task_id}: OK")
                    else:
                        _log(f"  {task_id}: FAIL — {reason}")
            elif next_idx < len(task_ids):
                time.sleep(0.5)
    return results


def main() -> int:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    trace_env_name = _trace_env_name(EXPERIMENT_NAME)
    if trace_env_name:
        os.environ.setdefault(trace_env_name, str(LOG_ROOT / "traces"))

    if not INPUT_ROOT.is_dir():
        _log(f"FATAL: input root {INPUT_ROOT} does not exist")
        return 1

    task_dirs = sorted(p for p in INPUT_ROOT.glob("task_*") if p.is_dir())
    task_ids = [td.name for td in task_dirs]
    board, wall_budget_seconds, retry_max_tasks, retry_max_timeout_seconds, retry_safety_seconds = (
        _resolve_board(len(task_ids))
    )
    _log(
        f"experiment={EXPERIMENT_NAME} tasks={len(task_dirs)} workers={MAX_WORKERS} "
        f"task_timeout={TASK_TIMEOUT_SECONDS}s board={board} "
        f"wall_budget={wall_budget_seconds}s"
    )
    if not task_dirs:
        _log("WARN: no task_* directories under /input")
        return 0

    # Pre-write stubs so a crash mid-run doesn't zero out remaining tasks.
    for task_id in task_ids:
        _write_stub(task_id)

    runner_mod, config_mod = _import_experiment(EXPERIMENT_NAME)
    app_config = _build_app_config(config_mod, INPUT_ROOT)

    started = time.time()
    if SUBMISSION_PIPELINED_PREPROCESS:
        initial_results = _run_ordered_preprocessed_tasks(
            task_ids=task_ids,
            app_config=app_config,
            runner_mod=runner_mod,
            workers=MAX_WORKERS,
            label="initial run",
            experiment_name=EXPERIMENT_NAME,
        )
    else:
        initial_results = _run_task_batch(
            task_ids=task_ids,
            app_config=app_config,
            runner_mod=runner_mod,
            workers=MAX_WORKERS,
            label="initial run",
        )
    initial_elapsed = time.time() - started
    empty_after_initial = _empty_prediction_tasks(task_ids)

    retry_results: dict[str, tuple[bool, str | None]] = {}
    retry_targets = list(empty_after_initial)
    retry_timeout_seconds = 0
    retry_skipped_reason = None
    remaining_before_retry = (
        wall_budget_seconds - initial_elapsed
        if wall_budget_seconds > 0
        else retry_max_timeout_seconds + retry_safety_seconds
    )
    retry_budget_seconds = max(0, remaining_before_retry - retry_safety_seconds)

    if not SUBMISSION_RETRY_EMPTY:
        retry_skipped_reason = "disabled"
    elif not retry_targets:
        retry_skipped_reason = "no empty predictions"
    elif retry_budget_seconds < SUBMISSION_RETRY_MIN_SECONDS:
        retry_skipped_reason = (
            f"insufficient remaining budget ({retry_budget_seconds:.1f}s "
            f"< {SUBMISSION_RETRY_MIN_SECONDS}s)"
        )
    else:
        if retry_max_tasks > 0 and len(retry_targets) > retry_max_tasks:
            retry_targets = retry_targets[:retry_max_tasks]
        # The batch runs in ceil(targets/workers) sequential waves; each wave can
        # take up to the per-task timeout. Divide the budget by the wave count so
        # the WHOLE retry pass fits the remaining wall clock — a flat
        # timeout=budget would overrun it by up to the wave count.
        retry_waves = max(
            1, -(-len(retry_targets) // max(1, SUBMISSION_RETRY_WORKERS))
        )
        retry_timeout_seconds = int(
            max(
                SUBMISSION_RETRY_MIN_SECONDS,
                min(retry_max_timeout_seconds, retry_budget_seconds / retry_waves),
            )
        )
        _log(
            f"retry empty predictions: targets={retry_targets} "
            f"timeout={retry_timeout_seconds}s remaining={remaining_before_retry:.1f}s"
        )
        retry_config = _build_app_config(
            config_mod,
            INPUT_ROOT,
            max_workers=SUBMISSION_RETRY_WORKERS,
            task_timeout_seconds=retry_timeout_seconds,
        )
        retry_results = _run_task_batch(
            task_ids=retry_targets,
            app_config=retry_config,
            runner_mod=runner_mod,
            workers=SUBMISSION_RETRY_WORKERS,
            label="retry run",
        )

    empty_after_retry = _empty_prediction_tasks(task_ids)
    final_succeeded = len(task_ids) - len(empty_after_retry)
    final_failed = len(empty_after_retry)

    elapsed = time.time() - started
    _log(
        f"done: {final_succeeded}/{len(task_dirs)} non-empty predictions, "
        f"{final_failed} empty, elapsed={elapsed:.1f}s"
    )
    # Write a small manifest for debugging.
    manifest = LOG_ROOT / "submission_manifest.json"
    env_prefix = _experiment_env_prefix(EXPERIMENT_NAME)
    manifest.write_text(
        json.dumps(
            {
                "experiment": EXPERIMENT_NAME,
                "task_count": len(task_dirs),
                "succeeded": final_succeeded,
                "failed": final_failed,
                "elapsed_seconds": round(elapsed, 1),
                "pipelined_preprocess": SUBMISSION_PIPELINED_PREPROCESS,
                "preprocess_workers": SUBMISSION_PREPROCESS_WORKERS,
                "asr_workers": SUBMISSION_ASR_WORKERS,
                "preprocess_root": (
                    os.environ.get(f"{env_prefix}_PREPROCESS_ROOT", "")
                    if env_prefix
                    else ""
                ),
                "initial_succeeded": sum(1 for ok, _ in initial_results.values() if ok),
                "initial_failed": sum(1 for ok, _ in initial_results.values() if not ok),
                "initial_failed_tasks": [
                    task_id for task_id, (ok, _) in initial_results.items() if not ok
                ],
                "empty_predictions_after_initial": empty_after_initial,
                "board": board,
                "wall_budget_seconds": wall_budget_seconds,
                "retry_enabled": SUBMISSION_RETRY_EMPTY,
                "retry_targets": retry_targets,
                "retry_timeout_seconds": retry_timeout_seconds,
                "retry_workers": SUBMISSION_RETRY_WORKERS,
                "retry_succeeded": sum(1 for ok, _ in retry_results.values() if ok),
                "retry_failed": sum(1 for ok, _ in retry_results.values() if not ok),
                "retry_failed_tasks": [
                    task_id for task_id, (ok, _) in retry_results.items() if not ok
                ],
                "retry_skipped_reason": retry_skipped_reason,
                "remaining_seconds_before_retry": round(remaining_before_retry, 1),
                "empty_predictions_after_retry": empty_after_retry,
            },
            indent=2,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
