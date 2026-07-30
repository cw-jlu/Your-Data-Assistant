#!/usr/bin/env python3
"""Local read-only trace viewer for phase 2 experiment runs.

Run:
  uv run python scripts/trace_viewer.py --port 8765

Then open:
  http://127.0.0.1:8765
"""
from __future__ import annotations

import argparse
import csv
import hmac
import importlib
import json
import os
import re
import sys
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse


REPO = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPO / "artifacts" / "runs"
LOGS_ROOT = REPO / "artifacts" / "logs"
SRC_ROOT = REPO / "src"
PHASE2_INPUT_ROOT = REPO / "data" / "phase2_demo" / "demo_samples_phase2" / "input"
PHASE2_GOLD_ROOT = REPO / "data" / "phase2_demo" / "demo_samples_phase2" / "output"
PHASE1_INPUT_ROOT = REPO / "data" / "public" / "input"
PHASE1_GOLD_ROOT = REPO / "data" / "public" / "output"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


TEXT_LIMIT = 1_500_000
CSV_PREVIEW_ROWS = 300


def _json_default(obj: Any) -> str:
    return str(obj)


def _read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_text(path: Path, *, limit: int = TEXT_LIMIT) -> dict[str, Any]:
    if not path.is_file():
        return {"exists": False, "text": "", "truncated": False, "size": 0}
    size = path.stat().st_size
    data = path.read_bytes()[:limit]
    return {
        "exists": True,
        "text": data.decode("utf-8", errors="replace"),
        "truncated": size > limit,
        "size": size,
    }


def _safe_run_dir(run_id: str) -> Path:
    name = unquote(run_id)
    if not name or "/" in name or "\\" in name or name in {".", ".."}:
        raise ValueError("invalid run id")
    path = RUNS_ROOT / name
    if not path.is_dir():
        raise FileNotFoundError(name)
    return path


def _safe_task_dir(run_dir: Path, task_id: str) -> Path:
    name = unquote(task_id)
    if not re.fullmatch(r"task_\d+", name):
        raise ValueError("invalid task id")
    path = run_dir / name
    if not path.is_dir():
        raise FileNotFoundError(name)
    return path


def _task_sort_key(task_id: str) -> tuple[int, str]:
    m = re.search(r"(\d+)$", task_id)
    return (int(m.group(1)) if m else 10**9, task_id)


def _load_task_json(task_id: str) -> dict[str, Any] | None:
    for root in (PHASE2_INPUT_ROOT, PHASE1_INPUT_ROOT):
        path = root / task_id / "task.json"
        if path.is_file():
            data = _read_json(path)
            if isinstance(data, dict):
                return data
    return None


def _gold_path(task_id: str) -> Path | None:
    for root in (PHASE2_GOLD_ROOT, PHASE1_GOLD_ROOT):
        path = root / task_id / "gold.csv"
        if path.is_file():
            return path
    return None


def _csv_preview(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {"exists": False, "columns": [], "rows": [], "row_count_preview": 0, "raw": ""}
    text_info = _read_text(path, limit=TEXT_LIMIT)
    text = text_info["text"]
    rows: list[list[str]] = []
    try:
        reader = csv.reader(text.splitlines())
        for i, row in enumerate(reader):
            if i > CSV_PREVIEW_ROWS:
                break
            rows.append(row)
    except Exception:
        rows = []
    columns = rows[0] if rows else []
    body = rows[1:] if len(rows) > 1 else []
    return {
        "exists": True,
        "columns": columns,
        "rows": body,
        "row_count_preview": len(body),
        "raw": text,
        "truncated": text_info["truncated"],
        "size": text_info["size"],
    }


def _summary_results(summary: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not isinstance(summary, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    rows = summary.get("results")
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            tid = row.get("tid") or row.get("task") or row.get("task_id")
            if isinstance(tid, str):
                out[tid] = row
    return out


_LOG_RUN_RE = re.compile(r"\s+->\s+([A-Za-z0-9_.-]+)\s+===")
_LOG_RUN_RE_UNICODE = re.compile(r"\s+[\u2192]\s+([A-Za-z0-9_.-]+)\s+===")
_LOG_SCORE_RE = re.compile(
    r"^\[\s*(?P<idx>\d+)\s*/\s*(?P<total>\d+)\]\s+\S+\s+"
    r"(?P<task>task_\d+)\s+score=(?P<score>[0-9.]+).*?"
    r"t=(?P<elapsed>[0-9.]+)s\s+mean=(?P<mean>[0-9.]+)",
    re.MULTILINE,
)


def _parse_terminal_logs() -> dict[str, dict[str, Any]]:
    """Map run_id to score lines parsed from wrapper terminal logs."""
    parsed: dict[str, dict[str, Any]] = {}
    log_paths: set[Path] = set()
    for root in (RUNS_ROOT, LOGS_ROOT):
        if not root.is_dir():
            continue
        for pattern in ("*.terminal.log", "*.log"):
            log_paths.update(path for path in root.glob(pattern) if path.is_file())
    for path in sorted(log_paths, key=lambda p: p.stat().st_mtime):
        text = _read_text(path, limit=TEXT_LIMIT)["text"]
        current: str | None = None
        for line in text.splitlines():
            m_run = _LOG_RUN_RE.search(line) or _LOG_RUN_RE_UNICODE.search(line)
            if m_run:
                current = m_run.group(1)
                parsed.setdefault(
                    current,
                    {"log_path": str(path), "mtime": path.stat().st_mtime, "scores": {}},
                )
                parsed[current]["log_path"] = str(path)
                parsed[current]["mtime"] = path.stat().st_mtime
                continue
            if current is None:
                continue
            m = _LOG_SCORE_RE.match(line)
            if not m:
                continue
            tid = m.group("task")
            parsed.setdefault(
                current,
                {"log_path": str(path), "mtime": path.stat().st_mtime, "scores": {}},
            )
            parsed[current]["scores"][tid] = {
                "idx": int(m.group("idx")),
                "total": int(m.group("total")),
                "score": float(m.group("score")),
                "elapsed": float(m.group("elapsed")),
                "mean": float(m.group("mean")),
            }
    return parsed


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _task_score(
    task_dir: Path,
    task_id: str,
    result: dict[str, Any] | None = None,
    log_row: dict[str, Any] | None = None,
) -> float | None:
    """Return the best available task score, including live prediction scoring."""
    score = _as_float((result or {}).get("score"))
    if score is not None:
        return score
    score = _as_float((log_row or {}).get("score"))
    if score is not None:
        return score
    if (task_dir / "prediction.csv").is_file():
        return _live_score(task_dir, task_id)
    return None


def _task_scores_for_run(
    task_dirs: list[Path],
    results: dict[str, dict[str, Any]],
    log_scores: dict[str, Any],
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for task_dir in task_dirs:
        task_id = task_dir.name
        score = _task_score(
            task_dir,
            task_id,
            results.get(task_id),
            log_scores.get(task_id) if isinstance(log_scores.get(task_id), dict) else None,
        )
        if score is not None:
            scores[task_id] = score
    return scores


def _run_summary(run_dir: Path, log_info: dict[str, Any] | None) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    summary = _read_json(summary_path)
    results = _summary_results(summary)
    task_dirs = sorted(
        [p for p in run_dir.iterdir() if p.is_dir() and re.fullmatch(r"task_\d+", p.name)],
        key=lambda p: _task_sort_key(p.name),
    )
    predictions = sum(1 for p in task_dirs if (p / "prediction.csv").is_file())
    traces = sum(1 for p in task_dirs if (p / "trace.json").is_file())
    scores = (log_info or {}).get("scores") or {}
    live_task_scores = _task_scores_for_run(task_dirs, results, scores)
    live_mean = None
    if scores:
        latest = max(scores.values(), key=lambda row: row.get("idx", 0))
        live_mean = latest.get("mean")
    elif live_task_scores:
        live_mean = sum(live_task_scores.values()) / len(live_task_scores)
    mean_score = None
    n_tasks = None
    elapsed_minutes = None
    if isinstance(summary, dict):
        mean_score = summary.get("mean_score")
        n_tasks = summary.get("n_tasks")
        elapsed_minutes = summary.get("elapsed_minutes")
    mtime = run_dir.stat().st_mtime
    if summary_path.is_file():
        mtime = max(mtime, summary_path.stat().st_mtime)
    if log_info and log_info.get("mtime"):
        mtime = max(mtime, float(log_info["mtime"]))
    return {
        "run_id": run_dir.name,
        "path": str(run_dir),
        "summary_exists": summary_path.is_file(),
        "status": "finished" if summary_path.is_file() else ("partial" if task_dirs else "empty"),
        "mean_score": mean_score if mean_score is not None else live_mean,
        "n_tasks": n_tasks if n_tasks is not None else (max((v.get("total", 0) for v in scores.values()), default=None)),
        "completed": len(scores) if scores else (len(live_task_scores) if live_task_scores else predictions),
        "prediction_count": predictions,
        "trace_count": traces,
        "task_dir_count": len(task_dirs),
        "live_score_count": len(live_task_scores),
        "live_n_perfect": sum(1 for score in live_task_scores.values() if score >= 0.999),
        "live_n_zero": sum(1 for score in live_task_scores.values() if score <= 0.001),
        "elapsed_minutes": elapsed_minutes,
        "mtime": mtime,
        "mtime_iso": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime)),
        "tag": summary.get("tag") if isinstance(summary, dict) else None,
        "levers": summary.get("levers") if isinstance(summary, dict) else None,
        "config": summary.get("config") if isinstance(summary, dict) else None,
        "terminal_log": (log_info or {}).get("log_path"),
    }


def list_runs() -> dict[str, Any]:
    logs = _parse_terminal_logs()
    runs = []
    if RUNS_ROOT.is_dir():
        for path in RUNS_ROOT.iterdir():
            if path.is_dir():
                runs.append(_run_summary(path, logs.get(path.name)))
    runs.sort(key=lambda r: (r["mtime"], r["run_id"]), reverse=True)
    return {"runs": runs[:800], "root": str(RUNS_ROOT)}


def get_run(run_id: str) -> dict[str, Any]:
    logs = _parse_terminal_logs()
    run_dir = _safe_run_dir(run_id)
    summary = _read_json(run_dir / "summary.json")
    results = _summary_results(summary)
    log_scores = (logs.get(run_dir.name) or {}).get("scores") or {}
    task_ids = set(results) | set(log_scores)
    for child in run_dir.iterdir():
        if child.is_dir() and re.fullmatch(r"task_\d+", child.name):
            task_ids.add(child.name)
    tasks = []
    for tid in sorted(task_ids, key=_task_sort_key):
        task_dir = run_dir / tid
        result = results.get(tid, {})
        log_row = log_scores.get(tid, {})
        score = _task_score(task_dir, tid, result, log_row)
        elapsed = result.get("elapsed", log_row.get("elapsed"))
        pred = task_dir / "prediction.csv"
        trace = task_dir / "trace.json"
        steps = task_dir / "attempt_0.steps.log"
        question = (_load_task_json(tid) or {}).get("question")
        tasks.append(
            {
                "task_id": tid,
                "score": score,
                "elapsed": elapsed,
                "n_ok": result.get("n_ok"),
                "has_prediction": pred.is_file(),
                "has_trace": trace.is_file(),
                "has_steps_log": steps.is_file(),
                "has_gold": _gold_path(tid) is not None,
                "question": question,
                "advisor_used": _read_text(task_dir / "advisor_used.txt", limit=100)["text"].strip()
                if (task_dir / "advisor_used.txt").is_file()
                else None,
                "anti_agg_label": result.get("anti_agg_label") or _anti_agg_label(task_dir),
                "asr_status": result.get("asr_status") or _asr_status(task_dir),
                "live": _live_progress(task_dir) if score is None else None,
            }
        )
    return {"run": _run_summary(run_dir, logs.get(run_dir.name)), "tasks": tasks}


def _anti_agg_label(task_dir: Path) -> str | None:
    info = _read_text(task_dir / "anti_aggregation.txt", limit=2000)
    if not info["exists"]:
        return None
    for line in info["text"].splitlines():
        if line.startswith("label="):
            return line.split("=", 1)[1].strip()
    return None


def _live_progress(task_dir: Path) -> dict[str, Any] | None:
    """Parse attempt_0.steps.log to report the live step/phase of a running task."""
    p = task_dir / "attempt_0.steps.log"
    if not p.is_file():
        return None
    try:
        text = p.read_text(errors="replace")
    except Exception:
        return None
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None
    step = None
    for m in re.finditer(r"step (\d+)", text):
        step = int(m.group(1))
    phase = None
    for ln in reversed(lines):
        m = re.search(r"phase[ =]([A-Za-z]+)", ln)
        if m:
            phase = m.group(1)
            break
    return {"step": step, "phase": phase, "last": lines[-1].strip()[:200]}


_LIVE_SCORE_CACHE: dict[tuple[str, float], float | None] = {}


def _live_score(task_dir: Path, task_id: str) -> float | None:
    """Compute the official per-task score from prediction.csv vs gold (used while a
    run is in progress, before summary.json exists). Cached by prediction mtime so a
    completed task is scored once, not on every matrix refresh."""
    pred = task_dir / "prediction.csv"
    if not pred.is_file():
        return None
    gold = _gold_path(task_id)
    if gold is None:
        return None
    try:
        key = (str(pred), pred.stat().st_mtime)
    except OSError:
        return None
    if key in _LIVE_SCORE_CACHE:
        return _LIVE_SCORE_CACHE[key]
    val: float | None
    try:
        from kobushi_core.eval.csv_compare import _evaluate_task, EvaluationOptions

        ev = _evaluate_task(
            task_id=task_id,
            prediction_path=pred,
            gold_path=gold,
            options=EvaluationOptions(),
        )
        val = float(ev.official_score_lambda_0_5)
    except Exception:
        val = None
    _LIVE_SCORE_CACHE[key] = val
    return val


def _asr_status(task_dir: Path) -> str | None:
    for name in ("audio_asr_meta.json", "asr_meta.json", "video_audio_asr_meta.json"):
        data = _read_json(task_dir / name)
        if isinstance(data, dict):
            return data.get("status") or data.get("state")
    return None


def _artifact_files(task_dir: Path) -> list[dict[str, Any]]:
    files = []
    for path in sorted(task_dir.iterdir(), key=lambda p: p.name):
        if path.is_file():
            files.append(
                {
                    "name": path.name,
                    "size": path.stat().st_size,
                    "mtime": path.stat().st_mtime,
                    "mtime_iso": time.strftime(
                        "%Y-%m-%d %H:%M:%S", time.localtime(path.stat().st_mtime)
                    ),
                }
            )
    return files


def _load_phase_prompts(run_id: str) -> dict[str, Any]:
    mapping = [
        ("exp_160", "experiments.exp_160_soft_final_sql_guard.prompt"),
        ("exp_159", "experiments.exp_159_finance_domain_prompt.prompt"),
        ("exp_155", "experiments.exp_155_phase_tool_visibility.prompt"),
        ("exp_154", "experiments.exp_154_v1_audio_asr.prompt"),
        ("exp_153", "experiments.exp_153_audio_asr.prompt"),
        ("exp_152", "experiments.exp_152_final_sql_guard.prompt"),
        ("exp_149", "experiments.exp_149_modality.prompt"),
    ]
    module_name = None
    for prefix, candidate in mapping:
        if run_id.startswith(prefix):
            module_name = candidate
            break
    if module_name is None:
        return {"source": None, "exact": False, "prompts": {}}
    try:
        mod = importlib.import_module(module_name)
        build = getattr(mod, "build_phased_system_prompt")
    except Exception as exc:
        return {"source": module_name, "exact": False, "error": str(exc), "prompts": {}}
    prompts = {}
    tool_placeholder = "[tool descriptions were not saved in this run]"
    for phase in ("plan", "explore", "answer", "verify"):
        try:
            prompts[phase] = build(phase, tool_placeholder)
        except Exception as exc:
            prompts[phase] = f"[could not build {phase} prompt: {exc}]"
    return {
        "source": module_name,
        "exact": False,
        "note": "Reconstructed with placeholder tool descriptions. Exact per-task input requires input_snapshot.json in future runs.",
        "prompts": prompts,
    }


def _normalize_trace(trace_data: Any) -> list[dict[str, Any]]:
    attempts = trace_data if isinstance(trace_data, list) else [trace_data]
    normalized = []
    for attempt in attempts:
        if not isinstance(attempt, dict):
            continue
        phase = "plan"
        steps_out = []
        for step in attempt.get("steps", []) or []:
            if not isinstance(step, dict):
                continue
            observation = step.get("observation")
            preview = step.get("observation_content_preview")
            if isinstance(observation, dict):
                step_phase = observation.get("phase") or phase
            else:
                step_phase = step.get("phase") or phase
            out = dict(step)
            out["phase"] = step_phase
            if "observation" not in out and preview is not None:
                out["observation"] = {"preview": preview, "ok": step.get("observation_ok")}
            steps_out.append(out)
            action = step.get("action")
            action_input = step.get("action_input") if isinstance(step.get("action_input"), dict) else {}
            if action == "complete_phase":
                next_phase = action_input.get("next_phase")
                if next_phase in {"plan", "explore", "answer", "verify"}:
                    phase = next_phase
            elif step_phase == "answer" and action == "answer_from_sql":
                phase = "verify"
        item = dict(attempt)
        item["steps"] = steps_out
        normalized.append(item)
    return normalized


def _last_answer_sql(trace_attempts: list[dict[str, Any]]) -> dict[str, Any] | None:
    last = None
    for attempt in trace_attempts:
        for step in attempt.get("steps", []) or []:
            if step.get("action") == "answer_from_sql":
                action_input = step.get("action_input") or {}
                last = {
                    "sql": action_input.get("sql"),
                    "source_route": action_input.get("_source_route"),
                    "phase": step.get("phase"),
                    "step": step.get("i") or step.get("step_index"),
                }
    return last


def get_task_detail(run_id: str, task_id: str) -> dict[str, Any]:
    run_dir = _safe_run_dir(run_id)
    task_dir = _safe_task_dir(run_dir, task_id)
    run_payload = get_run(run_id)
    task_row = next((t for t in run_payload["tasks"] if t["task_id"] == task_id), {})
    task_json = _load_task_json(task_id) or {}
    trace_raw = _read_json(task_dir / "trace.json")
    trace = _normalize_trace(trace_raw) if trace_raw is not None else []
    steps_log = _read_text(task_dir / "attempt_0.steps.log")
    pred_path = task_dir / "prediction.csv"
    gold = _gold_path(task_id)

    input_snapshot = _read_json(task_dir / "input_snapshot.json")
    injected = _read_text(task_dir / "injected_preamble.txt")
    if not injected["exists"]:
        injected = _read_text(task_dir / "preamble.txt")

    text_artifacts = {}
    for name in (
        "formula.txt",
        "anti_aggregation.txt",
        "advisor_used.txt",
        "video_summary.txt",
        "video_keyframe_note.md",
        "audio_asr.txt",
        "asr.txt",
    ):
        p = task_dir / name
        if p.is_file():
            text_artifacts[name] = _read_text(p)
    json_artifacts = {}
    for name in (
        "pdf_preprocess.json",
        "video_keyframe_note_meta.json",
        "audio_asr_meta.json",
        "asr_meta.json",
    ):
        data = _read_json(task_dir / name)
        if data is not None:
            json_artifacts[name] = data

    return {
        "run_id": run_id,
        "task_id": task_id,
        "task": task_row,
        "question": task_json.get("question"),
        "difficulty": task_json.get("difficulty"),
        "paths": {
            "task_dir": str(task_dir),
            "prediction": str(pred_path) if pred_path.is_file() else None,
            "gold": str(gold) if gold else None,
            "trace": str(task_dir / "trace.json") if (task_dir / "trace.json").is_file() else None,
            "steps_log": str(task_dir / "attempt_0.steps.log")
            if (task_dir / "attempt_0.steps.log").is_file()
            else None,
        },
        "prediction": _csv_preview(pred_path),
        "gold": _csv_preview(gold),
        "trace": trace,
        "trace_meta": {
            "exists": trace_raw is not None,
            "attempt_count": len(trace),
            "last_answer_sql": _last_answer_sql(trace),
            "is_truncated_trace": True,
            "note": "Current trace.json stores thought and observation previews. Future trace_full.json can preserve complete messages.",
        },
        "steps_log": steps_log,
        "input": {
            "snapshot": input_snapshot,
            "injected_preamble": injected,
            "phase_prompts": _load_phase_prompts(run_id),
        },
        "artifacts": {
            "files": _artifact_files(task_dir),
            "text": text_artifacts,
            "json": json_artifacts,
        },
    }


def get_file(run_id: str, task_id: str, file_name: str) -> dict[str, Any]:
    run_dir = _safe_run_dir(run_id)
    task_dir = _safe_task_dir(run_dir, task_id)
    if not file_name or "/" in file_name or "\\" in file_name:
        raise ValueError("invalid file name")
    path = task_dir / file_name
    info = _read_text(path)
    return {"name": file_name, "path": str(path), **info}


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Kobushi Trace Viewer</title>
  <style>__CSS__</style>
</head>
<body>
  <div id="app">
    <aside class="sidebar">
      <div class="brand">
        <div class="brand-title">Trace Viewer</div>
        <button id="refreshBtn" class="icon-button" title="Refresh runs">Refresh</button>
      </div>
      <input id="runSearch" class="search" placeholder="Filter runs">
      <div id="runList" class="run-list"></div>
    </aside>
    <main class="main">
      <section id="runHeader" class="run-header"></section>
      <section id="taskPicker" class="task-picker">
        <div class="task-picker-head">
          <div class="task-picker-copy">
            <div id="taskPickerTitle" class="task-picker-title">Tasks</div>
            <div id="taskPickerSub" class="task-picker-sub">Select a run first.</div>
          </div>
          <div class="task-picker-actions">
            <input id="taskSearch" class="search task-search" placeholder="Filter tasks">
            <button id="taskToggleBtn" class="small-button">Tasks</button>
          </div>
        </div>
        <div id="taskList" class="task-list"></div>
      </section>
      <nav class="tabs">
        <button class="tab active" data-tab="overview">Overview</button>
        <button class="tab" data-tab="input">Input</button>
        <button class="tab" data-tab="trace">Trace</button>
        <button class="tab" data-tab="output">Output</button>
        <button class="tab" data-tab="raw">Raw</button>
      </nav>
      <section id="content" class="content"></section>
    </main>
  </div>
  <script>__JS__</script>
</body>
</html>
"""


CSS = r"""
:root {
  --bg: #f6f7f9;
  --panel: #ffffff;
  --line: #d7dbe2;
  --muted: #667085;
  --text: #172033;
  --strong: #101828;
  --accent: #0f766e;
  --accent-soft: #e7f5f2;
  --bad: #b42318;
  --bad-soft: #fff1f0;
  --warn: #b54708;
  --warn-soft: #fff6e5;
  --good: #067647;
  --good-soft: #e9f8ef;
  --code: #111827;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  color: var(--text);
  background: var(--bg);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  font-size: 14px;
}
button, input { font: inherit; }
#app {
  display: grid;
  grid-template-columns: minmax(300px, 360px) 1fr;
  height: 100vh;
  min-height: 620px;
}
.sidebar {
  min-width: 0;
  border-right: 1px solid var(--line);
  background: #fbfcfe;
  display: grid;
  grid-template-rows: auto auto 1fr;
}
.brand {
  height: 54px;
  padding: 12px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  border-bottom: 1px solid var(--line);
}
.brand-title { font-weight: 700; color: var(--strong); }
.icon-button, .tab, .small-button {
  border: 1px solid var(--line);
  background: var(--panel);
  color: var(--text);
  border-radius: 6px;
  padding: 7px 10px;
  cursor: pointer;
}
.icon-button:hover, .tab:hover, .small-button:hover { border-color: #9aa4b2; }
.search {
  width: calc(100% - 24px);
  margin: 10px 12px;
  border: 1px solid var(--line);
  background: var(--panel);
  border-radius: 6px;
  padding: 9px 10px;
  outline: none;
}
.search:focus { border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-soft); }
.run-list, .task-list {
  overflow: auto;
  min-height: 0;
  border-top: 1px solid var(--line);
}
.section-title {
  padding: 10px 12px 0;
  font-size: 12px;
  text-transform: uppercase;
  color: var(--muted);
  font-weight: 700;
}
.run-item, .task-item {
  width: 100%;
  text-align: left;
  border: 0;
  border-bottom: 1px solid #edf0f4;
  background: transparent;
  padding: 10px 12px;
  cursor: pointer;
}
.run-item:hover, .task-item:hover { background: #f0f4f8; }
.run-item.active, .task-item.active {
  background: var(--accent-soft);
  box-shadow: inset 3px 0 0 var(--accent);
}
.run-name, .task-name { font-weight: 650; color: var(--strong); overflow-wrap: anywhere; }
.run-meta, .task-meta {
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
  margin-top: 6px;
  color: var(--muted);
  font-size: 12px;
}
.pill {
  display: inline-flex;
  align-items: center;
  height: 22px;
  padding: 0 7px;
  border-radius: 999px;
  background: #eef2f6;
  color: #344054;
  font-size: 12px;
  white-space: nowrap;
}
.pill.good { color: var(--good); background: var(--good-soft); }
.pill.bad { color: var(--bad); background: var(--bad-soft); }
.pill.warn { color: var(--warn); background: var(--warn-soft); }
.main {
  min-width: 0;
  display: grid;
  grid-template-rows: auto auto auto 1fr;
}
.run-header {
  min-height: 66px;
  padding: 12px 18px;
  border-bottom: 1px solid var(--line);
  background: var(--panel);
}
.header-title {
  font-size: 18px;
  font-weight: 750;
  color: var(--strong);
  overflow-wrap: anywhere;
}
.metric-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
.metric {
  border: 1px solid var(--line);
  background: #fbfcfe;
  border-radius: 6px;
  padding: 5px 8px;
  min-width: 92px;
}
.metric-label { display: block; color: var(--muted); font-size: 11px; }
.metric-value { display: block; color: var(--strong); font-weight: 700; margin-top: 2px; }
.task-picker {
  min-height: 0;
  border-bottom: 1px solid var(--line);
  background: var(--panel);
  display: grid;
  grid-template-rows: auto 1fr;
}
.task-picker.collapsed {
  grid-template-rows: auto;
}
.task-picker.collapsed .task-list {
  display: none;
}
.task-picker.collapsed .task-search {
  display: none;
}
.task-picker-head {
  display: flex;
  gap: 14px;
  align-items: center;
  justify-content: space-between;
  padding: 9px 18px;
}
.task-picker-copy {
  min-width: 0;
}
.task-picker-title {
  font-weight: 750;
  color: var(--strong);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.task-picker-sub {
  margin-top: 2px;
  color: var(--muted);
  font-size: 12px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.task-picker-actions {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-shrink: 0;
}
.task-search {
  width: min(280px, 45%);
  min-width: 190px;
  margin: 0;
}
.task-list .empty {
  margin: 10px 20px;
}
.task-picker:not(.collapsed) .task-list {
  max-height: min(240px, 28vh);
}
.task-list .task-item {
  display: grid;
  grid-template-columns: 84px minmax(104px, auto) minmax(132px, auto) minmax(0, 1fr);
  align-items: center;
  gap: 10px;
}
.task-list .task-meta {
  margin-top: 0;
}
.task-question {
  min-width: 0;
  color: var(--muted);
  font-size: 12px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.tabs {
  display: flex;
  gap: 8px;
  padding: 10px 20px;
  border-bottom: 1px solid var(--line);
  background: #fbfcfe;
}
.tab.active {
  color: var(--accent);
  border-color: var(--accent);
  background: var(--accent-soft);
}
.content {
  min-height: 0;
  overflow: auto;
  padding: 16px 18px 34px;
}
.empty {
  color: var(--muted);
  padding: 24px;
  border: 1px dashed var(--line);
  border-radius: 8px;
  background: var(--panel);
}
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 14px;
}
.block {
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--panel);
  overflow: hidden;
}
.block-title {
  padding: 10px 12px;
  border-bottom: 1px solid var(--line);
  font-weight: 700;
  color: var(--strong);
  background: #fbfcfe;
}
.block-body { padding: 12px; }
.question {
  font-size: 15px;
  line-height: 1.55;
}
pre, code {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
}
pre {
  margin: 0;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  line-height: 1.45;
  color: var(--code);
  background: #f8fafc;
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 10px;
  max-height: 560px;
  overflow: auto;
}
.table-wrap { overflow: auto; max-height: 420px; border: 1px solid var(--line); border-radius: 6px; }
table { width: 100%; border-collapse: collapse; background: var(--panel); }
th, td { border-bottom: 1px solid #edf0f4; padding: 7px 8px; text-align: left; vertical-align: top; }
th { position: sticky; top: 0; background: #f8fafc; color: #344054; z-index: 1; }
td { font-size: 13px; }
.toolbar { display: flex; align-items: center; gap: 10px; margin-bottom: 12px; flex-wrap: wrap; }
.toggle {
  display: inline-flex;
  gap: 8px;
  align-items: center;
  border: 1px solid var(--line);
  padding: 7px 9px;
  border-radius: 6px;
  background: var(--panel);
}
.step {
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--panel);
  margin-bottom: 10px;
  overflow: hidden;
}
.step-head {
  display: flex;
  gap: 8px;
  align-items: center;
  flex-wrap: wrap;
  padding: 9px 11px;
  background: #fbfcfe;
  border-bottom: 1px solid var(--line);
}
.step-title { font-weight: 750; color: var(--strong); }
.step-body { padding: 11px; display: grid; gap: 10px; }
.two-col {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
  gap: 14px;
}
@media (max-width: 900px) {
  #app { grid-template-columns: 1fr; grid-template-rows: 46vh 54vh; }
  .sidebar { border-right: 0; border-bottom: 1px solid var(--line); }
  .two-col { grid-template-columns: 1fr; }
  .task-picker-head { align-items: stretch; flex-direction: column; }
  .task-picker-actions { align-items: stretch; flex-direction: column; }
  .task-search { width: 100%; min-width: 0; }
  .task-list .task-item { grid-template-columns: 76px 1fr; }
  .task-question { grid-column: 1 / -1; }
}
"""


JS = r"""
const state = {
  runs: [],
  selectedRun: null,
  runDetail: null,
  selectedTask: null,
  taskDetail: null,
  tab: "overview",
  showThinking: false,
  authToken: null,
  tasksOpen: true,
};

const $ = (id) => document.getElementById(id);

async function api(path) {
  const res = await fetch(path, {
    headers: state.authToken ? {"X-Trace-Viewer-Token": state.authToken} : {},
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status} ${text}`);
  }
  return res.json();
}

function fmtScore(v) {
  if (v === null || v === undefined || Number.isNaN(Number(v))) return "-";
  return Number(v).toFixed(4);
}

function fmtSmall(v) {
  if (v === null || v === undefined || v === "") return "-";
  if (typeof v === "number") return v.toFixed(v >= 100 ? 0 : 1);
  return String(v);
}

function esc(s) {
  return String(s ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function pill(text, cls = "") {
  return `<span class="pill ${cls}">${esc(text)}</span>`;
}

function scoreClass(score) {
  if (score === null || score === undefined) return "";
  const n = Number(score);
  if (n >= 0.999) return "good";
  if (n <= 0.001) return "bad";
  return "warn";
}

async function loadRuns() {
  const data = await api("/api/runs");
  state.runs = data.runs || [];
  renderRuns();
  if (!state.selectedRun && state.runs.length) {
    await selectRun(state.runs[0].run_id);
  }
}

async function selectRun(runId) {
  state.selectedRun = runId;
  state.selectedTask = null;
  state.taskDetail = null;
  state.tasksOpen = true;
  state.runDetail = await api(`/api/runs/${encodeURIComponent(runId)}`);
  renderRuns();
  renderTasks();
  renderHeader();
  const first = (state.runDetail.tasks || [])[0];
  if (first) await selectTask(first.task_id, {collapse: false});
  else renderContent();
}

async function selectTask(taskId, opts = {}) {
  state.selectedTask = taskId;
  state.taskDetail = await api(`/api/runs/${encodeURIComponent(state.selectedRun)}/tasks/${encodeURIComponent(taskId)}`);
  if (opts.collapse !== false) state.tasksOpen = false;
  renderTasks();
  renderContent();
}

function filteredRuns() {
  const q = $("runSearch").value.trim().toLowerCase();
  if (!q) return state.runs;
  return state.runs.filter(r => JSON.stringify(r).toLowerCase().includes(q));
}

function filteredTasks() {
  const tasks = (state.runDetail && state.runDetail.tasks) || [];
  const q = $("taskSearch").value.trim().toLowerCase();
  if (!q) return tasks;
  return tasks.filter(t => JSON.stringify(t).toLowerCase().includes(q));
}

function currentTaskRow() {
  const tasks = (state.runDetail && state.runDetail.tasks) || [];
  return tasks.find(t => t.task_id === state.selectedTask) || null;
}

function renderRuns() {
  const html = filteredRuns().map(r => {
    const active = r.run_id === state.selectedRun ? "active" : "";
    const statusCls = r.status === "finished" ? "good" : (r.status === "partial" ? "warn" : "");
    return `<button class="run-item ${active}" data-run="${esc(r.run_id)}">
      <div class="run-name">${esc(r.run_id)}</div>
      <div class="run-meta">
        ${pill(r.status, statusCls)}
        ${pill(`mean ${fmtScore(r.mean_score)}`)}
        ${pill(`${fmtSmall(r.completed)}/${fmtSmall(r.n_tasks)}`)}
        ${pill(r.mtime_iso || "")}
      </div>
    </button>`;
  }).join("");
  $("runList").innerHTML = html || `<div class="empty">No runs found.</div>`;
  document.querySelectorAll("[data-run]").forEach(btn => {
    btn.addEventListener("click", () => selectRun(btn.dataset.run));
  });
}

function renderTasks() {
  const tasks = filteredTasks();
  const allTasks = (state.runDetail && state.runDetail.tasks) || [];
  const row = currentTaskRow();
  const picker = $("taskPicker");
  picker.classList.toggle("collapsed", !state.tasksOpen);
  $("taskToggleBtn").textContent = state.tasksOpen ? "Hide tasks" : "Tasks";
  $("taskPickerTitle").textContent = row
    ? `${row.task_id}  score ${fmtScore(row.score)}`
    : (state.selectedRun ? "Select a task" : "Tasks");
  $("taskPickerSub").textContent = row
    ? (row.question || `${tasks.length} shown / ${allTasks.length} total`)
    : (state.selectedRun ? `${tasks.length} shown / ${allTasks.length} total` : "Select a run on the left.");
  const html = tasks.map(t => {
    const active = t.task_id === state.selectedTask ? "active" : "";
    const sc = t.score;
    return `<button class="task-item ${active}" data-task="${esc(t.task_id)}">
      <span class="task-name">${esc(t.task_id)}</span>
      <span>${pill(`score ${fmtScore(sc)}`, scoreClass(sc))}</span>
      <span class="task-meta">
        ${t.has_trace ? pill("trace", "good") : pill("no trace", "warn")}
        ${t.asr_status ? pill(`asr ${t.asr_status}`) : ""}
      </span>
      <span class="task-question">${esc(t.question || "")}</span>
    </button>`;
  }).join("");
  $("taskList").innerHTML = html || `<div class="empty">No tasks.</div>`;
  document.querySelectorAll("[data-task]").forEach(btn => {
    btn.addEventListener("click", () => selectTask(btn.dataset.task));
  });
}

function renderHeader() {
  const r = state.runDetail && state.runDetail.run;
  if (!r) {
    $("runHeader").innerHTML = `<div class="header-title">Select a run</div>`;
    return;
  }
  $("runHeader").innerHTML = `
    <div class="header-title">${esc(r.run_id)}</div>
    <div class="metric-row">
      <div class="metric"><span class="metric-label">Status</span><span class="metric-value">${esc(r.status)}</span></div>
      <div class="metric"><span class="metric-label">Mean</span><span class="metric-value">${fmtScore(r.mean_score)}</span></div>
      <div class="metric"><span class="metric-label">Progress</span><span class="metric-value">${fmtSmall(r.completed)} / ${fmtSmall(r.n_tasks)}</span></div>
      <div class="metric"><span class="metric-label">Predictions</span><span class="metric-value">${fmtSmall(r.prediction_count)}</span></div>
      <div class="metric"><span class="metric-label">Traces</span><span class="metric-value">${fmtSmall(r.trace_count)}</span></div>
      <div class="metric"><span class="metric-label">Updated</span><span class="metric-value">${esc(r.mtime_iso || "-")}</span></div>
    </div>`;
}

function renderContent() {
  if (!state.taskDetail) {
    $("content").innerHTML = `<div class="empty">Select a task to inspect the trace.</div>`;
    return;
  }
  if (state.tab === "overview") renderOverview();
  if (state.tab === "input") renderInput();
  if (state.tab === "trace") renderTrace();
  if (state.tab === "output") renderOutput();
  if (state.tab === "raw") renderRaw();
}

function renderOverview() {
  const d = state.taskDetail;
  const t = d.task || {};
  $("content").innerHTML = `
    <div class="grid">
      <div class="block">
        <div class="block-title">${esc(d.task_id)} overview</div>
        <div class="block-body">
          <div class="metric-row">
            ${pill(`score ${fmtScore(t.score)}`, scoreClass(t.score))}
            ${pill(`elapsed ${fmtSmall(t.elapsed)}s`)}
            ${pill(t.advisor_used === "True" ? "advisor" : "no advisor")}
            ${t.anti_agg_label ? pill(`anti ${t.anti_agg_label}`) : ""}
            ${t.asr_status ? pill(`asr ${t.asr_status}`) : ""}
          </div>
          <p class="question">${esc(d.question || "Question not found.")}</p>
          <pre>${esc(d.paths.task_dir)}</pre>
        </div>
      </div>
      ${artifactBlock(d)}
    </div>
    <div style="margin-top:14px">
      ${finalSqlBlock(d)}
    </div>
    <div class="two-col" style="margin-top:14px">
      ${csvBlock("Prediction", d.prediction)}
      ${csvBlock("Gold", d.gold)}
    </div>`;
}

function finalSqlBlock(d) {
  const last = d.trace_meta && d.trace_meta.last_answer_sql;
  if (!last || !last.sql) {
    return `<div class="block">
      <div class="block-title">Final answer SQL</div>
      <div class="block-body"><div class="empty">No answer_from_sql step found.</div></div>
    </div>`;
  }
  const route = last.source_route || {};
  return `<div class="block">
    <div class="block-title">Final answer SQL</div>
    <div class="block-body">
      <div class="metric-row">
        ${last.phase ? pill(`phase ${last.phase}`) : ""}
        ${last.step !== undefined && last.step !== null ? pill(`step ${last.step}`) : ""}
        ${route.route ? pill(`route ${route.route}`) : ""}
        ${route.sql_role ? pill(`sql ${route.sql_role}`) : ""}
      </div>
      <pre>${esc(last.sql)}</pre>
      ${route.reason ? `<p class="question">${esc(route.reason)}</p>` : ""}
    </div>
  </div>`;
}

function artifactBlock(d) {
  const files = (d.artifacts.files || []).map(f =>
    `<tr><td>${esc(f.name)}</td><td>${fmtSmall(f.size)}</td><td>${esc(f.mtime_iso)}</td></tr>`
  ).join("");
  return `<div class="block">
    <div class="block-title">Artifacts</div>
    <div class="block-body">
      <div class="table-wrap"><table><thead><tr><th>File</th><th>Bytes</th><th>Modified</th></tr></thead><tbody>${files}</tbody></table></div>
    </div>
  </div>`;
}

function csvBlock(title, csv) {
  if (!csv || !csv.exists) {
    return `<div class="block"><div class="block-title">${esc(title)}</div><div class="block-body"><div class="empty">Missing.</div></div></div>`;
  }
  const head = (csv.columns || []).map(c => `<th>${esc(c)}</th>`).join("");
  const rows = (csv.rows || []).map(r => `<tr>${r.map(c => `<td>${esc(c)}</td>`).join("")}</tr>`).join("");
  return `<div class="block">
    <div class="block-title">${esc(title)} ${csv.truncated ? "(truncated)" : ""}</div>
    <div class="block-body">
      <div class="table-wrap"><table><thead><tr>${head}</tr></thead><tbody>${rows}</tbody></table></div>
    </div>
  </div>`;
}

function renderInput() {
  const d = state.taskDetail;
  const input = d.input || {};
  const snapshot = input.snapshot;
  const pre = input.injected_preamble || {};
  const prompts = (input.phase_prompts && input.phase_prompts.prompts) || {};
  const phaseBlocks = ["plan", "explore", "answer", "verify"].map(phase => `
    <div class="block">
      <div class="block-title">${phase.toUpperCase()} system prompt</div>
      <div class="block-body"><pre>${esc(prompts[phase] || "Not available.")}</pre></div>
    </div>`).join("");
  $("content").innerHTML = `
    <div class="block">
      <div class="block-title">Input snapshot</div>
      <div class="block-body">
        ${snapshot ? `<pre>${esc(JSON.stringify(snapshot, null, 2))}</pre>` :
          `<div class="empty">input_snapshot.json was not saved for this run. The prompts below are reconstructed when possible.</div>`}
      </div>
    </div>
    <div class="block" style="margin-top:14px">
      <div class="block-title">Injected preamble</div>
      <div class="block-body">${pre.exists ? `<pre>${esc(pre.text)}</pre>` : `<div class="empty">Injected preamble was not saved for this run.</div>`}</div>
    </div>
    <div class="block" style="margin-top:14px">
      <div class="block-title">Prompt reconstruction note</div>
      <div class="block-body"><pre>${esc(JSON.stringify(input.phase_prompts || {}, null, 2))}</pre></div>
    </div>
    <div class="grid" style="margin-top:14px">${phaseBlocks}</div>`;
}

function renderTrace() {
  const d = state.taskDetail;
  const attempts = d.trace || [];
  const controls = `<div class="toolbar">
    <label class="toggle"><input id="thinkingToggle" type="checkbox" ${state.showThinking ? "checked" : ""}> Show thinking</label>
    ${pill(`${attempts.length} attempts`)}
    ${d.trace_meta && d.trace_meta.is_truncated_trace ? pill("preview trace", "warn") : ""}
  </div>`;
  const body = attempts.map((attempt, idx) => renderAttempt(attempt, idx)).join("");
  $("content").innerHTML = controls + (body || `<div class="empty">No trace.json found.</div>`);
  $("thinkingToggle").addEventListener("change", (e) => {
    state.showThinking = e.target.checked;
    renderTrace();
  });
}

function renderAttempt(attempt, idx) {
  const steps = attempt.steps || [];
  return `<div class="block" style="margin-bottom:14px">
    <div class="block-title">Attempt ${esc(attempt.attempt ?? idx)} ${attempt.failure_reason ? "failure: " + esc(attempt.failure_reason) : ""}</div>
    <div class="block-body">
      ${steps.map(renderStep).join("")}
    </div>
  </div>`;
}

function renderStep(step) {
  const obs = step.observation || {};
  const input = step.action_input || {};
  const thought = step.thought || "";
  const ok = obs.ok ?? step.observation_ok;
  const okPill = ok === false ? pill("not ok", "bad") : pill("ok", "good");
  return `<div class="step">
    <div class="step-head">
      <span class="step-title">Step ${esc(step.i ?? step.step_index ?? "-")}</span>
      ${pill(String(step.phase || "-"))}
      ${pill(String(step.action || "-"))}
      ${okPill}
    </div>
    <div class="step-body">
      ${state.showThinking ? `<div><strong>Thought</strong><pre>${esc(thought)}</pre></div>` : ""}
      <div><strong>Action input</strong><pre>${esc(JSON.stringify(input, null, 2))}</pre></div>
      <div><strong>Observation</strong><pre>${esc(JSON.stringify(obs, null, 2))}</pre></div>
    </div>
  </div>`;
}

function renderOutput() {
  const d = state.taskDetail;
  const answer = ((d.trace || [])[0] || {}).answer;
  $("content").innerHTML = `
    <div class="grid">
      ${finalSqlBlock(d)}
      <div class="block">
        <div class="block-title">Trace answer</div>
        <div class="block-body"><pre>${esc(answer ? JSON.stringify(answer, null, 2) : "No answer object in trace.")}</pre></div>
      </div>
    </div>
    <div class="two-col" style="margin-top:14px">
      ${csvBlock("Prediction", d.prediction)}
      ${csvBlock("Gold", d.gold)}
    </div>`;
}

function renderRaw() {
  const d = state.taskDetail;
  const textArtifacts = d.artifacts.text || {};
  const jsonArtifacts = d.artifacts.json || {};
  const textBlocks = Object.entries(textArtifacts).map(([name, info]) => `
    <div class="block"><div class="block-title">${esc(name)}</div><div class="block-body"><pre>${esc(info.text || "")}</pre></div></div>`).join("");
  $("content").innerHTML = `
    <div class="block">
      <div class="block-title">Raw trace.json</div>
      <div class="block-body"><pre>${esc(JSON.stringify(d.trace || [], null, 2))}</pre></div>
    </div>
    <div class="block" style="margin-top:14px">
      <div class="block-title">attempt_0.steps.log</div>
      <div class="block-body">${d.steps_log.exists ? `<pre>${esc(d.steps_log.text)}</pre>` : `<div class="empty">Missing.</div>`}</div>
    </div>
    <div class="block" style="margin-top:14px">
      <div class="block-title">JSON artifacts</div>
      <div class="block-body"><pre>${esc(JSON.stringify(jsonArtifacts, null, 2))}</pre></div>
    </div>
    <div class="grid" style="margin-top:14px">${textBlocks}</div>`;
}

document.querySelectorAll(".tab").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    state.tab = btn.dataset.tab;
    if (state.tab === "trace") {
      state.tasksOpen = false;
      renderTasks();
    }
    renderContent();
  });
});

$("refreshBtn").addEventListener("click", async () => {
  const run = state.selectedRun;
  const task = state.selectedTask;
  await loadRuns();
  if (run) {
    await selectRun(run);
    if (task) await selectTask(task);
  }
});
$("runSearch").addEventListener("input", renderRuns);
$("taskSearch").addEventListener("input", renderTasks);
$("taskToggleBtn").addEventListener("click", () => {
  state.tasksOpen = !state.tasksOpen;
  renderTasks();
});

const TOKEN_TTL_MS = Number("__TOKEN_TTL_MS__") || 0;
const params = new URLSearchParams(window.location.search);
const nowMs = Date.now();
const storedExpires = Number(window.localStorage.getItem("traceViewerTokenExpiresAt") || "0");
if (storedExpires && storedExpires <= nowMs) {
  window.localStorage.removeItem("traceViewerToken");
  window.localStorage.removeItem("traceViewerTokenExpiresAt");
}
const urlToken = params.get("token");
state.authToken = urlToken || window.localStorage.getItem("traceViewerToken");
if (urlToken) {
  window.localStorage.setItem("traceViewerToken", urlToken);
  if (TOKEN_TTL_MS > 0) {
    window.localStorage.setItem("traceViewerTokenExpiresAt", String(nowMs + TOKEN_TTL_MS));
  } else {
    window.localStorage.removeItem("traceViewerTokenExpiresAt");
  }
  window.history.replaceState({}, "", window.location.pathname);
}

loadRuns().catch(err => {
  $("content").innerHTML = `<div class="empty">${esc(err.stack || err.message || err)}</div>`;
});
"""


class TraceViewerHandler(BaseHTTPRequestHandler):
    server_version = "KobushiTraceViewer/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; "
            "base-uri 'none'; frame-ancestors 'none'",
        )
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=_json_default).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def _error(self, status: int, message: str) -> None:
        self._json({"error": message}, status)

    def _access_email(self) -> str | None:
        # Header set by Cloudflare Access after authentication.
        email = self.headers.get("Cf-Access-Authenticated-User-Email")
        return email.strip().lower() if email else None

    def _request_token(self) -> str | None:
        token = self.headers.get("X-Trace-Viewer-Token")
        if token:
            return token.strip()
        auth = self.headers.get("Authorization", "")
        if auth.lower().startswith("bearer "):
            return auth[7:].strip()
        qs = parse_qs(urlparse(self.path).query)
        vals = qs.get("token") or []
        return vals[0].strip() if vals else None

    def _check_access(self) -> bool:
        server = self.server
        auth_token = getattr(server, "auth_token", None)
        if auth_token:
            expires_at = getattr(server, "auth_token_expires_at", None)
            if expires_at is not None and time.time() > expires_at:
                self._error(HTTPStatus.UNAUTHORIZED, "trace viewer token expired")
                return False
            supplied = self._request_token()
            if not supplied or not hmac.compare_digest(str(auth_token), supplied):
                self._error(HTTPStatus.UNAUTHORIZED, "missing or invalid trace viewer token")
                return False
        require = bool(getattr(server, "require_cf_access", False))
        if not require:
            return True
        email = self._access_email()
        if not email:
            self._error(
                HTTPStatus.UNAUTHORIZED,
                "missing Cf-Access-Authenticated-User-Email header",
            )
            return False
        allowed_emails = getattr(server, "allowed_access_emails", set())
        allowed_domains = getattr(server, "allowed_access_domains", set())
        if allowed_emails and email in allowed_emails:
            return True
        if allowed_domains and any(email.endswith("@" + d) for d in allowed_domains):
            return True
        if not allowed_emails and not allowed_domains:
            return True
        self._error(HTTPStatus.FORBIDDEN, f"access denied for {email}")
        return False

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        try:
            if path == "/healthz":
                self._json({"ok": True, "service": "trace_viewer"})
                return
            if not self._check_access():
                return
            if path == "/":
                ttl_ms = int(max(0.0, getattr(self.server, "auth_token_ttl_seconds", 0.0)) * 1000)
                html = (
                    HTML.replace("__CSS__", CSS)
                    .replace("__JS__", JS)
                    .replace("__TOKEN_TTL_MS__", str(ttl_ms))
                )
                self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")
                return
            if path == "/api/whoami":
                self._json({
                    "email": self._access_email(),
                    "cf_access_required": bool(getattr(self.server, "require_cf_access", False)),
                    "token_required": bool(getattr(self.server, "auth_token", None)),
                    "token_ttl_seconds": getattr(self.server, "auth_token_ttl_seconds", 0.0),
                    "token_expires_at": getattr(self.server, "auth_token_expires_at", None),
                })
                return
            if path == "/api/runs":
                self._json(list_runs())
                return
            parts = [unquote(p) for p in path.split("/") if p]
            if len(parts) == 3 and parts[0] == "api" and parts[1] == "runs":
                self._json(get_run(parts[2]))
                return
            if len(parts) == 5 and parts[0] == "api" and parts[1] == "runs" and parts[3] == "tasks":
                self._json(get_task_detail(parts[2], parts[4]))
                return
            if len(parts) == 6 and parts[0] == "api" and parts[1] == "runs" and parts[3] == "tasks" and parts[5] == "file":
                qs = parse_qs(parsed.query)
                name = (qs.get("name") or [""])[0]
                self._json(get_file(parts[2], parts[4], name))
                return
            self._error(HTTPStatus.NOT_FOUND, "not found")
        except FileNotFoundError as exc:
            self._error(HTTPStatus.NOT_FOUND, str(exc))
        except ValueError as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, repr(exc))


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a local Kobushi trace viewer.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--require-cf-access",
        action="store_true",
        help="Reject requests that do not include Cloudflare Access user headers.",
    )
    parser.add_argument(
        "--access-email",
        action="append",
        default=[],
        help="Allowed Cloudflare Access email. Repeatable. Only used with --require-cf-access.",
    )
    parser.add_argument(
        "--access-email-domain",
        action="append",
        default=[],
        help="Allowed email domain, e.g. example.com. Repeatable. Only used with --require-cf-access.",
    )
    parser.add_argument(
        "--auth-token",
        default=os.environ.get("TRACE_VIEWER_AUTH_TOKEN"),
        help="Require this bearer/header/query token. Useful for short-lived quick tunnels.",
    )
    parser.add_argument(
        "--auth-token-ttl-hours",
        type=float,
        default=float(os.environ.get("TRACE_VIEWER_AUTH_TOKEN_TTL_HOURS", "0") or 0),
        help="Expire --auth-token after this many hours. 0 disables token expiry.",
    )
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), TraceViewerHandler)
    server.require_cf_access = args.require_cf_access
    server.auth_token = args.auth_token
    token_ttl_seconds = max(0.0, args.auth_token_ttl_hours * 3600.0)
    server.auth_token_ttl_seconds = token_ttl_seconds
    server.auth_token_expires_at = (
        time.time() + token_ttl_seconds if args.auth_token and token_ttl_seconds > 0 else None
    )
    server.allowed_access_emails = {e.strip().lower() for e in args.access_email if e.strip()}
    server.allowed_access_domains = {
        d.strip().lower().removeprefix("@") for d in args.access_email_domain if d.strip()
    }
    print(f"Trace viewer: http://{args.host}:{args.port}", flush=True)
    print(f"Runs root: {RUNS_ROOT}", flush=True)
    if args.auth_token:
        print("Trace viewer token check: enabled", flush=True)
        if server.auth_token_expires_at is not None:
            print(f"Trace viewer token TTL: {args.auth_token_ttl_hours:g}h", flush=True)
    if args.require_cf_access:
        print("Cloudflare Access header check: enabled", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
