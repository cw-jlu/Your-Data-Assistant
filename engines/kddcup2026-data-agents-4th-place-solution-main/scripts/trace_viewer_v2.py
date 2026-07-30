#!/usr/bin/env python3
"""Matrix-first read-only trace viewer for Phase 2 experiment runs.

Run:
  uv run python scripts/trace_viewer_v2.py --port 8776

Then open:
  http://127.0.0.1:8776
"""
from __future__ import annotations

import argparse
import hmac
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


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import trace_viewer as v1  # noqa: E402


def _phase2_task_ids() -> list[str]:
    task_ids: set[str] = set()
    if v1.PHASE2_INPUT_ROOT.is_dir():
        for path in v1.PHASE2_INPUT_ROOT.iterdir():
            if path.is_dir() and re.fullmatch(r"task_\d+", path.name):
                task_ids.add(path.name)
    return sorted(task_ids, key=v1._task_sort_key)


def _run_task_ids(run_dir: Path, summary: dict[str, Any] | None, log_scores: dict[str, Any]) -> set[str]:
    task_ids = set(v1._summary_results(summary))
    task_ids.update(log_scores)
    for child in run_dir.iterdir():
        if child.is_dir() and re.fullmatch(r"task_\d+", child.name):
            task_ids.add(child.name)
    return task_ids


def _cell_status(task_dir: Path, score: Any) -> str:
    if score is not None:
        return "scored"
    if (task_dir / "attempt_0.steps.log").is_file():
        return "running"
    if task_dir.is_dir():
        return "started"
    return "missing"


def _build_cells(
    run_dir: Path,
    summary: dict[str, Any] | None,
    log_scores: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    results = v1._summary_results(summary)
    cells: dict[str, dict[str, Any]] = {}
    for tid in sorted(_run_task_ids(run_dir, summary, log_scores), key=v1._task_sort_key):
        task_dir = run_dir / tid
        result = results.get(tid, {})
        log_row = log_scores.get(tid, {})
        score = v1._task_score(task_dir, tid, result, log_row)
        elapsed = result.get("elapsed", log_row.get("elapsed"))
        cells[tid] = {
            "task_id": tid,
            "score": score,
            "elapsed": elapsed,
            "n_ok": result.get("n_ok"),
            "status": _cell_status(task_dir, score),
            "has_prediction": (task_dir / "prediction.csv").is_file(),
            "has_trace": (task_dir / "trace.json").is_file(),
            "has_steps_log": (task_dir / "attempt_0.steps.log").is_file(),
            "domain": result.get("domain"),
            "anti_agg_label": result.get("anti_agg_label") or v1._anti_agg_label(task_dir),
            "asr_status": result.get("asr_status") or v1._asr_status(task_dir),
            "live": v1._live_progress(task_dir) if score is None else None,
        }
    return cells


def _cell_scores(cells: dict[str, dict[str, Any]]) -> list[float]:
    scores: list[float] = []
    for cell in cells.values():
        score = cell.get("score")
        if score is None:
            continue
        try:
            scores.append(float(score))
        except (TypeError, ValueError):
            continue
    return scores


def _apply_live_rollups(row: dict[str, Any], cells: dict[str, dict[str, Any]], task_total: int) -> None:
    scores = _cell_scores(cells)
    if scores and row.get("mean_score") is None:
        row["mean_score"] = sum(scores) / len(scores)
    if row.get("n_tasks") is None and task_total:
        row["n_tasks"] = task_total
    if scores:
        row["completed"] = max(int(row.get("completed") or 0), len(scores))
        row["live_score_count"] = len(scores)
    if row.get("n_perfect") is None and scores:
        row["n_perfect"] = sum(1 for score in scores if score >= 0.999)
    if row.get("n_zero") is None and scores:
        row["n_zero"] = sum(1 for score in scores if score <= 0.001)


def get_matrix() -> dict[str, Any]:
    logs = v1._parse_terminal_logs()
    phase2_tasks = _phase2_task_ids()
    rows: list[dict[str, Any]] = []

    if not v1.RUNS_ROOT.is_dir():
        return {"runs": [], "task_ids": phase2_tasks, "tasks": [], "root": str(v1.RUNS_ROOT)}

    for run_dir in sorted(v1.RUNS_ROOT.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not run_dir.is_dir():
            continue
        summary = v1._read_json(run_dir / "summary.json")
        log_info = logs.get(run_dir.name)
        log_scores = (log_info or {}).get("scores") or {}
        cells = _build_cells(run_dir, summary, log_scores)
        if not cells and not isinstance(summary, dict):
            continue
        row = v1._run_summary(run_dir, log_info)
        row["cells"] = cells
        row["n_perfect"] = summary.get("n_perfect") if isinstance(summary, dict) else None
        row["n_zero"] = summary.get("n_zero") if isinstance(summary, dict) else None
        _apply_live_rollups(row, cells, len(phase2_tasks))
        rows.append(row)

    rows.sort(key=lambda r: (r.get("mtime") or 0, r.get("run_id") or ""), reverse=True)

    # The v2 matrix is intentionally Phase-2-first: rows may include older
    # experiments, but the task columns are the demo Phase 2 task set.
    ordered_tasks = phase2_tasks
    task_meta = []
    for tid in ordered_tasks:
        task_json = v1._load_task_json(tid) or {}
        task_meta.append(
            {
                "task_id": tid,
                "question": task_json.get("question"),
                "has_gold": v1._gold_path(tid) is not None,
            }
        )
    return {
        "runs": rows[:800],
        "task_ids": ordered_tasks,
        "tasks": task_meta,
        "root": str(v1.RUNS_ROOT),
        "generated_at": time.time(),
        "generated_at_iso": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    }


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Kobushi Trace Viewer v2</title>
  <style>__CSS__</style>
</head>
<body>
  <div id="app">
    <header class="topbar">
      <div class="brand">
        <div class="brand-title">Trace Viewer v2</div>
        <div id="statusLine" class="brand-sub">Loading...</div>
      </div>
      <div class="top-actions">
        <input id="runSearch" class="search" placeholder="Filter runs, exp, tag, lever">
        <label class="toggle"><input id="autoRefresh" type="checkbox" checked> Auto refresh</label>
        <button id="refreshBtn" class="button">Refresh</button>
      </div>
    </header>
    <main id="view" class="view"></main>
  </div>
  <script>__JS__</script>
</body>
</html>
"""


CSS = r"""
:root {
  --bg: #f5f7fa;
  --panel: #ffffff;
  --line: #d8dee8;
  --text: #162033;
  --muted: #667085;
  --strong: #101828;
  --accent: #0f766e;
  --accent-soft: #e6f4f1;
  --good: #067647;
  --good-bg: #e9f8ef;
  --mid: #b54708;
  --mid-bg: #fff6e5;
  --bad: #b42318;
  --bad-bg: #fff1f0;
  --none: #98a2b3;
  --none-bg: #f2f4f7;
  --blue: #175cd3;
  --blue-bg: #eff6ff;
  --code: #111827;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  min-width: 1040px;
  color: var(--text);
  background: var(--bg);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  font-size: 14px;
}
button, input { font: inherit; }
.topbar {
  height: 52px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 8px 12px;
  border-bottom: 1px solid var(--line);
  background: #fbfcfe;
  position: sticky;
  top: 0;
  z-index: 50;
}
.brand { min-width: 240px; }
.brand-title { color: var(--strong); font-weight: 800; font-size: 17px; }
.brand-sub { color: var(--muted); font-size: 12px; margin-top: 2px; }
.top-actions { display: flex; align-items: center; gap: 10px; min-width: 0; flex: 1; justify-content: flex-end; }
.search {
  width: min(520px, 52vw);
  border: 1px solid var(--line);
  border-radius: 6px;
  background: var(--panel);
  padding: 8px 10px;
  outline: none;
}
.search:focus { border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-soft); }
.button, .back-button, .tab {
  border: 1px solid var(--line);
  border-radius: 6px;
  background: var(--panel);
  color: var(--text);
  padding: 7px 10px;
  cursor: pointer;
}
.button:hover, .back-button:hover, .tab:hover { border-color: #98a2b3; }
.toggle {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  color: var(--muted);
  white-space: nowrap;
}
.view { min-height: calc(100vh - 52px); }
.matrix-shell { padding: 10px 12px 18px; }
.matrix-summary {
  display: flex;
  gap: 8px;
  align-items: center;
  flex-wrap: wrap;
  margin-bottom: 8px;
}
.pill {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-height: 22px;
  padding: 2px 7px;
  border-radius: 999px;
  background: var(--none-bg);
  color: #344054;
  font-size: 12px;
  white-space: nowrap;
}
.pill.good { background: var(--good-bg); color: var(--good); }
.pill.mid { background: var(--mid-bg); color: var(--mid); }
.pill.bad { background: var(--bad-bg); color: var(--bad); }
.pill.blue { background: var(--blue-bg); color: var(--blue); }
.matrix-wrap {
  height: calc(100vh - 108px);
  overflow: auto;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--panel);
}
.matrix {
  border-collapse: separate;
  border-spacing: 0;
  min-width: max-content;
  width: 100%;
}
.matrix th, .matrix td {
  border-right: 1px solid #edf0f4;
  border-bottom: 1px solid #edf0f4;
  padding: 0;
  vertical-align: middle;
}
.matrix tbody tr {
  height: 38px;
}
.matrix tbody td {
  height: 38px;
}
.matrix th {
  position: sticky;
  top: 0;
  z-index: 4;
  background: #f8fafc;
  color: #344054;
  font-size: 12px;
  font-weight: 750;
  text-align: center;
  height: 34px;
  padding: 0 6px;
}
.matrix .sticky-run {
  position: sticky;
  left: 0;
  z-index: 5;
  background: #fbfcfe;
  min-width: 300px;
  max-width: 360px;
}
.matrix th.sticky-run { z-index: 7; }
.matrix .sticky-score {
  position: sticky;
  left: 300px;
  z-index: 5;
  background: #fbfcfe;
  min-width: 86px;
}
.matrix th.sticky-score { z-index: 7; }
.matrix .meta-cell {
  min-width: 72px;
  padding: 4px 6px;
  font-size: 12px;
  color: var(--muted);
  white-space: nowrap;
}
.run-cell {
  width: 100%;
  height: 38px;
  border: 0;
  background: transparent;
  text-align: left;
  padding: 3px 8px;
  cursor: pointer;
  display: grid;
  grid-template-rows: 17px 16px;
  align-items: center;
  overflow: hidden;
}
.run-cell:hover { background: var(--accent-soft); }
.run-name {
  color: var(--strong);
  font-weight: 750;
  line-height: 17px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.run-tags {
  display: flex;
  gap: 4px;
  flex-wrap: nowrap;
  margin-top: 0;
  overflow: hidden;
  min-width: 0;
  height: 16px;
  align-items: center;
}
.run-tags .pill {
  min-height: 15px;
  height: 15px;
  padding: 0 5px;
  font-size: 10px;
}
.score-main { font-weight: 800; color: var(--strong); font-variant-numeric: tabular-nums; }
.score-sub { color: var(--muted); font-size: 11px; margin-top: 2px; }
.score-cell {
  width: 50px;
  height: 38px;
  display: flex;
  align-items: center;
  justify-content: center;
  border: 0;
  border-radius: 0;
  cursor: pointer;
  color: var(--text);
  background: var(--none-bg);
  font-size: 11px;
  font-weight: 700;
  font-variant-numeric: tabular-nums;
}
.score-cell:hover {
  outline: 2px solid var(--accent);
  outline-offset: -2px;
}
.score-cell.good { background: var(--good-bg); color: var(--good); }
.score-cell.mid { background: var(--mid-bg); color: var(--mid); }
.score-cell.bad { background: var(--bad-bg); color: var(--bad); }
.score-cell.partial { background: #fffbea; color: #854a0e; }
.score-cell.running { background: var(--blue-bg); color: var(--blue); border: 1px dashed var(--blue); }
.score-cell.missing { color: var(--none); }
.task-head {
  min-width: 50px;
  max-width: 50px;
  height: 32px;
  padding: 0 4px;
  text-align: center;
  white-space: nowrap;
  font-size: 11px;
  font-variant-numeric: tabular-nums;
}
.detail-page { min-height: calc(100vh - 52px); display: grid; grid-template-rows: auto auto 1fr; }
.detail-head {
  padding: 10px 14px;
  border-bottom: 1px solid var(--line);
  background: var(--panel);
}
.detail-actions { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin-bottom: 7px; }
.detail-title { font-weight: 800; font-size: 18px; color: var(--strong); overflow-wrap: anywhere; }
.detail-sub { color: var(--muted); margin-top: 5px; line-height: 1.4; }
.metrics { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 10px; }
.metric {
  border: 1px solid var(--line);
  border-radius: 7px;
  background: #fbfcfe;
  min-width: 86px;
  padding: 6px 8px;
}
.metric-label { display: block; color: var(--muted); font-size: 11px; }
.metric-value { display: block; color: var(--strong); font-weight: 800; margin-top: 2px; }
.run-task-table-wrap {
  min-height: 0;
  overflow: auto;
  padding: 12px 14px 24px;
}
.task-table {
  width: 100%;
  border-collapse: collapse;
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  overflow: hidden;
}
.task-table th, .task-table td {
  border-bottom: 1px solid #edf0f4;
  padding: 7px 8px;
  text-align: left;
  vertical-align: top;
}
.task-table th { position: sticky; top: 0; background: #f8fafc; z-index: 2; color: #344054; }
.task-row { cursor: pointer; }
.task-row:hover { background: var(--accent-soft); }
.tabs {
  display: flex;
  gap: 8px;
  padding: 8px 14px;
  border-bottom: 1px solid var(--line);
  background: #fbfcfe;
}
.tab.active { border-color: var(--accent); color: var(--accent); background: var(--accent-soft); }
.content { overflow: auto; padding: 12px 14px 28px; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 12px; }
.two-col { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 12px; }
.block {
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--panel);
  overflow: hidden;
}
.block-title {
  padding: 9px 11px;
  border-bottom: 1px solid var(--line);
  background: #fbfcfe;
  color: var(--strong);
  font-weight: 750;
}
.block-body { padding: 10px; }
.question { line-height: 1.55; font-size: 15px; }
.empty {
  color: var(--muted);
  padding: 20px;
  border: 1px dashed var(--line);
  border-radius: 8px;
  background: var(--panel);
}
pre, code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; }
pre {
  margin: 0;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  line-height: 1.45;
  color: var(--code);
  background: #f8fafc;
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 9px;
  max-height: 560px;
  overflow: auto;
}
.table-wrap { overflow: auto; max-height: 440px; border: 1px solid var(--line); border-radius: 6px; }
table.preview { width: 100%; border-collapse: collapse; background: var(--panel); }
table.preview th, table.preview td { border-bottom: 1px solid #edf0f4; padding: 7px 8px; text-align: left; vertical-align: top; }
table.preview th { position: sticky; top: 0; background: #f8fafc; color: #344054; z-index: 1; }
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
  padding: 8px 10px;
  background: #fbfcfe;
  border-bottom: 1px solid var(--line);
}
.step-title { font-weight: 800; color: var(--strong); }
.step-body { padding: 10px; display: grid; gap: 9px; }
@media (max-width: 980px) {
  body { min-width: 760px; }
  .topbar { align-items: stretch; height: auto; flex-direction: column; }
  .top-actions { justify-content: stretch; }
  .search { width: 100%; }
  .two-col { grid-template-columns: 1fr; }
}
"""


JS = r"""
const state = {
  matrix: null,
  view: "matrix",
  selectedRun: null,
  selectedTask: null,
  runDetail: null,
  taskDetail: null,
  tab: "overview",
  showThinking: false,
  authToken: null,
  autoTimer: null,
};

const $ = (id) => document.getElementById(id);

function esc(s) {
  return String(s ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

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

function fmtCellScore(v) {
  if (v === null || v === undefined || Number.isNaN(Number(v))) return "";
  const n = Number(v);
  if (n >= 0.999) return "1";
  if (n <= 0.001) return "0";
  return n.toFixed(2).replace(/^0/, "");
}

function fmtSmall(v) {
  if (v === null || v === undefined || v === "") return "-";
  if (typeof v === "number") return v.toFixed(v >= 100 ? 0 : 1);
  return String(v);
}

function pill(text, cls = "") {
  return `<span class="pill ${cls}">${esc(text)}</span>`;
}

function scoreClass(score) {
  if (score === null || score === undefined || Number.isNaN(Number(score))) return "missing";
  const n = Number(score);
  if (n >= 0.999) return "good";
  if (n <= 0.001) return "bad";
  return n >= 0.5 ? "mid" : "partial";
}

function taskMeta(taskId) {
  return (state.matrix?.tasks || []).find(t => t.task_id === taskId) || {};
}

async function loadMatrix({silent = false} = {}) {
  const data = await api("/api/matrix");
  state.matrix = data;
  $("statusLine").textContent = `${data.runs.length} runs, ${data.task_ids.length} task columns, generated ${data.generated_at_iso}`;
  if (state.view === "matrix" || !silent) render();
}

function updateRoute({view, run, task, tab, replace = false}) {
  const url = new URL(window.location.href);
  url.searchParams.delete("run");
  url.searchParams.delete("task");
  url.searchParams.delete("tab");
  if (view === "run" && run) {
    url.searchParams.set("run", run);
  }
  if (view === "task" && run && task) {
    url.searchParams.set("run", run);
    url.searchParams.set("task", task);
    if (tab && tab !== "overview") url.searchParams.set("tab", tab);
  }
  const next = url.pathname + url.search + url.hash;
  const payload = {view, run: run || null, task: task || null, tab: tab || "overview"};
  if (replace) window.history.replaceState(payload, "", next);
  else window.history.pushState(payload, "", next);
}

async function applyRouteFromLocation() {
  const params = new URLSearchParams(window.location.search);
  const run = params.get("run");
  const task = params.get("task");
  const tab = params.get("tab") || "overview";
  if (run && task) {
    await openTask(run, task, {push: false, tab});
    return;
  }
  if (run) {
    await openRun(run, {push: false});
    return;
  }
  state.view = "matrix";
  state.selectedRun = null;
  state.selectedTask = null;
  state.runDetail = null;
  state.taskDetail = null;
  state.tab = "overview";
  render();
}

function filteredRuns() {
  const q = $("runSearch").value.trim().toLowerCase();
  const runs = state.matrix?.runs || [];
  if (!q) return runs;
  return runs.filter(r => {
    const hay = [
      r.run_id, r.status, r.tag,
      ...(r.levers || []),
      JSON.stringify(r.config || {}),
    ].join(" ").toLowerCase();
    return hay.includes(q);
  });
}

function render() {
  if (state.view === "matrix") renderMatrix();
  if (state.view === "run") renderRun();
  if (state.view === "task") renderTask();
}

function renderMatrix() {
  const runs = filteredRuns();
  const tasks = state.matrix?.task_ids || [];
  const headTasks = tasks.map(t => {
    const meta = taskMeta(t);
    return `<th class="task-head" title="${esc(meta.question || t)}">${esc(t.replace("task_", "t"))}</th>`;
  }).join("");
  const rows = runs.map(r => {
    const cells = tasks.map(t => matrixCell(r, t)).join("");
    return `<tr>
      <td class="sticky-run">${runButton(r)}</td>
      <td class="sticky-score meta-cell">
        <div class="score-main">${fmtScore(r.mean_score)}</div>
        <div class="score-sub">${esc(r.status)}</div>
      </td>
      <td class="meta-cell">${fmtSmall(r.completed)} / ${fmtSmall(r.n_tasks)}</td>
      <td class="meta-cell">${fmtSmall(r.n_perfect)}</td>
      <td class="meta-cell">${fmtSmall(r.n_zero)}</td>
      <td class="meta-cell">${fmtSmall(r.elapsed_minutes)}m</td>
      ${cells}
    </tr>`;
  }).join("");
  $("view").innerHTML = `
    <section class="matrix-shell">
      <div class="matrix-summary">
        ${pill(`${runs.length} shown`)}
        ${pill(`${state.matrix?.runs?.length || 0} total runs`)}
        ${pill(`${tasks.length} task columns`)}
        ${pill("artifacts/runs", "blue")}
      </div>
      <div class="matrix-wrap">
        <table class="matrix">
          <thead>
            <tr>
              <th class="sticky-run">Run</th>
              <th class="sticky-score">Overall</th>
              <th>Done</th>
              <th>Perfect</th>
              <th>Zero</th>
              <th>Elapsed</th>
              ${headTasks}
            </tr>
          </thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
    </section>`;
  document.querySelectorAll("[data-open-run]").forEach(btn => {
    btn.addEventListener("click", () => openRun(btn.dataset.openRun));
  });
  document.querySelectorAll("[data-open-task]").forEach(btn => {
    btn.addEventListener("click", () => openTask(btn.dataset.run, btn.dataset.openTask));
  });
}

function runButton(r) {
  const title = [
    r.run_id,
    r.tag ? `tag=${r.tag}` : "",
    (r.levers || []).length ? `levers=${(r.levers || []).join(", ")}` : "",
  ].filter(Boolean).join("\n");
  return `<button class="run-cell" data-open-run="${esc(r.run_id)}" title="${esc(title)}">
    <div class="run-name">${esc(r.run_id)}</div>
    <div class="run-tags">
      ${pill(r.status, r.status === "finished" ? "good" : "blue")}
      ${pill(`${fmtSmall(r.completed)}/${fmtSmall(r.n_tasks)}`)}
      ${r.terminal_log ? pill("log") : ""}
      ${(r.levers || []).length ? pill(`${r.levers.length} levers`) : ""}
    </div>
  </button>`;
}

function matrixCell(run, taskId) {
  const cell = (run.cells || {})[taskId];
  if (!cell) {
    return `<td><button class="score-cell missing" data-run="${esc(run.run_id)}" data-open-task="${esc(taskId)}" title="${esc(taskId)}">-</button></td>`;
  }
  const cls = cell.status === "running" ? "running" : scoreClass(cell.score);
  const meta = taskMeta(taskId);
  const live = cell.live || null;
  const title = [
    `${run.run_id} / ${taskId}`,
    `score=${fmtScore(cell.score)}`,
    `status=${cell.status}`,
    live ? `live: ${live.phase || "?"} step ${live.step != null ? live.step : "?"}` : "",
    cell.elapsed !== undefined && cell.elapsed !== null ? `elapsed=${fmtSmall(cell.elapsed)}s` : "",
    meta.question || "",
  ].filter(Boolean).join("\n");
  return `<td><button class="score-cell ${cls}" data-run="${esc(run.run_id)}" data-open-task="${esc(taskId)}" title="${esc(title)}">${esc(fmtCellScore(cell.score) || (cell.status === "running" ? "..." : "-"))}</button></td>`;
}

async function openRun(runId, opts = {}) {
  state.selectedRun = runId;
  state.selectedTask = null;
  state.taskDetail = null;
  state.runDetail = await api(`/api/runs/${encodeURIComponent(runId)}`);
  state.view = "run";
  render();
  if (opts.push !== false) updateRoute({view: "run", run: runId});
}

async function openTask(runId, taskId, opts = {}) {
  state.selectedRun = runId;
  state.selectedTask = taskId;
  state.taskDetail = await api(`/api/runs/${encodeURIComponent(runId)}/tasks/${encodeURIComponent(taskId)}`);
  state.view = "task";
  state.tab = opts.tab || "overview";
  render();
  if (opts.push !== false) {
    updateRoute({view: "task", run: runId, task: taskId, tab: state.tab});
  }
}

function renderRun() {
  const run = state.runDetail?.run;
  const tasks = state.runDetail?.tasks || [];
  if (!run) {
    $("view").innerHTML = `<div class="empty">Run not loaded.</div>`;
    return;
  }
  const rows = tasks.map(t => {
    const done = t.has_prediction || t.has_trace;
    const liveTxt = t.score != null ? "scored"
      : (done ? "done"
        : (t.live ? `${t.live.phase || "?"} · step ${t.live.step != null ? t.live.step : "?"}` : "running"));
    const liveCls = (t.score != null || done) ? "good" : "warn";
    return `
    <tr class="task-row" data-run-task="${esc(t.task_id)}">
      <td><strong>${esc(t.task_id)}</strong></td>
      <td>${pill(`score ${fmtScore(t.score)}`, scoreClass(t.score))}</td>
      <td>${pill(liveTxt, liveCls)}</td>
      <td>${fmtSmall(t.elapsed)}s</td>
      <td>${t.has_prediction ? pill("prediction", "good") : pill("no prediction", "bad")}</td>
      <td>${t.has_trace ? pill("trace", "good") : pill("no trace", "mid")}</td>
      <td>${t.anti_agg_label ? esc(t.anti_agg_label) : "-"}</td>
      <td>${t.asr_status ? esc(t.asr_status) : "-"}</td>
      <td>${esc(t.question || "")}</td>
    </tr>`;
  }).join("");
  $("view").innerHTML = `
    <section class="detail-page">
      ${detailHead({
        title: run.run_id,
        subtitle: "Run detail",
        back: "Matrix",
        metrics: [
          ["Status", run.status],
          ["Mean", fmtScore(run.mean_score)],
          ["Progress", `${fmtSmall(run.completed)} / ${fmtSmall(run.n_tasks)}`],
          ["Perfect", fmtSmall(run.n_perfect)],
          ["Zero", fmtSmall(run.n_zero)],
          ["Elapsed", `${fmtSmall(run.elapsed_minutes)}m`],
        ],
      })}
      <div></div>
      <div class="run-task-table-wrap">
        <table class="task-table">
          <thead><tr><th>Task</th><th>Score</th><th>Progress</th><th>Elapsed</th><th>Pred</th><th>Trace</th><th>Anti agg</th><th>ASR</th><th>Question</th></tr></thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
    </section>`;
  wireBack();
  document.querySelectorAll("[data-run-task]").forEach(row => {
    row.addEventListener("click", () => openTask(run.run_id, row.dataset.runTask));
  });
}

function detailHead({title, subtitle, back, metrics}) {
  const metricHtml = (metrics || []).map(([label, value]) => `
    <div class="metric"><span class="metric-label">${esc(label)}</span><span class="metric-value">${esc(value)}</span></div>`
  ).join("");
  return `<header class="detail-head">
    <div class="detail-actions">
      <button class="back-button" data-back>${esc(back)}</button>
      ${state.view === "task" ? `<button class="back-button" data-back-run>Run</button>` : ""}
    </div>
    <div class="detail-title">${esc(title)}</div>
    <div class="detail-sub">${esc(subtitle || "")}</div>
    <div class="metrics">${metricHtml}</div>
  </header>`;
}

function wireBack() {
  document.querySelectorAll("[data-back]").forEach(btn => {
    btn.addEventListener("click", () => {
      state.view = "matrix";
      state.selectedRun = null;
      state.selectedTask = null;
      state.runDetail = null;
      state.taskDetail = null;
      state.tab = "overview";
      render();
      updateRoute({view: "matrix"});
    });
  });
  document.querySelectorAll("[data-back-run]").forEach(btn => {
    btn.addEventListener("click", async () => {
      await openRun(state.selectedRun);
    });
  });
}

function renderTask() {
  const d = state.taskDetail;
  if (!d) {
    $("view").innerHTML = `<div class="empty">Task not loaded.</div>`;
    return;
  }
  const t = d.task || {};
  $("view").innerHTML = `
    <section class="detail-page">
      ${detailHead({
        title: `${d.run_id} / ${d.task_id}`,
        subtitle: d.question || "",
        back: "Matrix",
        metrics: [
          ["Score", fmtScore(t.score)],
          ["Elapsed", `${fmtSmall(t.elapsed)}s`],
          ["n_ok", fmtSmall(t.n_ok)],
          ["Prediction", d.prediction?.exists ? "yes" : "no"],
          ["Gold", d.gold?.exists ? "yes" : "no"],
          ["Trace", d.trace_meta?.exists ? "yes" : "no"],
        ],
      })}
      <nav class="tabs">
        ${["overview", "input", "trace", "output", "raw"].map(tab =>
          `<button class="tab ${state.tab === tab ? "active" : ""}" data-tab="${tab}">${tab}</button>`
        ).join("")}
      </nav>
      <section id="content" class="content"></section>
    </section>`;
  wireBack();
  document.querySelectorAll("[data-tab]").forEach(btn => {
    btn.addEventListener("click", () => {
      state.tab = btn.dataset.tab;
      renderTaskContent();
      document.querySelectorAll("[data-tab]").forEach(b => b.classList.toggle("active", b.dataset.tab === state.tab));
      updateRoute({
        view: "task",
        run: state.selectedRun,
        task: state.selectedTask,
        tab: state.tab,
      });
    });
  });
  renderTaskContent();
}

function renderTaskContent() {
  if (state.tab === "overview") renderOverview();
  if (state.tab === "input") renderInput();
  if (state.tab === "trace") renderTrace();
  if (state.tab === "output") renderOutput();
  if (state.tab === "raw") renderRaw();
}

function finalSqlBlock(d) {
  const last = d.trace_meta && d.trace_meta.last_answer_sql;
  if (!last || !last.sql) {
    return `<div class="block"><div class="block-title">Final SQL</div><div class="block-body"><div class="empty">No answer_from_sql step found.</div></div></div>`;
  }
  const route = last.source_route || {};
  return `<div class="block">
    <div class="block-title">Final SQL</div>
    <div class="block-body">
      <div class="metrics">
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

function csvBlock(title, csv) {
  if (!csv || !csv.exists) {
    return `<div class="block"><div class="block-title">${esc(title)}</div><div class="block-body"><div class="empty">Missing.</div></div></div>`;
  }
  const head = (csv.columns || []).map(c => `<th>${esc(c)}</th>`).join("");
  const rows = (csv.rows || []).map(r => `<tr>${r.map(c => `<td>${esc(c)}</td>`).join("")}</tr>`).join("");
  return `<div class="block">
    <div class="block-title">${esc(title)} ${csv.truncated ? "(truncated)" : ""}</div>
    <div class="block-body"><div class="table-wrap"><table class="preview"><thead><tr>${head}</tr></thead><tbody>${rows}</tbody></table></div></div>
  </div>`;
}

function renderOverview() {
  const d = state.taskDetail;
  const files = (d.artifacts.files || []).map(f => `<tr><td>${esc(f.name)}</td><td>${fmtSmall(f.size)}</td><td>${esc(f.mtime_iso)}</td></tr>`).join("");
  $("content").innerHTML = `
    <div class="grid">
      <div class="block">
        <div class="block-title">Question</div>
        <div class="block-body">
          <p class="question">${esc(d.question || "Question not found.")}</p>
          <pre>${esc(d.paths.task_dir)}</pre>
        </div>
      </div>
      <div class="block">
        <div class="block-title">Artifacts</div>
        <div class="block-body"><div class="table-wrap"><table class="preview"><thead><tr><th>File</th><th>Bytes</th><th>Modified</th></tr></thead><tbody>${files}</tbody></table></div></div>
      </div>
    </div>
    <div style="margin-top:14px">${finalSqlBlock(d)}</div>
    <div class="two-col" style="margin-top:14px">${csvBlock("Prediction", d.prediction)}${csvBlock("Gold", d.gold)}</div>`;
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
      <div class="block-body">${snapshot ? `<pre>${esc(JSON.stringify(snapshot, null, 2))}</pre>` : `<div class="empty">input_snapshot.json was not saved for this run.</div>`}</div>
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
  $("content").innerHTML = `
    <div class="metrics" style="margin-bottom:12px">
      <label class="toggle"><input id="thinkingToggle" type="checkbox" ${state.showThinking ? "checked" : ""}> Show thinking</label>
      ${pill(`${attempts.length} attempts`)}
      ${d.trace_meta && d.trace_meta.is_truncated_trace ? pill("preview trace", "mid") : ""}
    </div>
    ${attempts.map(renderAttempt).join("") || (d.steps_log && d.steps_log.exists
      ? `<div class="block"><div class="block-title">Live step log &mdash; trace.json not written yet (task still running). Turn on Auto refresh to follow.</div><div class="block-body"><pre>${esc(d.steps_log.text)}</pre></div></div>`
      : `<div class="empty">No trace.json found.</div>`)}`;
  $("thinkingToggle").addEventListener("change", e => {
    state.showThinking = e.target.checked;
    renderTrace();
  });
}

function renderAttempt(attempt, idx) {
  const steps = attempt.steps || [];
  return `<div class="block" style="margin-bottom:14px">
    <div class="block-title">Attempt ${esc(attempt.attempt ?? idx)} ${attempt.failure_reason ? "failure: " + esc(attempt.failure_reason) : ""}</div>
    <div class="block-body">${steps.map(renderStep).join("")}</div>
  </div>`;
}

function renderStep(step) {
  const obs = step.observation || {};
  const input = step.action_input || {};
  const thought = step.thought || "";
  const ok = obs.ok ?? step.observation_ok;
  return `<div class="step">
    <div class="step-head">
      <span class="step-title">Step ${esc(step.i ?? step.step_index ?? "-")}</span>
      ${pill(String(step.phase || "-"))}
      ${pill(String(step.action || "-"))}
      ${ok === false ? pill("not ok", "bad") : pill("ok", "good")}
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
    <div class="two-col" style="margin-top:14px">${csvBlock("Prediction", d.prediction)}${csvBlock("Gold", d.gold)}</div>`;
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

$("refreshBtn").addEventListener("click", () => loadMatrix().catch(showError));
$("runSearch").addEventListener("input", () => {
  if (state.view === "matrix") renderMatrix();
});
$("autoRefresh").addEventListener("change", setupAutoRefresh);

async function refreshCurrentSilent() {
  try {
    if (state.view === "task" && state.selectedRun && state.selectedTask) {
      const stillRunning = !(state.taskDetail && Array.isArray(state.taskDetail.trace) && state.taskDetail.trace.length);
      state.taskDetail = await api(`/api/runs/${encodeURIComponent(state.selectedRun)}/tasks/${encodeURIComponent(state.selectedTask)}`);
      // Only re-render live (running) tasks so we don't disrupt reading a finished one.
      if (stillRunning) renderTaskContent();
    } else if (state.view === "run" && state.selectedRun) {
      state.runDetail = await api(`/api/runs/${encodeURIComponent(state.selectedRun)}`);
      renderRun();
    }
  } catch (e) { console.error(e); }
}

function setupAutoRefresh() {
  if (state.autoTimer) clearInterval(state.autoTimer);
  state.autoTimer = null;
  if ($("autoRefresh").checked) {
    state.autoTimer = setInterval(() => {
      if (state.view === "matrix") loadMatrix({silent: true}).catch(console.error);
      else refreshCurrentSilent();
    }, 15000);
  }
}

function showError(err) {
  $("view").innerHTML = `<div class="empty">${esc(err.stack || err.message || err)}</div>`;
}

const TOKEN_TTL_MS = Number("__TOKEN_TTL_MS__") || 0;
const currentUrl = new URL(window.location.href);
const params = currentUrl.searchParams;
const nowMs = Date.now();
const storedExpires = Number(window.localStorage.getItem("traceViewerV2TokenExpiresAt") || "0");
if (storedExpires && storedExpires <= nowMs) {
  window.localStorage.removeItem("traceViewerV2Token");
  window.localStorage.removeItem("traceViewerV2TokenExpiresAt");
}
const urlToken = params.get("token");
state.authToken = urlToken || window.localStorage.getItem("traceViewerV2Token");
if (urlToken) {
  window.localStorage.setItem("traceViewerV2Token", urlToken);
  if (TOKEN_TTL_MS > 0) {
    window.localStorage.setItem("traceViewerV2TokenExpiresAt", String(nowMs + TOKEN_TTL_MS));
  } else {
    window.localStorage.removeItem("traceViewerV2TokenExpiresAt");
  }
  currentUrl.searchParams.delete("token");
  window.history.replaceState({}, "", currentUrl.pathname + currentUrl.search + currentUrl.hash);
}

setupAutoRefresh();
window.addEventListener("popstate", () => {
  applyRouteFromLocation().catch(showError);
});
loadMatrix({silent: true})
  .then(() => applyRouteFromLocation())
  .catch(showError);
"""


class TraceViewerV2Handler(BaseHTTPRequestHandler):
    server_version = "KobushiTraceViewerV2/0.1"

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
            "default-src 'self'; script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
            "base-uri 'none'; frame-ancestors 'none'",
        )
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=v1._json_default).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def _error(self, status: int, message: str) -> None:
        self._json({"error": message}, status)

    def _access_email(self) -> str | None:
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
            self._error(HTTPStatus.UNAUTHORIZED, "missing Cloudflare Access user header")
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
                self._json({"ok": True, "service": "trace_viewer_v2"})
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
                self._json(
                    {
                        "email": self._access_email(),
                        "cf_access_required": bool(getattr(self.server, "require_cf_access", False)),
                        "token_required": bool(getattr(self.server, "auth_token", None)),
                        "token_ttl_seconds": getattr(self.server, "auth_token_ttl_seconds", 0.0),
                        "token_expires_at": getattr(self.server, "auth_token_expires_at", None),
                    }
                )
                return
            if path == "/api/matrix":
                self._json(get_matrix())
                return
            if path == "/api/runs":
                self._json(v1.list_runs())
                return
            parts = [unquote(p) for p in path.split("/") if p]
            if len(parts) == 3 and parts[0] == "api" and parts[1] == "runs":
                self._json(v1.get_run(parts[2]))
                return
            if len(parts) == 5 and parts[0] == "api" and parts[1] == "runs" and parts[3] == "tasks":
                self._json(v1.get_task_detail(parts[2], parts[4]))
                return
            if len(parts) == 6 and parts[0] == "api" and parts[1] == "runs" and parts[3] == "tasks" and parts[5] == "file":
                qs = parse_qs(parsed.query)
                name = (qs.get("name") or [""])[0]
                self._json(v1.get_file(parts[2], parts[4], name))
                return
            self._error(HTTPStatus.NOT_FOUND, "not found")
        except FileNotFoundError as exc:
            self._error(HTTPStatus.NOT_FOUND, str(exc))
        except ValueError as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, repr(exc))


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a matrix-first Kobushi trace viewer.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8776)
    parser.add_argument("--require-cf-access", action="store_true")
    parser.add_argument("--access-email", action="append", default=[])
    parser.add_argument("--access-email-domain", action="append", default=[])
    parser.add_argument("--auth-token", default=os.environ.get("TRACE_VIEWER_AUTH_TOKEN"))
    parser.add_argument(
        "--auth-token-ttl-hours",
        type=float,
        default=float(os.environ.get("TRACE_VIEWER_AUTH_TOKEN_TTL_HOURS", "0") or 0),
    )
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), TraceViewerV2Handler)
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

    print(f"Trace viewer v2: http://{args.host}:{args.port}", flush=True)
    print(f"Runs root: {v1.RUNS_ROOT}", flush=True)
    if args.auth_token:
        print("Trace viewer v2 token check: enabled", flush=True)
        if server.auth_token_expires_at is not None:
            print(f"Trace viewer v2 token TTL: {args.auth_token_ttl_hours:g}h", flush=True)
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
