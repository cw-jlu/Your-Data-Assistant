#!/usr/bin/env python3
"""Build a single-file HTML dashboard for ReAct trace inspection.

Discovers every `artifacts/eval_full_<run>/output/` directory under the
project root, reads per-task trace.json + gold.csv + prediction.csv +
task.json, scores predictions with mock_scorer, and emits a self-contained
HTML at `artifacts/dashboard/dashboard.html` (no external assets, opens
directly via file://).

Usage:
    uv run python scripts/build_dashboard.py
    uv run python scripts/build_dashboard.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from data_agent_baseline.benchmark.schema import AnswerTable
from data_agent_baseline.scoring.mock_scorer import _read_csv_table, score_one

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
INPUT_ROOT = PROJECT_ROOT / "data" / "public" / "input"
GOLD_ROOT = PROJECT_ROOT / "data" / "public" / "output"
OUTPUT_DIR = ARTIFACTS_ROOT / "dashboard"
OUTPUT_HTML = OUTPUT_DIR / "dashboard.html"

OBSERVATION_CONTENT_LIMIT = 2048


@dataclass
class TaskRecord:
    task_id: str
    difficulty: str
    question: str
    status: str
    score: float
    recall: float
    matched_columns: int
    gold_columns: int
    predicted_columns: int
    elapsed_seconds: float | None
    succeeded: bool
    failure_reason: str | None
    steps_count: int
    gold: dict[str, Any]
    prediction: dict[str, Any] | None
    steps: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class RunRecord:
    run: str
    label: str
    summary: dict[str, Any]
    tasks: list[TaskRecord]


def _read_task_json(task_id: str) -> dict[str, Any]:
    p = INPUT_ROOT / task_id / "task.json"
    if not p.exists():
        return {"task_id": task_id, "difficulty": "?", "question": ""}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {"task_id": task_id, "difficulty": "?", "question": "(unreadable task.json)"}


def _read_gold(task_id: str) -> AnswerTable | None:
    p = GOLD_ROOT / task_id / "gold.csv"
    if not p.exists():
        return None
    try:
        return _read_csv_table(p)
    except Exception:
        return None


def _read_trace(trace_path: Path) -> dict[str, Any] | None:
    if not trace_path.exists():
        return None
    try:
        return json.loads(trace_path.read_text())
    except Exception:
        return None


def _read_prediction(pred_path: Path) -> AnswerTable | None:
    if not pred_path.exists():
        return None
    try:
        return _read_csv_table(pred_path)
    except Exception:
        return None


def _table_to_dict(table: AnswerTable | None) -> dict[str, Any] | None:
    if table is None:
        return None
    return {"columns": list(table.columns), "rows": [list(r) for r in table.rows]}


def _truncate(s: str, limit: int) -> tuple[str, bool]:
    if s is None:
        return "", False
    if len(s) <= limit:
        return s, False
    return s[:limit], True


def _serialize_steps(raw_steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, step in enumerate(raw_steps or []):
        thought = step.get("thought") or ""
        action = step.get("action") or ""
        action_input = step.get("action_input")
        if isinstance(action_input, dict):
            action_input_str = json.dumps(action_input, indent=2, ensure_ascii=False)
        else:
            action_input_str = str(action_input) if action_input is not None else ""

        obs = step.get("observation") or {}
        if isinstance(obs, dict):
            obs_tool = obs.get("tool") or action
            obs_ok = obs.get("ok")
            obs_content_raw = obs.get("content")
            if isinstance(obs_content_raw, (dict, list)):
                obs_content_str = json.dumps(obs_content_raw, indent=2, ensure_ascii=False)
            else:
                obs_content_str = "" if obs_content_raw is None else str(obs_content_raw)
        else:
            obs_tool = action
            obs_ok = None
            obs_content_str = "" if obs is None else str(obs)

        truncated_content, truncated = _truncate(obs_content_str, OBSERVATION_CONTENT_LIMIT)
        out.append(
            {
                "i": step.get("step_index", i + 1),
                "thought": thought,
                "action": action,
                "action_input": action_input_str,
                "ok": step.get("ok"),
                "raw_response": step.get("raw_response", "")[:200] if step.get("raw_response") else "",
                "observation": {
                    "tool": obs_tool,
                    "ok": obs_ok,
                    "content": truncated_content,
                    "truncated": truncated,
                    "full_size": len(obs_content_str),
                },
            }
        )
    return out


def _decide_status(
    *,
    prediction: AnswerTable | None,
    score: float,
    failure_reason: str | None,
) -> str:
    if prediction is None:
        if failure_reason and ("timed out" in failure_reason.lower() or "timeout" in failure_reason.lower()):
            return "timeout"
        return "missing"
    if score >= 0.999:
        return "perfect"
    if score > 0.0:
        return "partial"
    if failure_reason and ("timed out" in failure_reason.lower() or "timeout" in failure_reason.lower()):
        return "timeout"
    return "zero"


def _load_task(run: str, task_dir: Path) -> TaskRecord | None:
    task_id = task_dir.name
    task_info = _read_task_json(task_id)
    gold = _read_gold(task_id)
    if gold is None:
        # gold missing — skip this task (cannot score)
        return None
    trace = _read_trace(task_dir / "trace.json")
    prediction = _read_prediction(task_dir / "prediction.csv")

    if prediction is not None:
        score, recall, matched, gold_cols, pred_cols = score_one(
            prediction=prediction, gold=gold
        )
    else:
        score, recall, matched, gold_cols, pred_cols = 0.0, 0.0, 0, len(gold.columns), 0

    failure_reason = trace.get("failure_reason") if trace else None
    succeeded = bool(trace.get("succeeded")) if trace else False
    elapsed = trace.get("e2e_elapsed_seconds") if trace else None
    raw_steps = trace.get("steps") or [] if trace else []
    serialized_steps = _serialize_steps(raw_steps)
    status = _decide_status(prediction=prediction, score=score, failure_reason=failure_reason)

    return TaskRecord(
        task_id=task_id,
        difficulty=task_info.get("difficulty", "?"),
        question=task_info.get("question", ""),
        status=status,
        score=round(score, 4),
        recall=round(recall, 4),
        matched_columns=matched,
        gold_columns=gold_cols,
        predicted_columns=pred_cols,
        elapsed_seconds=elapsed,
        succeeded=succeeded,
        failure_reason=failure_reason,
        steps_count=len(raw_steps),
        gold=_table_to_dict(gold) or {"columns": [], "rows": []},
        prediction=_table_to_dict(prediction),
        steps=serialized_steps,
    )


def _natural_task_key(task_id: str) -> int:
    try:
        return int(task_id.split("_")[1])
    except (IndexError, ValueError):
        return 10**9


def discover_runs() -> list[Path]:
    if not ARTIFACTS_ROOT.exists():
        return []
    runs: list[Path] = []
    for path in sorted(ARTIFACTS_ROOT.glob("eval_full_*")):
        if not path.is_dir():
            continue
        output_dir = path / "output"
        if not output_dir.is_dir():
            continue
        task_dirs = [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("task_")]
        if not task_dirs:
            continue
        runs.append(path)
    return runs


def _run_label(run_path: Path) -> str:
    # eval_full_v5 → v5 (shipped)
    name = run_path.name.replace("eval_full_", "")
    return name


def _summarize(tasks: list[TaskRecord]) -> dict[str, Any]:
    perfect = sum(1 for t in tasks if t.status == "perfect")
    partial = sum(1 for t in tasks if t.status == "partial")
    zero = sum(1 for t in tasks if t.status == "zero")
    timeout = sum(1 for t in tasks if t.status == "timeout")
    missing = sum(1 for t in tasks if t.status == "missing")
    total = len(tasks)
    scored = [t.score for t in tasks if t.prediction is not None]
    mean_score = sum(scored) / total if total else 0.0
    return {
        "total": total,
        "perfect": perfect,
        "partial": partial,
        "zero": zero,
        "timeout": timeout,
        "missing": missing,
        "mean_score": round(mean_score, 4),
    }


def build_run(run_path: Path) -> RunRecord:
    output_dir = run_path / "output"
    task_dirs = sorted(
        [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("task_")],
        key=lambda d: _natural_task_key(d.name),
    )
    tasks: list[TaskRecord] = []
    for td in task_dirs:
        rec = _load_task(run_path.name, td)
        if rec is not None:
            tasks.append(rec)
    return RunRecord(
        run=run_path.name,
        label=_run_label(run_path),
        summary=_summarize(tasks),
        tasks=tasks,
    )


def _record_to_dict(t: TaskRecord) -> dict[str, Any]:
    return {
        "task_id": t.task_id,
        "difficulty": t.difficulty,
        "question": t.question,
        "status": t.status,
        "score": t.score,
        "recall": t.recall,
        "matched_columns": t.matched_columns,
        "gold_columns": t.gold_columns,
        "predicted_columns": t.predicted_columns,
        "elapsed_seconds": t.elapsed_seconds,
        "succeeded": t.succeeded,
        "failure_reason": t.failure_reason,
        "steps_count": t.steps_count,
        "gold": t.gold,
        "prediction": t.prediction,
        "steps": t.steps,
    }


def assemble_payload(runs: list[RunRecord]) -> dict[str, Any]:
    runs_map: dict[str, Any] = {}
    for r in runs:
        runs_map[r.label] = {
            "label": r.label,
            "summary": r.summary,
            "tasks": [_record_to_dict(t) for t in r.tasks],
        }
    default_run = runs[-1].label if runs else ""
    return {"runs": runs_map, "default_run": default_run}


_HTML_TEMPLATE = """<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=1200" />
<title>Trace Dashboard</title>
<style>
  :root {
    --bg: #f5f5f4;
    --card: #ffffff;
    --border: #e5e5e5;
    --text: #1f2937;
    --muted: #6b7280;
    --perfect: #16a34a;
    --partial: #ca8a04;
    --zero: #dc2626;
    --timeout: #ea580c;
    --missing: #6b7280;
  }
  html, body {
    margin: 0;
    padding: 0;
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    font-size: 14px;
  }
  .wrap {
    max-width: 1400px;
    margin: 0 auto;
    padding: 16px;
  }
  h1 { font-size: 18px; margin: 0; }
  .topbar {
    display: flex;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
    padding: 12px 0;
    border-bottom: 1px solid var(--border);
  }
  .topbar select, .topbar input {
    padding: 4px 8px;
    border: 1px solid var(--border);
    border-radius: 4px;
    background: var(--card);
    font-size: 14px;
  }
  .stats {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 8px;
    margin: 12px 0;
  }
  .stat {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 10px 12px;
  }
  .stat .label { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; }
  .stat .value { font-size: 22px; font-weight: 600; }
  .stat.perfect .value { color: var(--perfect); }
  .stat.partial .value { color: var(--partial); }
  .stat.zero .value { color: var(--zero); }
  .stat.timeout .value { color: var(--timeout); }
  .stat.missing .value { color: var(--missing); }
  .filters {
    display: flex;
    align-items: center;
    gap: 16px;
    flex-wrap: wrap;
    padding: 8px 0;
    margin-bottom: 8px;
  }
  .filters label { font-size: 13px; }
  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
    gap: 8px;
    margin-top: 8px;
  }
  .card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 10px;
    cursor: pointer;
    transition: box-shadow 0.1s;
  }
  .card:hover { box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
  .card .head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 4px;
  }
  .card .tid { font-weight: 600; font-size: 13px; }
  .badge {
    font-size: 10px;
    padding: 1px 6px;
    border-radius: 3px;
    background: #f3f4f6;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.3px;
  }
  .badge.easy { background: #dcfce7; color: #166534; }
  .badge.medium { background: #fef3c7; color: #854d0e; }
  .badge.hard { background: #fed7aa; color: #9a3412; }
  .badge.extreme { background: #fecaca; color: #991b1b; }
  .indicator {
    font-size: 22px;
    line-height: 1;
    margin: 4px 0;
  }
  .indicator.perfect { color: var(--perfect); }
  .indicator.partial { color: var(--partial); }
  .indicator.zero { color: var(--zero); }
  .indicator.timeout { color: var(--timeout); }
  .indicator.missing { color: var(--missing); }
  .card .meta {
    font-size: 11px;
    color: var(--muted);
    margin-top: 2px;
  }
  .detail {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 14px;
    margin: 8px 0 16px;
  }
  .detail h2 { font-size: 16px; margin: 0 0 8px; }
  .detail .question {
    background: #f9fafb;
    border-left: 3px solid #3b82f6;
    padding: 8px 12px;
    margin: 8px 0;
    border-radius: 0 4px 4px 0;
  }
  .detail .failure {
    background: #fef2f2;
    border-left: 3px solid var(--zero);
    padding: 8px 12px;
    margin: 8px 0;
    border-radius: 0 4px 4px 0;
    color: #7f1d1d;
  }
  .compare {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin: 12px 0;
  }
  .table-wrap {
    border: 1px solid var(--border);
    border-radius: 4px;
    overflow: hidden;
    max-height: 320px;
    overflow-y: auto;
  }
  .table-wrap > .table-head {
    background: #f3f4f6;
    padding: 6px 10px;
    font-weight: 600;
    font-size: 12px;
    border-bottom: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
  }
  table.compare-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
  }
  table.compare-table th, table.compare-table td {
    padding: 4px 8px;
    border-bottom: 1px solid #f1f5f9;
    text-align: left;
    white-space: nowrap;
  }
  table.compare-table th { background: #fafafa; font-weight: 600; }
  .empty-pred {
    color: var(--muted);
    font-style: italic;
    padding: 16px;
    text-align: center;
  }
  .step-list { margin-top: 12px; }
  .step {
    border: 1px solid var(--border);
    border-radius: 4px;
    margin-bottom: 6px;
    overflow: hidden;
  }
  .step .step-head {
    background: #f9fafb;
    padding: 6px 12px;
    cursor: pointer;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 13px;
  }
  .step .step-head .action { font-weight: 600; }
  .step .step-head .preview {
    color: var(--muted);
    margin-left: 8px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    flex: 1;
    max-width: 600px;
  }
  .step .step-body {
    display: none;
    padding: 8px 12px;
    background: #fff;
    font-size: 12px;
  }
  .step.open .step-body { display: block; }
  .step .field { margin-bottom: 8px; }
  .step .field .key {
    font-weight: 600;
    color: var(--muted);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.3px;
    margin-bottom: 2px;
  }
  .step pre {
    background: #f9fafb;
    border: 1px solid #f1f5f9;
    border-radius: 3px;
    padding: 6px 8px;
    overflow-x: auto;
    font-family: ui-monospace, "SF Mono", Menlo, monospace;
    font-size: 11.5px;
    line-height: 1.4;
    white-space: pre-wrap;
    word-break: break-word;
    margin: 0;
  }
  .obs-ok { color: var(--perfect); }
  .obs-fail { color: var(--zero); }
  .truncated-marker {
    color: var(--muted);
    font-style: italic;
    font-size: 11px;
    padding-top: 4px;
  }
  .hidden { display: none !important; }
  .empty-state { color: var(--muted); padding: 20px; text-align: center; }
</style>
</head>
<body>
<div class="wrap">
  <div class="topbar">
    <h1>Trace Dashboard</h1>
    <span style="color:var(--muted)">·</span>
    <label>Run: <select id="runSelect"></select></label>
    <span style="color:var(--muted)">·</span>
    <label>Search: <input id="searchBox" type="text" placeholder="task_id or keyword" style="width:220px" /></label>
    <label>Sort:
      <select id="sortBox">
        <option value="task_id">Task ID</option>
        <option value="score_desc">Score (high→low)</option>
        <option value="score_asc">Score (low→high)</option>
        <option value="elapsed_desc">Elapsed (slow→fast)</option>
        <option value="elapsed_asc">Elapsed (fast→slow)</option>
        <option value="steps_desc">Steps (many→few)</option>
      </select>
    </label>
    <span style="color:var(--muted);margin-left:auto" id="runMeta"></span>
  </div>

  <div class="stats" id="stats"></div>

  <div class="filters">
    <span>Difficulty:</span>
    <label><input type="checkbox" class="diff" value="easy" checked /> easy</label>
    <label><input type="checkbox" class="diff" value="medium" checked /> medium</label>
    <label><input type="checkbox" class="diff" value="hard" checked /> hard</label>
    <label><input type="checkbox" class="diff" value="extreme" checked /> extreme</label>
    <span style="color:var(--muted)">|</span>
    <span>Status:</span>
    <label><input type="checkbox" class="stat-f" value="perfect" checked /> perfect</label>
    <label><input type="checkbox" class="stat-f" value="partial" checked /> partial</label>
    <label><input type="checkbox" class="stat-f" value="zero" checked /> zero</label>
    <label><input type="checkbox" class="stat-f" value="timeout" checked /> timeout</label>
    <label><input type="checkbox" class="stat-f" value="missing" checked /> missing</label>
  </div>

  <div id="grid" class="grid"></div>
  <div id="detail"></div>
</div>

<script id="dashboard-data" type="application/json">__DASHBOARD_DATA__</script>
<script>
(function () {
  const RAW = document.getElementById('dashboard-data').textContent;
  const DATA = JSON.parse(RAW);
  const STATUS_SYMBOL = {perfect:'●', partial:'◐', zero:'✕', timeout:'⌛', missing:'–'};
  const STATUS_LABEL = {perfect:'Perfect', partial:'Partial', zero:'Zero', timeout:'Timeout', missing:'Missing'};

  const runSelect = document.getElementById('runSelect');
  const searchBox = document.getElementById('searchBox');
  const sortBox = document.getElementById('sortBox');
  const stats = document.getElementById('stats');
  const grid = document.getElementById('grid');
  const detail = document.getElementById('detail');
  const runMeta = document.getElementById('runMeta');

  const runs = Object.keys(DATA.runs || {});
  if (runs.length === 0) {
    grid.innerHTML = '<div class="empty-state">No eval runs found. Run a benchmark first, then re-build the dashboard.</div>';
    return;
  }
  runs.forEach(r => {
    const opt = document.createElement('option');
    opt.value = r;
    opt.textContent = r;
    runSelect.appendChild(opt);
  });
  runSelect.value = DATA.default_run && DATA.runs[DATA.default_run] ? DATA.default_run : runs[runs.length-1];

  let openTaskId = null;

  function currentRun() { return DATA.runs[runSelect.value]; }

  function renderStats() {
    const s = currentRun().summary;
    const items = [
      {key:'total', label:'Total', cls:''},
      {key:'perfect', label:'Perfect', cls:'perfect'},
      {key:'partial', label:'Partial', cls:'partial'},
      {key:'zero', label:'Zero', cls:'zero'},
      {key:'timeout', label:'Timeout', cls:'timeout'},
      {key:'missing', label:'Missing', cls:'missing'},
    ];
    stats.innerHTML = items.map(it => (
      `<div class="stat ${it.cls}"><div class="label">${it.label}</div><div class="value">${s[it.key] ?? 0}</div></div>`
    )).join('');
    runMeta.textContent = `mean score: ${s.mean_score.toFixed(4)}`;
  }

  function activeFilters() {
    const diffs = Array.from(document.querySelectorAll('.diff:checked')).map(e => e.value);
    const statuses = Array.from(document.querySelectorAll('.stat-f:checked')).map(e => e.value);
    return {diffs, statuses, q: searchBox.value.trim().toLowerCase(), sort: sortBox.value};
  }

  function applyFilters(tasks, f) {
    let out = tasks.filter(t => f.diffs.includes(t.difficulty) && f.statuses.includes(t.status));
    if (f.q) {
      out = out.filter(t => t.task_id.toLowerCase().includes(f.q) || t.question.toLowerCase().includes(f.q));
    }
    const cmp = {
      task_id: (a,b) => parseInt(a.task_id.split('_')[1]||0) - parseInt(b.task_id.split('_')[1]||0),
      score_desc: (a,b) => b.score - a.score,
      score_asc: (a,b) => a.score - b.score,
      elapsed_desc: (a,b) => (b.elapsed_seconds||0) - (a.elapsed_seconds||0),
      elapsed_asc: (a,b) => (a.elapsed_seconds||0) - (b.elapsed_seconds||0),
      steps_desc: (a,b) => b.steps_count - a.steps_count,
    };
    out.sort(cmp[f.sort] || cmp.task_id);
    return out;
  }

  function escapeHTML(s) {
    if (s == null) return '';
    return String(s).replace(/[&<>"']/g, c => ({
      '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'
    }[c]));
  }

  function fmtElapsed(s) {
    if (s == null) return '—';
    if (s < 60) return s.toFixed(1) + 's';
    const m = Math.floor(s/60), r = Math.round(s%60);
    return `${m}m ${r}s`;
  }

  function renderCard(t) {
    const sym = STATUS_SYMBOL[t.status] || '?';
    return (
      `<div class="card" data-task="${t.task_id}" title="${escapeHTML(t.question)}">
        <div class="head">
          <span class="tid">${t.task_id}</span>
          <span class="badge ${t.difficulty}">${t.difficulty}</span>
        </div>
        <div class="indicator ${t.status}">${sym}</div>
        <div class="meta">score ${t.score.toFixed(3)} · ${t.steps_count} step${t.steps_count===1?'':'s'} · ${fmtElapsed(t.elapsed_seconds)}</div>
      </div>`
    );
  }

  function renderTable(label, table, otherTable) {
    if (!table) {
      return `<div class="table-wrap"><div class="table-head"><span>${label}</span><span style="color:var(--muted)">—</span></div><div class="empty-pred">no prediction</div></div>`;
    }
    const cols = table.columns || [];
    const rows = table.rows || [];
    const head = `<tr>${cols.map(c => `<th>${escapeHTML(c)}</th>`).join('')}</tr>`;
    const body = rows.slice(0, 200).map(r => (
      `<tr>${(r||[]).map(c => `<td>${escapeHTML(c)}</td>`).join('')}</tr>`
    )).join('');
    const truncNote = rows.length > 200 ? `<div class="truncated-marker">... and ${rows.length-200} more rows</div>` : '';
    return (
      `<div class="table-wrap"><div class="table-head"><span>${label}</span><span style="color:var(--muted)">${cols.length} col${cols.length===1?'':'s'} · ${rows.length} row${rows.length===1?'':'s'}</span></div>
        <table class="compare-table"><thead>${head}</thead><tbody>${body}</tbody></table>${truncNote}
      </div>`
    );
  }

  function renderStep(step) {
    const okBadge = step.observation && step.observation.ok === true ? '<span class="obs-ok">✓</span>'
                  : step.observation && step.observation.ok === false ? '<span class="obs-fail">✕</span>' : '';
    const preview = (step.thought || '').replace(/\\s+/g, ' ').slice(0, 140);
    const truncMark = step.observation && step.observation.truncated
      ? `<div class="truncated-marker">... truncated at ${step.observation.content.length} chars (full size: ${step.observation.full_size})</div>`
      : '';
    return (
      `<div class="step" data-i="${step.i}">
        <div class="step-head">
          <div><span class="action">Step ${step.i} · ${escapeHTML(step.action)}</span> ${okBadge}<span class="preview"> ${escapeHTML(preview)}</span></div>
          <span style="color:var(--muted)">▼</span>
        </div>
        <div class="step-body">
          <div class="field"><div class="key">thought</div><pre>${escapeHTML(step.thought)}</pre></div>
          <div class="field"><div class="key">action_input</div><pre>${escapeHTML(step.action_input)}</pre></div>
          <div class="field"><div class="key">observation · ${escapeHTML(step.observation.tool)} ${okBadge}</div><pre>${escapeHTML(step.observation.content)}</pre>${truncMark}</div>
        </div>
      </div>`
    );
  }

  function renderDetail(t) {
    const matchInfo = `matched ${t.matched_columns}/${t.gold_columns} gold cols · ${t.predicted_columns} predicted cols`;
    const failure = t.failure_reason
      ? `<div class="failure"><strong>Failure:</strong> ${escapeHTML(t.failure_reason)}</div>`
      : '';
    const steps = (t.steps || []).map(renderStep).join('');
    const stepsBlock = steps
      ? `<div class="step-list">${steps}</div>`
      : `<div class="empty-state">no steps recorded</div>`;
    return (
      `<div class="detail">
        <h2>${t.task_id} <span class="badge ${t.difficulty}">${t.difficulty}</span> <span class="indicator ${t.status}" style="font-size:14px;vertical-align:middle">${STATUS_SYMBOL[t.status]}</span> <span style="color:var(--muted);font-weight:normal;font-size:13px"> · score ${t.score.toFixed(3)} · recall ${t.recall.toFixed(3)} · ${matchInfo} · ${t.steps_count} step${t.steps_count===1?'':'s'} · ${fmtElapsed(t.elapsed_seconds)}</span></h2>
        <div class="question"><strong>Question:</strong> ${escapeHTML(t.question)}</div>
        ${failure}
        <div class="compare">
          ${renderTable('Gold', t.gold)}
          ${renderTable('Prediction', t.prediction)}
        </div>
        <h3 style="font-size:13px;margin:14px 0 4px;color:var(--muted);text-transform:uppercase;letter-spacing:0.3px">Reasoning timeline</h3>
        ${stepsBlock}
      </div>`
    );
  }

  function render() {
    renderStats();
    const f = activeFilters();
    const tasks = applyFilters(currentRun().tasks, f);
    grid.innerHTML = tasks.map(renderCard).join('') || '<div class="empty-state">No tasks match the current filters.</div>';
    // restore open detail if still in filtered set
    if (openTaskId && !tasks.find(t => t.task_id === openTaskId)) {
      openTaskId = null;
      detail.innerHTML = '';
    } else if (openTaskId) {
      const t = tasks.find(t => t.task_id === openTaskId);
      detail.innerHTML = renderDetail(t);
    }
  }

  // event wiring
  runSelect.addEventListener('change', () => { openTaskId = null; detail.innerHTML = ''; render(); });
  searchBox.addEventListener('input', render);
  sortBox.addEventListener('change', render);
  document.querySelectorAll('.diff,.stat-f').forEach(el => el.addEventListener('change', render));

  grid.addEventListener('click', (e) => {
    const card = e.target.closest('.card');
    if (!card) return;
    const tid = card.dataset.task;
    if (openTaskId === tid) {
      openTaskId = null;
      detail.innerHTML = '';
      return;
    }
    openTaskId = tid;
    const t = currentRun().tasks.find(x => x.task_id === tid);
    detail.innerHTML = renderDetail(t);
    detail.scrollIntoView({behavior:'smooth', block:'start'});
  });

  detail.addEventListener('click', (e) => {
    const head = e.target.closest('.step-head');
    if (!head) return;
    head.parentElement.classList.toggle('open');
  });

  render();
})();
</script>
</body>
</html>
"""


def render_html(payload: dict[str, Any]) -> str:
    # Embed JSON safely — escape </script> sequences
    raw = json.dumps(payload, ensure_ascii=False)
    raw = raw.replace("</script>", "<\\/script>")
    return _HTML_TEMPLATE.replace("__DASHBOARD_DATA__", raw)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="discover runs only; do not write HTML")
    ap.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_HTML,
        help="output HTML path (default: artifacts/dashboard/dashboard.html)",
    )
    args = ap.parse_args()

    run_paths = discover_runs()
    if not run_paths:
        print("No eval run directories found under artifacts/eval_full_*/.", file=sys.stderr)
        return 1

    print(f"discovered {len(run_paths)} run(s):")
    runs: list[RunRecord] = []
    for run_path in run_paths:
        run = build_run(run_path)
        runs.append(run)
        s = run.summary
        print(
            f"  {run.label:<28}  total={s['total']:>3}  "
            f"perfect={s['perfect']:>3}  partial={s['partial']:>3}  "
            f"zero={s['zero']:>3}  timeout={s['timeout']:>3}  missing={s['missing']:>3}  "
            f"mean={s['mean_score']:.4f}"
        )

    if args.dry_run:
        return 0

    payload = assemble_payload(runs)
    html_str = render_html(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html_str, encoding="utf-8")
    size_kb = args.output.stat().st_size / 1024
    print(f"\nwrote {args.output}  ({size_kb:.1f} KB)")
    print(f"open with: open {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
