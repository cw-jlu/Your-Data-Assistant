"""Render a `trace.json` to a Rich console for human-readable debugging.

Schema assumed (matches `agents/runtime.AgentRunResult.to_dict`):

    {
      "task_id":            str,
      "answer":             {"columns": [...], "rows": [[...], ...]} | None,
      "normalized_answer":  same shape | None,
      "steps": [
          {"step_index": int, "thought": str, "action": str,
           "action_input": dict, "raw_response": str,
           "observation": {"ok": bool, "tool": str, "content": ...} | {...},
           "ok": bool},
          ...
      ],
      "failure_reason":      str | None,
      "succeeded":           bool,
      "e2e_elapsed_seconds": float | None,
    }

Optional task.json sidecar (looked up via `--input-root`) supplies
``difficulty`` and ``question``; render gracefully degrades when absent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.text import Text


_DEFAULT_PREVIEW_CHARS = 200


def _short(text: str, *, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _format_action_input(action: str, action_input: dict[str, Any], *, full: bool) -> str:
    """Compact one-line summary of action_input; folds long `code` blocks."""
    if not action_input:
        return "{}"
    if action == "execute_python" and "code" in action_input:
        code = str(action_input.get("code", ""))
        line_count = code.count("\n") + 1
        if full:
            return f"code=<{line_count} lines>\n{code}"
        first = code.splitlines()[0] if code else ""
        return f"code=<{line_count} lines: {_short(first, limit=80)}>"
    serialized = json.dumps(action_input, ensure_ascii=False)
    if full or len(serialized) <= 200:
        return serialized
    return _short(serialized, limit=200)


def _format_observation(observation: dict[str, Any], *, full: bool) -> str:
    """Best-effort one-line summary of an observation payload."""
    if not isinstance(observation, dict):
        return str(observation)
    ok = observation.get("ok")
    tool = observation.get("tool")
    content = observation.get("content")
    parts: list[str] = []
    if ok is not None:
        parts.append(f"ok={ok}")
    if tool is not None and tool != observation.get("error"):
        parts.append(f"tool={tool}")
    if isinstance(content, dict):
        if "status" in content:
            parts.append(f"status={content['status']}")
        if "stdout" in content:
            stdout = str(content["stdout"]).strip()
            if stdout:
                parts.append("stdout=" + _short(stdout.replace("\n", " | "), limit=160))
        if "rows" in content and isinstance(content["rows"], list):
            parts.append(f"rows={len(content['rows'])}")
        if "columns" in content and isinstance(content["columns"], list):
            cols = content["columns"]
            cols_str = json.dumps(cols, ensure_ascii=False)
            parts.append(f"columns={_short(cols_str, limit=120)}")
        if "warnings" in content and isinstance(content["warnings"], list):
            for w in content["warnings"]:
                code = w.get("code") if isinstance(w, dict) else None
                sev = w.get("severity") if isinstance(w, dict) else None
                if code:
                    parts.append(f"warn[{sev}]={code}")
        if "entries" in content and isinstance(content["entries"], list):
            parts.append(f"entries={len(content['entries'])}")
    elif isinstance(content, list):
        parts.append(f"len={len(content)}")
    elif content is not None:
        parts.append(f"content={_short(str(content), limit=120)}")
    if "error" in observation:
        parts.append(f"error={_short(str(observation['error']), limit=160)}")

    summary = " · ".join(parts) if parts else json.dumps(observation, ensure_ascii=False)
    if full:
        full_dump = json.dumps(observation, ensure_ascii=False, indent=2)
        return summary + "\n" + full_dump
    return summary


def _format_answer_table(answer: dict[str, Any] | None, *, full: bool) -> str:
    if not isinstance(answer, dict):
        return "(no answer)"
    cols = answer.get("columns") or []
    rows = answer.get("rows") or []
    head = f"columns={cols}, rows={len(rows)}"
    if not full or not rows:
        return head
    body_lines = [json.dumps(row, ensure_ascii=False) for row in rows[:10]]
    if len(rows) > 10:
        body_lines.append(f"… (+{len(rows) - 10} more rows)")
    return head + "\n" + "\n".join(body_lines)


def _step_marker(step: dict[str, Any]) -> tuple[str, str]:
    """Return (glyph, style) for the step header bullet."""
    obs = step.get("observation") or {}
    ok = step.get("ok")
    if isinstance(obs, dict):
        content = obs.get("content")
        if isinstance(content, dict) and content.get("status") == "validation_blocking":
            return ("⚠", "yellow")
    if ok is False:
        return ("✗", "red")
    return ("▸", "cyan")


def _load_task_metadata(input_root: Path | None, task_id: str) -> dict[str, Any]:
    """Optional task.json lookup. Returns {} on failure."""
    if input_root is None:
        return {}
    candidate = input_root / task_id / "task.json"
    if not candidate.is_file():
        return {}
    try:
        return json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _load_gold(gold_root: Path | None, task_id: str) -> dict[str, Any] | None:
    """Optional gold.csv lookup. Returns {'columns': [...], 'rows': [...]} or None."""
    if gold_root is None:
        return None
    path = gold_root / task_id / "gold.csv"
    if not path.is_file():
        return None
    try:
        import csv

        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)
    except OSError:
        return None
    if not rows:
        return None
    return {"columns": rows[0], "rows": rows[1:]}


def render_trace(
    trace: dict[str, Any],
    *,
    console: Console,
    full: bool = False,
    task_metadata: dict[str, Any] | None = None,
    gold: dict[str, Any] | None = None,
) -> None:
    """Render a parsed trace.json to ``console`` in step-by-step view."""
    task_id = trace.get("task_id", "?")
    succeeded = trace.get("succeeded")
    elapsed = trace.get("e2e_elapsed_seconds")
    failure_reason = trace.get("failure_reason")
    steps = trace.get("steps") or []

    meta = task_metadata or {}
    difficulty = meta.get("difficulty") or "?"
    question = meta.get("question") or "(no task.json supplied)"

    status_text = "✓ succeeded" if succeeded else "✗ failed"
    status_style = "green" if succeeded else "red"
    elapsed_str = f" · {elapsed:.1f}s" if isinstance(elapsed, (int, float)) else ""

    header = Text()
    header.append(f"{task_id} ", style="bold")
    header.append(f"({difficulty}) ", style="magenta")
    header.append(f"· {len(steps)} steps · ", style="dim")
    header.append(status_text, style=status_style)
    header.append(elapsed_str, style="dim")
    if failure_reason:
        header.append(f"\nfailure: {failure_reason}", style="red")
    header.append(f"\nQuestion: {question}", style="white")
    console.print(Panel(header, border_style="bright_blue"))

    for step in steps:
        glyph, glyph_style = _step_marker(step)
        idx = step.get("step_index", "?")
        action = step.get("action") or "?"
        ok = step.get("ok")
        ok_style = "green" if ok else ("red" if ok is False else "white")

        title = Text()
        title.append(f"{glyph} ", style=glyph_style)
        title.append(f"Step {idx} ", style="bold")
        title.append(f"· {action} ", style="cyan")
        title.append(f"· ok={ok}", style=ok_style)

        obs = step.get("observation") or {}
        if isinstance(obs, dict):
            content = obs.get("content")
            if isinstance(content, dict) and content.get("status") == "validation_blocking":
                title.append("  validation_blocking", style="yellow")

        body = Text()
        thought = (step.get("thought") or "").strip()
        if thought:
            body.append("thought  ", style="dim")
            body.append(_short(thought, limit=300 if not full else 100_000) + "\n")
        action_input = step.get("action_input") or {}
        body.append("in       ", style="dim")
        body.append(_format_action_input(action, action_input, full=full) + "\n")
        body.append("obs      ", style="dim")
        body.append(_format_observation(obs, full=full))
        if full:
            raw = step.get("raw_response") or ""
            if raw:
                body.append("\nraw      ", style="dim")
                body.append(_short(raw, limit=10_000))

        console.print(Panel(body, title=title, border_style="grey50", padding=(0, 1)))

    answer = trace.get("answer")
    normalized = trace.get("normalized_answer")
    if answer is not None or normalized is not None or gold is not None:
        ans_text = Text()
        if answer is not None:
            ans_text.append("answer (raw):        ", style="dim")
            ans_text.append(_format_answer_table(answer, full=full) + "\n")
        if normalized is not None and normalized != answer:
            ans_text.append("answer (normalized): ", style="dim")
            ans_text.append(_format_answer_table(normalized, full=full) + "\n")
        if gold is not None:
            ans_text.append("gold:                ", style="dim cyan")
            ans_text.append(_format_answer_table(gold, full=full))
        console.print(Panel(ans_text, title="Final answer", border_style="green"))


def render_trace_from_path(
    trace_path: Path,
    *,
    console: Console,
    full: bool = False,
    input_root: Path | None = None,
    gold_root: Path | None = None,
) -> None:
    """Convenience wrapper — load trace.json, look up sidecar metadata, render."""
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    task_id = trace.get("task_id") or trace_path.parent.name
    metadata = _load_task_metadata(input_root, task_id)
    gold = _load_gold(gold_root, task_id)
    render_trace(trace, console=console, full=full, task_metadata=metadata, gold=gold)
