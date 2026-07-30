#!/usr/bin/env python3
"""Render ETL trace JSON(s) as a self-contained HTML Gantt timeline.

Usage:
    uv run python scripts/view_etl_trace.py <trace.json or task_dir>

If given a directory, finds all *_trace.json files in _cache/.
Opens the result in the default browser.
"""

from __future__ import annotations

import html
import json
import sys
import webbrowser
from pathlib import Path

PHASE_COLORS = {
    "schema_parse": "#4a90d9",
    "schema_merge": "#5b9bd5",
    "schema_infer": "#6baed6",
    "compress": "#2171b5",
    "pre_merge": "#74c476",
    "fix_format": "#41ab5d",
    "reconcile": "#238b45",
    "normalize": "#66c2a4",
    "dedup": "#99d8c9",
    "retry": "#fc8d59",
    "verify": "#ef6548",
    "csv": "#78c679",
    "identity_repair": "#a1d99b",
    "synonym_unify": "#c7e9c0",
}
SKIPPED_COLOR = "#bdbdbd"
ERROR_COLOR = "#e31a1c"
DEFAULT_COLOR = "#9ecae1"

BAR_HEIGHT = 28
ROW_SPACING = 6
LEFT_LABEL_WIDTH = 130
SVG_PADDING = 20
CHART_WIDTH = 700


def _color(phase: dict) -> str:
    if phase.get("error"):
        return ERROR_COLOR
    if phase.get("skipped"):
        return SKIPPED_COLOR
    return PHASE_COLORS.get(phase["name"], DEFAULT_COLOR)


def _render_file(trace: dict) -> str:
    phases = trace.get("phases", [])
    if not phases:
        return "<p>No phases recorded.</p>"

    total_ms = trace.get("duration_ms", 1) or 1
    cumulative = 0
    positioned = []
    for p in phases:
        dur = p.get("duration_ms", 0)
        positioned.append((cumulative, dur, p))
        cumulative += dur

    scale = CHART_WIDTH / max(cumulative, 1) if cumulative else 1

    rows = len(positioned)
    svg_h = SVG_PADDING * 2 + rows * (BAR_HEIGHT + ROW_SPACING)
    svg_w = LEFT_LABEL_WIDTH + CHART_WIDTH + SVG_PADDING * 2

    parts = [f'<svg width="{svg_w}" height="{svg_h}" xmlns="http://www.w3.org/2000/svg">']
    parts.append("<style>text{font-family:monospace;font-size:12px;fill:#333}</style>")

    for i, (offset, dur, p) in enumerate(positioned):
        y = SVG_PADDING + i * (BAR_HEIGHT + ROW_SPACING)
        x = LEFT_LABEL_WIDTH + offset * scale
        w = max(dur * scale, 2)
        color = _color(p)
        name = html.escape(p["name"])
        label = f"{dur}ms"
        if p.get("llm_calls"):
            label += f" ({p['llm_calls']} LLM)"

        parts.append(f'<text x="4" y="{y + BAR_HEIGHT // 2 + 4}" font-weight="bold">{name}</text>')
        parts.append(
            f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="{BAR_HEIGHT}" '
            f'rx="3" fill="{color}" opacity="0.85"/>'
        )
        parts.append(f'<text x="{x + w + 4:.1f}" y="{y + BAR_HEIGHT // 2 + 4}">{label}</text>')

    parts.append("</svg>")

    details_parts = []
    for p in phases:
        name = html.escape(p["name"])
        metrics = p.get("metrics", {})
        decisions = p.get("decisions", {})
        error = p.get("error")
        inner = ""
        if error:
            inner += f'<div style="color:red">Error: {html.escape(str(error))}</div>'
        if metrics:
            inner += (
                "<strong>Metrics:</strong><pre>"
                + html.escape(json.dumps(metrics, indent=2, ensure_ascii=False))
                + "</pre>"
            )
        if decisions:
            inner += (
                "<strong>Decisions:</strong><pre>"
                + html.escape(json.dumps(decisions, indent=2, ensure_ascii=False))
                + "</pre>"
            )
        if p.get("input_size"):
            inner += f"<div>Input: {p['input_size']} chars"
            if p.get("output_size"):
                inner += f" → Output: {p['output_size']} chars"
            inner += "</div>"
        if inner:
            details_parts.append(
                f"<details><summary><b>{name}</b> — {p.get('duration_ms', 0)}ms</summary>{inner}</details>"
            )

    stem = html.escape(trace.get("stem", "?"))
    total_llm = trace.get("total_llm_calls", 0)
    header = (
        f"<h3>{stem}</h3>"
        f"<div>Duration: {total_ms}ms | LLM calls: {total_llm} | "
        f"Started: {html.escape(trace.get('started_at', '?'))}</div>"
    )

    return (
        header
        + "\n".join(parts)
        + "<div style='margin-top:12px'>"
        + "\n".join(details_parts)
        + "</div>"
    )


def render_html(traces: list[dict]) -> str:
    style = """
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
               max-width: 1100px; margin: 24px auto; padding: 0 16px; background: #fafafa; }
        h2 { border-bottom: 2px solid #2171b5; padding-bottom: 6px; }
        details { margin: 4px 0; padding: 4px 8px; background: #fff; border: 1px solid #e0e0e0; border-radius: 4px; }
        pre { background: #f5f5f5; padding: 8px; border-radius: 3px; overflow-x: auto; }
        .tab { display: inline-block; padding: 6px 16px; margin: 2px 4px; cursor: pointer;
               border: 1px solid #ccc; border-radius: 4px 4px 0 0; background: #eee; }
        .tab.active { background: #fff; border-bottom-color: #fff; font-weight: bold; }
        .panel { display: none; padding: 12px; background: #fff; border: 1px solid #ccc; border-radius: 0 4px 4px 4px; }
        .panel.active { display: block; }
    </style>
    """

    if len(traces) == 1:
        body = f"<h2>ETL Trace</h2>{_render_file(traces[0])}"
    else:
        tabs = []
        panels = []
        for i, t in enumerate(traces):
            stem = html.escape(t.get("stem", f"file_{i}"))
            active = "active" if i == 0 else ""
            tabs.append(f'<span class="tab {active}" onclick="switchTab({i})">{stem}</span>')
            panels.append(f'<div class="panel {active}" id="panel-{i}">{_render_file(t)}</div>')
        tab_js = """
        <script>
        function switchTab(idx) {
            document.querySelectorAll('.tab').forEach((t,i) => t.classList.toggle('active', i===idx));
            document.querySelectorAll('.panel').forEach((p,i) => p.classList.toggle('active', i===idx));
        }
        </script>
        """
        body = (
            f"<h2>ETL Traces ({len(traces)} files)</h2>"
            + "\n".join(tabs)
            + "\n".join(panels)
            + tab_js
        )

    return f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>ETL Trace</title>{style}</head><body>{body}</body></html>"


def main() -> None:
    if len(sys.argv) < 2:
        print(
            "Usage: uv run python scripts/view_etl_trace.py <trace.json or task_dir>",
            file=sys.stderr,
        )
        sys.exit(1)

    target = Path(sys.argv[1])
    traces: list[dict] = []

    if target.is_file() and target.suffix == ".json":
        traces.append(json.loads(target.read_text(encoding="utf-8")))
    elif target.is_dir():
        cache = target / "_cache" if (target / "_cache").is_dir() else target
        for p in sorted(cache.glob("*_trace.json")):
            traces.append(json.loads(p.read_text(encoding="utf-8")))
    else:
        print(f"Error: {target} is not a JSON file or directory", file=sys.stderr)
        sys.exit(1)

    if not traces:
        print("No trace files found.", file=sys.stderr)
        sys.exit(1)

    out = target.parent / "etl_pipeline.html" if target.is_file() else target / "etl_pipeline.html"
    out.write_text(render_html(traces), encoding="utf-8")
    print(f"Written to {out}")
    webbrowser.open(out.as_uri())


if __name__ == "__main__":
    main()
