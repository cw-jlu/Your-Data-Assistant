"""Render a markdown score report from `mock_scorer --json` output.

Consumes the JSON array emitted by
`python -m data_agent_baseline.scoring.mock_scorer --json` (one summary per λ)
and produces:

  * a Markdown file (overall + by-difficulty + per-task tables, with multi-λ
    sensitivity at the top when more than one λ was scored), and
  * optionally a plain-text summary to stdout for terminal feedback.

Used by `scripts/local_eval.sh` to leave a permanent record at
`artifacts/sandbox/<version>/score.md` for every eval run.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def render_markdown(
    summaries: list[dict[str, Any]],
    *,
    version: str | None,
    commit: str | None,
    model_api_url: str | None,
    model_name: str | None,
    predictions_root: str | None,
) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines: list[str] = []
    title = "Score report" + (f" — {version}" if version else "")
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"_Generated: **{ts}**_")
    if commit:
        lines.append(f"_Commit: `{commit}`_")
    if model_api_url:
        lines.append(f"_Endpoint: `{model_api_url}`_")
    if model_name:
        lines.append(f"_Model: `{model_name}`_")
    if predictions_root:
        lines.append(f"_Predictions root: `{predictions_root}`_")
    lines.append("")

    if not summaries:
        lines.append("> No summaries to report.")
        return "\n".join(lines) + "\n"

    if len(summaries) > 1:
        lines.append("## λ sensitivity")
        lines.append("")
        lines.append("| λ | Mean Score | Mean Recall | Tasks | Missing predictions |")
        lines.append("|---|---|---|---|---|")
        for s in summaries:
            lines.append(
                f"| {s['lambda']:.2f} | {_fmt(s['mean_score'])} | "
                f"{_fmt(s['mean_recall'])} | {s['task_count']} | "
                f"{len(s.get('missing_predictions') or [])} |"
            )
        lines.append("")

    for s in summaries:
        header = f"λ = {s['lambda']:.2f}" if len(summaries) > 1 else "Summary"
        lines.append(f"## {header}")
        lines.append("")
        lines.append(f"- Tasks scored: **{s['task_count']}**")
        lines.append(f"- Mean Score: **{_fmt(s['mean_score'])}**")
        lines.append(f"- Mean Recall: **{_fmt(s['mean_recall'])}**")
        missing_pred = s.get("missing_predictions") or []
        missing_gold = s.get("missing_gold") or []
        if missing_pred:
            preview = missing_pred[:8]
            ellipsis = " …" if len(missing_pred) > 8 else ""
            lines.append(f"- Missing predictions: {len(missing_pred)} → `{preview}{ellipsis}`")
        if missing_gold:
            preview = missing_gold[:8]
            ellipsis = " …" if len(missing_gold) > 8 else ""
            lines.append(f"- Missing gold: {len(missing_gold)} → `{preview}{ellipsis}`")
        lines.append("")

        by_diff = s.get("by_difficulty") or {}
        if by_diff:
            lines.append("### By difficulty")
            lines.append("")
            lines.append("| Difficulty | n | Mean Score | Mean Recall |")
            lines.append("|---|---|---|---|")
            for diff in sorted(by_diff):
                stats = by_diff[diff]
                lines.append(
                    f"| {diff} | {int(stats['task_count'])} | "
                    f"{_fmt(stats['mean_score'])} | {_fmt(stats['mean_recall'])} |"
                )
            lines.append("")

        tasks = s.get("tasks") or []
        if tasks:
            tasks_sorted = sorted(tasks, key=lambda t: (t["score"], t["task_id"]))
            lines.append("### Per-task (sorted by score asc)")
            lines.append("")
            lines.append(
                "| Task | Difficulty | Score | Recall | Matched / Gold | Predicted |"
            )
            lines.append("|---|---|---|---|---|---|")
            for t in tasks_sorted:
                lines.append(
                    f"| {t['task_id']} | {t.get('difficulty') or '?'} | "
                    f"{_fmt(t['score'])} | {_fmt(t['recall'])} | "
                    f"{t['matched_columns']} / {t['gold_columns']} | "
                    f"{t['predicted_columns']} |"
                )
            lines.append("")

    return "\n".join(lines) + "\n"


def render_text(summaries: list[dict[str, Any]]) -> str:
    out: list[str] = []
    for s in summaries:
        out.append(f"λ = {s['lambda']:.2f}")
        out.append(f"  tasks scored: {s['task_count']}")
        out.append(f"  mean Score:   {_fmt(s['mean_score'])}")
        out.append(f"  mean Recall:  {_fmt(s['mean_recall'])}")
        missing_pred = s.get("missing_predictions") or []
        missing_gold = s.get("missing_gold") or []
        if missing_pred:
            out.append(f"  missing predictions: {len(missing_pred)} → {missing_pred[:8]}")
        if missing_gold:
            out.append(f"  missing gold: {len(missing_gold)} → {missing_gold[:8]}")
        by_diff = s.get("by_difficulty") or {}
        if by_diff:
            out.append("  by difficulty:")
            for diff in sorted(by_diff):
                stats = by_diff[diff]
                out.append(
                    f"    {diff:<10} n={int(stats['task_count']):>4}  "
                    f"score={_fmt(stats['mean_score'])}  "
                    f"recall={_fmt(stats['mean_recall'])}"
                )
        out.append("---")
    return "\n".join(out)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--json-input",
        type=Path,
        required=True,
        help="JSON file from mock_scorer --json. Use '-' to read from stdin.",
    )
    ap.add_argument("--md-output", type=Path, required=True)
    ap.add_argument("--version", help="Submission version label, e.g. v1")
    ap.add_argument("--commit", help="Git commit short SHA")
    ap.add_argument("--model-api-url")
    ap.add_argument("--model-name")
    ap.add_argument("--predictions-root")
    ap.add_argument(
        "--print-text",
        action="store_true",
        help="Also write a plain-text summary to stdout.",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    if str(args.json_input) == "-":
        raw = sys.stdin.read()
    else:
        raw = Path(args.json_input).read_text()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"failed to parse JSON: {exc}", file=sys.stderr)
        return 1

    summaries = data if isinstance(data, list) else [data]

    md = render_markdown(
        summaries,
        version=args.version,
        commit=args.commit,
        model_api_url=args.model_api_url,
        model_name=args.model_name,
        predictions_root=args.predictions_root,
    )
    args.md_output.parent.mkdir(parents=True, exist_ok=True)
    args.md_output.write_text(md, encoding="utf-8")

    if args.print_text:
        print(render_text(summaries))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
