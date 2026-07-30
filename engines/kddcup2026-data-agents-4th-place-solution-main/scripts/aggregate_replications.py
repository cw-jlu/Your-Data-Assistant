#!/usr/bin/env python3
"""Aggregate all replication summaries into a ranking table.

Walks artifacts/replications/<exp>/summary_*.json (newest per exp),
prints a markdown table sorted by mean λ0.5, and emits machine-readable
JSON so other tools can post-process.

Usage:
    python3 scripts/aggregate_replications.py             # markdown to stdout
    python3 scripts/aggregate_replications.py --json      # JSON to stdout
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPL_ROOT = Path(__file__).resolve().parent.parent / "artifacts" / "replications"


def latest_summary(exp_dir: Path) -> dict | None:
    summaries = sorted(exp_dir.glob("summary_*.json"))
    if not summaries:
        return None
    try:
        return json.loads(summaries[-1].read_text())
    except (OSError, json.JSONDecodeError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="emit JSON instead of markdown")
    args = ap.parse_args()

    if not REPL_ROOT.is_dir():
        print(f"# no replications dir at {REPL_ROOT}", file=sys.stderr)
        return 1

    rows: list[dict] = []
    for exp_dir in sorted(REPL_ROOT.iterdir()):
        if not exp_dir.is_dir():
            continue
        s = latest_summary(exp_dir)
        if s is None:
            continue
        l = s.get("lambda_0_5", {})
        ci = l.get("ci_95") or [None, None]
        rows.append({
            "exp": s.get("exp_name", exp_dir.name),
            "n": s.get("n"),
            "mean": l.get("mean"),
            "median": l.get("median"),
            "std": l.get("std"),
            "min": l.get("min"),
            "max": l.get("max"),
            "ci_lo": ci[0],
            "ci_hi": ci[1],
            "ci_half": l.get("ci_95_half_width"),
            "perfect_mean": (s.get("perfect") or {}).get("mean"),
            "missing_mean": (s.get("missing") or {}).get("mean"),
            "summary_path": str(sorted(exp_dir.glob("summary_*.json"))[-1].relative_to(REPL_ROOT.parent.parent)),
        })

    rows.sort(key=lambda r: -(r["mean"] if r["mean"] is not None else -1))

    if args.json:
        print(json.dumps(rows, indent=2))
        return 0

    if not rows:
        print("# (no replication summaries yet)")
        return 0

    # Markdown table
    print("| Rank | Experiment | n | mean λ0.5 | median | std | 95% CI | perfect | missing |")
    print("|------|------------|---|-----------|--------|-----|--------|---------|---------|")
    for i, r in enumerate(rows, 1):
        ci_str = (
            f"[{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]"
            if r["ci_lo"] is not None and r["ci_hi"] is not None
            else "—"
        )
        mean_str = f"{r['mean']:.4f}" if r['mean'] is not None else "?"
        median_str = f"{r['median']:.4f}" if r['median'] is not None else "?"
        std_str = f"{r['std']:.4f}" if r['std'] is not None else "?"
        perfect_str = f"{r['perfect_mean']}" if r['perfect_mean'] is not None else "?"
        missing_str = f"{r['missing_mean']}" if r['missing_mean'] is not None else "?"
        print(f"| {i} | `{r['exp']}` | {r['n']} | **{mean_str}** | {median_str} | {std_str} | {ci_str} | {perfect_str} | {missing_str} |")

    # Verdict block
    print()
    print("**Verdict notes** (mean - exp_040_baseline):")
    base_row = next((r for r in rows if r["exp"] == "exp_040_selfdbg_fence"), None)
    if base_row and base_row["mean"]:
        baseline_mean = base_row["mean"]
        for r in rows:
            if r["exp"] == "exp_040_selfdbg_fence":
                continue
            delta = (r["mean"] or 0) - baseline_mean
            ci_lo = r["ci_lo"] or 0
            base_ci_hi = (base_row["ci_hi"] or baseline_mean)
            ci_overlap = ci_lo <= base_ci_hi
            if delta > 0.02 and not ci_overlap:
                v = "✅ confirmed_win"
            elif delta > 0.01:
                v = "⚠ likely_win (CI overlaps)"
            elif delta < -0.01:
                v = "❌ regression"
            else:
                v = "— noise"
            print(f"- {r['exp']}: Δmean = {delta:+.4f} → {v}")
    else:
        print("- (exp_040_selfdbg_fence baseline not yet replicated)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
