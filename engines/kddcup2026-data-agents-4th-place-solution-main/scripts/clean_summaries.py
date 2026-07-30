"""Rewrite replication summary JSONs in-place, excluding runs whose run_dir
has been moved to artifacts/runs_discarded/ (= contaminated/aborted runs).

Re-computes lambda_0_5 / perfect / missing aggregate stats from the surviving
runs only. Skips files where all runs are still valid.

Usage:
    uv run python scripts/clean_summaries.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REPL = REPO / "artifacts" / "replications"


def _agg(scores: list[float]) -> dict:
    if not scores:
        return {}
    n = len(scores)
    mean = statistics.fmean(scores)
    std = statistics.stdev(scores) if n >= 2 else 0.0
    median = statistics.median(scores)
    half = 1.96 * std / (n**0.5) if n >= 2 else 0.0
    return {
        "mean": round(mean, 4),
        "median": round(median, 4),
        "std": round(std, 4),
        "min": min(scores),
        "max": max(scores),
        "ci_95_half_width": round(half, 4),
        "ci_95": [round(mean - half, 4), round(mean + half, 4)],
    }


def _agg_int(values: list[int]) -> dict:
    if not values:
        return {}
    return {
        "mean": round(statistics.fmean(values), 1),
        "min": min(values),
        "max": max(values),
    }


def clean_one(summary_path: Path, dry_run: bool) -> bool:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    runs = payload.get("runs") or []
    if not runs:
        return False

    surviving = []
    dropped = []
    for r in runs:
        rd = r.get("run_dir")
        if rd and (REPO / rd).is_dir():
            surviving.append(r)
        else:
            dropped.append(rd)
    if not dropped:
        return False  # nothing to clean

    if not surviving:
        # All runs dropped — leave file alone, log warning
        print(f"  [WARN] {summary_path.name}: all {len(runs)} runs missing, leaving file untouched")
        return False

    # Recompute aggregates
    lam = [r["lambda_0_5"] for r in surviving if "lambda_0_5" in r]
    perfect = [r["perfect"] for r in surviving if "perfect" in r]
    missing = [r["missing"] for r in surviving if "missing" in r]

    payload["n"] = len(surviving)
    if lam:
        payload["lambda_0_5"] = _agg(lam)
    if perfect:
        payload["perfect"] = _agg_int(perfect)
    if missing:
        payload["missing"] = _agg_int(missing)
    payload["runs"] = surviving
    payload["_meta"] = payload.get("_meta", {})
    payload["_meta"]["cleaned_dropped_runs"] = dropped

    print(
        f"  {summary_path.relative_to(REPO)}: "
        f"{len(runs)} → {len(surviving)} runs (dropped {len(dropped)}), "
        f"new mean={payload['lambda_0_5']['mean']:.4f}"
    )

    if not dry_run:
        summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not REPL.is_dir():
        print(f"no replications dir: {REPL}")
        sys.exit(0)

    n_changed = 0
    for exp_dir in sorted(REPL.iterdir()):
        if not exp_dir.is_dir():
            continue
        for summary_path in sorted(exp_dir.glob("summary_*.json")):
            if clean_one(summary_path, args.dry_run):
                n_changed += 1
    print(f"\n{'(dry-run) ' if args.dry_run else ''}cleaned {n_changed} summary files")


if __name__ == "__main__":
    main()
