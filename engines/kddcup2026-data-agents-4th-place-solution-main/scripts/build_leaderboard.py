#!/usr/bin/env python3
"""Build artifacts/leaderboard.tsv from artifacts/runs/ + artifacts/replications/.

For each experiment that has at least one run with evaluation.json:
  - latest run dir (timestamp-newest)
  - λ0.5, perfect, with_extras, missing
  - replication mean / std / n if artifacts/replications/<exp>/summary_*.json exists
  - normalized diff against the canonical baseline (default: exp_040_selfdbg_fence)
    written to artifacts/diffs/<exp>.diff (skipped for the baseline itself)

Output columns (tab-separated):
  exp_name  latest_run  lambda05  perfect  with_extras  missing
  rep_n  rep_mean  rep_std  rep_ci_lo  rep_ci_hi
  base_exp  diff_lines  diff_path

Usage:
    python scripts/build_leaderboard.py [--base exp_040_selfdbg_fence]
                                        [--out artifacts/leaderboard.tsv]
                                        [--diff-dir artifacts/diffs]
                                        [--no-diff]   # skip diff computation

The script is idempotent — re-running overwrites the leaderboard and any
diff files for experiments whose code changed.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "artifacts" / "runs"
REPL_DIR = REPO_ROOT / "artifacts" / "replications"
EXP_ROOT = REPO_ROOT / "src" / "experiments"

RUN_PAT = re.compile(r"^(?P<name>exp_\d+_.+)_(?P<idx>\d{3})$")


def list_runs_by_exp() -> dict[str, list[Path]]:
    """Return {exp_name: [sorted run dirs]} from artifacts/runs/."""
    out: dict[str, list[Path]] = {}
    if not RUNS_DIR.is_dir():
        return out
    for p in sorted(RUNS_DIR.iterdir()):
        if not p.is_dir():
            continue
        m = RUN_PAT.match(p.name)
        if not m:
            continue
        out.setdefault(m["name"], []).append(p)
    for v in out.values():
        v.sort()  # NNN suffix means lex sort == numeric sort
    return out


def latest_eval(
    run_dirs: list[Path], min_tasks: int = 0
) -> tuple[Path | None, dict | None]:
    """Pick the latest run dir whose evaluation.json has task_count >= min_tasks.

    If min_tasks > 0 and no run qualifies, returns (None, None) — caller should
    treat this exp as having no full-eval run yet (vs. smoke-only).
    """
    for d in reversed(run_dirs):
        ej = d / "evaluation.json"
        if not ej.is_file():
            continue
        try:
            ev = json.loads(ej.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if ev.get("task_count", 0) >= min_tasks:
            return d, ev
    return None, None


def latest_replication(exp_name: str) -> dict | None:
    rep_dir = REPL_DIR / exp_name
    if not rep_dir.is_dir():
        return None
    summaries = sorted(rep_dir.glob("summary_*.json"))
    if not summaries:
        return None
    try:
        return json.loads(summaries[-1].read_text())
    except (json.JSONDecodeError, OSError):
        return None


def compute_diff_for(exp_name: str, base: str, diff_dir: Path) -> tuple[int, Path | None]:
    """Run scripts/compute_exp_diff.py base exp_name -> diff_dir/<exp>.diff."""
    if exp_name == base:
        return 0, None
    if not (EXP_ROOT / exp_name).is_dir():
        return -1, None
    out_path = diff_dir / f"{exp_name}.diff"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "compute_exp_diff.py"),
        base,
        exp_name,
        "-o",
        str(out_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(
            f"  diff failed for {exp_name}: rc={proc.returncode} stderr={proc.stderr.strip()}\n"
        )
        return -1, None
    try:
        n_lines = sum(1 for _ in out_path.open())
    except OSError:
        n_lines = -1
    return n_lines, out_path


def fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="exp_040_selfdbg_fence",
                    help="canonical baseline experiment for diff comparison")
    ap.add_argument("--out", default="artifacts/leaderboard.tsv")
    ap.add_argument("--diff-dir", default="artifacts/diffs")
    ap.add_argument("--no-diff", action="store_true",
                    help="skip diff computation (faster; reuses existing diff files)")
    ap.add_argument("--min-tasks", type=int, default=50,
                    help="only include runs whose evaluation.json has task_count >= this "
                         "(default 50 = full benchmark). Use 0 to include smoke runs.")
    args = ap.parse_args()

    out_path = REPO_ROOT / args.out
    diff_dir = REPO_ROOT / args.diff_dir

    runs = list_runs_by_exp()
    print(f"found runs for {len(runs)} experiments", file=sys.stderr)

    rows: list[dict] = []
    for exp_name in sorted(runs.keys()):
        d, ev = latest_eval(runs[exp_name], min_tasks=args.min_tasks)
        if not ev:
            continue

        rep = latest_replication(exp_name)
        rep_block = rep.get("lambda_0_5", {}) if rep else {}

        diff_lines: int = -1
        diff_path: Path | None = None
        if not args.no_diff:
            diff_lines, diff_path = compute_diff_for(exp_name, args.base, diff_dir)
        else:
            # Reuse existing diff file if present
            cand = diff_dir / f"{exp_name}.diff"
            if cand.is_file():
                diff_path = cand
                try:
                    diff_lines = sum(1 for _ in cand.open())
                except OSError:
                    diff_lines = -1

        rel_diff = ""
        if diff_path is not None:
            try:
                rel_diff = str(diff_path.relative_to(REPO_ROOT))
            except ValueError:
                rel_diff = str(diff_path)

        rows.append(
            {
                "exp_name": exp_name,
                "latest_run": d.name if d else "",
                "task_count": ev.get("task_count"),
                "lambda05": ev.get("official_score_lambda_0_5_mean"),
                "perfect": ev.get("perfect_recall_no_extras_count"),
                "with_extras": ev.get("perfect_recall_with_extras_count"),
                "missing": ev.get("missing_prediction_count"),
                "rep_n": rep.get("n") if rep else None,
                "rep_mean": rep_block.get("mean"),
                "rep_std": rep_block.get("std"),
                "rep_ci_lo": (rep_block.get("ci_95") or [None, None])[0],
                "rep_ci_hi": (rep_block.get("ci_95") or [None, None])[1],
                "base_exp": args.base if exp_name != args.base else "",
                "diff_lines": diff_lines if diff_lines >= 0 else "",
                "diff_path": rel_diff,
            }
        )

    # Sort by lambda05 desc; ties: perfect desc.
    rows.sort(
        key=lambda r: (
            -(r["lambda05"] if r["lambda05"] is not None else -1),
            -(r["perfect"] if r["perfect"] is not None else -1),
        )
    )

    cols = [
        "exp_name", "latest_run", "task_count",
        "lambda05", "perfect", "with_extras", "missing",
        "rep_n", "rep_mean", "rep_std", "rep_ci_lo", "rep_ci_hi",
        "base_exp", "diff_lines", "diff_path",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(fmt(r[c]) for c in cols) + "\n")
    print(f"wrote {len(rows)} rows to {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
