"""exp_137 n=3 runner: 3 sequential 50-task benches, compute mean/stdev."""
from __future__ import annotations

import json, os, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT_BASE = REPO / "artifacts" / "runs"


def run_once(run_idx: int) -> dict:
    """Run scripts/math_advisor_full50.py once, output to exp_137_math_advisor_NNN."""
    # Patch the OUT inside math_advisor_full50.py via env or by post-rename
    run_dir = OUT_BASE / f"exp_137_math_advisor_{run_idx+1:03d}"
    if run_dir.exists():
        import shutil; shutil.rmtree(run_dir)
    # The script uses fixed OUT path = exp_137_math_advisor_001; we'll rename after.
    fixed_path = OUT_BASE / "exp_137_math_advisor_001"
    if fixed_path.exists():
        import shutil; shutil.rmtree(fixed_path)
    log = REPO / "artifacts" / "exp_137_n3" / f"run_{run_idx+1}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cmd = ["uv", "run", "python", "scripts/math_advisor_full50.py"]
    print(f"\n=== run {run_idx+1}/3 starting ===", flush=True)
    with open(log, "w") as f:
        proc = subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0
    # Rename output dir
    if fixed_path.exists():
        fixed_path.rename(run_dir)
    # Collect score
    score = None
    n_perfect = n_zero = None
    eval_file = run_dir / "results.json"
    if eval_file.exists():
        results = json.load(open(eval_file))
        scores = [r["score"] for r in results]
        if scores:
            score = sum(scores)/len(scores)
            n_perfect = sum(1 for s in scores if s >= 0.99)
            n_zero = sum(1 for s in scores if s < 0.01)
    return {
        "run_idx": run_idx+1, "score": score,
        "n_perfect": n_perfect, "n_zero": n_zero,
        "elapsed_s": int(elapsed), "exit_code": proc.returncode,
        "run_dir": str(run_dir),
    }


def main():
    OUT = REPO / "artifacts" / "exp_137_n3"
    OUT.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    runs = []
    for i in range(3):
        r = run_once(i)
        runs.append(r)
        print(f"  run {r['run_idx']}: score={r['score']} perfect={r['n_perfect']}/50 zero={r['n_zero']}/50 elapsed={r['elapsed_s']}s", flush=True)
        with open(OUT/"running_results.json", "w") as f:
            json.dump(runs, f, indent=2, default=str)
    total = time.time() - t_start

    # Stats
    scores = [r["score"] for r in runs if r["score"] is not None]
    if scores:
        mean = sum(scores)/len(scores)
        stdev = (sum((s-mean)**2 for s in scores)/(len(scores)-1))**0.5 if len(scores)>1 else 0
    else:
        mean = stdev = None

    summary = {
        "n_runs": len(runs), "scores": scores,
        "mean": mean, "stdev": stdev,
        "total_elapsed_s": int(total),
        "baseline_main_rerun": 0.760,
        "delta_vs_baseline": mean - 0.760 if mean is not None else None,
    }
    with open(OUT/"summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n=== exp_137 n=3 SUMMARY ===")
    print(f"  scores: {scores}")
    print(f"  mean = {mean:.4f} ± {stdev:.4f}")
    print(f"  baseline = 0.760")
    print(f"  Δ = {mean - 0.760:+.4f}")
    print(f"  total elapsed: {int(total/60)}m")


if __name__ == "__main__":
    main()
