"""Run v5 vs v6 comparison bench: 5 each, 2 in parallel (= 30 vLLM streams cap).

v5 = bad60a7 commit MINUS v6 patches (= 3dp Python round + no prose-txt guard)
v6 = bad60a7 HEAD (= 2dp Python round + prose-txt guard for .md/.txt/.markdown)

For each variant, run 5 sequential benches (= n=5). Pairs run concurrently.

Output:
  artifacts/v5_v6_compare/
    v5_run_<i>/ ...
    v6_run_<i>/ ...
    summary.csv  (= per-run mean score)
    stats.md     (= mean/stdev/t-test)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "artifacts" / "v5_v6_compare"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_RUNS = 5
PARALLEL_PAIRS = 1  # = 2 benches (1 v5 + 1 v6) per pair


def run_bench(variant: str, run_idx: int) -> dict:
    """Run a single bench. variant ∈ {'v5','v6'}."""
    dir_name = f"exp_122_v{variant[1]}_compare"  # v5_compare or v6_compare
    config = REPO / "src" / "experiments" / dir_name / "config.yaml"
    run_out_dir = OUT_DIR / f"{variant}_run_{run_idx}"
    run_out_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_out_dir / "bench.log"
    t0 = time.time()
    # Use unique run_id to avoid collision
    env = os.environ.copy()
    env["RUN_ID_SUFFIX"] = f"{variant}_compare_{run_idx}"
    cmd = [
        "uv", "run", "python", "-m", f"experiments.{dir_name}.run",
        "run-benchmark",
        "--config", str(config),
        "--evaluate",
    ]
    with open(log_path, "w") as logf:
        proc = subprocess.run(cmd, cwd=REPO, env=env, stdout=logf, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0
    # Find score in log
    log_text = log_path.read_text()
    score = None
    import re
    m = re.search(r"(?:mean|score|recall)[:\s]+([0-9.]+)", log_text, re.IGNORECASE)
    if m:
        try:
            score = float(m.group(1))
        except Exception:
            pass
    # Read evaluation.json from latest matching run dir
    runs_dir = REPO / "artifacts" / "runs"
    candidates = sorted(runs_dir.glob(f"{dir_name}_*"), key=lambda p: p.stat().st_mtime) if runs_dir.exists() else []
    latest = candidates[-1] if candidates else None
    if latest:
        ej = latest / "evaluation.json"
        if ej.exists():
            try:
                summary = json.load(open(ej))
                score = summary.get("official_score_lambda_0_5_mean", score)
            except Exception:
                pass
    return {
        "variant": variant,
        "run_idx": run_idx,
        "exit_code": proc.returncode,
        "elapsed_s": int(elapsed),
        "score": score,
        "run_dir": str(latest) if latest else None,
    }


def main():
    print(f"=== v5 vs v6 compare bench (n={N_RUNS} each, 1 pair parallel) ===", flush=True)

    jobs = []
    for i in range(N_RUNS):
        for variant in ("v5", "v6"):
            jobs.append((variant, i))

    results = []
    # Run 2 at a time (= 1 v5 + 1 v6 per pair)
    pair_size = 2
    for batch_start in range(0, len(jobs), pair_size):
        batch = jobs[batch_start: batch_start + pair_size]
        print(f"\n--- batch {batch_start//pair_size + 1}/{len(jobs)//pair_size}: {batch} ---", flush=True)
        with ThreadPoolExecutor(max_workers=pair_size) as ex:
            futs = {ex.submit(run_bench, v, i): (v, i) for v, i in batch}
            for fut in as_completed(futs):
                v, i = futs[fut]
                r = fut.result()
                results.append(r)
                print(f"  done: {v} run_{i} score={r['score']} elapsed={r['elapsed_s']}s", flush=True)

    # Save
    import csv
    with open(OUT_DIR / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["variant", "run_idx", "score", "elapsed_s", "exit_code", "run_dir"])
        w.writeheader()
        w.writerows(results)

    # Stats
    v5_scores = [r["score"] for r in results if r["variant"] == "v5" and r["score"] is not None]
    v6_scores = [r["score"] for r in results if r["variant"] == "v6" and r["score"] is not None]

    def stats(xs):
        n = len(xs)
        if n == 0: return (None, None, n)
        mean = sum(xs)/n
        var = sum((x-mean)**2 for x in xs)/(n-1) if n>1 else 0
        return (mean, var**0.5, n)

    m5, s5, n5 = stats(v5_scores)
    m6, s6, n6 = stats(v6_scores)

    lines = [
        "# v5 vs v6 compare — bench results\n",
        f"## v5 (n={n5}): mean={m5:.4f}, stdev={s5:.4f}",
        f"## v6 (n={n6}): mean={m6:.4f}, stdev={s6:.4f}",
        f"\n## Δ (v6 - v5): {m6-m5 if m5 and m6 else '?':.4f}",
        "",
        "## Per-run scores",
        f"- v5: {v5_scores}",
        f"- v6: {v6_scores}",
    ]
    if n5 > 1 and n6 > 1:
        # Welch's t-test (no scipy dep — manual)
        import math
        se = math.sqrt(s5**2/n5 + s6**2/n6)
        t = (m6 - m5) / se if se > 0 else 0
        # df: Welch-Satterthwaite (approx)
        df = (s5**2/n5 + s6**2/n6)**2 / (
            (s5**2/n5)**2/(n5-1) + (s6**2/n6)**2/(n6-1)
        ) if (n5>1 and n6>1) else 1
        lines.append(f"\n## Welch's t-test: t={t:.3f}, df={df:.1f}")
        lines.append(f"(|t| > ~2.4 = significant at p<0.05 two-tailed for n=5)")
    (OUT_DIR / "stats.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
