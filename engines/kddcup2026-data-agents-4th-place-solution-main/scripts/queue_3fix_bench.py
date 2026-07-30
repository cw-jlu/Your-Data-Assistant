"""Queue 3 fix experiments after v5_v6_compare bench finishes.

Waits for artifacts/v5_v6_compare/summary.csv to appear, then runs each fix
variant sequentially at default config (= 20 streams each, ~75min/bench).

Output: artifacts/3fix_compare/{summary.csv, stats.md}
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DONE_MARKER = REPO / "artifacts" / "v5_v6_compare" / "summary.csv"
OUT_DIR = REPO / "artifacts" / "3fix_compare"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VARIANTS = [
    "exp_122_v6_compare",        # baseline (= v6 as-is)
    "exp_134_zero_rows_hint",    # fix 1
    "exp_135_sample_values",     # fix 2
    "exp_136_doc_marker",        # fix 3
]


def wait_for_v5v6():
    print(f"[queue] waiting for {DONE_MARKER}...", flush=True)
    while not DONE_MARKER.exists():
        time.sleep(60)
    print(f"[queue] v5_v6 done, starting fixes", flush=True)


def run_bench(variant: str) -> dict:
    config = REPO / "src" / "experiments" / variant / "config.yaml"
    log = OUT_DIR / f"{variant}.log"
    t0 = time.time()
    cmd = [
        "uv", "run", "python", "-m", f"experiments.{variant}.run",
        "run-benchmark",
        "--config", str(config),
        "--evaluate",
    ]
    with open(log, "w") as logf:
        proc = subprocess.run(cmd, cwd=REPO, stdout=logf, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0
    runs = sorted((REPO / "artifacts" / "runs").glob(f"{variant}_*"),
                  key=lambda p: p.stat().st_mtime)
    latest = runs[-1] if runs else None
    score = None
    if latest:
        ej = latest / "evaluation.json"
        if ej.exists():
            try:
                summary = json.load(open(ej))
                score = summary.get("official_score_lambda_0_5_mean")
            except Exception:
                pass
    return {
        "variant": variant,
        "exit_code": proc.returncode,
        "elapsed_s": int(elapsed),
        "score": score,
        "run_dir": str(latest) if latest else None,
    }


def main():
    wait_for_v5v6()
    results = []
    for v in VARIANTS:
        print(f"\n[queue] running {v}...", flush=True)
        r = run_bench(v)
        results.append(r)
        print(f"[queue] {v} done: score={r['score']} elapsed={r['elapsed_s']}s", flush=True)
    with open(OUT_DIR / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["variant", "score", "elapsed_s", "exit_code", "run_dir"])
        w.writeheader()
        w.writerows(results)
    print("\n=== 3-fix bench summary ===")
    for r in results:
        print(f"  {r['variant']}: {r['score']}")


if __name__ == "__main__":
    main()
