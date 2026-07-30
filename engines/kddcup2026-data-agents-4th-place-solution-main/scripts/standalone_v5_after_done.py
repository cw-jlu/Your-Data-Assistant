"""After current benches finish, run a single v5_compare bench solo (= no contention).

Marker: artifacts/v5_v6_compare/summary.csv (= main script's last write)
Trigger: also wait for 3fix done (= artifacts/3fix_failures/results.csv)
Then: switch config max_workers=4, run v5_compare standalone, log score.

Output: artifacts/v5_standalone/{bench.log, evaluation.json link}
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
V5V6_DONE = REPO / "artifacts" / "v5_v6_compare" / "summary.csv"
THREEFIX_DONE = REPO / "artifacts" / "3fix_failures" / "results.csv"
OUT_DIR = REPO / "artifacts" / "v5_standalone"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print(f"[wait] waiting for {V5V6_DONE} AND {THREEFIX_DONE}", flush=True)
    while not (V5V6_DONE.exists() and THREEFIX_DONE.exists()):
        time.sleep(60)
    print(f"[wait] both done, sleeping 30s to let resources settle", flush=True)
    time.sleep(30)

    # Temporarily bump max_workers=4 on v5_compare config
    cfg = REPO / "src" / "experiments" / "exp_122_v5_compare" / "config.yaml"
    orig = cfg.read_text()
    patched = orig.replace("max_workers: 3", "max_workers: 4")
    cfg.write_text(patched)

    log = OUT_DIR / "bench.log"
    t0 = time.time()
    cmd = [
        "uv", "run", "python", "-m", "experiments.exp_122_v5_compare.run",
        "run-benchmark",
        "--config", str(cfg),
        "--evaluate",
    ]
    print(f"[run] {' '.join(cmd)}", flush=True)
    with open(log, "w") as f:
        proc = subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0
    cfg.write_text(orig)  # restore

    # Pull score from latest v5_compare run
    runs = sorted((REPO/"artifacts"/"runs").glob("exp_122_v5_compare_*"),
                  key=lambda p: p.stat().st_mtime)
    latest = runs[-1] if runs else None
    score = None
    if latest:
        ej = latest / "evaluation.json"
        if ej.exists():
            try:
                score = json.load(open(ej))["official_score_lambda_0_5_mean"]
            except: pass

    (OUT_DIR / "result.json").write_text(json.dumps({
        "score": score, "elapsed_s": int(elapsed),
        "run_dir": str(latest) if latest else None,
        "exit_code": proc.returncode,
    }, indent=2))
    print(f"\n[done] standalone v5 score = {score} (elapsed {int(elapsed)}s)", flush=True)
    if score is not None and score > 0.78:
        print("[verdict] >= 0.78 → 並列原因確定 (= 単独走で LB レンジに戻った)")
    elif score is not None and score < 0.76:
        print("[verdict] < 0.76 → 並列原因 NO (= vLLM 環境差確定)")
    else:
        print("[verdict] 0.76-0.78 → 部分的、追加検証要")


if __name__ == "__main__":
    main()
