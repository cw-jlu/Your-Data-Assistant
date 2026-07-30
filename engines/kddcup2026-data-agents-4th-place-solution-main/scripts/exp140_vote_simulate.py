"""Apply exp_140's adaptive_vote to per-task prediction.csv across 3 runs,
score the voted answer, and report mean / per-task delta.
"""
from __future__ import annotations
import csv
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from kobushi_core.benchmark.schema import AnswerTable
from kobushi_core.eval.csv_compare import EvaluationOptions, _evaluate_task
from experiments.exp_140_agentar_icl.adaptive_vote import adaptive_vote

RUN_DIRS = [
    ROOT / "artifacts" / "runs" / "exp140_icl_local_001",
    ROOT / "artifacts" / "runs" / "exp140_icl_local_002",
    ROOT / "artifacts" / "runs" / "exp140_icl_local_003",
]
GOLD_DIR = ROOT / "data" / "public" / "output"


def load_csv_as_answer(p: Path) -> AnswerTable | None:
    if not p.exists():
        return None
    with p.open() as f:
        r = csv.reader(f)
        try:
            cols = next(r)
        except StopIteration:
            return None
        rows = [tuple(row) for row in r]
    return AnswerTable(columns=tuple(cols), rows=tuple(rows))


def score_answer(ans: AnswerTable | None, tid: str) -> float:
    if ans is None or not ans.rows:
        return 0.0
    gold = GOLD_DIR / tid / "gold.csv"
    if not gold.exists():
        return 0.0
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", newline="", delete=False) as tf:
        w = csv.writer(tf)
        w.writerow(ans.columns)
        for row in ans.rows:
            w.writerow(row)
        pred = Path(tf.name)
    try:
        e = _evaluate_task(task_id=tid, prediction_path=pred, gold_path=gold, options=EvaluationOptions())
        return float(e.official_score_lambda_0_5)
    finally:
        pred.unlink(missing_ok=True)


def main():
    # Collect all task_ids that appear in any run
    tids = set()
    for d in RUN_DIRS:
        for tdir in d.glob("task_*"):
            tids.add(tdir.name)
    tids = sorted(tids)
    print(f"[init] total task ids: {len(tids)}")

    per_run_scores: dict[str, dict[str, float]] = {d.name: {} for d in RUN_DIRS}
    vote_scores: dict[str, float] = {}
    disagree_rows: list[tuple] = []

    for tid in tids:
        attempts: list[AnswerTable] = []
        per_run: list[float] = []
        for d in RUN_DIRS:
            ans = load_csv_as_answer(d / tid / "prediction.csv")
            sc = score_answer(ans, tid)
            per_run.append(sc)
            per_run_scores[d.name][tid] = sc
            if ans is not None:
                attempts.append(ans)
        if not attempts:
            vote_scores[tid] = 0.0
            continue
        voted = adaptive_vote(attempts)
        vs = score_answer(voted, tid)
        vote_scores[tid] = vs
        if max(per_run) - min(per_run) > 0.01 or vs != max(per_run):
            disagree_rows.append((tid, per_run + [vs]))

    # Use only tasks where all 3 runs completed
    common = [t for t in tids if all(t in per_run_scores[d.name] and per_run_scores[d.name][t] is not None for d in RUN_DIRS)
              and all((d / t / "prediction.csv").exists() for d in RUN_DIRS)]
    print(f"[init] tasks with all 3 attempts available: {len(common)}")

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    print()
    print(f"=== Per-run means (subset n={len(common)}) ===")
    for d in RUN_DIRS:
        m = mean([per_run_scores[d.name][t] for t in common])
        print(f"  {d.name:<35} {m:.4f}")
    print(f"  mean-of-runs (= no vote):           {mean([(per_run_scores[RUN_DIRS[0].name][t]+per_run_scores[RUN_DIRS[1].name][t]+per_run_scores[RUN_DIRS[2].name][t])/3 for t in common]):.4f}")
    print(f"  ADAPTIVE_VOTE (= official logic):   {mean([vote_scores[t] for t in common]):.4f}")
    print(f"  best-of-3 (= upper bound):          {mean([max(per_run_scores[d.name][t] for d in RUN_DIRS) for t in common]):.4f}")

    print()
    print(f"=== Tasks where vote differs (n={len(disagree_rows)}) ===")
    print(f"{'tid':<14} {'r1':>5} {'r2':>5} {'r3':>5} {'VOTE':>6}")
    for tid, scores in disagree_rows:
        if tid in common:
            print(f"{tid:<14} {scores[0]:>5.2f} {scores[1]:>5.2f} {scores[2]:>5.2f} {scores[3]:>6.2f}")


if __name__ == "__main__":
    main()
