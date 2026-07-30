"""POC: apply LLM-as-judge vote on EXISTING 3-run artifacts (= no re-bench).

For each task that has prediction.csv in all 3 runs (exp140_icl_local_001/002/003):
  1. Load 3 prediction CSVs → AnswerTable list
  2. Group by signature
  3. If single group: skip judge (= trivial agreement)
  4. Else: extract last SQL from each trace.log, call judge LLM, get pick
  5. Score chosen answer against gold

Compares:
  - per-run single-att means (= what we have)
  - adaptive_vote (= rule-based, current)
  - judge_vote (= LLM-based, new)
"""
from __future__ import annotations
import csv
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    os.environ["AGENT_API_BASE"] = "http://localhost:8000/v1"
    os.environ["AGENT_API_KEY"] = "local-key"
    os.environ["AGENT_MODEL"] = "qwen3.5-35b-a3b"
    os.environ["CF_ACCESS_CLIENT_ID"] = ""
    os.environ["CF_ACCESS_CLIENT_SECRET"] = ""

    from kobushi_core.benchmark.schema import AnswerTable
    from kobushi_core.eval.csv_compare import EvaluationOptions, _evaluate_task
    from kobushi_core.benchmark import DABenchPublicDataset
    from kobushi_core.model import OpenAIModelAdapter
    from experiments.exp_141_llm_judge_vote.llm_judge import judge_vote
    from experiments.exp_141_llm_judge_vote.adaptive_vote import adaptive_vote

    RUN_DIRS = [
        ROOT / "artifacts" / "runs" / "exp140_icl_local_001",
        ROOT / "artifacts" / "runs" / "exp140_icl_local_002",
        ROOT / "artifacts" / "runs" / "exp140_icl_local_003",
    ]
    GOLD_DIR = ROOT / "data" / "public" / "output"
    OUT = ROOT / "artifacts" / "exp141_judge_poc"
    OUT.mkdir(parents=True, exist_ok=True)

    def load_csv(p: Path) -> AnswerTable | None:
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
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", newline="", delete=False) as tf:
            w = csv.writer(tf)
            w.writerow(ans.columns)
            for row in ans.rows:
                w.writerow(row)
            pred = Path(tf.name)
        try:
            e = _evaluate_task(task_id=tid, prediction_path=pred, gold_path=gold,
                               options=EvaluationOptions())
            return float(e.official_score_lambda_0_5)
        finally:
            pred.unlink(missing_ok=True)

    ds = DABenchPublicDataset(root_dir=ROOT / "data" / "public" / "input")

    # Tasks with all 3 attempts
    tids = sorted({tdir.name for d in RUN_DIRS for tdir in d.glob("task_*")})
    common = [t for t in tids if all((d / t / "prediction.csv").exists() for d in RUN_DIRS)]
    print(f"[init] {len(common)} tasks have predictions in all 3 runs", flush=True)

    model = OpenAIModelAdapter(
        model="qwen3.5-35b-a3b",
        api_base="http://localhost:8000/v1",
        api_key="local-key",
        temperature=0.0,  # = judge wants deterministic
    )

    rows: list[dict] = []
    t_all = time.time()
    n_judge_called = 0
    for i, tid in enumerate(common, 1):
        attempts: list[AnswerTable | None] = []
        traces: list[str | None] = []
        per_run: list[float] = []
        for d in RUN_DIRS:
            ans = load_csv(d / tid / "prediction.csv")
            attempts.append(ans)
            traces.append(str(d / tid / "trace.log"))
            per_run.append(score_answer(ans, tid))

        rule_voted = adaptive_vote([a for a in attempts if a is not None])
        rule_score = score_answer(rule_voted, tid)

        question = ds.get_task(tid).question
        t0 = time.time()
        judge_ans, meta = judge_vote(attempts, question, model, trace_log_paths=traces)
        judge_dt = time.time() - t0
        judge_score = score_answer(judge_ans, tid)
        if not meta.get("skipped_judge"):
            n_judge_called += 1

        rows.append({
            "tid": tid,
            "per_run": per_run,
            "rule_vote": rule_score,
            "judge_vote": judge_score,
            "judge_meta": meta,
            "judge_dt": round(judge_dt, 2),
        })
        tag = "→J" if not meta.get("skipped_judge") else "skip"
        delta = judge_score - rule_score
        sym = " " if abs(delta) < 0.01 else ("+" if delta > 0 else "-")
        print(
            f"  [{i:2d}/{len(common)}] {tid}: rule={rule_score:.2f} judge={judge_score:.2f} "
            f"{sym}{abs(delta):.2f} {tag} ({judge_dt:.1f}s)",
            flush=True,
        )

    elapsed = time.time() - t_all
    n = len(rows)
    rule_mean = sum(r["rule_vote"] for r in rows) / n
    judge_mean = sum(r["judge_vote"] for r in rows) / n
    single_mean = sum(sum(r["per_run"]) / 3 for r in rows) / n
    best_of_3 = sum(max(r["per_run"]) for r in rows) / n

    print()
    print(f"=== POC summary (n={n} tasks, judge called {n_judge_called} times) ===")
    print(f"  single-att avg     : {single_mean:.4f}")
    print(f"  rule adaptive_vote : {rule_mean:.4f}")
    print(f"  LLM judge_vote     : {judge_mean:.4f}")
    print(f"  best-of-3 (upper)  : {best_of_3:.4f}")
    print(f"  elapsed            : {elapsed/60:.1f} min")

    (OUT / "summary.json").write_text(json.dumps({
        "n_tasks": n,
        "n_judge_called": n_judge_called,
        "elapsed_min": round(elapsed/60, 1),
        "single_att_avg": single_mean,
        "rule_vote_mean": rule_mean,
        "judge_vote_mean": judge_mean,
        "best_of_3_mean": best_of_3,
    }, indent=2))
    (OUT / "results.json").write_text(json.dumps(rows, indent=2, default=str))


if __name__ == "__main__":
    main()
