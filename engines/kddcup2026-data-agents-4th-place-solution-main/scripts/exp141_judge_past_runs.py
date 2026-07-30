"""Apply LLM-judge vote to past 3-attempt runs (= exp_137_math_advisor_001/002/003).

These runs store per-task `trace.json` as a list of 3 attempts each containing
{attempt, succeeded, answer={columns, rows}, steps=[...]}.

For each (run, task) we have 3 attempts. Pipeline:
  1. Parse 3 attempts → AnswerTables + steps lists
  2. Score each attempt against gold (= per-attempt single-att score)
  3. Apply rule adaptive_vote → score
  4. Apply LLM judge_vote (think OFF, T=0) → score
  5. Aggregate per-run + overall comparison

Output: artifacts/exp141_judge_past_runs/summary.json + results.json
"""
from __future__ import annotations
import csv
import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    os.environ["AGENT_API_BASE"] = "http://localhost:8000/v1"
    os.environ["AGENT_API_KEY"] = "local-key"
    os.environ["AGENT_MODEL"] = "qwen3.5-35b-a3b"
    os.environ["CF_ACCESS_CLIENT_ID"] = ""
    os.environ["CF_ACCESS_CLIENT_SECRET"] = ""

    from kobushi_core.benchmark.schema import AnswerTable
    from kobushi_core.benchmark import DABenchPublicDataset
    from kobushi_core.eval.csv_compare import EvaluationOptions, _evaluate_task
    from kobushi_core.model import OpenAIModelAdapter
    from experiments.exp_141_llm_judge_vote.llm_judge import judge_vote
    from experiments.exp_141_llm_judge_vote.adaptive_vote import adaptive_vote

    RUN_DIRS = [
        ROOT / "artifacts" / "runs" / "exp_137_math_advisor_001",
        ROOT / "artifacts" / "runs" / "exp_137_math_advisor_002",
        ROOT / "artifacts" / "runs" / "exp_137_math_advisor_003",
    ]
    GOLD_DIR = ROOT / "data" / "public" / "output"
    OUT = ROOT / "artifacts" / "exp141_judge_past_runs"
    OUT.mkdir(parents=True, exist_ok=True)

    ds = DABenchPublicDataset(root_dir=ROOT / "data" / "public" / "input")

    model = OpenAIModelAdapter(
        model="qwen3.5-35b-a3b",
        api_base="http://localhost:8000/v1",
        api_key="local-key",
        temperature=0.0,
    )

    def to_answer(d: dict | None) -> AnswerTable | None:
        if not d:
            return None
        cols = d.get("columns")
        rows = d.get("rows")
        if not cols or not isinstance(rows, list):
            return None
        return AnswerTable(columns=tuple(cols), rows=tuple(tuple(r) for r in rows))

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
            e = _evaluate_task(task_id=tid, prediction_path=pred, gold_path=gold,
                               options=EvaluationOptions())
            return float(e.official_score_lambda_0_5)
        finally:
            pred.unlink(missing_ok=True)

    all_rows: list[dict] = []
    per_run_stats: dict[str, dict] = {}
    t_all = time.time()

    for run_dir in RUN_DIRS:
        rname = run_dir.name
        print(f"\n=== {rname} ===", flush=True)
        per_run_stats[rname] = {
            "n_tasks": 0,
            "rule_sum": 0.0,
            "judge_sum": 0.0,
            "single_avg_sum": 0.0,
            "best_of_n_sum": 0.0,
            "n_judge_called": 0,
        }
        for tdir in sorted(run_dir.glob("task_*")):
            tid = tdir.name
            tr_path = tdir / "trace.json"
            if not tr_path.exists():
                continue
            try:
                attempts_raw = json.loads(tr_path.read_text())
            except Exception:
                continue
            if not isinstance(attempts_raw, list) or len(attempts_raw) < 2:
                continue

            answers: list[AnswerTable | None] = []
            steps_list: list[list[dict]] = []
            per_attempt: list[float] = []
            for att in attempts_raw:
                ans = to_answer(att.get("answer"))
                answers.append(ans)
                steps_list.append(att.get("steps") or [])
                per_attempt.append(score_answer(ans, tid))

            # Rule vote
            non_empty = [a for a in answers if a is not None]
            rule_ans = adaptive_vote(non_empty) if non_empty else None
            rule_score = score_answer(rule_ans, tid)

            # Judge vote
            question = ds.get_task(tid).question
            t0 = time.time()
            judge_ans, meta = judge_vote(
                answers, question, model,
                steps_per_attempt=steps_list,
            )
            judge_dt = time.time() - t0
            judge_score = score_answer(judge_ans, tid)

            if not meta.get("skipped_judge"):
                per_run_stats[rname]["n_judge_called"] += 1

            single_avg = sum(per_attempt) / len(per_attempt)
            best_of_n = max(per_attempt) if per_attempt else 0.0

            per_run_stats[rname]["n_tasks"] += 1
            per_run_stats[rname]["rule_sum"] += rule_score
            per_run_stats[rname]["judge_sum"] += judge_score
            per_run_stats[rname]["single_avg_sum"] += single_avg
            per_run_stats[rname]["best_of_n_sum"] += best_of_n

            all_rows.append({
                "run": rname,
                "tid": tid,
                "per_attempt": per_attempt,
                "rule_vote": rule_score,
                "judge_vote": judge_score,
                "single_avg": single_avg,
                "best_of_n": best_of_n,
                "judge_meta": meta,
                "judge_dt": round(judge_dt, 2),
            })
            tag = "→J" if not meta.get("skipped_judge") else "sk"
            delta = judge_score - rule_score
            sym = " " if abs(delta) < 0.01 else ("+" if delta > 0 else "-")
            print(
                f"  {tid:<12} attempts={per_attempt} rule={rule_score:.2f} "
                f"judge={judge_score:.2f} {sym}{abs(delta):.2f} {tag} ({judge_dt:.1f}s)",
                flush=True,
            )

    elapsed = time.time() - t_all

    print()
    print(f"=== Aggregate over {len(per_run_stats)} runs ===")
    grand = {"rule": 0.0, "judge": 0.0, "single_avg": 0.0, "best_of_n": 0.0, "n": 0, "j": 0}
    for rname, s in per_run_stats.items():
        n = s["n_tasks"]
        if n == 0:
            continue
        rmean = s["rule_sum"] / n
        jmean = s["judge_sum"] / n
        smean = s["single_avg_sum"] / n
        bmean = s["best_of_n_sum"] / n
        print(f"  {rname:<35} n={n}  single={smean:.4f}  rule={rmean:.4f}  judge={jmean:.4f}  best-of-n={bmean:.4f}  j_called={s['n_judge_called']}")
        grand["rule"] += s["rule_sum"]
        grand["judge"] += s["judge_sum"]
        grand["single_avg"] += s["single_avg_sum"]
        grand["best_of_n"] += s["best_of_n_sum"]
        grand["n"] += n
        grand["j"] += s["n_judge_called"]
    if grand["n"]:
        print()
        print(f"  POOLED (n={grand['n']} tasks): "
              f"single={grand['single_avg']/grand['n']:.4f}  "
              f"rule={grand['rule']/grand['n']:.4f}  "
              f"judge={grand['judge']/grand['n']:.4f}  "
              f"best-of-n={grand['best_of_n']/grand['n']:.4f}  "
              f"j_called={grand['j']}")
    print(f"  elapsed: {elapsed/60:.1f} min")

    (OUT / "summary.json").write_text(json.dumps({
        "per_run": per_run_stats,
        "pooled": {
            "n_tasks": grand["n"],
            "single_avg": grand["single_avg"]/grand["n"] if grand["n"] else 0,
            "rule": grand["rule"]/grand["n"] if grand["n"] else 0,
            "judge": grand["judge"]/grand["n"] if grand["n"] else 0,
            "best_of_n": grand["best_of_n"]/grand["n"] if grand["n"] else 0,
            "n_judge_called": grand["j"],
        },
        "elapsed_min": round(elapsed/60, 1),
    }, indent=2))
    (OUT / "results.json").write_text(json.dumps(all_rows, indent=2, default=str))


if __name__ == "__main__":
    main()
