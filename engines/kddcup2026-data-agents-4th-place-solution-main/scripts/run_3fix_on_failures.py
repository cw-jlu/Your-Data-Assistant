"""Run 4 variants × 10 failing tasks × 3 attempts. Compare to v5 baseline.

Variants:
  - exp_122_v6_compare    (= baseline)
  - exp_134_zero_rows_hint
  - exp_135_sample_values
  - exp_136_doc_marker

Failing tasks (= v5 GT score < 1.0):
  task_80, task_89, task_163, task_169, task_180,
  task_199, task_200, task_344, task_379, task_396

For each (variant, task) pair, run PhasedReActAgent 3 times, then adaptive_vote.
Score against gold.csv via the official evaluator.

Total: 4 × 10 × 3 = 120 agent runs.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from dotenv import load_dotenv
load_dotenv(REPO / ".env")

from kobushi_core.benchmark.dataset import DABenchPublicDataset
from kobushi_core.model import OpenAIModelAdapter
from kobushi_core.eval import EvaluationOptions, evaluate_run

OUT_DIR = REPO / "artifacts" / "3fix_failures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FAILING = ["task_80", "task_89", "task_163", "task_169", "task_180",
           "task_199", "task_200", "task_344", "task_379", "task_396"]

VARIANTS = [
    ("baseline", "exp_122_v6_compare"),
    ("zero_rows_hint", "exp_134_zero_rows_hint"),
    ("sample_values", "exp_135_sample_values"),
    ("doc_marker", "exp_136_doc_marker"),
]

N_ATTEMPTS = 3
MAX_WORKERS = 2  # outer × 3 inner = 6 streams (= alongside v5_v6 30 streams, total 36)


def make_model() -> OpenAIModelAdapter:
    return OpenAIModelAdapter(
        model="qwen3.5-35b-a3b",
        api_base=os.environ["AGENT_API_BASE"],
        api_key=os.environ["AGENT_API_KEY"],
        temperature=0.6,
        extra_headers={
            "CF-Access-Client-Id": os.environ.get("CF_ACCESS_CLIENT_ID", ""),
            "CF-Access-Client-Secret": os.environ.get("CF_ACCESS_CLIENT_SECRET", ""),
        },
    )


def run_one_attempt(variant_module: str, task_id: str, attempt_idx: int):
    PhasedReActAgent = importlib.import_module(
        f"experiments.{variant_module}.phased_agent"
    ).PhasedReActAgent
    PhasedAgentConfig = importlib.import_module(
        f"experiments.{variant_module}.phased_agent"
    ).PhasedAgentConfig
    build_preamble = importlib.import_module(
        f"experiments.{variant_module}.preamble"
    ).build_preamble
    create_default_tool_registry = importlib.import_module(
        f"experiments.{variant_module}.tools.registry"
    ).create_default_tool_registry

    ds = DABenchPublicDataset(root_dir=REPO / "data" / "public" / "input")
    task = ds.get_task(task_id)
    model = make_model()
    preamble = build_preamble(task)
    tools = create_default_tool_registry(
        auditor_model=model,
        question_provider=lambda: task.question,
        context_dir=task.context_dir,
    )
    agent = PhasedReActAgent(
        model=model, tools=tools,
        config=PhasedAgentConfig(max_steps=32, min_explore_queries=3),
        preamble=preamble.text,
    )
    try:
        result = agent.run(task)
    except Exception as exc:
        return None, str(exc)[:200]
    return result, None


def voted_score_for_task(variant_module: str, task_id: str, results: list) -> float:
    """Build adaptive_vote, write prediction.csv, score it."""
    adaptive_vote = importlib.import_module(
        f"experiments.{variant_module}.adaptive_vote"
    ).adaptive_vote
    answers = [r.answer for r in results if r and r.answer and r.answer.rows]
    if not answers:
        return 0.0
    voted = adaptive_vote(answers)
    if voted is None or not voted.rows:
        return 0.0
    # Write prediction.csv to a temp dir
    task_dir = OUT_DIR / "temp" / variant_module / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    pred_path = task_dir / "prediction.csv"
    import csv
    with open(pred_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(voted.columns)
        for row in voted.rows:
            w.writerow(row)
    # Score via official eval
    try:
        from kobushi_core.benchmark.schema import PublicTask
        ds = DABenchPublicDataset(root_dir=REPO / "data" / "public" / "input")
        task = ds.get_task(task_id)
        # Read gold
        gold_path = REPO / "data" / "public" / "output" / task_id / "gold.csv"
        from kobushi_core.eval import score_one_task
        score = score_one_task(
            task=task, prediction_csv=pred_path, gold_csv=gold_path,
            options=EvaluationOptions(lambda_=0.5, unordered=True),
        )
        return float(score.lambda_score) if hasattr(score, "lambda_score") else float(score)
    except Exception as exc:
        print(f"  scoring error {task_id}: {exc}")
        return 0.0


def run_one_task(variant_label: str, variant_module: str, task_id: str) -> dict:
    t0 = time.time()
    answers = []
    errors = []
    for i in range(N_ATTEMPTS):
        result, err = run_one_attempt(variant_module, task_id, i)
        if err:
            errors.append(err)
        if result:
            answers.append(result)
    score = voted_score_for_task(variant_module, task_id, answers)
    return {
        "variant": variant_label,
        "task_id": task_id,
        "score": score,
        "n_attempts_ok": len(answers),
        "errors": errors[:1],
        "elapsed_s": int(time.time() - t0),
    }


def main():
    print(f"=== 3-fix on failing tasks: {len(VARIANTS)} variants × {len(FAILING)} tasks × {N_ATTEMPTS} attempts ===", flush=True)
    jobs = [(label, mod, tid) for label, mod in VARIANTS for tid in FAILING]
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(run_one_task, *j): j for j in jobs}
        done = 0
        for fut in as_completed(futs):
            label, mod, tid = futs[fut]
            done += 1
            try:
                r = fut.result()
            except Exception as exc:
                r = {"variant": label, "task_id": tid, "error": str(exc), "score": 0.0}
            results.append(r)
            print(f"  [{done:3d}/{len(jobs)}] {r['variant']:18s} {r['task_id']}: score={r.get('score',0.0):.2f} ({r.get('elapsed_s','?')}s)", flush=True)

    # Aggregate per variant
    import csv
    with open(OUT_DIR / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["variant", "task_id", "score", "n_attempts_ok", "elapsed_s"], extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    from collections import defaultdict
    by_variant = defaultdict(list)
    for r in results:
        by_variant[r["variant"]].append(r.get("score", 0))
    print(f"\n=== Per-variant mean score (10 failing tasks) ===")
    for v, scores in by_variant.items():
        print(f"  {v}: {sum(scores)/len(scores):.3f}  ({[f'{s:.2f}' for s in scores]})")


if __name__ == "__main__":
    main()
