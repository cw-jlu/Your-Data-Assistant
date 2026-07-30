"""Phase 2 first baseline run — exp_142_phase2_baseline (= v11 + video).

Same per-task pipeline as v11 (= scripts/math_advisor_full50.py):
  1. math_expert(question) → one-line pseudo-formula
  2. inject formula into preamble (skip on NO_CALC)
  3. PhasedReActAgent × N_ATTEMPTS attempts
  4. adaptive_vote → prediction.csv
  5. trace.json per attempt saved

Phase 2 deltas:
  - DATA roots point to data/phase2_demo/demo_samples_phase2/
  - PhasedReActAgent in exp_142 auto-attaches any video found under
    context/ to the first user message (= multimodal content list).
  - N_ATTEMPTS=1, MAX_WORKERS=12 for first sample.
"""
from __future__ import annotations

import csv, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from dotenv import load_dotenv
load_dotenv(REPO / ".env")

from kobushi_core.benchmark.dataset import DABenchPublicDataset
from kobushi_core.model import OpenAIModelAdapter
from experiments.exp_142_phase2_baseline.phased_agent import PhasedReActAgent, PhasedAgentConfig
from experiments.exp_142_phase2_baseline.preamble import build_preamble
from experiments.exp_142_phase2_baseline.tools.registry import create_default_tool_registry
from experiments.exp_142_phase2_baseline.adaptive_vote import adaptive_vote
from experiments.exp_142_phase2_baseline.math_advisor import generate_formula

# Phase 2 demo dataset paths
DATA_ROOT = REPO / "data" / "phase2_demo" / "demo_samples_phase2"
INPUT_ROOT = DATA_ROOT / "input"
GOLD_ROOT = DATA_ROOT / "output"

# Auto-pick next free dir
EXP_NAME = "exp_142_phase2_baseline"
_runs = REPO / "artifacts" / "runs"
_runs.mkdir(parents=True, exist_ok=True)
_i = 1
while (_runs / f"{EXP_NAME}_{_i:03d}").exists():
    _i += 1
OUT = _runs / f"{EXP_NAME}_{_i:03d}"
OUT.mkdir(parents=True, exist_ok=True)
print(f"[init] output dir: {OUT}", flush=True)

N_ATTEMPTS = 1
MAX_WORKERS = 12
MAX_STEPS = 64


def make_model(temp: float = 0.6) -> OpenAIModelAdapter:
    return OpenAIModelAdapter(
        model="qwen3.5-35b-a3b",
        api_base=os.environ.get("AGENT_API_BASE", ""),
        api_key=os.environ.get("AGENT_API_KEY", ""),
        temperature=temp,
        # CF Access headers via extra_headers
        extra_headers={
            "CF-Access-Client-Id": os.environ.get("CF_ACCESS_CLIENT_ID", ""),
            "CF-Access-Client-Secret": os.environ.get("CF_ACCESS_CLIENT_SECRET", ""),
        },
    )


def _official_score(pred_csv: Path, gold_csv: Path) -> float:
    """Official column-multiset λ=0.5 score (= same as LB metric)."""
    from kobushi_core.eval.csv_compare import _evaluate_task, EvaluationOptions
    if not gold_csv.is_file():
        return 0.0
    try:
        ev = _evaluate_task(
            task_id=str(pred_csv.parent.name),
            prediction_path=pred_csv,
            gold_path=gold_csv,
            options=EvaluationOptions(),
        )
        return float(ev.official_score_lambda_0_5)
    except Exception as e:
        print(f"  [score err] {e}", flush=True)
        return 0.0


def run_one(tid: str) -> dict:
    t0 = time.time()
    ds = DABenchPublicDataset(root_dir=INPUT_ROOT)
    try:
        task = ds.get_task(tid)
    except Exception as e:
        return {"tid": tid, "error": f"load: {e}", "score": 0.0, "elapsed": time.time() - t0}

    task_dir = OUT / tid
    task_dir.mkdir(parents=True, exist_ok=True)

    formula = "NO_CALC"
    try:
        formula = generate_formula(task.question)
    except Exception as e:
        formula = f"NO_CALC ({e})"
    (task_dir / "formula.txt").write_text(formula)
    skip_advisor = formula.strip().upper().startswith("NO_CALC")
    (task_dir / "advisor_used.txt").write_text("False" if skip_advisor else "True")

    answers = []
    attempt_traces = []
    for i in range(N_ATTEMPTS):
        try:
            m = make_model(temp=0.6)
            preamble = build_preamble(task)
            if skip_advisor:
                injected = preamble.text
            else:
                injected = (
                    "# MATH FORMULA HINT (= expert calculation guide)\n"
                    f"  {formula}\n\n"
                    "Follow this aggregation/division/filter structure EXACTLY.\n\n"
                ) + preamble.text
            tools = create_default_tool_registry(
                auditor_model=m, question_provider=lambda: task.question, context_dir=task.context_dir,
            )
            agent = PhasedReActAgent(
                model=m, tools=tools,
                config=PhasedAgentConfig(max_steps=MAX_STEPS, min_explore_queries=3),
                preamble=injected,
            )
            r = agent.run(task)
            if r.answer and r.answer.rows:
                answers.append(r.answer)
            attempt_traces.append({
                "attempt": i,
                "answer": {"columns": list(r.answer.columns), "rows": [list(row) for row in r.answer.rows]} if r.answer else None,
                "failure_reason": getattr(r, "failure_reason", None),
                "steps": [
                    {
                        "i": j,
                        "action": s.action,
                        "action_input": s.action_input,
                        "thought": (s.thought or "")[:2000],
                        "observation_ok": s.observation.get("ok") if isinstance(s.observation, dict) else None,
                        "observation_content_preview": str(s.observation.get("content") if isinstance(s.observation, dict) else s.observation)[:400],
                    }
                    for j, s in enumerate(r.steps)
                ],
            })
        except Exception as e:
            attempt_traces.append({"attempt": i, "error": str(e)[:300]})

    (task_dir / "trace.json").write_text(json.dumps(attempt_traces, ensure_ascii=False, indent=2, default=str))

    if not answers:
        return {"tid": tid, "formula": formula, "score": 0.0, "n_ok": 0,
                "elapsed": time.time() - t0}
    voted = adaptive_vote(answers)
    if voted is None or not voted.rows:
        return {"tid": tid, "formula": formula, "score": 0.0, "n_ok": len(answers),
                "elapsed": time.time() - t0}
    pred = task_dir / "prediction.csv"
    with open(pred, "w", newline="") as f:
        w = csv.writer(f); w.writerow(voted.columns); [w.writerow(r) for r in voted.rows]
    gold = GOLD_ROOT / tid / "gold.csv"
    score = _official_score(pred, gold)
    return {"tid": tid, "formula": formula, "score": score, "n_ok": len(answers),
            "cols": list(voted.columns), "elapsed": time.time() - t0}


def main():
    ds = DABenchPublicDataset(root_dir=INPUT_ROOT)
    task_ids = sorted([t.task_id for t in ds.iter_tasks()],
                      key=lambda x: int(x.split("_")[1]))
    print(f"=== exp_142 Phase 2 baseline: workers={MAX_WORKERS}, n_attempts={N_ATTEMPTS}, "
          f"max_steps={MAX_STEPS}, T=0.6, dataset={len(task_ids)} tasks ===\n", flush=True)

    results = []
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(run_one, tid): tid for tid in task_ids}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            results.append(r)
            mark = "✓" if r.get("score", 0) >= 0.999 else "✗"
            err = (r.get("error") or "")[:50]
            running = sum(rr["score"] for rr in results) / len(results)
            print(f"[{i:3d}/{len(task_ids)}] {mark} {r['tid']:<10} "
                  f"score={r.get('score',0):.2f} n_ok={r.get('n_ok',0)} "
                  f"t={r.get('elapsed',0):.0f}s mean={running:.4f} {err}", flush=True)

    elapsed = time.time() - t_start
    mean = sum(r["score"] for r in results) / len(results) if results else 0
    n_perfect = sum(1 for r in results if r["score"] >= 0.999)
    n_zero = sum(1 for r in results if r["score"] < 0.01)
    summary = {
        "n_tasks": len(results),
        "mean_score": mean,
        "n_perfect": n_perfect,
        "n_zero": n_zero,
        "elapsed_minutes": elapsed / 60,
        "config": {
            "n_attempts": N_ATTEMPTS, "max_workers": MAX_WORKERS,
            "max_steps": MAX_STEPS, "model": "qwen3.5-35b-a3b",
            "temperature": 0.6, "data_root": str(INPUT_ROOT),
        },
        "results": results,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print()
    print(f"=== DONE mean={mean:.4f} perfect={n_perfect}/{len(results)} "
          f"zero={n_zero}, elapsed={elapsed/60:.1f}min ===")
    print(f"saved {OUT}/summary.json")


if __name__ == "__main__":
    main()
