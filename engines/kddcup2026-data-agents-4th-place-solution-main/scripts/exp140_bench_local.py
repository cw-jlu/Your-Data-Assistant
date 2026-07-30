"""Full 50-task bench: exp_140 (= BIRD-ICL) on local llama.cpp Q4_K_M, single-attempt.

Same protocol as scripts/math_advisor_bench_local.py for direct comparability:
- Sequential tasks (= llama.cpp parallel=1)
- 1 attempt per task (= no spawn, no adaptive_vote)
- Official scorer (= kobushi_core.eval._evaluate_task λ=0.5)
"""
from __future__ import annotations
import csv as _csv
import json
import os
import sys
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

    from kobushi_core.eval.csv_compare import _evaluate_task, EvaluationOptions
    from kobushi_core.benchmark import DABenchPublicDataset
    from kobushi_core.model import OpenAIModelAdapter
    from experiments.exp_140_agentar_icl.preamble import build_preamble
    from experiments.exp_140_agentar_icl.phased_agent import PhasedReActAgent, PhasedAgentConfig
    from experiments.exp_140_agentar_icl.tools.registry import create_default_tool_registry

    _runs = ROOT / "artifacts" / "runs"
    _i = 1
    while (_runs / f"exp140_icl_local_{_i:03d}").exists():
        _i += 1
    OUT = _runs / f"exp140_icl_local_{_i:03d}"
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"[init] output dir: {OUT}", flush=True)

    ds = DABenchPublicDataset(root_dir=ROOT / "data" / "public" / "input")
    tids = sorted(ds.list_task_ids())
    print(f"[{time.strftime('%H:%M:%S')}] tasks={len(tids)}, single-attempt on local llama.cpp + BIRD-ICL", flush=True)

    results = []
    t_all = time.time()
    for i, tid in enumerate(tids, 1):
        task = ds.get_task(tid)
        model = OpenAIModelAdapter(
            model="qwen3.5-35b-a3b",
            api_base="http://localhost:8000/v1",
            api_key="local-key",
            temperature=0.6,
        )
        preamble = build_preamble(task)
        tools = create_default_tool_registry(
            auditor_model=model, question_provider=lambda: task.question, context_dir=task.context_dir,
        )
        agent = PhasedReActAgent(
            model=model, tools=tools,
            config=PhasedAgentConfig(max_steps=64, min_explore_queries=3),
            preamble=preamble.text,
        )
        td = OUT / tid
        td.mkdir(parents=True, exist_ok=True)
        (td / "trace.log").write_text("")
        agent.trace_log_path = str(td / "trace.log")

        t0 = time.time()
        score = 0.0
        n_steps = 0
        err = None
        try:
            r = agent.run(task)
            n_steps = len(r.steps)
            if r.answer and r.answer.rows:
                pred = td / "prediction.csv"
                with pred.open("w", newline="") as f:
                    w = _csv.writer(f)
                    w.writerow(r.answer.columns)
                    for row in r.answer.rows:
                        w.writerow(row)
                gold = ROOT / "data" / "public" / "output" / tid / "gold.csv"
                e = _evaluate_task(task_id=tid, prediction_path=pred, gold_path=gold, options=EvaluationOptions())
                score = float(e.official_score_lambda_0_5)
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:200]}"
        elapsed = time.time() - t0
        results.append({"tid": tid, "score": score, "elapsed_s": round(elapsed, 1),
                        "n_steps": n_steps, "error": err})
        tag = "✓" if score >= 0.99 else ("✗" if score < 0.01 else "○")
        running_mean = sum(r['score'] for r in results) / len(results)
        print(f"  [{i:2d}/{len(tids)}] {tag} {tid}: score={score:.2f} t={elapsed:.0f}s n={n_steps}  running={running_mean:.4f}", flush=True)

    total_elapsed = time.time() - t_all
    mean = sum(r['score'] for r in results) / len(results)
    perfect = sum(1 for r in results if r['score'] >= 0.99)
    zero = sum(1 for r in results if r['score'] < 0.01)
    print(f"\n=== DONE n={len(results)}, mean={mean:.4f}, perfect={perfect}, zero={zero}, elapsed={total_elapsed/60:.1f}min ===", flush=True)
    (OUT / "summary.json").write_text(json.dumps({
        "n_tasks": len(results),
        "mean_score": mean,
        "n_perfect": perfect,
        "n_zero": zero,
        "elapsed_minutes": round(total_elapsed/60, 1),
        "config": {
            "endpoint": "http://localhost:8000/v1 (= local llama.cpp Q4_K_M)",
            "n_attempts_per_task": 1,
            "max_steps": 64,
            "experiment": "exp_140_agentar_icl",
            "icl_top_k": 5,
            "icl_min_sim": 0.40,
            "icl_embedding": "all-MiniLM-L6-v2",
            "icl_source": "BIRD train (9428)",
            "scorer": "kobushi_core.eval._evaluate_task (λ=0.5)",
        }
    }, indent=2))
    (OUT / "results.json").write_text(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
