"""exp_140 gpuhost FP8, n=1 single-attempt, workers=12.

Bypasses the 3-attempt spawn + adaptive_vote pipeline. Each task runs in a
single agent.run() call. Provides faster wall-time + cleaner variance estimate
(= no vote-induced compounding).
"""
from __future__ import annotations
import csv as _csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

MAX_WORKERS = 12


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_one(tid: str, gold_dir: Path, ds, out_dir: Path):
    from kobushi_core.eval.csv_compare import EvaluationOptions, _evaluate_task
    from kobushi_core.model import OpenAIModelAdapter
    from experiments.exp_140_agentar_icl.preamble import build_preamble
    from experiments.exp_140_agentar_icl.phased_agent import (
        PhasedAgentConfig, PhasedReActAgent,
    )
    from experiments.exp_140_agentar_icl.tools.registry import (
        create_default_tool_registry,
    )

    task = ds.get_task(tid)
    td = out_dir / tid
    td.mkdir(parents=True, exist_ok=True)
    (td / "trace.log").write_text("")

    t0 = time.time()
    score = 0.0
    n_steps = 0
    err = None
    try:
        model = OpenAIModelAdapter(
            model=os.environ.get("AGENT_MODEL", "qwen3.5-35b-a3b"),
            api_base=os.environ["AGENT_API_BASE"],
            api_key=os.environ["AGENT_API_KEY"],
            temperature=0.6,
            extra_headers={
                "CF-Access-Client-Id": os.environ.get("CF_ACCESS_CLIENT_ID", ""),
                "CF-Access-Client-Secret": os.environ.get("CF_ACCESS_CLIENT_SECRET", ""),
            },
        )
        preamble = build_preamble(task)
        tools = create_default_tool_registry(
            auditor_model=model, question_provider=lambda: task.question,
            context_dir=task.context_dir,
        )
        agent = PhasedReActAgent(
            model=model, tools=tools,
            config=PhasedAgentConfig(max_steps=64, min_explore_queries=3),
            preamble=preamble.text,
        )
        agent.trace_log_path = str(td / "trace.log")
        r = agent.run(task)
        n_steps = len(r.steps)
        if r.answer and r.answer.rows:
            pred = td / "prediction.csv"
            with pred.open("w", newline="") as f:
                w = _csv.writer(f)
                w.writerow(r.answer.columns)
                for row in r.answer.rows:
                    w.writerow(row)
            gold = gold_dir / tid / "gold.csv"
            e = _evaluate_task(task_id=tid, prediction_path=pred, gold_path=gold,
                               options=EvaluationOptions())
            score = float(e.official_score_lambda_0_5)
    except Exception as e:
        err = f"{type(e).__name__}: {str(e)[:200]}"
    return {"tid": tid, "score": score, "elapsed_s": round(time.time() - t0, 1),
            "n_steps": n_steps, "error": err}


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    # gpuhost endpoint (= already set in .env via AGENT_API_BASE)
    from kobushi_core.benchmark import DABenchPublicDataset

    _runs = ROOT / "artifacts" / "runs"
    _i = 1
    while (_runs / f"exp140_gpuhost_1att_{_i:03d}").exists():
        _i += 1
    OUT = _runs / f"exp140_gpuhost_1att_{_i:03d}"
    OUT.mkdir(parents=True, exist_ok=True)
    _log(f"=== exp_140 gpuhost 1-att bench (workers={MAX_WORKERS}) ===")
    _log(f"output: {OUT}")

    ds = DABenchPublicDataset(root_dir=ROOT / "data" / "public" / "input")
    tids = sorted(ds.list_task_ids())
    gold_dir = ROOT / "data" / "public" / "output"

    results = []
    t_all = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(run_one, tid, gold_dir, ds, OUT): tid for tid in tids}
        for i, fut in enumerate(as_completed(futures), 1):
            tid = futures[fut]
            try:
                r = fut.result()
            except Exception as e:
                r = {"tid": tid, "score": 0.0,
                     "error": f"outer {type(e).__name__}: {str(e)[:200]}"}
            results.append(r)
            tag = "✓" if r["score"] >= 0.99 else ("✗" if r["score"] < 0.01 else "○")
            running = sum(x["score"] for x in results) / len(results)
            _log(f"  [{i:2d}/{len(tids)}] {tag} {r['tid']}: score={r['score']:.2f} "
                 f"t={r.get('elapsed_s', 0):.0f}s n={r.get('n_steps', 0)}  running={running:.4f}")

    total = time.time() - t_all
    mean = sum(r["score"] for r in results) / len(results)
    perfect = sum(1 for r in results if r["score"] >= 0.99)
    zero = sum(1 for r in results if r["score"] < 0.01)
    _log(f"=== DONE n={len(results)}, mean={mean:.4f}, perfect={perfect}, zero={zero}, "
         f"elapsed={total/60:.1f}min ===")

    (OUT / "summary.json").write_text(json.dumps({
        "n_tasks": len(results),
        "mean_score": mean,
        "n_perfect": perfect,
        "n_zero": zero,
        "elapsed_minutes": round(total / 60, 1),
        "config": {
            "endpoint": "gpuhost FP8 (https://gpu-host.internal/v1)",
            "experiment": "exp_140_agentar_icl",
            "n_attempts": 1,
            "max_workers": MAX_WORKERS,
            "max_steps": 64,
        },
    }, indent=2))
    (OUT / "results.json").write_text(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
