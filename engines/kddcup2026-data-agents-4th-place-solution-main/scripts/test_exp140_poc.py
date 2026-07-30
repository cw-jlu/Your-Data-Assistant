"""POC: exp_140 BIRD-ICL on a curated 5-task subset.

Subset (= mix of local-specific failures + clean baselines):
  - task_303 (= percentage, local zero, exp_137 8/9 → does ICL help?)
  - task_200 (= chemistry, local zero, exp_137 7/9 → ICL likely weak signal)
  - task_25  (= chemistry, local pass)
  - task_22  (= movie pattern, local pass)
  - task_180 (= partial 0.75 baseline)
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


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    os.environ["AGENT_API_BASE"] = "http://localhost:8000/v1"
    os.environ["AGENT_API_KEY"] = "local-key"
    os.environ["AGENT_MODEL"] = "qwen3.5-35b-a3b"
    os.environ["CF_ACCESS_CLIENT_ID"] = ""
    os.environ["CF_ACCESS_CLIENT_SECRET"] = ""

    from kobushi_core.eval.csv_compare import EvaluationOptions, _evaluate_task
    from kobushi_core.benchmark import DABenchPublicDataset
    from kobushi_core.model import OpenAIModelAdapter
    from experiments.exp_140_agentar_icl.preamble import build_preamble
    from experiments.exp_140_agentar_icl.phased_agent import (
        PhasedAgentConfig,
        PhasedReActAgent,
    )
    from experiments.exp_140_agentar_icl.tools.registry import (
        create_default_tool_registry,
    )

    TIDS = ["task_303", "task_200", "task_25", "task_22", "task_180"]

    ds = DABenchPublicDataset(root_dir=ROOT / "data" / "public" / "input")
    out_root = ROOT / "artifacts" / "exp140_poc"
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[{time.strftime('%H:%M:%S')}] starting POC ({len(TIDS)} tasks)")
    results: list[dict] = []
    t_all = time.time()

    for tid in TIDS:
        task = ds.get_task(tid)
        print(f"\n[{time.strftime('%H:%M:%S')}] === {tid}: {task.question[:90]}")

        model = OpenAIModelAdapter(
            model="qwen3.5-35b-a3b",
            api_base="http://localhost:8000/v1",
            api_key="local-key",
            temperature=0.6,
        )
        preamble = build_preamble(task)
        tools = create_default_tool_registry(
            auditor_model=model,
            question_provider=lambda: task.question,
            context_dir=task.context_dir,
        )
        agent = PhasedReActAgent(
            model=model,
            tools=tools,
            config=PhasedAgentConfig(max_steps=64, min_explore_queries=3),
            preamble=preamble.text,
        )
        td = out_root / tid
        td.mkdir(parents=True, exist_ok=True)
        (td / "trace.log").write_text("")
        agent.trace_log_path = str(td / "trace.log")

        t0 = time.time()
        score = 0.0
        try:
            r = agent.run(task)
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {str(e)[:200]}")
            results.append({"tid": tid, "score": 0.0,
                            "elapsed_s": round(time.time() - t0, 1),
                            "error": f"{type(e).__name__}: {str(e)[:120]}"})
            continue
        elapsed = time.time() - t0
        if r.answer and r.answer.rows:
            pred = td / "prediction.csv"
            with pred.open("w", newline="") as f:
                w = _csv.writer(f)
                w.writerow(r.answer.columns)
                for row in r.answer.rows:
                    w.writerow(row)
            gold = ROOT / "data" / "public" / "output" / tid / "gold.csv"
            e = _evaluate_task(task_id=tid, prediction_path=pred, gold_path=gold,
                               options=EvaluationOptions())
            score = float(e.official_score_lambda_0_5)
        results.append({"tid": tid, "score": score,
                        "elapsed_s": round(elapsed, 1),
                        "n_steps": len(r.steps), "succeeded": r.succeeded})
        tag = "✓" if score >= 0.99 else ("✗" if score < 0.01 else "○")
        print(f"  {tag} done in {elapsed:.1f}s, score={score:.2f}, n_steps={len(r.steps)}")

    total_elapsed = time.time() - t_all
    mean = sum(r['score'] for r in results) / len(results)
    print(f"\n[{time.strftime('%H:%M:%S')}] === DONE n={len(results)}, mean={mean:.3f}, "
          f"elapsed={total_elapsed/60:.1f}min ===")
    for r in results:
        tag = "✓" if r['score'] >= 0.99 else ("✗" if r['score'] < 0.01 else "○")
        print(f"  {tag} {r['tid']}: score={r['score']:.2f} t={r['elapsed_s']}s")

    (out_root / "summary.json").write_text(json.dumps(
        {"results": results, "mean": mean, "elapsed_minutes": round(total_elapsed/60, 1)},
        indent=2, default=str))


if __name__ == "__main__":
    main()
