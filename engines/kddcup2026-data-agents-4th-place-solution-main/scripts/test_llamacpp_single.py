"""Single-attempt smoke against llama.cpp (= no spawn, no vote).
Direct PhasedReActAgent call to verify llama.cpp compatibility end-to-end.
"""
from __future__ import annotations
import csv as _csv
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    # Override to llama.cpp
    os.environ["AGENT_API_BASE"] = "http://localhost:8000/v1"
    os.environ["AGENT_API_KEY"] = "local-key"
    os.environ["AGENT_MODEL"] = "qwen3.5-35b-a3b"
    os.environ["CF_ACCESS_CLIENT_ID"] = ""
    os.environ["CF_ACCESS_CLIENT_SECRET"] = ""

    from kobushi_core.eval.csv_compare import _evaluate_task, EvaluationOptions
    from kobushi_core.benchmark import DABenchPublicDataset
    from kobushi_core.model import OpenAIModelAdapter
    from experiments.exp_137_math_advisor.preamble import build_preamble
    from experiments.exp_137_math_advisor.phased_agent import PhasedReActAgent, PhasedAgentConfig
    from experiments.exp_137_math_advisor.tools.registry import create_default_tool_registry

    TID = "task_25"

    ds = DABenchPublicDataset(root_dir=ROOT / "data" / "public" / "input")
    task = ds.get_task(TID)
    print(f"[{time.strftime('%H:%M:%S')}] Q: {task.question}", flush=True)

    model = OpenAIModelAdapter(
        model="qwen3.5-35b-a3b",
        api_base="http://localhost:8000/v1",
        api_key="local-key",
        temperature=0.6,
    )

    preamble = build_preamble(task)
    print(f"[{time.strftime('%H:%M:%S')}] preamble built: {len(preamble.text)} chars", flush=True)

    tools = create_default_tool_registry(
        auditor_model=model, question_provider=lambda: task.question, context_dir=task.context_dir,
    )
    agent = PhasedReActAgent(
        model=model, tools=tools,
        config=PhasedAgentConfig(max_steps=64, min_explore_queries=3),
        preamble=preamble.text,
    )
    trace_path = ROOT / "artifacts" / "llamacpp_single" / TID / "trace.log"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text("")
    agent.trace_log_path = str(trace_path)

    print(f"[{time.strftime('%H:%M:%S')}] agent.run() starting...", flush=True)
    t0 = time.time()
    r = agent.run(task)
    elapsed = time.time() - t0
    print(f"[{time.strftime('%H:%M:%S')}] done in {elapsed:.1f}s, succeeded={r.succeeded}, n_steps={len(r.steps)}", flush=True)

    if r.answer and r.answer.rows:
        out_dir = trace_path.parent
        pred = out_dir / "prediction.csv"
        with pred.open("w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(r.answer.columns)
            for row in r.answer.rows:
                w.writerow(row)
        gold = ROOT / "data" / "public" / "output" / TID / "gold.csv"
        e = _evaluate_task(task_id=TID, prediction_path=pred, gold_path=gold, options=EvaluationOptions())
        print(f"  ANSWER cols={list(r.answer.columns)}, rows={[list(row) for row in r.answer.rows[:5]]}")
        print(f"  GOLD: {gold.read_text().strip()}")
        print(f"  OFFICIAL SCORE: {e.official_score_lambda_0_5}")
    else:
        print("  NO ANSWER")


if __name__ == "__main__":
    main()
