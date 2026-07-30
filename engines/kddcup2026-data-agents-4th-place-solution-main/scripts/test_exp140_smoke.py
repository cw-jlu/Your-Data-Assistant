"""Smoke test exp_140 (= BIRD-ICL injection) on a single task.

Default task = task_303 (= local-specific failure in llamacpp_single_001;
exp_137 passes 8/9; retrieval surfaces a perfect percentage SQL example).
"""
from __future__ import annotations

import csv as _csv
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
    from experiments.exp_140_agentar_icl.bird_icl import retrieve_and_format

    tid = sys.argv[1] if len(sys.argv) > 1 else "task_303"

    ds = DABenchPublicDataset(root_dir=ROOT / "data" / "public" / "input")
    task = ds.get_task(tid)
    print(f"[{time.strftime('%H:%M:%S')}] Q: {task.question}")

    block, examples = retrieve_and_format(task.question)
    print(f"[icl] retrieved {len(examples)} examples")
    for e in examples:
        print(f"  sim={e.sim:.3f} db={e.db_id} Q={e.question[:70]}")

    model = OpenAIModelAdapter(
        model="qwen3.5-35b-a3b",
        api_base="http://localhost:8000/v1",
        api_key="local-key",
        temperature=0.6,
    )
    preamble = build_preamble(task)
    has_icl = "ICL EXAMPLES" in preamble.text
    print(f"[preamble] built {len(preamble.text)} chars, has_icl={has_icl}")

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
    trace_path = ROOT / "artifacts" / "exp140_smoke" / tid / "trace.log"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text("")
    agent.trace_log_path = str(trace_path)

    print(f"[{time.strftime('%H:%M:%S')}] agent.run() starting...")
    t0 = time.time()
    r = agent.run(task)
    elapsed = time.time() - t0
    print(
        f"[{time.strftime('%H:%M:%S')}] done in {elapsed:.1f}s, "
        f"succeeded={r.succeeded}, n_steps={len(r.steps)}"
    )

    if r.answer and r.answer.rows:
        out_dir = trace_path.parent
        pred = out_dir / "prediction.csv"
        with pred.open("w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(r.answer.columns)
            for row in r.answer.rows:
                w.writerow(row)
        gold = ROOT / "data" / "public" / "output" / tid / "gold.csv"
        e = _evaluate_task(
            task_id=tid,
            prediction_path=pred,
            gold_path=gold,
            options=EvaluationOptions(),
        )
        print(f"  ANSWER cols={list(r.answer.columns)}, rows={[list(row) for row in r.answer.rows[:5]]}")
        print(f"  GOLD: {gold.read_text().strip()}")
        print(f"  OFFICIAL SCORE: {e.official_score_lambda_0_5}")
    else:
        print("  NO ANSWER")


if __name__ == "__main__":
    main()
