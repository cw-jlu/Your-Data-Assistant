"""POC: schema_link sub-agent (= extract column/value candidates from question,
match against schema + cell values, inject as SCHEMA LINK HINTS).

Standalone POC — does NOT modify exp_140 preamble. Runs a fresh agent with
the schema_link block appended to its preamble.

Local 4bit so we don't compete with gpuhost bench.
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

    from kobushi_core.eval.csv_compare import EvaluationOptions, _evaluate_task
    from kobushi_core.benchmark import DABenchPublicDataset
    from kobushi_core.model import OpenAIModelAdapter
    from experiments.exp_140_agentar_icl.preamble import build_preamble
    from experiments.exp_140_agentar_icl.phased_agent import (
        PhasedAgentConfig, PhasedReActAgent,
    )
    from experiments.exp_140_agentar_icl.tools.registry import (
        create_default_tool_registry,
    )
    from experiments.exp_140_agentar_icl.schema_link import (
        extract_link_candidates, build_schema_link_block,
    )

    # Same task set as 2-pass POC for apples-to-apples
    TIDS = ["task_163", "task_169", "task_199", "task_80", "task_22"]

    ds = DABenchPublicDataset(root_dir=ROOT / "data" / "public" / "input")
    OUT = ROOT / "artifacts" / "exp143_schema_link_poc"
    OUT.mkdir(parents=True, exist_ok=True)

    results = []
    for tid in TIDS:
        task = ds.get_task(tid)
        print(f"\n{'='*70}\n[{time.strftime('%H:%M:%S')}] === {tid} ===", flush=True)
        print(f"Q: {task.question}", flush=True)

        td = OUT / tid
        td.mkdir(parents=True, exist_ok=True)

        model = OpenAIModelAdapter(
            model="qwen3.5-35b-a3b", api_base="http://localhost:8000/v1",
            api_key="local-key", temperature=0.6,
        )
        sub_model = OpenAIModelAdapter(
            model="qwen3.5-35b-a3b", api_base="http://localhost:8000/v1",
            api_key="local-key", temperature=0.0,
        )

        # === STEP 1: schema link extraction ===
        t0 = time.time()
        link = extract_link_candidates(task.question, sub_model)
        link_block = build_schema_link_block(task.question, task.context_dir, sub_model)
        link_elapsed = time.time() - t0
        print(f"\n[link] extracted in {link_elapsed:.0f}s:", flush=True)
        if link_block:
            print(link_block[:1200], flush=True)
        else:
            print("  (none)", flush=True)
        (td / "schema_link.txt").write_text(link_block or "")
        (td / "candidates.json").write_text(json.dumps({
            "column_candidates": [{"phrase": c.phrase, "candidates": list(c.candidates)}
                                  for c in link.column_candidates],
            "value_candidates": [{"phrase": c.phrase, "candidates": list(c.candidates)}
                                 for c in link.value_candidates],
        }, indent=2))

        # === STEP 2: agent run with link block injected ===
        p2_preamble = build_preamble(task)
        if link_block:
            p2_preamble_text = link_block + "\n\n" + p2_preamble.text
        else:
            p2_preamble_text = p2_preamble.text
        tools = create_default_tool_registry(
            auditor_model=model, question_provider=lambda: task.question,
            context_dir=task.context_dir,
        )
        agent = PhasedReActAgent(
            model=model, tools=tools,
            config=PhasedAgentConfig(max_steps=64, min_explore_queries=3),
            preamble=p2_preamble_text,
        )
        trace_path = td / "trace.log"
        trace_path.write_text("")
        agent.trace_log_path = str(trace_path)

        t0 = time.time()
        r = agent.run(task)
        elapsed = time.time() - t0
        score = 0.0
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
        print(f"\n[agent] done in {elapsed:.0f}s, score={score:.2f}, n_steps={len(r.steps)}", flush=True)
        results.append({
            "tid": tid, "score": score, "link_elapsed_s": round(link_elapsed, 1),
            "agent_elapsed_s": round(elapsed, 1), "n_steps": len(r.steps),
        })

    print()
    print(f"=== POC summary ({len(results)} tasks) ===")
    print(f"{'tid':<14} {'score':>6} {'link_s':>7} {'agent_s':>8} {'n':>4}")
    for r in results:
        print(f"{r['tid']:<14} {r['score']:>6.2f} {r['link_elapsed_s']:>7.0f} {r['agent_elapsed_s']:>8.0f} {r['n_steps']:>4}")
    mean = sum(r['score'] for r in results) / len(results)
    print(f"\n  mean: {mean:.4f}")
    (OUT / "summary.json").write_text(json.dumps({"results": results, "mean": mean}, indent=2))


if __name__ == "__main__":
    main()
