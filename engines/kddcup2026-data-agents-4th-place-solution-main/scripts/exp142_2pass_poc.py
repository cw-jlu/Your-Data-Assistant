"""POC: 2-pass agent (= open exploration before question reveal).

Pass 1: agent sees data but NOT the question; explores freely (= "survey
this database"). Produces a data summary from its trace.
Pass 2: standard exp_140 agent with the real question + pass-1 summary
injected into preamble as "## PRIOR DATA EXPLORATION".

Target: see if "always-fail" tasks where current PLAN locks a wrong
interpretation (task_163 "type of expenses" etc.) become solvable when the
agent has prior open exploration.

Local 4bit (= llama.cpp) so we don't compete with gpuhost bench.
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


SURVEY_QUESTION = (
    "Examine the data files available in this workspace. "
    "Use `describe_data` first to see the view catalog, then run 5-10 SELECT queries "
    "to understand the schema, value distributions, and key columns. "
    "Take notes about: column names + types, distinct value samples, "
    "linking columns between tables, any unusual columns. "
    "At the end, call `answer` with a single-column 'summary' table containing "
    "ONE row whose value is a concise paragraph summarizing your findings. "
    "Do NOT try to answer any specific question — just describe the data."
)


def _summarize_explore_trace(steps: list[dict]) -> str:
    """Compile pass-1 trace into a compact summary for pass-2 preamble injection."""
    lines = []
    for s in steps:
        ph = (s.get("phase") or "").lower()
        ac = (s.get("action") or "").lower()
        th = (s.get("thought") or "").strip().replace("\n", " ")[:400]
        if not th:
            continue
        if ac == "execute_sql":
            sql = (s.get("action_input") or {}).get("sql", "")[:200] if isinstance(s.get("action_input"), dict) else ""
            obs = (s.get("observation_content_preview") or "")[:300].replace("\n", " ")
            lines.append(f"- ran: {sql[:120]}")
            lines.append(f"  result: {obs}")
        elif ac == "complete_phase":
            lines.append(f"- thought: {th[:300]}")
    return "\n".join(lines[:60])  # cap


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    # Local 4bit endpoint
    os.environ["AGENT_API_BASE"] = "http://localhost:8000/v1"
    os.environ["AGENT_API_KEY"] = "local-key"
    os.environ["AGENT_MODEL"] = "qwen3.5-35b-a3b"
    os.environ["CF_ACCESS_CLIENT_ID"] = ""
    os.environ["CF_ACCESS_CLIENT_SECRET"] = ""

    from kobushi_core.eval.csv_compare import EvaluationOptions, _evaluate_task
    from kobushi_core.benchmark import DABenchPublicDataset
    from kobushi_core.benchmark.schema import PublicTask, TaskRecord
    from kobushi_core.model import OpenAIModelAdapter
    from experiments.exp_140_agentar_icl.preamble import build_preamble
    from experiments.exp_140_agentar_icl.phased_agent import (
        PhasedAgentConfig, PhasedReActAgent,
    )
    from experiments.exp_140_agentar_icl.tools.registry import (
        create_default_tool_registry,
    )

    # Test tasks: "always fail" series + 1 known-good control
    TIDS = ["task_163", "task_169", "task_199", "task_80", "task_22"]

    ds = DABenchPublicDataset(root_dir=ROOT / "data" / "public" / "input")
    OUT = ROOT / "artifacts" / "exp142_2pass_poc"
    OUT.mkdir(parents=True, exist_ok=True)

    results = []
    for tid in TIDS:
        task = ds.get_task(tid)
        print(f"\n{'='*70}\n[{time.strftime('%H:%M:%S')}] === {tid} ===", flush=True)
        print(f"REAL Q: {task.question}", flush=True)

        td = OUT / tid
        td.mkdir(parents=True, exist_ok=True)

        # ===== PASS 1: open exploration =====
        print(f"\n[{time.strftime('%H:%M:%S')}] PASS 1: open exploration (no question)", flush=True)
        model = OpenAIModelAdapter(
            model="qwen3.5-35b-a3b", api_base="http://localhost:8000/v1",
            api_key="local-key", temperature=0.6,
        )
        # Build preamble WITHOUT the real question — substitute survey prompt
        survey_task = PublicTask(
            record=TaskRecord(
                task_id=task.task_id,
                difficulty=task.difficulty,
                question=SURVEY_QUESTION,
            ),
            assets=task.assets,
        )
        p1_preamble = build_preamble(survey_task)
        tools = create_default_tool_registry(
            auditor_model=model, question_provider=lambda: SURVEY_QUESTION,
            context_dir=task.context_dir,
        )
        p1_agent = PhasedReActAgent(
            model=model, tools=tools,
            config=PhasedAgentConfig(max_steps=20, min_explore_queries=5),
            preamble=p1_preamble.text,
        )
        p1_trace = td / "pass1_trace.log"
        p1_trace.write_text("")
        p1_agent.trace_log_path = str(p1_trace)

        t0 = time.time()
        p1_result = p1_agent.run(survey_task)
        p1_elapsed = time.time() - t0
        # Compile summary from steps
        summary = _summarize_explore_trace(
            [{"phase": (s.observation or {}).get("phase", ""),
              "action": s.action,
              "action_input": s.action_input,
              "thought": s.thought,
              "observation_content_preview": str((s.observation or {}).get("content", ""))[:300]}
             for s in p1_result.steps]
        )
        # Also include the agent's final "answer" if it produced one
        if p1_result.answer and p1_result.answer.rows:
            summary_text = " ".join(str(c) for row in p1_result.answer.rows for c in row)
            summary = f"AGENT SUMMARY: {summary_text[:1500]}\n\n--- TRACE ---\n{summary}"
        (td / "pass1_summary.txt").write_text(summary)
        print(f"PASS 1 done in {p1_elapsed:.0f}s, summary={len(summary)} chars, n_steps={len(p1_result.steps)}", flush=True)

        # ===== PASS 2: standard agent + summary =====
        print(f"\n[{time.strftime('%H:%M:%S')}] PASS 2: real question + pass-1 summary injected", flush=True)
        p2_preamble = build_preamble(task)
        # Inject pass-1 summary at top of preamble
        p2_preamble_text = (
            "# PRIOR DATA EXPLORATION (= survey done before question revealed)\n"
            "This summary was produced by a separate agent that explored the data\n"
            "WITHOUT knowing the question. Treat it as factual data understanding.\n"
            "Use it to inform your column selection and filter choices.\n\n"
            + summary + "\n\n"
            + p2_preamble.text
        )
        tools2 = create_default_tool_registry(
            auditor_model=model, question_provider=lambda: task.question,
            context_dir=task.context_dir,
        )
        p2_agent = PhasedReActAgent(
            model=model, tools=tools2,
            config=PhasedAgentConfig(max_steps=64, min_explore_queries=3),
            preamble=p2_preamble_text,
        )
        p2_trace = td / "pass2_trace.log"
        p2_trace.write_text("")
        p2_agent.trace_log_path = str(p2_trace)

        t0 = time.time()
        p2_result = p2_agent.run(task)
        p2_elapsed = time.time() - t0

        # Score pass-2
        score = 0.0
        if p2_result.answer and p2_result.answer.rows:
            pred = td / "prediction.csv"
            with pred.open("w", newline="") as f:
                w = _csv.writer(f)
                w.writerow(p2_result.answer.columns)
                for row in p2_result.answer.rows:
                    w.writerow(row)
            gold = ROOT / "data" / "public" / "output" / tid / "gold.csv"
            e = _evaluate_task(task_id=tid, prediction_path=pred, gold_path=gold,
                               options=EvaluationOptions())
            score = float(e.official_score_lambda_0_5)
        print(f"PASS 2 done in {p2_elapsed:.0f}s, score={score:.2f}, n_steps={len(p2_result.steps)}", flush=True)
        results.append({
            "tid": tid, "score": score,
            "p1_elapsed_s": round(p1_elapsed, 1),
            "p2_elapsed_s": round(p2_elapsed, 1),
            "p1_n_steps": len(p1_result.steps),
            "p2_n_steps": len(p2_result.steps),
        })

    print()
    print(f"=== POC summary ({len(results)} tasks) ===")
    print(f"{'tid':<14} {'score':>6} {'p1_s':>6} {'p2_s':>6} {'p1_n':>5} {'p2_n':>5}")
    for r in results:
        print(f"{r['tid']:<14} {r['score']:>6.2f} {r['p1_elapsed_s']:>6.0f} {r['p2_elapsed_s']:>6.0f} {r['p1_n_steps']:>5} {r['p2_n_steps']:>5}")
    mean = sum(r['score'] for r in results) / len(results)
    print(f"\n  mean: {mean:.4f}")
    (OUT / "summary.json").write_text(json.dumps({"results": results, "mean": mean}, indent=2))


if __name__ == "__main__":
    main()
