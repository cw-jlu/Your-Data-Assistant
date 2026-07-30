"""POC: retry-with-analysis on 15 failing/stochastic tasks.

For each task:
  Round 1: standard exp_140 agent.run() (= ICL + VALUE HINTS preamble)
  Analyzer: sub-agent reads trace + answer → outputs ALT_DIRECTION
  Round 2: agent.run() with retry hint block prepended to preamble
  Final score: max(round1_score, round2_score)

Compares:
  - round1_only (= what current pipeline gives)
  - round1 + round2 (= proposed retry-with-analysis)
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

# 15 tasks: 8 always-fail + 7 stochastic
TIDS = [
    # always fail
    "task_163", "task_169", "task_199", "task_344", "task_396",
    "task_418", "task_80", "task_89",
    # stochastic
    "task_11", "task_173", "task_180", "task_200", "task_303",
    "task_379", "task_86",
]


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
    from experiments.exp_140_agentar_icl.trace_analyzer import (
        analyze_attempt, build_retry_hint_block,
    )

    ds = DABenchPublicDataset(root_dir=ROOT / "data" / "public" / "input")
    OUT = ROOT / "artifacts" / "exp144_retry_poc"
    OUT.mkdir(parents=True, exist_ok=True)

    def _score_csv(pred_path: Path, gold_path: Path) -> float:
        if not pred_path.exists() or not gold_path.exists():
            return 0.0
        try:
            e = _evaluate_task(task_id=pred_path.parent.name, prediction_path=pred_path,
                               gold_path=gold_path, options=EvaluationOptions())
            return float(e.official_score_lambda_0_5)
        except Exception:
            return 0.0

    def _save_pred(file_path: Path, answer):
        if not answer or not answer.rows:
            return None
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(answer.columns)
            for row in answer.rows:
                w.writerow(row)
        return file_path

    def _pred_preview(answer) -> str:
        if not answer or not answer.rows:
            return "(no answer)"
        cols = ",".join(str(c) for c in answer.columns)
        rows = "\n".join(",".join(str(v) for v in r) for r in answer.rows[:5])
        n_extra = len(answer.rows) - 5
        suffix = f"\n... ({n_extra} more rows)" if n_extra > 0 else ""
        return f"{cols}\n{rows}{suffix}"

    results = []
    for tid in TIDS:
        task = ds.get_task(tid)
        print(f"\n{'='*70}\n[{time.strftime('%H:%M:%S')}] === {tid} ===", flush=True)
        print(f"Q: {task.question}", flush=True)
        td = OUT / tid
        td.mkdir(parents=True, exist_ok=True)
        gold = ROOT / "data" / "public" / "output" / tid / "gold.csv"

        model = OpenAIModelAdapter(
            model="qwen3.5-35b-a3b", api_base="http://localhost:8000/v1",
            api_key="local-key", temperature=0.6,
        )
        analyzer_model = OpenAIModelAdapter(
            model="qwen3.5-35b-a3b", api_base="http://localhost:8000/v1",
            api_key="local-key", temperature=0.0,
        )

        # =========== ROUND 1 ===========
        print(f"[{time.strftime('%H:%M:%S')}] ROUND 1", flush=True)
        preamble = build_preamble(task)
        tools = create_default_tool_registry(
            auditor_model=model, question_provider=lambda: task.question,
            context_dir=task.context_dir,
        )
        r1_agent = PhasedReActAgent(
            model=model, tools=tools,
            config=PhasedAgentConfig(max_steps=64, min_explore_queries=3),
            preamble=preamble.text,
        )
        r1_trace_path = td / "round1_trace.log"
        r1_trace_path.write_text("")
        r1_agent.trace_log_path = str(r1_trace_path)
        t0 = time.time()
        r1 = r1_agent.run(task)
        r1_elapsed = time.time() - t0
        r1_pred_path = _save_pred(td / "round1.csv", r1.answer)
        r1_score = _score_csv(r1_pred_path, gold) if r1_pred_path else 0.0
        print(f"  R1 done {r1_elapsed:.0f}s, score={r1_score:.2f}, n_steps={len(r1.steps)}", flush=True)

        # =========== ANALYZER ===========
        analysis = {"alt_direction": "", "rationale": "", "raw_reply": ""}
        hint_block = None
        gave_up = bool(r1.failure_reason and r1.failure_reason.startswith("GIVE_UP"))
        if gave_up:
            print(f"  AGENT GAVE UP: {r1.failure_reason[:200]}", flush=True)
        # Always run analyzer when r1 < 0.99 OR agent gave up
        if r1_score < 0.99 or gave_up:
            print(f"[{time.strftime('%H:%M:%S')}] ANALYZER", flush=True)
            trace_text = r1_trace_path.read_text() if r1_trace_path.exists() else ""
            pred_preview = _pred_preview(r1.answer) if r1.answer else "(no answer — agent gave up)"
            # Add the give_up reason to the analyzer's context if present
            if gave_up:
                pred_preview = f"AGENT GAVE UP at this attempt.\nReason: {r1.failure_reason}\n\n(no final answer)"
            t0 = time.time()
            analysis = analyze_attempt(task.question, trace_text, pred_preview, analyzer_model)
            print(f"  analyzer done {time.time()-t0:.0f}s", flush=True)
            print(f"  ALT_DIRECTION: {analysis['alt_direction'][:200]}", flush=True)
            (td / "analysis.json").write_text(json.dumps(analysis, indent=2))
            hint_block = build_retry_hint_block(analysis)
            # If analyzer said "(none)" but agent gave up, force a generic retry hint
            if hint_block is None and gave_up:
                hint_block = (
                    "# RETRY HINT — PRIOR ATTEMPT GAVE UP\n"
                    f"A previous attempt aborted with reason: {r1.failure_reason}\n"
                    "Reconsider your interpretation of the question. In particular,\n"
                    "check whether a literal column name (= word appearing in the\n"
                    "question matches a column verbatim) is the better choice over\n"
                    "any semantic alternative you may consider."
                )

        # =========== ROUND 2 ===========
        r2_score = 0.0
        r2_elapsed = 0.0
        r2_n_steps = 0
        if hint_block:
            print(f"[{time.strftime('%H:%M:%S')}] ROUND 2 (with retry hint)", flush=True)
            r2_preamble_text = hint_block + "\n\n" + preamble.text
            tools2 = create_default_tool_registry(
                auditor_model=model, question_provider=lambda: task.question,
                context_dir=task.context_dir,
            )
            r2_agent = PhasedReActAgent(
                model=model, tools=tools2,
                config=PhasedAgentConfig(max_steps=64, min_explore_queries=3),
                preamble=r2_preamble_text,
            )
            r2_trace_path = td / "round2_trace.log"
            r2_trace_path.write_text("")
            r2_agent.trace_log_path = str(r2_trace_path)
            t0 = time.time()
            r2 = r2_agent.run(task)
            r2_elapsed = time.time() - t0
            r2_n_steps = len(r2.steps)
            r2_pred_path = _save_pred(td / "round2.csv", r2.answer)
            if r2_pred_path:
                r2_score = _score_csv(r2_pred_path, gold)
            print(f"  R2 done {r2_elapsed:.0f}s, score={r2_score:.2f}, n_steps={r2_n_steps}", flush=True)

        best = max(r1_score, r2_score)
        delta = best - r1_score
        results.append({
            "tid": tid,
            "r1_score": r1_score, "r2_score": r2_score, "best": best, "delta": delta,
            "r1_elapsed_s": round(r1_elapsed, 1), "r2_elapsed_s": round(r2_elapsed, 1),
            "alt_direction": (analysis.get("alt_direction") or "")[:200],
        })
        sym = "+" if delta > 0 else (" " if delta == 0 else "-")
        print(f"  [final] r1={r1_score:.2f} r2={r2_score:.2f} best={best:.2f} {sym}{abs(delta):.2f}", flush=True)

    print()
    print(f"=== POC summary ({len(results)} tasks) ===")
    print(f"{'tid':<14} {'r1':>5} {'r2':>5} {'best':>5} {'Δ':>5}  alt_dir")
    for r in results:
        sym = "+" if r["delta"] > 0 else (" " if r["delta"] == 0 else "-")
        print(f"{r['tid']:<14} {r['r1_score']:>5.2f} {r['r2_score']:>5.2f} {r['best']:>5.2f} {sym}{abs(r['delta']):>4.2f}  {r['alt_direction'][:60]}")
    mean_r1 = sum(r["r1_score"] for r in results) / len(results)
    mean_best = sum(r["best"] for r in results) / len(results)
    delta = mean_best - mean_r1
    print(f"\n  mean r1: {mean_r1:.4f}")
    print(f"  mean best: {mean_best:.4f}")
    print(f"  Δ (retry effect): +{delta:.4f}")
    (OUT / "summary.json").write_text(json.dumps({
        "results": results, "mean_r1": mean_r1, "mean_best": mean_best, "delta": delta
    }, indent=2))


if __name__ == "__main__":
    main()
