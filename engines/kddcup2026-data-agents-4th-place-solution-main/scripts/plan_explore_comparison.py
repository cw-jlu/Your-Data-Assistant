"""PLAN-only comparison: exp_122 (= list_context-only) vs exp_131 (= explore tools).

For each task in TARGET_TASKS, run PhasedReActAgent twice:
  A. exp_122 PLAN phase only: tools = list_context + complete_phase
  B. exp_131 PLAN phase only: tools = + describe_data, execute_sql, read_doc, grep

Both stop when the agent calls complete_phase (= phase transition).
We capture ALL steps for inspection.

Output: artifacts/plan_explore_compare/<task_id>/{exp_122,exp_131}_<n>.md
        artifacts/plan_explore_compare/summary.md

Usage:
    uv run python scripts/plan_explore_comparison.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from dotenv import load_dotenv
load_dotenv(REPO / ".env")

from kobushi_core.benchmark.dataset import DABenchPublicDataset
from kobushi_core.model import OpenAIModelAdapter


OUT_DIR = REPO / "artifacts" / "plan_explore_compare"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_TASKS = ["task_25", "task_163", "task_169", "task_180", "task_199"]
N_SAMPLES = 3


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


def run_plan_only(exp_module: str, task_id: str, model) -> dict:
    """Run agent and stop at first complete_phase transition (= end of PLAN)."""
    # Dynamic import per exp variant
    if exp_module == "exp_122_column_auditor":
        from experiments.exp_122_column_auditor.phased_agent import (
            PhasedReActAgent, PhasedAgentConfig,
        )
        from experiments.exp_122_column_auditor.preamble import build_preamble
        from experiments.exp_122_column_auditor.tools.registry import create_default_tool_registry
    elif exp_module == "exp_131_plan_explore":
        from experiments.exp_131_plan_explore.phased_agent import (
            PhasedReActAgent, PhasedAgentConfig,
        )
        from experiments.exp_131_plan_explore.preamble import build_preamble
        from experiments.exp_131_plan_explore.tools.registry import create_default_tool_registry
    else:
        raise ValueError(f"unknown exp_module: {exp_module}")

    ds = DABenchPublicDataset(root_dir=REPO / "data" / "public" / "input")
    task = ds.get_task(task_id)
    preamble = build_preamble(task)
    tools = create_default_tool_registry(
        auditor_model=model,
        question_provider=lambda: task.question,
        context_dir=task.context_dir,
    )

    # Cap max_steps to ~12 — plan phase shouldn't need many. We also break on
    # the transition to "explore" externally.
    agent = PhasedReActAgent(
        model=model,
        tools=tools,
        config=PhasedAgentConfig(max_steps=12, min_explore_queries=0),
        preamble=preamble.text,
    )

    # We're going to run, then post-hoc trim steps to "up to first
    # complete_phase". The agent will naturally try to transition to explore.
    # We let it run a bit beyond if it lingers, since min_explore_queries=0 it
    # should commit ASAP.
    t0 = time.time()
    result = agent.run(task)
    latency = time.time() - t0

    # Find the step that transitioned plan→explore (= first complete_phase)
    plan_steps = []
    for s in result.steps:
        plan_steps.append(s)
        if s.action == "complete_phase":
            break

    return {
        "task_id": task_id,
        "question": task.question,
        "latency_s": int(latency),
        "n_plan_steps": len(plan_steps),
        "steps": [
            {
                "i": i,
                "action": s.action,
                "action_input": s.action_input,
                "thought": s.thought[:1500] if hasattr(s, "thought") else "",
                "observation": str(s.observation)[:800] if hasattr(s, "observation") else "",
            }
            for i, s in enumerate(plan_steps)
        ],
    }


def _save_one(tid: str, cfg: str, i: int, r: dict) -> None:
    task_dir = OUT_DIR / tid
    task_dir.mkdir(parents=True, exist_ok=True)
    path = task_dir / f"{cfg}_{i}.md"
    lines = [
        f"# {tid} | {cfg} | sample {i}",
        f"",
        f"## Q: {r['question']}",
        f"## Latency: {r['latency_s']}s, plan_steps: {r['n_plan_steps']}",
        f"",
    ]
    for s in r["steps"]:
        lines.append(f"### Step {s['i']}: action={s['action']}")
        if s["action_input"]:
            lines.append(f"**input**: `{str(s['action_input'])[:300]}`")
        if s.get("thought"):
            lines.append(f"**thought**:\n```\n{s['thought']}\n```")
        if s.get("observation"):
            lines.append(f"**observation**:\n```\n{s['observation']}\n```")
        lines.append("")
    path.write_text("\n".join(lines))


def _one_job(args):
    tid, cfg, i, exp_module = args
    model = make_model()
    try:
        r = run_plan_only(exp_module, tid, model)
        _save_one(tid, cfg, i, r)
        return (tid, cfg, i, r["n_plan_steps"], r["latency_s"], None)
    except Exception as exc:
        return (tid, cfg, i, 0, 0, str(exc))


def main():
    print(f"=== PARALLEL PLAN-only comparison: {len(TARGET_TASKS)} tasks × {N_SAMPLES} samples × 2 configs ===", flush=True)

    # Build all job tuples
    jobs = []
    for tid in TARGET_TASKS:
        for cfg, exp_module in [
            ("exp_122", "exp_122_column_auditor"),
            ("exp_131", "exp_131_plan_explore"),
        ]:
            for i in range(N_SAMPLES):
                jobs.append((tid, cfg, i, exp_module))

    print(f"Total jobs: {len(jobs)}", flush=True)

    from concurrent.futures import ThreadPoolExecutor, as_completed
    summary = {"exp_122": {}, "exp_131": {}}
    with ThreadPoolExecutor(max_workers=6) as ex:
        futures = [ex.submit(_one_job, j) for j in jobs]
        done = 0
        for fut in as_completed(futures):
            tid, cfg, i, n_steps, lat, err = fut.result()
            done += 1
            if err:
                print(f"  [{done}/{len(jobs)}] {tid} {cfg}#{i} ERROR: {err}", flush=True)
            else:
                print(f"  [{done}/{len(jobs)}] {tid} {cfg}#{i} steps={n_steps} lat={lat}s", flush=True)
                summary[cfg].setdefault(tid, []).append({"i": i, "n_steps": n_steps, "latency_s": lat})

    # Summary table
    lines = ["# PLAN-only comparison — summary\n"]
    lines.append("| task | cfg | n_samples | avg_steps | avg_latency_s |")
    lines.append("|---|---|---|---|---|")
    for tid in TARGET_TASKS:
        for cfg in ["exp_122", "exp_131"]:
            samples = summary[cfg].get(tid, [])
            if not samples: continue
            avg_steps = sum(s["n_steps"] for s in samples) / len(samples)
            avg_lat = sum(s["latency_s"] for s in samples) / len(samples)
            lines.append(f"| {tid} | {cfg} | {len(samples)} | {avg_steps:.1f} | {avg_lat:.0f} |")
    (OUT_DIR / "summary.md").write_text("\n".join(lines))
    print(f"\nSummary: {OUT_DIR / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
