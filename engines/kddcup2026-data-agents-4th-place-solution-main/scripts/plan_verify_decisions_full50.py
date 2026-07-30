"""Run PLAN verifier on all 50 public tasks using REAL phased agent PLAN.

Unlike plan_verify_full50.py (= 1-shot LLM), this runs the actual phased agent
up to the plan→explore transition and uses the agent's true PLAN thought.

Pipeline:
  1. For each task, run PhasedReActAgent until first complete_phase
  2. Extract the PLAN-phase thought text
  3. Run plan_verifier on (question, plan_text, schema)
  4. Compare flag against v5 ground truth scores

Usage:
    uv run python scripts/plan_verify_phased_full50.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from dotenv import load_dotenv
load_dotenv(REPO / ".env")

from kobushi_core.benchmark.dataset import DABenchPublicDataset
from kobushi_core.model import OpenAIModelAdapter
from experiments.exp_133_plan_decisions.phased_agent import (
    PhasedReActAgent, PhasedAgentConfig,
)
from experiments.exp_122_column_auditor.preamble import build_preamble
from experiments.exp_122_column_auditor.tools.registry import create_default_tool_registry
from experiments.exp_122_column_auditor.tools.duckdb_unified import execute_sql
from experiments.exp_122_column_auditor.plan_verifier import verify_plan


OUT_DIR = REPO / "artifacts" / "plan_verify_decisions_full50"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GT_PATH = REPO / "artifacts" / "bench_phased_vote3_full50_exp126_exp123" / "summary.json"


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


def load_schema(task) -> dict[str, list[str]]:
    try:
        res = execute_sql(task.context_dir, "SHOW TABLES")
        tables = [r[0] for r in res["rows"]]
    except Exception:
        return {}
    schema = {}
    for tbl in tables:
        try:
            cols = execute_sql(task.context_dir, f"DESCRIBE {tbl}")
            schema[tbl] = [r[0] for r in cols["rows"]]
        except Exception:
            schema[tbl] = []
    return schema


def run_phased_until_explore(task, model: OpenAIModelAdapter) -> str:
    """Run phased agent, return the PLAN-phase thought text (= up to first complete_phase)."""
    preamble = build_preamble(task)
    tools = create_default_tool_registry(
        auditor_model=model,
        question_provider=lambda: task.question,
        context_dir=task.context_dir,
    )
    agent = PhasedReActAgent(
        model=model,
        tools=tools,
        config=PhasedAgentConfig(max_steps=8, min_explore_queries=0),
        preamble=preamble.text,
    )
    result = agent.run(task)
    # Collect thought text from PLAN-phase steps (= steps before first plan→explore)
    plan_thoughts = []
    for s in result.steps:
        obs = getattr(s, "observation", {}) or {}
        if obs.get("phase") == "plan":
            plan_thoughts.append(s.thought or "")
        # Stop after the transition step itself
        if s.action == "complete_phase" and obs.get("transition", {}).get("from") == "plan":
            break
    return "\n\n".join(plan_thoughts)


def run_one(task_id: str) -> dict:
    model = make_model()
    ds = DABenchPublicDataset(root_dir=REPO / "data" / "public" / "input")
    task = ds.get_task(task_id)
    t0 = time.time()
    try:
        schema = load_schema(task)
        plan_text = run_phased_until_explore(task, model)
        result = verify_plan(task.question, plan_text, schema, model)
        err = None
    except Exception as exc:
        plan_text = ""
        result = None
        err = str(exc)
    elapsed = time.time() - t0
    out_path = OUT_DIR / f"{task_id}.md"
    if result is not None:
        flags_lines = [f"- '{fl.noun}': {fl.unused_matches}" for fl in result.flags]
    else:
        flags_lines = [f"<error: {err}>"]
    out_path.write_text(
        f"# {task_id}\nQ: {task.question}\nElapsed: {elapsed:.1f}s\n\n"
        f"## Nouns\n{result.nouns if result else '?'}\n\n"
        f"## Flags ({len(result.flags) if result else 0})\n" + "\n".join(flags_lines) +
        f"\n\n## PLAN\n```\n{plan_text}\n```\n"
    )
    return {
        "task_id": task_id,
        "elapsed_s": elapsed,
        "nouns": result.nouns if result else [],
        "flags": [{"noun": f.noun, "unused": f.unused_matches} for f in (result.flags if result else [])],
        "any_flag": result.has_issue if result else False,
        "plan_chars": len(plan_text),
        "error": err,
    }


def main():
    gt = json.load(open(GT_PATH))["scores"]
    task_ids = sorted(gt.keys(), key=lambda x: int(x.split("_")[1]))
    print(f"=== PHASED PLAN verifier on {len(task_ids)} tasks ===", flush=True)
    n_fail = sum(1 for v in gt.values() if v < 1)
    print(f"GT: n_fail={n_fail}/{len(gt)} (mean={sum(gt.values())/len(gt):.3f})", flush=True)

    results = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        futures = {ex.submit(run_one, tid): tid for tid in task_ids}
        done = 0
        for fut in as_completed(futures):
            tid = futures[fut]
            done += 1
            r = fut.result()
            r["gt_score"] = gt.get(tid, None)
            results.append(r)
            gtv = r["gt_score"]
            cat = "FAIL" if gtv < 1.0 else "ok  "
            flag_str = ",".join(f["noun"][:15] for f in r["flags"]) if r["flags"] else "-"
            err = f" ERR={r['error']}" if r.get("error") else ""
            print(
                f"  [{done:2d}/{len(task_ids)}] [{cat}] {tid} gt={gtv:.2f} "
                f"flag={'X' if r['any_flag'] else '-'} {flag_str}{err}",
                flush=True,
            )

    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2, default=str))

    tp = sum(1 for r in results if r["gt_score"] < 1.0 and r["any_flag"])
    fn = sum(1 for r in results if r["gt_score"] < 1.0 and not r["any_flag"])
    fp = sum(1 for r in results if r["gt_score"] >= 1.0 and r["any_flag"])
    tn = sum(1 for r in results if r["gt_score"] >= 1.0 and not r["any_flag"])
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    lines = [
        "# PHASED PLAN verifier full-50 — summary\n",
        f"|       | flag=X | flag=- |",
        f"|-------|--------|--------|",
        f"| FAIL  | TP={tp}  | FN={fn}  |",
        f"| OK    | FP={fp}  | TN={tn}  |",
        f"",
        f"- Precision: {precision:.3f}",
        f"- Recall:    {recall:.3f}",
        f"- Specificity: {specificity:.3f}",
        f"",
        "| task | gt | flag | nouns_n | flagged |",
        "|---|---|---|---|---|",
    ]
    for r in sorted(results, key=lambda x: int(x["task_id"].split("_")[1])):
        lines.append(
            f"| {r['task_id']} | {r['gt_score']:.1f} | {'X' if r['any_flag'] else '-'} | "
            f"{len(r['nouns'])} | {[f['noun'] for f in r['flags']]} |"
        )
    (OUT_DIR / "summary.md").write_text("\n".join(lines))
    print(f"\nP={precision:.3f}, R={recall:.3f}, Spec={specificity:.3f}", flush=True)
    print(f"Summary: {OUT_DIR / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
