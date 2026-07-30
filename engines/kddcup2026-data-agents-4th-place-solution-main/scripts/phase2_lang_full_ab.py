"""Full-harness prompt-language A/B: en vs zh on Chinese-language Phase 2 tasks.

Same phased harness, math_advisor, column_auditor, adaptive_vote, DuckDB
unified, video attach — the ONLY difference is the system-prompt language:
  --variant en  → experiments.exp_142_phase2_baseline (English phased prompt)
  --variant zh  → experiments.exp_142_phase2_zh       (Chinese phased prompt)

Runs the SAME Chinese-language task subset for both variants (att=1), so the
comparison isolates the system-prompt language.
"""
from __future__ import annotations

import argparse, csv, importlib, json, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from dotenv import load_dotenv
load_dotenv(REPO / ".env")

from kobushi_core.benchmark.dataset import DABenchPublicDataset
from kobushi_core.model import OpenAIModelAdapter
from kobushi_core.eval.csv_compare import _evaluate_task, EvaluationOptions

DATA = REPO / "data" / "phase2_demo" / "demo_samples_phase2"
INPUT, GOLD = DATA / "input", DATA / "output"

VARIANT_MODULE = {
    "en": "experiments.exp_142_phase2_baseline",
    "zh": "experiments.exp_142_phase2_zh",
}


def make_model(temp=0.6):
    return OpenAIModelAdapter(
        model="qwen3.5-35b-a3b",
        api_base=os.environ.get("AGENT_API_BASE", ""),
        api_key=os.environ.get("AGENT_API_KEY", ""),
        temperature=temp,
        extra_headers={"CF-Access-Client-Id": os.environ.get("CF_ACCESS_CLIENT_ID", ""),
                       "CF-Access-Client-Secret": os.environ.get("CF_ACCESS_CLIENT_SECRET", "")},
    )


def official_score(pred, gold):
    if not gold.is_file():
        return 0.0
    try:
        ev = _evaluate_task(task_id=pred.parent.name, prediction_path=pred,
                            gold_path=gold, options=EvaluationOptions())
        return float(ev.official_score_lambda_0_5)
    except Exception:
        return 0.0


def is_chinese(q):
    return bool(re.search(r"[一-鿿]", q))


def run_one(tid, mod, out_dir, max_steps=64):
    t0 = time.time()
    PhasedReActAgent = mod["PhasedReActAgent"]
    PhasedAgentConfig = mod["PhasedAgentConfig"]
    build_preamble = mod["build_preamble"]
    create_registry = mod["create_default_tool_registry"]
    generate_formula = mod["generate_formula"]
    adaptive_vote = mod["adaptive_vote"]

    ds = DABenchPublicDataset(root_dir=INPUT)
    task = ds.get_task(tid)
    td = out_dir / tid
    td.mkdir(parents=True, exist_ok=True)
    try:
        formula = generate_formula(task.question)
    except Exception as e:
        formula = f"NO_CALC ({e})"
    skip = formula.strip().upper().startswith("NO_CALC")
    (td / "formula.txt").write_text(formula)

    answers = []
    try:
        m = make_model(0.6)
        pre = build_preamble(task)
        injected = pre.text if skip else (
            "# MATH FORMULA HINT\n" + f"  {formula}\n\n"
            "Follow this aggregation/division/filter structure EXACTLY.\n\n"
        ) + pre.text
        tools = create_registry(auditor_model=m, question_provider=lambda: task.question,
                                context_dir=task.context_dir)
        agent = PhasedReActAgent(model=m, tools=tools,
                                 config=PhasedAgentConfig(max_steps=max_steps, min_explore_queries=3),
                                 preamble=injected)
        r = agent.run(task)
        if r.answer and r.answer.rows:
            answers.append(r.answer)
        (td / "trace.json").write_text(json.dumps([{
            "answer": {"columns": list(r.answer.columns), "rows": [list(x) for x in r.answer.rows]} if r.answer else None,
            "failure_reason": getattr(r, "failure_reason", None),
            "steps": [{"i": j, "action": s.action, "action_input": s.action_input,
                       "thought": (s.thought or "")[:1500]} for j, s in enumerate(r.steps)],
        }], ensure_ascii=False, indent=2, default=str))
    except Exception as e:
        (td / "error.txt").write_text(str(e))
        return {"tid": tid, "score": 0.0, "error": str(e)[:120], "elapsed": time.time() - t0}

    if not answers:
        return {"tid": tid, "score": 0.0, "n_ok": 0, "elapsed": time.time() - t0}
    voted = adaptive_vote(answers)
    if not voted or not voted.rows:
        return {"tid": tid, "score": 0.0, "n_ok": len(answers), "elapsed": time.time() - t0}
    pred = td / "prediction.csv"
    with open(pred, "w", newline="") as f:
        w = csv.writer(f); w.writerow(voted.columns); [w.writerow(x) for x in voted.rows]
    s = official_score(pred, GOLD / tid / "gold.csv")
    return {"tid": tid, "score": s, "n_ok": len(answers), "elapsed": time.time() - t0}


def load_module(variant):
    base = VARIANT_MODULE[variant]
    return {
        "PhasedReActAgent": importlib.import_module(f"{base}.phased_agent").PhasedReActAgent,
        "PhasedAgentConfig": importlib.import_module(f"{base}.phased_agent").PhasedAgentConfig,
        "build_preamble": importlib.import_module(f"{base}.preamble").build_preamble,
        "create_default_tool_registry": importlib.import_module(f"{base}.tools.registry").create_default_tool_registry,
        "generate_formula": importlib.import_module(f"{base}.math_advisor").generate_formula,
        "adaptive_vote": importlib.import_module(f"{base}.adaptive_vote").adaptive_vote,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["en", "zh"], required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-steps", type=int, default=64)
    ap.add_argument("--n", type=int, default=16, help="number of Chinese tasks")
    a = ap.parse_args()

    ds = DABenchPublicDataset(root_dir=INPUT)
    zh_tasks = sorted([t.task_id for t in ds.iter_tasks() if is_chinese(t.question)],
                      key=lambda s: int(s.split("_")[1]))[:a.n]
    mod = load_module(a.variant)

    runs = REPO / "artifacts" / "runs"
    i = 1
    while (runs / f"exp_142_lang_{a.variant}_{i:03d}").exists():
        i += 1
    out_dir = runs / f"exp_142_lang_{a.variant}_{i:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== lang A/B variant={a.variant} ({VARIANT_MODULE[a.variant]}) ===")
    print(f"  {len(zh_tasks)} Chinese tasks, workers={a.workers}, att=1, out={out_dir.name}", flush=True)

    results = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(run_one, t, mod, out_dir, a.max_steps): t for t in zh_tasks}
        for k, fut in enumerate(as_completed(futs), 1):
            r = fut.result(); results.append(r)
            mark = "✓" if r.get("score", 0) >= 0.999 else "✗"
            run_mean = sum(x["score"] for x in results) / len(results)
            print(f"[{k:2d}/{len(zh_tasks)}] {mark} {r['tid']:<9} score={r.get('score',0):.2f} "
                  f"t={r.get('elapsed',0):.0f}s mean={run_mean:.4f} {(r.get('error') or '')[:40]}", flush=True)

    mean = sum(r["score"] for r in results) / len(results)
    (out_dir / "summary.json").write_text(json.dumps({
        "variant": a.variant, "module": VARIANT_MODULE[a.variant],
        "n_tasks": len(results), "mean_score": mean,
        "n_perfect": sum(1 for r in results if r["score"] >= 0.999),
        "elapsed_min": (time.time() - t0) / 60, "results": results,
    }, ensure_ascii=False, indent=2))
    print(f"\n=== DONE variant={a.variant} mean={mean:.4f} "
          f"perfect={sum(1 for r in results if r['score']>=0.999)}/{len(results)} ===")


if __name__ == "__main__":
    main()
