"""Run PLAN verifier on all 50 public tasks, measure precision/recall vs v5 ground truth.

Pipeline:
  1. For each task, generate PLAN text via 1-shot LLM call (= no tools, no thinking)
  2. Run plan_verifier on (question, plan_text, schema)
  3. Compare flag against ground truth (= v5 bench scores)
  4. Output: confusion matrix, per-task table

Usage:
    uv run python scripts/plan_verify_full50.py
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
from kobushi_core.model import OpenAIModelAdapter, ModelMessage
from experiments.exp_122_column_auditor.preamble import build_preamble
from experiments.exp_122_column_auditor.plan_verifier import verify_plan
from experiments.exp_122_column_auditor.tools.duckdb_unified import execute_sql


OUT_DIR = REPO / "artifacts" / "plan_verify_full50"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GT_PATH = REPO / "artifacts" / "bench_phased_vote3_full50_exp126_exp123" / "summary.json"

_ONESHOT_PLAN_SYS = """You are running the PLAN phase of a SQL agent. Given a question and schema preview, output a PLAN that:
  - states column_count and per_column for the answer
  - INTERPRET each noun: list candidate columns, then CHOOSING <table>.<column> with reason
  - INTERPRET semantics (filters, joins) similarly
Be terse and explicit. Use the format from earlier examples."""


def make_model() -> OpenAIModelAdapter:
    return OpenAIModelAdapter(
        model="qwen3.5-35b-a3b",
        api_base=os.environ["AGENT_API_BASE"],
        api_key=os.environ["AGENT_API_KEY"],
        temperature=0.3,
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


def generate_plan_text(task, model: OpenAIModelAdapter) -> str:
    preamble = build_preamble(task)
    user = f"{preamble.text}\n\n# Question\n{task.question}\n\nWrite the PLAN now."
    try:
        return model.complete(
            [
                ModelMessage(role="system", content=_ONESHOT_PLAN_SYS),
                ModelMessage(role="user", content=user),
            ],
            enable_thinking=False,
            max_tokens=2048,
        )
    except Exception as exc:
        return f"<plan generation failed: {exc}>"


def run_one(task_id: str) -> dict:
    model = make_model()
    ds = DABenchPublicDataset(root_dir=REPO / "data" / "public" / "input")
    task = ds.get_task(task_id)
    t0 = time.time()
    schema = load_schema(task)
    plan_text = generate_plan_text(task, model)
    result = verify_plan(task.question, plan_text, schema, model)
    elapsed = time.time() - t0
    # Save per-task artifact
    out_path = OUT_DIR / f"{task_id}.md"
    lines = [
        f"# {task_id}",
        f"Q: {task.question}",
        f"Elapsed: {elapsed:.1f}s",
        f"",
        f"## Nouns extracted",
        f"{result.nouns}",
        f"",
        f"## Flags ({len(result.flags)})",
    ]
    for fl in result.flags:
        lines.append(f"- '{fl.noun}': unused exact matches {fl.unused_matches}")
    lines.extend(["", "## PLAN", "```", plan_text, "```"])
    out_path.write_text("\n".join(lines))
    return {
        "task_id": task_id,
        "elapsed_s": elapsed,
        "nouns": result.nouns,
        "flags": [{"noun": f.noun, "unused": f.unused_matches} for f in result.flags],
        "any_flag": result.has_issue,
        "plan_chars": len(plan_text),
    }


def main():
    gt = json.load(open(GT_PATH))["scores"]
    task_ids = sorted(gt.keys(), key=lambda x: int(x.split("_")[1]))
    print(f"=== PLAN verifier on {len(task_ids)} tasks ===", flush=True)
    print(f"GT mean: {sum(gt.values())/len(gt):.3f}, n_fail={sum(1 for v in gt.values() if v<1)}", flush=True)

    results = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(run_one, tid): tid for tid in task_ids}
        done = 0
        for fut in as_completed(futures):
            tid = futures[fut]
            done += 1
            try:
                r = fut.result()
                r["gt_score"] = gt.get(tid, None)
                results.append(r)
                gtv = r["gt_score"]
                cat = "FAIL" if gtv < 1.0 else "ok  "
                flag_chars = ",".join(f["noun"][:15] for f in r["flags"]) if r["flags"] else "-"
                print(f"  [{done}/{len(task_ids)}] [{cat}] {tid} gt={gtv:.2f} flag={'X' if r['any_flag'] else '-'} {flag_chars}", flush=True)
            except Exception as exc:
                print(f"  [{done}/{len(task_ids)}] {tid}: ERROR {exc}", flush=True)

    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2, default=str))

    # Confusion matrix
    tp = sum(1 for r in results if r["gt_score"] < 1.0 and r["any_flag"])
    fn = sum(1 for r in results if r["gt_score"] < 1.0 and not r["any_flag"])
    fp = sum(1 for r in results if r["gt_score"] >= 1.0 and r["any_flag"])
    tn = sum(1 for r in results if r["gt_score"] >= 1.0 and not r["any_flag"])
    n_fail = tp + fn
    n_ok = fp + tn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / n_fail if n_fail > 0 else 0.0
    specificity = tn / n_ok if n_ok > 0 else 0.0

    summary_lines = [
        "# PLAN verifier full-50 — summary\n",
        f"## Confusion matrix",
        f"|       | flag=X | flag=- |",
        f"|-------|--------|--------|",
        f"| FAIL  | TP={tp}  | FN={fn}  |",
        f"| OK    | FP={fp}  | TN={tn}  |",
        f"",
        f"- Precision: {precision:.3f} (flag → fail)",
        f"- Recall:    {recall:.3f} (failure → flagged)",
        f"- Specificity: {specificity:.3f} (success → not flagged)",
        f"",
        "## Per-task",
        "| task | gt | flag | nouns | flagged |",
        "|---|---|---|---|---|",
    ]
    for r in sorted(results, key=lambda x: int(x["task_id"].split("_")[1])):
        flagged = [f["noun"] for f in r["flags"]]
        summary_lines.append(
            f"| {r['task_id']} | {r['gt_score']:.1f} | {'X' if r['any_flag'] else '-'} | "
            f"{len(r['nouns'])} | {flagged} |"
        )
    (OUT_DIR / "summary.md").write_text("\n".join(summary_lines))
    print(f"\nP={precision:.3f}, R={recall:.3f}, Specificity={specificity:.3f}", flush=True)
    print(f"Summary: {OUT_DIR / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
