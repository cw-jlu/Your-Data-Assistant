"""Standalone POC for doc-preprocessing approach (= Sub-problem A + B).

Tests on 2 known-failing tasks:
- task_169 (Type B): knowledge.md formula ambiguity
- task_344 (Type F+G): narrative doc structuring + medical threshold extraction

For each:
1. Pass 1 (knowledge disambiguate): LLM reads question + knowledge.md, outputs
   operational pseudo-SQL with explicit scope.
2. Pass 2 (narrative structure): LLM reads question + narrative doc, outputs
   structured CSV table.
3. Augment preamble with the preprocess outputs.
4. Run a single-attempt ReAct (= exp_109 base, no PSV) on the augmented preamble.
5. Score vs gold.

Usage:
    uv run python scripts/test_doc_preprocess.py
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from dotenv import load_dotenv
load_dotenv()

from kobushi_core.benchmark.dataset import DABenchPublicDataset
from kobushi_core.eval.csv_compare import _evaluate_task, EvaluationOptions
from kobushi_core.model import OpenAIModelAdapter, ModelMessage
from experiments.exp_109_plan_first_strengthen.agent import ReActAgent, ReActAgentConfig
from experiments.exp_109_plan_first_strengthen.preamble import build_preamble
from experiments.exp_109_plan_first_strengthen.tools.registry import (
    ToolRegistry,
    create_default_tool_registry,
)


def make_model() -> OpenAIModelAdapter:
    api_base = os.environ["AGENT_API_BASE"]
    api_key = os.environ["AGENT_API_KEY"]
    headers = {
        "CF-Access-Client-Id": os.environ.get("CF_ACCESS_CLIENT_ID", ""),
        "CF-Access-Client-Secret": os.environ.get("CF_ACCESS_CLIENT_SECRET", ""),
    }
    return OpenAIModelAdapter(
        model="qwen3.5-35b-a3b",
        api_base=api_base,
        api_key=api_key,
        temperature=0.6,
        extra_headers=headers,
        presence_penalty=1.0,
    )


# ============================================================================
# Pass 1: Knowledge.md disambiguator
# ============================================================================

PASS1_PROMPT = """\
You are a knowledge-base disambiguator for data-analysis questions.

Given a question and the dataset's knowledge.md, produce a focused
"operational knowledge" markdown section that:

1. Identifies the metric / formula relevant to the question.
2. Restates the formula in OPERATIONAL form (= unambiguous SQL pseudo-code
   with explicit scope, e.g. per-customer aggregation vs population sum).
3. Notes any glossary terms used in the question (= what each ambiguous
   noun refers to in the schema).
4. Surfaces relevant Example sections from knowledge.md, expanding any
   abstract Metric descriptions into concrete SQL.

Output ONLY a markdown section. No JSON, no fenced code blocks at the top
level. The section will be appended to the agent's preamble verbatim.

Format:
# Operational Knowledge (LLM-disambiguated)

## Relevant metric: <name>
Formula:
  <SQL-like pseudo-code with EXPLICIT scope, e.g.:
   SELECT AVG(yearly.total) / 12 FROM (
     SELECT CustomerID, SUM(Consumption) AS total FROM yearmonth
     WHERE substr(Date, 1, 4) = '2013' GROUP BY CustomerID
   ) yearly
   WHERE CustomerID IN (SELECT CustomerID FROM customers WHERE Segment='SME')>

## Glossary terms used in question
- <term>: <schema mapping>
- <term>: <schema mapping>

## Example reference
- knowledge.md "Example 2" matches this question (= "Average Monthly Consumption for SME in 2013").
"""


def pass1_disambiguate(task, model: OpenAIModelAdapter) -> str:
    knowledge_md = (task.context_dir / "knowledge.md").read_text()
    user = (
        f"# Question\n{task.question}\n\n"
        f"# knowledge.md\n{knowledge_md}\n\n"
        f"Generate the Operational Knowledge section."
    )
    messages = [
        ModelMessage(role="system", content=PASS1_PROMPT),
        ModelMessage(role="user", content=user),
    ]
    raw = model.complete(messages, enable_thinking=True, max_tokens=131072)
    return raw.strip()


# ============================================================================
# Pass 2: Narrative doc structurer
# ============================================================================

PASS2_PROMPT = """\
You are a narrative-to-structured-data extractor.

Given a question and a narrative document (= prose describing entities like
patients / heroes / events), produce a CSV table with one row per entity
and only the columns the question needs.

Use extended thinking to:
- Identify what entity type the question asks about (e.g. patients, races)
- Determine which columns are needed (= IDs, key attributes, dates, values)
- Parse the narrative to extract those columns
- Be CONSERVATIVE: only include rows where you can confidently extract values.
  Skip rows with missing or ambiguous data; do NOT hallucinate.

Output ONLY the CSV (header + rows). No JSON, no markdown fences.

If the document is too large to fully process within the thinking budget,
extract as many rows as you can and note "[truncated]" as the last row.
"""


def pass2_structure_narrative(task, doc_relpath: str, model: OpenAIModelAdapter) -> str:
    doc_path = task.context_dir / doc_relpath
    if not doc_path.exists():
        return f"# Pass 2 skipped: {doc_relpath} not found"
    text = doc_path.read_text()
    # Cap at 80KB for the LLM input (= save tokens for thinking)
    text = text[:80000]
    user = (
        f"# Question\n{task.question}\n\n"
        f"# Narrative document ({doc_relpath}, first 80KB)\n{text}\n\n"
        f"Extract a CSV table relevant to answering this question."
    )
    messages = [
        ModelMessage(role="system", content=PASS2_PROMPT),
        ModelMessage(role="user", content=user),
    ]
    raw = model.complete(messages, enable_thinking=True, max_tokens=131072)
    return raw.strip()


# ============================================================================
# Test harness
# ============================================================================

def run_test_169(model: OpenAIModelAdapter, ds: DABenchPublicDataset, tools: ToolRegistry) -> dict:
    print("\n" + "=" * 70)
    print("TEST: task_169 (= AVG/12 formula ambiguity, Type B)")
    print("=" * 70)
    task = ds.get_task("task_169")

    # Baseline: build standard rich preamble
    base_preamble = build_preamble(task)
    print(f"baseline preamble: {base_preamble.char_count} chars")

    # Pass 1: disambiguate knowledge.md
    t0 = time.time()
    pass1_out = pass1_disambiguate(task, model)
    print(f"\n--- pass1 latency: {time.time()-t0:.1f}s ---")
    print("--- pass1 output (first 1500 chars) ---")
    print(pass1_out[:1500])

    # Build augmented preamble = base + pass1 inject
    augmented = base_preamble.text + "\n\n---\n\n" + pass1_out

    # Run single-attempt agent on augmented preamble
    print(f"\n--- running agent on augmented preamble ({len(augmented)} chars) ---")
    agent = ReActAgent(
        model=model,
        tools=tools,
        config=ReActAgentConfig(max_steps=24, min_steps=4),
        preamble=augmented,
    )
    t1 = time.time()
    result = agent.run(task)
    print(f"agent latency: {time.time()-t1:.1f}s, steps: {len(result.steps)}, succeeded: {result.succeeded}")

    if not result.succeeded or not result.answer:
        print(f"FAIL: {result.failure_reason}")
        return {"task": "task_169", "score": 0.0, "fail_reason": result.failure_reason}

    # Score against gold
    pred_dir = REPO / "artifacts" / "doc_preprocess_test" / "task_169"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_p = pred_dir / "prediction.csv"
    with open(pred_p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(result.answer.columns)
        for row in result.answer.rows:
            w.writerow(row)

    gold_p = REPO / "data" / "public" / "output" / "task_169" / "gold.csv"
    e = _evaluate_task(
        task_id="task_169",
        prediction_path=pred_p,
        gold_path=gold_p,
        options=EvaluationOptions(),
    )
    print(f"\n--- score ---")
    print(f"  λ0.5 = {e.official_score_lambda_0_5:.4f}")
    print(f"  recall = {e.recall:.2f}")
    print(f"  pred: {e.rows_pred}r × {len(e.cols_pred or []) if isinstance(e.cols_pred, list) else e.cols_pred}c")
    print(f"  pred values: {result.answer.rows[:3]}")
    print(f"  gold values: 459.9562642871061 (= 1 row × 1 col)")

    return {
        "task": "task_169",
        "score": e.official_score_lambda_0_5,
        "pred": result.answer.to_dict(),
    }


def run_test_344(model: OpenAIModelAdapter, ds: DABenchPublicDataset, tools: ToolRegistry) -> dict:
    print("\n" + "=" * 70)
    print("TEST: task_344 (= medical narrative + missing thresholds, Type F+G)")
    print("=" * 70)
    task = ds.get_task("task_344")

    base_preamble = build_preamble(task)
    print(f"baseline preamble: {base_preamble.char_count} chars")

    # Pass 1: disambiguate knowledge.md
    t0 = time.time()
    pass1_out = pass1_disambiguate(task, model)
    print(f"\n--- pass1 latency: {time.time()-t0:.1f}s ---")
    print("--- pass1 output (first 1200 chars) ---")
    print(pass1_out[:1200])

    # Pass 2: structure narrative (Patient.md)
    t1 = time.time()
    pass2_out = pass2_structure_narrative(task, "doc/Patient.md", model)
    print(f"\n--- pass2 latency: {time.time()-t1:.1f}s ---")
    print("--- pass2 output (first 1500 chars) ---")
    print(pass2_out[:1500])

    augmented = (
        base_preamble.text + "\n\n---\n\n" + pass1_out + "\n\n---\n\n"
        "# Structured patient table (= Pass 2 narrative extraction)\n\n```csv\n"
        + pass2_out
        + "\n```"
    )

    print(f"\n--- running agent on augmented preamble ({len(augmented)} chars) ---")
    agent = ReActAgent(
        model=model,
        tools=tools,
        config=ReActAgentConfig(max_steps=24, min_steps=4),
        preamble=augmented,
    )
    t2 = time.time()
    result = agent.run(task)
    print(f"agent latency: {time.time()-t2:.1f}s, steps: {len(result.steps)}, succeeded: {result.succeeded}")

    if not result.succeeded or not result.answer:
        print(f"FAIL: {result.failure_reason}")
        return {"task": "task_344", "score": 0.0, "fail_reason": result.failure_reason}

    pred_dir = REPO / "artifacts" / "doc_preprocess_test" / "task_344"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_p = pred_dir / "prediction.csv"
    with open(pred_p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(result.answer.columns)
        for row in result.answer.rows:
            w.writerow(row)

    gold_p = REPO / "data" / "public" / "output" / "task_344" / "gold.csv"
    e = _evaluate_task(
        task_id="task_344",
        prediction_path=pred_p,
        gold_path=gold_p,
        options=EvaluationOptions(),
    )
    print(f"\n--- score ---")
    print(f"  λ0.5 = {e.official_score_lambda_0_5:.4f}")
    print(f"  pred values: {result.answer.rows[:3]}")
    print(f"  gold values: 4 (= 1 row × 1 col)")

    return {
        "task": "task_344",
        "score": e.official_score_lambda_0_5,
        "pred": result.answer.to_dict(),
    }


def main():
    model = make_model()
    ds = DABenchPublicDataset(root_dir=REPO / "data" / "public" / "input")
    tools = create_default_tool_registry()

    results = []
    results.append(run_test_169(model, ds, tools))
    results.append(run_test_344(model, ds, tools))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for r in results:
        print(f"  {r['task']}: λ0.5 = {r.get('score', 0):.4f}")
        if r.get("fail_reason"):
            print(f"    FAIL: {r['fail_reason']}")
    print("\nBaseline (= no preprocessing) on these tasks:")
    print("  task_169: 0.0000 (= consistently zero across 130 runs)")
    print("  task_344: 0.0000 (= consistently zero across 130 runs)")

    # Save results
    out_p = REPO / "artifacts" / "doc_preprocess_test" / "results.json"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str))
    print(f"\nResults: {out_p}")


if __name__ == "__main__":
    main()
