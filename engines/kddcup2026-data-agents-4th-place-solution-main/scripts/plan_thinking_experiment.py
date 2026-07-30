"""Compare PLAN-phase thinking: exp_122 baseline vs deep-think variant.

For each of 5 failed tasks, sample N=5 PLAN-phase responses under two configs:
  A. baseline_122 = current PLAN system prompt + default max_tokens (= 32K)
  B. deep_think   = baseline + "think deeply, consider multiple interpretations"
                    suffix + max_tokens=64K (= 2x thinking budget)

Output: artifacts/plan_thinking/<task_id>/<config>_<n>.md
        artifacts/plan_thinking/summary.md (= side-by-side compact)

Usage:
    uv run python scripts/plan_thinking_experiment.py
"""
from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from dotenv import load_dotenv
load_dotenv(REPO / ".env")

from kobushi_core.benchmark.dataset import DABenchPublicDataset
from kobushi_core.model import OpenAIModelAdapter, ModelMessage
from experiments.exp_122_column_auditor.preamble import build_preamble
from experiments.exp_122_column_auditor.prompt import (
    build_phased_system_prompt,
    build_task_prompt,
)


OUT_DIR = REPO / "artifacts" / "plan_thinking"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 5 failed tasks spanning different stumble patterns
TARGET_TASKS = ["task_25", "task_163", "task_169", "task_180", "task_199"]

N_SAMPLES = 5


def make_model(max_tokens: int = 32768) -> OpenAIModelAdapter:
    return OpenAIModelAdapter(
        model="qwen3.5-35b-a3b",
        api_base=os.environ["AGENT_API_BASE"],
        api_key=os.environ["AGENT_API_KEY"],
        temperature=0.6,
        max_tokens=max_tokens,
        extra_headers={
            "CF-Access-Client-Id": os.environ.get("CF_ACCESS_CLIENT_ID", ""),
            "CF-Access-Client-Secret": os.environ.get("CF_ACCESS_CLIENT_SECRET", ""),
        },
    )


# Synthetic tool description (since we don't run actual tools, just need stub)
_PLAN_TOOL_DESC = """- list_context: list the task's context files (= optional, max_depth controls recursion).
- complete_phase: advance to the next phase. action_input={"next_phase": "explore"}."""


DEEP_THINK_SUFFIX = """

THINKING BUDGET: This is a critical planning step. Take time to deeply reason about:
- What does the question REALLY ask? Are there multiple interpretations? Pick the most natural one explicitly.
- Are there ambiguous terms (e.g. "type", "amount", "lowest", "per unit")? Map each to a concrete schema column with reasoning.
- What is the expected RESULT SHAPE? Will the gold csv have 1 column or many? 1 row or many? Are ties possible (in which case all tied rows must be returned, not just one)?
- What is the ARITHMETIC LEVEL? Is this per-row, per-group, or aggregate? E.g. "average" = AVG(...), not SUM/N if data has per-customer rows.
- What is the FILTER SCOPE? Is "X-related" loose or specific? Should you include all rows matching X or only a specific subset?
- Walk through the SQL skeleton in your head before committing. Spell out: SELECT <what cols?> FROM <which tables?> JOIN <on what keys?> WHERE <what conditions?> GROUP BY (if needed).
You may produce a multi-paragraph thought analyzing each of these points. Do not rush."""


def build_plan_messages(task, preamble_text: str, deep_think: bool) -> list[ModelMessage]:
    system = build_phased_system_prompt("plan", _PLAN_TOOL_DESC)
    if deep_think:
        system += DEEP_THINK_SUFFIX
    task_prompt = build_task_prompt(task)
    user = task_prompt
    if preamble_text:
        user = f"{preamble_text}\n\n{task_prompt}"
    return [
        ModelMessage(role="system", content=system),
        ModelMessage(role="user", content=user),
    ]


def extract_thinking_and_response(raw: str) -> tuple[str, str]:
    """Qwen3.5 thinking mode outputs <think>...</think> blocks. Separate them."""
    think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    thinking_chunks = think_pattern.findall(raw)
    response = think_pattern.sub("", raw).strip()
    thinking = "\n\n---\n\n".join(thinking_chunks).strip()
    return thinking, response


def run_one(task_id: str, n_samples: int = N_SAMPLES) -> dict:
    ds = DABenchPublicDataset(root_dir=REPO / "data" / "public" / "input")
    task = ds.get_task(task_id)
    preamble = build_preamble(task)

    task_dir = OUT_DIR / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    results = {"task_id": task_id, "question": task.question, "samples": []}

    for config_name, deep_think, max_tokens in [
        ("baseline_122", False, 32768),
        ("deep_think", True, 65536),
    ]:
        model = make_model(max_tokens=max_tokens)
        messages = build_plan_messages(task, preamble.text, deep_think=deep_think)

        for i in range(n_samples):
            print(f"  [{config_name} #{i}]", flush=True)
            t0 = time.time()
            try:
                raw = model.complete(messages)
                thinking, response = extract_thinking_and_response(raw)
                latency = time.time() - t0
                results["samples"].append({
                    "config": config_name,
                    "sample_idx": i,
                    "latency_s": int(latency),
                    "thinking_len": len(thinking),
                    "response_len": len(response),
                    "thinking_preview": thinking[:300],
                    "response_preview": response[:500],
                })
                # Save full output
                path = task_dir / f"{config_name}_{i}.md"
                path.write_text(
                    f"# {task_id} | {config_name} | sample {i}\n\n"
                    f"## Question\n{task.question}\n\n"
                    f"## Latency: {latency:.1f}s\n\n"
                    f"## Thinking ({len(thinking)} chars)\n```\n{thinking}\n```\n\n"
                    f"## Response ({len(response)} chars)\n```\n{response}\n```\n"
                )
            except Exception as exc:
                print(f"    ERROR: {exc}", flush=True)
                results["samples"].append({
                    "config": config_name,
                    "sample_idx": i,
                    "error": str(exc),
                })

    return results


def main():
    print(f"=== PLAN thinking experiment: {len(TARGET_TASKS)} tasks × {N_SAMPLES} samples × 2 configs ===", flush=True)
    all_results = []
    for tid in TARGET_TASKS:
        print(f"\n## {tid}", flush=True)
        r = run_one(tid)
        all_results.append(r)

    # Summary report
    lines = ["# PLAN thinking experiment — summary\n"]
    for r in all_results:
        lines.append(f"\n## {r['task_id']}: {r['question'][:80]}\n")
        # Group by config
        for cfg in ["baseline_122", "deep_think"]:
            samples = [s for s in r["samples"] if s.get("config") == cfg and "error" not in s]
            if not samples:
                lines.append(f"### {cfg}: all failed\n")
                continue
            avg_thinking = sum(s["thinking_len"] for s in samples) / len(samples)
            avg_response = sum(s["response_len"] for s in samples) / len(samples)
            avg_latency = sum(s["latency_s"] for s in samples) / len(samples)
            lines.append(
                f"### {cfg}\n"
                f"- avg thinking chars: {avg_thinking:.0f}\n"
                f"- avg response chars: {avg_response:.0f}\n"
                f"- avg latency: {avg_latency:.0f}s\n"
                f"- n successful: {len(samples)}/{N_SAMPLES}\n"
            )
    (OUT_DIR / "summary.md").write_text("\n".join(lines))
    print(f"\nSummary written to {OUT_DIR / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
