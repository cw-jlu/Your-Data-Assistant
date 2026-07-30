"""Compare PLAN output: exp_122 (= baseline) vs exp_132 (= + literal match rule).

1-shot LLM call, no tool execution. We just want to see if adding the
LITERAL SCHEMA MATCH RULE flips the CHOOSING decision on the failed tasks.

Usage:
    uv run python scripts/plan_literal_match_test.py
"""
from __future__ import annotations

import os
import re
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


OUT_DIR = REPO / "artifacts" / "plan_literal_match"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_TASKS = ["task_25", "task_163", "task_169", "task_180", "task_199",
                "task_11", "task_67"]  # last 2 = ok baseline for control
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


# Synthetic tool description (= PLAN phase 1-shot, no actual tools needed)
_PLAN_TOOL_DESC = """- list_context: list the task's context files. - complete_phase: advance phase."""


def build_messages(exp_module: str, task):
    if exp_module == "exp_122":
        from experiments.exp_122_column_auditor.prompt import (
            build_phased_system_prompt, build_task_prompt,
        )
        from experiments.exp_122_column_auditor.preamble import build_preamble
    elif exp_module == "exp_132":
        from experiments.exp_132_literal_match.prompt import (
            build_phased_system_prompt, build_task_prompt,
        )
        from experiments.exp_132_literal_match.preamble import build_preamble
    else:
        raise ValueError(exp_module)

    preamble = build_preamble(task)
    system = build_phased_system_prompt("plan", _PLAN_TOOL_DESC)
    user = f"{preamble.text}\n\n{build_task_prompt(task)}"
    return [
        ModelMessage(role="system", content=system),
        ModelMessage(role="user", content=user),
    ]


def run_one(args):
    tid, cfg, i = args
    ds = DABenchPublicDataset(root_dir=REPO / "data" / "public" / "input")
    task = ds.get_task(tid)
    messages = build_messages(cfg, task)
    model = make_model()
    t0 = time.time()
    try:
        text = model.complete(messages, enable_thinking=True, max_tokens=16384)
    except Exception as exc:
        return (tid, cfg, i, None, None, f"err: {exc}")
    latency = time.time() - t0
    # Extract first column choice (= "CHOOSING <table>.<col>" or "CHOOSING <col>")
    choices = re.findall(r"CHOOSING\s+(\S+?)(?:\s|,|;|\.|$|because|;)", text, re.IGNORECASE)
    choice = choices[0] if choices else "?"
    # Save full text
    task_dir = OUT_DIR / tid
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / f"{cfg}_{i}.md").write_text(
        f"# {tid} | {cfg} | sample {i}\n\nQ: {task.question}\n\nLatency: {latency:.1f}s, choice='{choice}'\n\n## Response\n```\n{text}\n```\n"
    )
    return (tid, cfg, i, choice, latency, None)


def main():
    print(f"=== PLAN literal match test: {len(TARGET_TASKS)} tasks × {N_SAMPLES} samples × 2 cfg ===", flush=True)
    jobs = []
    for tid in TARGET_TASKS:
        for cfg in ["exp_122", "exp_132"]:
            for i in range(N_SAMPLES):
                jobs.append((tid, cfg, i))

    results = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        futures = [ex.submit(run_one, j) for j in jobs]
        for fut in as_completed(futures):
            tid, cfg, i, choice, lat, err = fut.result()
            if err:
                print(f"  {tid} {cfg}#{i} ERR: {err}", flush=True)
            else:
                print(f"  {tid} {cfg}#{i} choice='{choice}' lat={lat:.0f}s", flush=True)
                results.append((tid, cfg, choice))

    # Summary: per task, count choices per config
    from collections import Counter
    lines = ["# PLAN literal match test — summary\n", "| task | cfg | choices (count) |", "|---|---|---|"]
    for tid in TARGET_TASKS:
        for cfg in ["exp_122", "exp_132"]:
            choices = [r[2] for r in results if r[0] == tid and r[1] == cfg]
            c = Counter(choices)
            lines.append(f"| {tid} | {cfg} | {dict(c)} |")
    (OUT_DIR / "summary.md").write_text("\n".join(lines))
    print(f"\nSummary: {OUT_DIR / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
