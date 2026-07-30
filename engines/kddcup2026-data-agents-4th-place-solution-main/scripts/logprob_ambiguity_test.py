"""Logprob-based ambiguity detection POC.

For each task:
  1. Run a single PLAN-style LLM call with `complete_with_logprobs`
  2. Locate decision tokens (= top of "CHOOSING <choice>" patterns)
  3. Compute per-token Shannon entropy from top-K logprobs
  4. Report: max entropy, mean entropy at decision points

Hypothesis: failed tasks (= ambiguous interpretation) show higher entropy at
the choice token vs success tasks where the model is confident.

Usage:
    uv run python scripts/logprob_ambiguity_test.py
"""
from __future__ import annotations

import math
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
from experiments.exp_122_column_auditor.preamble import build_preamble


OUT_DIR = REPO / "artifacts" / "logprob_ambiguity"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Mix of failed tasks (= expected ambiguous) + success tasks (= expected clear)
FAILED_TASKS = ["task_25", "task_163", "task_169", "task_180", "task_199"]
SUCCESS_TASKS = ["task_11", "task_19", "task_67", "task_74", "task_75"]


_PLAN_SYSTEM = """You are doing a 1-shot SQL planning step. Given a question + schema preview,
choose interpretations for each ambiguous noun.

Your output MUST be a single line of the form:
  CHOOSING <table>.<column> because <one short reason>

Where <table>.<column> is the SINGLE column you'd use as the main answer source.
If multiple candidates exist, pick ONE — the model's logprob distribution at
this token will reveal whether you were torn between them.

Output ONLY that line. No other text.
"""


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


def shannon_entropy(top_logprobs: dict[str, float]) -> float:
    """Compute Shannon entropy in bits from a top-K logprob dict.
    Note: logprobs may not sum to 1 (= only top-K). We normalize the captured
    mass."""
    if not top_logprobs:
        return 0.0
    probs = [math.exp(lp) for lp in top_logprobs.values()]
    total = sum(probs)
    if total <= 0:
        return 0.0
    probs = [p / total for p in probs]
    return -sum(p * math.log2(p) for p in probs if p > 0)


def find_choice_token_entropies(
    text: str, top_buf: list[dict[str, float]]
) -> dict:
    """Locate the choice-decision tokens in the output and return their entropy.

    Strategy: find "CHOOSING " in text, then look at the next few tokens — the
    column-name choice (e.g. "expense.cost") spans 2-4 tokens. Report the
    HIGHEST per-token entropy across these tokens (= measures whether the
    model was torn at any sub-token of the choice).

    The OpenAI API returns one logprob entry per generated token, in order.
    We don't get exact char→token offsets, so we approximate by matching the
    output text position.
    """
    if not top_buf:
        return {"available": False}

    # Reconstruct token positions by re-joining top_buf's "token" (= the
    # actually-chosen one). Each entry in top_buf is {token_str: logprob}.
    # The chosen token is the one with the MAX logprob within that entry's
    # top-K dict. But we lost which token was actually generated. Approximation:
    # accumulate top-1 tokens.
    char_pos = 0
    token_positions = []  # list of (start_char, end_char, entry_idx, entropy)
    for i, entry in enumerate(top_buf):
        if not entry:
            continue
        # Pick the highest-prob token (likely the chosen one)
        top_tok = max(entry.items(), key=lambda x: x[1])[0]
        ent = shannon_entropy(entry)
        token_positions.append((char_pos, char_pos + len(top_tok), i, ent, top_tok))
        char_pos += len(top_tok)

    # Find "CHOOSING " in text
    m = re.search(r"CHOOSING\s+", text)
    if not m:
        return {"available": True, "found_choosing": False, "n_tokens": len(top_buf)}
    choice_start = m.end()
    # Find end of choice (= space or 'because')
    end_match = re.search(r"\s*(because|\.|,)", text[choice_start:])
    choice_end = choice_start + (end_match.start() if end_match else 30)

    # Find entries overlapping the choice span
    choice_entropies = []
    for (start, end, idx, ent, tok) in token_positions:
        # Token overlaps choice span?
        if end > choice_start and start < choice_end:
            choice_entropies.append({"idx": idx, "entropy": ent, "token": tok[:30]})

    if not choice_entropies:
        return {"available": True, "found_choosing": True, "no_overlap": True}

    return {
        "available": True,
        "found_choosing": True,
        "choice_text": text[choice_start:choice_end].strip(),
        "n_choice_tokens": len(choice_entropies),
        "max_entropy": max(e["entropy"] for e in choice_entropies),
        "mean_entropy": sum(e["entropy"] for e in choice_entropies) / len(choice_entropies),
        "details": choice_entropies[:6],
    }


def run_one(task_id: str) -> dict:
    ds = DABenchPublicDataset(root_dir=REPO / "data" / "public" / "input")
    task = ds.get_task(task_id)
    preamble = build_preamble(task)
    model = make_model()

    user = f"{preamble.text}\n\n# Question\n{task.question}\n\nNow output ONLY the CHOOSING line."
    messages = [
        ModelMessage(role="system", content=_PLAN_SYSTEM),
        ModelMessage(role="user", content=user),
    ]

    t0 = time.time()
    text, top_buf = model.complete_with_logprobs(
        messages, top_logprobs=5, enable_thinking=False, max_tokens=512
    )
    latency = time.time() - t0

    result = find_choice_token_entropies(text, top_buf)
    result["task_id"] = task_id
    result["latency_s"] = int(latency)
    result["text"] = text[:200]
    return result


def main():
    print("=== Logprob ambiguity test ===", flush=True)
    all_tasks = [(t, "fail") for t in FAILED_TASKS] + [(t, "ok") for t in SUCCESS_TASKS]
    print(f"Total tasks: {len(all_tasks)}", flush=True)

    rows = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(run_one, t): (t, cat) for t, cat in all_tasks}
        for fut in as_completed(futures):
            tid, cat = futures[fut]
            try:
                r = fut.result()
                r["category"] = cat
                rows.append(r)
                me = r.get("max_entropy", 0.0)
                ct = r.get("choice_text", "?")
                print(f"  [{cat:4s}] {tid}: max_ent={me:.3f} choice='{ct[:40]}' lat={r.get('latency_s')}s", flush=True)
            except Exception as exc:
                print(f"  [{cat:4s}] {tid}: ERROR {exc}", flush=True)

    # Save + summary
    import json as J
    (OUT_DIR / "raw.json").write_text(J.dumps(rows, indent=2, default=str))

    # Summary table
    lines = ["# Logprob Ambiguity Test — Summary\n",
             "Higher entropy at the CHOOSING decision token = model is uncertain.",
             "Hypothesis: failed tasks should show higher entropy than success tasks.\n",
             "| task | category | max_entropy | choice |",
             "|---|---|---|---|"]
    rows.sort(key=lambda r: r.get("max_entropy", 0), reverse=True)
    for r in rows:
        ent = r.get("max_entropy", 0)
        choice = r.get("choice_text", "?")[:40]
        lines.append(f"| {r['task_id']} | {r['category']} | {ent:.3f} | {choice} |")

    # By-category stats
    for cat in ["fail", "ok"]:
        cat_rows = [r for r in rows if r.get("category") == cat and r.get("max_entropy") is not None]
        if cat_rows:
            avg = sum(r["max_entropy"] for r in cat_rows) / len(cat_rows)
            mx = max(r["max_entropy"] for r in cat_rows)
            lines.append(f"\n## {cat}: n={len(cat_rows)}, avg_max_entropy={avg:.3f}, max={mx:.3f}")
    (OUT_DIR / "summary.md").write_text("\n".join(lines))
    print(f"\nSummary: {OUT_DIR / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
