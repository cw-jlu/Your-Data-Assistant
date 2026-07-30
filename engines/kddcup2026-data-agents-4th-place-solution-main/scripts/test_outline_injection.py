"""POC: inject doc outline into preamble + run single agent attempt on task_418.

Compare to baseline (= no outline) for one task to see if step count drops
and/or score improves.
"""
import csv as _csv
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def parse_doc_outline(md_path: Path) -> list[dict]:
    """Extract only section title + line range from markdown headers.
    No domain-specific extraction (= no whitelist of anchor words / units / etc.)
    so the hint contains NO test-data prior knowledge.
    """
    text = md_path.read_text()
    lines = text.split("\n")
    n = len(lines)
    headers = []
    for i, line in enumerate(lines, start=1):
        m = re.match(r"^(#+)\s+(.*)$", line)
        if m:
            headers.append({"level": len(m.group(1)), "title": m.group(2).strip(), "line_start": i})
    for idx, h in enumerate(headers):
        end_line = n
        for nxt in headers[idx + 1:]:
            if nxt["level"] <= h["level"]:
                end_line = nxt["line_start"] - 1
                break
        h["line_end"] = end_line
    return headers


def format_outline_for_agent(md_path: Path) -> str:
    headers = parse_doc_outline(md_path)
    if not headers:
        # no headers — show file length only
        n_lines = len(md_path.read_text().split("\n"))
        return f"## {md_path.name} ({n_lines} lines) — no section headers"
    lines = [f"## {md_path.name} ({headers[0]['line_end']} lines)"]
    for h in headers:
        indent = "  " * (h["level"] - 1)
        title = h["title"]
        if len(title) > 80:
            title = title[:77] + "..."
        line_range = f"L{h['line_start']:>4}-{h['line_end']:>4}"
        lines.append(f"{indent}{line_range}  {title}")
    return "\n".join(lines)


OUTLINE_BLOCK_HEADER = "\n\n# DOC OUTLINE (= pre-extracted section tree, use `read_doc(path, offset=L_start)` to jump)\n\n"


def build_outline_block(task_context_dir: Path) -> str:
    docs = sorted(task_context_dir.glob("doc/*.md"))
    if not docs:
        return ""
    parts = [OUTLINE_BLOCK_HEADER]
    for md in docs:
        parts.append(format_outline_for_agent(md))
        parts.append("")  # blank line between docs
    parts.append(
        "Use `read_doc(path, offset=L_start, max_chars=N)` to jump directly into\n"
        "a relevant section without grep scanning the whole file.\n"
    )
    return "\n".join(parts)


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")

    from kobushi_core.eval.csv_compare import _evaluate_task, EvaluationOptions
    from kobushi_core.benchmark.dataset import DABenchPublicDataset
    from kobushi_core.model import OpenAIModelAdapter

    from experiments.exp_137_math_advisor.preamble import build_preamble
    from experiments.exp_137_math_advisor.phased_agent import PhasedReActAgent, PhasedAgentConfig
    from experiments.exp_137_math_advisor.tools.registry import create_default_tool_registry

    TID = "task_418"

    ds = DABenchPublicDataset(root_dir=ROOT / "data" / "public" / "input")
    task = ds.get_task(TID)
    print(f"Q: {task.question}", flush=True)

    outline_block = build_outline_block(task.context_dir)
    print(f"\n=== OUTLINE BLOCK ({len(outline_block)} chars) ===")
    print(outline_block[:2500])
    print("...\n=== END OUTLINE ===\n", flush=True)

    # Build preamble with formula injection (= built-in to exp_137.preamble) + outline append
    preamble = build_preamble(task)
    injected = preamble.text + outline_block

    m = OpenAIModelAdapter(
        model="qwen3.5-35b-a3b",
        api_base=os.environ["AGENT_API_BASE"],
        api_key=os.environ["AGENT_API_KEY"],
        temperature=0.6,
        extra_headers={
            "CF-Access-Client-Id": os.environ.get("CF_ACCESS_CLIENT_ID", ""),
            "CF-Access-Client-Secret": os.environ.get("CF_ACCESS_CLIENT_SECRET", ""),
        },
    )
    tools = create_default_tool_registry(
        auditor_model=m, question_provider=lambda: task.question, context_dir=task.context_dir,
    )
    agent = PhasedReActAgent(
        model=m, tools=tools,
        config=PhasedAgentConfig(max_steps=64, min_explore_queries=3),
        preamble=injected,
    )
    # Per-step trace persistence (= written immediately by agent.run as each step completes)
    out_dir = ROOT / "artifacts" / "test_outline_injection" / TID
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / "trace.log"
    trace_path.write_text("")  # truncate
    agent.trace_log_path = str(trace_path)
    print(f"[{time.strftime('%H:%M:%S')}] agent.run() starting (preamble: {len(injected)} chars, trace→{trace_path})...", flush=True)
    t0 = time.time()
    r = agent.run(task)
    print(f"[{time.strftime('%H:%M:%S')}] done in {time.time() - t0:.1f}s, succeeded={r.succeeded}, n_steps={len(r.steps)}", flush=True)

    if r.answer and r.answer.rows:
        out_dir = ROOT / "artifacts" / "test_outline_injection" / TID
        out_dir.mkdir(parents=True, exist_ok=True)
        pred = out_dir / "prediction.csv"
        with pred.open("w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(r.answer.columns)
            for row in r.answer.rows:
                w.writerow(row)
        gold = ROOT / "data" / "public" / "output" / TID / "gold.csv"
        e = _evaluate_task(task_id=TID, prediction_path=pred, gold_path=gold, options=EvaluationOptions())
        print(f"  ANSWER cols={list(r.answer.columns)}, rows={[list(row) for row in r.answer.rows[:5]]}")
        print(f"  GOLD: {gold.read_text().strip()}")
        print(f"  OFFICIAL SCORE: {e.official_score_lambda_0_5}")
    else:
        print("  NO ANSWER")

    # Print tool action summary (= count grep/read_doc vs other actions)
    from collections import Counter
    actions = Counter(s.action for s in r.steps)
    print(f"\n=== action histogram ({len(r.steps)} steps) ===")
    for a, c in actions.most_common():
        print(f"  {a}: {c}")


if __name__ == "__main__":
    main()
