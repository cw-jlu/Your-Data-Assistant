"""Quick POC: measure question ambiguity by asking LLM to enumerate
all reasonable interpretations of a question, multiple times.

Hypothesis: more interpretations + higher variance across samples =
more ambiguous question = higher agent failure rate.

Run in parallel with the math_advisor bench (= shares vLLM endpoint).
"""
from __future__ import annotations
import os, sys, time, json, re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kobushi_core.model import OpenAIModelAdapter, ModelMessage


def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


SYS = """You are evaluating the ambiguity of a data analysis question.

Given a question and the available data context, enumerate ALL DISTINCT REASONABLE
INTERPRETATIONS of what the question could mean. Each interpretation should
correspond to a different SQL query that a competent analyst might write.

Consider ambiguity from:
- Multiple matching columns/tables (= which "name" do they mean?)
- Vague predicates (= what counts as "abnormal" / "high" / "recent"?)
- Filter scope (= just this period / overall / both?)
- Output column choice (= just the value / the entity with the value / both?)
- Tie handling (= one row / all tied rows / first only?)
- Unit / format inference (= "0:01:54" = HH:MM:SS or MM:SS.ms?)
- Aggregation choice (= AVG of records / SUM / 1 of N?)

Format: numbered list, ONE interpretation per line, max 1 sentence each.
Output ONLY the numbered list. No preamble, no summary. Be concise."""


def make_model(temp=0.8):
    return OpenAIModelAdapter(
        model="qwen3.5-35b-a3b",
        api_base=os.environ["AGENT_API_BASE"],
        api_key=os.environ["AGENT_API_KEY"],
        temperature=temp,
        extra_headers={
            "CF-Access-Client-Id": os.environ.get("CF_ACCESS_CLIENT_ID", ""),
            "CF-Access-Client-Secret": os.environ.get("CF_ACCESS_CLIENT_SECRET", ""),
        },
    )


def build_context_summary(context_dir: Path) -> str:
    """Compact summary of the task's data sources for the LLM."""
    parts = []
    # CSV column headers
    for csv_p in context_dir.glob("csv/*.csv"):
        try:
            with csv_p.open() as f:
                hdr = f.readline().strip()
            parts.append(f"CSV {csv_p.name}: {hdr[:200]}")
        except Exception:
            pass
    # JSON keys
    for j_p in list(context_dir.glob("json/*.json"))[:3]:
        try:
            d = json.loads(j_p.read_text())
            if isinstance(d, list) and d:
                keys = list(d[0].keys()) if isinstance(d[0], dict) else []
                parts.append(f"JSON {j_p.name}: list of dicts with keys {keys[:8]}")
            elif isinstance(d, dict):
                parts.append(f"JSON {j_p.name}: keys {list(d.keys())[:8]}")
        except Exception:
            pass
    # Doc files (titles only)
    for md_p in list(context_dir.glob("doc/*.md"))[:2]:
        try:
            with md_p.open() as f:
                for line in f:
                    if line.startswith("# ") or line.startswith("### "):
                        parts.append(f"DOC {md_p.name}: section '{line.strip()[:100]}'")
                        break
        except Exception:
            pass
    # knowledge.md presence
    if (context_dir / "knowledge.md").exists():
        parts.append("knowledge.md: present (= doc with field definitions)")
    return "\n".join(parts) if parts else "(no schema info)"


def count_interpretations(response: str) -> int:
    """Count numbered list items: lines starting with 1./2./3. etc."""
    lines = [l.strip() for l in response.strip().split("\n")]
    return sum(1 for l in lines if re.match(r"^\d+[.)\s]", l))


def query_once(model, q: str, ctx: str) -> tuple[int, str]:
    r = model.complete(
        [ModelMessage(role="system", content=SYS),
         ModelMessage(role="user", content=f"Question: {q}\n\nAvailable data:\n{ctx}\n\nList all distinct reasonable interpretations:")],
        enable_thinking=False, max_tokens=1500,
    )
    n = count_interpretations(r)
    return n, r.strip()


def main():
    tasks = [
        ("task_22",  "easy lookup (= Connor Hilton dues date)"),
        ("task_25",  "clear superlative (= lowest cost event)"),
        ("task_80",  "known ambiguous (= 0:01:54 time format)"),
        ("task_418", "ambiguous (= 'abnormal' creatinine concept)"),
        ("task_180", "multi-filter complex"),
        ("task_396", "doc-heavy (= percentage Marvel)"),
        ("task_173", "unknown"),
    ]
    N_SAMPLES = 5
    model = make_model(temp=0.8)

    log(f"=== Question Ambiguity POC ({len(tasks)} tasks × {N_SAMPLES} samples, temp=0.8) ===\n")

    results = []
    for tid, note in tasks:
        task_json = json.load(open(ROOT / f"data/public/input/{tid}/task.json"))
        q = task_json["question"]
        ctx_dir = ROOT / f"data/public/input/{tid}/context"
        ctx = build_context_summary(ctx_dir)
        log(f"--- {tid} ({note}) ---")
        log(f"  Q: {q[:120]}")
        counts = []
        responses = []
        for i in range(N_SAMPLES):
            try:
                n, r = query_once(model, q, ctx)
                counts.append(n)
                responses.append(r)
                log(f"    sample {i+1}: {n} interpretations")
            except Exception as e:
                log(f"    sample {i+1}: ERR {str(e)[:120]}")
        if counts:
            import statistics
            m = sum(counts)/len(counts)
            sd = statistics.stdev(counts) if len(counts) > 1 else 0
            log(f"  mean={m:.2f}, stdev={sd:.2f}, range=[{min(counts)}, {max(counts)}]")
            results.append({"tid": tid, "mean": m, "stdev": sd,
                            "counts": counts, "samples": responses[:1]})  # save 1 sample
        else:
            results.append({"tid": tid, "mean": 0, "stdev": 0, "counts": [], "samples": []})
        log("")

    # Save
    out_path = ROOT / "artifacts" / "ambiguity_poc.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, default=str))
    log(f"Saved: {out_path}")

    # Summary table
    log("\n=== Summary (sorted by mean interpretation count) ===")
    for r in sorted(results, key=lambda x: -x["mean"]):
        log(f"  {r['tid']:10s}  mean={r['mean']:.2f}  stdev={r['stdev']:.2f}  counts={r['counts']}")


if __name__ == "__main__":
    main()
